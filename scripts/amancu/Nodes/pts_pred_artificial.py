import glob
import numpy as np
import torch
import os
from os import getpid
from tqdm import tqdm
import pandas as pd
import multiprocessing as mp
import random as ran

try:
    import open3d as o3d
except ImportError:
    pass

from typing import Iterable, Union, Optional, Tuple, Callable, List
from morphx.classes.hybridcloud import HybridCloud
import morphx.processing.clouds as clouds
from morphx.processing.hybrids import extract_subset
from morphx.processing.objects import context_splitting_kdt
from elektronn3.models.convpoint import SegSmall
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, precision_recall_curve, balanced_accuracy_score
from syconn.proc.meshes import mesh2obj_file_colors
from syconn.handler.prediction_pts import evaluate_preds
from scipy.spatial import cKDTree

def sample_cloud(hc: HybridCloud, vertex_number: int, node_number: int, random_seed: int = None,
                    jitter: int = 0, padding: int = None) -> Tuple[HybridCloud, np.ndarray, np.ndarray]:
    """ Creates a (pseudo)random sample point cloud with a specific number of points from the given subset of mesh
    vertices. If the requested number of points is larger than the given subset, the subset gets enriched with slightly
    augmented points before sampling if padding is None. If padding is not None, points with padding as coordinates are
    added to the resulting sample. These padded points have all properties (e.g. features, labels) from the point at
    pc.vertices[0].

    Args:
        hc: MorphX PointCloud object which should be sampled.
        vertex_number: The number of points which should make up the sample point cloud.
        random_seed: Possibility for making the sampling deterministic.
        jitter: Add small jitter to possible duplicate points due to oversampling. A jitter of
            np.random.random((deficit, 3))*jitter is added to the duplicate points.
        padding: If not None, any point deficit in the input point cloud is compensated by points with this padding.

    Returns:
        PointCloud with sampled points (and labels) and indices of the original vertices where samples are from.
    """
    if len(hc.vertices) == 0:
        return hc, np.array([])
    if random_seed is not None:
        np.random.seed(random_seed)
    samplel = None
    samplef = None
    samplet = None
    samplen = None
    samplenl = None
    samplepl = None
    vert_ixs = np.arange(len(hc.vertices))
    np.random.shuffle(vert_ixs)
    # cache vertex indices of sample for later mapping
    sample_ixs = np.zeros(vertex_number, dtype=int)
    sample_ixs[:min(len(hc.vertices), vertex_number)] = vert_ixs[:vertex_number]

    node_jxs = np.arange(len(hc.nodes))
    np.random.shuffle(node_jxs)
    # cache vertex indices of sample for later mapping
    sample_jxs = np.zeros(node_number, dtype=int)
    sample_jxs[:min(len(hc.nodes), node_number)] = node_jxs[:node_number]

    if padding is None:
        # add augmented points in case of deficit
        deficit = max(0, vertex_number - len(hc.vertices))
        offset = len(hc.vertices)
        # while loop and replace=False ensures uniform oversampling
        while deficit != 0:
            next_compensation = min(len(vert_ixs), deficit)
            sample_ixs[offset:offset+next_compensation] = np.random.choice(vert_ixs, next_compensation, replace=False)
            deficit -= next_compensation
            offset += next_compensation
        samplev = hc.vertices[sample_ixs].astype(float)
    else:
        # add padded points in case of deficit
        samplev = np.ones((vertex_number, 3)) * padding
        samplev[:len(hc.vertices)] = hc.vertices[sample_ixs[:len(hc.vertices)]]

    if len(hc.labels) != 0:
        samplel = hc.labels[sample_ixs]
    if len(hc.features) != 0:
        samplef = hc.features[sample_ixs]
    if len(hc.nodes) != 0:
        samplen = hc.nodes[sample_jxs]
    if len(hc.node_labels) != 0:
        samplenl = hc.node_labels[sample_jxs]
    if len(hc.types) != 0:
        samplet = hc.types[sample_ixs]
    if len(hc.pred_labels) != 0:
        samplepl = hc.pred_labels[sample_ixs]
    # add jitter to duplicate points
    samplev[len(hc.vertices):] += np.random.random((max(0, vertex_number - len(hc.vertices)), 3))*jitter
    return HybridCloud(vertices=samplev, labels=samplel, features=samplef, nodes=samplen, node_labels=samplenl, types=samplet, pred_labels=samplepl,
                      encoding=hc.encoding, no_pred=hc.no_pred), sample_ixs, sample_jxs


def extract_subhcs(hc: HybridCloud, ctx_size, ctx_dst_fac, npoints, transform: Callable):
    # choose base nodes with context overlap
    base_node_dst = ctx_size / ctx_dst_fac
    # select source nodes for context extraction
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(hc.nodes)
    pcd, idcs = pcd.voxel_down_sample_and_trace(base_node_dst, pcd.get_min_bound(), pcd.get_max_bound())
    source_nodes = np.max(idcs, axis=1)
    bs = 1
    n_batches = int(np.ceil(len(source_nodes) / bs))
    # add additional source nodes to fill batches
    if len(source_nodes) % bs != 0:
        source_nodes = np.concatenate([np.random.choice(source_nodes, bs - len(source_nodes) % bs),
                                       source_nodes])
    node_arrs = context_splitting_kdt(hc, source_nodes, ctx_size)
    # print(f'Node arrs:  {node_arrs}')

    # collect contexts into batches (each batch contains every n_batches contexts
    # (e.g. every 4th if n_batches = 4)
    for ii in range(n_batches):
        nnodes = len(node_arrs[ii])
        # initialize list of data
        batch_v = np.zeros((bs, npoints, 3))
        batch_f = np.zeros((bs, npoints), dtype=bool)
        mask = np.zeros((bs, nnodes), dtype=bool)
        batch_sn = np.zeros((bs, 1))
        batch_n = np.zeros((bs, nnodes, 3))
        batch_l = np.zeros((bs, nnodes, 1), dtype=bool)
        idcs_list = []
        arr_list = {'verts': batch_v,
                    'feats': batch_f,
                    'nodes': batch_n,
                    'labels': batch_l,
                    'margin_mask': mask,
                    'source_node': batch_sn,
                    'global_vert_indices': idcs_list,
                    'global_node_indices': idcs_list}

        # generate contexts
        cnt = 0
        for node_arr in node_arrs[ii::n_batches]:
            hc_sub, idcs_sub = extract_subset(hc, node_arr)
            ix = 0
            while len(hc_sub.vertices) == 0:
                if ix >= len(hc.nodes):
                    raise IndexError(f'Could not find suitable context in HC during "extract_subhcs".')
                elif ix >= len(node_arrs):
                    # if the cell fragment, represented by hc, is small and its skeleton not well centered,
                    # it can happen that all extracted sub-skeletons do not contain any vertex. in that case
                    # use any node of the skeleton
                    sn = np.random.randint(0, len(hc.nodes))
                    hc_sub, idcs_sub = extract_subset(hc, context_splitting_kdt(hc, sn, ctx_size))
                else:
                    hc_sub, idcs_sub = extract_subset(hc, node_arrs[ix])
                ix += 1
            # print(f'hc nodes')
            assert len(hc_sub.nodes) == len(node_arr)
            # fill batches with sampled and transformed subsets
            hc_sample, idcs_sample, node_idcs_sample = sample_cloud(hc_sub, npoints, nnodes)
            tree = cKDTree(data=hc_sample.nodes)
            inner_ids = tree.query_ball_point(hc.nodes[source_nodes[ii]], r=15000)
            inner_mask = np.zeros(shape=(len(hc_sample.nodes),), dtype=bool)
            inner_mask[inner_ids] = bool(1)
            # get vertex indices respective to total hc
            global_vert_idcs = idcs_sub[idcs_sample.astype(int)]
            # global_vert_idcs = global_vert_idcs[inner_mask]             # uncomment for inner focus of context
            global_node_idcs = node_arr
            # print(f'global_node_idcs: {global_node_idcs}')

            if transform is not None:
                transform(hc_sample)
            arr_list['verts'][cnt] = hc_sample.vertices
            arr_list['feats'][cnt] = hc_sample.features
            # masks get used later when mapping predictions back onto the cell surface during postprocessing
            arr_list['nodes'][cnt] = hc_sample.nodes
            arr_list['labels'][cnt] = hc_sample.node_labels
            arr_list['margin_mask'][cnt] = inner_mask
            arr_list['source_node'][cnt] = source_nodes[ii]
            arr_list['global_vert_indices'].append(global_vert_idcs)
            arr_list['global_node_indices'].append(global_node_idcs)
            cnt += 1
        yield (arr_list['feats'], arr_list['verts'], arr_list['nodes'], arr_list['labels']), arr_list['source_node'], arr_list['margin_mask'], arr_list['global_vert_indices'], arr_list['global_node_indices']

def process_data_slice(slice, pred_files, model, ctx_size, ctx_dst_fac, npoints, pred_transform, device, global_dict):
    """
    Adds average results of metrics (precision, recall, accuracy, f1score) per cell in the res_dc dictionary of results.

    Args:
        slice: np.s_ slice to process the files
        pred_files: file paths containing HybridCLoud pickles

    """
    hc = HybridCloud()
    # for each whole cell
    for i in tqdm(pred_files[slice], desc='Predict HCs'):
        path = i
        hc.load_from_pkl(i)
        predictions = []
        pred_indices = []

        vert_tree = cKDTree(data=hc.vertices,)
        ii = 0
        for (sample_feats, sample_pts, sample_out_pts, sample_labels), source_node, mask, vert_indices, node_indices in extract_subhcs(hc, ctx_size, ctx_dst_fac,
                                                                                      npoints, pred_transform):

            sample_feats = sample_feats[:, :, None]

            dpts = torch.from_numpy(sample_pts).to(device).float()
            dfeats = torch.from_numpy(sample_feats).to(device).float()
            dtarget_pts = torch.from_numpy(sample_out_pts).to(device).float()

            with torch.no_grad():
                pred = model(dfeats, dpts, dtarget_pts)
                pred = pred.detach().cpu().numpy()
                # eliminate batch axis
                pred = pred[0, :, :]

            # prepare predictions
            # print(f'pred: {pred}')
            # print(f'argmax: {np.argmax(pred, 1)}')
            pred = np.argmax(pred, 1)

            # place the external context of the prediction on null, so that they focus only on the middle
            # pred= pred[mask[0,:]]

            predictions.append(pred)
            pred_indices.append(node_indices[1])
            ii += 1

        # "merge" contexts and their predictions, taking the majority vote over all predictions per vertex
        predictions = np.concatenate(predictions)
        pred_indices = np.concatenate(pred_indices)

        # 0: background, 1: foreground, 3: no prediction -> for the use of bincount
        pred_labels = np.ones((len(hc.nodes), 1)) * (3)
        # pred labels will have the vertices labels of values [0,1,3]
        evaluate_preds(pred_indices, predictions, pred_labels)

        if len(np.where(pred_labels==3)[0]) != 0:
            print(f'There are unlabeled nodes')
            # pred_labels[np.where(pred_labels==3)[0]] = int(0)


        # # render the contexts
        # colors = np.full(shape=(hc.vertices.shape[0], 4,), fill_value=GREY)
        # mask = np.zeros(len(hc.vertices))
        # mask[inner_vert_idcs] = 1
        # mask = mask.astype(bool)
        # mask = np.array([[x] for x in mask])
        # try:
        #     np.put_along_axis(colors, mask, PINK, axis=0)
        # except:
        #     # print("No foreground labels in original context.")
        #     pass
        # mesh2obj_file_colors(os.path.expanduser(
        #     # f'/wholebrain/scratch/amancu/mergeError/preds/lcp/ConvPoint/2000/archLrg/' + os.path.basename(
        #     f'/wholebrain/scratch/amancu/mergeError/preds/lcp/ConvPoint/2000/test/parallel/' + os.path.basename(
        #         path) + f'_context.ply'),
        #     [np.array([]), hc.vertices, np.array([])], colors)

        # for 3 labeled vertices, we employ a kdtree approach to inherit the label of the neighbouring vertices
        # propagate labels to unlabeled vertices
        labeled_indices = np.where(pred_labels!=3)[0]
        labeled_nodes = hc.nodes[labeled_indices]
        labeled_tree = cKDTree(data=labeled_nodes, )
        node_idcs = np.where(pred_labels == 3)[0]
        unlabeled_nodes = hc.nodes[node_idcs]
        labeled_neighbors = labeled_tree.query_ball_point(x=unlabeled_nodes, r=500)
        # print(f'neighbors {labeled_neighbors}')
        for i, cluster in enumerate(labeled_neighbors):
            if len(cluster) == 0:
                pred_labels[node_idcs[i]] = int(0)
            cnt = np.bincount(pred_labels[labeled_indices[cluster]][:, 0].astype(np.int64))
            try:
                node_label = np.argmax(cnt)
            except:
                print(f'Could not find neighbor in propagation for cell pair {os.path.basename(path)} and vert {unlabeled_nodes[i]} in idc of hc.vertices {node_idcs[i]}')
                node_label = int(0)
            pred_labels[node_idcs[i]] = node_label

        # calculate true and pred node labels
        true_labels = hc.node_labels
        true_node_labels = np.zeros(shape=(len(true_labels),))
        one_idcs = np.where(true_labels >= 0)[0]
        np.put(true_node_labels, one_idcs, np.ones(len(one_idcs)))

        # evaluate cell metric results
        pred_node_labels = pred_labels

        precision = precision_score(true_node_labels, pred_node_labels, average='binary', zero_division=0)
        recall = recall_score(true_node_labels, pred_node_labels, average='binary', zero_division=0)
        accuracy = accuracy_score(true_node_labels, pred_node_labels)
        balanced_accuracy = balanced_accuracy_score(true_node_labels, pred_node_labels)
        f1score = f1_score(true_node_labels, pred_node_labels, average='binary', zero_division=0)

        global_dict['precision'] = global_dict['precision'] + [precision]
        global_dict['recall'] = global_dict['recall'] + [recall]
        global_dict['accuracy'] = global_dict['accuracy'] + [accuracy]
        global_dict['balanced_accuracy'] = global_dict['balanced_accuracy'] + [balanced_accuracy]
        global_dict['fscore'] = global_dict['fscore'] + [f1score]

        # random node predictions to inspect in meshlab
        if ran.random() > 0.95:
            print(f'For {os.path.basename(path)} \n Precision: {precision} \n Recall: {recall} \n Accuracy: {accuracy} \n Fscore: {f1score} \n Balanced Accuracy: {balanced_accuracy}')
            # original nodes
            colors = np.full(shape=(hc.nodes.shape[0], 4,), fill_value=GREY)
            mask = np.where(true_node_labels == 1)[0]
            mask = np.array([[x] for x in mask])
            try:
                np.put_along_axis(colors, mask, PINK, axis=0)
            except:
                # print("No foreground labels in original context.")
                pass
            mesh2obj_file_colors(os.path.expanduser(
                f'/wholebrain/scratch/amancu/mergeError/Nodes/Preds/quantitative/meshes_{os.path.dirname(os.path.basename(path))}/' + os.path.basename(
                    path) + f'_original_nodes_.ply'),
                [np.array([]), hc.nodes, np.array([])], colors)

            # prediction
            colors = np.full(shape=(hc.nodes.shape[0], 4,), fill_value=GREY)
            mask = np.where(pred_node_labels == 1)[0].astype(np.int64)
            mask = np.array([[x] for x in mask])
            try:
                np.put_along_axis(colors, mask, PINK, axis=0)
            except:
                # print("No foreground labels in prediction.")
                pass
            mesh2obj_file_colors(os.path.expanduser(
                f'//wholebrain/scratch/amancu/mergeError/Nodes/Preds/quantitative/meshes_{os.path.dirname(os.path.basename(path))}/' + os.path.basename(
                    path) + f'_prediction_nodes.ply'),
                [np.array([]), hc.nodes, np.array([])], colors)
    return getpid()

# colors for labels
PINK = np.array([10., 255., 10., 255.])
BLUE = np.array([255., 125., 125., 255.])
GREY = np.array([180., 180., 180., 255.])
nproc = 1

if __name__ == '__main__':
    # define model args
    input_channels = 1
    num_classes = 2
    use_norm = False
    dr = 0.3
    track_running_stats = False
    act = 'swish'
    use_bias = True
    npoints = int(10e3)
    scale_norm = 5e3
    ctx_size = 20e3
    ctx_dst_fac = 3
    pred_transform = clouds.Compose([clouds.Center(), clouds.Normalization(scale_norm)])

    torch.multiprocessing.set_start_method('spawn')
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    folder = f'/wholebrain/scratch/amancu/mergeError/Nodes/TestGT/R3000_downsample300/*.pkl'
    # save_path = f'/wholebrain/scratch/amancu/mergeError/Nodes/Trainings/lcp_r3000_ConvPoint_SearchQuantized_None_Adam_StepLR_CrossEntropy/state_dict.pth'
    save_path = f'/wholebrain/scratch/amancu/mergeError/Nodes/Trainings/SegSmall_betterOutNodes_r3000_ConvPoint_SearchQuantized_Adam_StepLR_CrossEntropy_Classification/state_dict.pth'
    # save_path = f'/wholebrain/scratch/amancu/mergeError/Nodes/Trainings/SegSmall_r3000_ConvPoint_SearchQuantized_Adam_StepLR_MSE_Regression/state_dict.pth'
    # save_path = f'/wholebrain/scratch/amancu/mergeError/Nodes/Trainings/SegSmall_betterOutNodes_r3000_ConvPoint_SearchQuantized_run2_Adam_StepLR_MSE_Regression/state_dict.pth'
    pred_files = glob.glob(folder)

    model_name = os.path.basename(os.path.dirname(save_path))
    print(f'Predicting for model {model_name} with nr of files {len(pred_files)}')

    with torch.no_grad():
        search = 'SearchQuantized'
        conv = 'ConvPoint'
        convol = dict(layer=conv, kernel_separation=False)
        layer = convol['layer']
        model = SegSmall(input_channels, num_classes, dropout=dr, use_norm=use_norm,
                         track_running_stats=track_running_stats, act=act, use_bias=use_bias).to(device)
        model.load_state_dict(torch.load(save_path, map_location=device)['model_state_dict'])
        model.eval()                # no dropout layers, no gradient memorization

        # split tasks for processes
        proc_slices = []
        offset = len(pred_files) // nproc
        for i in range(nproc):
            slice_start = offset * i
            slice_end = offset * (i+1) if i < nproc - 1 else len(pred_files)
            proc_slices.append(np.s_[slice_start:slice_end])

        print(f'slices {proc_slices}')
        with mp.Manager() as manager:
            dict = manager.dict()
            dict['precision'] = []
            dict['recall'] = []
            dict['accuracy'] = []
            dict['balanced_accuracy'] = []
            dict['fscore'] = []

            params = [(slice, pred_files, model, ctx_size, ctx_dst_fac, npoints, pred_transform,
                       device, dict) for slice in proc_slices]
            running_tasks = [mp.Process(target=process_data_slice, args=param) for param in params]
            _ = [running_task.start() for running_task in running_tasks]
            _ = [running_task.join() for running_task in running_tasks]
            print(f'Processing finished')

            print(f'dict: {dict}')

            precisions = dict['precision']
            recalls = dict['recall']
            accuracies = dict['accuracy']
            balanced_accuracies = dict['balanced_accuracy']
            fscores = dict['fscore']
            # print(f'prec {precisions}')
            # print(f'recalls {recalls}')
            # print(f'accur {accuracies}')
            # print(f'fscores {fscores}')

            result = {
                'model name': model_name,
                'precision': np.mean(precisions),
                'recall': np.mean(recalls),
                'accuracy': np.mean(accuracies),
                'balanced_accuracy': np.mean(balanced_accuracies),
                'fscore': np.mean(fscores),
            }

            vals = list(result.values())

            print(f'Number zero Fscores: {len(np.where(fscores == 0.0)[0])}')
            print(f'Mean precisions: {vals[1]}')
            print(f'Mean recall: {vals[2]}')
            print(f'Mean accuracy: {vals[3]}')
            print(f'Mean balanced_accuracy: {vals[4]}')
            print(f'Mean fscores: {vals[5]}')

            df = pd.DataFrame.from_dict(result, orient='index')
            csv_path = f'/wholebrain/scratch/amancu/mergeError/Nodes/Preds/quantitative/{model_name}_test_result.csv'
            df.to_csv(csv_path)