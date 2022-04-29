import argparse
import collections
import numpy as np
import torch
import os
from torch import nn
from tqdm import tqdm
import pandas as pd
import multiprocessing as mp
import random as ran
from collections import defaultdict
import networkx as nx
try:
    import open3d as o3d
except ImportError:
    pass

from syconn.reps.super_segmentation import SuperSegmentationDataset
from syconn.handler.prediction import certainty_estimate
from syconn.handler.prediction_pts import sso2hc
from typing import Tuple, Callable
from morphx.classes.hybridcloud import HybridCloud
import morphx.processing.clouds as clouds
from morphx.processing.hybrids import extract_subset
from morphx.processing.objects import context_splitting_kdt
from elektronn3.models.convpoint import SegSmall
from syconn.proc.meshes import mesh2obj_file_colors
from scipy.spatial import cKDTree

PINK = np.array([10., 255., 10., 255.])
BLUE = np.array([255., 125., 125., 255.])
GREY = np.array([180., 180., 180., 255.])

pts_feat_dict = dict(sv=0)

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


# adapted version of the evaluate method evaluate_preds in prediction_pts.py
# also takes into consideration the raw predictions of each vertex
def evaluate_real_preds(preds_idcs: np.ndarray, zipped_preds, pred_labels):
    """ ith entry in ``preds_idcs`` contains vertex index of prediction saved at ith entry of preds.
        Predictions for each vertex index are gathered and then evaluated by a majority vote.
        The result gets saved at the respective index in the pred_labels array, along with the raw predictions of the vertices.
        Raw prediction of a vertex is taken with the help of a maximum among all candidates of that vertex. """
    pred_dict = defaultdict(list)
    arg_list = defaultdict(list)
    u_preds_idcs = np.unique(preds_idcs)
    for i in range(len(preds_idcs)):
        try:
            pred_dict[preds_idcs[i]].append(zipped_preds[i])
        except:
            print(f'problem i: {i}')
            return
    #get the strongest raw predictions of all and pick
    for u_ix in u_preds_idcs:
        arg_list = [a[0] for a in pred_dict[u_ix]]
        counts = np.bincount(arg_list)
        max_pred_label = np.argmax(counts)
        # get all raw predictions for the chosen label and calculate max
        raw_preds = np.array([a[1] for a in pred_dict[u_ix]])[np.where(arg_list==max_pred_label)[0]]
        max_raw = np.max(raw_preds,0)
        pred_labels[u_ix] = (max_pred_label, max_raw)


def extract_subhcs(hc: HybridCloud, ctx_size, ctx_dst_fac, npoints, transform: Callable):
    # choose base nodes with context overlap
    base_node_dst = ctx_size / ctx_dst_fac
    # select source nodes for context extraction
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(hc.nodes)
    # pick source nodes by downsampling the skeleton cloud
    pcd, idcs = pcd.voxel_down_sample_and_trace(base_node_dst, pcd.get_min_bound(), pcd.get_max_bound())
    source_nodes = np.max(idcs, axis=1)
    bs = 1
    n_batches = int(np.ceil(len(source_nodes) / bs))
    if len(source_nodes) % bs != 0:
        source_nodes = np.concatenate([np.random.choice(source_nodes, bs - len(source_nodes) % bs),
                                       source_nodes])
    node_arrs = context_splitting_kdt(hc, source_nodes, ctx_size)
    nnodes = 100
    # (e.g. every 4th if n_batches = 4)
    for ii in range(n_batches):
        # initialize list of data
        batch_v = np.zeros((bs, npoints, 3))
        batch_f = np.zeros((bs, npoints, 1), dtype=bool)
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
        for i, node_arr in enumerate(node_arrs[ii::n_batches]):
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
            # fill batches with sampled and transformed subsets
            hc_sample, idcs_sample, node_idcs_sample = sample_cloud(hc_sub, npoints, nnodes)

            # get target locations
            # if nnodes == 1:
            #     out_coords = np.array([hc.nodes[source_nodes[i]]])
            # elif len(hc_sub.nodes) < nnodes:
            #     sample_pts = hc_sub.vertices
            #     # add surface points
            #     add_verts = sample_pts[np.random.choice(len(sample_pts), nnodes - len(hc_sub.nodes))]
            #     out_coords = np.concatenate([hc_sub.nodes, add_verts])
            # # down sample to ~500nm apart
            # else:
            #     pcd = o3d.geometry.PointCloud()
            #     pcd.points = o3d.utility.Vector3dVector(hc_sub.nodes)
            #     pcd, idcs = pcd.voxel_down_sample_and_trace(500, pcd.get_min_bound(), pcd.get_max_bound())
            #     base_points = np.max(idcs, axis=1)
            #     base_points = np.random.choice(base_points, nnodes,
            #                                    replace=len(base_points) < nnodes)
            #     out_coords = hc_sub.nodes[base_points]


            tree = cKDTree(data=hc.nodes)
            node_idcs = np.unique(np.concatenate(tree.query_ball_point(hc_sample.nodes, r=100)))
            # np.random.shuffle(node_idcs)
            # print(f'hc_sample nodes: {hc_sample.nodes}')
            # print(f'hc nodes: {hc.nodes}')
            # print(f'node_idcs: {node_idcs}')
            sample_nodes = hc_sample.nodes
            while len(node_idcs) > nnodes:
                print(f'more than {nnodes}')
                if len(node_idcs) > nnodes + 20:
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(sample_nodes)
                    # pick source nodes by downsampling the skeleton cloud
                    pcd, idcs = pcd.voxel_down_sample_and_trace(500, pcd.get_min_bound(), pcd.get_max_bound())
                    sample_node_idcs = np.max(idcs, axis=1)
                    base_points = np.random.choice(sample_node_idcs, nnodes,
                                                   replace=len(sample_node_idcs) < nnodes)
                    sample_nodes = sample_nodes[base_points]
                    node_idcs = np.unique(np.concatenate(tree.query_ball_point(sample_nodes, r=100)))
                    print(f'len node_idcs: {len(node_idcs)}')
                    # np.random.shuffle(node_idcs)
                if len(node_idcs) > nnodes:
                    node_idcs = node_idcs[:nnodes]

            if len(node_idcs) < nnodes:
                # add surface points
                add_verts = hc_sample.vertices[np.random.choice(len(hc_sample.vertices), nnodes - len(sample_nodes))]
                sample_nodes = np.concatenate([sample_nodes, add_verts])
                node_idcs = np.unique(np.concatenate(tree.query_ball_point(sample_nodes, r=100)))
                # np.random.shuffle(node_idcs)

            hc_sample._nodes = sample_nodes
            # print(f'node indices in extract: {node_idcs}')

            inner_mask = np.zeros(shape=(len(hc_sample.nodes),), dtype=bool)
            # tree = cKDTree(data=hc_sample.nodes)
            # inner_ids = tree.query_ball_point(hc.nodes[source_nodes[ii]], r=15000)
            # inner_mask[inner_ids] = bool(1)
            global_vert_idcs = idcs_sub[idcs_sample.astype(int)]
            # global_vert_idcs = global_vert_idcs[inner_mask]             # uncomment for inner focus of context
            global_node_idcs = node_idcs

            if transform is not None:
                transform(hc_sample)
            arr_list['verts'][cnt] = hc_sample.vertices
            arr_list['feats'][cnt] = hc_sample.features
            # masks get used later when mapping predictions back onto the cell surface during postprocessing
            arr_list['nodes'][cnt] = hc_sample.nodes
            arr_list['margin_mask'][cnt] = inner_mask
            arr_list['source_node'][cnt] = source_nodes[ii]
            arr_list['global_vert_indices'].append(global_vert_idcs)
            arr_list['global_node_indices'].append(global_node_idcs.astype(int))
            cnt += 1
        yield (arr_list['feats'], arr_list['verts'], arr_list['nodes']), arr_list['source_node'], arr_list['margin_mask'], arr_list['global_vert_indices'], arr_list['global_node_indices']


def process_data_slice(slice, ssd, ssv_ids, model, ctx_size, ctx_dst_fac, npoints, pred_transform, device):
    """
    Adds average results of  per cell in the res_dc dictionary of results.

    Args:
        slice: np.s_ slice to process the files
        pred_files: file paths containing HybridCLoud pickles

    """
    # for each whole cell in slice
    for ssv_id in tqdm(ssv_ids[slice], desc='Predict SSOs'):
        print(f'Do ssv: {ssv_id}')
        res_dc = {
            'merge_coordinates': [],
            'confidences': []
        }

        sso = ssd.get_super_segmentation_object(ssv_id)
        sso.load_skeleton()

        hc, voxel_dict = sso2hc(sso, tuple(pts_feat_dict.keys()), tuple(pts_feat_dict.values()), 'merger')
        verts = hc.nodes

        predictions = []
        pred_indices = []
        raw_predictions = []
        res_list = []

        ii = 0
        source_nodes=[]

        # do predictions on contexts of the cell
        for (sample_feats, sample_pts, sample_out_pts), source_node, mask, vert_indices, node_indices in extract_subhcs(hc, ctx_size, ctx_dst_fac,
                                                                                      npoints, pred_transform):

            sample_feats = sample_feats[:, :, None]
            source_node = int(source_node[0,0])
            source_nodes.append(source_node)
            # print(f'len sample out nodes: {sample_out_pts.shape} \n node idcs {len(node_indices[0])}')

            dpts = torch.from_numpy(sample_pts).to(device).float()
            dfeats = torch.from_numpy(sample_feats).to(device).float()
            dtarget_pts = torch.from_numpy(sample_out_pts).to(device).float()

            with torch.no_grad():
                pred = model(dfeats, dpts, dtarget_pts)
                pred = pred.detach().cpu().numpy()
                # eliminate batch axis for further processing
                raw_preds = pred[0, :, :]

            # prepare predictions
            pred = np.argmax(raw_preds, 1)
            if len(np.where(pred == 1)[0]) > 0:
                  print(f'foreground labels')
            predictions.append(pred)
            # print(f'node_indices in process_data {len(node_indices[1])} \n{node_indices[1]}')
            pred_indices.append(node_indices[1])
            raw_predictions.append(raw_preds)

            ii += 1

        # "merge" contexts and their predictions, taking the majority vote over all predictions per vertex
        print(f'Merge context predictions...')
        predictions = np.concatenate(predictions)
        pred_indices = np.concatenate(pred_indices)
        raw_preds = np.concatenate(raw_predictions)

        zipped_preds = list(zip(predictions, raw_preds))
        print(f'len predictions: {len(predictions)}')
        print(f'len raw preds: {len(raw_preds)}')
        print(f'zipped len: {len(zipped_preds)}')
        print(f'idcs len: {len(pred_indices)}')

        # 0: background, 1: foreground, 3: no prediction -> for the use of bincount
        pred_labels = list(zip(np.ones((len(hc.nodes), 1)) * (3), np.ones((len(hc.nodes), 1)) * (3)))
        # pred labels will have the vertices labels of values [0,1,3]
        evaluate_real_preds(pred_indices, zipped_preds, pred_labels)

        preds = [a[0] for a in pred_labels]
        labeled_indices = np.where(preds != 3)[0]
        labeled_vertices = hc.nodes[labeled_indices]
        labeled_tree = cKDTree(data=labeled_vertices, )
        node_idcs = np.where(pred_labels == 3)[0]
        if len(node_idcs) > 0:
            print(f'unlabeled nodes exist: {len(node_idcs)}')
        unlabeled_verts = hc.nodes[node_idcs]
        labeled_neighbors = labeled_tree.query_ball_point(x=unlabeled_verts, r=500)

        # propagate labels to the unlabeled vertices
        for i, cluster in enumerate(labeled_neighbors):
            if len(cluster) == 0:
                pred_labels[node_idcs[i]] = int(0)
            cnt = np.bincount(pred_labels[labeled_indices[cluster]][:, 0].astype(np.int64))
            try:
                vert_label = np.argmax(cnt)
            except:
                print(
                    f'Could not find neighbor in propagation for cell {ssv_id} and vert {unlabeled_verts[i]} in idc of hc.nodes {node_idcs[i]}')
                vert_label = int(0)
            pred_labels[node_idcs[i]] = vert_label

        # update labels and raw predictions
        preds = np.array([a[0] for a in pred_labels], dtype='object')
        raw_preds = [a[1] for a in pred_labels]
        rp = []
        for gen in raw_preds:
            a=[]
            for i in gen:
                a.append(i)
            if a == [3.0]:
                a = [3.0,3.0]
            rp.append(a)
        raw_preds = np.array(rp)

        #evaluate confidence score
        print(f'Evaluating confidence score...')
        pos_indices = np.where(preds == 1)[0]
        positive_vertices = verts[pos_indices]
        raw_pos_preds = raw_preds[pos_indices]
        edges = []
        vert_NN = cKDTree(data=positive_vertices, )

        # TODO optimize this
        # search for every neighbor of every vertex
        for i, pos_vertex in enumerate(positive_vertices):
            neighbors = vert_NN.query_ball_point(x=pos_vertex, r=1000, workers=2)
            curr_edges = [(i, x) for x in neighbors]
            edges.extend(curr_edges)

        G = nx.Graph(edges)
        connected_components = nx.connected_components(G)
        coords = []
        confidences = []

        # for each cluster calculate confidence score
        for component in connected_components:
            component_vertices = positive_vertices[list(component)]
            raw_component_preds = raw_pos_preds[list(component)]

            # Ignore clusters under 50 vertices
            if len(raw_component_preds) < 50:
                continue
            else:
                entr = certainty_estimate(raw_component_preds, is_logit=True)

            if entr < 0.2:
                continue

            # calculate confidence score
            confidences.append(entr)

            # get coordinates of a connected component/merge error site
            mean_coords = np.mean(component_vertices, axis=0)
            coords.append(list(mean_coords))

        res_dc['merge_coordinates'] = coords
        res_dc['confidences'] = confidences

        df = pd.DataFrame.from_dict(res_dc, orient='index')
        csv_path = f'/wholebrain/scratch/amancu/mergeError/Nodes/Preds/betterOutNodes_classification/qualitative/sso_{ssv_id}.csv'
        df.to_csv(csv_path)

        res_list.append(res_dc)

        print(f'File written, render point cloud...')
        coord = res_dc['merge_coordinates']

        # prediction
        tree = cKDTree(data=verts)
        ids=[]
        if len(res_dc['merge_coordinates']) != 0:
            ids = tree.query_ball_point(coord,r=2000)
            ids = np.concatenate(ids)
        else:
            ids=[]

        colors = np.full(shape=(hc.nodes.shape[0], 4,), fill_value=GREY)
        mask = np.array([[x] for x in ids])
        try:
            np.put_along_axis(colors, mask, PINK, axis=0)
        except:
            print("No foreground labels in prediction.")
            pass
        mesh2obj_file_colors(os.path.expanduser(
            f'/wholebrain/scratch/amancu/mergeError/Nodes/Preds/betterOutNodes_classification/qualitative/{ssv_id}_prediction.ply'),
            [np.array([]), hc.nodes, np.array([])], colors)

    print(f'Queue put done. Store dict...')
    return

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')

    input_channels = 1
    num_classes = 2
    use_norm = 'gn'
    dr = 0.3
    track_running_stats = False
    act = 'swish'
    use_bias = True
    npoints = int(10e3)
    scale_norm = 5e3
    ctx_size = 20e3
    ctx_dst_fac = 3
    pred_transform = clouds.Compose([clouds.Center(), clouds.Normalization(scale_norm)])

    radius = 2000
    nproc = 1

    lcp_flag = True
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    ssv_ids = [46495592, 1471641453, 1303277836, 89548345, 1639126257, 18894494, 9855203]

    ssd_kwargs = dict(working_dir='/ssdscratch/pschuber/songbird/j0251/j0251_72_seg_20210127_agglo2/')
    ssd = SuperSegmentationDataset(**ssd_kwargs)
    mdir = f'/wholebrain/scratch/amancu/mergeError/Nodes/Trainings/SegSmall_betterOutNodes_r3000_ConvPoint_SearchQuantized_Adam_StepLR_CrossEntropy_Classification/state_dict.pth'
    # mdir = f'/wholebrain/scratch/amancu/mergeError/Nodes/Trainings/SegSmall_betterOutNodes_r3000_ConvPoint_SearchQuantized_Adam_StepLR_CrossEntropy_Classification/state_dict.pth'
    # mdir = f'/wholebrain/scratch/amancu/mergeError/Nodes/Trainings/SegSmall_betterOutNodes_r3000_ConvPoint_SearchQuantized_Adam_StepLR_CrossEntropy_Classification/state_dict.pth'
    model = SegSmall(input_channels, num_classes, dropout=dr, use_norm=use_norm,
                     track_running_stats=track_running_stats, act=act, use_bias=use_bias).to(device)
    model.load_state_dict(torch.load(mdir, map_location=device)['model_state_dict'])
    model.eval()                # no dropout layers, no gradient memorization

    # split tasks for processes
    proc_slices = []
    offset = len(ssv_ids) // nproc
    for i in range(nproc):
        slice_start = offset * i
        slice_end = offset * (i+1) if i < nproc - 1 else len(ssv_ids)
        # slice_end = offset * (i+1) if i < nproc - 1 else 10
        proc_slices.append(np.s_[slice_start:slice_end])

    print(f'slices {proc_slices}')
    queue = mp.Queue()
    jobs = []
    running_tasks = []
    params = [(slice, ssd, ssv_ids, model, ctx_size, ctx_dst_fac, npoints, pred_transform,
               device) for slice in proc_slices]

    running_tasks = [mp.Process(target=process_data_slice, args=param) for param in params]
    for running_task in running_tasks:
        running_task.start()
    for running_task in running_tasks:
        running_task.join()
    print(f'Processing finished')

    # get results
    results = [queue.get() for task in running_tasks]