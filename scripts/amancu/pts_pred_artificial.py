import glob
import argparse
import collections
import numpy as np
import torch
import os
from torch import nn
from tqdm import tqdm
import pandas as pd
import multiprocessing as mp

try:
    import open3d as o3d
except ImportError:
    pass

from typing import Iterable, Union, Optional, Tuple, Callable, List
from morphx.classes.hybridcloud import HybridCloud
import morphx.processing.clouds as clouds
from morphx.processing.hybrids import extract_subset
from morphx.processing.objects import context_splitting_kdt, context_splitting_graph_many
from elektronn3.models.convpoint import SegSmall
from elektronn3.models.lcp_adapt import ConvAdaptSeg
from lightconvpoint.utils.network import get_search, get_conv
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, precision_recall_curve
from scipy.special import softmax
from matplotlib import pyplot
from syconn.proc.meshes import mesh2obj_file_colors
from syconn.handler.prediction_pts import evaluate_preds


def prob(pred):
    """
    Transform prediciton to range [0,1], to which is the binary prediction closer.
    Args:
        pred: Raw prediction of shape (N,1)

    """
    # determine indices
    ind = np.argmax(pred, 1).astype(np.float64)
    ind = ind.astype(np.int64)
    # normalize predictions
    softened = softmax(pred, axis=1)
    # print(softened)
    res = []
    for i in range(len(pred)):
        nr = softened[i]
        nr = nr[ind[i]]
        res.append(nr if ind[i] == 1 else 1 - nr)
    res = np.array(res)
    # print(res)
    return res


def process_data_slice(slice, pred_files, lock, model, ctx_size, ctx_dst_fac, npoints, pred_transform, device, lcp_flag,
                       result_dc):
    """
    Adds average results of metrics (precision, recall, accuracy, f1score) per cell in the result_dc dictionary of results.

    Args:
        slice: np.s_ slice to process the files
        pred_files: file paths containing HybridCLoud pickles

    """
    hc = HybridCloud()
    # for each whole cells
    for i in tqdm(range(len(pred_files))):
        path = pred_files[i]
        hc.load_from_pkl(path)
        predictions = []
        pred_indices = []

        ii = 0
        for (sample_feats, sample_pts, sample_labels), vert_indices in extract_subhcs(hc, ctx_size, ctx_dst_fac,
                                                                                      npoints, pred_transform):

            sample_feats = sample_feats[:, :, None]

            dpts = torch.from_numpy(sample_pts).to(device).float()
            dfeats = torch.from_numpy(sample_feats).to(device).float()

            if lcp_flag:
                dpts = dpts.transpose(1, 2)
                dfeats = dfeats.transpose(1, 2)

            # print(f'pts shape {dpts.shape} feats shape {dfeats.shape}')

            with torch.no_grad():
                # try:
                pred = model(dfeats, dpts)
                # except:
                #     print('No se puede')
                #     continue
                if lcp_flag:
                    pred = pred.transpose(1, 2)

                pred = pred.detach().cpu().numpy()
                # eliminate batch axis
                pred = pred[0, :, :]
                # print(pred)

            # prepare predictions
            pred = np.argmax(pred, 1)
            predictions.append(pred)
            pred_indices.append(vert_indices[0])

            # uncomment to see meshlab results
            # # original
            colors = np.full(shape=(sample_pts.shape[1], 4,), fill_value=GREY)
            mask = np.where(sample_labels == 1)[0]
            mask = np.array([[x] for x in mask])
            try:
                np.put_along_axis(colors, mask, RED, axis=0)
            except:
                # print("No foreground labels in original context.")
                pass
            mesh2obj_file_colors(os.path.expanduser(
                f'/wholebrain/scratch/amancu/mergeError/preds/lcp/ConvPoint/1000/Adam_ExponentialLR/' + os.path.basename(
                # f'/wholebrain/scratch/amancu/mergeError/preds/lcp/ConvPoint/{radius}/SGD_CyclicLR/' + os.path.basename(
                    pred_files[i]) + f'_original_{ii}.ply'),
                [np.array([]), sample_pts[0, :, :], np.array([])], colors)
            #
            # # prediction
            # colors = np.full(shape=(sample_pts.shape[1], 4,), fill_value=GREY)
            # mask = np.where(pred == 1)[0].astype(np.int64)
            # mask = np.array([[x] for x in mask])
            # try:
            #     np.put_along_axis(colors, mask, RED, axis=0)
            # except:
            #     # print("No foreground labels in prediction.")
            #     pass
            # mesh2obj_file_colors(os.path.expanduser(
            #     f'/wholebrain/scratch/amancu/mergeError/preds/lcp/ConvPoint/1000/Adam_ExponentialLR/' + os.path.basename(
            #     # f'/wholebrain/scratch/amancu/mergeError/preds/lcp/ConvPoint/1000/SGD_CyclicLR/' + os.path.basename(
            #         pred_files[i]) + f'_prediction_{ii}.ply'),
            #     [np.array([]), sample_pts[0, :, :], np.array([])], colors)
            ii += 1

        # "merge" contexts and their predictions, taking the majority vote over all predictions per vertex
        predictions = np.concatenate(predictions)
        pred_indices = np.concatenate(pred_indices)
        print(f'pred shape {predictions.shape}, indcs shape {pred_indices.shape}')
        pred_labels = np.ones((len(hc.vertices), 1)) * (-1)
        evaluate_preds(pred_indices, predictions, pred_labels)
        print(f'Number of foreground labels in whole cell: {len(np.where(pred_labels == 1)[0])}')

        pred = pred.astype(np.float64)
        # evaluate cell metric results
        true_labels = hc.labels
        precision = precision_score(true_labels, pred_labels, average='binary', zero_division=0)
        recall = recall_score(true_labels, pred_labels, average='binary', zero_division=0)
        accuracy = accuracy_score(true_labels, pred_labels)
        f1score = f1_score(true_labels, pred_labels, average='binary', zero_division=0)
        # for precision recall curve
        # prec_rec = []
        # auc = []
        # for i in range(2):
        #     prec, rec, thresh = precision_recall_curve(true_labels[np.where(true_labels==i)], pred_labels[np.where(pred_labels==i)])
        #     prec_rec.append((prec, rec))
        #     auc.append(average_precision_score(true_labels[np.where(true_labels==i)], pred_labels[np.where(pred_labels==i)]))

        # TODO metrics on cell prediction
        # pro cell precision, recall, accuracy, fscore
        # -> global metrics

        # append metrics for a cell pair
        lock.acquire()
        try:
            result_dc['precision'].append(precision)
            result_dc['recall'].append(recall)
            result_dc['accuracy'].append(accuracy)
            result_dc['fscore'].append(f1score)
            # result_dc['pr_curve'].append((prec_rec, auc))
        finally:
            lock.release()


def extract_subhcs(hc: HybridCloud, ctx_size, ctx_dst_fac, npoints, transform: Callable):
    # choose base nodes with context overlap
    base_node_dst = ctx_size / ctx_dst_fac
    # print(f'base node dist {base_node_dst}')
    # select source nodes for context extraction
    pcd = o3d.geometry.PointCloud()
    # transform hc nodes to voxel coordinates
    # nodes = hc.nodes / [10,10,25]
    pcd.points = o3d.utility.Vector3dVector(hc.nodes)

    # TODO ANSCHAUEN
    pcd, idcs = pcd.voxel_down_sample_and_trace(base_node_dst, pcd.get_min_bound(), pcd.get_max_bound())

    source_nodes = np.max(idcs, axis=1)
    bs = 1
    n_batches = int(np.ceil(len(source_nodes) / bs))
    # print(f'nbatches: {n_batches}')
    # add additional source nodes to fill batches
    if len(source_nodes) % bs != 0:
        source_nodes = np.concatenate([np.random.choice(source_nodes, bs - len(source_nodes) % bs),
                                       source_nodes])
    node_arrs = context_splitting_kdt(hc, source_nodes, ctx_size)
    # print(f'Original {len(hc.nodes)}, extracted {len(node_arrs)}')
    # print(f'Node arrs {node_arrs}')
    # collect contexts into batches (each batch contains every n_batches contexts
    # (e.g. every 4th if n_batches = 4)
    for ii in range(n_batches):
        # initialize list of data
        batch_v = np.zeros((bs, npoints, 3))
        batch_f = np.zeros((bs, npoints), dtype=bool)
        # used later for removing cell organelles
        batch_l = np.zeros((bs, npoints, 1), dtype=bool)
        idcs_list = []
        arr_list = {'verts': batch_v,
                    'feats': batch_f,
                    'labels': batch_l,
                    'global_vert_indices': idcs_list}
        # arr_list.append((batch, batch_f, batch_mask, idcs_list))
        # generate contexts
        cnt = 0
        for node_arr in node_arrs[ii::n_batches]:
            hc_sub, idcs_sub = extract_subset(hc, node_arr)
            # replace subsets with zero vertices by another subset (this is probably very rare)
            ix = 0
            while len(hc_sub.vertices) == 0:
                if ix >= len(hc.nodes):
                    raise IndexError(f'Could not find suitable context in {ssv} during "pts_loader_cpmt".')
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
            hc_sample, idcs_sample = clouds.sample_cloud(hc_sub, npoints)
            # get vertex indices respective to total hc
            global_idcs = idcs_sub[idcs_sample.astype(int)]
            # prepare masks for filtering sv vertices
            # bounds = hc.obj_bounds['sv']
            # sv_mask = np.logical_and(global_idcs < bounds[1], global_idcs >= bounds[0])
            # hc_sample.set_features(label_binarize(hc_sample.features, classes=np.arange(len(feat_dc))))
            if transform is not None:
                transform(hc_sample)
            arr_list['verts'][cnt] = hc_sample.vertices
            arr_list['feats'][cnt] = hc_sample.features
            # masks get used later when mapping predictions back onto the cell surface during postprocessing
            arr_list['labels'][cnt] = hc_sample.labels
            arr_list['global_vert_indices'].append(global_idcs)
            cnt += 1
        # batch_progress = ii + 1
        yield (arr_list['feats'], arr_list['verts'], arr_list['labels']), arr_list['global_vert_indices']


# cs_merge_radii = [100, 500, 1000, 2000, 5000]
# cs_merge_radii = [2000]
radii = [1000]
radius = 1000
# colors for labels
RED = np.array([255., 125., 125., 255.])
GREY = np.array([180., 180., 180., 255.])
nproc = 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prediction pipeline for merge error detection')
    parser.add_argument('--r', type=int, help='Radius of cs merger',
                        default=100)
    args = parser.parse_args()
    radius = args.r

    # define model args
    input_channels = 1
    num_classes = 2
    use_norm = 'gn'
    dr = 0.2
    track_running_stats = False
    act = 'relu'
    use_bias = True
    npoints = int(15e3)
    scale_norm = 5e3
    ctx_size = 20e3
    ctx_dst_fac = 3
    pred_transform = clouds.Compose([clouds.Center(), clouds.Normalization(scale_norm)])

    lcp_flag = True

    torch.multiprocessing.set_start_method('spawn')
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    for radius in radii:
        # test set
        # folder = f'/wholebrain/scratch/amancu/mergeError/test_dataset/R{radius}/*.pkl'
        # training set
        folder = f'/wholebrain/scratch/amancu/mergeError/ptclouds/R{radius}/Hybridcloud/*.pkl'
        if lcp_flag:
            # save_path = f'/wholebrain/scratch/amancu/mergeError/trainings/lcp/mergeError_pts_model_lcp_radius{radius}_eval0_ConvPoint_SearchQuantized/state_dict.pth'
            # save_path = f'/wholebrain/scratch/amancu/mergeError/trainings/lcp/mergeError_pts_model_lcp_radius{radius}_ConvPoint_SearchQuantized_Adam/state_dict.pth'
            # save_path = f'/wholebrain/scratch/amancu/mergeError/trainings/lcp/mergeError_pts_model_lcp_radius{radius}_ConvPoint_SearchQuantized_Adam_weights1,4/state_dict.pth'
            save_path = f'/wholebrain/scratch/amancu/mergeError/trainings/lcp/see/mergeError_pts_model_lcp_radius1000_ConvPoint_SearchQuantized_Adam_ExponentialLR_weights1,2/state_dict.pth'
            # save_path = f'/wholebrain/scratch/amancu/mergeError/trainings/lcp/mergeError_pts_model_lcp_radius{radius}_ConvPoint_SearchQuantized_SGD_CyclicLR/state_dict.pth'
            # save_path = f'/wholebrain/scratch/amancu/mergeError/trainings/lcp/mergeError_pts_model_lcp_radius{radius}_ConvPoint_SearchQuantized_SGD_CyclicLR_weights1,2/state_dict.pth'
            # save_path = f'/wholebrain/scratch/amancu/mergeError/trainings/lcp/mergeError_pts_model_lcp_radius{radius}_eval0_FKAConv_SearchQuantized/state_dict.pth'
        # pred_files = glob.glob(folder)
        pred_files = ['/wholebrain/scratch/amancu/mergeError/ptclouds/R1000/Hybridcloud/sso_240259084_314052640.pkl']

        model_name = os.path.basename(os.path.dirname(save_path))
        print(f'Predicting for radius {radius} with nr of files {len(pred_files)} on {model_name}')

        with torch.no_grad():
            if lcp_flag:
                search = 'SearchQuantized'
                conv = dict(layer='ConvPoint', kernel_separation=False)
                # conv = dict(layer='FKAConv', kernel_separation=False)
                act = torch.nn.ReLU
                model = ConvAdaptSeg(input_channels, num_classes, get_conv(conv), get_search(search), kernel_num=64,
                                     architecture=None, activation=act, norm='gn').to(device)
            else:
                model = SegSmall(input_channels, num_classes + 1, dropout=dr, use_norm=use_norm,
                                 track_running_stats=track_running_stats, act=act, use_bias=use_bias).to(device)
            model.load_state_dict(torch.load(save_path, map_location=device)['model_state_dict'])
            # print(model)
            model.eval()

            # dictionary with lists of all metrics for each cell pair
            result_dc = {
                'precision': [],
                'recall': [],
                'accuracy': [],
                'fscore': [],
                'pr_curve': []
            }

            # split tasks for processes
            proc_slices = []
            offset = len(pred_files) // nproc
            for i in range(nproc):
                slice_start = offset * i
                slice_end = offset * i if i < nproc - 1 else len(pred_files)
                proc_slices.append(np.s_[slice_start:slice_end])

            result_dict_lock = mp.Lock()

            params = [(slice, pred_files, result_dict_lock, model, ctx_size, ctx_dst_fac, npoints, pred_transform,
                       device, lcp_flag, result_dc) for slice in proc_slices]

            running_tasks = [mp.Process(target=process_data_slice, args=param) for param in params]
            for running_task in running_tasks:
                running_task.start()
            for running_task in running_tasks:
                running_task.join()

            print(f'Processing finished')

            precisions = result_dc['precision']
            recalls = result_dc['recall']
            accuracies = result_dc['accuracy']
            fscores = result_dc['fscore']
            pr_curve = result_dc['pr_curve']
            print(f'prec {precisions}')
            print(f'recalls {recalls}')
            print(f'accur {accuracies}')
            print(f'fscores {fscores}')

            result = {
                'model name': model_name,
                'precision': np.mean(precisions),
                'recall': np.mean(recalls),
                'accuracy': np.mean(accuracies),
                'fscore': np.mean(fscores),
            }

            vals = list(result.values())

            print(f'Number zero Fscores: {len(np.where(fscores == 0.0)[0])}')
            print(f'Mean precisions: {vals[1]}')
            print(f'Mean recall: {vals[2]}')
            print(f'Mean accuracy: {vals[3]}')
            print(f'Mean fscores: {vals[4]}')

            # # plot precision recall curve
            # plt.figure()
            # # Plot Precision-Recall curve
            # target_names = ['background', 'foreground']
            # for i in range(2):
            #     lines, = plt.plot(pr_curve[0][1], pr_curve[0], lw=3,
            #                       label='%s: %0.4f' % (target_names[i], pr_curve[2]))
            # plt.xlabel('Recall')
            # plt.ylabel('Precision')
            # plt.ylim([0.0, 1.05])
            # plt.xlim([0.0, 1.05])
            # plt.title('Precision-Recall')
            # plt.legend(loc="lower left")
            # plt.show(block=False)
            # plt.savefig(fold + "/%s_valid_prec_rec.png" % prefix)

            df = pd.DataFrame.from_dict(result)
            csv_path = f'/wholebrain/scratch/amancu/mergeError/preds/lcp/ConvPoint/1000/Adam_ExponentialLR/{model_name}.csv'
            df.to_csv(csv_path)
