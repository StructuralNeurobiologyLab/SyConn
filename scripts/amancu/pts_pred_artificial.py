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
import random as ran

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
from scipy.spatial import cKDTree


def merge_dict(dict1, dict2):
    ''' Merge dictionaries and keep values of common keys in list'''
    # dict3 = {}
    for key, value in dict1.items():
        dict1[key].extend(dict2[key])
    return dict1

def merge_multiple_dicts(list_of_dicts):
    res = {
        'precision': [],
        'recall': [],
        'accuracy': [],
        'fscore': [],
    }
    for d in list_of_dicts:
        merge_dict(res,d)
        print(res)
        print(f'in merge')
    return res


def process_data_slice(slice, pred_files, model, ctx_size, ctx_dst_fac, npoints, pred_transform, device, lcp_flag, queue):
    """
    Adds average results of metrics (precision, recall, accuracy, f1score) per cell in the res_dc dictionary of results.

    Args:
        slice: np.s_ slice to process the files
        pred_files: file paths containing HybridCLoud pickles

    """
    hc = HybridCloud()
    res_dc = {
        'precision': [],
        'recall': [],
        'accuracy': [],
        'fscore': [],
    }
    # for each whole cells
    for i in tqdm(pred_files[slice], desc='Predict HCs'):
        path = i
        hc.load_from_pkl(i)
        predictions = []
        pred_indices = []

        vert_tree = cKDTree(data=hc.vertices,)
        ii = 0
        source_nodes=[]
        for (sample_feats, sample_pts, sample_labels), source_node, mask, vert_indices in extract_subhcs(hc, ctx_size, ctx_dst_fac,
                                                                                      npoints, pred_transform):

            sample_feats = sample_feats[:, :, None]
            source_node = int(source_node[0,0])
            source_nodes.append(source_node)
            # print(f'source node: {source_node}')

            dpts = torch.from_numpy(sample_pts).to(device).float()
            dfeats = torch.from_numpy(sample_feats).to(device).float()

            if lcp_flag:
                dpts = dpts.transpose(1, 2)
                dfeats = dfeats.transpose(1, 2)

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

            # place the external context of the prediction on null, so that they focus only on the middle
            pred= pred[mask[0,:]]

            predictions.append(pred)
            pred_indices.append(vert_indices[0])
            ii += 1

        # "merge" contexts and their predictions, taking the majority vote over all predictions per vertex
        predictions = np.concatenate(predictions)
        pred_indices = np.concatenate(pred_indices)

        # 0: background, 1: foreground, 3: no prediction -> for the use of bincount
        pred_labels = np.ones((len(hc.vertices), 1)) * (3)
        # pred labels will have the vertices labels of values [0,1,3]
        evaluate_preds(pred_indices, predictions, pred_labels)

        if len(np.where(pred_labels==3)[0]) != 0:
            # print(f'Found {os.path.basename(path)} with {len(np.where(pred_labels==3)[0])} unpredicted (3) labels')
            pred_labels[np.where(pred_labels==3)[0]] = int(0)

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
        # labeled_indices = np.where(pred_labels!=3)[0]
        # labeled_vertices = hc.vertices[labeled_indices]
        # labeled_tree = cKDTree(data=labeled_vertices, )
        # vert_idcs = np.where(pred_labels == 3)[0]
        # unlabeled_verts = hc.vertices[vert_idcs]
        # labeled_neighbors = labeled_tree.query_ball_point(x=unlabeled_verts, r=500)
        # # print(f'neighbors {labeled_neighbors}')
        # for i, cluster in enumerate(labeled_neighbors):
        #     if len(cluster) == 0:
        #         pred_labels[vert_idcs[i]] = int(0)
        #     cnt = np.bincount(pred_labels[labeled_indices[cluster]][:, 0].astype(np.int64))
        #     try:
        #         vert_label = np.argmax(cnt)
        #     except:
        #         print(f'Could not find neighbor in propagation for cell pair {os.path.basename(path)} and vert {unlabeled_verts[i]} in idc of hc.vertices {vert_idcs[i]}')
        #         vert_label = int(0)
        #     pred_labels[vert_idcs[i]] = vert_label

        # evaluate cell metric results
        true_labels = hc.labels

        # calculate true and pred node labels
        true_node_labels = np.zeros(shape=(hc.nodes.shape[0],))
        pred_node_labels = np.zeros(shape=(hc.nodes.shape[0],))
        for i, node in enumerate(hc.nodes):
            try:
                vert_idcs = vert_tree.query_ball_point(x=node, r=700)
                # vert_idcs = vert_tree.query(x=node, k=10)
                vert_true = true_labels[vert_idcs]
                vert_preds = pred_labels[vert_idcs]
                cnt_true = np.bincount(vert_true[:,0].astype(np.int64))
                cnt_pred = np.bincount(vert_preds[:,0].astype(np.int64))
                true_node_label = np.argmax(cnt_true)
                pred_node_label = np.argmax(cnt_pred)
                true_node_labels[i] = true_node_label
                pred_node_labels[i] = pred_node_label
            except:
                true_node_labels[i] = int(0)
                pred_node_labels[i] = int(0)

        # # prediction per vertex
        # precision = precision_score(true_labels, pred_labels, average='binary', zero_division=0)
        # recall = recall_score(true_labels, pred_labels, average='binary', zero_division=0)
        # accuracy = accuracy_score(true_labels, pred_labels)
        # f1score = f1_score(true_labels, pred_labels, average='binary', zero_division=0)

        # prediction per skeleton node
        precision = precision_score(true_node_labels, pred_node_labels, average='binary', zero_division=0)
        recall = recall_score(true_node_labels, pred_node_labels, average='binary', zero_division=0)
        accuracy = accuracy_score(true_node_labels, pred_node_labels)
        f1score = f1_score(true_node_labels, pred_node_labels, average='binary', zero_division=0)

        res_dc['precision'].append(precision)
        res_dc['recall'].append(recall)
        res_dc['accuracy'].append(accuracy)
        res_dc['fscore'].append(f1score)

        # uncomment for vertices render
        # if ran.random() > 0.05:
        #     print(f'For {os.path.basename(path)} \n Precision: {precision} \n Recall: {recall} \n Accuracy: {accuracy} \n Fscore: {f1score}')
        #     # original nodes
        #     colors = np.full(shape=(hc.vertices.shape[0], 4,), fill_value=GREY)
        #     mask = np.where(hc.labels == 1)[0]
        #     mask = np.array([[x] for x in mask])
        #     try:
        #         np.put_along_axis(colors, mask, PINK, axis=0)
        #     except:
        #         # print("No foreground labels in original context.")
        #         pass
        #     mesh2obj_file_colors(os.path.expanduser(
        #         f'/wholebrain/scratch/amancu/mergeError/preds/lcp/ConvPoint/contexts/' + os.path.basename(
        #             path) + f'_original_verts.ply'),
        #         [np.array([]), hc.vertices, np.array([])], colors)
        #
        #     # prediction
        #     colors = np.full(shape=(hc.vertices.shape[0], 4,), fill_value=GREY)
        #     mask = np.where(pred_labels == 1)[0].astype(np.int64)
        #     mask = np.array([[x] for x in mask])
        #     try:
        #         np.put_along_axis(colors, mask, PINK, axis=0)
        #     except:
        #         # print("No foreground labels in prediction.")
        #         pass
        #     mesh2obj_file_colors(os.path.expanduser(
        #         f'/wholebrain/scratch/amancu/mergeError/preds/lcp/ConvPoint/2000/contexts/' + os.path.basename(
        #             path) + f'_prediction_verts.ply'),
        #         [np.array([]), hc.vertices, np.array([])], colors)


        # random node predictions to inspect in meshlab
        if ran.random() > 0.95:
            print(f'For {os.path.basename(path)} \n Precision: {precision} \n Recall: {recall} \n Accuracy: {accuracy} \n Fscore: {f1score}')
            # original nodes
            colors = np.full(shape=(hc.nodes.shape[0], 4,), fill_value=GREY)
            mask = np.where(hc.node_labels == 1)[0]
            mask = np.concatenate((mask, np.where(hc.node_labels == 2)[0]))
            mask = np.array([[x] for x in mask])
            try:
                np.put_along_axis(colors, mask, PINK, axis=0)
            except:
                # print("No foreground labels in original context.")
                pass
            mesh2obj_file_colors(os.path.expanduser(
                # f'/wholebrain/scratch/amancu/mergeError/preds/lcp/ConvPoint/2000/archLrg/testSet/' + os.path.basename(
                f'/wholebrain/scratch/amancu/mergeError/preds/lcp/ConvPoint/2000/arch2048/meshes/' + os.path.basename(
                # f'/wholebrain/scratch/amancu/mergeError/preds/lcp/ConvPoint/2000/test/parallel/' + os.path.basename(
                    path) + f'_run5_original_nodes.ply'),
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
                # f'/wholebrain/scratch/amancu/mergeError/preds/lcp/ConvPoint/2000/archLrg/testSet/' + os.path.basename(
                f'/wholebrain/scratch/amancu/mergeError/preds/lcp/ConvPoint/2000/arch2048/meshes/' + os.path.basename(
                # f'/wholebrain/scratch/amancu/mergeError/preds/lcp/ConvPoint/2000/test/parallel/' + os.path.basename(
                    path) + f'_run5_prediction_nodes.ply'),
                [np.array([]), hc.nodes, np.array([])], colors)
    print(res_dc)
    queue.put(res_dc)

def extract_subhcs(hc: HybridCloud, ctx_size, ctx_dst_fac, npoints, transform: Callable):
    # choose base nodes with context overlap
    base_node_dst = ctx_size / ctx_dst_fac
    # print(f'dist {base_node_dst}')
    # print(f'base node dist {base_node_dst}')
    # select source nodes for context extraction
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(hc.nodes)
    pcd, idcs = pcd.voxel_down_sample_and_trace(base_node_dst, pcd.get_min_bound(), pcd.get_max_bound())
    # print(f'idcs {idcs}')
    source_nodes = np.max(idcs, axis=1)
    bs = 1
    n_batches = int(np.ceil(len(source_nodes) / bs))
    # print(f'nbatches: {n_batches}')
    # add additional source nodes to fill batches
    if len(source_nodes) % bs != 0:
        source_nodes = np.concatenate([np.random.choice(source_nodes, bs - len(source_nodes) % bs),
                                       source_nodes])
    node_arrs = context_splitting_kdt(hc, source_nodes, ctx_size)
    # print(f'source nodes len {len(source_nodes)} and node arrs len {len(node_arrs)}')
    # print(f'src_nodes: {source_nodes}')
    # print(f'Extracted number of contexts {len(node_arrs)}')

    # collect contexts into batches (each batch contains every n_batches contexts
    # (e.g. every 4th if n_batches = 4)
    for ii in range(n_batches):
        # initialize list of data
        batch_v = np.zeros((bs, npoints, 3))
        batch_f = np.zeros((bs, npoints), dtype=bool)
        mask = np.zeros((bs, npoints), dtype=bool)
        batch_sn = np.zeros((bs,1))
        # used later for removing cell organelles
        batch_l = np.zeros((bs, npoints, 1), dtype=bool)
        idcs_list = []
        arr_list = {'verts': batch_v,
                    'feats': batch_f,
                    'labels': batch_l,
                    'margin_mask': mask,
                    'source_node': batch_sn,
                    'global_vert_indices': idcs_list}
        # arr_list.append((batch, batch_f, batch_mask, idcs_list))
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
            # fill batches with sampled and transformed subsets
            hc_sample, idcs_sample = clouds.sample_cloud(hc_sub, npoints)
            tree = cKDTree(data=hc_sample.vertices)
            inner_ids = tree.query_ball_point(hc.nodes[source_nodes[ii]], r=15000)
            mask = np.ones(shape=(len(hc_sample.vertices),), dtype=bool)
            mask[inner_ids] = bool(0)
            inner_mask = np.zeros(shape=(len(hc_sample.vertices),), dtype=bool)
            inner_mask[inner_ids] = bool(1)
            # get vertex indices respective to total hc
            global_idcs = idcs_sub[idcs_sample.astype(int)]
            global_idcs = global_idcs[inner_mask]
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
            # arr_list['margin_mask'][cnt] = mask
            arr_list['margin_mask'][cnt] = inner_mask
            arr_list['source_node'][cnt] = source_nodes[ii]
            arr_list['global_vert_indices'].append(global_idcs)
            cnt += 1
        yield (arr_list['feats'], arr_list['verts'], arr_list['labels']), arr_list['source_node'], arr_list['margin_mask'], arr_list['global_vert_indices']


# cs_merge_radii = [100, 500, 1000, 2000, 5000]
# cs_merge_radii = [2000]
radii = [2000]
radius = 2000
# colors for labels
PINK = np.array([10., 255., 10., 255.])
BLUE = np.array([255., 125., 125., 255.])
GREY = np.array([180., 180., 180., 255.])
nproc = 50

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prediction pipeline for merge error detection')
    parser.add_argument('--r', type=int, help='Radius of cs merger',
                        default=2000)
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
    npoints = int(10e3)
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
        # folder = f'/wholebrain/scratch/amancu/mergeError/ptclouds/R{radius}/Hybridcloud/*.pkl'
        folder = f'/wholebrain/scratch/amancu/mergeError/test_dataset/R{radius}/*.pkl'
        if lcp_flag:
            save_path = f'/wholebrain/scratch/amancu/mergeError/trainings/lcp/see/lcp_r2000_ConvPoint_SearchQuantized_arch2048_run5_Adam_StepLR_weights1,2_CrossEntropy/state_dict.pth'
            # save_path = f'/wholebrain/scratch/amancu/mergeError/trainings/lcp/see/lcp_r2000_ConvPoint_SearchQuantized_archLrg_run6_Adam_StepLR_weights1,2_CrossEntropy/state_dict.pth'
        pred_files = glob.glob(folder)
        # pred_files = ['/wholebrain/scratch/amancu/mergeError/ptclouds/R2000/Hybridcloud/sso_3807046_15922193.pkl',
        #             '/wholebrain/scratch/amancu/mergeError/ptclouds/R2000/Hybridcloud/sso_44948783_70288384.pkl',
        #             '/wholebrain/scratch/amancu/mergeError/ptclouds/R2000/Hybridcloud/sso_209549121_209742637.pkl',
        #             '/wholebrain/scratch/amancu/mergeError/ptclouds/R2000/Hybridcloud/sso_9529398_19721758.pkl',
        #             '/wholebrain/scratch/amancu/mergeError/ptclouds/R2000/Hybridcloud/sso_42482050_43253052.pkl']
        # pred_files = ['/wholebrain/scratch/amancu/mergeError/ptclouds/R2000/Hybridcloud/sso_1574929_1767377.pkl',
        #             '/wholebrain/scratch/amancu/mergeError/ptclouds/R2000/Hybridcloud/sso_209549121_209742637.pkl',
        #             '/wholebrain/scratch/amancu/mergeError/ptclouds/R2000/Hybridcloud/sso_130650624_188296720.pkl',
        #             '/wholebrain/scratch/amancu/mergeError/ptclouds/R2000/Hybridcloud/sso_9529398_19721758.pkl',
        #             '/wholebrain/scratch/amancu/mergeError/ptclouds/R2000/Hybridcloud/sso_226082525_238186872.pkl',
        #             '/wholebrain/scratch/amancu/mergeError/ptclouds/R2000/Hybridcloud/sso_40022718_113258940.pkl',
        #             '/wholebrain/scratch/amancu/mergeError/ptclouds/R2000/Hybridcloud/sso_28958042_54095396.pkl',
        #             '/wholebrain/scratch/amancu/mergeError/ptclouds/R2000/Hybridcloud/sso_2208821_2571960.pkl',
        #             '/wholebrain/scratch/amancu/mergeError/ptclouds/R2000/Hybridcloud/sso_217066591_241081409.pkl',
        #             '/wholebrain/scratch/amancu/mergeError/ptclouds/R2000/Hybridcloud/sso_3807046_15922193.pkl',            # 10
        #             '/wholebrain/scratch/amancu/mergeError/ptclouds/R2000/Hybridcloud/sso_212101191_232639528.pkl',
        #             '/wholebrain/scratch/amancu/mergeError/ptclouds/R2000/Hybridcloud/sso_42482050_43253052.pkl',
        #             '/wholebrain/scratch/amancu/mergeError/ptclouds/R2000/Hybridcloud/sso_50307606_160236492.pkl',
        #             '/wholebrain/scratch/amancu/mergeError/ptclouds/R2000/Hybridcloud/sso_42866227_162377533.pkl',
        #             '/wholebrain/scratch/amancu/mergeError/ptclouds/R2000/Hybridcloud/sso_44948783_70288384.pkl',
        #             '/wholebrain/scratch/amancu/mergeError/ptclouds/R2000/Hybridcloud/sso_50792978_303842028.pkl',
        #             '/wholebrain/scratch/amancu/mergeError/ptclouds/R2000/Hybridcloud/sso_130750890_177395993.pkl',
        #             '/wholebrain/scratch/amancu/mergeError/ptclouds/R2000/Hybridcloud/sso_1437627_1443233.pkl',
        #             '/wholebrain/scratch/amancu/mergeError/ptclouds/R2000/Hybridcloud/sso_19450892_82261946.pkl',
        #             # '/wholebrain/scratch/amancu/mergeError/ptclouds/R2000/Hybridcloud/sso_42866227_162377533.pkl',
        #               ]
        # pred_files = ['/wholebrain/scratch/amancu/mergeError/ptclouds/R2000/Hybridcloud/sso_130750890_177395993.pkl',
        #               '/wholebrain/scratch/amancu/mergeError/ptclouds/R2000/Hybridcloud/sso_217066591_241081409.pkl',
        #               '/wholebrain/scratch/amancu/mergeError/ptclouds/R2000/Hybridcloud/sso_135276895_160863880.pkl',
        #               '/wholebrain/scratch/amancu/mergeError/ptclouds/R2000/Hybridcloud/sso_249125218_261419284.pkl',
        #               '/wholebrain/scratch/amancu/mergeError/ptclouds/R2000/Hybridcloud/sso_9529398_19721758.pkl',
        #               '/wholebrain/scratch/amancu/mergeError/ptclouds/R2000/Hybridcloud/sso_1037781_24879303.pkl',
        #               '/wholebrain/scratch/amancu/mergeError/ptclouds/R2000/Hybridcloud/sso_209549121_209742637.pkl',
        #               '/wholebrain/scratch/amancu/mergeError/ptclouds/R2000/Hybridcloud/sso_3807046_15922193.pkl',
        #               '/wholebrain/scratch/amancu/mergeError/ptclouds/R2000/Hybridcloud/sso_50307606_160236492.pkl',
        #               '/wholebrain/scratch/amancu/mergeError/ptclouds/R2000/Hybridcloud/sso_28958042_54095396.pkl',
        #               # 10
        #               '/wholebrain/scratch/amancu/mergeError/ptclouds/R2000/Hybridcloud/sso_130650624_188296720.pkl',
        #               '/wholebrain/scratch/amancu/mergeError/ptclouds/R2000/Hybridcloud/sso_86196746_158670784.pkl',
        #               '/wholebrain/scratch/amancu/mergeError/ptclouds/R2000/Hybridcloud/sso_42482050_43253052.pkl',
        #               '/wholebrain/scratch/amancu/mergeError/ptclouds/R2000/Hybridcloud/sso_109957093_122059097.pkl',
        #               '/wholebrain/scratch/amancu/mergeError/ptclouds/R2000/Hybridcloud/sso_44948783_70288384.pkl',
        #               '/wholebrain/scratch/amancu/mergeError/ptclouds/R2000/Hybridcloud/sso_40022718_113258940.pkl',
        #               '/wholebrain/scratch/amancu/mergeError/ptclouds/R2000/Hybridcloud/sso_140734639_140923411.pkl',
        #               '/wholebrain/scratch/amancu/mergeError/ptclouds/R2000/Hybridcloud/sso_1437627_1443233.pkl',
        #               '/wholebrain/scratch/amancu/mergeError/ptclouds/R2000/Hybridcloud/sso_1574929_1767377.pkl',
        #               # '/wholebrain/scratch/amancu/mergeError/ptclouds/R2000/Hybridcloud/sso_42866227_162377533.pkl',
        #               ]
        # pred_files = ['/wholebrain/scratch/amancu/mergeError/ptclouds/R2000/Hybridcloud/sso_42482050_43253052.pkl']

        model_name = os.path.basename(os.path.dirname(save_path))
        print(f'Predicting for radius {radius} with nr of files {len(pred_files)} on {model_name}')

        with torch.no_grad():
            if lcp_flag:
                search = 'SearchQuantized'
                conv = dict(layer='ConvPoint', kernel_separation=False)
                # conv = dict(layer='FKAConv', kernel_separation=False)
                act = torch.nn.ReLU
                architecture_2048 = [{'ic': -1, 'oc': 1, 'ks': 16, 'nn': 32, 'np': -1},
                                     {'ic': 1, 'oc': 1, 'ks': 16, 'nn': 32, 'np': 1024},
                                     {'ic': 1, 'oc': 1, 'ks': 16, 'nn': 32, 'np': 512},
                                     {'ic': 1, 'oc': 1, 'ks': 16, 'nn': 32, 'np': 256},
                                     {'ic': 1, 'oc': 2, 'ks': 16, 'nn': 32, 'np': 64},
                                     {'ic': 2, 'oc': 2, 'ks': 16, 'nn': 16, 'np': 16},
                                     {'ic': 2, 'oc': 2, 'ks': 16, 'nn': 8, 'np': 8},
                                     {'ic': 2, 'oc': 2, 'ks': 16, 'nn': 4, 'np': 'd'},
                                     {'ic': 4, 'oc': 2, 'ks': 16, 'nn': 4, 'np': 'd'},
                                     {'ic': 4, 'oc': 1, 'ks': 16, 'nn': 8, 'np': 'd'},
                                     {'ic': 2, 'oc': 1, 'ks': 16, 'nn': 16, 'np': 'd'},
                                     {'ic': 2, 'oc': 1, 'ks': 16, 'nn': 16, 'np': 'd'},
                                     {'ic': 2, 'oc': 1, 'ks': 16, 'nn': 16, 'np': 'd'}]
                architecture_large = [{'ic': -1, 'oc': 1, 'ks': 16, 'nn': 32, 'np': -1},
                                      {'ic': 1, 'oc': 1, 'ks': 16, 'nn': 32, 'np': 2048},
                                      {'ic': 1, 'oc': 1, 'ks': 16, 'nn': 32, 'np': 1024},
                                      {'ic': 1, 'oc': 1, 'ks': 16, 'nn': 32, 'np': 256},
                                      {'ic': 1, 'oc': 2, 'ks': 16, 'nn': 32, 'np': 64},
                                      {'ic': 2, 'oc': 2, 'ks': 16, 'nn': 16, 'np': 16},
                                      {'ic': 2, 'oc': 2, 'ks': 16, 'nn': 8, 'np': 8},
                                      {'ic': 2, 'oc': 2, 'ks': 16, 'nn': 4, 'np': 'd'},
                                      {'ic': 4, 'oc': 2, 'ks': 16, 'nn': 4, 'np': 'd'},
                                      {'ic': 4, 'oc': 1, 'ks': 16, 'nn': 8, 'np': 'd'},
                                      {'ic': 2, 'oc': 1, 'ks': 16, 'nn': 16, 'np': 'd'},
                                      {'ic': 2, 'oc': 1, 'ks': 16, 'nn': 16, 'np': 'd'},
                                      {'ic': 2, 'oc': 1, 'ks': 16, 'nn': 16, 'np': 'd'}]
                model = ConvAdaptSeg(input_channels, num_classes, get_conv(conv), get_search(search), kernel_num=64,
                                     architecture=architecture_2048, activation=act, norm='gn').to(device)
            else:
                model = SegSmall(input_channels, num_classes + 1, dropout=dr, use_norm=use_norm,
                                 track_running_stats=track_running_stats, act=act, use_bias=use_bias).to(device)
            model.load_state_dict(torch.load(save_path, map_location=device)['model_state_dict'])
            # print(model)
            model.eval()

            # dictionary with lists of all metrics for each cell pair
            global result_dc
            result_dc = {
                'precision': [],
                'recall': [],
                'accuracy': [],
                'fscore': [],
            }

            # split tasks for processes
            proc_slices = []
            offset = len(pred_files) // nproc
            # offset = 10 // nproc
            for i in range(nproc):
                slice_start = offset * i
                slice_end = offset * (i+1) if i < nproc - 1 else len(pred_files)
                # slice_end = offset * (i+1) if i < nproc - 1 else 10
                proc_slices.append(np.s_[slice_start:slice_end])

            print(f'slices {proc_slices}')
            queue = mp.Queue()
            jobs = []
            running_tasks = []
            params = [(slice, pred_files, model, ctx_size, ctx_dst_fac, npoints, pred_transform,
                       device, lcp_flag, queue) for slice in proc_slices]

            running_tasks = [mp.Process(target=process_data_slice, args=param) for param in params]
            for running_task in running_tasks:
                running_task.start()
            for running_task in running_tasks:
                running_task.join()
            print(f'Processing finished')

            # get results
            results = [queue.get() for task in running_tasks]
            print(results)
            result_dc = merge_multiple_dicts(results)

            precisions = result_dc['precision']
            recalls = result_dc['recall']
            accuracies = result_dc['accuracy']
            fscores = result_dc['fscore']
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

            df = pd.DataFrame.from_dict(result, orient='index')
            # csv_path = f'/wholebrain/scratch/amancu/mergeError/preds/lcp/ConvPoint/2000/archLrg/{model_name}_testSet_context_focus.csv'
            csv_path = f'/wholebrain/scratch/amancu/mergeError/preds/lcp/ConvPoint/2000/arch2048/{model_name}_testSet_context_focus.csv'
            # csv_path = f'/wholebrain/scratch/amancu/mergeError/preds/lcp/ConvPoint/2000/test/parallel/{model_name}.csv'
            df.to_csv(csv_path)
