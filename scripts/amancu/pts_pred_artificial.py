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


def process_data_slice(slice, pred_files, model, ctx_size, ctx_dst_fac, npoints, pred_transform, device, lcp_flag,
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
            ii += 1

        # "merge" contexts and their predictions, taking the majority vote over all predictions per vertex
        predictions = np.concatenate(predictions)
        pred_indices = np.concatenate(pred_indices)

        # 0: background, 1: foreground, 3: no prediction -> for the use of bincount
        pred_labels = np.ones((len(hc.vertices), 1)) * (3)
        # pred labels will have the vertices labels of values [0,1,3]
        evaluate_preds(pred_indices, predictions, pred_labels)

        # for 3 labeled vertices, we employ a kdtree approach to inherit the label of the neighbouring vertices
        tree = cKDTree(data=hc.vertices, )
        # loop until there are no more non-predicted labels
        while np.any(pred_labels == 3):
            vert_idcs = np.where(pred_labels == 3)[0]
            verts = hc.vertices[vert_idcs]
            for i, vert in enumerate(verts):
                res = tree.query_ball_point(x=vert, r=400)              # gets approx 10 neighbors
                # cast majority vote on neighbor predictions
                cnt = np.bincount(pred_labels[res][:,0].astype(np.int64))
                vert_label = np.argmax(cnt)
                pred_labels[vert_idcs[i]] = vert_label

        # evaluate cell metric results
        true_labels = hc.labels

        vert_tree = cKDTree(data=hc.vertices,)
        # calculate true node labels
        true_node_labels = np.zeros(shape=(hc.nodes.shape[0],))
        pred_node_labels = np.zeros(shape=(hc.nodes.shape[0],))
        for i, node in enumerate(hc.nodes):
            vert_idcs = vert_tree.query_ball_point(x=node, r=700)
            vert_true = true_labels[vert_idcs]
            vert_preds = pred_labels[vert_idcs]
            cnt_true = np.bincount(vert_true[:,0].astype(np.int64))
            cnt_pred = np.bincount(vert_preds[:,0].astype(np.int64))
            true_node_label = np.argmax(cnt_true)
            pred_node_label = np.argmax(cnt_pred)
            true_node_labels[i] = true_node_label
            pred_node_labels[i] = pred_node_label

        # random node predictions to inspect in meshlab
        if ran.random() > 0.8:
            # original
            colors = np.full(shape=(hc.nodes.shape[0], 4,), fill_value=GREY)
            mask = np.where(true_node_labels == 1)[0]
            mask = np.array([[x] for x in mask])
            try:
                np.put_along_axis(colors, mask, RED, axis=0)
            except:
                # print("No foreground labels in original context.")
                pass
            mesh2obj_file_colors(os.path.expanduser(
                f'/wholebrain/scratch/amancu/mergeError/preds/lcp/ConvPoint/2000/arch_2048/meshes/' + os.path.basename(
                    path) + f'_arch2048_original.ply'),
                [np.array([]), hc.nodes, np.array([])], colors)

            # prediction
            colors = np.full(shape=(hc.nodes.shape[0], 4,), fill_value=GREY)
            mask = np.where(pred_node_labels == 1)[0].astype(np.int64)
            mask = np.array([[x] for x in mask])
            try:
                np.put_along_axis(colors, mask, RED, axis=0)
            except:
                # print("No foreground labels in prediction.")
                pass
            mesh2obj_file_colors(os.path.expanduser(
                f'/wholebrain/scratch/amancu/mergeError/preds/lcp/ConvPoint/2000/arch_2048/meshes/' + os.path.basename(
                    path) + f'_arch2048_prediction.ply'),
                [np.array([]), hc.nodes, np.array([])], colors)

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

        result_dc['precision'].append(precision)
        result_dc['recall'].append(recall)
        result_dc['accuracy'].append(accuracy)
        result_dc['fscore'].append(f1score)

def extract_subhcs(hc: HybridCloud, ctx_size, ctx_dst_fac, npoints, transform: Callable):
    # choose base nodes with context overlap
    base_node_dst = ctx_size / ctx_dst_fac
    # print(f'base node dist {base_node_dst}')
    # select source nodes for context extraction
    pcd = o3d.geometry.PointCloud()
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
    # print(f'Node arrs {len(node_arrs)}')
    # print(f'HC vert num {len(hc.vertices)}')
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
radii = [2000]
radius = 2000
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
        folder = f'/wholebrain/scratch/amancu/mergeError/ptclouds/R{radius}/Hybridcloud/*.pkl'
        if lcp_flag:
            # save_path = f'/wholebrain/scratch/amancu/mergeError/trainings/lcp/mergeError_pts_model_lcp_radius{radius}_eval0_ConvPoint_SearchQuantized/state_dict.pth'
            # save_path = f'/wholebrain/scratch/amancu/mergeError/trainings/lcp/mergeError_pts_model_lcp_radius{radius}_ConvPoint_SearchQuantized_Adam/state_dict.pth'
            # save_path = f'/wholebrain/scratch/amancu/mergeError/trainings/lcp/mergeError_pts_model_lcp_radius{radius}_ConvPoint_SearchQuantized_Adam_weights1,4/state_dict.pth'
            # save_path = f'/wholebrain/scratch/amancu/mergeError/trainings/lcp/see/mergeError_pts_model_lcp_radius1000_ConvPoint_SearchQuantized_Adam_ExponentialLR_weights1,2/state_dict.pth'
            # save_path = f'/wholebrain/scratch/amancu/mergeError/trainings/lcp/mergeError_pts_model_lcp_radius{radius}_ConvPoint_SearchQuantized_SGD_CyclicLR/state_dict.pth'
            # save_path = f'/wholebrain/scratch/amancu/mergeError/trainings/lcp/mergeError_pts_model_lcp_radius{radius}_ConvPoint_SearchQuantized_SGD_CyclicLR_weights1,2/state_dict.pth'
            # save_path = f'/wholebrain/scratch/amancu/mergeError/trainings/lcp/mergeError_pts_model_lcp_radius{radius}_eval0_FKAConv_SearchQuantized/state_dict.pth'
            # save_path = f'/wholebrain/scratch/amancu/mergeError/trainings/test/see/lcp_r2000_ConvPoint_SearchQuantized_architecture1024_augmentations_bn_Adam_StepLR_weights1,2_CrossEntropy_5samples/state_dict.pth'
            # save_path = f'/wholebrain/scratch/amancu/mergeError/trainings/test/see/lcp_r2000_ConvPoint_SearchQuantized_architecture1024_augmentations_bn_Adam_StepLR_weights1,2_FocalLoss_20samples/state_dict.pth'
            # save_path = f'/wholebrain/scratch/amancu/mergeError/trainings/test/see/lcp_r2000_ConvPoint_SearchQuantized_architecture1024_augmentations_bn_Adam_StepLR_weights1,2_CrossEntropy_1sample/state_dict.pth'
            # save_path = f'/wholebrain/scratch/amancu/mergeError/trainings/test/see/lcp_r2000_ConvPoint_SearchQuantized_architecture1024_augmentations_nonorm_Adam_StepLR_weights1,2_CrossEntropy_1sample/state_dict.pth'
            # save_path = f'/wholebrain/scratch/amancu/mergeError/trainings/test/see/lcp_r2000_ConvPoint_SearchQuantized_architecture1024_augmentations_gn_Adam_StepLR_weights1,2_CrossEntropy_1sample/state_dict.pth'
            # save_path = f'/wholebrain/scratch/amancu/mergeError/trainings/test/see/lcp_r2000_ConvPoint_SearchQuantized_architecture_2048_augmentations_gn_Adam_StepLR_weights1,2_CrossEntropy_1sample/state_dict.pth'
            # save_path = f'/wholebrain/scratch/amancu/mergeError/trainings/test/see/lcp_r2000_ConvPoint_SearchQuantized_architecture_large_augmentations_gn_Adam_StepLR_weights1,2_CrossEntropy_1sample/state_dict.pth'
            save_path = f'/wholebrain/scratch/amancu/mergeError/trainings/lcp/see/lcp_r2000_ConvPoint_SearchQuantized_architecture2048_Adam_StepLR_weights1,2_CrossEntropy/state_dict.pth'
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
                architecture_1024 = [dict(ic=-1, oc=1, ks=16, nn=16, np=1024),
                                     dict(ic=1, oc=1, ks=16, nn=16, np=512),
                                     dict(ic=1, oc=1, ks=16, nn=16, np=256),
                                     dict(ic=1, oc=2, ks=16, nn=16, np=64),
                                     dict(ic=2, oc=2, ks=16, nn=16, np=16),
                                     dict(ic=2, oc=2, ks=16, nn=16, np=8),
                                     dict(ic=2, oc=2, ks=16, nn=4, np='d'),
                                     dict(ic=4, oc=2, ks=16, nn=4, np='d'),
                                     dict(ic=4, oc=1, ks=16, nn=4, np='d'),
                                     dict(ic=2, oc=1, ks=16, nn=8, np='d'),
                                     dict(ic=2, oc=1, ks=16, nn=8, np='d')]
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
                'pr_curve': []
            }

            process_data_slice(np.s_[0:len(pred_files)], pred_files, model, ctx_size, ctx_dst_fac, npoints,
                               pred_transform,
                               device, lcp_flag, result_dc)

            # split tasks for processes
            # proc_slices = []
            # offset = len(pred_files) // nproc
            # for i in range(nproc):
            #     slice_start = offset * i
            #     slice_end = offset * i if i < nproc - 1 else len(pred_files)
            #     proc_slices.append(np.s_[slice_start:slice_end])
            #
            # result_dict_lock = mp.Lock()
            #
            # params = [(slice, pred_files, result_dict_lock, model, ctx_size, ctx_dst_fac, npoints, pred_transform,
            #            device, lcp_flag, result_dc) for slice in proc_slices]
            #
            # running_tasks = [mp.Process(target=process_data_slice, args=param) for param in params]
            # for running_task in running_tasks:
            #     running_task.start()
            # for running_task in running_tasks:
            #     running_task.join()

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

            df = pd.DataFrame.from_dict(result, orient='index')
            csv_path = f'/wholebrain/scratch/amancu/mergeError/preds/lcp/ConvPoint/2000/arch_2048/{model_name}.csv'
            df.to_csv(csv_path)
