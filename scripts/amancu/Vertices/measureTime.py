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

from scipy.stats import entropy
from syconn.reps.super_segmentation import SuperSegmentationObject, SuperSegmentationDataset
from syconn.handler.prediction import certainty_estimate
from typing import Iterable, Union, Optional, Tuple, Callable, List
from morphx.classes.hybridcloud import HybridCloud
import morphx.processing.clouds as clouds
from morphx.processing.hybrids import extract_subset
from morphx.processing.objects import context_splitting_kdt, context_splitting_graph_many
from elektronn3.models.convpoint import SegSmall
from elektronn3.models.lcp_adapt import ConvAdaptSeg
from lightconvpoint.utils.network import get_search, get_conv
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, precision_recall_curve
from syconn.proc.meshes import mesh2obj_file_colors
from syconn.handler.prediction_pts import evaluate_preds
from scipy.spatial import cKDTree

PINK = np.array([10., 255., 10., 255.])
BLUE = np.array([255., 125., 125., 255.])
GREY = np.array([180., 180., 180., 255.])

def evaluate_real_preds(preds_idcs: np.ndarray, zipped_preds, pred_labels):
    """ ith entry in ``preds_idcs`` contains vertex index of prediction saved at ith entry of preds.
        Predictions for each vertex index are gathered and then evaluated by a majority vote.
        The result gets saved at the respective index in the pred_labels array. """
    pred_dict = defaultdict(list)
    arg_list = defaultdict(list)
    u_preds_idcs = np.unique(preds_idcs)
    for i in range(len(preds_idcs)):
        pred_dict[preds_idcs[i]].append(zipped_preds[i])
    #get the strongest raw predictions of all and pick
    for u_ix in u_preds_idcs:
        arg_list = [a[0] for a in pred_dict[u_ix]]
        counts = np.bincount(arg_list)
        max_pred_label = np.argmax(counts)
        # get all raw predictions for the chosen label and calculate max
        raw_preds = np.array([a[1] for a in pred_dict[u_ix]])[np.where(arg_list==max_pred_label)[0]]
        max_raw = np.max(raw_preds,0)
        pred_labels[u_ix] = (max_pred_label, max_raw)


def process_data_slice(slice, ssd, ssv_ids, model, ctx_size, ctx_dst_fac, npoints, pred_transform, device, lcp_flag):
    """
    Adds average results of metrics (precision, recall, accuracy, f1score) per cell in the res_dc dictionary of results.

    Args:
        slice: np.s_ slice to process the files
        pred_files: file paths containing HybridCLoud pickles

    """
    # GPU Warm-up
    # fea=np.array([[[1,1,1]]])
    # ver=np.array([[1]])
    # dummy_f=torch.tensor(fea)
    # dummy_v = torch.tensor(ver)
    # for _ in range(10):
    #     _ = model(dummy_f, dummy_v)

    starter, ender = torch.cuda.Event(enable_timing=True), \
                     torch.cuda.Event(enable_timing=True)
    starter.record()

    # for each whole cells
    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        for ssv_id in tqdm(ssv_ids[slice], desc='Predict SSOs'):
            res_dc = {
                # 'ssv_id': [],
                'merge_coordinates': [],
                'confidences': []
            }

            sso = ssd.get_super_segmentation_object(ssv_id)
            sso.load_skeleton()
            verts = sso.mesh[1].reshape((-1, 3))
            skel = sso.skeleton['nodes']
            features = np.zeros(shape=(len(verts),), dtype=np.int32)
            print(f'skeleton shape {skel.shape}')

            hc = HybridCloud(vertices=verts, features=features, nodes=sso.skeleton['nodes'] * sso.scaling,
                             edges=sso.skeleton['edges'])

            vert_tree = cKDTree(data=hc.vertices,)
            predictions = []
            pred_indices = []
            raw_predictions = []
            res_list = []
            probs = []
            ii = 0
            source_nodes=[]
            for (sample_feats, sample_pts), source_node, mask, vert_indices in extract_subhcs(hc, ctx_size, ctx_dst_fac,
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
                    raw_preds = pred[0, :, :]
                    # print(raw_pred)

                # prepare predictions
                # raw_preds = raw_preds[mask[0,:]]
                pred = np.argmax(raw_preds, 1)
                # raw_preds = np.max(raw_preds,1)

                # place the external context of the prediction on null, so that they focus only on the middle
                # pred= pred[mask[0,:]]

                predictions.append(pred)
                pred_indices.append(vert_indices[0])
                raw_predictions.append(raw_preds)

                ii += 1

            print(f'Merge context predictions...')
            # "merge" contexts and their predictions, taking the majority vote over all predictions per vertex
            predictions = np.concatenate(predictions)
            pred_indices = np.concatenate(pred_indices)
            raw_preds = np.concatenate(raw_predictions)

            zipped_preds = list(zip(predictions,raw_preds))

            # 0: background, 1: foreground, 3: no prediction -> for the use of bincount
            pred_labels = list(zip(np.ones((len(hc.vertices), 1)) * (3), np.ones((len(hc.vertices), 1)) * (3)))
            # pred labels will have the vertices labels of values [0,1,3]
            evaluate_real_preds(pred_indices, zipped_preds, pred_labels)

            preds = [a[0] for a in pred_labels]
            labeled_indices = np.where(preds != 3)[0]
            labeled_vertices = hc.vertices[labeled_indices]
            labeled_tree = cKDTree(data=labeled_vertices, )
            vert_idcs = np.where(preds == 3)[0]
            unlabeled_verts = hc.vertices[vert_idcs]
            labeled_neighbors = labeled_tree.query_ball_point(x=unlabeled_verts, r=500)
            # print(f'neighbors {labeled_neighbors}')
            for i, cluster in enumerate(labeled_neighbors):
                if len(cluster) == 0:
                    pred_labels[vert_idcs[i]] = int(0)
                cnt = np.bincount(pred_labels[labeled_indices[cluster]][:, 0].astype(np.int64))
                try:
                    vert_label = np.argmax(cnt)
                except:
                    print(
                        f'Could not find neighbor in propagation for cell pair {os.path.basename(path)} and vert {unlabeled_verts[i]} in idc of hc.vertices {vert_idcs[i]}')
                    vert_label = int(0)
                pred_labels[vert_idcs[i]] = vert_label

            # # update labels and raw predictions
            # preds = np.array([a[0] for a in pred_labels], dtype='object')
            # raw_preds = [a[1] for a in pred_labels]
            # rp = []
            # for gen in raw_preds:
            #     a=[]
            #     for i in gen:
            #         a.append(i)
            #     # print(f'array ios {a}')
            #     if a == [3.0]:
            #         a = [3.0,3.0]
            #     rp.append(a)
            # raw_preds = np.array(rp)
            # # raw_preds = [[a[0], a[1]] for a in raw_preds]
            # # print(f'After eval {raw_preds.shape}')
            # # print(f'preds shape {preds.shape}')
            # # print(f'raw_preds shape {raw_preds.shape}')
            # # print(f'Toal number vertices: {len(verts)}')
            # # print(f'nr pos labels {len(np.where(preds==1)[0])}')
            # # print(f'pos labels with raw preds: {raw_preds[np.where(preds==1)[0]]}')
            #
            # #evaluate confidence score
            # print(f'Evaluating confidence score...')
            # pos_indices = np.where(preds == 1)[0]
            # positive_vertices = verts[pos_indices]
            # raw_pos_preds = raw_preds[pos_indices]
            # # print(f'raw shape {raw_pos_preds.shape}')
            #
            # edges = []
            #
            # vert_NN = cKDTree(data=positive_vertices, )
            # # search for every neighbor of every vertex
            # for i, pos_vertex in enumerate(positive_vertices):
            #     neighbors = vert_NN.query_ball_point(x=pos_vertex, r=1000, workers=2)
            #     # neighbors = np.unique(np.concatenate(neighbors)).astype(int)
            #     curr_edges = [(i, x) for x in neighbors]
            #     edges.extend(curr_edges)
            #
            # G = nx.Graph(edges)
            # connected_components = nx.connected_components(G)
            # # print(f'length of connected components {len(connected_components)}')
            # coords = []
            # confidences = []
            #
            # for component in connected_components:
            #     component_vertices = positive_vertices[list(component)]
            #     raw_component_preds = raw_pos_preds[list(component)]
            #
            #     # print(f'raw component preds in component {raw_component_preds} and type {type(raw_component_preds)}')
            #
            #     # Ignore single predicted vertices
            #     if len(raw_component_preds) < 80:
            #         # print(f'Component has 1 vertex. Continue...')
            #         continue
            #     else:
            #         # print(f'Target: {raw_component_preds}')
            #         entr = certainty_estimate(raw_component_preds, is_logit=True)
            #     # print(f'raw preds {raw_component_preds}')
            #     # print(f'len component {len(raw_component_preds)} and entropy for component {entr}')
            #
            #     if entr < 0.4:
            #         continue
            #
            #     # calculate confidence score
            #     confidences.append(entr)
            #
            #     # get coordinates of a connected component/merge error site
            #     mean_coords = np.mean(component_vertices, axis=0)
            #     # print(f'mean coords: {mean_coords}')
            #     coords.append(list(mean_coords))
            #     # print(mean_coords)
            #
            #
            # # res_dc['ssv_id'].append(ssv_id)
            # res_dc['merge_coordinates'] = coords
            # res_dc['confidences'] = confidences
            #
            # df = pd.DataFrame.from_dict(res_dc, orient='index')
            # # csv_path = f'/wholebrain/scratch/amancu/mergeError/preds/lcp/ConvPoint/2000/archLrg/SGD_CyclicLR_CrossEntropy/qualitative/sso_{ssv_id}.csv'
            # csv_path = f'/wholebrain/scratch/amancu/mergeError/preds/lcp/ConvPoint/2000/archLrg/SGD_CyclicLR_FocalLoss/qualitative/sso_{ssv_id}.csv'
            # # csv_path = f'/wholebrain/scratch/amancu/mergeError/preds/lcp/ConvPoint/2000/archLrg/_Adam_StepLR_FocalLoss/qualitative/sso_{ssv_id}.csv'
            # df.to_csv(csv_path)
            #
            # res_list.append(res_dc)
            #
            # print(f'File written, render point cloud...')
            #
            # coord = res_dc['merge_coordinates']
            # # print(f'verts: {verts[5]} \n and coords: {coord}')
            #
            # # prediction
            # tree = cKDTree(data=verts)
            # ids=[]
            # if len(res_dc['merge_coordinates']) != 0:
            #     # merge_coords = res_dc['merge_coordinates']
            #     # print(f'concatenated coords: {merge_coords} \n types {type(merge_coords)} and {type(merge_coords[0])} \n shape {merge_coords.shape}')
            #     ids = tree.query_ball_point(coord,r=2000)
            #     ids = np.concatenate(ids)
            # else:
            #     ids=[]
            #
            # colors = np.full(shape=(hc.vertices.shape[0], 4,), fill_value=GREY)
            # # mask = np.where(pred_labels == 1)[0].astype(np.int64)
            # mask = np.array([[x] for x in ids])
            # try:
            #     np.put_along_axis(colors, mask, PINK, axis=0)
            # except:
            #     # print("No foreground labels in prediction.")
            #     pass
            # mesh2obj_file_colors(os.path.expanduser(
            #     # f'/wholebrain/scratch/amancu/mergeError/preds/lcp/ConvPoint/2000/archLrg/SGD_CyclicLR_CrossEntropy/qualitative/{ssv_id}_prediction.ply'),
            #     # f'/wholebrain/scratch/amancu/mergeError/preds/lcp/ConvPoint/2000/archLrg/SGD_CyclicLR_CrossEntropy/qualitative/{ssv_id}_prediction.ply'),
            #     f'/wholebrain/scratch/amancu/mergeError/preds/lcp/ConvPoint/2000/archLrg/SGD_CyclicLR_FocalLoss/qualitative/{ssv_id}_prediction.ply'),
            #     # f'/wholebrain/scratch/amancu/mergeError/preds/lcp/ConvPoint/2000/archLrg/_Adam_StepLR_FocalLoss/qualitative/{ssv_id}_prediction.ply'),
            #     [np.array([]), hc.vertices, np.array([])], colors)

    ender.record()
    torch.cuda.synchronize()
    print(f'Elapsed time with record: {starter.elapsed_time(ender)}')

    print(f'Elapsed time with autograd: {prof}')

    # print(res_list)
    # queue.put(res_list)
    print(f'Queue put done. Store dict...')
    return

def extract_subhcs(hc: HybridCloud, ctx_size, ctx_dst_fac, npoints, transform: Callable):
    # choose base nodes with context overlap
    base_node_dst = ctx_size / ctx_dst_fac
    # select source nodes for context extraction
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(hc.nodes)
    pcd, idcs = pcd.voxel_down_sample_and_trace(base_node_dst, pcd.get_min_bound(), pcd.get_max_bound())
    # print(f'idcs {idcs}')
    source_nodes = np.max(idcs, axis=1)
    # print(f'len source_nodes {len(source_nodes)}')
    bs = 1
    n_batches = int(np.ceil(len(source_nodes) / bs))
    if len(source_nodes) % bs != 0:
        source_nodes = np.concatenate([np.random.choice(source_nodes, bs - len(source_nodes) % bs),
                                       source_nodes])
    node_arrs = context_splitting_kdt(hc, source_nodes, ctx_size)
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
            # global_idcs = global_idcs[inner_mask]
            # prepare masks for filtering sv vertices
            # bounds = hc.obj_bounds['sv']
            # sv_mask = np.logical_and(global_idcs < bounds[1], global_idcs >= bounds[0])
            # hc_sample.set_features(label_binarize(hc_sample.features, classes=np.arange(len(feat_dc))))
            if transform is not None:
                transform(hc_sample)
            arr_list['verts'][cnt] = hc_sample.vertices
            arr_list['feats'][cnt] = hc_sample.features
            # masks get used later when mapping predictions back onto the cell surface during postprocessing
            # arr_list['margin_mask'][cnt] = mask
            arr_list['margin_mask'][cnt] = inner_mask
            arr_list['source_node'][cnt] = source_nodes[ii]
            arr_list['global_vert_indices'].append(global_idcs)
            cnt += 1
        yield (arr_list['feats'], arr_list['verts']), arr_list['source_node'], arr_list['margin_mask'], arr_list['global_vert_indices']


def load_model(mkwargs, device):
    search = 'SearchQuantized'
    conv = dict(layer='ConvPoint', kernel_separation=False)
    # conv = dict(layer='FKAConv', kernel_separation=False)
    act = nn.ReLU
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
    archLrg_noFstConv = [{'ic': -1, 'oc': 1, 'ks': 16, 'nn': 32, 'np': 2048},
                         {'ic': 1, 'oc': 1, 'ks': 16, 'nn': 32, 'np': 1024},
                         {'ic': 1, 'oc': 1, 'ks': 16, 'nn': 32, 'np': 256},
                         {'ic': 1, 'oc': 2, 'ks': 16, 'nn': 32, 'np': 64},
                         {'ic': 2, 'oc': 2, 'ks': 16, 'nn': 16, 'np': 16},
                         {'ic': 2, 'oc': 2, 'ks': 16, 'nn': 8, 'np': 8},
                         {'ic': 2, 'oc': 2, 'ks': 16, 'nn': 4, 'np': 'd'},
                         {'ic': 4, 'oc': 2, 'ks': 16, 'nn': 4, 'np': 'd'},
                         {'ic': 4, 'oc': 1, 'ks': 16, 'nn': 8, 'np': 'd'},
                         {'ic': 2, 'oc': 1, 'ks': 16, 'nn': 16, 'np': 'd'},
                         # {'ic': 2, 'oc': 1, 'ks': 16, 'nn': 16, 'np': 'd'},
                         {'ic': 2, 'oc': 1, 'ks': 16, 'nn': 16, 'np': 'd'}]
    arch2048_noFstConv = [dict(ic=-1, oc=1, ks=16, nn=32, np=1024),
                          dict(ic=1, oc=1, ks=16, nn=32, np=512),
                          dict(ic=1, oc=1, ks=16, nn=32, np=256),
                          dict(ic=1, oc=2, ks=16, nn=32, np=64),
                          dict(ic=2, oc=2, ks=16, nn=16, np=16),
                          dict(ic=2, oc=2, ks=16, nn=8, np=8),
                          dict(ic=2, oc=2, ks=16, nn=4, np='d'),
                          dict(ic=4, oc=2, ks=16, nn=4, np='d'),
                          dict(ic=4, oc=1, ks=16, nn=8, np='d'),
                          dict(ic=2, oc=1, ks=16, nn=16, np='d'),
                          dict(ic=2, oc=1, ks=16, nn=16, np='d')]
    model = ConvAdaptSeg(input_channels, num_classes, get_conv(conv), get_search(search), kernel_num=64,
                         architecture=architecture_large, activation=act, norm='gn').to(device)
    model.load_state_dict(torch.load(mdir, map_location=device)['model_state_dict'])
    model.eval()
    return model

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')

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

    radius = 2000
    nproc = 1

    lcp_flag = True
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    ssv_ids = [224145260, 271546135,23256852]

    ssd_kwargs = dict(working_dir='/ssdscratch/songbird/j0251/rag_flat_Jan2019_v3/')
    ssd = SuperSegmentationDataset(**ssd_kwargs)
    mdir = f'/wholebrain/scratch/amancu/mergeError/trainings/lcp/see/lcp_r2000_ConvPoint_SearchQuantized_archLrg_run6_Adam_StepLR_weights1,2_CrossEntropy/state_dict.pth'
    # mdir = f'/wholebrain/scratch/amancu/mergeError/trainings/lcp/see/lcp_r2000_ConvPoint_SearchQuantized_archLrg_run2_SGD_CyclicLR_weights1,2_CrossEntropy/state_dict.pth'
    # mdir = f'/wholebrain/scratch/amancu/mergeError/trainings/lcp/see/lcp_r2000_ConvPoint_SearchQuantized_archLrg_Adam_StepLR_weights1,2_FocalLoss/state_dict.pth'
    # mdir = f'/wholebrain/scratch/amancu/mergeError/trainings/lcp/see/lcp_r2000_ConvPoint_SearchQuantized_archLrg_run2_SGD_CyclicLR_weights1,2_FocalLoss/state_dict.pth'
    # mdir = f'/wholebrain/scratch/amancu/mergeError/trainings/lcp/see/lcp_r2000_ConvPoint_SearchQuantized_archLrg_Adam_StepLR_weights1,2_FocalLoss/state_dict.pth'
    mkwargs = dict(use_bn=False, track_running_stats=False)
    # ssd_kwargs = [{'ssv_id': ssv_id, 'working_dir': ssd_kwargs['working_dir']} for ssv_id in ssv_ids]
    model = load_model(mkwargs, device)
    model.eval()

    slice = np.s_[0:len(ssv_ids)]

    for _ in range(5):
        process_data_slice(slice, ssd, ssv_ids, model, ctx_size, ctx_dst_fac, npoints, pred_transform, device, lcp_flag)

    print(f'Processing finished')

    # get results
    results = [queue.get() for task in running_tasks]