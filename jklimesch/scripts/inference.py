import os
import numpy as np
import open3d as o3d
import torch
import math
import time
from tqdm import tqdm
from typing import List, Tuple
from collections import defaultdict
from morphx.preprocessing import splitting
from morphx.classes.hybridcloud import HybridCloud
from morphx.processing import objects, clouds
from morphx.classes.pointcloud import PointCloud
from sklearn.preprocessing import label_binarize
from elektronn3.models.convpoint import SegBig, SegAdapt
from neuronx.classes.argscontainer import ArgsContainer
from syconn.reps.super_segmentation_dataset import SuperSegmentationDataset


def predict_sso(sso_ids: List[int], ssd: SuperSegmentationDataset, model_p: str, model_args_p: str, pred_key: str,
                redundancy: int, border_exclusion: int = 0):
    model_p = os.path.expanduser(model_p)
    model_args_p = os.path.expanduser(model_args_p)

    argscont = ArgsContainer().load_from_pkl(model_args_p)

    if argscont.use_cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # load model
    if argscont.use_big:
        model = SegBig(argscont.input_channels, argscont.class_num, trs=argscont.track_running_stats, dropout=0,
                       use_bias=argscont.use_bias, norm_type=argscont.norm_type, use_norm=argscont.use_norm,
                       kernel_size=argscont.kernel_size, neighbor_nums=argscont.neighbor_nums,
                       reductions=argscont.reductions, first_layer=argscont.first_layer,
                       padding=argscont.padding, nn_center=argscont.nn_center, centroids=argscont.centroids,
                       pl=argscont.pl, normalize=argscont.cp_norm)
    else:
        model = SegAdapt(argscont.input_channels, argscont.class_num, architecture=argscont.architecture,
                         trs=argscont.track_running_stats, dropout=argscont.dropout, use_bias=argscont.use_bias,
                         norm_type=argscont.norm_type, kernel_size=argscont.kernel_size, padding=argscont.padding,
                         nn_center=argscont.nn_center, centroids=argscont.centroids, kernel_num=argscont.pl,
                         normalize=argscont.cp_norm, act=argscont.act)
    try:
        full = torch.load(model_p)
        model.load_state_dict(full)
    except RuntimeError:
        model.load_state_dict(full['model_state_dict'])
    model.to(device)
    model.eval()

    voxel_dc = {'sv': 80, 'mi': 100, 'vc': 100, 'sy': 100}
    feats = argscont.features
    if 'hc' in feats:
        feats['sv'] = feats['hc']
        feats.pop('hc')
    parts = {}
    for key in feats:
        parts[key] = (voxel_dc[key], feats[key])

    for sso_id in tqdm(sso_ids):
        sso = ssd.get_super_segmentation_object(sso_id)
        vert_dc = {}
        voxel_idcs = {}
        offset = 0
        obj_bounds = {}
        for ix, k in enumerate(parts):
            # build cell representation by adding cell surface and possible organelles
            pcd = o3d.geometry.PointCloud()
            verts = sso.load_mesh(k)[1].reshape(-1, 3)
            pcd.points = o3d.utility.Vector3dVector(verts)
            pcd, idcs = pcd.voxel_down_sample_and_trace(parts[k][0], pcd.get_min_bound(), pcd.get_max_bound())
            voxel_idcs[k] = np.max(idcs, axis=1)
            vert_dc[k] = np.asarray(pcd.points)
            obj_bounds[k] = [offset, offset + len(pcd.points)]
            offset += len(pcd.points)

        if type(parts['sv'][1]) == int:
            sample_feats = np.concatenate([[parts[k][1]] * len(vert_dc[k]) for k in parts]).reshape(-1, 1)
            sample_feats = label_binarize(sample_feats, classes=np.arange(len(parts)))
        else:
            feats_dc = {}
            for key in parts:
                sample_feats = np.ones((len(vert_dc[key]), len(parts[key][1])))
                sample_feats[:] = parts[key][1]
                feats_dc[key] = sample_feats
            sample_feats = np.concatenate([feats_dc[k] for k in parts])

        sample_pts = np.concatenate([vert_dc[k] for k in parts])
        if not sso.load_skeleton():
            raise ValueError(f'Couldnt find skeleton of {sso}')
        nodes, edges = sso.skeleton['nodes'] * sso.scaling, sso.skeleton['edges']
        hc = HybridCloud(nodes, edges, vertices=sample_pts, labels=np.ones((len(sample_pts), 1)),
                         features=sample_feats, obj_bounds=obj_bounds)
        node_arrs, source_nodes = splitting.split_single(hc, argscont.chunk_size, argscont.chunk_size / redundancy)

        transform = clouds.Compose(argscont.val_transforms)
        samples = []
        for ix, node_arr in enumerate(tqdm(node_arrs)):
            # vertices which correspond to nodes in node_arr
            sample, idcs_sub = objects.extract_cloud_subset(hc, node_arr)
            # random subsampling of the corresponding vertices
            sample, idcs_sample = clouds.sample_cloud(sample, argscont.sample_num, padding=argscont.padding)
            if border_exclusion > 0:
                sample.mark_borders(-1, argscont.chunk_size-border_exclusion, centroid=hc.nodes[source_nodes[ix]])
                marked_idcs = idcs_sample[(sample.labels == -1).reshape(-1)]
                # exclude borders from majority votes
                idcs_sub[marked_idcs] = -1
            # indices with respect to the total cell
            idcs_global = idcs_sub[idcs_sample.astype(int)]
            bounds = hc.obj_bounds['sv']
            sv_mask = np.logical_and(idcs_global < bounds[1], idcs_global >= bounds[0])
            idcs_global[np.logical_not(sv_mask)] = -1
            if len(sample.vertices) == 0:
                continue
            transform(sample)
            samples.append((sample, idcs_global))

        preds = []
        idcs_preds = []
        with torch.no_grad():
            batches = batch_builder(samples, argscont.batch_size, argscont.input_channels)
            for batch in batches:
                pts = batch[0].to(device, non_blocking=True)
                features = batch[1].to(device, non_blocking=True)
                outputs = model(features, pts)
                outputs = outputs.cpu().detach().numpy()
                for ix in range(argscont.batch_size):
                    preds.append(np.argmax(outputs[ix], axis=1))
                    idcs_preds.append(batch[2][ix])

        preds = np.concatenate(preds)
        idcs_preds = np.concatenate(idcs_preds)
        # filter possible organelles
        preds = preds[idcs_preds != -1]
        idcs_preds = idcs_preds[idcs_preds != -1].astype(int)
        pred_labels = np.ones((len(voxel_idcs['sv']), 1))*-1
        print("Evaluating predictions...")
        start = time.time()
        evaluate_preds(idcs_preds, preds, pred_labels)
        print(f"Finished evaluation in {time.time()-start} seconds.")
        sso_vertices = sso.mesh[1].reshape((-1, 3))
        sso_preds = np.ones((len(sso_vertices), 1)) * -1
        sso_preds[voxel_idcs['sv']] = pred_labels
        ld = sso.label_dict('vertex')
        ld[pred_key] = sso_preds
        ld.push()


def evaluate_preds(preds_idcs: np.ndarray, preds: np.ndarray, pred_labels: np.ndarray):
    """ ith entry in ``preds_idcs`` contains vertex index of prediction saved at ith entry of preds.
        Predictions for each vertex index are gathered and then evaluated by a majority vote.
        The result gets saved at the respective index in the pred_labels array. """
    pred_dict = defaultdict(list)
    u_preds_idcs = np.unique(preds_idcs)
    for i in range(len(preds_idcs)):
        pred_dict[preds_idcs[i]].append(preds[i])
    for u_ix in u_preds_idcs:
        counts = np.bincount(pred_dict[u_ix])
        pred_labels[u_ix] = np.argmax(counts)


def batch_builder(samples: List[Tuple[PointCloud, np.ndarray]], batch_size: int, input_channels: int):
    point_num = len(samples[0][0].vertices)
    batch_num = math.ceil(len(samples)/batch_size)
    batches = []
    ix = -1
    for batch_ix in range(batch_num):
        pts = torch.zeros((batch_size, point_num, 3))
        features = torch.ones((batch_size, point_num, input_channels))
        mapping_idcs = torch.ones((batch_size, point_num))
        for sample_ix in range(batch_size):
            ix += 1
            if ix == len(samples):
                ix = 0
            pts[sample_ix] = torch.from_numpy(samples[ix][0].vertices).float()
            features[sample_ix] = torch.from_numpy(samples[ix][0].features).float()
            mapping_idcs[sample_ix] = torch.from_numpy(samples[ix][1])
        batches.append((pts, features, mapping_idcs))
    return batches


if __name__ == '__main__':
    base_path = '~/thesis/current_work/paper/dnh/2020_09_18_4000_4000/'
    m_path = base_path + 'models/state_dict_e250.pth'
    argscont_path = base_path + 'argscont.pkl'
    predict_sso([141995, 11833344, 28410880, 28479489],
                SuperSegmentationDataset(working_dir="/wholebrain/scratch/areaxfs3/"),
                m_path, argscont_path, pred_key='dnh_borders', redundancy=5, border_exclusion=1000)

