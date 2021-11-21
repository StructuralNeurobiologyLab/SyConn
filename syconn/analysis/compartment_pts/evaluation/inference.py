import os
import numpy as np
import open3d as o3d
import torch
import math
import time
import pickle as pkl
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
from lightconvpoint.utils import get_network
from lightconvpoint.utils.network import get_search, get_conv
from elektronn3.models.lcp_adapt import ConvAdaptSeg


def predict_sso_thread(kwargs):
    predict_sso(**kwargs)


def predict_sso(sso_ids: List[int], wd: str, model_p: str, model_args_p: str, pred_key: str,
                redundancy: int, border_exclusion: int = 0, v3: bool = True, out_p: str = None):
    model_p = os.path.expanduser(model_p)
    model_args_p = os.path.expanduser(model_args_p)

    ssd = SuperSegmentationDataset(working_dir=wd)
    argscont = ArgsContainer().load_from_pkl(model_args_p)

    if argscont.use_cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # load model (lcp = LightConvPoint)
    lcp_flag = False
    if argscont.architecture == 'lcp' or argscont.model == 'ConvAdaptSeg':
        kwargs = {}
        if argscont.model == 'ConvAdaptSeg':
            kwargs = dict(kernel_num=argscont.pl, architecture=argscont.architecture, activation=argscont.act,
                          norm=argscont.norm_type)
        conv = dict(layer=argscont.conv[0], kernel_separation=argscont.conv[1])
        model = get_network(argscont.model, argscont.input_channels, argscont.class_num, conv, argscont.search,
                            **kwargs)
        lcp_flag = True
    elif argscont.use_big:
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

    voxel_dc = {'sv': 80, 'mi': 100, 'vc': 100, 'sy': 100, 'sj': 100}
    feats = argscont.features
    if 'hc' in feats:
        feats['sv'] = feats['hc']
        feats.pop('hc')
    if v3 and 'sy' in feats:
        feats['sj'] = feats['sy']
        feats.pop('sy')
    parts = {}
    for key in feats:
        parts[key] = (voxel_dc[key], feats[key])

    start_total = time.time()
    for sso_id in sso_ids:
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
        for ix, node_arr in enumerate(node_arrs):
            # vertices which correspond to nodes in node_arr
            sample, idcs_sub = objects.extract_cloud_subset(hc, node_arr)
            # random subsampling of the corresponding vertices
            sample, idcs_sample = clouds.sample_cloud(sample, argscont.sample_num, padding=argscont.padding)
            if border_exclusion > 0:
                sample.mark_borders(-1, argscont.chunk_size - border_exclusion, centroid=hc.nodes[source_nodes[ix]])
                marked_idcs = idcs_sample[(sample.labels == -1).reshape(-1)]
                # exclude borders from majority votes
                idcs_sub[marked_idcs] = -1
            # indices with respect to the total HybridCloud
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
                # lcp and convpoint use different axis order
                if lcp_flag:
                    pts = pts.transpose(1, 2)
                    features = features.transpose(1, 2)
                outputs = model(features, pts)
                if lcp_flag:
                    outputs = outputs.transpose(1, 2)
                outputs = outputs.cpu().detach().numpy()
                for ix in range(argscont.batch_size):
                    preds.append(np.argmax(outputs[ix], axis=1))
                    idcs_preds.append(batch[2][ix])

        preds = np.concatenate(preds)
        idcs_preds = np.concatenate(idcs_preds)
        # filter possible organelles and borders
        preds = preds[idcs_preds != -1]
        idcs_preds = idcs_preds[idcs_preds != -1].astype(int) - hc.obj_bounds['sv'][0]
        pred_labels = np.ones((len(voxel_idcs['sv']), 1)) * -1
        print("Evaluating predictions...")
        start = time.time()
        evaluate_preds(idcs_preds, preds, pred_labels)
        print(f"Finished evaluation in {(time.time() - start):.2f} seconds.")
        sso_vertices = sso.mesh[1].reshape((-1, 3))
        sso_preds = np.ones((len(sso_vertices), 1)) * -1
        sso_preds[voxel_idcs['sv']] = pred_labels

        if out_p is None:
            ld = sso.label_dict('vertex')
            ld[pred_key] = sso_preds
            ld.push()
        else:
            if not os.path.exists(out_p):
                os.makedirs(out_p)
            with open(os.path.join(out_p, str(sso_id) + '.pkl'), 'wb') as f:
                pkl.dump(sso_preds, f)
    return time.time() - start_total


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
    batch_num = math.ceil(len(samples) / batch_size)
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


def predict_sso_thread_dnho(sso_ids: List[int], wd: str, model_p: str, pred_key: str,
                            redundancy: int, v3: bool = True, out_p: str = None, architecture=None):
    from syconn.handler.prediction_pts import pts_feat_dict, pts_feat_ds_dict
    model_p = os.path.expanduser(model_p)

    ssd = SuperSegmentationDataset(working_dir=wd)

    device = torch.device('cuda')

    # load model (lcp = LightConvPoint)
    # Model selection
    # TODO: change hardcoded mode parameters
    normalize_pts = True
    lcp_flag = True
    search = 'SearchQuantized'
    conv = dict(layer='ConvPoint', kernel_separation=False, normalize_pts=normalize_pts)
    act = torch.nn.ReLU
    inp_channels = 4
    out_channels = 4
    context_size = 15000
    nb_points = 15000
    batch_size = 8
    scale_norm = 5000
    valid_transform = clouds.Compose([clouds.Center(),
                                      # clouds.Normalization(scale_norm)
                                      ])
    model = ConvAdaptSeg(inp_channels, out_channels, get_conv(conv), get_search(search), kernel_num=64,
                         architecture=architecture, activation=act, norm='gn')
    try:
        full = torch.load(model_p)
        model.load_state_dict(full)
    except RuntimeError:
        model.load_state_dict(full['model_state_dict'])
    model.to(device)
    model.eval()

    voxel_dc = dict(pts_feat_ds_dict['compartment'])
    feats = dict(pts_feat_dict)
    if 'hc' in feats:
        feats['sv'] = feats['hc']
        feats.pop('hc')
    if v3 and 'syn_ssv' in feats:
        feats['sj'] = feats['syn_ssv']
        feats.pop('syn_ssv')
    if v3 and 'syn_ssv' in voxel_dc:
        voxel_dc['sj'] = voxel_dc['syn_ssv']
        voxel_dc.pop('syn_ssv')
    del feats['syn_ssv_asym']
    del feats['syn_ssv_sym']
    del feats['sv_myelin']
    parts = {}
    for key in feats:
        parts[key] = (voxel_dc[key], feats[key])

    start_total = time.time()
    for sso_id in sso_ids:
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
            raise ValueError(f"Couldn't find skeleton of {sso}")
        nodes, edges = sso.skeleton['nodes'] * sso.scaling, sso.skeleton['edges']
        hc = HybridCloud(nodes, edges, vertices=sample_pts, labels=np.ones((len(sample_pts), 1)),
                         features=sample_feats, obj_bounds=obj_bounds)
        node_arrs, source_nodes = splitting.split_single(hc, context_size, context_size / redundancy)

        samples = []
        for ix, node_arr in enumerate(node_arrs):
            # vertices which correspond to nodes in node_arr
            sample, idcs_sub = objects.extract_cloud_subset(hc, node_arr)
            # random subsampling of the corresponding vertices
            sample, idcs_sample = clouds.sample_cloud(sample, nb_points, padding=None)
            # indices with respect to the total HybridCloud
            idcs_global = idcs_sub[idcs_sample.astype(int)]
            bounds = hc.obj_bounds['sv']
            sv_mask = np.logical_and(idcs_global < bounds[1], idcs_global >= bounds[0])
            idcs_global[np.logical_not(sv_mask)] = -1
            if len(sample.vertices) == 0:
                continue
            valid_transform(sample)
            samples.append((sample, idcs_global))

        preds = []
        idcs_preds = []
        with torch.no_grad():
            batches = batch_builder(samples, batch_size, inp_channels)
            for batch in batches:
                pts = batch[0].to(device, non_blocking=True)
                features = batch[1].to(device, non_blocking=True)
                # lcp and convpoint use different axis order
                if lcp_flag:
                    pts = pts.transpose(1, 2)
                    features = features.transpose(1, 2)
                outputs = model(features, pts)
                if lcp_flag:
                    outputs = outputs.transpose(1, 2)
                outputs = outputs.cpu().detach().numpy()
                for ix in range(batch_size):
                    preds.append(np.argmax(outputs[ix], axis=1))
                    idcs_preds.append(batch[2][ix])

        preds = np.concatenate(preds)
        idcs_preds = np.concatenate(idcs_preds)
        # filter possible organelles and borders
        preds = preds[idcs_preds != -1]
        idcs_preds = idcs_preds[idcs_preds != -1].astype(int) - hc.obj_bounds['sv'][0]
        pred_labels = np.ones((len(voxel_idcs['sv']), 1)) * -1
        print("Evaluating predictions...")
        start = time.time()
        evaluate_preds(idcs_preds, preds, pred_labels)
        print(f"Finished evaluation in {(time.time() - start):.2f} seconds.")
        sso_vertices = sso.mesh[1].reshape((-1, 3))
        sso_preds = np.ones((len(sso_vertices), 1)) * -1
        sso_preds[voxel_idcs['sv']] = pred_labels
        ld = sso.label_dict('vertex')
        ld[pred_key] = sso_preds
        if out_p is None:
            ld.push()
        else:
            sso.semseg2mesh(pred_key, dest_path=f'{out_p}/{sso_id}_dnho.k.zip')
            if not os.path.exists(out_p):
                os.makedirs(out_p)
            with open(os.path.join(out_p, str(sso_id) + '.pkl'), 'wb') as f:
                pkl.dump(sso_preds, f)
    return time.time() - start_total


def predict_sso_thread_do(sso_ids: List[int], wd: str, model_p: str, pred_key: str,
                          redundancy: int, v3: bool = True, out_p: str = None, architecture=None):
    from syconn.handler.prediction_pts import pts_feat_dict, pts_feat_ds_dict
    model_p = os.path.expanduser(model_p)

    ssd = SuperSegmentationDataset(working_dir=wd)

    device = torch.device('cuda')

    # load model (lcp = LightConvPoint)
    # Model selection
    # TODO: change hardcoded mode parameters
    normalize_pts = True
    lcp_flag = True
    search = 'SearchQuantized'
    conv = dict(layer='ConvPoint', kernel_separation=False, normalize_pts=normalize_pts)
    act = torch.nn.ReLU
    inp_channels = 4
    out_channels = 2
    context_size = 15000
    nb_points = 15000
    batch_size = 8
    scale_norm = 5000
    valid_transform = clouds.Compose([clouds.Center(),
                                      # clouds.Normalization(scale_norm)
                                      ])
    model = ConvAdaptSeg(inp_channels, out_channels, get_conv(conv), get_search(search), kernel_num=64,
                         architecture=architecture, activation=act, norm='gn')
    try:
        full = torch.load(model_p)
        model.load_state_dict(full)
    except RuntimeError:
        model.load_state_dict(full['model_state_dict'])
    model.to(device)
    model.eval()

    voxel_dc = dict(pts_feat_ds_dict['compartment'])
    feats = dict(pts_feat_dict)
    if 'hc' in feats:
        feats['sv'] = feats['hc']
        feats.pop('hc')
    if v3 and 'syn_ssv' in feats:
        feats['sj'] = feats['syn_ssv']
        feats.pop('syn_ssv')
    if v3 and 'syn_ssv' in voxel_dc:
        voxel_dc['sj'] = voxel_dc['syn_ssv']
        voxel_dc.pop('syn_ssv')
    del feats['syn_ssv_asym']
    del feats['syn_ssv_sym']
    del feats['sv_myelin']
    parts = {}
    for key in feats:
        parts[key] = (voxel_dc[key], feats[key])

    start_total = time.time()
    for sso_id in sso_ids:
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
            raise ValueError(f"Couldn't find skeleton of {sso}")
        nodes, edges = sso.skeleton['nodes'] * sso.scaling, sso.skeleton['edges']
        hc = HybridCloud(nodes, edges, vertices=sample_pts, labels=np.ones((len(sample_pts), 1)),
                         features=sample_feats, obj_bounds=obj_bounds)
        node_arrs, source_nodes = splitting.split_single(hc, context_size, context_size / redundancy)

        samples = []
        for ix, node_arr in enumerate(node_arrs):
            # vertices which correspond to nodes in node_arr
            sample, idcs_sub = objects.extract_cloud_subset(hc, node_arr)
            # random subsampling of the corresponding vertices
            sample, idcs_sample = clouds.sample_cloud(sample, nb_points, padding=None)
            # indices with respect to the total HybridCloud
            idcs_global = idcs_sub[idcs_sample.astype(int)]
            bounds = hc.obj_bounds['sv']
            sv_mask = np.logical_and(idcs_global < bounds[1], idcs_global >= bounds[0])
            idcs_global[np.logical_not(sv_mask)] = -1
            if len(sample.vertices) == 0:
                continue
            valid_transform(sample)
            samples.append((sample, idcs_global))

        preds = []
        idcs_preds = []
        with torch.no_grad():
            batches = batch_builder(samples, batch_size, inp_channels)
            for batch in batches:
                pts = batch[0].to(device, non_blocking=True)
                features = batch[1].to(device, non_blocking=True)
                # lcp and convpoint use different axis order
                if lcp_flag:
                    pts = pts.transpose(1, 2)
                    features = features.transpose(1, 2)
                outputs = model(features, pts)
                if lcp_flag:
                    outputs = outputs.transpose(1, 2)
                outputs = outputs.cpu().detach().numpy()
                for ix in range(batch_size):
                    preds.append(np.argmax(outputs[ix], axis=1))
                    idcs_preds.append(batch[2][ix])

        preds = np.concatenate(preds)
        idcs_preds = np.concatenate(idcs_preds)
        # filter possible organelles and borders
        preds = preds[idcs_preds != -1]
        idcs_preds = idcs_preds[idcs_preds != -1].astype(int) - hc.obj_bounds['sv'][0]
        pred_labels = np.ones((len(voxel_idcs['sv']), 1)) * -1
        print("Evaluating predictions...")
        start = time.time()
        evaluate_preds(idcs_preds, preds, pred_labels)
        print(f"Finished evaluation in {(time.time() - start):.2f} seconds.")
        sso_vertices = sso.mesh[1].reshape((-1, 3))
        sso_preds = np.ones((len(sso_vertices), 1)) * -1
        sso_preds[voxel_idcs['sv']] = pred_labels
        ld = sso.label_dict('vertex')
        ld[pred_key] = sso_preds
        if out_p is None:
            ld.push()
        else:
            sso.semseg2mesh(pred_key, dest_path=f'{out_p}/{sso_id}_do.k.zip')
            if not os.path.exists(out_p):
                os.makedirs(out_p)
            with open(os.path.join(out_p, str(sso_id) + '.pkl'), 'wb') as f:
                pkl.dump(sso_preds, f)
    return time.time() - start_total


if __name__ == '__main__':
    # base = os.path.expanduser('~/working_dir/paper/hierarchy/')
    #
    # # path_list = [('abt', 180), ('dnh', 390), ('ads', 760)]
    # path_list = [('abt', 370)]
    #
    # # path_list = [('2020_10_14_8000_8192_cp_cp_q', 570),
    # #              ('2020_11_08_2000_2048_cp_cp_q', 90),
    # #              ('2020_11_08_2000_2048_cp_cp_q_2', 90),
    # #              ('2020_11_08_8000_2048_cp_cp_q', 480),
    # #              ('2020_11_08_8000_2048_cp_cp_q_2', 480),
    # #              ('2020_11_08_8000_8192_cp_cp_q', 300),
    # #              ('2020_11_08_8000_8192_cp_cp_q_co', 510),
    # #              ('2020_11_08_8000_8192_cp_cp_q_co_2', 510),
    # #              ('2020_11_08_8000_8192_cp_cp_q_nn', 450),
    # #              ('2020_11_08_8000_8192_cp_cp_q_nn_2', 420),
    # #              ('2020_11_08_8000_32768_cp_cp_q', 120),
    # #              ('2020_11_08_8000_32768_cp_cp_q_2', 270),
    # #              ('2020_11_08_24000_32768_cp_cp_q', 570),
    # #              ('2020_11_08_24000_32768_cp_cp_q_2', 540),
    # #              ('2020_11_09_2000_2048_cp_cp_q_3', 120),
    # #              ('2020_11_09_8000_1024_cp_cp_q', 600),
    # #              ('2020_11_09_8000_1024_cp_cp_q_2', 600),
    # #              ('2020_11_09_8000_8192_cp_cp_q_2', 360),
    # #              ('2020_11_09_8000_8192_cp_cp_q_bn', 540),
    # #              ('2020_11_09_8000_8192_cp_cp_q_bn_2', 570),
    # #              ('2020_11_11_2000_2048_cp_cp_q_3', 360),
    # #              ('2020_11_11_8000_1024_cp_cp_q_3', 450),
    # #              ('2020_11_11_8000_2048_cp_cp_q_3', 450),
    # #              ('2020_11_11_8000_8192_cp_cp_q_bn_3', 480),
    # #              ('2020_11_11_8000_8192_cp_cp_q_co_3', 450),
    # #              ('2020_11_11_8000_32768_cp_cp_q_3', 390),
    # #              ('2020_11_11_24000_32768_cp_cp_q_3', 480),
    # #              ('2020_11_16_8000_8192_cp_cp_q_nn_3', 690),
    # #              ('2020_11_16_8000_8192_cp_cp_q_nn_4', 690)]
    #
    # durations = {}
    # red = 5
    # for path in path_list:
    #     print(f'Processing: {path} with redundancy {red}')
    #     base_path = base + path[0] + '/'
    #     m_path = base_path + f'models/state_dict_e{path[1]}.pth'
    #     argscont_path = base_path + 'argscont.pkl'
    #     duration = predict_sso([141995, 11833344, 28410880, 28479489], "/wholebrain/scratch/areaxfs3/",
    #                            m_path, argscont_path, pred_key=f'{path[0]}_e{path[1]}_red{red}_border', redundancy=red, border_exclusion=0,
    #                            out_p=base_path + f'syn_eval_red{red}/')
    #
    #     if path[0] in durations:
    #         durations[path[0]].append(duration)
    #     else:
    #         durations[path[0]] = [duration]
    # # with open(base + 'timing_border.pkl', 'wb') as f:
    # #     pkl.dump(durations, f)

    # path = '/wholebrain/scratch/jklimesch/paper/paper_models/2020_09_28_10000_15000_ads_cmnGT/'
    path = '/wholebrain/scratch/pschuber/e3_trainings/lcp_semseg_j0251_dnho/semseg_pts_nb15000_ctx15000_dnho_nclass4_ptconv_GN_strongerWeighted_noKernelSep_eval/'
    red = 5
    print(f'Processing model "{path}" with redundancy {red}.')
    m_path = path + f'state_dict_final.pth'
    argscont_path = path + 'argscont.pkl'
    duration = predict_sso([141995, 11833344, 28410880, 28479489], "/wholebrain/scratch/areaxfs3/",
                           m_path, argscont_path, pred_key=f'dnho_cmn_new', redundancy=red, border_exclusion=0)
