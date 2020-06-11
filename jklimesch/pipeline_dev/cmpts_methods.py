try:
    import open3d as o3d
except ImportError:
    pass  # for sphinx build
import re
import os
import time
from numba.typed import List, Dict
import tqdm
import collections
import numba as nb
import morphx.processing.clouds as clouds
from syconn.reps.super_segmentation import SuperSegmentationDataset
from elektronn3.models.convpoint import SegBig
from typing import Iterable, Union, Optional, Any, Tuple, Callable, List
from multiprocessing import Process, Queue, Manager
import numpy as np
import scipy.special
from scipy import spatial
import morphx.processing.clouds as clouds
import functools
from morphx.classes.hybridcloud import HybridCloud
import networkx as nx
from scipy.spatial import cKDTree
from sklearn.preprocessing import label_binarize
from morphx.processing.hybrids import extract_subset
from morphx.processing.objects import bfs_vertices, context_splitting, context_splitting_v2
from syconn.reps.super_segmentation import SuperSegmentationDataset
from syconn.handler.basics import chunkify_successive
from syconn.reps.super_segmentation_helper import sparsify_skeleton_fast
from syconn import global_params
from syconn.handler import log_handler
from morphx.classes.hybridmesh import HybridMesh
from morphx.classes.pointcloud import PointCloud
from morphx.classes.cloudensemble import CloudEnsemble
from syconn.reps.super_segmentation import SuperSegmentationObject
from syconn.reps.super_segmentation_helper import map_myelin2coords, majorityvote_skeleton_property
# for readthedocs build
try:
    import torch
except ImportError:
    pass

pts_feat_dict = dict(sv=0, mi=1, syn_ssv=3, syn_ssv_sym=3, syn_ssv_asym=4, vc=2)
pts_feat_ds_dict = dict(celltype=dict(sv=70, mi=100, syn_ssv=70, syn_ssv_sym=70, syn_ssv_asym=70, vc=100),
                        glia=dict(sv=50, mi=100, syn_ssv=100, syn_ssv_sym=100, syn_ssv_asym=100, vc=100),
                        compartment=dict(sv=80, mi=100, syn_ssv=100, syn_ssv_sym=100, syn_ssv_asym=100, vc=100))


# -------------------------------------------- COMPARTMENT PREDICTION ---------------------------------------------#


def get_cpmt_model_pts(mpath: Optional[str] = None, device='cuda') -> 'InferenceModel':
    """
    Args:
        mpath: path to model.

    Returns:
        Inference model.
    """
    if mpath is None:
        mpath = global_params.config.mpath_comp_pts
    mpath = os.path.expanduser(mpath)
    from elektronn3.models.convpoint import SegBig
    m = SegBig(4, 3, norm_type='gn').to(device)
    m.load_state_dict(torch.load(mpath)['model_state_dict'])
    return m


def pts_loader_cpmt(ssv_params=None, batchsize: Optional[int] = None,
                    npoints: Optional[int] = None, ctx_size: Optional[float] = None, use_myelin: bool = False,
                    transform: Optional[Callable] = None, base_node_dst: float = 10000, use_subcell: bool = True,
                    ssd_kwargs: Optional[dict] = None, label_remove: List[int] = None,
                    label_mappings: List[Tuple[int, int]] = None):
    """
    Args:
        ssv_params: SuperSegmentationObject kwargs for which samples are generated.
        batchsize: Number of contexts in one batch.
        npoints: Number of points which get sampled from one context.
        ctx_size: Context size.
        use_myelin: Include myelin. This makes loading very slow.
        ssd_kwargs: kwargs for SuperSegmentationDataset.
        transform: Transformations which are applied to each context.
        base_node_dst: Distance between base nodes around which contexts are extracted.
        use_subcell: Flag for including cell organelles.
        label_remove: Remove nodes with certain labels (can only be used after e.g. ads-prediction).
        label_mappings: List of label mappings (also only useful after first ads-prediction).

    Yields:
        SSV params, (features of samples in batch, vertices of samples in batch), original vertex indices of vertices
        in batch, batch_progess, n_batches
    """
    feat_dc = dict(pts_feat_dict)
    # TODO: add use_syntype
    del feat_dc['syn_ssv_asym']
    del feat_dc['syn_ssv_sym']
    if not use_subcell:
        del feat_dc['mi']
        del feat_dc['vc']
        del feat_dc['syn_ssv']
    if ssv_params is None:
        if ssd_kwargs is None:
            raise ValueError
        ssv_params = ssd_kwargs
    for curr_ssv_params in ssv_params:
        # do not write SSV mesh in case it does not exist (will be build from SV meshes)
        ssv = SuperSegmentationObject(mesh_caching=False, **curr_ssv_params)
        hc, voxel_dict = sso2hc(ssv, tuple(feat_dc.keys()), tuple(feat_dc.values()), 'compartment',
                                myelin=use_myelin, label_remove=label_remove, label_mappings=label_mappings)
        ssv.clear_cache()

        # select source nodes for context extraction
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(hc.nodes)
        pcd, idcs = pcd.voxel_down_sample_and_trace(
            base_node_dst, pcd.get_min_bound(), pcd.get_max_bound())
        source_nodes = np.max(idcs, axis=1)
        batchsize = min(len(source_nodes), batchsize)
        n_batches = int(np.ceil(len(source_nodes) / batchsize))

        # add additional source nodes to fill batches
        if len(source_nodes) % batchsize != 0:
            source_nodes = np.concatenate([np.random.choice(source_nodes, batchsize - len(source_nodes) % batchsize),
                                           source_nodes])
        # collect contexts into batches (each batch contains every n_batches contexts (e.g. every 4th if n_batches = 4)
        for ii in range(n_batches):
            batch = np.zeros((batchsize, npoints, 3))
            batch_f = np.zeros((batchsize, npoints, len(feat_dc)))
            batch_mask = np.zeros((batchsize, npoints), dtype=bool)
            idcs_list = []
            # generate contexts
            cnt = 0
            for source_node in source_nodes[ii::n_batches]:
                node_ids = context_splitting_v2(hc, source_node, ctx_size, 1000)
                hc_sub, idcs_sub = extract_subset(hc, node_ids)
                hc_sample, idcs_sample = clouds.sample_cloud(hc_sub, npoints)
                # get vertex indices respective to total hc
                global_idcs = idcs_sub[idcs_sample]
                # prepare masks for filtering sv vertices
                bounds = hc.obj_bounds['sv']
                sv_mask = np.logical_and(global_idcs < bounds[1], global_idcs >= bounds[0])
                hc_sample.set_features(label_binarize(hc_sample.features, classes=np.arange(len(feat_dc))))
                if transform is not None:
                    transform(hc_sample)
                batch[cnt] = hc_sample.vertices
                batch_f[cnt] = hc_sample.features
                idcs_list.append(global_idcs[sv_mask])
                batch_mask[cnt] = sv_mask
                cnt += 1
            batch_progress = ii + 1
            yield curr_ssv_params, (batch_f, batch), (idcs_list, batch_mask, voxel_dict['sv']), batch_progress, \
                  n_batches


def pts_pred_cmpt(m, inp, q_out, d_out, q_cnt, device, bs):
    """
    Args:
        m: Inference model
        inp: Output of loader function (ssv_params, model_inp, batch_idcs, batch_progress, n_batches.
        q_out: Queue which contains SSV IDs.
        d_out: Dict (key: SSV ID, value: list of prediction outputs.
        q_cnt: Progress queue.
        device: Device.
        bs: Batchsize.
    """
    ssv_params, model_inp, batch_info, batch_progress, n_batches = inp
    idcs_list = batch_info[0]
    batch_mask = batch_info[1]
    idcs_voxel = batch_info[2]
    res = []
    with torch.no_grad():
        for ii in range(0, int(np.ceil(len(model_inp[0]) / bs))):
            low = bs * ii
            high = bs * (ii + 1)
            with torch.no_grad():
                g_inp = [torch.from_numpy(i[low:high]).to(device).float() for i in model_inp]
                out = m(*g_inp).cpu().numpy()
                masks = batch_mask[low:high]
                # filter vertices which belong to sv (discard predictions for cell organelles)
                out = out[masks]
            res.append(out)
    if batch_progress == 1:
        res = dict(idcs=np.concatenate(idcs_list), preds=np.concatenate(res),
                   batch_progress=(batch_progress, n_batches), idcs_voxel=idcs_voxel)
    else:
        res = dict(idcs=np.concatenate(idcs_list), preds=np.concatenate(res),
                   batch_progress=(batch_progress, n_batches))

    q_cnt.put(1./n_batches)
    if batch_progress == 1:
        q_out.put(ssv_params['ssv_id'])
    d_out[ssv_params['ssv_id']].append(res)


def pts_postproc_cpmt(sso_id: int, d_in: dict, working_dir: Optional[str] = None, version: Optional[str] = None):
    curr_ix = 0
    sso = SuperSegmentationObject(ssv_id=sso_id, working_dir=working_dir, version=version)
    # TODO: Remove hc stuff after forwarding of voxel_idcs and testing
    hc, _ = sso2hc(sso, 'sv', 0, 'compartment')
    preds = []
    preds_idcs = []
    voxel_idcs = None
    while True:
        if len(d_in[sso_id]) < curr_ix + 1:
            time.sleep(0.5)
            continue
        res = d_in[sso_id][curr_ix]
        preds.append(np.argmax(res['preds'], axis=1))
        preds_idcs.append(res['idcs'])
        if voxel_idcs is None:
            voxel_idcs = res['idcs_voxel']
        d_in[sso_id][curr_ix] = None
        curr_ix += 1
        if res['batch_progress'][0] == res['batch_progress'][1]:
            break
    del d_in[sso_id]
    preds = np.concatenate(preds)
    preds_idcs = np.concatenate(preds_idcs)
    pred_labels = np.ones((len(hc.vertices), 1))*-1
    evaluate_preds(preds_idcs, preds, pred_labels)
    hc.set_labels(pred_labels)
    # TODO: Implement direct forwarding of voxel_idcs (between loader and postproc)
    # sso_vertices = sso.mesh[1].reshape((-1, 3))
    # sso_preds = np.ones((len(sso_vertices), 1))*-1
    # sso_preds[voxel_idcs] = pred_labels
    # ld = sso.label_dict('vertex')
    # ld['cmpt'] = sso_preds
    # ld.push()
    return [sso_id], [True], hc

# ------------------------------------------------- HELPER METHODS --------------------------------------------------#


def evaluate_preds(preds_idcs: np.ndarray, preds: np.ndarray, pred_labels: np.ndarray):
    from collections import defaultdict
    pred_dict = defaultdict(list)
    u_preds_idcs = np.unique(preds_idcs)
    for i in range(len(preds_idcs)):
        pred_dict[preds_idcs[i]].append(preds[i])
    for u_ix in u_preds_idcs:
        counts = np.bincount(pred_dict[u_ix])
        pred_labels[u_ix] = np.argmax(counts)


# -------------------------------------------- SSO TO MORPHX CONVERSION ---------------------------------------------#


@functools.lru_cache(256)
def sso2hc(sso: SuperSegmentationObject, feats: Union[Tuple, str], feat_labels: Union[Tuple, int], pt_type: str, myelin: bool = False,
           radius: int = None, label_remove: List[int] = None, label_mappings: List[Tuple[int, int]] = None):
    if type(feats) == str:
        feats = [feats]
    if type(feat_labels) == int:
        feat_labels = [feat_labels]
    vert_dc = dict()
    obj_bounds = {}
    offset = 0
    idcs_dict = {}
    for k in feats:
        pcd = o3d.geometry.PointCloud()
        verts = sso.load_mesh(k)[1].reshape(-1, 3)
        pcd.points = o3d.utility.Vector3dVector(verts)
        pcd, idcs = pcd.voxel_down_sample_and_trace(pts_feat_ds_dict[pt_type][k], pcd.get_min_bound(),
                                                    pcd.get_max_bound())
        idcs = np.max(idcs, axis=1)
        idcs_dict[k] = idcs
        vert_dc[k] = np.asarray(pcd.points)
        obj_bounds[k] = [offset, offset+len(pcd.points)]
        offset += len(pcd.points)
    sample_feats = np.concatenate([[feat_labels[ii]] * len(vert_dc[k])
                                   for ii, k in enumerate(feats)]).reshape(-1, 1)
    sample_pts = np.concatenate([vert_dc[k] for k in feats])
    if not sso.load_skeleton():
        raise ValueError(f'Couldnt find skeleton of {sso}')
    nodes, edges = sso.skeleton['nodes'] * sso.scaling, sso.skeleton['edges']
    hc = HybridCloud(nodes, edges, vertices=sample_pts, features=sample_feats, obj_bounds=obj_bounds)
    if myelin:
        add_myelin(sso, hc)
    if label_remove is not None:
        hc.remove_nodes(label_remove)
    if label_mappings is not None:
        hc.map_labels(label_mappings)
    # cache verts2node
    _ = hc.verts2node
    if radius is not None:
        # add edges within radius
        kdt = spatial.cKDTree(hc.nodes)
        pairs = list(kdt.query_pairs(radius))
        # remap to subset of indices
        hc._edges = np.concatenate([hc._edges, pairs])
    return hc, idcs_dict


def add_myelin(ssv: SuperSegmentationObject, hc: HybridCloud, average: bool = True):
    """ Tranfers myelin prediction from a SuperSegmentationObject to an existing
        HybridCloud (hc). Myelin is added in form of the types array of the hc,
        where myelinated vertices have type 1 and 0 otherwise. Works in-place.

    Args:
        ssv: SuperSegmentationObject which contains skeleton to which myelin should get mapped.
        hc: HybridCloud to which myelin should get added.
        average: Flag for applying majority vote to the myelin property
    """
    ssv.skeleton['myelin'] = map_myelin2coords(ssv.skeleton['nodes'], mag=4)
    if average:
        majorityvote_skeleton_property(ssv, 'myelin')
        myelinated = ssv.skeleton['myelin_avg10000']
    else:
        myelinated = ssv.skeleton['myelin']
    nodes_idcs = np.arange(len(hc.nodes))
    myel_nodes = nodes_idcs[myelinated.astype(bool)]
    myel_vertices = []
    for node in myel_nodes:
        myel_vertices.extend(hc.verts2node[node])
    # myelinated vertices get type 1, not myelinated vertices get type 0
    types = np.zeros(len(hc.vertices))
    types[myel_vertices] = 1
    hc.set_types(types)


# -------------------------------------------- SIMULATION METHODS ---------------------------------------------#

def test_cmpt_pipeline():
    wd = "/wholebrain/songbird/j0126/areaxfs_v6/"
    ssd_kwargs = dict(working_dir=wd)
    ssd = SuperSegmentationDataset(**ssd_kwargs)
    ssv_ids = np.array([2734465, 2854913, 8003584, 8339462, 10919937, 26169344, 26501121])
    ssv_params = [dict(ssv_id=ssv.id, sv_ids=ssv.sv_ids, **ssd_kwargs)
                  for ssv in ssd.get_super_segmentation_object(ssv_ids)]

    transform = [clouds.Normalization(20000), clouds.Center()]
    transform = clouds.Compose(transform)

    # pipeline simulation
    sim_q_load = Queue()
    sim_q_out = Queue()
    sim_q_cnt = Queue()
    sim_m_postproc = Manager()
    sim_d_out = sim_m_postproc.dict()
    ix = 2
    for k in ssv_ids:
        sim_d_out[k] = sim_m_postproc.list()
    res = pts_loader_cpmt([ssv_params[ix]], 16, 11000, 20000, use_myelin=False, transform=transform,
                          base_node_dst=10000)
    for el in res:
        sim_q_load.put(el)
    m = get_cpmt_model_pts('~/thesis/current_work/paper/ads/2020_06_10_20000_11000/models/state_dict_e17.pth')
    cnt = 0
    while True:
        if not sim_q_load.empty():
            print("Not empty!")
            inp = sim_q_load.get()
            pts_pred_cmpt(m, inp, sim_q_out, sim_d_out, sim_q_cnt, 'cuda', 8)
            cnt = 0
        else:
            print("Empty!")
            cnt += 1
            time.sleep(2)
            if cnt == 2:
                break
    id_list, success_list, hc = pts_postproc_cpmt(ssv_ids[ix], sim_d_out, wd)
    import ipdb
    ipdb.set_trace()


if __name__ == '__main__':
    test_cmpt_pipeline()
