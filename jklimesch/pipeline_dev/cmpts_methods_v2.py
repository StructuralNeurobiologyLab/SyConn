try:
    import open3d as o3d
except ImportError:
    pass  # for sphinx build
import re
import os
import time
import glob
from typing import Iterable, Union, Optional, Tuple, Callable, List
from multiprocessing import Process, Queue, Manager
import numpy as np
from scipy import spatial
import morphx.processing.clouds as clouds
import functools
from morphx.classes.hybridcloud import HybridCloud
from sklearn.preprocessing import label_binarize
from morphx.processing.hybrids import extract_subset
from morphx.processing.objects import context_splitting_v2
from syconn.reps.super_segmentation import SuperSegmentationDataset
from syconn import global_params
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

def predict_cmpt_ssd(ssd_kwargs, mpath: Optional[str] = None,
                     ssv_ids: Optional[Iterable[int]] = None,
                     pred_types: List[str] = None):
    """
    Performs compartment predictions on the ssv's given with ``ssv_params``. The kwargs for predict_pts_plain are
    organized as dicts with the respective values, keyed by the pred_type (see Args below). This enables the pred
    worker to apply multiple different models at once.

    Args:
         ssd_kwargs: Keyword arguments which specify the ssd in use.
         mpath: Path to model folder (which contains models with model identifier) or to single model file.
         ssv_ids: Ids of ssv objects which should get processed.
         pred_types: List of model identifiers (e.g. ['ads', 'abt', 'dnh'] for axon, dendrite, soma; axon, bouton,
            terminal; dendrite, neck, head; models.
    """
    if mpath is None:
        mpath = global_params.config.mpath_compartment_pts
    if os.path.isdir(mpath):
        # multiple models
        mpaths = glob.glob(mpath + '*.pth')
    else:
        # single model, must contain 'cmpt' in its name
        mpaths = [mpath]
        if pred_types is None:
            pred_types = ['cmpt']
    if len(mpaths) != len(pred_types):
        raise ValueError(f"{len(mpaths)} models were found, but {len(pred_types)} prediction types were given.")
    # These variables are needed in predict_pts_plain
    ctx_size = {}
    npoints = {}
    scale_fact = {}
    for path in mpaths:
        for p_t in pred_types:
            if p_t in path:
                kwargs = get_cmpt_kwargs(path)[1]
                if p_t in ctx_size:
                    raise ValueError(f"Found multiple models for prediction type {p_t}.")
                ctx_size[p_t] = kwargs['ctx_size']
                npoints[p_t] = kwargs['npoints']
                scale_fact[p_t] = kwargs['scale_fact']
    kwargs = dict(ctx_size=ctx_size, npoints=npoints, scale_fact=scale_fact)
    loader_kwargs = dict(pred_types=pred_types)
    ssd = SuperSegmentationDataset(**ssd_kwargs)
    if ssv_ids is None:
        ssv_ids = ssd.ssv_ids
    out_dc = predict_pts_plain(ssd_kwargs,
                               ssv_ids=ssv_ids,
                               nloader=10,
                               npredictor=5,
                               bs=10,
                               model_loader=get_cpmt_model_pts,
                               loader_func=pts_loader_cpmt,
                               pred_func=pts_pred_cmpt,
                               postproc_func=pts_postproc_cpmt,
                               mpath=mpath,
                               loader_kwargs=loader_kwargs,
                               model_loader_kwargs=loader_kwargs,
                               **kwargs)
    if not np.all(list(out_dc.values())) or len(out_dc) != len(ssv_ids):
        raise ValueError('Invalid output during compartment prediction.')


def get_cpmt_model_pts(mpath: Optional[str] = None, device='cuda', model_loader_kwargs: Optional[List] = None):
    """ Loads multiple (or one) models with respect to the given pred_types. """
    if mpath is None:
        mpath = global_params.config.mpath_compartment_pts
    if os.path.isdir(mpath):
        # multiple models
        mpaths = glob.glob(mpath + '*.pth')
    else:
        # single model, must contain 'cmpt' in its name
        mpaths = [mpath]
    if model_loader_kwargs is None:
        pred_types = ['cmpt']
    else:
        pred_types = model_loader_kwargs
    models = {}
    from elektronn3.models.convpoint import SegBig
    for path in mpaths:
        for p_t in pred_types:
            if p_t in path:
                mkwargs = get_cmpt_kwargs(path)[0]
                if p_t in models:
                    raise ValueError(f"Found multiple models for prediction type {p_t}.")
                if p_t == 'dnh' or p_t == 'abt':
                    m = SegBig(**mkwargs, reductions=[1024, 512, 256, 64, 16, 8],
                               neighbor_nums=[32, 32, 32, 16, 8, 8, 4, 8, 8, 8, 16, 16, 16]).to(device)
                else:
                    m = SegBig(**mkwargs).to(device)
                m.load_state_dict(torch.load(path)['model_state_dict'])
                models[p_t] = m
    return models


def pts_loader_cpmt(ssv_params = None, pred_types: List[str] = None, batchsize: Optional[int] = None,
                    npoints: Optional[Union[int, dict]] = None,
                    ctx_size: Optional[Union[float, dict]] = None,
                    transform: Optional[Union[Callable, dict]] = None,
                    use_subcell: bool = True, use_myelin: bool = False,
                    ssd_kwargs: Optional[dict] = None):
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
        ssv = SuperSegmentationObject(mesh_caching=False, **curr_ssv_params)
        hc, voxel_dict = sso2hc(ssv, tuple(feat_dc.keys()), tuple(feat_dc.values()), 'compartment',
                                myelin=use_myelin)
        ssv.clear_cache()
        if pred_types is None:
            raise ValueError("pred_types is None. However, pred_types must at least contain one pred_type such as "
                             "'cmpt'")
        for p_t in pred_types:
            base_node_dst = ctx_size[p_t]
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
                batch = np.zeros((batchsize, npoints[p_t], 3))
                batch_f = np.zeros((batchsize, npoints[p_t], len(feat_dc)))
                batch_mask = np.zeros((batchsize, npoints[p_t]), dtype=bool)
                idcs_list = []
                # generate contexts
                cnt = 0
                for source_node in source_nodes[ii::n_batches]:
                    node_ids = context_splitting_v2(hc, source_node, ctx_size[p_t], 1000)
                    hc_sub, idcs_sub = extract_subset(hc, node_ids)
                    hc_sample, idcs_sample = clouds.sample_cloud(hc_sub, npoints[p_t])
                    # get vertex indices respective to total hc
                    global_idcs = idcs_sub[idcs_sample]
                    # prepare masks for filtering sv vertices
                    bounds = hc.obj_bounds['sv']
                    sv_mask = np.logical_and(global_idcs < bounds[1], global_idcs >= bounds[0])
                    hc_sample.set_features(label_binarize(hc_sample.features, classes=np.arange(len(feat_dc))))
                    if transform is not None:
                        transform[p_t](hc_sample)
                    batch[cnt] = hc_sample.vertices
                    batch_f[cnt] = hc_sample.features
                    idcs_list.append(global_idcs[sv_mask])
                    batch_mask[cnt] = sv_mask
                    cnt += 1
                batch_progress = ii + 1
                if batch_progress == 1:
                    yield curr_ssv_params, (batch_f, batch), (idcs_list, batch_mask, voxel_dict['sv']), \
                          (batch_progress, n_batches, p_t, pred_types)
                else:
                    yield curr_ssv_params, (batch_f, batch), (idcs_list, batch_mask, voxel_dict['sv']), \
                          (batch_progress, n_batches, p_t)


def pts_pred_cmpt(m, inp, q_out, d_out, q_cnt, device, bs):
    """
    Args:
        inp: Tuple of ssv_params, (feats, verts), (mapping_indices, masks (for removing cell organelles),
        voxel_indices), (batch_progress, n_batches, p_t (prediction type of this batch), pred_types (only
        present in first batches)
        m: Dict with pytorch models keyed by the prediction type
    """
    ssv_params, model_inp, batch_info, batch_progress = inp
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
                out = m[batch_progress[2]](*g_inp).cpu().numpy()
                masks = batch_mask[low:high]
                # filter vertices which belong to sv (discard predictions for cell organelles)
                out = out[masks]
            res.append(out)
    # batch_progress: (batch_progress, n_batches, p_t, pred_types), bzw. (batch_progress, n_batches, p_t)
    if batch_progress[0] == 1:
        res = dict(idcs=np.concatenate(idcs_list), preds=np.concatenate(res),
                   batch_progress=batch_progress, idcs_voxel=idcs_voxel)
    else:
        res = dict(idcs=np.concatenate(idcs_list), preds=np.concatenate(res),
                   batch_progress=batch_progress)
    q_cnt.put(1./batch_progress[1])
    if batch_progress[0] == 1:
        q_out.put(ssv_params['ssv_id'])
    d_out[ssv_params['ssv_id']].append(res)


def pts_postproc_cpmt(sso_id: int, d_in: dict, working_dir: Optional[str] = None, version: Optional[str] = None):
    """
    Receives predictions from the prediction queue, waits until all predictions for one sso have been received and
    then concatenates and evaluates all the predictions (taking the majority vote over all predictions per vertex).
    The resulting label arrays (number dependent on the number of prediction types (e.g. ads, abt, dnh)) get saved
    in the original sso object.
    """
    curr_ix = 0
    sso = SuperSegmentationObject(ssv_id=sso_id, working_dir=working_dir, version=version)
    preds = {}
    preds_idcs = {}
    # indices of vertices which were chosen during voxelization (allows mapping between hc and sso)
    voxel_idcs = None
    # predictions types which were forwarded from the loading function
    pred_types = None
    p_t_done = {}
    while True:
        if len(d_in[sso_id]) < curr_ix + 1:
            time.sleep(0.5)
            continue
        res = d_in[sso_id][curr_ix]
        if voxel_idcs is None:
            voxel_idcs = res['idcs_voxel']
        if pred_types is None:
            pred_types = res['batch_progress'][3]
            for p_t in pred_types:
                p_t_done[p_t] = False
                preds[p_t] = []
                preds_idcs[p_t] = []
        p_t = res['batch_progress'][2]
        preds[p_t].append(np.argmax(res['preds'], axis=1))
        preds_idcs[p_t].append(res['idcs'])
        d_in[sso_id][curr_ix] = None
        curr_ix += 1
        # check if all predictions for this sso were received (all pred_types must evaluate to True)
        if res['batch_progress'][0] == res['batch_progress'][1]:
            p_t_done[p_t] = True
            done = True
            for p_t in p_t_done:
                done = done and p_t_done[p_t]
            if done:
                break
    del d_in[sso_id]

    # evaluate predictions and map them to the original sso vertices (with respect to
    # indices which were chosen during voxelization
    sso_vertices = sso.mesh[1].reshape((-1, 3))
    ld = sso.label_dict('vertex')
    hc, _ = sso2hc(sso, feats='sv', feat_labels=1, pt_type='compartment')
    for p_t in pred_types:
        preds[p_t] = np.concatenate(preds[p_t])
        preds_idcs[p_t] = np.concatenate(preds_idcs[p_t])
        pred_labels = np.ones((len(voxel_idcs), 1))*-1
        evaluate_preds(preds_idcs[p_t], preds[p_t], pred_labels)
        import ipdb
        ipdb.set_trace()
        # sso_preds = np.ones((len(sso_vertices), 1))*-1
        # sso_preds[voxel_idcs] = pred_labels
        # ld[p_t] = sso_preds
    # ld.push()
    return [sso_id], [True]


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


def get_cmpt_kwargs(mdir: str) -> Tuple[dict, dict]:
    use_norm = True
    use_bias = True
    norm_type = 'gn'
    if 'noBias' in mdir:
        use_bias = False
    if 'noNorm' in mdir:
        use_norm = False
    if '_bn_' in mdir:
        norm_type = 'bn'
    npoints = int(re.findall(r'_nb(\d+)_', mdir)[0])
    scale_fact = int(re.findall(r'_scale(\d+)_', mdir)[0])
    ctx = int(re.findall(r'_ctx(\d+)_', mdir)[0])
    feat_dim = int(re.findall(r'_fdim(\d+)_', mdir)[0])
    class_num = int(re.findall(r'_cnum(\d+)_', mdir)[0])
    # TODO: Fix neighbor_nums or create extra model
    mkwargs = dict(input_channels=feat_dim, output_channels=class_num, use_norm=use_norm, use_bias=use_bias,
                   norm_type=norm_type)
    loader_kwargs = dict(ctx_size=ctx, scale_fact=scale_fact, npoints=npoints)
    return mkwargs, loader_kwargs


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

    mpath = '~/thesis/current_work/paper/test_models/'
    mpath = os.path.expanduser(mpath)
    pred_types = ['ads', 'abt', 'dnh']
    if mpath is None:
        mpath = global_params.config.mpath_compartment_pts
    if os.path.isdir(mpath):
        # multiple models
        mpaths = glob.glob(mpath + '*.pth')
    else:
        # single model, must contain 'cmpt' in its name
        mpaths = [mpath]
        if pred_types is None:
            pred_types = ['cmpt']
    if len(mpaths) != len(pred_types):
        raise ValueError(f"{len(mpaths)} models were found, but {len(pred_types)} prediction types were given.")
    # These variables are needed in predict_pts_plain
    ctx_size = {}
    npoints = {}
    scale_fact = {}
    for path in mpaths:
        for p_t in pred_types:
            if p_t in path:
                kwargs = get_cmpt_kwargs(path)[1]
                if p_t in ctx_size:
                    raise ValueError(f"Found multiple models for prediction type {p_t}.")
                ctx_size[p_t] = kwargs['ctx_size']
                npoints[p_t] = kwargs['npoints']
                scale_fact[p_t] = kwargs['scale_fact']

    transform = {}
    for key in pred_types:
        transform[key] = clouds.Compose([clouds.Normalization(scale_fact[key]), clouds.Center()])

    # pipeline simulation
    sim_q_load = Queue()
    sim_q_out = Queue()
    sim_q_cnt = Queue()
    sim_m_postproc = Manager()
    sim_d_out = sim_m_postproc.dict()
    ix = 6
    for k in ssv_ids:
        sim_d_out[k] = sim_m_postproc.list()
    res = pts_loader_cpmt([ssv_params[ix]], pred_types=pred_types, batchsize=8, ctx_size=ctx_size,
                          npoints=npoints, use_myelin=False, transform=transform)
    for el in res:
        sim_q_load.put(el)
    m = get_cpmt_model_pts(mpath, model_loader_kwargs=pred_types)
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
    id_list, success_list = pts_postproc_cpmt(ssv_ids[ix], sim_d_out, wd)


if __name__ == '__main__':
    test_cmpt_pipeline()
