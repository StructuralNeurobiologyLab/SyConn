# SyConn-dev
# Copyright (c) 2016 Philipp J. Schubert
# All rights reserved

# import here, otherwise it might fail if it is imported after importing torch
# see https://github.com/pytorch/pytorch/issues/19739
try:
    import open3d as o3d
except ImportError:
    pass  # for sphinx build
import collections
import functools
import sys
import re
import os
import time
import glob
from collections import defaultdict
from multiprocessing import Process, Queue, Manager, queues
from typing import Iterable, Union, Optional, Tuple, Callable, List
import morphx.processing.clouds as clouds
import networkx as nx
import numpy as np
import scipy.special
import tqdm
from morphx.classes.hybridcloud import HybridCloud
from morphx.processing.hybrids import extract_subset
from morphx.processing.objects import bfs_vertices, context_splitting_kdt, context_splitting_graph_many
from scipy import spatial
from scipy.spatial import cKDTree
from sklearn.preprocessing import label_binarize
from syconn import global_params
from syconn.handler import log_handler
from syconn.handler.basics import chunkify_successive, chunkify
from syconn.mp.mp_utils import start_multiprocess_imap
from syconn.handler.prediction import certainty_estimate
from syconn.reps.super_segmentation import SuperSegmentationDataset
from syconn.reps.super_segmentation import SuperSegmentationObject, semsegaxoness2skel
from syconn.reps.super_segmentation_helper import map_myelin2coords, majorityvote_skeleton_property

# for readthedocs build
try:
    import torch
except ImportError:
    pass
# TODO: specify further, add to config
pts_feat_dict = dict(sv=0, mi=1, syn_ssv=3, syn_ssv_sym=3, syn_ssv_asym=4, vc=2, sv_myelin=5)
# in nm, should be replaced by Poisson disk sampling
pts_feat_ds_dict = dict(celltype=dict(sv=70, mi=100, syn_ssv=70, syn_ssv_sym=70, syn_ssv_asym=70, vc=100),
                        glia=dict(sv=50, mi=100, syn_ssv=100, syn_ssv_sym=100, syn_ssv_asym=100, vc=100),
                        compartment=dict(sv=80, mi=100, syn_ssv=100, syn_ssv_sym=100, syn_ssv_asym=100, vc=100))


# TODO: move to handler.basics
def write_ply(fn, verts, colors):
    ply_header = '''ply
    format ascii 1.0
    element vertex %(vert_num)d
    property float x
    property float y
    property float z
    property uchar red
    property uchar green
    property uchar blue
    end_header

    '''
    verts = np.hstack([verts, colors])
    with open(fn, 'wb') as f:
        f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
        np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')


def write_pts_ply(fname: str, pts: np.ndarray, feats: np.ndarray, binarized=False):
    assert pts.ndim == 2
    assert feats.ndim == 2
    pts = np.asarray(pts)
    feats = np.asarray(feats)
    col_dc = {0: [[200, 200, 200]], 1: [[100, 100, 200]], 3: [[200, 100, 200]],
              4: [[250, 100, 100]], 2: [[100, 200, 100]], 5: [[100, 200, 200]],
              6: [0, 0, 0]}
    if not binarized:
        feats = label_binarize(feats, np.arange(np.max(feats) + 1))
    cols = np.zeros(pts.shape, dtype=np.uint8)
    for k in range(feats.shape[1]):
        mask = feats[:, k] == 1
        cols[mask] = col_dc[k]
    write_ply(fname, pts, cols)


def worker_postproc(q_out: Queue, q_postproc: Queue, d_postproc: dict,
                    postproc_func: Callable, postproc_kwargs: dict,
                    n_worker_pred):
    """

    Args:
        q_out:
        q_postproc:
        d_postproc:
        postproc_func:
        postproc_kwargs:
        n_worker_pred:
    """
    stops_received = set()
    while True:
        try:
            inp = q_postproc.get_nowait()
            if 'STOP' in inp:
                if inp not in stops_received:
                    stops_received.add(inp)
                else:
                    q_postproc.put_nowait(inp)
                    time.sleep(np.random.randint(2))
                if len(stops_received) == n_worker_pred:
                    break
                continue
        except queues.Empty:
            if len(stops_received) == n_worker_pred:
                break
            time.sleep(0.5)
            continue
        try:
            res = postproc_func(inp, d_postproc, **postproc_kwargs)
            q_out.put_nowait(res)
        except Exception as e:
            log_handler.error(f'Error during worker_postproc "{str(postproc_func)}": {str(e)}')
            break
    log_handler.debug(f'Worker postproc done.')
    q_out.put('END')


def worker_pred(worker_cnt: int, q_out: Queue, d_out: dict, q_progress: Queue, q_in: Queue,
                model_loader: Callable, pred_func: Callable, n_worker_load: int, n_worker_postproc: int,
                device: str, mpath: Optional[str] = None, bs: Optional[int] = None,
                model_loader_kwargs: Optional[dict] = None):
    """

    Args:
        worker_cnt: Index of this worker.
        q_out: Queue which contains SSV IDs.
        d_out: Dict (key: SSV ID, value: list of prediction outputs).
        q_progress: Progress queue.
        q_in: Queue with the output of the loaders.
        model_loader: Factory method for the pytorch model.
        pred_func: Method used to perform the prediction. Must have the syntax
            (model, input, q_out, d_out, q_progress, device, batchsize). The return needs to be placed in the dict.
             Only put the SSV ID in `q_out` for the first batch of an SSV! Otherwise multiple postproc worker might
             be assigned to the same object!
        mpath: Path to the pytorch model.
        device: Device
        bs: Batch size.
        n_worker_load: Number of loader.
        n_worker_postproc: Number of postproc worker.
        model_loader_kwargs: Additional keyword arguments for the model loader.
    """
    # try:
    if model_loader_kwargs is None:
        model_loader_kwargs = dict()
    m = model_loader(mpath, device, **model_loader_kwargs)
    stops_received = set()
    while True:
        try:
            inp = q_in.get_nowait()
            if 'STOP' in inp:
                if inp not in stops_received:
                    stops_received.add(inp)
                else:
                    q_in.put_nowait(inp)
                    time.sleep(np.random.randint(25) / 10)
                if len(stops_received) == n_worker_load:
                    break
                continue
        except queues.Empty:
            time.sleep(0.25)
            continue
        pred_func(m, inp, q_out, d_out, q_progress, device, bs)
    # except Exception as e:
    #     log_handler.error(f'Error during worker_pred "{str(model_loader)}" or "{str(pred_func)}": {str(e)}')
    for _ in range(n_worker_postproc):
        q_out.put(f'STOP{worker_cnt}')
    log_handler.debug(f'Pred worker {worker_cnt} done.')


def worker_load(worker_cnt: int, q_loader: Queue, q_out: Queue, q_loader_sync: Queue, loader_func: Callable,
                n_worker_pred: int):
    """

    Args:
        worker_cnt:
        q_loader:
        q_out:
        q_loader_sync:
        loader_func:
        n_worker_pred:
    """
    while True:
        kwargs = q_loader.get()
        if kwargs is None:
            break
        try:
            res = loader_func(**kwargs)
            for el in res:
                dt = 0
                while True:
                    if dt > 60:
                        log_handler.error(f'Loader {worker_cnt}: Locked for {dt} s.')
                    try:
                        q_out.put_nowait(el)
                    except queues.Full:
                        time.sleep(0.25)
                        dt += 1
                    finally:
                        break
        except Exception as e:
            log_handler.error(f'Error during loader_func {str(loader_func)}: {str(e)}')
            break

    time.sleep(1)
    for _ in range(n_worker_pred):
        q_out.put(f'STOP{worker_cnt}')
    log_handler.debug(f'Loader {worker_cnt} done.')
    q_loader_sync.put('DONE')


def listener(q_progress: Queue, q_loader_sync: Queue, nloader: int, total: int,
             show_progress: bool = True):
    """

    Args:
        q_progress:
        q_loader_sync:
        nloader:
        total:
        show_progress
    """
    if show_progress:
        pbar = tqdm.tqdm(total=total, leave=False)
    cnt_loder_done = 0
    while True:
        if q_progress.empty():
            time.sleep(0.25)
        else:
            res = q_progress.get_nowait()
            if res is None:  # final stop
                if show_progress:
                    pbar.close()
                if cnt_loder_done != nloader:
                    log_handler.error(f'Only {cnt_loder_done}/{nloader} loader finished.')
                    sys.exit(1)
                break
            if show_progress:
                pbar.update(res)
        if q_loader_sync.empty() or cnt_loder_done == nloader:
            pass
        else:
            _ = q_loader_sync.get_nowait()
            cnt_loder_done += 1
    log_handler.debug(f'Listener done')


def _size_counter(args):
    ssv_id, ssd_kwargs = args
    return SuperSegmentationObject(ssv_id, **ssd_kwargs).size


def predict_pts_plain(ssd_kwargs: Union[dict, Iterable], model_loader: Callable,
                      loader_func: Callable, pred_func: Callable,
                      npoints: Union[int, dict], scale_fact: Union[float, dict], ctx_size: Union[int, dict],
                      postproc_func: Optional[Callable] = None,
                      postproc_kwargs: Optional[dict] = None,
                      output_func: Optional[Callable] = None,
                      mpath: Optional[str] = None,
                      nloader: int = 4, npredictor: int = 2, npostproc: int = 2,
                      ssv_ids: Optional[Union[list, np.ndarray]] = None,
                      use_test_aug: bool = False,
                      seeded: bool = False,
                      device: str = 'cuda', bs: Union[int, dict] = 40,
                      loader_kwargs: Optional[dict] = None,
                      model_loader_kwargs: Optional[dict] = None,
                      show_progress: bool = True) -> dict:
    """
    Perform cell type predictions of cell reconstructions on sampled point sets from the
    cell's vertices. The number of predictions `npreds` per cell is calculated based on the
    fraction of the total number of vertices over `npoints` times two, but at least 5.
    Every point set is constructed by collecting the vertices associated with skeleton within a
    breadth-first search up to a maximum of `npoints`.

    Overview:
        * loader (load_func) -> input queue
        * input queue -> prediction worker
        * prediction worker (pred_func) -> postprocessing queue (default: identity)
        * postprocessing worker (postproc_func) -> output queue
        * output queue -> result dictionary (return)

    Args:
        ssd_kwargs: Keyword arguments to specify the underlying ``SuperSegmentationDataset``. If type dict,
            `redundancy` kwarg will be used to process each cell at minimum `redundancy` times and at most as many
            times as `npoints` fits into the number of cell vertices (times three). The `loader_func` needs to handle
            ``ssd_kwargs`` and ``ssv_ids`` as kwargs (see :py:func:`~pts_loader_scalar`). If type iterable, then the
            `loader_func` needs to handle `ssv_params` which is a tuple of SSV ID and working directory
            (see :py:func:`~pts_loader_local_skel`).
        model_loader: Function which returns the pytorch model object.
        mpath: Path to model.
        loader_func: Loader function, used by `nloader` workers retrieving samples.
        pred_func: Predict function, used by `npredictor` workers performing the inference.
        npoints: Number of points used to generate a sample.
        ctx_size: .
        scale_fact: Scale factor; used to normalize point clouds prior to model inference.
        postproc_kwargs: Keyword arguments for post-processing.
        postproc_func: Optional post-processing layer for the output of pred_func.
        output_func: Transforms the elements in the output queue and stores it in the final dictionary.
            If None, elements as returned by `pred_func` are assumed to be of form ``(ssv_ids, results)``:

                def output_func(res_dc, (ssv_ids, predictions)):
                    for ssv_id, ssv_pred in zip(*(ssv_ids, predictions)):
                        res_dc[ssv_id].append(ssv_pred)

        nloader: Number of workers loading samples from the given cell IDs via `load_func`.
        npredictor: Number of workers which will call `model_loader` and process (via `pred_func`) the output of
            the "loaders", i.e. workers which retrieve samples via `loader_func`.
        npostproc: Optional worker for post processing, see `postproc_func`.
        ssv_ids: IDs of cells to predict.
        use_test_aug: Use test-time augmentations. Currently this adds the following transformation
            to the basic transforms:

                transform = [clouds.Normalization(scale_fact), clouds.Center()]
                [clouds.RandomVariation((-5, 5), distr='normal')] + transform + [clouds.RandomRotate()]

        device: pytorch device.
        bs: Batch size.
        loader_kwargs: Optional keyword arguments for loader func.
        seeded: Loader will hash ssv, sample and batch IDs to generate a random seed.
        model_loader_kwargs: Optional keyword arguments for model_loader func.
        show_progress: Show progress bar.

    Examples:

        from syconn.handler.prediction_pts import pts_loader_scalar, pts_pred_scalar


        def load_model(mkwargs, device):
            from elektronn3.models.convpoint import ModelNet40
            import torch
            m = ModelNet40(5, 8, **mkwargs).to(device)
            m.load_state_dict(torch.load(mpath)['model_state_dict'])
            m = torch.nn.DataParallel(m)
            m.eval()
            return m

        ssd_kwargs = dict(working_dir='', version='ctgt')
        mdir = ''
        ssv_ids = []
        npoints = 25000
        scale_fact = 30000
        mkwargs = dict(use_bn=False, track_running_stats=False)
        dict_out = predict_pts_plain(ssd_kwargs, load_model, pts_loader_scalar, pts_pred_scalar,
                                     npoints, scale_fact, ssv_ids=ssv_ids,
                                     nloader=2, npredictor=1, use_test_aug=True)

    Returns:
        Dictionary with the prediction result. Key: SSV ID, value: output of `pred_func` to output queue.

    """
    apply_proxy_fix()
    m = Manager()

    if loader_kwargs is None:
        loader_kwargs = dict()
    if model_loader_kwargs is None:
        model_loader_kwargs = dict()
    if output_func is None:
        def output_func(res_dc, ret):
            for ix, out in zip(*ret):
                res_dc[ix].append(out)
    if postproc_func is None:
        def postproc_func(x, *args, **kwargs): return x
    if postproc_kwargs is None:
        postproc_kwargs = dict()

    if type(scale_fact) == dict:
        transform = {}
        for p_t in loader_kwargs['pred_types']:
            transform[p_t] = clouds.Compose([clouds.Normalization(scale_fact[p_t]), clouds.Center()])
    else:
        transform = [clouds.Normalization(scale_fact), clouds.Center()]
        if use_test_aug:
            transform = [clouds.RandomVariation((-5, 5), distr='normal')] + transform + [clouds.RandomRotate()]
        transform = clouds.Compose(transform)

    if type(ssd_kwargs) is dict:
        params_kwargs = dict(batchsize=bs, npoints=npoints, ssd_kwargs=ssd_kwargs,
                             transform=transform, ctx_size=ctx_size, seeded=seeded, **loader_kwargs)
        if ssv_ids is None:
            ssd = SuperSegmentationDataset(**ssd_kwargs)
            ssv_ids = ssd.ssv_ids
        else:
            ssv_ids = np.array(ssv_ids, np.uint64)
        ssv_sizes = start_multiprocess_imap(_size_counter, [(ssv_id, ssd_kwargs) for ssv_id in ssv_ids],
                                            nb_cpus=None)
        ssv_sizes = np.array(ssv_sizes)
        sorted_ix = np.argsort(ssv_sizes)[::-1]
        ssv_ids = ssv_ids[sorted_ix]
        params_in = [{**params_kwargs, **dict(ssv_ids=[ssv_id])} for ssv_id in ssv_ids]
    else:
        params_kwargs = dict(batchsize=bs, npoints=npoints, transform=transform, ctx_size=ctx_size, **loader_kwargs)
        params_in = [{**params_kwargs, **dict(ssv_params=[ch])} for ch in ssd_kwargs]
        ssv_ids = np.array([el['ssv_id'] for el in ssd_kwargs])

    nsamples_tot = len(ssv_ids)
    if 'redundancy' in loader_kwargs:
        nsamples_tot *= loader_kwargs['redundancy']

    q_loader = Queue()
    # fill 1 param per worker now; fill the rest one after another s.t. number of elements in 'q_load' is controlled
    for el in params_in[:nloader]:
        q_loader.put_nowait(el)

    q_load = Queue()
    q_progress = Queue()
    d_postproc = m.dict()
    for k in ssv_ids:
        d_postproc[k] = m.Queue()
    q_postproc = Queue()

    q_out = Queue()
    q_loader_sync = Queue()
    producers = [Process(target=worker_load, args=(ii, q_loader, q_load, q_loader_sync, loader_func, npredictor))
                 for ii in range(nloader)]
    for p in producers:
        p.start()
    consumers = [Process(target=worker_pred, args=(ii, q_postproc, d_postproc, q_progress, q_load, model_loader,
                                                   pred_func, nloader, npostproc, device, mpath, bs,
                                                   model_loader_kwargs)) for ii in range(npredictor)]
    for c in consumers:
        c.start()
    postprocs = [Process(target=worker_postproc, args=(q_out, q_postproc, d_postproc, postproc_func, postproc_kwargs,
                                                       npredictor)) for _ in range(npostproc)]
    for c in postprocs:
        c.start()

    for el in params_in[nloader:] + [None] * nloader:
        while q_load.qsize() + q_loader.qsize() >= 2 * npredictor:
            time.sleep(1)
        q_loader.put(el)

    dict_out = collections.defaultdict(list)
    cnt_end = 0
    lsnr = Process(target=listener, args=(q_progress, q_loader_sync, nloader, nsamples_tot, show_progress))
    lsnr.start()
    while True:
        if q_out.empty():
            if cnt_end == npostproc:
                break
            time.sleep(0.5)
            continue
        res = q_out.get_nowait()
        if res == 'END':
            cnt_end += 1
            continue
        output_func(dict_out, res)
    q_progress.put_nowait(None)
    lsnr.join()
    # at this point all jobs should have finished
    for p in producers:
        p.join(timeout=10)
        if p.is_alive():
            raise ValueError(f'Job {p} is still running.')
        p.close()
    for c in consumers:
        c.join(timeout=10)
        if c.is_alive():
            raise ValueError(f'Job {c} is still running.')
        c.close()
    for c in postprocs:
        c.join(timeout=10)
        if c.is_alive():
            raise ValueError(f'Job {c} is still running.')
        c.close()
    if len(dict_out) != len(ssv_ids):
        raise ValueError(f'Missing {len(ssv_ids) - len(dict_out)} cell predictions: '
                         f'{np.setdiff1d(ssv_ids, list(dict_out.keys()))}')
    m.shutdown()
    return dict_out


@functools.lru_cache(256)
def _load_ssv_hc_cached(args):
    return _load_ssv_hc(args)


def _load_ssv_hc(args):
    """

    Args:
        args:

    Returns:

    """
    # TODO: refactor
    map_myelin = False
    recalc_skeletons = False
    if len(args) == 5:
        ssv, feats, feat_labels, pt_type, radius = args
    elif len(args) == 6:
        ssv, feats, feat_labels, pt_type, radius, map_myelin = args
    else:
        ssv, feats, feat_labels, pt_type, radius, map_myelin, recalc_skeletons = args
    vert_dc = dict()

    if pt_type == 'glia' and recalc_skeletons:  # at this point skeletons have not been computed
        ssv.calculate_skeleton(force=True, save=False)

    if not ssv.load_skeleton():
        raise ValueError(f'Couldnt find skeleton of {ssv}')
    if map_myelin:
        _, _, myelinated = ssv._pred2mesh(ssv.skeleton['nodes'] * ssv.scaling, ssv.skeleton['myelin_avg10000'],
                                          return_color=False)
        myelinated = myelinated.astype(np.bool)
    for k in feats:
        if k == 'sv_myelin':  # do not process - 'sv_myelin' is processed together with 'sv'
            continue
        pcd = o3d.geometry.PointCloud()
        verts = ssv.load_mesh(k)[1].reshape(-1, 3)
        pcd.points = o3d.utility.Vector3dVector(verts)
        if map_myelin and k == 'sv':
            pcd, idcs = pcd.voxel_down_sample_and_trace(
                pts_feat_ds_dict[pt_type][k], pcd.get_min_bound(), pcd.get_max_bound())
            vert_ixs = np.max(idcs, axis=1)
            sv_verts = np.asarray(pcd.points, dtype=np.float32)
            vert_dc[k] = sv_verts[~myelinated[vert_ixs]]
            vert_dc['sv_myelin'] = sv_verts[myelinated[vert_ixs]]
        else:
            pcd = pcd.voxel_down_sample(voxel_size=pts_feat_ds_dict[pt_type][k])
            vert_dc[k] = np.asarray(pcd.points)
    sample_feats = np.concatenate([[feat_labels[ii]] * len(vert_dc[k])
                                   for ii, k in enumerate(feats)])
    sample_pts = np.concatenate([vert_dc[k] for k in feats])
    nodes, edges = ssv.skeleton['nodes'] * ssv.scaling, ssv.skeleton['edges']
    hc = HybridCloud(nodes, edges, vertices=sample_pts, features=sample_feats)
    # cache verts2node
    _ = hc.verts2node
    if radius is not None:
        # add edges within radius
        kdt = spatial.cKDTree(hc.nodes)
        pairs = list(kdt.query_pairs(radius))
        # remap to subset of indices
        hc._edges = np.concatenate([hc._edges, pairs])
    return hc


def pts_loader_scalar_infer(ssd_kwargs: dict, ssv_ids: Tuple[Union[list, np.ndarray], int],
                            batchsize: int, npoints: int, ctx_size: float,
                            transform: Optional[Callable] = None, seeded: bool = False,
                            use_ctx_sampling: bool = True, redundancy: int = 20, map_myelin: bool = False,
                            use_syntype: bool = True, cellshape_only: bool = False,
                            ) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Generator for SSV point cloud samples of size `npoints`. Currently used for
    per-cell point-to-scalar tasks, e.g. cell type prediction.

    Args:
        ssd_kwargs: SuperSegmentationDataset keyword arguments specifying e.g.
            working directory, version, etc.
        ssv_ids: SuperSegmentationObject IDs and redundancy.
        batchsize: Only used during training.
        npoints: Number of points used to generate sample context.
        ctx_size: Euclidean distance between the two most distant nodes in nm.
            1/4 samples will fluctuate around factor 0.7+-0.2 (mean, s.d.) during training.
        transform: Transformation/agumentation applied to every sample.
        seeded: If True, will set the seed to ``hash(frozenset(ssv_id, n_samples, curr_batch_count))``.
        use_ctx_sampling: Use context based sampling. If True, uses `ctx_size` (in nm). Otherwise vist skeleton nodes
            until `npoints` have been collected.
        redundancy: Number of samples generated from each SSV.
        map_myelin: Use myelin as vertex feature. Requires myelin node attribute 'myelin_avg10000'.
        use_syntype:
        cellshape_only:

    Yields: SSV IDs [M, ], (point feature [N, C], point location [N, 3])

    """
    np.random.shuffle(ssv_ids)
    ssd = SuperSegmentationDataset(**ssd_kwargs)
    feat_dc = dict(pts_feat_dict)
    if cellshape_only:
        feat_dc = dict(sv=feat_dc['sv'])
    else:
        if use_syntype:
            if 'syn_ssv' in feat_dc:
                del feat_dc['syn_ssv']
        else:
            del feat_dc['syn_ssv_sym']
            del feat_dc['syn_ssv_asym']
            assert 'syn_ssv' in feat_dc
        if not map_myelin:
            del feat_dc['sv_myelin']
    for ssv_id in ssv_ids:
        redundancy_ssv = int(redundancy)
        n_batches = int(np.ceil(redundancy_ssv / batchsize))
        ssv = ssd.get_super_segmentation_object(ssv_id)
        hc = _load_ssv_hc((ssv, tuple(feat_dc.keys()), tuple(feat_dc.values()), 'celltype', None, map_myelin))
        ssv.clear_cache()

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(hc.nodes)
        pcd, idcs = pcd.voxel_down_sample_and_trace(2500, pcd.get_min_bound(), pcd.get_max_bound())
        nodes = np.max(idcs, axis=1)
        source_nodes_all = np.random.choice(nodes, redundancy_ssv, replace=len(nodes) < redundancy_ssv)
        rand_ixs = chunkify(np.random.choice(redundancy_ssv, redundancy_ssv, replace=False), n_batches)
        npoints_ssv = min(len(hc.vertices), npoints)
        if npoints_ssv == 0:
            log_handler.warn(f'Found SSV with 0 vertices: {ssv}')
        if use_ctx_sampling:
            node_ids_all = np.array(context_splitting_kdt(hc, source_nodes_all, ctx_size), dtype=object)
        else:
            node_ids_all = np.array([bfs_vertices(hc, sn, npoints_ssv) for sn in source_nodes_all], dtype=object)
        for ii in range(n_batches):
            n_samples = min(redundancy_ssv, batchsize)
            redundancy_ssv -= batchsize
            if seeded:
                np.random.seed(np.uint32(hash(frozenset((ssv_id, n_samples, ii)))))
            batch = np.zeros((n_samples, npoints_ssv, 3))
            batch_f = np.zeros((n_samples, npoints_ssv, len(feat_dc)))
            cnt = 0
            curr_batch_ixs = rand_ixs[ii]
            source_nodes_batch = source_nodes_all[curr_batch_ixs]
            node_ids_batch = node_ids_all[curr_batch_ixs]
            for source_node, node_ids in zip(source_nodes_batch, node_ids_batch):
                node_ids = node_ids.astype(np.int32)
                # This might be slow
                sn_cnt = 1
                while True:
                    hc_sub = extract_subset(hc, node_ids)[0]  # only pass HybridCloud
                    sample_feats = hc_sub.features
                    if len(sample_feats) > 0 or npoints_ssv == 0:
                        break
                    if sn_cnt >= len(source_nodes_all):
                        msg = (f'Crould not find context with > 0 vertices during batch generation of {ssv} '
                               f'in method "pts_loader_scalar_infer".')
                        log_handler.error(msg)
                        raise ValueError(msg)
                    source_node = source_nodes_all[sn_cnt]
                    if use_ctx_sampling:
                        node_ids = context_splitting_kdt(hc, [source_node], ctx_size)[0]
                    else:
                        node_ids = bfs_vertices(hc, source_node, npoints_ssv)
                    sn_cnt += 1
                sample_feats = hc_sub.features
                sample_pts = hc_sub.vertices
                # make sure there is always the same number of points within a batch
                sample_ixs = np.arange(len(sample_pts))
                sample_pts = sample_pts[sample_ixs][:npoints_ssv]
                sample_feats = sample_feats[sample_ixs][:npoints_ssv]
                npoints_add = npoints_ssv - len(sample_pts)
                idx = np.random.choice(len(sample_pts), npoints_add)
                sample_pts = np.concatenate([sample_pts, sample_pts[idx]])
                sample_feats = np.concatenate([sample_feats, sample_feats[idx]])
                # one hot encoding
                sample_feats = label_binarize(sample_feats, classes=np.arange(len(feat_dc)))
                hc_sub._vertices = sample_pts
                hc_sub._features = sample_feats
                if transform is not None:
                    transform(hc_sub)
                batch[cnt] = hc_sub.vertices
                batch_f[cnt] = hc_sub.features
                cnt += 1
            assert cnt == n_samples
            yield ssv.ssv_kwargs, (batch_f, batch), ii + 1, n_batches


def pts_loader_scalar(ssd_kwargs: dict, ssv_ids: Union[list, np.ndarray], batchsize: int, npoints: int, ctx_size: float,
                      transform: Optional[Callable] = None, train: bool = False, draw_local: bool = False,
                      draw_local_dist: int = 1000, use_ctx_sampling: bool = True, cache: Optional[bool] = True,
                      map_myelin: bool = False, use_syntype: bool = True, cellshape_only: bool = False
                      ) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Generator for SSV point cloud samples of size `npoints`. Currently used for
    per-cell point-to-scalar tasks, e.g. cell type prediction.

    Args:
        ssd_kwargs: SuperSegmentationDataset keyword arguments specifying e.g.
            working directory, version, etc.
        ssv_ids: SuperSegmentationObject IDs for which samples are generated.
        batchsize: Only used during training.
        npoints: Number of points used to generate sample context.
        ctx_size: Euclidean distance between the two most distant nodes in nm.
            1/4 samples will fluctuate around factor 0.7+-0.2 (mean, s.d.) during training.
        transform: Transformation/agumentation applied to every sample.
        train: If false, eval mode -> batch size won't be used.
        draw_local: Will draw similar contexts from approx.
            the same location, requires a single unique element in ssv_ids
        draw_local_dist: Maximum distance to similar source node in nm.
        use_ctx_sampling: Use context based sampling. If True, uses `ctx_size` (in nm). Otherwise vist skeleton nodes
            until `npoints` have been collected.
        cache: Cache loaded SSVs.
        map_myelin: Use myelin as vertex feature. Requires myelin node attribute 'myelin_avg10000'.
        use_syntype:
        cellshape_only:

    Yields: SSV IDs [M, ], (point feature [N, C], point location [N, 3])

    """
    np.random.shuffle(ssv_ids)
    ssd = SuperSegmentationDataset(**ssd_kwargs)
    feat_dc = dict(pts_feat_dict)
    if cellshape_only:
        feat_dc = dict(sv=feat_dc['sv'])
    else:
        if use_syntype:
            if 'syn_ssv' in feat_dc:
                del feat_dc['syn_ssv']
        else:
            del feat_dc['syn_ssv_sym']
            del feat_dc['syn_ssv_asym']
            assert 'syn_ssv' in feat_dc
        if not map_myelin:
            del feat_dc['sv_myelin']
    if cache is None:
        cache = train
    if not train:
        raise NotImplementedError('Use "pts_loader_scalar_infer" for inference.')
    else:
        ssv_ids = np.unique(ssv_ids)
        for curr_ssvid in ssv_ids:
            ssv = ssd.get_super_segmentation_object(curr_ssvid)
            args = (ssv, tuple(feat_dc.keys()), tuple(feat_dc.values()), 'celltype', None, map_myelin)
            if cache:
                hc = _load_ssv_hc_cached(args)
            else:
                hc = _load_ssv_hc(args)
            ssv.clear_cache()
            # fluctuate context size in 1/4 samples
            if np.random.randint(0, 4) == 0:
                ctx_size_fluct = max((np.random.randn(1)[0] * 0.1 + 0.7), 0.33) * ctx_size
            else:
                ctx_size_fluct = ctx_size
            npoints_ssv = min(len(hc.vertices), npoints)
            # add a +-10% fluctuation in the number of input points
            npoints_add = np.random.randint(-int(npoints_ssv * 0.1), int(npoints_ssv * 0.1))
            npoints_ssv += npoints_add
            batch = np.zeros((batchsize, npoints_ssv, 3))
            batch_f = np.zeros((batchsize, npoints_ssv, len(feat_dc)))
            ixs = np.ones((batchsize,), dtype=np.uint64) * ssv.id
            cnt = 0
            source_nodes = np.random.choice(len(hc.nodes), batchsize, replace=len(hc.nodes) < batchsize)
            if draw_local:
                # only use half of the nodes and choose a close-by node as root for similar context retrieval
                source_nodes = source_nodes[::2]
                sn_new = []
                g = hc.graph(simple=False)
                for n in source_nodes:
                    sn_new.append(n)
                    # just choose any node within the cell randomly
                    if np.isinf(draw_local_dist):
                        sn_new.append(np.random.randint(0, len(hc.nodes)))
                    else:
                        paths = nx.single_source_dijkstra_path(g, n, draw_local_dist)
                        neighs = np.array(list(paths.keys()), dtype=np.int32)
                        sn_new.append(np.random.choice(neighs, 1)[0])
                source_nodes = sn_new
            for source_node in source_nodes:
                cnt_ctx = 0
                while True:
                    if cnt_ctx > 2*len(source_nodes):
                        raise ValueError(f'Could not find context with > 0 vertices in {ssv}.')
                    cnt_ctx += 1
                    if use_ctx_sampling:
                        node_ids = context_splitting_kdt(hc, source_node, ctx_size_fluct)
                    else:
                        node_ids = bfs_vertices(hc, source_node, npoints_ssv)
                    hc_sub = extract_subset(hc, node_ids)[0]  # only pass HybridCloud
                    sample_feats = hc_sub.features
                    if len(sample_feats) > 0:
                        break
                    source_node = np.random.choice(source_nodes)

                sample_feats = hc_sub.features
                sample_pts = hc_sub.vertices
                # shuffling
                sample_ixs = np.arange(len(sample_pts))
                np.random.shuffle(sample_ixs)
                sample_pts = sample_pts[sample_ixs][:npoints_ssv]
                sample_feats = sample_feats[sample_ixs][:npoints_ssv]
                # add duplicated points before applying the transform if sample_pts
                # has less points than npoints_ssv
                npoints_add = npoints_ssv - len(sample_pts)
                idx = np.random.choice(len(sample_pts), npoints_add)
                sample_pts = np.concatenate([sample_pts, sample_pts[idx]])
                sample_feats = np.concatenate([sample_feats, sample_feats[idx]])
                # one hot encoding
                sample_feats = label_binarize(sample_feats, classes=np.arange(len(feat_dc)))
                hc_sub._vertices = sample_pts
                hc_sub._features = sample_feats
                if transform is not None:
                    transform(hc_sub)
                batch[cnt] = hc_sub.vertices
                batch_f[cnt] = hc_sub.features
                cnt += 1
            assert cnt == batchsize
            yield ixs, (batch_f, batch)


def pts_pred_scalar(m, inp, q_out, d_out, q_cnt, device, bs):
    """

    Args:
        m: Model instance.
        inp: Input as given by the loader_func.
        q_out:
        d_out:
        q_cnt:
        device:
        bs:

    Returns:

    """
    ssv_kwargs, model_inp, batch_progress, n_batches = inp
    n_samples = len(model_inp[0])
    res = []
    for ii in range(0, int(np.ceil(n_samples / bs))):
        low = bs * ii
        high = bs * (ii + 1)
        with torch.no_grad():
            g_inp = [torch.from_numpy(i[low:high]).to(device).float() for i in model_inp]
            out = m(*g_inp).cpu().numpy()
        res.append(out)

    del inp
    res = dict(probas=np.concatenate(res), n_batches=n_batches)

    q_cnt.put_nowait(n_samples)
    d_out[ssv_kwargs['ssv_id']].put(res)
    if batch_progress == 1:
        q_out.put_nowait(ssv_kwargs)


def pts_pred_scalar_nopostproc(m, inp, q_out, d_out, q_cnt, device, bs):
    """

    Args:
        m: Model instance.
        inp: Input as given by the loader_func.
        q_out:
        d_out:
        q_cnt:
        device:
        bs:

    Returns:

    """
    ssv_kwargs, model_inp, _, _ = inp
    n_samples = len(model_inp[0])
    res = []
    for ii in range(0, int(np.ceil(n_samples / bs))):
        low = bs * ii
        high = bs * (ii + 1)
        with torch.no_grad():
            g_inp = [torch.from_numpy(i[low:high]).to(device).float() for i in model_inp]
            out = m(*g_inp).cpu().numpy()
        res.append(out)
    del inp
    q_cnt.put_nowait(n_samples)
    q_out.put_nowait(([ssv_kwargs['ssv_id']] * n_samples, res))


def pts_postproc_scalar(ssv_kwargs: dict, d_in: dict, pred_key: Optional[str] = None,
                        da_equals_tan: bool = True) -> Tuple[List[int], List[bool]]:
    """
    Framework is very similar to what will be needed for semantic segmentation of surface points.
    Requires adaptions in pts_loader_semseg and correct merge of vertex indices instead of skeleton node cKDTree.

    Args:
        ssv_kwargs:
        d_in:
        pred_key:
        da_equals_tan: This flag is only applied if the working directory belongs to j0126.

    Returns:

    """
    if pred_key is None:
        pred_key = 'celltype_cnn_e3'
    curr_ix = 0
    sso = SuperSegmentationObject(**ssv_kwargs)
    sso.load_attr_dict()
    celltype_probas = []

    while True:
        try:
            # res: [(dict(t_pts=.., t_label, batch_process)]
            res = d_in[sso.id].get_nowait()
            curr_ix += 1
        except queues.Empty:
            time.sleep(0.25)
            continue
        celltype_probas.append(res['probas'])
        if curr_ix == res['n_batches']:
            break
    logit = np.concatenate(celltype_probas)

    if 'j0126' in sso.config.working_dir and da_equals_tan:
        # accumulate evidence for DA and TAN
        logit[:, 1] += logit[:, 6]
        # remove TAN in proba array
        logit = np.delete(logit, [6], axis=1)
        # INT is now at index 6 -> label 6 is INT

    cls = np.argmax(logit, axis=1).squeeze()
    cls_maj = collections.Counter(cls).most_common(1)[0][0]

    sso.save_attributes([pred_key, f"{pred_key}_probas", f"{pred_key}_certainty"],
                        [cls_maj, logit, certainty_estimate(logit)])

    return [sso.id], [True]


def pts_loader_local_skel(ssv_params: List[dict], out_point_label: Optional[List[Union[str, int]]] = None,
                          batchsize: Optional[int] = None, npoints: Optional[int] = None,
                          ctx_size: Optional[float] = None, transform: Optional[Callable] = None,
                          n_out_pts: int = 100, train=False, base_node_dst: float = 10000,
                          use_ctx_sampling: bool = True, use_syntype: bool = False, use_myelin: bool = False,
                          recalc_skeletons: bool = False,
                          use_subcell: bool = False) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Generator for SSV point cloud samples of size `npoints`. Currently used for
    local point-to-scalar tasks, e.g. morphology embeddings or glia detection.

    Args:
        ssv_params: SuperSegmentationObject kwargs for which samples are generated.
        out_point_label: Either key for sso.skeleton attribute or int (used for all out locations). Currently only
            int is supported!
        batchsize: Only used during training.
        ctx_size: Context size in nm.
        npoints: Number of points used to generate sample context.
        transform: Transformation/agumentation applied to every sample.
        n_out_pts: Maximum number of out points.
        use_subcell: Use points of subcellular structure.
        train: True, train; False, eval mode:
            * train: choose `batchsize` random nodes from the SSV skeleton
              as base points for the context retrieval.
            * eval: return as many batches (size `batchsize`) as there are base nodes in the SSV
              skeleton (distance between nodes see `base_node_dst`).
        base_node_dst: Distance between base nodes for context retrieval during eval mode.
        use_syntype: Use synapse type as point feature.
        use_ctx_sampling: Use context based sampling. If True, uses `ctx_size` (in nm).
        recalc_skeletons: Do not use existing cell skeleton but recalculate it with
            :py:func:`syconn.reps.super_segmentation_object.SuperSegmentationObject.calculate_skeleton`.
        use_myelin: Use myelin point cloud as inpute feature.

    Yields: SSV IDs [M, ], (point location [N, 3], point feature [N, C]), (out_pts [N, 3], out_labels [N, 1])
        If train is False, outpub_labels will be a scalar indicating the current SSV progress, i.e.
        the last batch will have output_label=1.

    """
    if ctx_size is None:
        ctx_size = 20000
    if train and type(out_point_label) == str:
        raise NotImplementedError('Type str is not implemented yet for out_point_label!')
    feat_dc = dict(pts_feat_dict)
    if not use_subcell:
        del feat_dc['mi']
        del feat_dc['vc']
        del feat_dc['syn_ssv']
        del feat_dc['syn_ssv_asym']
        del feat_dc['syn_ssv_sym']
    else:
        if not use_syntype:
            del feat_dc['syn_ssv_asym']
            del feat_dc['syn_ssv_sym']
        else:
            del feat_dc['syn_ssv']
    if not use_myelin:
        del feat_dc['sv_myelin']
    default_kwargs = dict(mesh_caching=False, create=False)
    for curr_ssv_params in ssv_params:
        default_kwargs.update(curr_ssv_params)
        curr_ssv_params = default_kwargs
        # do not write SSV mesh in case it does not exist (will be build from SV meshes)
        ssv = SuperSegmentationObject(**curr_ssv_params)
        loader_kwargs = (ssv, tuple(feat_dc.keys()), tuple(feat_dc.values()), 'glia', None, use_myelin,
                         recalc_skeletons)
        if train:
            hc = _load_ssv_hc_cached(loader_kwargs)
        else:
            hc = _load_ssv_hc(loader_kwargs)
        ssv.clear_cache()
        if train:
            source_nodes = np.random.choice(len(hc.nodes), batchsize, replace=len(hc.nodes) < batchsize)
        else:
            # source_nodes = hc.base_points(threshold=base_node_dst, source=len(hc.nodes) // 2)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(hc.nodes)
            pcd, idcs = pcd.voxel_down_sample_and_trace(
                base_node_dst, pcd.get_min_bound(), pcd.get_max_bound())
            source_nodes = np.max(idcs, axis=1)
            batchsize = min(len(source_nodes), batchsize)
        n_batches = int(np.ceil(len(source_nodes) / batchsize))
        if len(source_nodes) % batchsize != 0:
            source_nodes = np.concatenate([np.random.choice(source_nodes, batchsize - len(source_nodes) % batchsize),
                                           source_nodes])
        for ii in range(n_batches):
            if train and np.random.randint(0, 4) == 0:
                ctx_size_fluct = (np.random.randn(1)[0] * 0.1 + 0.6) * ctx_size
            else:
                ctx_size_fluct = ctx_size
            npoints_ssv = min(len(hc.vertices), npoints)
            # add a +-10% fluctuation in the number of input and output points
            if n_out_pts > 1:  # n_out_pts == 1 for embedding generation
                npoints_add = np.random.randint(-int(n_out_pts * 0.1), int(n_out_pts * 0.1))
                n_out_pts_curr = n_out_pts + npoints_add
            else:
                n_out_pts_curr = n_out_pts
            npoints_add = np.random.randint(-int(npoints_ssv * 0.1), int(npoints_ssv * 0.1))
            npoints_ssv += npoints_add
            batch = np.zeros((batchsize, npoints_ssv, 3))
            batch_f = np.zeros((batchsize, npoints_ssv, len(feat_dc)))
            batch_out = np.zeros((batchsize, n_out_pts_curr, 3))
            if not train:
                batch_out_orig = np.zeros((batchsize, n_out_pts_curr, 3))
            batch_out_l = np.zeros((batchsize, n_out_pts_curr, 1))
            cnt = 0
            for source_node in source_nodes[ii::n_batches]:
                # create local context

                while True:
                    if use_ctx_sampling:
                        node_ids = context_splitting_kdt(hc, source_node, ctx_size_fluct)
                    else:
                        node_ids = bfs_vertices(hc, source_node, npoints_ssv)
                    hc_sub = extract_subset(hc, node_ids)[0]  # only pass HybridCloud
                    sample_feats = hc_sub.features
                    if len(sample_feats) > 0:
                        break
                    source_node = np.random.choice(source_nodes)

                sample_feats = hc_sub.features
                sample_pts = hc_sub.vertices
                # get target locations
                if n_out_pts_curr == 1:
                    out_coords = np.array([hc.nodes[source_node]])
                elif len(hc_sub.nodes) < n_out_pts_curr:
                    # add surface points
                    add_verts = sample_pts[np.random.choice(len(sample_pts), n_out_pts_curr - len(hc_sub.nodes))]
                    out_coords = np.concatenate([hc_sub.nodes, add_verts])
                # down sample to ~500nm apart
                else:
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(hc_sub.nodes)
                    pcd, idcs = pcd.voxel_down_sample_and_trace(500, pcd.get_min_bound(), pcd.get_max_bound())
                    base_points = np.max(idcs, axis=1)
                    base_points = np.random.choice(base_points, n_out_pts_curr,
                                                   replace=len(base_points) < n_out_pts_curr)
                    out_coords = hc_sub.nodes[base_points]
                if train:
                    n_add_verts = min(1, int(n_out_pts_curr * 0.1))
                    add_verts = sample_pts[np.random.choice(len(sample_pts), n_add_verts)]
                    out_coords[np.random.randint(0, n_add_verts)] = add_verts
                # sub-sample vertices
                sample_ixs = np.arange(len(sample_pts))
                np.random.shuffle(sample_ixs)
                sample_pts = sample_pts[sample_ixs][:npoints_ssv]
                sample_feats = sample_feats[sample_ixs][:npoints_ssv]
                # add duplicate points before applying the transform if sample_pts
                # has less points than npoints_ssv
                npoints_add = npoints_ssv - len(sample_pts)
                idx = np.random.choice(len(sample_pts), npoints_add)
                sample_pts = np.concatenate([sample_pts, sample_pts[idx]])
                sample_feats = np.concatenate([sample_feats, sample_feats[idx]])
                # one hot encoding
                sample_feats = label_binarize(sample_feats, classes=np.arange(len(feat_dc)))
                hc_sub._vertices = sample_pts
                hc_sub._features = sample_feats
                hc_sub._nodes = np.array(out_coords)
                # apply augmentations
                if transform is not None:
                    transform(hc_sub)
                batch[cnt] = hc_sub.vertices
                batch_f[cnt] = hc_sub.features
                batch_out[cnt] = hc_sub.nodes
                if not train:
                    batch_out_orig[cnt][:] = out_coords
                batch_out_l[cnt] = out_point_label
                cnt += 1
            assert cnt == batchsize
            if not train:
                batch_progress = ii + 1
                yield curr_ssv_params, (batch_f, batch, batch_out), batch_out_orig, batch_progress, n_batches
            else:
                yield curr_ssv_params, (batch_f, batch), (batch_out, batch_out_l)


def pts_pred_local_skel(m, inp, q_out, d_out, q_cnt, device, bs):
    """

    Args:
        m: Model instance.
        inp: Input as given by the loader_func.
        q_out:
        d_out:
        q_cnt:
        device:
        bs:

    Returns:

    """
    ssv_params, model_inp, out_pts_orig, batch_progress, n_batches = inp
    res = []
    with torch.no_grad():
        for ii in range(0, int(np.ceil(len(model_inp[0]) / bs))):
            low = bs * ii
            high = bs * (ii + 1)
            with torch.no_grad():
                g_inp = [torch.from_numpy(i[low:high]).to(device).float() for i in model_inp]
                out = m(*g_inp).cpu().numpy()
            res.append(out)
    res = dict(t_pts=out_pts_orig, t_l=np.concatenate(res), n_batches=n_batches)

    q_cnt.put_nowait(1. / n_batches)
    d_out[ssv_params['ssv_id']].put(res)
    if batch_progress == 1:
        q_out.put_nowait(ssv_params)


def pts_postproc_glia(ssv_params: dict, d_in: dict, pred_key: str, lo_first_n: Optional[int] = None,
                      partitioned: Optional[bool] = False, apply_softmax: bool = True,
                      sample_loc_ds: float = 100, pred2loc_knn: int = 5) -> Tuple[List[int], List[bool]]:
    curr_ix = 0
    sso = SuperSegmentationObject(**ssv_params)
    node_probas = []
    node_coords = []

    while True:
        try:
            # res: [(dict(t_pts=.., t_label, batch_process)]
            res = d_in[sso.id].get_nowait()
            curr_ix += 1
        except queues.Empty:
            time.sleep(0.25)
            continue
        # el['t_l'] has shape (b, num_points, n_classes) -> (n_nodes, n_classes)
        node_probas.append(res['t_l'].reshape(-1, 2))
        # el['t_pts'] has shape (b, num_points, 3) -> (n_nodes, 3)
        node_coords.append(res['t_pts'].reshape(-1, 3))
        if curr_ix == res['n_batches']:
            break
    node_probas = np.concatenate(node_probas)
    if apply_softmax:
        node_probas = scipy.special.softmax(node_probas, axis=1)
    node_coords = np.concatenate(node_coords)
    kdt = cKDTree(node_coords)
    max_sv = len(sso.svs)
    # only write results for the first N supervoxels (as it was flagged "partitioned", meaning the rest of the
    # supervoxels were added to generate additional context
    if partitioned is not None and lo_first_n is not None and partitioned[sso.id]:
        max_sv = lo_first_n
    for sv in sso.svs[:max_sv]:
        if sv.skeleton_exists:
            coords = sv.skeleton['nodes'] * sv.scaling
        else:
            coords = sv.sample_locations(ds_factor=sample_loc_ds, save=False)
        dists, ixs = kdt.query(coords, k=pred2loc_knn)
        skel_probas = np.ones((len(coords), 2)) * -1
        for ii, nn_dists, nn_ixs in zip(np.arange(len(coords)), dists, ixs):
            nn_ixs = nn_ixs[nn_dists != np.inf]
            probas = node_probas[nn_ixs].squeeze()
            # get mean probability per node
            if len(probas) == 0:
                msg = f'Did not find close-by node predictions in {sso} at {coords[ii]}! {sso.ssv_kwargs}.' \
                      f'\nGot {len(node_probas)} predictions for {len(skel_probas)} sample locations nodes.'
                log_handler.error(msg)
                raise ValueError(msg)
            skel_probas[ii] = np.mean(probas, axis=0)
        # every node has at least one prediction; get mean proba for this super voxel
        sv.enable_locking = True
        sv.save_attributes([pred_key], [skel_probas])
    return [sso.id], [True]


def pts_pred_embedding(m, inp, q_out, d_out, q_cnt, device, bs):
    """
    Uses loader method: :py:func:`~pts_loader_local_skel`.
    Args:
        m: Model instance.
        inp: Input as given by the loader_func.
        q_out:
        d_out:
        q_cnt:
        device:
        bs:

    Returns:

    """
    ssv_params, model_inp, out_pts_orig, batch_progress, n_batches = inp
    # ignore target points, not needed for the representation network (e.g. ModelNet40) which is pts2scalar
    model_inp = model_inp[:2]
    res = []
    with torch.no_grad():
        for ii in range(0, int(np.ceil(len(model_inp[0]) / bs))):
            low = bs * ii
            high = bs * (ii + 1)
            with torch.no_grad():
                g_inp = [torch.from_numpy(i[low:high]).to(device).float() for i in model_inp]
                out = m(g_inp, None, None).cpu().numpy()
            res.append(out)
    res = dict(t_pts=out_pts_orig, t_l=np.concatenate(res), n_batches=n_batches)

    q_cnt.put_nowait(1. / n_batches)
    d_out[ssv_params['ssv_id']].put(res)
    if batch_progress == 1:
        q_out.put(ssv_params)


def pts_postproc_embedding(ssv_params: dict, d_in: dict, pred_key: Optional[str] = None
                           ) -> Tuple[List[int], List[bool]]:
    curr_ix = 0
    sso = SuperSegmentationObject(**ssv_params)
    node_embedding = []
    node_coords = []
    while True:
        # res: [(dict(t_pts=.., t_label, batch_process)]
        try:
            res = d_in[sso.id].get_nowait()
            curr_ix += 1
        except queues.Empty:
            time.sleep(0.5)
            continue
        # el['t_l'] has shape (b, num_points, n_latent_dim) -> (n_nodes, n_latent_dim)
        node_embedding.append(res['t_l'].reshape(-1, res['t_l'].shape[-1]))
        # el['t_pts'] has shape (b, num_points, 3) -> (n_nodes, 3)
        node_coords.append(res['t_pts'].reshape(-1, 3))
        if curr_ix == res['n_batches']:
            break

    node_embedding = np.concatenate(node_embedding)
    node_coords = np.concatenate(node_coords)

    # map inference sites of latent vecs to skeleton node locations via nearest neighbor
    # TODO: perform interpolation?
    sso.load_skeleton()
    hull_tree = spatial.cKDTree(node_coords)
    dists, ixs = hull_tree.query(sso.skeleton["nodes"] * sso.scaling, n_jobs=sso.nb_cpus, k=1)
    sso.skeleton[pred_key] = node_embedding[ixs]
    sso.save_skeleton()
    return [sso.id], [True]


def pts_loader_semseg_train(fnames_pkl: Iterable[str], batchsize: int,
                            npoints: int, ctx_size: float,
                            transform: Optional[Callable] = None,
                            use_subcell: bool = False, mask_borders_with_id: Optional[int] = None
                            ) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Generator for SSV point cloud samples of size `npoints`. Currently used for
    semantic segmentation tasks, e.g. spine, bouton and functional compartment
    prediction.

    Args:
        fnames_pkl:
        batchsize:
        npoints:
        ctx_size:
        transform:
        use_subcell:
        mask_borders_with_id:

    Yields: SSV IDs [M, ], (point feature [N, C], point location [N, 3])

    """
    feat_dc = dict(pts_feat_dict)
    del feat_dc['syn_ssv_asym']
    del feat_dc['syn_ssv_sym']
    del feat_dc['sv_myelin']
    if not use_subcell:
        del feat_dc['mi']
        del feat_dc['vc']
        del feat_dc['syn_ssv']

    if np.random.randint(0, 4) == 0:
        ctx_size_fluct = max((np.random.randn(1)[0] * 0.1 + 0.7), 0.33) * ctx_size
    else:
        ctx_size_fluct = ctx_size

    for pkl_f in fnames_pkl:
        hc = load_hc_pkl(pkl_f, 'compartment')
        npoints_ssv = min(len(hc.vertices), npoints)
        # filter valid skeleton nodes (i.e. which were close to manually annotated nodes)
        source_nodes = np.where(hc.node_labels == 1)[0]
        source_nodes = np.random.choice(len(source_nodes), batchsize,
                                        replace=len(source_nodes) < batchsize)
        n_batches = int(np.ceil(len(source_nodes) / batchsize))
        if len(source_nodes) % batchsize != 0:
            source_nodes = np.random.choice(source_nodes, batchsize * n_batches)
        for ii in range(n_batches):
            # cell/sv vertices proportional to total npoints
            n_out_pts = int(np.sum(hc.features == 0) / len(hc.vertices) * npoints_ssv)
            # add a +-10% fluctuation in the number of input and output points
            npoints_add = np.random.randint(-int(n_out_pts * 0.1), int(n_out_pts * 0.1))
            n_out_pts_curr = n_out_pts + npoints_add
            npoints_add = np.random.randint(-int(npoints_ssv * 0.1), int(npoints_ssv * 0.1))
            npoints_ssv += npoints_add
            batch = np.zeros((batchsize, n_out_pts_curr, 3))
            batch_f = np.ones((batchsize, n_out_pts_curr, len(feat_dc)))
            # batch_out = np.zeros((batchsize, n_out_pts_curr, 3))
            batch_out_l = np.zeros((batchsize, n_out_pts_curr, 1))
            cnt = 0
            for source_node in source_nodes[ii::n_batches]:
                # create local context
                # node_ids = bfs_vertices(hc, source_node, npoints_ssv)
                while True:
                    node_ids = context_splitting_graph_many(hc, [source_node], ctx_size_fluct)[0]
                    hc_sub = extract_subset(hc, node_ids)[0]  # only pass HybridCloud
                    # if mask_borders_with_id is not None:
                    #     source_node_c = hc_sub.nodes[hc_sub.relabel_dc[source_node]]
                    #     boarder_vert_mask = np.linalg.norm(hc_sub.vertices - source_node_c, axis=1) > \
                    #                         ctx_size_fluct * 0.8
                    #     hc_sub._labels[boarder_vert_mask] = mask_borders_with_id
                    sample_feats = hc_sub.features
                    if len(sample_feats) > 0:
                        break
                    source_node = np.random.choice(source_nodes)
                sample_pts = hc_sub.vertices
                sample_labels = hc_sub.labels

                # sub-sample vertices
                sample_ixs = np.arange(len(sample_pts))
                np.random.shuffle(sample_ixs)
                sample_pts = sample_pts[sample_ixs][:n_out_pts_curr]
                sample_feats = sample_feats[sample_ixs][:n_out_pts_curr]
                sample_labels = sample_labels[sample_ixs][:n_out_pts_curr]
                # add duplicate points before applying the transform if sample_pts
                # has less points than n_out_pts_curr
                npoints_add = n_out_pts_curr - len(sample_pts)
                if npoints_add > 0:
                    idx = np.random.choice(len(sample_pts), npoints_add)
                    sample_pts = np.concatenate([sample_pts, sample_pts[idx]])
                    sample_feats = np.concatenate([sample_feats, sample_feats[idx]])
                    sample_labels = np.concatenate([sample_labels, sample_labels[idx]])

                hc_sub._vertices = sample_pts
                hc_sub._features = sample_feats
                hc_sub._labels = sample_labels
                # apply augmentations
                if transform is not None:
                    transform(hc_sub)
                batch[cnt] = hc_sub.vertices
                # one hot encoding
                if use_subcell:
                    batch_f[cnt] = label_binarize(hc_sub.features, classes=np.arange(len(feat_dc)))
                # get target locations
                # TODO: Add masking if beneficial - for now just use all input points and their labels
                # out_pts_mask = (hc_sub.features == 0).squeeze()
                # n_out_pts_actual = np.sum(out_pts_mask)
                # idx = np.random.choice(n_out_pts_actual, n_out_pts_curr,
                #                        replace=n_out_pts_actual < n_out_pts_curr)
                # batch_out[cnt] = hc_sub.vertices[out_pts_mask][idx]
                # TODO: currently only supports type(out_point_label) = int
                # batch_out_l[cnt] = hc_sub.labels[out_pts_mask][idx]
                batch_out_l[cnt] = hc_sub.labels
                if -1 in batch_out_l[cnt]:
                    assert mask_borders_with_id is not None
                    batch_out_l[cnt][batch_out_l[cnt] == -1] = mask_borders_with_id
                cnt += 1
            assert cnt == batchsize
            # TODO: Add masking if beneficial - for now just use all input points and their labels
            # yield (batch_f, batch), (batch_out, batch_out_l)
            yield batch_f, batch, batch_out_l


def pts_loader_semseg(ssv_params: Optional[List[Tuple[int, dict]]] = None,
                      out_point_label: Optional[List[Union[str, int]]] = None,
                      batchsize: Optional[int] = None, npoints: Optional[int] = None,
                      ctx_size: Optional[float] = None, transform: Optional[Callable] = None,
                      n_out_pts: int = 100, train=False, base_node_dst: float = 10000, use_subcell: bool = True,
                      ssd_kwargs: Optional[dict] = None) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Generator for SSV point cloud samples of size `npoints`. Currently used for
    local point-to-scalar tasks, e.g. morphology embeddings or glia detection.

    Args:
        ssv_params: SuperSegmentationObject IDs and SSD kwargs for which samples are generated.
        out_point_label: Either key for sso.skeleton attribute or int (used for all out locations).
        batchsize: Only used during training.
        ctx_size:
        npoints: Number of points used to generate sample context.
        transform: Transformation/agumentation applied to every sample.
        n_out_pts: Maximum number of out points.
        use_subcell: Use points of subcellular structure.
        train: True, train; False, eval mode:
            * train: choose `batchsize` random nodes from the SSV skeleton
              as base points for the context retrieval.
            * eval: return as many batches (size `batchsize`) as there are base nodes in the SSV
              skeleton (distance between nodes see `base_node_dst`).
        base_node_dst: Distance between base nodes for context retrieval during eval mode.
        ssd_kwargs:

    Yields: SSV IDs [M, ], (point location [N, 3], point feature [N, C]), (out_pts [N, 3], out_labels [N, 1])
        If train is False, outpub_labels will be a scalar indicating the current SSV progress, i.e.
        the last batch will have output_label=1.

    """
    if ctx_size is None:
        ctx_size = 20000
    # TODO: support node attributes in hybrid cloud graph also
    if type(out_point_label) == str:
        raise NotImplementedError
    feat_dc = dict(pts_feat_dict)
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
    if train and np.random.randint(0, 4) == 0:
        ctx_size_fluct = (np.random.randn(1)[0] * 0.1 + 0.7) * ctx_size
    else:
        ctx_size_fluct = ctx_size
    for curr_ssv_params in ssv_params:
        # do not write SSV mesh in case it does not exist (will be build from SV meshes)
        ssv = SuperSegmentationObject(mesh_caching=False, **curr_ssv_params)
        if train:
            hc = _load_ssv_hc_cached((ssv, tuple(feat_dc.keys()), tuple(feat_dc.values()), 'compartment', None))
        else:
            hc = _load_ssv_hc((ssv, tuple(feat_dc.keys()), tuple(feat_dc.values()), 'compartment', None))
        ssv.clear_cache()
        npoints_ssv = min(len(hc.vertices), npoints)
        # add a +-10% fluctuation in the number of input and output points
        npoints_add = np.random.randint(-int(n_out_pts * 0.1), int(n_out_pts * 0.1))
        n_out_pts_curr = n_out_pts + npoints_add
        npoints_add = np.random.randint(-int(npoints_ssv * 0.1), int(npoints_ssv * 0.1))
        npoints_ssv += npoints_add
        if train:
            source_nodes = np.random.choice(len(hc.nodes), batchsize, replace=len(hc.nodes) < batchsize)
        else:
            # source_nodes = hc.base_points(threshold=base_node_dst, source=len(hc.nodes) // 2)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(hc.nodes)
            pcd, idcs = pcd.voxel_down_sample_and_trace(
                base_node_dst, pcd.get_min_bound(), pcd.get_max_bound())
            source_nodes = np.max(idcs, axis=1)
            batchsize = min(len(source_nodes), batchsize)
        n_batches = int(np.ceil(len(source_nodes) / batchsize))
        if len(source_nodes) % batchsize != 0:
            source_nodes = np.concatenate(
                [np.random.choice(source_nodes, batchsize - len(source_nodes) % batchsize), source_nodes])
        for ii in range(n_batches):
            batch = np.zeros((batchsize, npoints_ssv, 3))
            batch_f = np.zeros((batchsize, npoints_ssv, len(feat_dc)))
            batch_out = np.zeros((batchsize, n_out_pts_curr, 3))
            # TODO: add vertex indices for quick merge at the end
            if not train:
                batch_out_orig = np.zeros((batchsize, n_out_pts_curr, 3))
            batch_out_l = np.zeros((batchsize, n_out_pts_curr, 1))
            cnt = 0
            for source_node in source_nodes[ii::n_batches]:
                # create local context
                node_ids = context_splitting_kdt(hc, source_node, ctx_size_fluct)
                hc_sub = extract_subset(hc, node_ids)[0]  # only pass HybridCloud
                sample_feats = hc_sub.features
                sample_pts = hc_sub.vertices
                # get target locations ~1um apart
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(hc_sub.nodes)
                pcd, idcs = pcd.voxel_down_sample_and_trace(
                    1000, pcd.get_min_bound(), pcd.get_max_bound())
                base_points = np.max(idcs, axis=1)
                base_points = np.random.choice(base_points, n_out_pts_curr,
                                               replace=len(base_points) < n_out_pts_curr)
                out_coords = hc_sub.nodes[base_points]
                # sub-sample vertices
                sample_ixs = np.arange(len(sample_pts))
                np.random.shuffle(sample_ixs)
                sample_pts = sample_pts[sample_ixs][:npoints_ssv]
                sample_feats = sample_feats[sample_ixs][:npoints_ssv]
                # add duplicate points before applying the transform if sample_pts
                # has less points than npoints_ssv
                npoints_add = npoints_ssv - len(sample_pts)
                idx = np.random.choice(len(sample_pts), npoints_add)
                sample_pts = np.concatenate([sample_pts, sample_pts[idx]])
                sample_feats = np.concatenate([sample_feats, sample_feats[idx]])
                # TODO: batch_out needs to be adapted, also add batch_out_ixs to store the vertex indices for quick
                #  merge at the end
                # one hot encoding
                sample_feats = label_binarize(sample_feats, classes=np.arange(len(feat_dc)))
                hc_sub._vertices = sample_pts
                hc_sub._features = sample_feats
                # apply augmentations
                if transform is not None:
                    transform(hc_sub)
                batch[cnt] = hc_sub.vertices
                batch_f[cnt] = hc_sub.features
                batch_out[cnt] = hc_sub.nodes
                if not train:
                    batch_out_orig[cnt] = out_coords
                # TODO: currently only supports type(out_point_label) = int
                batch_out_l[cnt] = out_point_label
                cnt += 1
            assert cnt == batchsize
            if not train:
                batch_progress = ii + 1
                yield curr_ssv_params, (batch_f, batch, batch_out), batch_out_orig, batch_progress, n_batches
            else:
                yield curr_ssv_params, (batch_f, batch), (batch_out, batch_out_l)


def pts_pred_semseg(m, inp, q_out, d_out, q_cnt, device, bs):
    """

    Args:
        m: Model instance.
        inp: Input as given by the loader_func.
        q_out:
        d_out:
        q_cnt:
        device:
        bs:

    Returns:

    """
    # TODO: is it possible to get 'device' directly from model 'm'?
    ssv_params, model_inp, out_pts_orig, batch_progress, n_batches = inp
    res = []
    with torch.no_grad():
        for ii in range(0, int(np.ceil(len(model_inp[0]) / bs))):
            low = bs * ii
            high = bs * (ii + 1)
            with torch.no_grad():
                g_inp = [torch.from_numpy(i[low:high]).to(device).float() for i in model_inp]
                out = m(*g_inp).cpu().numpy()
            res.append(out)
    res = dict(t_pts=out_pts_orig, t_l=np.concatenate(res), n_batches=n_batches)

    q_cnt.put_nowait(1. / n_batches)
    d_out[ssv_params['ssv_id']].put(res)
    if batch_progress == 1:
        q_out.put_nowait(ssv_params)


def pts_postproc_semseg(ssv_id: int, d_in: dict, working_dir: Optional[str] = None,
                        version: Optional[str] = None) -> Tuple[List[int], List[bool]]:
    """
    Framework is very similar to what will be needed for semantic segmentation of surface points.
    Requires adaptions in pts_loader_semseg and correct merge of vertex indices instead of skeleton node cKDTree.

    Args:
        ssv_id:
        d_in:
        working_dir:
        version:

    Returns:

    """
    curr_ix = 0
    sso = SuperSegmentationObject(ssv_id=ssv_id, working_dir=working_dir, version=version)
    sso.load_skeleton()
    skel = sso.skeleton
    node_preds = []
    node_coords = []

    while True:
        try:
            # res: [(dict(t_pts=.., t_label, batch_process)]
            res = d_in[sso.id].get_nowait()
            curr_ix += 1
        except queues.Empty:
            time.sleep(0.25)
            continue
        # el['t_l'] has shape (b, num_points, n_classes) -> (n_nodes, 1)
        node_preds.append(np.argmax(res['t_l'].reshape(-1, 2), axis=1)[..., None])
        # el['t_pts'] has shape (b, num_points, 3) -> (n_nodes, 3)
        node_coords.append(res['t_pts'].reshape(-1, 3))
        if curr_ix == res['n_batches']:
            break
    node_preds = np.concatenate(node_preds)
    node_coords = np.concatenate(node_coords)
    kdt = cKDTree(node_coords)
    dists, ixs = kdt.query(skel['nodes'] * sso.scaling, k=10, distance_upper_bound=2000)
    node_pred = np.ones(len(skel['nodes'])) * -1
    for ii, nn_dists, nn_ixs in zip(np.arange(len(skel['nodes'])), dists, ixs):
        nn_ixs = nn_ixs[nn_dists != np.inf]
        preds = node_preds[nn_ixs].squeeze()
        node_pred[ii] = collections.Counter(preds).most_common(1)[0][0]
    # every node has at least one prediction
    assert np.sum(node_pred == -1) == 0, "Unpredicted skeleton node."
    return [ssv_id], [True]


@functools.lru_cache(maxsize=128)
def load_hc_pkl(path: str, gt_type: str, radius: Optional[float] = None) -> HybridCloud:
    """
    TODO: move pts_feat_dict and pts_feat_ds_dict to config.

    Load HybridCloud from pkl file (cached via functools.lur_cache).
    The following properties must be met:

    * Vertex features are labeled according to ``pts_feat_dict`` in
      handler.prediction_pts.
    * Skeleton nodes require to have labels (0, 1) indicating whether they can
      be used as source node for the context generation (1) or not (0).

    Down sampling will be performed via open3d's ``voxel_down_sample`` with
    voxel sizes defined in pts_feat_ds_dict (handler.prediction_pts)

    Args:
        path: Path to HybridCloud pickle file.
        gt_type: See pts_feat_ds_dict in handler.prediction_pts.
        radius: Add additional edges between skeleton nodes within `radius`.

    Returns:
        Populated HybridCloud.
    """
    feat_ds_dict = dict(pts_feat_ds_dict[gt_type])
    # requires cloud keys to be in [vc, syn_ssv, mi, hybrid]
    feat_ds_dict['hybrid'] = feat_ds_dict['sv']
    hc = HybridCloud()
    hc.load_from_pkl(path)
    new_verts = []
    new_labels = []
    new_feats = []
    for ident_str, feat_id in pts_feat_dict.items():
        pcd = o3d.geometry.PointCloud()
        m = (hc.features == feat_id).squeeze()
        if np.sum(m) == 0:
            continue
        verts = hc.vertices[m]
        labels = hc.labels[m]
        feats = hc.features[m]
        pcd.points = o3d.utility.Vector3dVector(verts)
        pcd, idcs = pcd.voxel_down_sample_and_trace(
            pts_feat_ds_dict[gt_type][ident_str], pcd.get_min_bound(),
            pcd.get_max_bound())
        idcs = np.max(idcs, axis=1)
        new_verts.append(np.asarray(pcd.points))
        new_labels.append(labels[idcs])
        new_feats.append(feats[idcs])
    hc._vertices = np.concatenate(new_verts)
    hc._labels = np.concatenate(new_labels)
    hc._features = np.concatenate(new_feats)
    # reset verts2node mapping and cache it
    hc._verts2node = None
    _ = hc.verts2node
    if radius is not None:
        # add edges within radius
        kdt = spatial.cKDTree(hc.nodes)
        pairs = list(kdt.query_pairs(radius))
        # remap to subset of indices
        hc._edges = np.concatenate([hc._edges, pairs])
    return hc


# factory methods for models
def get_pt_kwargs(mdir: str) -> Tuple[dict, dict]:
    use_norm = False
    track_running_stats = False
    activation = 'relu'
    use_bias = True
    ctx = int(re.findall(r'_ctx(\d+)_', mdir)[0])
    if 'swish' in mdir:
        activation = 'swish'
    if '_noBN_' in mdir:
        use_norm = False
    if '_gn_' in mdir:
        use_norm = 'gn'
    elif '_bn_' in mdir:
        use_norm = 'bn'
        if 'trackRunStats' in mdir:
            track_running_stats = True
    if 'noBias' in mdir:
        use_bias = False
    npoints = int(re.findall(r'_nb(\d+)_', mdir)[0])
    scale_fact = int(re.findall(r'_scale(\d+)_', mdir)[0])
    mkwargs = dict(use_norm=use_norm, track_running_stats=track_running_stats, act=activation, use_bias=use_bias)
    loader_kwargs = dict(ctx_size=ctx, scale_fact=scale_fact, npoints=npoints)
    return mkwargs, loader_kwargs


def get_glia_model_pts(mpath: Optional[str] = None, device: str = 'cuda') -> 'InferenceModel':
    if mpath is None:
        mpath = global_params.config.mpath_glia_pts
    from elektronn3.models.convpoint import SegSmall
    mkwargs, loader_kwargs = get_pt_kwargs(mpath)
    try:
        m = SegSmall(1, 2, **mkwargs).to(device)
        m.load_state_dict(torch.load(mpath)['model_state_dict'])
    except RuntimeError:
        mkwargs['use_bias'] = False
        m = SegSmall(1, 2, **mkwargs).to(device)
        m.load_state_dict(torch.load(mpath)['model_state_dict'])
    m.loader_kwargs = loader_kwargs
    return m.eval()


def get_compartment_model_pts(mpath: Optional[str] = None, device='cuda') -> 'InferenceModel':
    if mpath is None:
        mpath = global_params.config.mpath_comp_pts
    from elektronn3.models.convpoint import SegSmall2
    mkwargs, loader_kwargs = get_pt_kwargs(mpath)
    m = SegSmall2(5, 7, **mkwargs).to(device)
    m.load_state_dict(torch.load(mpath)['model_state_dict'])
    m.loader_kwargs = loader_kwargs
    return m.eval()


def get_celltype_model_pts(mpath: Optional[str] = None, device='cuda') -> 'InferenceModel':
    if mpath is None:
        mpath = global_params.config.mpath_celltype_pts
    from elektronn3.models.convpoint import ModelNet40
    mkwargs, loader_kwargs = get_pt_kwargs(mpath)
    n_classes = 8
    n_inputs = 5
    if 'j0251' in mpath:
        n_classes = 11
    if 'myelin' in mpath:
        n_inputs += 1
    if '_noSyntype' in mpath:
        n_inputs -= 1
    if '_cellshapeOnly' in mpath:
        n_inputs = 1
    try:
        m = ModelNet40(n_inputs, n_classes, **mkwargs).to(device)
    except RuntimeError as e:
        if not mkwargs['use_bias']:
            mkwargs['use_bias'] = True
        else:
            raise RuntimeError(e)
        m = ModelNet40(n_inputs, n_classes, **mkwargs).to(device)
    m.load_state_dict(torch.load(mpath)['model_state_dict'])
    m.loader_kwargs = loader_kwargs
    return m.eval()


def get_tnet_model_pts(mpath: Optional[str] = None, device='cuda') -> 'InferenceModel':
    if mpath is None:
        mpath = global_params.config.mpath_tnet_pts
    from elektronn3.models.convpoint import ModelNet40, TripletNet
    mkwargs, loader_kwargs = get_pt_kwargs(mpath)
    if 'myelin' in mpath:
        inp_dim = 6
    else:
        inp_dim = 5
    m = TripletNet(ModelNet40(inp_dim, 10, **mkwargs).to(device))
    m.load_state_dict(torch.load(mpath)['model_state_dict'])
    m.loader_kwargs = loader_kwargs
    return m.eval()


# prediction wrapper
def predict_glia_ssv(ssv_params: List[dict], mpath: Optional[str] = None,
                     postproc_kwargs: Optional[dict] = None, show_progress: bool = True, **add_kwargs):
    """
    Perform glia predictions of cell reconstructions on sampled point sets from the
    cell's vertices. The number of predictions ``npreds`` per cell is calculated based on the
    fraction of the total number of vertices over ``npoints`` times two, but at least 5.
    Every point set is constructed by collecting the vertices associated with skeleton within a
    breadth-first search up to a maximum of ``npoints``.

    Notes:
        * :py:func:`~pts_postproc_glia` currently requires locking.

    Args:
        ssv_params: List of kwargs to initialize SSVs.
        mpath: Path to model.
        postproc_kwargs: Postprocessing kwargs.
        show_progress: Show progress bar.

    Returns:

    """
    if mpath is None:
        mpath = global_params.config.mpath_glia_pts
    loader_kwargs = get_pt_kwargs(mpath)[1]
    default_kwargs = dict(nloader=8, npredictor=4, npostproc=2, bs=10,
                          loader_kwargs=dict(n_out_pts=200, base_node_dst=loader_kwargs['ctx_size'] / 3,
                                             recalc_skeletons=True))
    default_kwargs.update(add_kwargs)
    postproc_kwargs_def = global_params.config['points']['glia']['mapping']
    if postproc_kwargs is None:
        postproc_kwargs = {}
    postproc_kwargs_def.update(postproc_kwargs)
    out_dc = predict_pts_plain(ssv_params, get_glia_model_pts, pts_loader_local_skel, pts_pred_local_skel,
                               postproc_func=pts_postproc_glia, mpath=mpath, postproc_kwargs=postproc_kwargs_def,
                               show_progress=show_progress, **loader_kwargs, **default_kwargs)
    if not np.all(list(out_dc.values())) or len(out_dc) != len(ssv_params):
        raise ValueError('Invalid output during glia prediction.')


def infere_cell_morphology_ssd(ssv_params, mpath: Optional[str] = None, pred_key_appendix: str = '', **add_kwargs):
    """
    Extract local morphology embeddings of cell reconstructions on sampled point sets from the
    cell's vertices. The number of predictions ``npreds`` per cell is calculated based on the
    fraction of the total number of vertices over ``npoints`` times two, but at least 5.
    Every point set is constructed by collecting the vertices associated with skeleton within a
    breadth-first search up to a maximum of ``npoints``.
    Result is stored with key 'latent_morph' (+ `pred_key_appendix`) in the SSV skeleton.


    Args:
        ssv_params:
        mpath:
        pred_key_appendix:

    Returns:

    """
    pred_key = "latent_morph"
    pred_key += pred_key_appendix
    if mpath is None:
        mpath = global_params.config.mpath_tnet_pts
    loader_kwargs = get_pt_kwargs(mpath)[1]
    if 'myelin' in mpath:
        use_myelin = True
    else:
        use_myelin = False
    default_kwargs = dict(nloader=8, npredictor=4, npostproc=2, bs=10, loader_kwargs=dict(
        n_out_pts=1, base_node_dst=loader_kwargs['ctx_size'] / 2, use_syntype=True, use_subcell=True,
        use_myelin=use_myelin))
    postproc_kwargs = dict(pred_key=pred_key)
    default_kwargs.update(add_kwargs)
    out_dc = predict_pts_plain(ssv_params, get_tnet_model_pts, pts_loader_local_skel, pts_pred_embedding,
                               postproc_kwargs=postproc_kwargs, postproc_func=pts_postproc_embedding,
                               show_progress=False, mpath=mpath, **loader_kwargs, **default_kwargs)
    if not np.all(list(out_dc.values())) or len(out_dc) != len(ssv_params):
        raise ValueError('Invalid output during glia prediction.')


def predict_celltype_ssd(ssd_kwargs, mpath: Optional[str] = None, ssv_ids: Optional[Iterable[int]] = None,
                         da_equals_tan: bool = True, pred_key: Optional[str] = None,
                         show_progress: bool = True, **add_kwargs):
    """
    Perform cell type predictions of cell reconstructions on sampled point sets from the
    cell's vertices. The number of predictions ``npreds`` per cell is calculated based on the
    fraction of the total number of vertices over ``npoints`` times two, but at least 5.
    Every point set is constructed by collecting the vertices associated with skeleton within a
    breadth-first search up to a maximum of ``npoints``.


    Args:
        ssd_kwargs:
        mpath:
        ssv_ids:
        da_equals_tan: Only relevant for j0126.
        pred_key: Key used to store predictions in `attr_dict` of cell SSOs.
        show_progress: Show progress bar.

    Returns:

    """
    if pred_key is None:
        pred_key = 'celltype_cnn_e3'
    if mpath is None:
        mpath = global_params.config.mpath_celltype_pts
    loader_kwargs = get_pt_kwargs(mpath)[1]
    cellshape_only = False
    use_syntype = True
    if 'myelin' in mpath:
        map_myelin = True
    else:
        map_myelin = False
    if '_noSyntype' in mpath:
        use_syntype = False
    if '_cellshapeOnly' in mpath:
        cellshape_only = True
    default_kwargs = dict(nloader=8, npredictor=4, bs=10, npostproc=2,
                          loader_kwargs=dict(redundancy=20, map_myelin=map_myelin, use_syntype=use_syntype,
                                             cellshape_only=cellshape_only),
                          postproc_kwargs=dict(pred_key=pred_key, da_equals_tan=da_equals_tan))
    default_kwargs.update(add_kwargs)
    ssd = SuperSegmentationDataset(**ssd_kwargs)
    if ssv_ids is None:
        ssv_ids = ssd.ssv_ids
    out_dc = predict_pts_plain(ssd_kwargs, get_celltype_model_pts, pts_loader_scalar_infer, pts_pred_scalar,
                               postproc_func=pts_postproc_scalar, mpath=mpath, ssv_ids=ssv_ids,
                               show_progress=show_progress, **loader_kwargs, **default_kwargs)
    if not np.all(list(out_dc.values())) or len(out_dc) != len(ssv_ids):
        raise ValueError('Invalid output during cell type prediction.')


# -------------------------------------------- COMPARTMENT PREDICTION ---------------------------------------------#


def predict_cmpt_ssd(ssd_kwargs, mpath: Optional[str] = None, ssv_ids: Optional[Iterable[int]] = None,
                     ctx_dst_fac: Optional[int] = None, show_progress: bool = True, **add_kwargs):
    """
    Performs compartment predictions on the ssv's given with ``ssv_ids``, based on the dataset initialized with
    ``ssd_kwargs``. The kwargs for predict_pts_plain are organized as dicts with the respective values, keyed
    by the pred_type which is inferred from the models at mpath. This enables the pred worker to apply multiple
    different models at once. E.g. when mpath contains models with identifiers 'ads', 'abt' and 'dnh', ctx_size would
    e.g. be {'ads': 20000, 'abt': 3000, 'dnh': 3000}.

    Args:
         ssd_kwargs: Keyword arguments which specify the ssd in use.
         mpath: Path to model folder (which contains models with model identifier) or to single model file.
         ssv_ids: Ids of ssv objects which should get processed.
         ctx_dst_fac: Defines the redundancy of the predictions by determining the distance of the base nodes
            used for context extraction. Higher ``ctx_dst_fac`` means smaller distance (ctx / ctx_dst_fac) and
            therefore larger context overlap and longer processing time.
         add_kwargs: Can for example contain parameter ``bs`` for batchsize. ``bs`` is supposed to be a factor
            which gets multiplied with the model dependent batch sizes.
        show_progress: Show progress bar.
    """
    if mpath is None:
        mpath = global_params.config.mpath_compartment_pts
    mpath = os.path.expanduser(mpath)
    if os.path.isdir(mpath):
        # multiple models
        mpaths = glob.glob(mpath + '*.pth')
    else:
        # single model
        mpaths = [mpath]
    # These variables are needed in predict_pts_plain
    ctx_size = defaultdict(list)
    batchsizes = {}
    npoints = {}
    scale_fact = {}
    # find model with each identifier and pack all the loader parameters into the dicts which are then handed
    # over to predict_pts_plain
    pred_types = []
    for path in mpaths:
        kwargs = get_cmpt_kwargs(path)[1]
        # infer pred_types and all other parameters from models at m_path
        p_t = kwargs['pred_type']
        if p_t in pred_types:
            raise ValueError(f"Found multiple models for prediction type {p_t}.")
        pred_types.append(p_t)
        ctx_size[kwargs['ctx_size']].append(p_t)
        batchsizes[kwargs['ctx_size']] = kwargs['bs']
        npoints[p_t] = kwargs['npoints']
        scale_fact[p_t] = kwargs['scale_fact']
    kwargs = dict(ctx_size=ctx_size, npoints=npoints, scale_fact=scale_fact)
    if ctx_dst_fac is None:
        ctx_dst_fac = 2
    loader_kwargs = dict(pred_types=pred_types, ctx_dst_fac=ctx_dst_fac)
    ssd = SuperSegmentationDataset(**ssd_kwargs)
    if ssv_ids is None:
        ssv_ids = ssd.ssv_ids
    ssd_kwargs = [{'ssv_id': ssv_id, 'working_dir': ssd_kwargs['working_dir']} for ssv_id in ssv_ids]
    default_kwargs = dict(nloader=6, npredictor=4, npostproc=4, bs=batchsizes)
    if 'bs' in add_kwargs and type(add_kwargs['bs']) == dict:
        raise ValueError('Non default batch size is meant to be a factor which is multiplied with the model'
                         ' dependent batch sizes.')
    default_kwargs.update(add_kwargs)
    if 'bs' in add_kwargs:
        for ctx in batchsizes:
            batchsizes[ctx] = int(batchsizes[ctx]*default_kwargs['bs'])
        default_kwargs['bs'] = batchsizes
    out_dc = predict_pts_plain(ssd_kwargs,
                               model_loader=get_cpmt_model_pts,
                               loader_func=pts_loader_cpmt,
                               pred_func=pts_pred_cmpt,
                               postproc_func=pts_postproc_cpmt,
                               mpath=mpath,
                               loader_kwargs=loader_kwargs,
                               model_loader_kwargs=dict(pred_types=pred_types),
                               show_progress=show_progress,
                               **default_kwargs,
                               **kwargs)
    if not np.all(list(out_dc.values())) or len(out_dc) != len(ssv_ids):
        raise ValueError('Invalid output during compartment prediction.')


def get_cpmt_model_pts(mpath: Optional[str] = None, device='cuda', pred_types: Optional[List] = None):
    """ Loads multiple models (or only one), depending on ``pred_types``. Models which should be used
        must contain one of the pred_types in their names. If ``mpath`` points to a single model, this
        model must contain 'cmpt' in its name.

    Args:
        mpath: Path to model folder (containing multiple models) or to single model file. Models should
            have one of the pred_types in their names, a single model must contain 'cmpt' in its name.
        device: Device onto which the models should get transfered.
        pred_types: List of prediction types, e.g. ['ads', 'abt', dnh'] for axon, dendrite, soma; ...
    """
    if mpath is None:
        mpath = global_params.config.mpath_compartment_pts
    mpath = os.path.expanduser(mpath)
    if os.path.isdir(mpath):
        # multiple models
        mpaths = glob.glob(mpath + '*.pth')
    else:
        # single model, must contain 'cmpt' in its name
        mpaths = [mpath]
    if pred_types is None:
        pred_types = ['cmpt']
    else:
        pred_types = pred_types
    models = {}
    from elektronn3.models.convpoint import SegBig
    # for each pred_type, check if there is a corresponding model and load it
    for p_t in pred_types:
        for path in mpaths:
            if p_t in path:
                mkwargs = get_cmpt_kwargs(path)[0]
                if p_t in models:
                    raise ValueError(f"Found multiple models for prediction type {p_t}.")
                # TODO: Remove hardcoding of model parameters
                if p_t == 'dnh' or p_t == 'abt':
                    m = SegBig(**mkwargs, reductions=[1024, 512, 256, 64, 16, 8],
                               neighbor_nums=[32, 32, 32, 16, 8, 8, 4, 8, 8, 8, 16, 16, 16]).to(device)
                else:
                    m = SegBig(**mkwargs).to(device)
                m.load_state_dict(torch.load(path)['model_state_dict'])
                # save models in dict, e.g. {'ads': ads-m, 'abt': abt-m, 'dnh': dnh-m}
                models[p_t] = m
    return models


def pts_loader_cpmt(ssv_params, pred_types: List[str], batchsize: dict, npoints: dict, ctx_size: dict, transform: dict,
                    ctx_dst_fac: int, use_subcell: bool = True, use_myelin: bool = False,
                    ssd_kwargs: Optional[dict] = None):
    """ Given multiple ssvs, defined by ssv_params, this function produces samples for each ssv which are
        later processed by the prediction function. Different models of the pred_func need different contexts.
        Therefore, this function splits each ssv multiple times, depending on the entries in the given dicts,
        like e.g. ``ctx_size``.

    Args:
        ssv_params: Parameters of the ssvs which should get processed.
        pred_types: List of prediction types, e.g. ['ads', 'abt', dnh'] for axon, dendrite, soma; ...
        batchsize: Dict of batch sizes, keyed by the respective context size. Models with the same context size
            will have the same batch size, no matter how many points are sampled from the contexts.
        npoints: Dict with numbers of sample points (extracted from subset) keyed by the respective prediction key.
        ctx_size: Dict with context sizes, keyed by the respective prediction key.
        transform: Dict of transformations, keyed by the respective prediction key.
        use_subcell: Flag for using cell organelles
        use_myelin: Flag for using myelin.
        ssd_kwargs: Keyword arguments to initialize the ssd.
        ctx_dst_fac: Defines the redundancy of the predictions by determining the distance of the base nodes
            used for context extraction. Higher ``ctx_dst_fac`` means smaller distance (ctx / ctx_dst_fac) and
            therefore larger context overlap and longer processing time.
    """
    if pred_types is None:
        raise ValueError("pred_types is None. However, pred_types must at least contain one pred_type such as "
                         "'cmpt'")
    feat_dc = dict(pts_feat_dict)
    # TODO: add use_syntype
    del feat_dc['syn_ssv_asym']
    del feat_dc['syn_ssv_sym']
    del feat_dc['sv_myelin']
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
        # transform ssv into a HybridCloud. voxel_dict will be forwarded to postprocessing function to perform
        # final mapping between hc and ssv.
        hc, voxel_dict = sso2hc(ssv, tuple(feat_dc.keys()), tuple(feat_dc.values()), 'compartment',
                                myelin=use_myelin)
        ssv.clear_cache()
        # pred_types with the same ctx_size use the same chunks (possibly with different sampling)
        for ctx in ctx_size:
            # choose base nodes with context overlap
            base_node_dst = ctx / ctx_dst_fac
            # select source nodes for context extraction
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(hc.nodes)
            pcd, idcs = pcd.voxel_down_sample_and_trace(
                    base_node_dst, pcd.get_min_bound(), pcd.get_max_bound())
            source_nodes = np.max(idcs, axis=1)
            bs = min(len(source_nodes), batchsize[ctx])
            n_batches = int(np.ceil(len(source_nodes) / bs))
            # add additional source nodes to fill batches
            if len(source_nodes) % bs != 0:
                source_nodes = np.concatenate([np.random.choice(source_nodes, bs - len(source_nodes) % bs),
                                               source_nodes])
            node_arrs = context_splitting_kdt(hc, source_nodes, ctx)
            # collect contexts into batches (each batch contains every n_batches contexts
            # (e.g. every 4th if n_batches = 4)
            for ii in range(n_batches):
                arr_list = []
                # Splitting is the same for the same ctx size, but sampling and transform is different each time
                for p_t in ctx_size[ctx]:
                    batch = np.zeros((bs, npoints[p_t], 3))
                    batch_f = np.zeros((bs, npoints[p_t], len(feat_dc)))
                    # used later for removing cell organelles
                    batch_mask = np.zeros((bs, npoints[p_t]), dtype=bool)
                    idcs_list = []
                    arr_list.append((batch, batch_f, batch_mask, idcs_list))
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
                            hc_sub, idcs_sub = extract_subset(hc, context_splitting_kdt(hc, sn, ctx))
                        else:
                            hc_sub, idcs_sub = extract_subset(hc, node_arrs[ix])
                        ix += 1
                    # fill batches with sampled and transformed subsets
                    for ix, p_t in enumerate(ctx_size[ctx]):
                        hc_sample, idcs_sample = clouds.sample_cloud(hc_sub, npoints[p_t])
                        # get vertex indices respective to total hc
                        global_idcs = idcs_sub[idcs_sample.astype(int)]
                        # prepare masks for filtering sv vertices
                        bounds = hc.obj_bounds['sv']
                        sv_mask = np.logical_and(global_idcs < bounds[1], global_idcs >= bounds[0])
                        hc_sample.set_features(label_binarize(hc_sample.features, classes=np.arange(len(feat_dc))))
                        if transform is not None:
                            transform[p_t](hc_sample)
                        arr_list[ix][0][cnt] = hc_sample.vertices
                        arr_list[ix][1][cnt] = hc_sample.features
                        # masks get used later when mapping predictions back onto the cell surface during postprocessing
                        arr_list[ix][2][cnt] = sv_mask
                        arr_list[ix][3].append(global_idcs[sv_mask])
                    cnt += 1
                batch_progress = ii + 1
                # return samples with same ctx, but possibly different sampling and transform
                for ix, p_t in enumerate(ctx_size[ctx]):
                    # each batch is defined by its ssv_params (id), the batch progress and the prediction type
                    yield curr_ssv_params, (arr_list[ix][1], arr_list[ix][0]), \
                          (arr_list[ix][3], arr_list[ix][2], voxel_dict['sv']), \
                          (batch_progress, n_batches, p_t, pred_types, ctx)


def pts_pred_cmpt(m, inp, q_out, d_out, q_cnt, device, bs):
    """
    Args:
        m: Dict with pytorch models keyed by the prediction type
        inp: Tuple of ssv_params, (feats, verts), (mapping_indices, masks (for removing cell organelles),
        voxel_indices), (batch_progress, n_batches, p_t (prediction type of this batch), pred_types (only
        present in first batches)
        q_out: Queue for worker syncing
        d_out: Dict to save prediction results
        q_cnt: Queue for worker syncing
        device: Device to which models in m have been transfered
        bs: Dict of batch sizes, keyed by the respective context size. Models with the same context size
            will have the same batch size, no matter how many points are sampled from the contexts.
    """
    ssv_params, model_inp, batch_info, batch_progress = inp
    idcs_list = batch_info[0]
    batch_mask = batch_info[1]
    idcs_voxel = batch_info[2]
    # get context dependent batch size
    bs = bs[batch_progress[4]]
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
    # batch_progress: (batch_progress, n_batches, p_t, pred_types), or (batch_progress, n_batches, p_t)
    res = dict(idcs=np.concatenate(idcs_list), preds=np.concatenate(res),
               batch_progress=batch_progress, idcs_voxel=idcs_voxel)
    q_cnt.put_nowait(1./batch_progress[1]/len(batch_progress[3]))
    pred_types = batch_progress[3]

    d_out[ssv_params['ssv_id']].put(res)
    if batch_progress[0] == 1 and batch_progress[2] == pred_types[0]:
        q_out.put(ssv_params)


def pts_postproc_cpmt(sso_params: dict, d_in: dict):
    """
    Receives predictions from the prediction queue, waits until all predictions for one sso have been received and
    then concatenates and evaluates all the predictions (taking the majority vote over all predictions per vertex).
    The resulting label arrays (number dependent on the number of prediction types (e.g. ads, abt, dnh)) get saved
    in the original sso object.

    Args:
        sso_params: Params of sso object for which the predictions should get evaluated.
        d_in: Dict with prediction results
    """
    sso = SuperSegmentationObject(**sso_params)
    preds = {}
    preds_idcs = {}
    # indices of vertices which were chosen during voxelization (allows mapping between hc and sso)
    voxel_idcs = None
    # predictions types which were forwarded from the loading function
    pred_types = None
    p_t_progress = {}
    p_t_done = {}

    while True:
        try:
            # res: [(dict(t_pts=.., t_label, batch_process)]
            res = d_in[sso.id].get_nowait()
        except queues.Empty:
            time.sleep(0.25)
            continue
        if voxel_idcs is None:
            voxel_idcs = res['idcs_voxel']
        if pred_types is None:
            pred_types = res['batch_progress'][3]
            for p_t in pred_types:
                p_t_progress[p_t] = 0
                p_t_done[p_t] = False
                preds[p_t] = []
                preds_idcs[p_t] = []
        p_t = res['batch_progress'][2]
        preds[p_t].append(np.argmax(res['preds'], axis=1))
        preds_idcs[p_t].append(res['idcs'])
        # check if all predictions for this sso were received (all pred_types must evaluate to True)
        p_t_progress[p_t] += 1
        if p_t_progress[p_t] == res['batch_progress'][1]:
            p_t_done[p_t] = True
        done = True
        for p_t in p_t_progress:
            done = done and p_t_done[p_t]
        if done:
            break
    # evaluate predictions and map them to the original sso vertices (with respect to
    # indices which were chosen during voxelization
    sso_vertices = sso.mesh[1].reshape((-1, 3))
    ld = sso.label_dict('vertex')
    for p_t in pred_types:
        preds[p_t] = np.concatenate(preds[p_t])
        preds_idcs[p_t] = np.concatenate(preds_idcs[p_t])
        pred_labels = np.ones((len(voxel_idcs), 1))*-1
        evaluate_preds(preds_idcs[p_t], preds[p_t], pred_labels)
        # pred labels now contain the prediction with respect to the hc vertices
        sso_preds = np.ones((len(sso_vertices), 1))*-1
        sso_preds[voxel_idcs] = pred_labels
        # save prediction in the vertex prediction attributes of the sso, keyed by their prediction type.
        ld[p_t] = sso_preds
    # TODO: use single array for all compartment predictions in the entire pipeline
    # convert to conventional
    # 'axoness' (0: dendrite, 1: axon, 2: soma, 3: en-passant, 4: terminal, 5: background, 6: unpredicted)
    # and
    # 'spiness' (0: shaft, 1: head, 2: neck, 3: other, 4: background, 5: unpredicted)
    cmpt_preds = convert_cmpt_preds(sso)
    ax_pred = np.array(cmpt_preds)  # trigger copy
    # convert spine labels to dendrite
    ax_pred[ax_pred == 5] = 0
    ax_pred[ax_pred == 6] = 0
    ax_pred[ax_pred == -1] = 5  # unpredicted to unpredicted
    # prepare dendritic compartment labels in multi-view layout
    sp_pred = np.array(cmpt_preds)
    sp_pred[cmpt_preds == 1] = 3  # axon to 'other'
    sp_pred[cmpt_preds == 2] = 3  # soma to 'other'
    sp_pred[cmpt_preds == 3] = 3  # en-passant to 'other'
    sp_pred[cmpt_preds == 4] = 3  # terminal to 'other'
    sp_pred[cmpt_preds == 5] = 1  # head to head
    sp_pred[cmpt_preds == 6] = 2  # neck to neck
    sp_pred[cmpt_preds == -1] = 5  # unpredicted to unpredicted

    pred_key_sp = sso.config['spines']['semseg2mesh_spines']['semseg_key']
    pred_key_ax = sso.config['compartments']['view_properties_semsegax']['semseg_key']

    ld[pred_key_ax] = ax_pred.astype(np.int32)
    ld[pred_key_sp] = sp_pred.astype(np.int32)
    del ld['dnh']
    del ld['abt']
    del ld['ads']
    ld.push()
    sso.load_skeleton()
    node_preds = sso.semseg_for_coords(sso.skeleton['nodes'], pred_key_sp, **sso.config['spines']['semseg2coords_spines'])
    sso.skeleton[pred_key_sp] = node_preds  # skeleton key will be saved to file with `semsegaxoness2skel` call below
    map_properties = sso.config['compartments']['map_properties_semsegax']
    max_dist = sso.config['compartments']['dist_axoness_averaging']
    semsegaxoness2skel(sso, map_properties, pred_key_ax, max_dist)
    return [sso.id], [True]


def convert_cmpt_preds(sso: SuperSegmentationObject) -> np.ndarray:
    """
    Convert vertex predictions in ``label_dict`` of cell reconstruction object to common layout.

    Expected keys in label dict for point cloud based predictions:
        * Coarse compartments ['ads']: dendrite (0), axon (1), soma (2).
        * Axon compartments ['abt']: axon (0), en-passant bouton (1), terminal bouton (2).
        * Dendritic compartments ['dnh']: dendritic shaft (0), spine neck (1), spine head (2)

    Alternative layout from multi-views:
        * 'axoness': 0: dendrite, 1: axon, 2: soma, 3: en-passant, 4: terminal, 5: background, 6: unpredicted.
        * 'spiness': 0: shaft, 1: head, 2: neck, 3: other, 4: background, 5: unpredicted.

    Resulting layout:
        Single array with dendrite (0), axon (1), soma (2), en-passant bouton (3), terminal bouton (4),
        spine head (5), spine neck (6).

    Args:
        sso: Cell reconstruction.

    Returns:
        Single array with compartment predictions.
    """
    ld = sso.label_dict('vertex')
    if 'ads' in ld and 'abt' in ld and 'dnh' in ld:
        ads = ld['ads']
        abt = ld['abt']
        dnh = ld['dnh']
        a_mask = (ads == 1).reshape(-1)
        d_mask = (ads == 0).reshape(-1)
        abt[abt == 1] = 3
        abt[abt == 2] = 4
        dnh[dnh == 1] = 6
        dnh[dnh == 2] = 5
        ads[a_mask] = abt[a_mask]
        ads[d_mask] = dnh[d_mask]
    elif 'axoness' in ld and 'spiness' in ld:
        adsbt = ld['axoness']
        shnobu = ld['spiness']
        raise NotImplementedError('Conversion for multi-view predictions is not implemented yet.')
    else:
        raise KeyError(f'Key required for conversion not found. Available keys: {ld.keys()}')
    if np.ndim(ads) == 2:
        ads = ads.squeeze(1)
    return ads

# ------------------------------------------------- HELPER METHODS --------------------------------------------------#


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


# TODO: Merge with get_pt_kwargs
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
    npoints = int(re.findall(r'_nb(\d+)_', mdir)[-1])
    scale_fact = int(re.findall(r'_scale(\d+)_', mdir)[-1])
    ctx = int(re.findall(r'_ctx(\d+)_', mdir)[-1])
    feat_dim = int(re.findall(r'_fdim(\d+)_', mdir)[-1])
    class_num = int(re.findall(r'_cnum(\d+)_', mdir)[-1])
    pred_type = re.findall(r'_t([^_]+)_', mdir)[-1]
    batchsize = int(re.findall(r'_bs(\d+)_', mdir)[-1])
    # TODO: Fix neighbor_nums or create extra model
    mkwargs = dict(input_channels=feat_dim, output_channels=class_num, use_norm=use_norm, use_bias=use_bias,
                   norm_type=norm_type)
    loader_kwargs = dict(ctx_size=ctx, scale_fact=scale_fact, npoints=npoints, pred_type=pred_type, bs=batchsize)
    return mkwargs, loader_kwargs


# -------------------------------------------- SSO TO MORPHX CONVERSION ---------------------------------------------#


# TODO: Merge with _load_ssv_hc
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



# Backport of https://github.com/python/cpython/pull/4819
# Improvements to the Manager / proxied shared values code
# broke handling of proxied objects without a custom proxy type,
# as the AutoProxy function was not updated.
#
# This code adds a wrapper to AutoProxy if it is missing the
# new argument.

from inspect import signature
from functools import wraps
from multiprocessing import managers
orig_AutoProxy = managers.AutoProxy


@wraps(managers.AutoProxy)
def AutoProxy(*args, incref=True, manager_owned=False, **kwargs):
    # Create the autoproxy without the manager_owned flag, then
    # update the flag on the generated instance. If the manager_owned flag
    # is set, `incref` is disabled, so set it to False here for the same
    # result.
    autoproxy_incref = False if manager_owned else incref
    proxy = orig_AutoProxy(*args, incref=autoproxy_incref, **kwargs)
    proxy._owned_by_manager = manager_owned
    return proxy


def apply_proxy_fix():
    """
    See https://stackoverflow.com/questions/46779860/multiprocessing-managers-and-custom-classes
    """
    if "manager_owned" in signature(managers.AutoProxy).parameters:
        return

    log_handler.debug("Patching multiprocessing.managers.AutoProxy to add manager_owned")
    managers.AutoProxy = AutoProxy

    # re-register any types already registered to SyncManager without a custom
    # proxy type, as otherwise these would all be using the old unpatched AutoProxy
    SyncManager = managers.SyncManager
    registry = managers.SyncManager._registry
    for typeid, (callable, exposed, method_to_typeid, proxytype) in registry.items():
        if proxytype is not orig_AutoProxy:
            continue
        create_method = hasattr(managers.SyncManager, typeid)
        SyncManager.register(
            typeid,
            callable=callable,
            exposed=exposed,
            method_to_typeid=method_to_typeid,
            create_method=create_method,
        )