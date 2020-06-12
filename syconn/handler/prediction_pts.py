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
import re
import time
from multiprocessing import Process, Queue, Manager
from typing import Iterable, Union, Optional, Tuple, Callable, List

import morphx.processing.clouds as clouds
import networkx as nx
import numpy as np
import scipy.special
import tqdm
from morphx.classes.cloudensemble import CloudEnsemble
from morphx.classes.hybridcloud import HybridCloud
from morphx.classes.hybridmesh import HybridMesh
from morphx.classes.pointcloud import PointCloud
from morphx.processing.hybrids import extract_subset
from morphx.processing.objects import bfs_vertices, context_splitting_v2
from scipy import spatial
from scipy.spatial import cKDTree
from sklearn.preprocessing import label_binarize

from syconn import global_params
from syconn.handler import log_handler
from syconn.handler.basics import chunkify_successive
from syconn.mp.mp_utils import start_multiprocess_imap
from syconn.reps.super_segmentation import SuperSegmentationDataset
from syconn.reps.super_segmentation import SuperSegmentationObject
from syconn.reps.super_segmentation_helper import map_myelin2coords, majorityvote_skeleton_property

# for readthedocs build
try:
    import torch
except ImportError:
    pass
# TODO: specify further, add to config
pts_feat_dict = dict(sv=0, mi=1, syn_ssv=3, syn_ssv_sym=3, syn_ssv_asym=4, vc=2)
# in nm, should be replaced by Poisson disk sampling
pts_feat_ds_dict = dict(celltype=dict(sv=70, mi=100, syn_ssv=70, syn_ssv_sym=70, syn_ssv_asym=70, vc=100),
                        glia=dict(sv=50, mi=100, syn_ssv=100, syn_ssv_sym=100, syn_ssv_asym=100, vc=100),
                        compartment=dict(sv=80, mi=100, syn_ssv=100, syn_ssv_sym=100, syn_ssv_asym=100, vc=100))
m_postproc = Manager()


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
        if not q_postproc.empty():
            inp = q_postproc.get()
            if 'STOP' in inp:
                if inp not in stops_received:
                    stops_received.add(inp)
                else:
                    q_postproc.put(inp)
                    time.sleep(np.random.randint(25) / 10)
                if len(stops_received) == n_worker_pred:
                    break
                continue
        else:
            if len(stops_received) == n_worker_pred:
                break
            time.sleep(0.5)
            continue
        q_out.put(postproc_func(inp, d_postproc, **postproc_kwargs))
    log_handler.debug(f'Worker postproc done.')
    q_out.put('END')


def worker_pred(worker_cnt: int, q_out: Queue, d_out: dict, q_progress: Queue, q_in: Queue,
                model_loader: Callable, pred_func: Callable, n_worker_load: int, n_worker_postporc: int,
                device: str, mpath: Optional[str] = None, bs: Optional[int] = None):
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
        n_worker_postporc: Number of postproc worker.
    """
    m = model_loader(mpath, device)
    stops_received = set()
    while True:
        if not q_in.empty():
            inp = q_in.get()
            if 'STOP' in inp:
                if inp not in stops_received:
                    stops_received.add(inp)
                else:
                    q_in.put(inp)
                    time.sleep(np.random.randint(25) / 10)
                if len(stops_received) == n_worker_load:
                    break
                continue
        else:
            time.sleep(0.5)
            continue
        pred_func(m, inp, q_out, d_out, q_progress, device, bs)
    log_handler.debug(f'Pred worker {worker_cnt} stopped.')
    for _ in range(n_worker_postporc):
        q_out.put(f'STOP{worker_cnt}')


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
        if q_loader.empty():
            break
        else:
            kwargs = q_loader.get()
        res = loader_func(**kwargs)
        for el in res:
            while True:
                if q_out.full():
                    time.sleep(1)
                else:
                    break
            q_out.put(el)
    time.sleep(1)
    for _ in range(n_worker_pred):
        q_out.put(f'STOP{worker_cnt}')
    q_loader_sync.put('DONE')


def listener(q_progress: Queue, q_loader_sync: Queue, nloader: int, total: int):
    """

    Args:
        q_progress:
        q_loader_sync:
        nloader:
        total:
    """
    pbar = tqdm.tqdm(total=total, leave=False)
    cnt_loder_done = 0
    while True:
        if q_progress.empty():
            time.sleep(0.2)
        else:
            res = q_progress.get()
            if res is None:  # final stop
                assert cnt_loder_done == nloader
                pbar.close()
                break
            pbar.update(res)
        if q_loader_sync.empty() or cnt_loder_done == nloader:
            pass
        else:
            _ = q_loader_sync.get()
            cnt_loder_done += 1


def _vert_counter(args):
    ssv_id, ssd_kwargs = args
    return SuperSegmentationObject(ssv_id, **ssd_kwargs).mesh[1].shape[0]


def predict_pts_plain(ssd_kwargs: Union[dict, Iterable], model_loader: Callable,
                      loader_func: Callable, pred_func: Callable,
                      npoints: int, scale_fact: float, ctx_size: int,
                      postproc_func: Optional[Callable] = None,
                      postproc_kwargs: Optional[dict] = None,
                      output_func: Optional[Callable] = None,
                      mpath: Optional[str] = None,
                      nloader: int = 4, npredictor: int = 2, npostptroc: int = 2,
                      ssv_ids: Optional[Union[list, np.ndarray]] = None,
                      use_test_aug: bool = False,
                      seeded: bool = False,
                      device: str = 'cuda', bs: int = 40,
                      loader_kwargs: Optional[dict] = None,
                      redundancy: Union[int, tuple] = (25, 100)) -> dict:
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
        npostptroc: Optional worker for post processing, see `postproc_func`.
        ssv_ids: IDs of cells to predict.
        use_test_aug: Use test-time augmentations. Currently this adds the following transformation
            to the basic transforms:

                transform = [clouds.Normalization(scale_fact), clouds.Center()]
                [clouds.RandomVariation((-5, 5), distr='normal')] + transform + [clouds.RandomRotate()]

        device: pytorch device.
        bs: Batch size.
        loader_kwargs: Optional keyword arguments for loader func.
        redundancy: Only used if redundancy is a dict, e.g. as for cell type prediction. If int it is used as minimum
            redundancy, if tuple it is (min redundancy and max redundancy).
        seeded:

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
    if loader_kwargs is None:
        loader_kwargs = dict()
    if output_func is None:
        def output_func(res_dc, ret):
            for ix, out in zip(*ret):
                res_dc[ix].append(out)
    if postproc_func is None:
        def postproc_func(x, *args, **kwargs): return x
    if postproc_kwargs is None:
        postproc_kwargs = dict()

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
            ssv_ids = np.array(ssv_ids, np.uint)
        # redundancy, default: 3 * npoints / #vertices
        ssv_n_vertices = start_multiprocess_imap(_vert_counter, [(ssv_id, ssd_kwargs) for ssv_id in ssv_ids],
                                                 nb_cpus=None)
        if type(redundancy) is tuple:
            ssv_redundancy = [min(max(nverts // npoints, redundancy[0]), redundancy[1]) for nverts in ssv_n_vertices]
        else:
            ssv_redundancy = [max(nverts // npoints, redundancy) for nverts in ssv_n_vertices]
        ssv_redundancy = np.array(ssv_redundancy)
        sorted_ix = np.argsort(ssv_redundancy)[::-1]
        ssv_redundancy = ssv_redundancy[sorted_ix]
        ssv_ids = ssv_ids[sorted_ix]
        ssv_ids = np.concatenate([np.array([ssv_ids[ii]] * ssv_redundancy[ii], dtype=np.uint)
                                  for ii in range(len(ssv_ids))])
        params_in = [{**params_kwargs, **dict(ssv_ids=ch)} for ch in chunkify_successive(
            ssv_ids, int(np.ceil(len(ssv_ids) / nloader)))]
    else:
        params_kwargs = dict(batchsize=bs, npoints=npoints, transform=transform, ctx_size=ctx_size, **loader_kwargs)
        params_in = [{**params_kwargs, **dict(ssv_params=[ch])} for ch in ssd_kwargs]
        ssv_ids = [el['ssv_id'] for el in ssd_kwargs]

    nsamples_tot = len(ssv_ids)

    q_loader = Queue()
    for el in params_in:
        q_loader.put(el)
    q_load = Queue(maxsize=20 * npredictor)
    q_progress = Queue()
    d_postproc = m_postproc.dict()
    for k in ssv_ids:
        d_postproc[k] = m_postproc.list()
    q_postproc = m_postproc.Queue()
    q_out = Queue()
    q_loader_sync = Queue()
    producers = [Process(target=worker_load, args=(ii, q_loader, q_load, q_loader_sync, loader_func, npredictor))
                 for ii in range(nloader)]
    for p in producers:
        p.start()
    consumers = [
        Process(target=worker_pred, args=(ii, q_postproc, d_postproc, q_progress, q_load, model_loader, pred_func,
                                          nloader, npostptroc, device, mpath, bs)) for ii in range(npredictor)]
    for c in consumers:
        c.start()
    postprocs = [Process(target=worker_postproc, args=(q_out, q_postproc, d_postproc, postproc_func, postproc_kwargs,
                                                       npredictor)) for _ in range(npostptroc)]
    for c in postprocs:
        c.start()
    dict_out = collections.defaultdict(list)
    cnt_end = 0
    lsnr = Process(target=listener, args=(q_progress, q_loader_sync, nloader, nsamples_tot))
    lsnr.start()
    while True:
        if q_out.empty():
            if cnt_end == npostptroc:
                break
            time.sleep(0.5)
            continue
        res = q_out.get()
        if res == 'END':
            cnt_end += 1
            continue
        output_func(dict_out, res)

    q_progress.put(None)
    lsnr.join()
    for p in producers:
        p.join()
        p.close()
    for c in consumers:
        c.join()
        c.close()
    for c in postprocs:
        c.join()
        c.close()
    return dict_out


@functools.lru_cache(256)
def _load_ssv_hc(args):
    ssv, feats, feat_labels, pt_type, radius = args
    vert_dc = dict()
    for k in feats:
        pcd = o3d.geometry.PointCloud()
        verts = ssv.load_mesh(k)[1].reshape(-1, 3)
        pcd.points = o3d.utility.Vector3dVector(verts)
        pcd = pcd.voxel_down_sample(voxel_size=pts_feat_ds_dict[pt_type][k])
        vert_dc[k] = np.asarray(pcd.points)
    sample_feats = np.concatenate([[feat_labels[ii]] * len(vert_dc[k])
                                   for ii, k in enumerate(feats)])
    sample_pts = np.concatenate([vert_dc[k] for k in feats])
    if not ssv.load_skeleton():
        raise ValueError(f'Couldnt find skeleton of {ssv}')
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


def pts_loader_scalar(ssd_kwargs: dict, ssv_ids: Union[list, np.ndarray],
                      batchsize: int, npoints: int, ctx_size: float,
                      transform: Optional[Callable] = None,
                      train: bool = False, draw_local: bool = False,
                      draw_local_dist: int = 1000, seeded: bool = False,
                      use_ctx_sampling: bool = True,
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
        seeded: If True, will set the seed to ``hash(frozenset(ssv_id, n_samples, curr_batch_count))``.
        use_ctx_sampling: Use context based sampling. If True, uses `ctx_size` (in nm). Otherwise vist skeleton nodes
            until `npoints` have been collected.

    Yields: SSV IDs [M, ], (point feature [N, C], point location [N, 3])

    """
    np.random.shuffle(ssv_ids)
    ssd = SuperSegmentationDataset(**ssd_kwargs)
    # TODO: add `use_syntype kwarg and cellshape only
    feat_dc = dict(pts_feat_dict)
    if 'syn_ssv' in feat_dc:
        del feat_dc['syn_ssv']
    if not train:
        if draw_local:
            raise NotImplementedError()
        for ssv_id, occ in zip(*np.unique(ssv_ids, return_counts=True)):
            n_batches = int(np.ceil(occ / batchsize))
            for ii in range(n_batches):
                n_samples = min(occ, batchsize)
                occ -= batchsize
                if seeded:
                    np.random.seed(np.uint32(hash(frozenset((ssv_id, n_samples, ii)))))
                ssv = ssd.get_super_segmentation_object(ssv_id)
                hc = _load_ssv_hc((ssv, tuple(feat_dc.keys()), tuple(
                    feat_dc.values()), 'celltype', None))
                ssv.clear_cache()
                npoints_ssv = min(len(hc.vertices), npoints)
                batch = np.zeros((n_samples, npoints_ssv, 3))
                batch_f = np.zeros((n_samples, npoints_ssv, len(feat_dc)))
                cnt = 0
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(hc.nodes)
                pcd, idcs = pcd.voxel_down_sample_and_trace(2500, pcd.get_min_bound(), pcd.get_max_bound())
                nodes = np.max(idcs, axis=1)
                # nodes = hc.base_points(threshold=2500, source=len(hc.nodes) // 2)
                # nodes = sparsify_skeleton_fast(hc.graph(), scal=np.array([1, 1, 1]), min_dist_thresh=2500,
                #                                max_dist_thresh=2500, dot_prod_thresh=0).nodes()
                source_nodes = np.random.choice(nodes, n_samples, replace=len(nodes) < n_samples)
                for source_node in source_nodes:
                    # This might be slow
                    while True:
                        if use_ctx_sampling:
                            node_ids = context_splitting_v2(hc, source_node, ctx_size)
                        else:
                            node_ids = bfs_vertices(hc, source_node, npoints_ssv)
                        hc_sub = extract_subset(hc, node_ids)[0]  # only pass HybridCloud
                        sample_feats = hc_sub.features
                        if len(sample_feats) > 0:
                            break
                        print(f'FOUND SOURCE NODE WITH ZERO VERTICES AT {hc.nodes[source_node]} IN "{ssv}".')
                        source_node = np.random.choice(source_nodes)
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
    else:
        ssv_ids = np.unique(ssv_ids)
        for curr_ssvid in ssv_ids:
            ssv = ssd.get_super_segmentation_object(curr_ssvid)
            hc = _load_ssv_hc((ssv, tuple(feat_dc.keys()), tuple(
                feat_dc.values()), 'celltype', None))
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
            ixs = np.ones((batchsize,), dtype=np.uint) * ssv.id
            cnt = 0
            source_nodes = np.random.choice(len(hc.nodes), batchsize, replace=len(hc.nodes) < batchsize)
            if draw_local:
                # only use half of the nodes and choose a close-by node as root for similar context retrieval
                source_nodes = source_nodes[::2]
                sn_new = []
                g = hc.graph(simple=False)
                for n in source_nodes:
                    sn_new.append(n)
                    paths = nx.single_source_dijkstra_path(g, n, draw_local_dist)
                    neighs = np.array(list(paths.keys()), dtype=np.int)
                    sn_new.append(np.random.choice(neighs, 1)[0])
                source_nodes = sn_new
            for source_node in source_nodes:
                # local_bfs = bfs_vertices(hc, source_node, npoints_ssv)
                while True:
                    if use_ctx_sampling:
                        node_ids = context_splitting_v2(hc, source_node, ctx_size_fluct)
                    else:
                        node_ids = bfs_vertices(hc, source_node, npoints_ssv)
                    hc_sub = extract_subset(hc, node_ids)[0]  # only pass HybridCloud
                    sample_feats = hc_sub.features
                    if len(sample_feats) > 0:
                        break
                    log_handler.debug(f'FOUND SOURCE NODE WITH ZERO VERTICES AT {hc.nodes[source_node]} IN "{ssv}".')
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
    res = dict(probas=np.concatenate(res), batch_progress=(batch_progress, n_batches))

    q_cnt.put(n_samples)

    if batch_progress == 1:
        q_out.put(ssv_kwargs)
    d_out[ssv_kwargs['ssv_id']].append(res)


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
    q_cnt.put(n_samples)
    q_out.put(([ssv_kwargs['ssv_id']] * n_samples, res))


def pts_postproc_scalar(ssv_kwargs: dict, d_in: dict, pred_key: Optional[str] = None) -> Tuple[List[int], List[bool]]:
    """
    Framework is very similar to what will be needed for semantic segmentation of surface points.
    Requires adaptions in pts_loader_semseg and correct merge of vertex indices instead of skeleton node cKDTree.

    Args:
        ssv_kwargs:
        d_in:
        pred_key:

    Returns:

    """
    if pred_key is None:
        pred_key = 'celltype_cnn_e3'
    curr_ix = 0
    sso = SuperSegmentationObject(**ssv_kwargs)
    sso.load_attr_dict()
    celltype_probas = []
    while True:
        if len(d_in[sso.id]) < curr_ix + 1:
            time.sleep(0.5)
            continue
        # res: [(dict(probas=.., batch_process)]
        res = d_in[sso.id][curr_ix]
        celltype_probas.append(res['probas'])
        d_in[sso.id][curr_ix] = None
        curr_ix += 1
        if res['batch_progress'][0] == res['batch_progress'][1]:
            break

    logit = np.concatenate(celltype_probas)
    cls = np.argmax(logit, axis=1).squeeze()
    cls_maj = collections.Counter(cls).most_common(1)[0][0]

    sso.save_attributes([pred_key], [cls_maj])
    sso.save_attributes([f"{pred_key}_probas"], [logit])

    d_in[sso.id][curr_ix] = None
    return [sso.id], [True]


def pts_loader_local_skel(ssv_params: List[dict], out_point_label: Optional[List[Union[str, int]]] = None,
                          batchsize: Optional[int] = None, npoints: Optional[int] = None,
                          ctx_size: Optional[float] = None, transform: Optional[Callable] = None,
                          n_out_pts: int = 100, train=False, base_node_dst: float = 10000,
                          use_ctx_sampling: bool = True, use_syntype: bool = False,
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
    default_kwargs = dict(mesh_caching=False, create=False)
    for curr_ssv_params in ssv_params:
        default_kwargs.update(curr_ssv_params)
        curr_ssv_params = default_kwargs
        # do not write SSV mesh in case it does not exist (will be build from SV meshes)
        ssv = SuperSegmentationObject(**curr_ssv_params)
        hc = _load_ssv_hc((ssv, tuple(feat_dc.keys()), tuple(feat_dc.values()), 'glia', None))
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
                        node_ids = context_splitting_v2(hc, source_node, ctx_size_fluct)
                    else:
                        node_ids = bfs_vertices(hc, source_node, npoints_ssv)
                    hc_sub = extract_subset(hc, node_ids)[0]  # only pass HybridCloud
                    sample_feats = hc_sub.features
                    if len(sample_feats) > 0:
                        break
                    log_handler.debug(f'FOUND SOURCE NODE WITH ZERO VERTICES AT {hc.nodes[source_node]} IN "{ssv}".')
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
                hc_sub._nodes = out_coords
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
    res = dict(t_pts=out_pts_orig, t_l=np.concatenate(res), batch_progress=(batch_progress, n_batches))

    q_cnt.put(1. / n_batches)
    if batch_progress == 1:
        q_out.put(ssv_params)
    d_out[ssv_params['ssv_id']].append(res)


def pts_postproc_glia(ssv_params: dict, d_in: dict, pred_key: str, lo_first_n: Optional[int] = None,
                      partitioned: Optional[bool] = False, apply_softmax: bool = True,
                      sample_loc_ds: float = 100, pred2loc_knn: int = 5) -> Tuple[List[dict], List[bool]]:
    curr_ix = 0
    sso = SuperSegmentationObject(**ssv_params)
    node_probas = []
    node_coords = []
    while True:
        if len(d_in[sso.id]) < curr_ix + 1:
            time.sleep(0.5)
            continue
        # res: [(dict(t_pts=.., t_label, batch_process)]
        res = d_in[sso.id][curr_ix]
        # el['t_l'] has shape (b, num_points, n_classes) -> (n_nodes, n_classes)
        node_probas.append(res['t_l'].reshape(-1, 2))
        # el['t_pts'] has shape (b, num_points, 3) -> (n_nodes, 3)
        node_coords.append(res['t_pts'].reshape(-1, 3))
        d_in[sso.id][curr_ix] = None
        curr_ix += 1
        if res['batch_progress'][0] == res['batch_progress'][1]:
            break
    del d_in[sso.id]
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
        # every node has at least one prediction
        # get mean proba for this super voxel
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
    res = dict(t_pts=out_pts_orig, t_l=np.concatenate(res), batch_progress=(batch_progress, n_batches))

    q_cnt.put(1. / n_batches)
    if batch_progress == 1:
        q_out.put(ssv_params)
    d_out[ssv_params['ssv_id']].append(res)


def pts_postproc_embedding(ssv_params: dict, d_in: dict, pred_key: Optional[str] = None
                           ) -> Tuple[List[dict], List[bool]]:
    curr_ix = 0
    sso = SuperSegmentationObject(**ssv_params)
    node_embedding = []
    node_coords = []
    while True:
        if len(d_in[sso.id]) < curr_ix + 1:
            time.sleep(0.5)
            continue
        # res: [(dict(t_pts=.., t_label, batch_process)]
        res = d_in[sso.id][curr_ix]
        # el['t_l'] has shape (b, num_points, n_latent_dim) -> (n_nodes, n_latent_dim)
        node_embedding.append(res['t_l'].reshape(-1, res['t_l'].shape[-1]))
        # el['t_pts'] has shape (b, num_points, 3) -> (n_nodes, 3)
        node_coords.append(res['t_pts'].reshape(-1, 3))
        d_in[sso.id][curr_ix] = None
        curr_ix += 1
        if res['batch_progress'][0] == res['batch_progress'][1]:
            break
    del d_in[sso.id]
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
                            use_subcell: bool = False,
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

    Yields: SSV IDs [M, ], (point feature [N, C], point location [N, 3])

    """
    feat_dc = dict(pts_feat_dict)
    # TODO: add use_syntype
    del feat_dc['syn_ssv_asym']
    del feat_dc['syn_ssv_sym']
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
            batch = np.zeros((batchsize, npoints_ssv, 3))
            batch_f = np.zeros((batchsize, npoints_ssv, len(feat_dc)))
            batch_out = np.zeros((batchsize, n_out_pts_curr, 3))
            batch_out_l = np.zeros((batchsize, n_out_pts_curr, 1))
            cnt = 0
            for source_node in source_nodes[ii::n_batches]:
                # create local context
                # node_ids = bfs_vertices(hc, source_node, npoints_ssv)
                while True:
                    node_ids = context_splitting_v2(hc, source_node, ctx_size_fluct)
                    hc_sub = extract_subset(hc, node_ids)[0]  # only pass HybridCloud
                    sample_feats = hc_sub.features
                    if len(sample_feats) > 0:
                        break
                    print(f'FOUND SOURCE NODE WITH ZERO VERTICES AT {hc.nodes[source_node]} IN "{pkl_f}".')
                    source_node = np.random.choice(source_nodes)
                sample_pts = hc_sub.vertices
                sample_labels = hc_sub.labels

                # sub-sample vertices
                sample_ixs = np.arange(len(sample_pts))
                np.random.shuffle(sample_ixs)
                sample_pts = sample_pts[sample_ixs][:npoints_ssv]
                sample_feats = sample_feats[sample_ixs][:npoints_ssv]
                sample_labels = sample_labels[sample_ixs][:npoints_ssv]
                # add duplicate points before applying the transform if sample_pts
                # has less points than npoints_ssv
                npoints_add = npoints_ssv - len(sample_pts)
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
                batch_f[cnt] = label_binarize(hc_sub.features, classes=np.arange(len(feat_dc)))
                # get target locations
                out_pts_mask = (hc_sub.features == 0).squeeze()
                n_out_pts_actual = np.sum(out_pts_mask)
                idx = np.random.choice(n_out_pts_actual, n_out_pts_curr,
                                       replace=n_out_pts_actual < n_out_pts_curr)
                batch_out[cnt] = hc_sub.vertices[out_pts_mask][idx]
                # TODO: currently only supports type(out_point_label) = int
                batch_out_l[cnt] = hc_sub.labels[out_pts_mask][idx]
                assert -1 not in batch_out_l[cnt]
                cnt += 1
            assert cnt == batchsize
            yield (batch_f, batch), (batch_out, batch_out_l)


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
    if train and np.random.randint(0, 4) == 0:
        ctx_size_fluct = (np.random.randn(1)[0] * 0.1 + 0.7) * ctx_size
    else:
        ctx_size_fluct = ctx_size
    for curr_ssv_params in ssv_params:
        # do not write SSV mesh in case it does not exist (will be build from SV meshes)
        ssv = SuperSegmentationObject(mesh_caching=False, **curr_ssv_params)
        hc = _load_ssv_hc((ssv, tuple(feat_dc.keys()), tuple(feat_dc.values()), 'glia', None))
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
            # source_nodes = np.array(list(sparsify_skeleton_fast(hc.graph(), scal=np.array([1, 1, 1]), min_dist_thresh=base_node_dst,
            #                                       max_dist_thresh=base_node_dst, dot_prod_thresh=0).nodes()))
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
                # node_ids = bfs_vertices(hc, source_node, npoints_ssv)
                node_ids = context_splitting_v2(hc, source_node, ctx_size_fluct)
                hc_sub = extract_subset(hc, node_ids)[0]  # only pass HybridCloud
                sample_feats = hc_sub.features
                sample_pts = hc_sub.vertices
                # get target locations
                # ~1um apart
                # base_points = sparsify_skeleton_fast(hc_sub.graph(), scal=np.array([1, 1, 1]), min_dist_thresh=1000,
                #                                      max_dist_thresh=1000, dot_prod_thresh=0, verbose=False).nodes()
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
    res = dict(t_pts=out_pts_orig, t_l=np.concatenate(res), batch_progress=(batch_progress, n_batches))

    q_cnt.put(1. / n_batches)
    if batch_progress == 1:
        q_out.put(ssv_params)
    d_out[ssv_params['ssv_id']].append(res)


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
        if len(d_in[ssv_id]) < curr_ix + 1:
            time.sleep(0.5)
            continue
        # res: [(dict(t_pts=.., t_label, batch_process)]
        res = d_in[ssv_id][curr_ix]
        # el['t_l'] has shape (b, num_points, n_classes) -> (n_nodes, 1)
        node_preds.append(np.argmax(res['t_l'].reshape(-1, 2), axis=1)[..., None])
        # el['t_pts'] has shape (b, num_points, 3) -> (n_nodes, 3)
        node_coords.append(res['t_pts'].reshape(-1, 3))
        if res['batch_progress'][0] == res['batch_progress'][1]:
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
    d_in[ssv_id][curr_ix] = None
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
    if 'j0251' in mpath:
        n_classes = 10
    try:
        m = ModelNet40(5, n_classes, **mkwargs).to(device)
    except RuntimeError as e:
        if not mkwargs['use_bias']:
            mkwargs['use_bias'] = True
        else:
            raise RuntimeError(e)
        m = ModelNet40(5, 8, **mkwargs).to(device)
    m.load_state_dict(torch.load(mpath)['model_state_dict'])
    m.loader_kwargs = loader_kwargs
    return m.eval()


def get_tnet_model_pts(mpath: Optional[str] = None, device='cuda') -> 'InferenceModel':
    if mpath is None:
        mpath = global_params.config.mpath_tnet_pts
    from elektronn3.models.convpoint import ModelNet40, TripletNet
    mkwargs, loader_kwargs = get_pt_kwargs(mpath)
    m = TripletNet(ModelNet40(5, 10, **mkwargs).to(device))
    m.load_state_dict(torch.load(mpath)['model_state_dict'])
    m.loader_kwargs = loader_kwargs
    return m.eval()


# prediction wrapper
def predict_glia_ssv(ssv_params: List[dict], mpath: Optional[str] = None,
                     postproc_kwargs: Optional[dict] = None, **add_kwargs):
    """
    Perform glia predictions of cell reconstructions on sampled point sets from the
    cell's vertices. The number of predictions ``npreds`` per cell is calculated based on the
    fraction of the total number of vertices over ``npoints`` times two, but at least 5.
    Every point set is constructed by collecting the vertices associated with skeleton within a
    breadth-first search up to a maximum of ``npoints``.


    Args:
        ssv_params: List of kwargs to initialize SSVs.
        mpath: Path to model.
        postproc_kwargs: Postprocessing kwargs.

    Returns:

    """
    if mpath is None:
        mpath = global_params.config.mpath_glia_pts
    loader_kwargs = get_pt_kwargs(mpath)[1]
    default_kwargs = dict(nloader=10, npredictor=5, bs=10,
                          loader_kwargs=dict(n_out_pts=200, base_node_dst=loader_kwargs['ctx_size'] / 3))
    default_kwargs.update(add_kwargs)
    postproc_kwargs_def = global_params.config['points']['glia']['mapping']
    if postproc_kwargs is None:
        postproc_kwargs = {}
    postproc_kwargs_def.update(postproc_kwargs)
    out_dc = predict_pts_plain(ssv_params, get_glia_model_pts, pts_loader_local_skel, pts_pred_local_skel,
                               postproc_func=pts_postproc_glia, mpath=mpath, postproc_kwargs=postproc_kwargs_def,
                               **loader_kwargs, **default_kwargs)
    if not np.all(list(out_dc.values())) or len(out_dc) != len(ssv_params):
        raise ValueError('Invalid output during glia prediction.')


def infere_cell_morphology_ssd(ssv_params, mpath: Optional[str] = None, pred_key_appendix: str = '', **add_kwargs):
    """
    Extract local morphology embeddings of cell reconstructions on sampled point sets from the
    cell's vertices. The number of predictions ``npreds`` per cell is calculated based on the
    fraction of the total number of vertices over ``npoints`` times two, but at least 5.
    Every point set is constructed by collecting the vertices associated with skeleton within a
    breadth-first search up to a maximum of ``npoints``.


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
    default_kwargs = dict(nloader=10, npredictor=5, bs=10, loader_kwargs=dict(
        n_out_pts=1, base_node_dst=loader_kwargs['ctx_size'] / 2, use_syntype=True, use_subcell=True))
    postproc_kwargs = dict(pred_key=pred_key)
    default_kwargs.update(add_kwargs)
    out_dc = predict_pts_plain(ssv_params, get_tnet_model_pts, pts_loader_local_skel, pts_pred_embedding,
                               postproc_kwargs=postproc_kwargs, postproc_func=pts_postproc_embedding,
                               mpath=mpath, **loader_kwargs, **default_kwargs)
    if not np.all(list(out_dc.values())) or len(out_dc) != len(ssv_params):
        raise ValueError('Invalid output during glia prediction.')


def predict_celltype_ssd(ssd_kwargs, mpath: Optional[str] = None, ssv_ids: Optional[Iterable[int]] = None,
                         da_equals_tan: bool = True, pred_key: Optional[str] = None, **add_kwargs):
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
        da_equals_tan:
        pred_key:

    Returns:

    """
    if pred_key is None:
        pred_key = 'celltype_cnn_e3'
    if mpath is None:
        mpath = global_params.config.mpath_celltype_pts
    loader_kwargs = get_pt_kwargs(mpath)[1]
    default_kwargs = dict(nloader=10, npredictor=4, bs=10, redundancy=(25, 100))
    default_kwargs.update(add_kwargs)
    ssd = SuperSegmentationDataset(**ssd_kwargs)
    if ssv_ids is None:
        ssv_ids = ssd.ssv_ids
    out_dc = predict_pts_plain(ssd_kwargs, get_celltype_model_pts, pts_loader_scalar, pts_pred_scalar_nopostproc,
                               mpath=mpath, ssv_ids=ssv_ids, **loader_kwargs, **default_kwargs)
    for ssv in ssd.get_super_segmentation_object(out_dc.keys()):
        logit = np.concatenate(out_dc[ssv.id])
        if 'j0126' in ssd.config.working_dir and da_equals_tan:
            # accumulate evidence for DA and TAN
            logit[:, 1] += logit[:, 6]
            # remove TAN in proba array
            logit = np.delete(logit, [6], axis=1)
            # INT is now at index 6 -> label 6 is INT
        cls = np.argmax(logit, axis=1).squeeze()
        cls_maj = collections.Counter(cls).most_common(1)[0][0]
        ssv.save_attributes([pred_key], [cls_maj])
        ssv.save_attributes([f"{pred_key}_probas"], [logit])
    if not np.all(list(out_dc.values())) or len(out_dc) != len(ssv_ids):
        raise ValueError('Invalid output during cell type prediction.')


# -------------------------------------------- SSO TO MORPHX CONVERSION ---------------------------------------------#

def sso2ce(sso: SuperSegmentationObject, mi: bool = True, vc: bool = True,
           sy: bool = True, my: bool = False, my_avg: bool = True, mesh: bool = False) -> CloudEnsemble:
    """ Converts a SuperSegmentationObject into a CloudEnsemble (ce). Cell organelles are saved
        as additional cloud in the ce, named as in the function parameters (e.g. 'mi' for
        mitochondria). The no_pred (no prediction) flags of the ce are set for all additional
        clouds. Myelin is added in form of the types array of the HybridCloud, where myelinated
        vertices have type 1 and 0 otherwise.

    Args:
        sso: The SuperSegmentationObject which should get converted to a CloudEnsemble.
        mi: Flag for including mitochondria.
        vc: Flag for including vesicle clouds.
        sy: Flag for including synapses.
        my: Flag for including myelin.
        my_avg: Flag for applying majority vote on myelin property.
        mesh: Flag for storing all objects as HybridMesh objects with additional faces.

    Returns:
        CloudEnsemble object as described above.
    """
    # convert cell organelle meshes
    clouds = {}
    if mi:
        indices, vertices, normals = sso.mi_mesh
        if mesh:
            clouds['mi'] = HybridMesh(vertices=vertices.reshape((-1, 3)), faces=indices.reshape((-1, 3)))
        else:
            clouds['mi'] = PointCloud(vertices=vertices.reshape((-1, 3)))
    if vc:
        indices, vertices, normals = sso.vc_mesh
        if mesh:
            clouds['vc'] = HybridMesh(vertices=vertices.reshape((-1, 3)), faces=indices.reshape((-1, 3)))
        else:
            clouds['vc'] = PointCloud(vertices=vertices.reshape((-1, 3)))
    if sy:
        indices, vertices, normals = sso._load_obj_mesh('syn_ssv', rewrite=False)
        if mesh:
            clouds['sy'] = HybridMesh(vertices=vertices.reshape((-1, 3)), faces=indices.reshape((-1, 3)))
        else:
            clouds['sy'] = PointCloud(vertices=vertices.reshape((-1, 3)))
    # convert cell mesh
    indices, vertices, normals = sso.mesh
    sso.load_skeleton()
    if mesh:
        hm = HybridMesh(vertices=vertices.reshape((-1, 3)), faces=indices.reshape((-1, 3)),
                        nodes=sso.skeleton['nodes'] * sso.scaling, edges=sso.skeleton['edges'])
    else:
        hm = HybridCloud(vertices=vertices.reshape((-1, 3)), nodes=sso.skeleton['nodes'] * sso.scaling,
                         edges=sso.skeleton['edges'])
    # merge all clouds into a CloudEnsemble
    ce = CloudEnsemble(clouds, hm, no_pred=[obj for obj in clouds])
    if my:
        add_myelin(sso, hm, average=my_avg)
    return ce


def sso2hc(sso: SuperSegmentationObject, mi: bool = True, vc: bool = True,
           sy: bool = True, my: bool = False, my_avg: bool = True) -> HybridCloud:
    """ Converts a SuperSegmentationObject into a HybridCloud (hc). The object boundaries
        are stored in the obj_bounds attribute of the hc. The no_pred (no prediction) flags
        are set for all cell organelles. Myelin is added in form of the types array of the
        hc, where myelinated vertices have type 1 and 0 otherwise.

    Args:
        sso: The SuperSegmentationObject which should get converted to a CloudEnsemble.
        mi: Flag for including mitochondria.
        vc: Flag for including vesicle clouds.
        sy: Flag for including synapses.
        my: Flag for including myelin.
        my_avg: Flag for applying majority vote on myelin property.

    Returns:
        HybridCloud object as described above.
    """
    vertex_num = 0
    # convert cell organelle meshes
    clouds = []
    obj_names = []
    if mi:
        indices, vertices, normals = sso.mi_mesh
        clouds.append(vertices.reshape((-1, 3)))
        obj_names.append('mi')
        vertex_num += len(vertices.reshape((-1, 3)))
    if vc:
        indices, vertices, normals = sso.vc_mesh
        clouds.append(vertices.reshape((-1, 3)))
        obj_names.append('vc')
        vertex_num += len(vertices.reshape((-1, 3)))
    if sy:
        indices, vertices, normals = sso._load_obj_mesh('syn_ssv', rewrite=False)
        clouds.append(vertices.reshape((-1, 3)))
        obj_names.append('sy')
        vertex_num += len(vertices.reshape((-1, 3)))
    # convert cell mesh
    indices, vertices, normals = sso.mesh
    hc_vertices = vertices.reshape((-1, 3))
    vertex_num += len(hc_vertices)
    # merge all clouds into a HybridCloud
    total_verts = np.zeros((vertex_num, 3))
    bound = len(hc_vertices)
    obj_bounds = {'hc': [0, bound]}
    total_verts[0:bound] = hc_vertices
    for ix, cloud in enumerate(clouds):
        if len(cloud) == 0:
            # ignore cell organelles with zero vertices
            continue
        obj_bounds[obj_names[ix]] = [bound, bound + len(cloud)]
        total_verts[bound:bound + len(cloud)] = cloud
        bound += len(cloud)
    sso.load_skeleton()
    hc = HybridCloud(vertices=total_verts, nodes=sso.skeleton['nodes'] * sso.scaling, edges=sso.skeleton['edges'],
                     obj_bounds=obj_bounds, no_pred=obj_names)
    if my:
        add_myelin(sso, hc, average=my_avg)
    return hc


def add_myelin(sso: SuperSegmentationObject, hc: HybridCloud, average: bool = True):
    """ Tranfers myelin prediction from a SuperSegmentationObject to an existing
        HybridCloud (hc). Myelin is added in form of the types array of the hc,
        where myelinated vertices have type 1 and 0 otherwise. Works in-place.

    Args:
        sso: SuperSegmentationObject which contains skeleton to which myelin should get mapped.
        hc: HybridCloud to which myelin should get added.
        average: Flag for applying majority vote to the myelin property
    """
    sso.skeleton['myelin'] = map_myelin2coords(sso.skeleton["nodes"], mag=4)
    if average:
        majorityvote_skeleton_property(sso, 'myelin')
        myelinated = sso.skeleton['myelin_avg10000']
    else:
        myelinated = sso.skeleton['myelin']
    nodes_idcs = np.arange(len(hc.nodes))
    myel_nodes = nodes_idcs[myelinated.astype(bool)]
    myel_vertices = []
    for node in myel_nodes:
        myel_vertices.extend(hc.verts2node[node])
    # myelinated vertices get type 1, not myelinated vertices get type 0
    types = np.zeros(len(hc.vertices))
    types[myel_vertices] = 1
    hc.set_types(types)
