# SyConn-dev
# Copyright (c) 2016 Philipp J. Schubert
# All rights reserved

# import here, otherwise it might fail if it is imported after importing torch
# see https://github.com/pytorch/pytorch/issues/19739
import open3d as o3d
import collections
from typing import Iterable, Union, Optional, Any, Tuple, Callable, List
from multiprocessing import Process, Queue
from syconn.handler.basics import chunkify_successive
import numpy as np
import time
import tqdm
import tqdm
import torch
from scipy import spatial
import morphx.processing.clouds as clouds
import functools
from morphx.classes.hybridcloud import HybridCloud
import networkx as nx
from sklearn.preprocessing import label_binarize
from morphx.processing.hybrids import extract_subset
from morphx.processing.objects import bfs_vertices, context_splitting, context_splitting_v2
from morphx.processing.graphs import bfs_num
from syconn.reps.super_segmentation import SuperSegmentationDataset
from syconn.reps.super_segmentation_helper import sparsify_skeleton_fast

# TODO: specify further, add to config
pts_feat_dict = dict(sv=0, mi=1, syn_ssv=3, syn_ssv_sym=3, syn_ssv_asym=4, vc=2)
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


def write_pts_ply(fname: str, pts: np.ndarray, feats: np.ndarray,
                  binarized=False):
    assert pts.ndim == 2
    assert feats.ndim == 2
    pts = np.asarray(pts)
    feats = np.asarray(feats)
    col_dc = {0: [[200, 200, 200]], 1: [[100, 100, 200]], 3: [[200, 100, 200]],
              4: [[250, 100, 100]], 2: [[100, 200, 100]], 5: [[100, 200, 200]],
              6: [0, 0, 0]}
    if not binarized:
        feats = label_binarize(feats, np.arange(np.max(feats)+1))
    cols = np.zeros(pts.shape, dtype=np.uint8)
    for k in range(feats.shape[1]):
        mask = feats[:, k] == 1
        cols[mask] = col_dc[k]
    write_ply(fname, pts, cols)


def worker_postproc(q_out: Queue, q_postproc: Queue,
                    postproc_func: Callable, postproc_kwargs: dict):
    """

    Args:
        q_out:
        q_postproc:
        postproc_func:
        postproc_kwargs:
    """
    stop_received = False
    while True:
        if not q_postproc.empty():
            inp = q_postproc.get()
            if inp == 'STOP':
                if stop_received:
                    # already got STOP signal, put back in queue for other worker.
                    q_postproc.put('STOP')
                    # wait for the other worker to get the signal
                    time.sleep(2)
                    continue
                stop_received = True
                if not q_postproc.empty():
                    continue
                # put another stop to the queue. Global stop is only triggered by a single stop signal
                q_postproc.put('STOP')
                break
        else:
            if stop_received:
                # put another stop to the queue. Global stop is only triggered by a single stop signal
                q_postproc.put('STOP')
                break
            time.sleep(0.5)
            continue
        q_out.put(postproc_func(inp, **postproc_kwargs))
    q_out.put('END')


def worker_pred(q_out: Queue, q_cnt: Queue, q_in: Queue,
                model_loader: Callable, pred_func: Callable,
                mkwargs: dict, device: str,
                bs: Optional[int] = None):
    """

    Args:
        q_out:
        q_cnt:
        q_in:
        model_loader:
        pred_func:
        mkwargs:
        device:
        bs:
    """
    m = model_loader(mkwargs, device)
    stop_received = False
    while True:
        if not q_in.empty():
            inp = q_in.get()
            if inp == 'STOP':
                if stop_received:
                    # already got STOP signal, put back in queue for other worker.
                    q_in.put('STOP')
                    # wait for the other worker to get the signal
                    time.sleep(2)
                    continue
                stop_received = True
                if not q_in.empty():
                    continue
                break
        else:
            if stop_received:
                break
            time.sleep(0.5)
            continue
        pred_func(m, inp, q_out, q_cnt, device, bs)
    print('Predicter done.')
    q_out.put('STOP')


def worker_load(q_loader: Queue, q_out: Queue, q_loader_sync: Queue, loader_func: Callable):
    """

    Args:
        q_loader:
        q_out:
        q_loader_sync:
        loader_func:
        kwargs:
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
    print('Loader done.')
    q_loader_sync.put('DONE')


def listener(q_cnt: Queue, q_in: Queue, q_loader_sync: Queue,
             npredictor: int, nloader: int, total: int):
    """

    Args:
        q_cnt:
        q_in:
        q_loader_sync:
        npredictor:
        nloader:
        total:
    """
    pbar = tqdm.tqdm(total=total)
    cnt_loder_done = 0
    while True:
        if q_cnt.empty():
            time.sleep(0.2)
        else:
            res = q_cnt.get()
            if res is None:  # final stop
                assert cnt_loder_done == nloader
                pbar.close()
                break
            pbar.update(res)
        if q_loader_sync.empty() or cnt_loder_done == nloader:
            time.sleep(0.2)
        else:
            _ = q_loader_sync.get()
            cnt_loder_done += 1
            if cnt_loder_done == nloader:
                for _ in range(npredictor):
                    time.sleep(0.2)
                    q_in.put('STOP')


def predict_pts_plain(ssd_kwargs: dict, model_loader: Callable,
                      loader_func: Callable, pred_func: Callable,
                      npoints: int, scale_fact: float, ctx_size: int,
                      postproc_func: Optional[Callable] = None,
                      postproc_kwargs: Optional[dict] = None,
                      output_func: Optional[Callable] = None,
                      mkwargs: Optional[dict] = None,
                      nloader: int = 4, npredictor: int = 2, npostptroc: int = 1,
                      ssv_ids: Optional[Union[list, np.ndarray]] = None,
                      use_test_aug: bool = False,
                      device: str = 'cuda', bs: int = 40,
                      loader_kwargs: Optional[dict] = None) -> dict:
    """
    # TODO: Use 'mode' kwarg to switch between point2scalar, point2points for skel and surface classifaction.
    # TODO: remove quick-fix with ssd_kwargs vs. ssv_ids kwargs

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
        ssd_kwargs: Keyword arguments to specify the underlying ``SuperSegmentationDataset``.
        model_loader: Function which returns the pytorch model object.
        mkwargs: Model keyword arguments used in `model_loader` call.
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
        def postproc_func(x): return x
    if postproc_kwargs is None:
        postproc_kwargs = dict()

    transform = [clouds.Normalization(scale_fact), clouds.Center()]
    if use_test_aug:
        transform = [clouds.RandomVariation((-5, 5), distr='normal')] + transform + [clouds.RandomRotate()]
    transform = clouds.Compose(transform)

    if type(ssd_kwargs) is dict:
        kwargs = dict(batchsize=bs, npoints=npoints, ssd_kwargs=ssd_kwargs,
                      transform=transform, ctx_size=ctx_size, **loader_kwargs)
        ssd = SuperSegmentationDataset(**ssd_kwargs)
        if ssv_ids is None:
            ssv_ids = ssd.ssv_ids
        # minimum redundanc
        min_redundancy = 25
        # three times as many predictions as npoints fit into the ssv vertices
        ssv_redundancy = [max(len(ssv.mesh[1]) // 3 // npoints * 3, min_redundancy) for ssv in
                          ssd.get_super_segmentation_object(ssv_ids)]
        ssv_ids = np.concatenate([np.array([ssv_ids[ii]] * ssv_redundancy[ii], dtype=np.uint)
                                  for ii in range(len(ssv_ids))])
        params_in = [{**kwargs, **dict(ssv_ids=[ch])} for ch in ssv_ids]
    else:
        kwargs = dict(batchsize=bs, npoints=npoints,
                      transform=transform, ctx_size=ctx_size, **loader_kwargs)
        params_in = [{**kwargs, **dict(ssv_params=[ch])} for ch in ssd_kwargs]

    # total samples:
    nsamples_tot = len(ssv_ids)

    q_loader = Queue()
    for el in params_in:
        q_loader.put(el)
    q_in = Queue(maxsize=20*npredictor)
    q_cnt = Queue()
    q_out = Queue()
    q_postproc = Queue()
    q_loader_sync = Queue()
    producers = [Process(target=worker_load, args=(q_loader, q_in, q_loader_sync, loader_func)) for _ in range(nloader)]
    for p in producers:
        p.start()
    consumers = [Process(target=worker_pred, args=(q_postproc, q_cnt, q_in, model_loader, pred_func,
                                                   mkwargs, device, bs)) for _ in range(npredictor)]
    for c in consumers:
        c.start()
    postprocs = [Process(target=worker_postproc, args=(q_out, q_postproc, postproc_func, postproc_kwargs
                                                       )) for _ in range(npostptroc)]
    for c in postprocs:
        c.start()
    dict_out = collections.defaultdict(list)
    cnt_end = 0
    lsnr = Process(target=listener, args=(q_cnt, q_in, q_loader_sync, npredictor,
                                          nloader, nsamples_tot))
    lsnr.start()
    while True:
        if q_out.empty():
            if cnt_end == npostptroc:
                break
            time.sleep(1)
            continue
        res = q_out.get()
        if res == 'END':
            cnt_end += 1
            continue
        output_func(dict_out, res)

    print('Finished collection of results.')
    q_cnt.put(None)
    lsnr.join()
    print('Joined listener.')
    for p in producers:
        p.join()
        p.close()
    print('Joined producers.')
    for c in consumers:
        c.join()
        c.close()
    print('Joined consumers.')
    for c in postprocs:
        c.join()
        c.close()
    print('Joined post-processor.')
    return dict_out


def generate_pts_sample(sample_pts: dict, feat_dc: dict, cellshape_only: bool,
                        num_obj_types: int, onehot: bool = True,
                        npoints: Optional[int] = None, use_syntype: bool = True,
                        downsample: Optional[float] = None):
    # TODO: add optional downsampling here
    feat_dc = dict(feat_dc)
    if use_syntype:
        if 'syn_ssv' in feat_dc:
            del feat_dc['syn_ssv']
    else:
        if 'syn_ssv_sym' in feat_dc:
            del feat_dc['syn_ssv_sym']
        if 'syn_ssv_asym' in feat_dc:
            del feat_dc['syn_ssv_asym']
    if cellshape_only is True:
        sample_pts = sample_pts['sv']
        sample_feats = np.ones((len(sample_pts), 1)) * feat_dc['sv']
    else:
        sample_feats = np.concatenate([[feat_dc[k]] * len(sample_pts[k])
                                       for k in feat_dc.keys()])
        if onehot:
            sample_feats = label_binarize(sample_feats, classes=np.arange(num_obj_types))
        else:
            sample_feats = sample_feats[..., None]
        # len(sample_feats) is the equal to the total number of vertices
        sample_pts = np.concatenate([sample_pts[k] for k in feat_dc.keys()])
    if npoints is not None:
        idx_arr = np.random.choice(np.arange(len(sample_pts)),
                                   npoints, replace=len(sample_pts) < npoints)
        sample_pts = sample_pts[idx_arr]
        sample_feats = sample_feats[idx_arr]
    return sample_pts, sample_feats


@functools.lru_cache(256)
def _load_ssv_hc(args):
    ssv, feats, feat_labels, pt_type, radius = args
    vert_dc = dict()
    # TODO: replace by poisson disk sampling
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


@functools.lru_cache(maxsize=128)
def load_hc_pkl(path: str, gt_type: str,
                radius: Optional[float] = None) -> HybridCloud:
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


def pts_loader_scalar(ssd_kwargs: dict, ssv_ids: Union[list, np.ndarray],
                      batchsize: int, npoints: int, ctx_size: float,
                      transform: Optional[Callable] = None,
                      train: bool = False, draw_local: bool = False,
                      draw_local_dist: int = 1000,
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
            if draw_local and occ % 2:
                raise ValueError(f'draw_local is set to True but the number of SSV samples is uneven.')
            ssv = ssd.get_super_segmentation_object(ssv_id)
            hc = _load_ssv_hc((ssv, tuple(feat_dc.keys()), tuple(
                feat_dc.values()), 'celltype', None))
            npoints_ssv = min(len(hc.vertices), npoints)
            batch = np.zeros((occ, npoints_ssv, 3))
            batch_f = np.zeros((occ, npoints_ssv, len(feat_dc)))
            ixs = np.ones((occ,), dtype=np.uint) * ssv_id
            cnt = 0
            # TODO: this should be deterministic during inference
            # nodes = hc.base_points(threshold=5000, source=len(hc.nodes) // 2)
            nodes = sparsify_skeleton_fast(hc.graph(), scal=np.array([1, 1, 1]), min_dist_thresh=5000,
                                           max_dist_thresh=5000, dot_prod_thresh=0).nodes()
            source_nodes = np.random.choice(nodes, occ, replace=len(nodes) < occ)
            for source_node in source_nodes:
                # local_bfs = bfs_vertices(hc, source_node, npoints_ssv)
                local_bfs = context_splitting_v2(hc, source_node, ctx_size)
                pc = extract_subset(hc, local_bfs)[0]  # only pass PointCloud
                sample_feats = pc.features
                sample_pts = pc.vertices
                # make sure there is always the same number of points within a batch
                sample_ixs = np.arange(len(sample_pts))
                sample_pts = sample_pts[sample_ixs][:npoints_ssv]
                sample_feats = sample_feats[sample_ixs][:npoints_ssv]
                npoints_add = npoints_ssv - len(sample_pts)
                idx = np.random.choice(np.arange(len(sample_pts)), npoints_add)
                sample_pts = np.concatenate([sample_pts, sample_pts[idx]])
                sample_feats = np.concatenate([sample_feats, sample_feats[idx]])
                # one hot encoding
                sample_feats = label_binarize(sample_feats, classes=np.arange(len(feat_dc)))
                pc._vertices = sample_pts
                pc._features = sample_feats
                if transform is not None:
                    transform(pc)
                batch[cnt] = pc.vertices
                batch_f[cnt] = pc.features
                cnt += 1
            assert cnt == occ
            yield ixs, (batch_f, batch)
    else:
        ssv_ids = np.unique(ssv_ids)
        # fluctuate context size in 1/4 samples
        if np.random.randint(0, 4) == 0:
            ctx_size_fluct = max((np.random.randn(1)[0] * 0.1 + 0.7), 0.33) * ctx_size
        else:
            ctx_size_fluct = ctx_size
        for curr_ssvid in ssv_ids:
            ssv = ssd.get_super_segmentation_object(curr_ssvid)
            hc = _load_ssv_hc((ssv, tuple(feat_dc.keys()), tuple(
                feat_dc.values()), 'celltype', None))
            npoints_ssv = min(len(hc.vertices), npoints)
            # add a +-10% fluctuation in the number of input points
            npoints_add = np.random.randint(-int(npoints_ssv * 0.1), int(npoints_ssv * 0.1))
            npoints_ssv += npoints_add
            batch = np.zeros((batchsize, npoints_ssv, 3))
            batch_f = np.zeros((batchsize, npoints_ssv, len(feat_dc)))
            ixs = np.ones((batchsize,), dtype=np.uint) * ssv.id
            cnt = 0
            source_nodes = np.random.choice(np.arange(len(hc.nodes)), batchsize,
                                            replace=len(hc.nodes) < batchsize)
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
                local_bfs = context_splitting_v2(hc, source_node, ctx_size_fluct)
                pc = extract_subset(hc, local_bfs)[0]  # only pass PointCloud
                sample_feats = pc.features
                sample_pts = pc.vertices
                # shuffling
                sample_ixs = np.arange(len(sample_pts))
                np.random.shuffle(sample_ixs)
                sample_pts = sample_pts[sample_ixs][:npoints_ssv]
                sample_feats = sample_feats[sample_ixs][:npoints_ssv]
                # add duplicated points before applying the transform if sample_pts
                # has less points than npoints_ssv
                npoints_add = npoints_ssv - len(sample_pts)
                idx = np.random.choice(np.arange(len(sample_pts)), npoints_add)
                sample_pts = np.concatenate([sample_pts, sample_pts[idx]])
                sample_feats = np.concatenate([sample_feats, sample_feats[idx]])
                # one hot encoding
                sample_feats = label_binarize(sample_feats, classes=np.arange(len(feat_dc)))
                pc._vertices = sample_pts
                pc._features = sample_feats
                if transform is not None:
                    transform(pc)
                batch[cnt] = pc.vertices
                batch_f[cnt] = pc.features
                cnt += 1
            assert cnt == batchsize
            yield ixs, (batch_f, batch)


def pts_pred_scalar(m, inp, q_out, q_cnt, device, bs):
    """

    Args:
        m: Model instance.
        inp: Input as given by the loader_func.
        q_out:
        q_cnt:
        device:
        bs:

    Returns:

    """
    # TODO: is it possible to get 'device' directly from model 'm'?
    ssv_ids, inp = inp
    res = []
    n_samples = len(inp[0])
    for ii in range(0, int(np.ceil(n_samples / bs))):
        low = bs * ii
        high = bs * (ii + 1)
        with torch.no_grad():
            g_inp = [torch.from_numpy(i[low:high]).to(device).float() for i in inp]
            out = m(*g_inp).cpu().numpy()
        res.append(out)

    del inp
    q_cnt.put(len(ssv_ids))
    q_out.put((ssv_ids, res))


def pts_loader_glia(ssv_params: Optional[List[Tuple[int, dict]]] = None,
                    out_point_label: Optional[List[Union[str, int]]] = None,
                    batchsize: Optional[int] = None, npoints: Optional[int] = None,
                    ctx_size: Optional[float] = None,
                    transform: Optional[Callable] = None,
                    n_out_pts: int = 100, train=False,
                    base_node_dst: float = 10000,
                    use_subcell: bool = False,
                    ssd_kwargs: Optional[dict] = None
                    ) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Generator for SSV point cloud samples of size `npoints`. Currently used for
    local point-to-scalar tasks, e.g. morphology embeddings or glia detection.

    Args:
        ssv_params: SuperSegmentationObject IDs and SSD kwargs for which samples are generated.
        out_point_label: Either key for sso.skeleton attribute or int (used for all out locations).
        batchsize: Only used during training.
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
        ssd = SuperSegmentationDataset(**curr_ssv_params[1])
        ssv = ssd.get_super_segmentation_object(curr_ssv_params[0])
        hc = _load_ssv_hc((ssv, tuple(feat_dc.keys()), tuple(
            feat_dc.values()), 'glia', None))
        npoints_ssv = min(len(hc.vertices), npoints)
        # add a +-10% fluctuation in the number of input and output points
        npoints_add = np.random.randint(-int(n_out_pts * 0.1), int(n_out_pts * 0.1))
        n_out_pts_curr = n_out_pts + npoints_add
        npoints_add = np.random.randint(-int(npoints_ssv * 0.1), int(npoints_ssv * 0.1))
        npoints_ssv += npoints_add
        if train:
            source_nodes = np.random.choice(np.arange(len(hc.nodes)), batchsize,
                                            replace=len(hc.nodes) < batchsize)
        else:
            # source_nodes = hc.base_points(threshold=base_node_dst, source=len(hc.nodes) // 2)
            source_nodes = np.array(list(sparsify_skeleton_fast(hc.graph(), scal=np.array([1, 1, 1]), min_dist_thresh=base_node_dst,
                                                  max_dist_thresh=base_node_dst, dot_prod_thresh=0).nodes()))
            batchsize = min(len(source_nodes), batchsize)
        n_batches = int(np.ceil(len(source_nodes) / batchsize))
        if len(source_nodes) % batchsize != 0:
            source_nodes = np.concatenate([np.random.choice(source_nodes, batchsize - len(source_nodes) % batchsize), source_nodes])
        for ii in range(n_batches):
            print(len(source_nodes), batchsize, ssv)
            batch = np.zeros((batchsize, npoints_ssv, 3))
            batch_f = np.zeros((batchsize, npoints_ssv, len(feat_dc)))
            batch_out = np.zeros((batchsize, n_out_pts_curr, 3))
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
                base_points = sparsify_skeleton_fast(hc_sub.graph(), scal=np.array([1, 1, 1]), min_dist_thresh=1000,
                                                     max_dist_thresh=1000, dot_prod_thresh=0, verbose=False).nodes()
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
                idx = np.random.choice(np.arange(len(sample_pts)), npoints_add)
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
                    batch_out_orig[cnt] = out_coords
                # TODO: currently only supports type(out_point_label) = int
                batch_out_l[cnt] = out_point_label
                cnt += 1
            assert cnt == batchsize
            if not train:
                batch_process = (ii+1)/n_batches
                yield ssv.id, (batch_f, batch, batch_out), batch_out_orig, batch_process, n_batches
            else:
                yield ssv.id, (batch_f, batch), (batch_out, batch_out_l)


def pts_pred_glia(m, inp, q_out, q_cnt, device, bs):
    """

    Args:
        m: Model instance.
        inp: Input as given by the loader_func.
        q_out:
        q_cnt:
        device:
        bs:

    Returns:

    """
    # TODO: is it possible to get 'device' directly from model 'm'?
    ssv_id, model_inp,  out_pts_orig, batch_process, n_batches = inp
    with torch.no_grad():
        g_inp = [torch.from_numpy(i).to(device).float() for i
                 in model_inp]
        out = m(*g_inp).cpu().numpy()
    res = dict(t_pts=out_pts_orig, t_l=out, b_process=batch_process)

    del inp
    q_cnt.put(1./n_batches)
    q_out.put(([ssv_id], [res]))


def pts_loader_semseg(ssd_kwargs: dict, ssv_ids: Union[list, np.ndarray],
                      batchsize: int, npoints: int,
                      transform: Optional[Callable] = None,
                      train: bool = False, draw_local: bool = False,
                      draw_local_dist: int = 1000
                      ) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Generator for SSV point cloud samples of size `npoints`.

    Args:
        ssd_kwargs: SuperSegmentationDataset keyword arguments specifying e.g.
            working directory, version, etc.
        ssv_ids: SuperSegmentationObject IDs for which samples are generated.
        batchsize: Only used during training.
        npoints: Number of points used to generate sample context.
        transform: Transformation/agumentation applied to every sample.
        train: If false, eval mode -> batch size won't be used.
        draw_local: Will draw similar contexts from approx.
            the same location, requires a single unique element in ssv_ids
        draw_local_dist: Maximum distance to similar source node in nm.

    Yields: SSV IDs [M, ], (point feature [N, C], point location [N, 3])

    """
    raise NotImplementedError('TBD')


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
        # cell/sv vertices proportional to total npoints
        n_out_pts = int(np.sum(hc.features == 0) / len(hc.vertices) * npoints_ssv)
        # # with 1/4 draw a small sample during training (only if sample has a minimum number of vertices - 0.75*npoints)
        # if np.random.randint(0, 4) == 1 and npoints_ssv / npoints > 0.75:
        #     fluct = np.random.randint(-int(n_out_pts * 0.8), 0) / n_out_pts
        #     npoints_add = int(fluct * n_out_pts)
        #     n_out_pts_curr = n_out_pts + npoints_add
        #     # randomize the fluctuation with +-std=0.1 and add those number of vertices
        #     npoints_add = int((1 + np.random.randn(1)[0] * 0.1) * fluct * npoints_ssv)
        #     # lower bound of 2k points due to non-truncated Gauss distribution
        #     npoints_ssv = max(2000, npoints_ssv + npoints_add)
        # else:
        # add a +-10% fluctuation in the number of input and output points
        npoints_add = np.random.randint(-int(n_out_pts * 0.1), int(n_out_pts * 0.1))
        n_out_pts_curr = n_out_pts + npoints_add
        npoints_add = np.random.randint(-int(npoints_ssv * 0.1), int(npoints_ssv * 0.1))
        npoints_ssv += npoints_add
        source_nodes = np.where(hc.node_labels == 1)[0]
        source_nodes = np.random.choice(len(source_nodes), batchsize,
                                        replace=len(source_nodes) < batchsize)
        n_batches = int(np.ceil(len(source_nodes) / batchsize))
        if len(source_nodes) % batchsize != 0:
            source_nodes = np.random.choice(source_nodes, batchsize * n_batches)
        dt_context = 0
        for ii in range(n_batches):
            batch = np.zeros((batchsize, npoints_ssv, 3))
            batch_f = np.zeros((batchsize, npoints_ssv, len(feat_dc)))
            batch_out = np.zeros((batchsize, n_out_pts_curr, 3))
            batch_out_l = np.zeros((batchsize, n_out_pts_curr, 1))
            cnt = 0
            for source_node in source_nodes[ii::n_batches]:
                # create local context
                # node_ids = bfs_vertices(hc, source_node, npoints_ssv)
                start = time.time()
                node_ids = context_splitting_v2(hc, source_node, ctx_size_fluct)
                dt_context += time.time() - start
                # print(f'DT context: {dt_context:.2f} s vs DT overall: {(time.time()-start_overall):.2f} s')
                hc_sub = extract_subset(hc, node_ids)[0]  # only pass HybridCloud
                sample_feats = hc_sub.features
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
                idx = np.random.choice(np.arange(len(sample_pts)), npoints_add)
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
