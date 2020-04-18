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
import morphx.processing.clouds as clouds
import functools
from morphx.classes.hybridcloud import HybridCloud
import networkx as nx
from sklearn.preprocessing import label_binarize
from morphx.processing.hybrids import extract_subset
from morphx.processing.objects import bfs_vertices
from morphx.processing.graphs import bfs_num
from syconn.reps.super_segmentation import SuperSegmentationDataset

# TODO: specify further, add to config
pts_feat_dict = dict(sv=0, mi=1, syn_ssv=3, syn_ssv_sym=3, syn_ssv_asym=4, vc=2)
# in nm, should be replaced by Poisson disk sampling
pts_feat_ds_dict = dict(sv=80, mi=100, syn_ssv=100, syn_ssv_sym=100, syn_ssv_asym=100, vc=100)


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


def write_pts_ply(fname: str, pts: np.ndarray, feats: np.ndarray):
    assert pts.ndim == 2
    assert feats.ndim == 2
    col_dc = {0: [[200, 200, 200]], 1: [[100, 100, 200]], 3: [[200, 100, 200]],
              4: [[250, 100, 100]], 2: [[100, 200, 100]]}
    cols = np.zeros(pts.shape, dtype=np.uint8)
    for k in range(feats.shape[1]):
        mask = feats[:, k] == 1
        cols[mask] = col_dc[k]
    write_ply(fname, pts, cols)


def worker_pred(q_out: Queue, q_cnt: Queue, q_in: Queue,
                model_loader: Callable, pred_func: Callable,
                mkwargs: dict, device: str):
    """

    Args:
        q_out:
        q_cnt:
        q_in:
        model_loader:
        pred_func:
        mkwargs:
        device:
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
        pred_func(m, inp, q_out, q_cnt, device)
    q_out.put('END')


def worker_load(q: Queue, q_loader_sync: Queue, loader_func: Callable,
                kwargs: dict):
    """

    Args:
        q:
        q_loader_sync:
        loader_func:
        kwargs:
    """
    for el in loader_func(**kwargs):
        while True:
            if q.full():
                time.sleep(1)
            else:
                break
        q.put(el)
    time.sleep(1)
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
            time.sleep(1)
        else:
            res = q_cnt.get()
            if res is None:  # final stop
                assert cnt_loder_done == nloader
                pbar.close()
                break
            pbar.update(res)
        if q_loader_sync.empty() or cnt_loder_done == nloader:
            time.sleep(1)
        else:
            _ = q_loader_sync.get()
            cnt_loder_done += 1
            print('Loader finished.')
            if cnt_loder_done == nloader:
                for _ in range(npredictor):
                    time.sleep(1)
                    q_in.put('STOP')


def predict_pts_plain(ssd_kwargs: dict, model_loader: Callable,
                      loader_func: Callable, pred_func: Callable,
                      npoints: int, scale_fact: float,
                      output_func: Optional[Callable] = None,
                      mkwargs: Optional[dict] = None,
                      nloader: int = 4, npredictor: int = 2,
                      ssv_ids: Optional[Union[list, np.ndarray]] = None,
                      use_test_aug: bool = False,
                      device: str = 'cuda') -> dict:
    """
    Perform cell type predictions of cell reconstructions on sampled point sets from the
    cell's vertices. The number of predictions `npreds` per cell is calculated based on the
    fraction of the total number of vertices over `npoints` times two, but at least 5.
    Every point set is constructed by collecting the vertices associated with skeleton within a
    breadth-first search up to a maximum of `npoints`.

    Overview:
        * loader (load_func) -> input queue
        * input queue -> prediction worker
        * prediction worker (pred_func) -> output queue
        * output queue -> result dictionary (return)

    Args:
        ssd_kwargs: Keyword arguments to specify the underlying ``SuperSegmentationDataset``.
        model_loader: Function which returns the pytorch model object.
        loader_func: Loader function, used by `nloader` workers retrieving samples.
        pred_func: Predict function, used by `npredictor` workers performing the inference.
        mkwargs: Model keyword arguments.
        npoints: Number of points used to generate a sample.
        scale_fact: Scale factor; used to normalize point clouds prior to model inference.
        output_func: Transforms the elements in the output queue and stores it in the final dictionary.
            If None, elements as returned by `pred_func` are assumed to be of form ``(ssv_ids, results)``:

                def output_func(res_dc, (ssv_ids, predictions)):
                    for ssv_id, ssv_pred in zip(*(ssv_ids, predictions)):
                        res_dc[ssv_id].append(ssv_pred)

        nloader: Number of workers loading samples from the given cell IDs via `load_func`.
        npredictor: Number of workers which will call `model_loader` and process (via `pred_func`) the output of
            the "loaders", i.e. workers which retrieve samples via `loader_func`.
        ssv_ids: IDs of cells to predict.
        use_test_aug: Use test-time augmentations. Currently this adds the following transformation
            to the basic transforms:

                transform = [clouds.Normalization(scale_fact), clouds.Center()]
                [clouds.RandomVariation((-5, 5), distr='normal')] + transform + [clouds.RandomRotate()]

        device: pytorch device.

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
    if output_func is None:
        def output_func(res_dc, ret):
            for ix, out in zip(*ret):
                res_dc[ix].append(out)
    transform = [clouds.Normalization(scale_fact), clouds.Center()]
    if use_test_aug:
        transform = [clouds.RandomVariation((-5, 5), distr='normal')] + transform + [clouds.RandomRotate()]
    transform = clouds.Compose(transform)
    bs = 40  # ignored during inference
    ssd = SuperSegmentationDataset(**ssd_kwargs)
    if ssv_ids is None:
        ssv_ids = ssd.ssv_ids
    # minimum redundanc
    min_redundancy = 25
    # three times as many predictions as npoints fit into the ssv vertices
    ssv_redundancy = [max(len(ssv.mesh[1]) // 3 // npoints * 3, min_redundancy) for ssv in
                      ssd.get_super_segmentation_object(ssv_ids)]
    kwargs = dict(batchsize=bs, npoints=npoints, ssd_kwargs=ssd_kwargs, transform=transform)
    ssv_ids = np.concatenate([np.array([ssv_ids[ii]] * ssv_redundancy[ii], dtype=np.uint)
                              for ii in range(len(ssv_ids))])
    params_in = [{**kwargs, **dict(ssv_ids=ch)} for ch in chunkify_successive(
        ssv_ids, int(np.ceil(len(ssv_ids) / nloader)))]

    # total samples:
    nsamples_tot = len(ssv_ids)

    q_in = Queue(maxsize=20*npredictor)
    q_cnt = Queue()
    q_out = Queue()
    q_loader_sync = Queue()
    producers = [Process(target=worker_load, args=(q_in, q_loader_sync, loader_func, el))
                 for el in params_in]
    for p in producers:
        p.start()
    consumers = [Process(target=worker_pred, args=(q_out, q_cnt, q_in, model_loader, pred_func,
                                                   mkwargs, device)) for _ in range(npredictor)]
    for c in consumers:
        c.start()
    dict_out = collections.defaultdict(list)
    cnt_end = 0
    lsnr = Process(target=listener, args=(q_cnt, q_in, q_loader_sync, npredictor,
                                          nloader, nsamples_tot))
    lsnr.start()
    while True:
        if q_out.empty():
            if cnt_end == npredictor:
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
    ssv, feats, feat_labels = args
    vert_dc = dict()
    # TODO: replace by poisson disk sampling
    for k in feats:
        pcd = o3d.geometry.PointCloud()
        verts = ssv.load_mesh(k)[1].reshape(-1, 3)
        pcd.points = o3d.utility.Vector3dVector(verts)
        pcd = pcd.voxel_down_sample(voxel_size=pts_feat_ds_dict[k])
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
    return hc


def pts_loader_scalar(ssd_kwargs: dict, ssv_ids: Union[list, np.ndarray],
                      batchsize: int, npoints: int,
                      transform: Optional[Callable] = None,
                      train: bool = False, draw_local: bool = False,
                      draw_local_dist: int = 1000
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
                feat_dc.values())))
            npoints_ssv = min(len(hc.vertices), npoints)
            batch = np.zeros((occ, npoints_ssv, 3))
            batch_f = np.zeros((occ, npoints_ssv, len(feat_dc)))
            ixs = np.ones((occ,), dtype=np.uint) * ssv_id
            cnt = 0
            # TODO: this should be deterministic during inference
            nodes = hc.base_points(density_mode=False, threshold=5000, source=len(hc.nodes) // 2)
            source_nodes = np.random.choice(nodes, occ, replace=len(nodes) < occ)
            for source_node in source_nodes:
                local_bfs = bfs_vertices(hc, source_node, npoints_ssv)
                pc = extract_subset(hc, local_bfs)[0]  # only pass PointCloud
                sample_feats = pc.features
                sample_pts = pc.vertices
                # make sure there is always the same number of points within a batch
                sample_ixs = np.arange(len(sample_pts))
                # np.random.shuffle(sample_ixs)
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
            yield (ixs, (batch_f, batch))
    else:
        ssv_ids = np.unique(ssv_ids)

        for curr_ssvids in ssv_ids:
            ssv = ssd.get_super_segmentation_object(curr_ssvids)
            hc = _load_ssv_hc((ssv, tuple(feat_dc.keys()), tuple(
                feat_dc.values())))
            print(_load_ssv_hc.cache_info())
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
                local_bfs = bfs_vertices(hc, source_node, npoints_ssv)
                pc = extract_subset(hc, local_bfs)[0]  # only pass PointCloud
                sample_feats = pc.features
                sample_pts = pc.vertices
                # shuffling
                sample_ixs = np.arange(len(sample_pts))
                np.random.shuffle(sample_ixs)
                sample_pts = sample_pts[sample_ixs][:npoints_ssv]
                sample_feats = sample_feats[sample_ixs][:npoints_ssv]
                # add up to 10% of duplicated points before applying the transform if sample_pts
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
            yield (ixs, (batch_f, batch))


def pts_pred_scalar(m, inp, q_out, q_cnt, device):
    """

    Args:
        m: Model instance.
        inp: Input as given by the loader_func.
        q_out:
        q_cnt:
        device:

    Returns:

    """
    # TODO: is it possible to get 'device' directly frm model 'm'?
    ssv_ids, inp = inp
    with torch.no_grad():
        g_inp = [torch.from_numpy(i).to(device).float() for i in inp]
        res = m(*g_inp).cpu().numpy()[:, np.newaxis]
        del g_inp, inp
    q_cnt.put(len(ssv_ids))
    q_out.put((ssv_ids, res))


def pts_loader_glia(ssv_params: List[Tuple[int, dict]],
                    out_point_label: List[Union[str, int]],
                    batchsize: int, npoints: int,
                    transform: Optional[Callable] = None,
                    n_out_pts: int = 100,
                    use_subcell: bool = False,
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

    Yields: SSV IDs [M, ], (point location [N, 3], point feature [N, C]), (out_pts [N, 3], out_labels [N, 1])

    """
    # TODO: support node attributes in hybrid cloud graph also
    if type(out_point_label) == str:
        raise NotImplementedError
    # TODO: add noise on output points?
    feat_dc = dict(pts_feat_dict)
    # TODO: add use_syntype
    del feat_dc['syn_ssv_asym']
    del feat_dc['syn_ssv_sym']
    if not use_subcell:
        del feat_dc['mi']
        del feat_dc['vc']
        del feat_dc['syn_ssv']

    for curr_ssv_params in ssv_params:
        ssd = SuperSegmentationDataset(**curr_ssv_params[1])
        ssv = ssd.get_super_segmentation_object(curr_ssv_params[0])
        hc = _load_ssv_hc((ssv, tuple(feat_dc.keys()), tuple(
            feat_dc.values())))
        npoints_ssv = min(len(hc.vertices), npoints)
        # add a +-10% fluctuation in the number of input points
        npoints_add = np.random.randint(-int(npoints_ssv * 0.1), int(npoints_ssv * 0.1))
        npoints_ssv += npoints_add
        batch = np.zeros((batchsize, npoints_ssv, 3))
        batch_f = np.zeros((batchsize, npoints_ssv, len(feat_dc)))
        batch_out = np.zeros((batchsize, n_out_pts, 3))
        batch_out_l = np.zeros((batchsize, n_out_pts, 1))
        ixs = np.ones((batchsize,), dtype=np.uint) * ssv.id
        cnt = 0
        source_nodes = np.random.choice(np.arange(len(hc.nodes)), batchsize,
                                        replace=len(hc.nodes) < batchsize)
        # g = hc.graph(simple=False)
        for source_node in source_nodes:
            # create local context
            node_ids = bfs_vertices(hc, source_node, npoints_ssv)
            hc_sub = extract_subset(hc, node_ids)[0]  # only pass HybridCloud
            sample_feats = hc.features
            sample_pts = hc.vertices
            # get target locations
            base_points = hc_sub.base_points(threshold=1000)  # ~1um apart
            base_points = np.random.choice(base_points, n_out_pts,
                                           replace=len(base_points) < n_out_pts)
            out_coords = hc_sub.nodes[base_points]
            # sub-sample vertices
            sample_ixs = np.arange(len(sample_pts))
            np.random.shuffle(sample_ixs)
            sample_pts = sample_pts[sample_ixs][:npoints_ssv]
            sample_feats = sample_feats[sample_ixs][:npoints_ssv]
            # add up to 10% of duplicated points before applying the transform if sample_pts
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
            # TODO: currently only supports type(out_point_label) = int
            batch_out_l[cnt] = out_point_label
            cnt += 1
        assert cnt == batchsize
        yield (ixs, (batch_f, batch), (batch_out, batch_out_l))


def pts_loader_semseg(ssd_kwargs: dict, ssv_ids: Union[list, np.ndarray],
                      batchsize: int, npoints: int,
                      transform: Optional[Callable] = None,
                      train: bool = False, draw_local: bool = False,
                      draw_local_dist: int = 1000
                      ) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Generator for SSV point cloud samples of size `npoints`. Currently used for
    semantic segmentation tasks, e.g. morphology embeddings.

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


