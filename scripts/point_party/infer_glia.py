import collections
import re
from syconn.handler import basics, config
from syconn.handler.prediction import certainty_estimate
import numpy as np
from collections import defaultdict, Counter
from sklearn.metrics.classification import classification_report
from syconn.handler.prediction_pts import predict_pts_plain, \
    pts_loader_glia, pts_pred_glia
from syconn.reps.super_segmentation import SuperSegmentationDataset
import os


def load_model(mkwargs, device):
    from elektronn3.models.convpoint import SegSmall
    import torch
    mpath = mkwargs['mpath']
    del mkwargs['mpath']
    m = SegSmall(1, 2, **mkwargs).to(device)
    m.load_state_dict(torch.load(mpath)['model_state_dict'])
    # pts_pred_scalar (pred_func used in predict_pts_plain) requires the model object to have a .predict method
    # m = torch.nn.DataParallel(m)
    return m


def predict_glia_wd(ssd_kwargs, model_loader, mkwargs, npoints, scale_fact, ssv_ids=None, **kwargs_add):
    """
    Perform cell type predictions of cell reconstructions on sampled point sets from the
    cell's vertices. The number of predictions ``npreds`` per cell is calculated based on the
    fraction of the total number of vertices over ``npoints`` times two, but at least 5.
    Every point set is constructed by collecting the vertices associated with skeleton within a
    breadth-first search up to a maximum of ``npoints``.


    Args:
        ssd_kwargs:
        model_loader:
        mkwargs:
        npoints:
        scale_fact:
        ssv_ids:

    Returns:

    """
    tmp_dir = '/wholebrain/scratch/pschuber/tmp/glia_inference/'
    os.makedirs(tmp_dir, exist_ok=True)
    from sklearn.preprocessing import label_binarize
    from scipy.spatial import cKDTree
    ssd = SuperSegmentationDataset(**ssd_kwargs)
    ixs = np.random.choice(len(ssd.ssv_ids), 1000, replace=False)
    ssv_ids = ssd.ssv_ids[ixs]
    sizes = ssd.load_cached_data('size')[ixs]
    ssv_params = [(ssv_id, ssd_kwargs) for ssv_id in ssv_ids[np.argsort(sizes)[::-1]]]
    out_dc = predict_pts_plain(ssv_params, model_loader, pts_loader_glia, pts_pred_glia, mkwargs=mkwargs,
                               npoints=npoints, scale_fact=scale_fact, ssv_ids=ssv_ids, **kwargs_add)
    from syconn.handler.prediction_pts import write_pts_ply
    for ix, out in out_dc.items():
        # out: [(dict(t_pts=.., t_label, batch_process)]

        # el['t_l'] has shape (b, num_points, n_classes)
        prediction = np.argmax(np.concatenate([el['t_l'].reshape(-1, 2) for el in out]), axis=1)[..., None]
        if not np.sum(prediction == 1) / len(prediction) > 0.05:
            continue
        fname = f'{tmp_dir}/glia_test_{ix}_cellmesh.ply'
        sso = ssd.get_super_segmentation_object(ix)
        cell_mesh = sso.mesh[1].reshape(-1, 3)
        write_pts_ply(fname, cell_mesh, np.zeros((cell_mesh.shape[0], 1)))
        fname = f'{tmp_dir}/glia_test_{ix}_pred.ply'
        # el['t_pts'] has shape (b, num_points, 3)
        print(ix, np.unique(prediction, return_counts=True)[1] / len(prediction))
        # 4 will be red in write_pts_ply
        orig_coords = np.concatenate([el['t_pts'].reshape(-1, 3) for el in out])
        sso.load_skeleton()
        skel = sso.skeleton
        kdt = cKDTree(orig_coords)
        dists, ixs = kdt.query(skel['nodes'] * sso.scaling, k=10, distance_upper_bound=2000)
        node_pred = np.ones(len(skel['nodes'])) * -1
        for ii, nn_dists, nn_ixs in zip(np.arange(len(skel['nodes'])), dists, ixs):
            nn_ixs = nn_ixs[nn_dists != np.inf]
            preds = prediction[nn_ixs].squeeze()
            node_pred[ii] = Counter(preds).most_common(1)[0][0]
        # every node has at least one prediction
        assert np.sum(node_pred == -1) == 0
        prediction = label_binarize(prediction * 4, classes=np.arange(5))
        write_pts_ply(fname, orig_coords, prediction, binarized=True)
    raise()
    return out_dc


if __name__ == '__main__':
    wd = "/wholebrain/songbird/j0126/areaxfs_v6/"
    base_dir = '/wholebrain/scratch/pschuber/e3_trainings_convpoint/'
    ssd_kwargs = dict(working_dir=wd)
    mdir = base_dir + '/glia_pts_scale1500_nb30000_ctx15000_swish_gn_eval0/'
    use_norm = False
    track_running_stats = False
    activation = 'relu'
    use_bias = False
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
    npoints = int(re.findall(r'_nb(\d+)_', mdir)[0])
    scale_fact = int(re.findall(r'_scale(\d+)_', mdir)[0])
    log = config.initialize_logging(f'log_eval_sp{npoints}k', mdir)
    assert os.path.isfile(f'{mdir}/state_dict_final.pth')
    mkwargs = dict(use_norm=use_norm, track_running_stats=track_running_stats, act=activation,
                   mpath=f'{mdir}/state_dict_final.pth', use_bias=use_bias)
    predict_glia_wd(ssd_kwargs, load_model, mkwargs, npoints, scale_fact, ctx_size=ctx,
                    nloader=10, npredictor=5, bs=25, loader_kwargs=dict(n_out_pts=100, base_node_dst=ctx/2))
