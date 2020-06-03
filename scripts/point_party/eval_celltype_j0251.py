import collections
import re
from syconn.handler import basics, config
from syconn.handler.prediction import certainty_estimate
from syconn.mp.mp_utils import start_multiprocess_imap
from syconn.handler.prediction import get_celltype_model_e3
import numpy as np
from sklearn.metrics.classification import classification_report
from sklearn.metrics import confusion_matrix
from syconn.reps.super_segmentation import SuperSegmentationDataset
from syconn.handler.prediction_pts import predict_pts_plain, \
    pts_loader_scalar, pts_pred_scalar, get_celltype_model_pts, get_pt_kwargs
import os
import tqdm


def predict_celltype_wd(ssd_kwargs, model_loader, mpath, npoints, scale_fact, ctx_size, nloader=4, npredictor=2,
                        ssv_ids=None, use_test_aug=False, device='cuda', **kwargs):
    """
    Perform cell type predictions of cell reconstructions on sampled point sets from the
    cell's vertices. The number of predictions ``npreds`` per cell is calculated based on the
    fraction of the total number of vertices over ``npoints`` times two, but at least 5.
    Every point set is constructed by collecting the vertices associated with skeleton within a
    breadth-first search up to a maximum of ``npoints``.


    Args:
        ssd_kwargs:
        model_loader:
        mpath:
        npoints:
        scale_fact:
        nloader:
        npredictor:
        ssv_ids:
        use_test_aug:
        device:

    Returns:

    """
    out_dc = predict_pts_plain(ssd_kwargs, model_loader, pts_loader_scalar, pts_pred_scalar, mpath=mpath,
                               npoints=npoints, scale_fact=scale_fact, nloader=nloader, npredictor=npredictor,
                               ssv_ids=ssv_ids, use_test_aug=use_test_aug, device=device, ctx_size=ctx_size, **kwargs)
    out_dc = dict(out_dc)
    for ssv_id in out_dc:
        logit = np.concatenate(out_dc[ssv_id])
        if da_equals_tan:
            # accumulate evidence for DA and TAN
            logit[:, 1] += logit[:, 6]
            # remove TAN in proba array
            logit = np.delete(logit, [6], axis=1)
            # INT is now at index 6 -> label 6 is INT
        cls = np.argmax(logit, axis=1).squeeze()
        cls_maj = collections.Counter(cls).most_common(1)[0][0]
        out_dc[ssv_id] = (cls_maj, certainty_estimate(logit, is_logit=True))
    return out_dc


def _preproc_syns(args):
    ssv_ids, ssd_kwargs = args
    ssd = SuperSegmentationDataset(**ssd_kwargs)
    celltypes = []
    celltypes_old = []
    m = get_celltype_model_e3()
    for ssv in ssd.get_super_segmentation_object(ssv_ids):
        ssv._view_caching = True
        ssv.load_attr_dict()
        _ = ssv.syn_ssv_mesh
        _ = ssv.typedsyns2mesh()
        if 'celltype_cnn_e3_v2' not in ssv.attr_dict:
            ssv.predict_celltype_cnn(m, '_v2', onthefly_views=True, verbose=True,
                                     view_props=dict(nb_views=2))
        celltypes.append(ssv.celltype('celltype_cnn_e3_v2'))
        celltypes_old.append(ssv.celltype())
    return celltypes, celltypes_old


if __name__ == '__main__':
    # load GT
    da_equals_tan = True
    import pandas
    str2int_label = dict(STN=0, DA=1, MSN=2, LMAN=3, HVC=4, GP=5, TAN=6, GPe=5,
                         INT=7, FS=7, GLIA=9, LTS=7, GPi=5)
    del str2int_label['GLIA']
    str2int_label["GP "] = 5  # typo
    int2str_label = {v: k for k, v in str2int_label.items()}
    target_names = [int2str_label[kk] for kk in range(8)]
    if da_equals_tan:
        target_names[1] = 'Modulatory'
        target_names.remove('TAN')
    csv_p = "/wholebrain/songbird/j0251/groundtruth/j0251_celltype_gt_v0.csv"
    df = pandas.io.parsers.read_csv(csv_p, header=None, names=['ID', 'type']).values
    ssv_ids = df[:, 0].astype(np.uint)
    if len(np.unique(ssv_ids)) != len(ssv_ids):
        raise ValueError('Multi-usage of IDs!')
    str_labels = df[:, 1]
    ssv_labels = np.array([str2int_label[el] for el in str_labels], dtype=np.uint16)

    for ix, curr_id in enumerate(ssv_ids):
        curr_l = ssv_labels[ix]
        if da_equals_tan:
            # adapt GT labels
            if curr_l == 6: curr_l = 1  # TAN and DA are the same now
            if curr_l == 7: curr_l = 6  # INT now has label 6
        ssv_labels[ix] = curr_l
    #
    base_dir = '/wholebrain/scratch/pschuber/e3trainings_BAK/ptconv_2020_06_03/celltype_eval0_sp50k/'
    CV = 0
    mdir = base_dir + '/celltype_pts_scale2000_nb50000_ctx20000_swish_gn_CV{}_eval0/'
    mkwargs, loader_kwargs = get_pt_kwargs(mdir)
    npoints = loader_kwargs['npoints']
    log = config.initialize_logging(f'log_eval_j0251_celltype_gt_v0_sp{npoints}k', base_dir)

    # eval multiview data
    wd = "/ssdscratch/pschuber/songbird/j0251/rag_flat_Jan2019/"
    ssd_kwargs = dict(working_dir=wd)
    ssd = SuperSegmentationDataset(**ssd_kwargs)
    params = [([ssv_id], ssd_kwargs) for ssv_id in ssv_ids]
    mv_preds = start_multiprocess_imap(_preproc_syns, params, nb_cpus=1)

    mv_preds_old = np.array([p[1] for p in mv_preds])
    mv_preds = np.array([p[0] for p in mv_preds])
    print(f'Multi view data (sj):')
    log.info(classification_report(ssv_labels, mv_preds_old, labels=np.arange(7),
                                   target_names=target_names))
    for ix, curr_id in enumerate(ssv_ids):
        if mv_preds_old[ix] != ssv_labels[ix]:
            log.info(f'id: {curr_id} target: {ssv_labels[ix]} prediction: {mv_preds_old[ix]}')
    print(confusion_matrix(ssv_labels, mv_preds_old, labels=np.arange(7)))
    print(f'Multi view data (syn_ssv):')
    log.info(classification_report(ssv_labels, mv_preds, labels=np.arange(7),
                                   target_names=target_names))
    for ix, curr_id in enumerate(ssv_ids):
        if mv_preds_old[ix] != ssv_labels[ix]:
            log.info(f'id: {curr_id} target: {ssv_labels[ix]} prediction: {mv_preds[ix]}')
    print(confusion_matrix(ssv_labels, mv_preds, labels=np.arange(7)))

    # eval point based data
    log.info(f'\nStarting evaluation of model with npoints={npoints}, eval. j0251_celltype_gt_v0, '
             f'model_kwargs={mkwargs} and da_equals_tan={da_equals_tan}.\n')
    mpath = f'{mdir.format(CV, 0)}/state_dict_final.pth'
    mkwargs['mpath'] = mpath
    log.info(f'Using model "{mpath}".')
    fname_pred = f'{base_dir}/j0251_celltype_gt_v0_PRED.pkl'
    assert os.path.isfile(mpath)

    res_dc = predict_celltype_wd(ssd_kwargs, get_celltype_model_pts, mpath, ssv_ids=ssv_ids,
                                 nloader=6, npredictor=3, use_test_aug=False, seeded=True, **loader_kwargs)
    basics.write_obj2pkl(fname_pred, res_dc)

    target_ids_local, pts_preds = [], []
    for ix, curr_id in enumerate(ssv_ids):
        curr_pred, curr_cert = res_dc[curr_id]
        pts_preds.append(curr_pred)
        target_ids_local.append(curr_id)
        if curr_pred != ssv_labels[ix]:
            log.info(f'id: {curr_id} target: {ssv_labels[ix]} prediction: {curr_pred} certainty: {curr_cert:.2f}')
    print(f'Point cloud based prediction:')
    log.info(classification_report(ssv_labels, pts_preds, labels=np.arange(7), target_names=target_names))
    print(confusion_matrix(ssv_labels, pts_preds, labels=np.arange(7)))
