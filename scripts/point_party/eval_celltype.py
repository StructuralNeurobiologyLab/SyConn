import collections
import re
from syconn.handler import basics, config
from syconn.handler.prediction import certainty_estimate
import numpy as np
from sklearn.metrics.classification import classification_report
from syconn.handler.prediction_pts import predict_pts_plain, \
    pts_loader_scalar, pts_pred_scalar, get_celltype_model_pts, get_pt_kwargs
import os


def predict_celltype_wd(ssd_kwargs, model_loader, mpath, npoints, scale_fact, ctx_size, nloader=4, npredictor=2,
                        ssv_ids=None, use_test_aug=False, device='cuda'):
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
        nloader:
        npredictor:
        ssv_ids:
        use_test_aug:

    Returns:

    """
    out_dc = predict_pts_plain(ssd_kwargs, model_loader, pts_loader_scalar, pts_pred_scalar, mpath=mpath,
                               npoints=npoints, scale_fact=scale_fact, nloader=nloader, npredictor=npredictor,
                               ssv_ids=ssv_ids, use_test_aug=use_test_aug, device=device, ctx_size=ctx_size)
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


if __name__ == '__main__':
    ncv_min = 0
    n_cv = 10
    da_equals_tan = True
    wd = "/wholebrain/songbird/j0126/areaxfs_v6/"
    gt_version = "ctgt_v4"
    base_dir_init = '/wholebrain/scratch/pschuber/e3_trainings_convpoint/celltype_eval{}_sp0k/'
    for run in range(3):
        base_dir = base_dir_init.format(run)
        ssd_kwargs = dict(working_dir=wd, version=gt_version)
        mdir = base_dir + '/celltype_pts_scale2000_nb500_ctx10000_swish_gn_CV{}_eval{}/'
        mkwargs, loader_kwargs = get_pt_kwargs(mdir)
        npoints = loader_kwargs['npoints']
        log = config.initialize_logging(f'log_eval{run}_sp{npoints}k', base_dir)
        log.info(f'\nStarting evaluation of model with npoints={npoints}, eval. run={run}, '
                 f'model_kwargs={mkwargs} and da_equals_tan={da_equals_tan}.\n'
                 f'GT: version={gt_version} at wd={wd}\n')
        for CV in range(ncv_min, n_cv):
            split_dc = basics.load_pkl2obj(f'/wholebrain/songbird/j0126/areaxfs_v6/ssv_ctgt_v4'
                                           f'/ctgt_v4_splitting_cv{CV}_10fold.pkl')
            mpath = f'{mdir.format(CV, run)}/state_dict_final.pth'
            mkwargs['mpath'] = mpath
            log.info(f'Using model "{mpath}" for cross-validation split {CV}.')
            fname_pred = f'{base_dir}/ctgt_v4_splitting_cv{CV}_10fold_PRED.pkl'
            assert os.path.isfile(mpath)

            res_dc = predict_celltype_wd(ssd_kwargs, get_celltype_model_pts, mpath, ssv_ids=split_dc['valid'],
                                         nloader=10, npredictor=5, use_test_aug=True, **loader_kwargs)
            basics.write_obj2pkl(fname_pred, res_dc)

        # compare to GT
        import pandas
        str2int_label = dict(STN=0, DA=1, MSN=2, LMAN=3, HVC=4, GP=5, TAN=6, GPe=5,
                             INT=7, FS=8, GLIA=9)
        del str2int_label['GLIA']
        del str2int_label['FS']
        str2int_label["GP "] = 5  # typo
        int2str_label = {v: k for k, v in str2int_label.items()}
        target_names = [int2str_label[kk] for kk in range(8)]
        if da_equals_tan:
            target_names[1] = 'Modulatory'
            target_names.remove('TAN')
        csv_p = '/wholebrain/songbird/j0126/GT/celltype_gt/j0126_cell_type_gt_areax_fs6_v3.csv'
        df = pandas.io.parsers.read_csv(csv_p, header=None, names=['ID', 'type']).values
        ssv_ids = df[:, 0].astype(np.uint)
        if len(np.unique(ssv_ids)) != len(ssv_ids):
            raise ValueError('Multi-usage of IDs!')
        str_labels = df[:, 1]
        ssv_labels = np.array([str2int_label[el] for el in str_labels], dtype=np.uint16)
        valid_ids, valid_ls, valid_preds, valid_certainty = [], [], [], []

        for CV in range(ncv_min, n_cv):
            res_dc = basics.load_pkl2obj(f'{base_dir}/ctgt_v4_splitting_cv{CV}_10fold_PRED.pkl')
            split_dc = basics.load_pkl2obj(f'/wholebrain/songbird/j0126/areaxfs_v6/ssv_ctgt_v4'
                                           f'/ctgt_v4_splitting_cv{CV}_10fold.pkl')
            valid_ids_local, valid_ls_local, valid_preds_local, valid_certainty_local = [], [], [], []
            for ix, curr_id in enumerate(ssv_ids):
                if curr_id not in split_dc['valid']:
                    continue
                curr_l = ssv_labels[ix]
                if da_equals_tan:
                    # adapt GT labels
                    if curr_l == 6: curr_l = 1  # TAN and DA are the same now
                    if curr_l == 7: curr_l = 6  # INT now has label 6
                valid_ls_local.append(curr_l)
                curr_pred, curr_cert = res_dc[curr_id]
                valid_preds_local.append(curr_pred)
                valid_certainty.append(curr_cert)
                valid_ids_local.append(curr_id)
                if curr_pred != curr_l:
                    log.info(f'id: {curr_id} target: {curr_l} prediction: {curr_pred} certainty: {curr_cert:.2f}')
            log.info(f'\nCV split: {CV}')
            log.info(classification_report(valid_ls_local, valid_preds_local, labels=np.arange(7),
                                           target_names=target_names))
            valid_ls.extend(valid_ls_local)
            valid_preds.extend(valid_preds_local)
            valid_ids.extend(valid_ids_local)

        log.info(f'Final prediction result for run {run} with {loader_kwargs} and {mkwargs}.')
        log.info(classification_report(valid_ls, valid_preds, labels=np.arange(7),
                                       target_names=target_names))
        log.info('-------------------------------')
