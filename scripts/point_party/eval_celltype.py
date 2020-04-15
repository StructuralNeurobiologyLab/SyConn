import collections
import os
import re
import gc
from multiprocessing import Process, Queue
from syconn.handler import basics, config
from syconn.handler.prediction_pts import worker_load, \
    worker_pred, pts_loader_scalar, pts_pred_scalar, listener
from syconn.handler.prediction import certainty_estimate
from syconn.handler.basics import chunkify, chunkify_successive
import numpy as np
import time
import tqdm
import morphx.processing.clouds as clouds
from sklearn.metrics.classification import classification_report
from syconn.reps.super_segmentation import SuperSegmentationDataset


def load_model(mkwargs, device):
    from elektronn3.models.convpoint import ModelNet40
    import torch
    m = ModelNet40(5, 8, **mkwargs).to(device)
    m.load_state_dict(torch.load(mpath)['model_state_dict'])
    m = torch.nn.DataParallel(m)
    m.eval()
    return m


def predict_celltype_wd(ssd_kwargs, model_loader, mkwargs, npoints, scale_fact, nloader=4, npredictor=2,
                        ssv_ids=None, use_test_aug=False):
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
    producers = [Process(target=worker_load, args=(q_in, q_loader_sync, pts_loader_scalar, el))
                 for el in params_in]
    for p in producers:
        p.start()
    consumers = [Process(target=worker_pred, args=(q_out, q_cnt, q_in, model_loader, pts_pred_scalar, mkwargs)) for _ in
                 range(npredictor)]
    for c in consumers:
        c.start()
    res_dc = collections.defaultdict(list)
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
        for ssv_id, logit in zip(*res):
            res_dc[ssv_id].append(logit[None, ])

    res_dc = dict(res_dc)
    for ssv_id in res_dc:
        logit = np.concatenate(res_dc[ssv_id])
        if da_equals_tan:
            # accumulate evidence for DA and TAN
            logit[:, 1] += logit[:, 6]
            # remove TAN in proba array
            logit = np.delete(logit, [6], axis=1)
            # INT is now at index 6 -> label 6 is INT
        cls = np.argmax(logit, axis=1).squeeze()
        cls_maj = collections.Counter(cls).most_common(1)[0][0]
        res_dc[ssv_id] = (cls_maj, certainty_estimate(logit, is_logit=True))
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
    return res_dc


if __name__ == '__main__':
    ncv_min = 0
    n_cv = 10
    da_equals_tan = True
    wd = "/wholebrain/songbird/j0126/areaxfs_v6/"
    gt_version = "ctgt_v4"
    base_dir_init = '/wholebrain/scratch/pschuber/e3_trainings_convpoint/celltype_eval{}_sp75k/'
    for run in range(1):
        base_dir = base_dir_init.format(run)
        ssd_kwargs = dict(working_dir=wd, version=gt_version)
        mdir = base_dir + '/celltype_pts_scale30000_nb75000_noBN_moreAug4_CV{}_eval{}/'
        use_bn = True
        track_running_stats = False
        if 'noBN' in mdir:
            use_bn = False
        if 'trackRunStats' in mdir:
            track_running_stats = True
        npoints = int(re.findall(r'_nb(\d+)_', mdir)[0])
        scale_fact = int(re.findall(r'_scale(\d+)_', mdir)[0])
        log = config.initialize_logging(f'log_eval{run}_sp{npoints}k', base_dir)
        mkwargs = dict(use_bn=use_bn, track_running_stats=track_running_stats)
        log.info(f'\nStarting evaluation of model with npoints={npoints}, eval. run={run}, '
                 f'model_kwargs={mkwargs} and da_equals_tan={da_equals_tan}.\n'
                 f'GT: version={gt_version} at wd={wd}\n')
        for CV in range(ncv_min, n_cv):
            split_dc = basics.load_pkl2obj(f'/wholebrain/songbird/j0126/areaxfs_v6/ssv_ctgt_v4'
                                           f'/ctgt_v4_splitting_cv{CV}_10fold.pkl')
            mpath = f'{mdir.format(CV, run)}/state_dict.pth'
            log.info(f'Using model "{mpath}" for cross-validation split {CV}.')
            fname_pred = f'{base_dir}/ctgt_v4_splitting_cv{CV}_10fold_PRED.pkl'

            # if os .path.isfile(fname_pred):
            #     continue
            res_dc = predict_celltype_wd(ssd_kwargs, load_model, mkwargs, npoints, scale_fact, ssv_ids=split_dc['valid'],
                                         nloader=2, npredictor=1, use_test_aug=True)
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
                    log.info(f'id: {curr_id}  targtet: {curr_l}  pred: {curr_pred}  ce: {curr_cert}')
            log.info(f'\nCV split: {CV}')
            log.info(classification_report(valid_ls_local, valid_preds_local, labels=np.arange(7),
                                           target_names=target_names))
            valid_ls.extend(valid_ls_local)
            valid_preds.extend(valid_preds_local)
            valid_ids.extend(valid_ids_local)

        log.info(f'Final prediction result for run {run} with npoints={npoints}, '
                 f'track_running_stats={track_running_stats}, use_bn={use_bn}.')
        log.info(classification_report(valid_ls, valid_preds, labels=np.arange(7),
                                       target_names=target_names))
        log.info('-------------------------------')
