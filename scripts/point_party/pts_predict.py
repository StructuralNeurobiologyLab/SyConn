# SyConn-dev
# Copyright (c) 2016 Philipp J. Schubert
# All rights reserved

import collections
import os
from multiprocessing import Process, Queue
from syconn.handler import basics
from syconn.handler.prediction import generate_pts_sample, \
    pts_feat_dict, certainty_estimate, pts_loader_ssvs
from syconn.handler.basics import chunkify, chunkify_successive
import numpy as np
import time
import tqdm
import morphx.processing.clouds as clouds
from sklearn.metrics.classification import classification_report
from syconn.reps.super_segmentation import SuperSegmentationDataset


def worker_pred(q_out: Queue, q_cnt: Queue, q_in: Queue, model_loader):
    import torch
    m = model_loader()
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
        ssv_ids, inp = inp
        with torch.no_grad():
            inp = (torch.from_numpy(i).cuda().float() for i in inp)
            res = m(*inp).cpu().numpy()
        q_cnt.put(len(ssv_ids))
        q_out.put((ssv_ids, res))
    q_out.put('END')


def worker_load(q: Queue, q_loader_sync: Queue, gen, kwargs: dict):
    for el in gen(**kwargs):
        while True:
            if q.full():
                time.sleep(1)
            else:
                break
        q.put(el)
    time.sleep(1)
    q_loader_sync.put('DONE')


def load_model():
    from elektronn3.models.convpoint import ModelNet40
    import torch
    m = ModelNet40(5, 8).to('cuda')
    m.load_state_dict(torch.load(mpath)['model_state_dict'])
    m.eval()
    return m


def listener(q_cnt: Queue, q_in, q_loader_sync, npredictor, nloader, total):
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


def predict_pts_wd(ssd_kwargs, model_loader, npoints, scale_fact, nloader=4, npredictor=2,
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
        transform = [clouds.RandomVariation((-10, 10))] + transform + [clouds.RandomRotate()]
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
    producers = [Process(target=worker_load, args=(q_in, q_loader_sync, pts_loader_ssvs, el))
                 for el in params_in]
    for p in producers:
        p.start()
    consumers = [Process(target=worker_pred, args=(q_out, q_cnt, q_in, model_loader)) for _ in
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
    da_equals_tan = True
    split_dc = basics.load_pkl2obj('/wholebrain/songbird/j0126/areaxfs_v6/ssv_ctgt_v4'
                                   '/ctgt_v4_splitting_cv0_10fold.pkl')
    model_dir = '/wholebrain/u/pschuber/e3_training_convpoint/'
    # mpath = f'{model_dir}/celltype_pts_tnet_scale30000_nb75000_cv
    # -1_nDim10_SNAPSHOT/state_dict.pth'
    mpath = f'{model_dir}/celltype_pts_scale30000_nb50000_moreAug3_CV0_eval0' \
            f'/state_dict_minlr_step35000.pth'
    assert os.path.isfile(mpath)
    # wd = '/ssdscratch/pschuber/songbird/j0126/areaxfs_v10_v4b_base_20180214_full_agglo_cbsplit/'
    # version = None

    wd = "/wholebrain/songbird/j0126/areaxfs_v6/"
    gt_version = "ctgt_v4"
    ssd_kwargs = dict(working_dir=wd, version=gt_version)
    res_dc = predict_pts_wd(ssd_kwargs, load_model, 50000, 30000, ssv_ids=split_dc['valid'],
                            nloader=2, npredictor=1, use_test_aug=True)
    basics.write_obj2pkl('/wholebrain/scratch/pschuber/test_celltype_pred.pkl',
                         res_dc)

    # compare to GT
    res_dc = basics.load_pkl2obj('/wholebrain/scratch/pschuber/test_celltype_pred.pkl')

    import pandas
    str2int_label = dict(STN=0, DA=1, MSN=2, LMAN=3, HVC=4, GP=5, TAN=6, GPe=5,
                         INT=7, FS=8, GLIA=9)
    del str2int_label['GLIA']
    del str2int_label['FS']
    str2int_label["GP "] = 5  # typo
    csv_p = '/wholebrain/songbird/j0126/GT/celltype_gt/j0126_cell_type_gt_areax_fs6_v3.csv'
    df = pandas.io.parsers.read_csv(csv_p, header=None, names=['ID', 'type']).values
    ssv_ids = df[:, 0].astype(np.uint)
    if len(np.unique(ssv_ids)) != len(ssv_ids):
        raise ValueError('Multi-usage of IDs!')
    str_labels = df[:, 1]
    ssv_labels = np.array([str2int_label[el] for el in str_labels], dtype=np.uint16)
    valid_ids, valid_ls, valid_preds = [], [], []
    for ix, curr_id in enumerate(ssv_ids):
        if curr_id not in split_dc['valid']:
            continue
        curr_l = ssv_labels[ix]
        if da_equals_tan:
            # adapt GT labels
            if curr_l == 6: curr_l = 1  # TAN and DA are the same now
            if curr_l == 7: curr_l = 6  # INT now has label 6
        valid_ls.append(curr_l)
        valid_preds.append(res_dc[curr_id][0])
        valid_ids.append(curr_id)
    int2str_label = {v: k for k, v in str2int_label.items()}
    target_names = [int2str_label[kk] for kk in range(8)]
    if da_equals_tan:
        target_names[1] = 'Modulatory'
        target_names.remove('TAN')

    print(classification_report(valid_ls, valid_preds, labels=np.arange(7),
                                target_names=target_names))
    raise()