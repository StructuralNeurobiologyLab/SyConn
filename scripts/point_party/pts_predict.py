# SyConn-dev
# Copyright (c) 2016 Philipp J. Schubert
# All rights reserved

import collections
from multiprocessing import Process, Queue
from syconn.handler import basics
from syconn.handler.prediction import generate_pts_sample, \
    pts_feat_dict, certainty_estimate
from syconn.handler.basics import chunkify
import numpy as np
import time
import tqdm
from sklearn.metrics.classification import classification_report
import open3d as o3d
from syconn.reps.super_segmentation import SuperSegmentationDataset


def worker_pred(q_out: Queue, q_cnt: Queue, q_in: Queue, model_loader):
    import torch
    m = model_loader()
    ixs = []
    res = []
    while True:
        if not q_in.empty():
            inp = q_in.get()
            if inp == 'STOP':
                print('Predictor finished.', end="")
                break
        else:
            time.sleep(1)
            print('Idle predictor.', end="")
            continue
        ssv_ids, inp = inp
        with torch.no_grad():
            inp = (torch.from_numpy(i).cuda().float() for i in inp)
            res.append(m(*inp).cpu().numpy())
        ixs.append(ssv_ids)
        q_cnt.put(len(ssv_ids))
    q_out.put((np.concatenate(ixs), np.concatenate(res)))
    q_out.put('END')


def worker_load(q: Queue, q_loader_sync: Queue, gen, kwargs: dict):
    for el in gen(**kwargs):
        while True:
            if q.full():
                time.sleep(1)
            else:
                break
        q.put(el)
    q_loader_sync.put('DONE')


def _pts_loader_ssvs(ssd_kwargs, ssv_ids, batchsize, npoints, scale_fact, redundancy=5):
    np.random.shuffle(ssv_ids)
    ssd = SuperSegmentationDataset(**ssd_kwargs)
    nbatches = int(np.ceil(len(ssv_ids) / batchsize)) * redundancy
    nsamples = nbatches * batchsize
    ndiff = int(nsamples / redundancy - len(ssv_ids))
    ssv_ids = np.concatenate([ssv_ids, ssv_ids[:ndiff]])
    # TODO: add `use_syntype kwarg and cellshape only
    feat_dc = dict(pts_feat_dict)
    if 'syn_ssv' in feat_dc:
        del feat_dc['syn_ssv']
    for curr_ssvids in chunkify(ssv_ids, nbatches):
        ssvs = ssd.get_super_segmentation_object(curr_ssvids)
        batch = np.zeros((batchsize, npoints, 3))
        batch_f = np.zeros((batchsize, npoints, len(feat_dc)))
        ixs = np.zeros((batchsize, ), dtype=np.uint)
        cnt = 0
        for ssv in ssvs:
            vert_dc = dict()
            for k in feat_dc:
                pcd = o3d.geometry.PointCloud()
                verts = ssv.load_mesh(k)[1].reshape(-1, 3)
                pcd.points = o3d.utility.Vector3dVector(verts)
                pcd = pcd.voxel_down_sample(voxel_size=50)
                vert_dc[k] = np.asarray(pcd.points)
            for _ in range(redundancy):
                v_s, f_s = generate_pts_sample(vert_dc, pts_feat_dict, False, len(feat_dc),
                                               True, npoints, True)
                v_s -= v_s.mean(axis=0)
                batch[cnt] = v_s / scale_fact
                batch_f[cnt] = f_s
                ixs[cnt] = ssv.id
                cnt += 1
        assert cnt == batchsize
        yield (ixs, (batch_f, batch))


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
                break
            pbar.update(res)
        if q_loader_sync.empty() or cnt_loder_done == nloader:
            time.sleep(1)
        else:
            _ = q_loader_sync.get()
            cnt_loder_done += 1
            print('Loader finished.', end="")
            if cnt_loder_done == nloader:
                for _ in range(npredictor):
                    q_in.put('STOP')


def predict_pts_wd(ssd_kwargs, model_loader, npoints, scale_fact, nloader=4, npredictor=2,
                   ssv_ids=None):
    bs = 80
    redundancy = 10
    assert bs % redundancy == 0  # divisible by redundancy
    ssd = SuperSegmentationDataset(**ssd_kwargs)
    if ssv_ids is None:
        ssv_ids = ssd.ssv_ids
    kwargs = dict(batchsize=bs, npoints=npoints, ssd_kwargs=ssd_kwargs, scale_fact=scale_fact, redundancy=redundancy)
    params_in = [{**kwargs, **dict(ssv_ids=ch)} for ch in chunkify(ssv_ids, nloader)]
    q_in = Queue(maxsize=20*npredictor)
    q_cnt = Queue()
    q_out = Queue()
    q_loader_sync = Queue()
    producers = [Process(target=worker_load, args=(q_in, q_loader_sync, _pts_loader_ssvs, el)) for el in params_in]
    for p in producers:
        p.start()
    consumers = [Process(target=worker_pred, args=(q_out, q_cnt, q_in, model_loader)) for _ in
                 range(npredictor)]
    for c in consumers:
        c.start()
    res_dc = dict()
    cnt_end = 0
    lsnr = Process(target=listener, args=(q_cnt, q_in, q_loader_sync, npredictor,
                                          nloader, len(ssv_ids) * redundancy))
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
        ids, logits = res
        for ssv_id in np.unique(ids):
            logit = logits[ids == ssv_id]
            cls = np.argmax(logit, axis=1).squeeze()
            cls_maj = collections.Counter(cls).most_common(1)[0][0]
            res_dc[ssv_id] = (cls_maj, certainty_estimate(logit, is_logit=True))
    print('Finished collection of results.', end="")
    q_cnt.put(None)
    lsnr.join()
    print('Joined listener.', end="")
    for p in producers:
        p.join()
        p.close()
    print('Joined producers.', end="")
    for c in consumers:
        c.join()
        c.close()
    print('Joined consumers.', end="")
    return res_dc


if __name__ == '__main__':
    model_dir = '/wholebrain/u/pschuber/e3_training_convpoint/'
    # mpath = f'{model_dir}/celltype_pts_tnet_scale30000_nb75000_cv
    # -1_nDim10_SNAPSHOT/state_dict.pth'
    mpath = f'{model_dir}/celltype_pts_scale30000_nb75000_cv0_SNAPSHOT' \
            f'/state_dict_final.pth'
    # wd = '/ssdscratch/pschuber/songbird/j0126/areaxfs_v10_v4b_base_20180214_full_agglo_cbsplit/'
    # version = None

    # wd = "/wholebrain/songbird/j0126/areaxfs_v6/"
    # gt_version = "ctgt_v4"
    # ssd_kwargs = dict(working_dir=wd, version=gt_version)
    # res_dc = predict_pts_wd(ssd_kwargs, load_model, 75000, 30000)
    # basics.write_obj2pkl('/wholebrain/scratch/pschuber/test_celltype_pred.pkl',
    #                      res_dc)

    # compare to GT
    res_dc = basics.load_pkl2obj('/wholebrain/scratch/pschuber/test_celltype_pred.pkl')
    import pandas
    str2int_label = dict(STN=0, DA=1, MSN=2, LMAN=3, HVC=4, GP=5, TAN=6, GPe=5, INT=7, FS=8, GLIA=9)
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
    print(classification_report(ssv_labels, [res_dc[ix][0] for ix in ssv_ids],
                                labels=list(str2int_label.values()), target_names=list(str2int_label.keys())))
    raise()