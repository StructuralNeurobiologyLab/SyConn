# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max-Planck-Institute of Neurobiology, Munich, Germany
# Authors: Philipp Schubert, Joergen Kornfeld

from syconn.reps.super_segmentation import *
from syconn.handler import log_main
from syconn.handler.multiviews import generate_rendering_locs
import numpy as np
import logging
import pandas

if __name__ == "__main__":
    # based on /wholebrain/songbird/j0126/areaxfs_v6/

    # INT and ? are label 8, GPe and GP are label 5
    str2int_label = dict(STN=0, DA=1, MSN=2, LMAN=3, HVC=4, GP=5, FS=6, TAN=7, GPe=5, INT=8, GLIA=9)
    str2int_label["?"] = 8
    str2int_label["GP "] = 5  # typo
    csv_p = '/wholebrain/songbird/j0126/cell_type_gt_AREAXFS6_updated_April_06_2019.csv'
    gt = df = pandas.io.parsers.read_csv(csv_p).values
    ssv_ids = df[:, 0].astype(np.uint)
    str_labels = df[:, 1]
    ssv_labels = np.array([str2int_label[el] for el in str_labels], dtype=np.uint16)
    classes, c_cnts = np.unique(ssv_labels, return_counts=True)
    log_main.setLevel(20)  # This is INFO level (to filter copied file messages)
    log_main.info('Successfully parsed "{}" with the following cell type class '
                  'distribution [labels, counts]: {}, {}'.format(csv_p, classes,
                                                                 c_cnts))

    WD ="/wholebrain/songbird/j0126/areaxfs_v6/"
    ssd = SuperSegmentationDataset(working_dir=WD)
    orig_ssvs = ssd.get_super_segmentation_object(ssv_ids)
    for ssv in orig_ssvs:
        if not (ssv.id in ssd.ssv_ids and os.path.isfile(ssv.attr_dict_path)):
            msg = 'GT file contains SSV with ID {} which is not part of the ' \
                  'used SSD.'.format(ssv.id)
            log_main.warning(msg)
            raise ValueError(msg)
    gt_version = "ctgt_v2"
    new_ssd = SuperSegmentationDataset(working_dir=WD, version=gt_version)
    copy_ssvs2new_SSD_simple(orig_ssvs, new_version=gt_version,
                             target_wd=WD, safe=False)  # `safe=False` will overwrite existing data
    new_ssd.save_dataset_deep(extract_only=True)
    pbar = tqdm.tqdm(total=len(ssv_ids))
    for ii, ssv_id in enumerate(ssv_ids):
        ssv = new_ssd.get_super_segmentation_object(ssv_id)
        ssv.save_attributes(["cellttype_gt"], [ssv_labels[ii]])
        # # previous run for "normal" bootstrap N-view predictions
        ssv._render_rawviews(4, verbose=True, force_recompute=True)  # TODO: Make sure that copied files are always the same if `force_recompute` is set to False!
        # # Large FoV bootstrapping
        # # downsample vertices and get ~3 locations per comp_window
        verts = ssv.mesh[1].reshape(-1, 3)
        comp_window = 40960  # nm  -> pixel size: 80 nm
        rendering_locs = generate_rendering_locs(verts, comp_window)
        views = render_sso_coords(ssv, rendering_locs, verbose=True, ws=(512, 512), add_cellobjects=True,
                                  return_rot_mat=False, comp_window=comp_window, nb_views=4)
        ssv.save_views(views, view_key="4_large_fov")

        pbar.update()
    write_obj2pkl(new_ssd.path + "/{}_labels.pkl".format(gt_version),
                 {ssv_ids[kk]: ssv_labels[kk] for kk in range(len(ssv_ids))})
    # # test prediction, copy trained models, see paths to 'models' folder defined in global_params.config
    # from syconn.handler.prediction import get_celltype_model_large_e3, get_tripletnet_model_large_e3
    # from syconn.reps.super_segmentation import SuperSegmentationDataset, SuperSegmentationObject
    # from syconn import global_params
    # import tqdm
    # ssd = SuperSegmentationDataset(working_dir=global_params.wd, version="ctgt_v2")
    # pbar = tqdm.tqdm(total=len(ssd.ssv_ids))
    # m = get_celltype_model_large_e3()
    # m_tnet = get_tripletnet_model_large_e3()
    # for ssv in ssd.ssvs:
    #     ssv._view_caching = True
    #     ssv.predict_celltype_cnn(model=m, pred_key_appendix='test_pred_v1',
    #                              model_tnet=m_tnet)
    #     pbar.update()
