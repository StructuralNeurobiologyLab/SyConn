# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max-Planck-Institute of Neurobiology, Munich, Germany
# Authors: Philipp Schubert, Joergen Kornfeld

from syconn.reps.super_segmentation import *
from syconn.handler.logger import log_main
import numpy as np
import logging
import pandas

# based on /wholebrain/songbird/j0126/areaxfs_v6/

str2int_label = dict(STN=0, DA=1, MSN=2, LMAN=3, HVC=4, GP=5, FS=6, TAN=7, GPe=5)
str2int_label["?"] = 8
csv_p = '/wholebrain/songbird/j0126/cell_type_gt_AREAXFS6.csv'
gt = df = pandas.io.parsers.read_csv(csv_p).values
ssv_ids = df[:, 0].astype(np.uint)
str_labels = df[:, 1]
ssv_labels = [str2int_label[el] for el in str_labels]
classes, c_cnts = np.unique(ssv_labels, return_counts=True)
log_main.setLevel(20)  # This is INFO level (to filter copied file messages)
log_main.info('Successfully parsed "{}" with the following cell type class '
              'distribution [labels, counts]: {}, {}'.format(csv_p, classes,
                                                             c_cnts))

if __name__ == "__main__":
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
    new_ssd.save_dataset_deep(extract_only=True, qsub_pe='pe')
    pbar = tqdm.tqdm(total=len(ssv_ids))
    for ii, ssv_id in enumerate(ssv_ids):
        ssv = new_ssd.get_super_segmentation_object(ssv_id)
        ssv.save_attributes(["cellttype_gt"], [ssv_labels[ii]])
        ssv._render_rawviews(4, verbose=True, force_recompute=False)  # TODO: Make sure that copied files are always the same if `force_recompute` is set to False!
        pbar.update()
    write_obj2pkl(new_ssd.path + "/{}_labels.pkl".format(gt_version),
                 {ssv_ids[kk]: ssv_labels[kk] for kk in range(len(ssv_ids))})
