# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max-Planck-Institute of Neurobiology, Munich, Germany
# Authors: Philipp Schubert, Joergen Kornfeld

from syconn.reps.super_segmentation import *
import pandas

# based on /wholebrain/songbird/j0126/areaxfs_v6/

str2int_label = dict(STN=0, DA=1, MSN=2, LMAN=3, HVC=4, GPe=5, FS=6, TAN=7)
csv_p = '/wholebrain/songbird/j0126/axon_type_gt.csv'
gt = df = pandas.io.parsers.read_csv(csv_p).values
ssv_ids = df[:, 0].astype(np.uint)
str_labels = df[:, 1]
ssv_labels = [str2int_label[el] for el in str_labels]


if __name__ == "__main__":
    WD ="/wholebrain/songbird/j0126/areaxfs_v6/"
    ssd = SuperSegmentationDataset(working_dir=WD)
    orig_ssvs = ssd.get_super_segmentation_object(ssv_ids)
    for ssv in orig_ssvs:
        assert os.path.isfile(ssv.attr_dict_path)
    gt_version = "ctgt_v2"
    new_ssd = SuperSegmentationDataset(working_dir=WD, version=gt_version)
    copy_ssvs2new_SSD_simple(orig_ssvs, new_version=gt_version,
                             target_wd=WD)
    new_ssd.save_dataset_deep(extract_only=True, qsub_pe='pe')
    pbar = tqdm.tqdm(total=len(ssv_ids))
    for ii, ssv_id in enumerate(ssv_ids):
        ssv = new_ssd.get_super_segmentation_object(ssv_id)
        ssv.save_attributes(["cellttype_gt"], [ssv_labels[ii]])
        ssv._render_rawviews(4, verbose=True)
        pbar.update()
    write_obj2pkl(new_ssd.path + "/{}_labels.pkl".format(gt_version),
                 {ssv_ids[kk]: ssv_labels[kk] for kk in range(len(ssv_ids))})
