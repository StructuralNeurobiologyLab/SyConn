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

# INT and ? are label 8, GPe and GP are label 5
str2int_label = dict(STN=0, DA=1, MSN=2, LMAN=3, HVC=4, GP=5, FS=6, TAN=7, GPe=5, INT=8)
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
    # for ssv in orig_ssvs:
    #     if not (ssv.id in ssd.ssv_ids and os.path.isfile(ssv.attr_dict_path)):
    #         msg = 'GT file contains SSV with ID {} which is not part of the ' \
    #               'used SSD.'.format(ssv.id)
    #         log_main.warning(msg)
    #         raise ValueError(msg)
    gt_version = "ctgt_v2"
    new_ssd = SuperSegmentationDataset(working_dir=WD, version=gt_version)
    # copy_ssvs2new_SSD_simple(orig_ssvs, new_version=gt_version,
    #                          target_wd=WD, safe=False)  # `safe=False` will overwrite existing data
    # new_ssd.save_dataset_deep(extract_only=True, qsub_pe='pe')
    pbar = tqdm.tqdm(total=len(ssv_ids))
    for ii, ssv_id in enumerate(ssv_ids):
        ssv = new_ssd.get_super_segmentation_object(ssv_id)
        ssv.save_attributes(["cellttype_gt"], [ssv_labels[ii]])
        # # previous run for "normal" bootstrap N-view predictions
        # ssv._render_rawviews(4, verbose=True, force_recompute=True)  # TODO: Make sure that copied files are always the same if `force_recompute` is set to False!
        # # Large FoV bootstrapping
        # # downsample vertices and get ~3 locations per comp_window
        verts = ssv.mesh[1].reshape(-1, 3)
        comp_window = 40960  # nm  -> pixel size: 80 nm
        ds_factor = comp_window / 3
        # get unique array of downsampled vertex locations (scaled back to nm)
        verts_ixs = np.arange(len(verts))
        np.random.seed(0)
        np.random.shuffle(verts_ixs)
        ds_locs_encountered = {}
        rendering_locs = []
        for kk, c in enumerate(verts[verts_ixs]):
            ds_loc = tuple((c / ds_factor).astype(np.int))
            if ds_loc in ds_locs_encountered:  # always gets first coordinate which is in downsampled voxel, the others are skipped
                continue
            rendering_locs.append(c)
            ds_locs_encountered[ds_loc] = None
        rendering_locs = np.array(rendering_locs)
        views = render_sso_coords(ssv, rendering_locs, verbose=True, ws=(512, 512), add_cellobjects=True,
                                  return_rot_mat=False, comp_window=comp_window, nb_views=4)
        ssv.save_views(views, view_key="4_large_fov")

        pbar.update()
    write_obj2pkl(new_ssd.path + "/{}_labels.pkl".format(gt_version),
                 {ssv_ids[kk]: ssv_labels[kk] for kk in range(len(ssv_ids))})
