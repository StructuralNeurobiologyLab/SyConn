# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Sven Dorkenwald, Philipp Schubert, Joergen Kornfeld

import numpy as np

import super_segmentation as ss
import segmentation
import connectivity


def extract_connectivity_thread(args):
    sj_obj_ids = args[0]
    sj_version = args[1]
    ssd_version = args[2]
    working_dir = args[3]

    ssd = ss.SuperSegmentationDataset(working_dir,
                                      version=ssd_version)

    sd = segmentation.SegmentationDataset("sj",
                                          version=sj_version,
                                          working_dir=working_dir)

    cons = []
    for sj_obj_id in sj_obj_ids:
        sj = sd.get_segmentation_object(sj_obj_id)
        con = extract_connectivity_information(sj, ssd)
        if con is not None:
            if len(cons) == 0:
                cons = con
            else:
                cons = np.concatenate((cons, con))

    return cons


def sv_id_to_partner_ids_vec(cs_ids):
    sv_ids = np.right_shift(cs_ids, 32)
    sv_ids = np.concatenate((sv_ids[:, None],
                             (cs_ids - np.left_shift(sv_ids, 32))[:, None]),
                            axis=1)
    return sv_ids


def extract_connectivity_information(sj, ssd):
    sj.load_attr_dict()

    if not "connectivity" in sj.attr_dict:
        return

    ss_con_ids = ssd.id_changer[np.array(sj.attr_dict["connectivity"].keys(),
                                         dtype=np.int)]
    if len(ss_con_ids) == 0:
        return

    con_cnts = np.array(sj.attr_dict["connectivity"].values(), dtype=np.int)

    # Removing intracellular sjs
    ss_con_cnts = con_cnts[ss_con_ids[:, 0] != ss_con_ids[:, 1]]
    if len(ss_con_cnts) == 0:
        return

    ss_con_ids = ss_con_ids[ss_con_ids[:, 0] != ss_con_ids[:, 1]]

    # Adding the counts up
    cs_ids = np.left_shift(np.max(ss_con_ids, axis=1), 32) + \
             np.min(ss_con_ids, axis=1)
    unique_cs_ids, idx = np.unique(cs_ids, return_inverse=True)
    cs_con_cnts = np.bincount(idx, ss_con_cnts)
    cs_con_cnts = cs_con_cnts / np.sum(cs_con_cnts)

    # Going back to ssd domain
    sso_ids = np.right_shift(unique_cs_ids, 32)
    sso_ids = np.concatenate((sso_ids[:, None],
                              (unique_cs_ids -
                               np.left_shift(sso_ids, 32))[:, None]), axis=1)

    # Threshold overlap
    sso_ids = sso_ids[cs_con_cnts > .3]

    if len(sso_ids) == 0:
        return

    cs_con_cnts = cs_con_cnts[cs_con_cnts > .3]
    cs_con_cnts /= np.sum(cs_con_cnts)

    sizes = sj.size * cs_con_cnts * np.product(sj.scaling) / 1e9

    sj_ids = np.array([sj.id] * len(sizes))
    sj_types = np.array([sj.attr_dict["type_ratio"]] * len(sizes))
    sj_coords = np.array([sj.rep_coord] * len(sizes))

    return np.concatenate([sso_ids, sj_ids[:, None], sizes[:, None], sj_types[:, None], sj_coords], axis=1)


def get_sso_specific_info_thread(args):
    sso_ids = args[0]
    sj_version = args[1]
    ssd_version = args[2]
    working_dir = args[3]
    version = args[4]

    ssd = ss.SuperSegmentationDataset(working_dir,
                                      version=ssd_version)

    cm = connectivity.ConnectivityMatrix(working_dir,
                                         version=version,
                                         sj_version=sj_version,
                                         create=False)

    axoness_entries = []
    cell_types = {}
    blacklist = []
    shapes = {}
    for sso_id in sso_ids:
        print sso_id
        sso = ssd.get_super_segmentation_object(sso_id)

        if not sso.load_skeleton():
            blacklist.append(sso_id)
            continue

        if "axoness" not in sso.skeleton:
            blacklist.append(sso_id)
            continue

        if sso.cell_type is None:
            blacklist.append(sso_id)
            continue

        con_mask, pos = np.where(cm.connectivity[:, :2] == sso_id)

        sj_coords = cm.connectivity[con_mask, -3:]
        sj_axoness = sso.axoness_for_coords(sj_coords)

        con_ax = np.concatenate([con_mask[:, None], pos[:, None],
                                 sj_axoness[:, None]], axis=1)

        if len(axoness_entries) == 0:
            axoness_entries = con_ax
        else:
            axoness_entries = np.concatenate((axoness_entries, con_ax))

        cell_types[sso_id] = sso.cell_type
        shapes[sso_id] = sso.shape

    axoness_entries = np.array(axoness_entries, dtype=np.int)
    return axoness_entries, cell_types, shapes, blacklist
