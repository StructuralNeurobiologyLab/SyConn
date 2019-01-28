# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Philipp Schubert, Joergen Kornfeld
from collections import defaultdict
import numpy as np
import os
from scipy import spatial
from sklearn import ensemble, externals
from sklearn.model_selection import cross_val_score
from knossos_utils.chunky import load_dataset
from knossos_utils import knossosdataset, skeleton_utils, skeleton
knossosdataset._set_noprint(True)
import time
import datetime

from ..mp import batchjob_utils as qu
from ..mp import mp_utils as sm
from ..reps import super_segmentation, segmentation, connectivity_helper as ch
from ..reps.rep_helper import subfold_from_ix, ix_from_subfold
from ..backend.storage import AttributeDict, VoxelStorage
from ..handler.basics import chunkify
from . import log_extraction
from .. import global_params


# code for splitting 'syn' objects, which are generated as overlap between CS and SJ, see below.
def filter_relevant_syn(sd_syn, ssd):
    """
    This function filters (likely ;) ) the intra-ssv contact
    sites (inside of a ssv, not between ssvs) that do not need to be agglomerated.

    Parameters
    ----------
    sd_syn : SegmentationDataset
    ssd : SuperSegmentationDataset

    Returns
    -------
    Dict[list]
        lookup from SSV-wide synapses to SV syn. objects, keys: SSV syn ID;
        values: List of SV syn IDs
    """
    # get all cs IDs belonging to syn objects and then retrieve corresponding
    # SVs IDs via bit shift
    sv_ids = ch.sv_id_to_partner_ids_vec([syn.lookup_in_attribute_dict('cs_id')
                                          for syn in sd_syn.sos])

    syn_ids = sd_syn.ids.copy()
    # this might mean that all syn between svs with IDs>max(np.uint32) are discarded
    sv_ids[sv_ids >= len(ssd.id_changer)] = -1
    mapped_sv_ids = ssd.id_changer[sv_ids]

    mask = np.all(mapped_sv_ids > 0, axis=1)
    syn_ids = syn_ids[mask]
    filtered_mapped_sv_ids = mapped_sv_ids[mask]

    # this identifies all inter-ssv contact sites
    mask = filtered_mapped_sv_ids[:, 0] - filtered_mapped_sv_ids[:, 1] != 0
    syn_ids = syn_ids[mask]
    relevant_syns = filtered_mapped_sv_ids[mask]

    relevant_synssv_ids = np.left_shift(np.max(relevant_syns, axis=1), 32) + \
                          np.min(relevant_syns, axis=1)

    # create lookup from SSV-wide synapses to SV syn. objects
    rel_synssv_to_syn_ids = defaultdict(list)
    for i_entry in range(len(relevant_synssv_ids)):
        rel_synssv_to_syn_ids[relevant_synssv_ids[i_entry]].\
            append(syn_ids[i_entry])

    return rel_synssv_to_syn_ids


def combine_and_split_syn(wd, cs_gap_nm=300, ssd_version=None, syn_version=None,
                          stride=1000, qsub_pe=None, qsub_queue=None, nb_cpus=None,
                          resume_job=False, n_max_co_processes=None):
    """
    Creates 'syn_ssv' objects from 'syn' objects. Therefore, computes connected
    syn-objects on SSV level and aggregates the respective 'syn' attributes
    ['sj_id', 'cs_id', 'id_sj_ratio', 'id_cs_ratio', 'background_overlap_ratio',
    'cs_size', 'sj_size_pseudo']. This method requires the execution of
    'syn_gen_via_cset' (or equivalent) beforehand.

    All objects of the resulting 'syn_ssv' SegmentationDataset contain the
    following attributes:
    ['sj_ids', 'cs_ids', 'id_sj_ratio', 'id_cs_ratio', 'background_overlap_ratio',
    'neuron_partners']

    Parameters
    ----------
    wd :
    cs_gap_nm :
    ssd_version :
    syn_version :
    stride :
    qsub_pe :
    qsub_queue :
    resume_job :
    nb_cpus :
    n_max_co_processes :

    """
    ssd = super_segmentation.SuperSegmentationDataset(wd, version=ssd_version)
    syn_sd = segmentation.SegmentationDataset("syn", working_dir=wd,
                                              version=syn_version)

    rel_synssv_to_syn_ids = filter_relevant_syn(syn_sd, ssd)

    voxel_rel_paths_2stage = np.unique([subfold_from_ix(ix, 100000)[:-2]
                                        for ix in range(100000)])

    voxel_rel_paths = [subfold_from_ix(ix, 100000) for ix in range(100000)]
    block_steps = np.linspace(0, len(voxel_rel_paths),
                              int(np.ceil(float(len(rel_synssv_to_syn_ids)) /
                                          stride)) + 1).astype(np.int)
    # target SD for SSV syn objects
    sd_syn_ssv = segmentation.SegmentationDataset("syn_ssv", working_dir=wd,
                                                  version="0", create=True,
                                                  n_folders_fs=100000)

    for p in voxel_rel_paths_2stage:
        try:
            os.makedirs(sd_syn_ssv.so_storage_path + p)
        except FileExistsError:
            msg = 'SegmentationDataset of type "syn_ssv" already exists. ' \
                  '"combine_and_split_syn" is only executed once, please make sure '\
                  'that no previous SegmentationDataset exists/will be overwritten.'
            log_extraction.critical(msg)
            raise FileExistsError(msg)

    rel_synssv_to_syn_ids_items = list(rel_synssv_to_syn_ids.items())
    i_block = 0
    multi_params = []
    for block in [rel_synssv_to_syn_ids_items[i:i + stride]
                  for i in range(0, len(rel_synssv_to_syn_ids_items), stride)]:
        multi_params.append([wd, block, voxel_rel_paths[block_steps[i_block]: block_steps[i_block+1]],
                             syn_sd.version, sd_syn_ssv.version, ssd.scaling, cs_gap_nm])
        i_block += 1
    if (qsub_pe is None and qsub_queue is None) or not qu.batchjob_enabled():
        _ = sm.start_multiprocess(_combine_and_split_syn_thread,
                                  multi_params, nb_cpus=nb_cpus)

    elif qu.batchjob_enabled():
        _ = qu.QSUB_script(multi_params, "combine_and_split_syn", pe=qsub_pe,
                           resume_job=resume_job, script_folder=None,
                           queue=qsub_queue, n_max_co_processes=n_max_co_processes)
    else:
        raise Exception("QSUB not available")

    return sd_syn_ssv


def _combine_and_split_syn_thread(args):
    wd = args[0]
    rel_cs_to_cs_agg_ids_items = args[1]
    voxel_rel_paths = args[2]
    syn_version = args[3]
    cs_version = args[4]
    scaling = args[5]
    cs_gap_nm = args[6]

    sd_syn_ssv = segmentation.SegmentationDataset("syn_ssv", working_dir=wd,
                                                  version=cs_version)

    sd_syn = segmentation.SegmentationDataset("syn", working_dir=wd,
                                              version=syn_version)

    n_per_voxel_path = np.ceil(float(len(rel_cs_to_cs_agg_ids_items)) / len(voxel_rel_paths))

    n_items_for_path = 0
    cur_path_id = 0
    base_dir = sd_syn_ssv.so_storage_path + voxel_rel_paths[cur_path_id]
    os.makedirs(base_dir, exist_ok=True)
    voxel_dc = VoxelStorage(base_dir + "/voxel.pkl", read_only=False)
    attr_dc = AttributeDict(base_dir + "/attr_dict.pkl", read_only=False)

    p_parts = voxel_rel_paths[cur_path_id].strip("/").split("/")
    next_id = int("%.2d%.2d%d" % (int(p_parts[0]), int(p_parts[1]),
                                  int(p_parts[2])))

    for item in rel_cs_to_cs_agg_ids_items:
        n_items_for_path += 1

        ssv_ids = ch.sv_id_to_partner_ids_vec([item[0]])[0]

        syn = sd_syn.get_segmentation_object(item[1][0])
        syn.load_attr_dict()
        syn_attr_list = [syn.attr_dict]  # used to collect syn properties
        voxel_list = syn.voxel_list
        # store index of syn. objects for attribute dict retrieval
        synix_list = [0] * len(voxel_list)
        for syn_ix, syn_id in enumerate(item[1][1:]):
            syn_object = sd_syn.get_segmentation_object(syn_id)
            syn_object.load_attr_dict()
            syn_attr_list.append(syn_object.attr_dict)
            voxel_list = np.concatenate([voxel_list, syn_object.voxel_list])
            synix_list += [syn_ix] * len(syn_object.voxel_list)
        syn_attr_list = np.array(syn_attr_list)
        synix_list = np.array(synix_list)

        ccs = cc_large_voxel_lists(voxel_list * scaling, cs_gap_nm)
        for this_cc in ccs:
            this_cc_mask = np.array(list(this_cc))
            # retrieve the index of the syn objects selected for this CC
            this_syn_ixs = np.unique(synix_list[this_cc_mask])
            this_attr = syn_attr_list[this_syn_ixs]
            this_vx = voxel_list[this_cc_mask]
            abs_offset = np.min(this_vx, axis=0)
            this_vx -= abs_offset
            id_mask = np.zeros(np.max(this_vx, axis=0) + 1, dtype=np.bool)
            id_mask[this_vx[:, 0], this_vx[:, 1], this_vx[:, 2]] = True

            try:
                voxel_dc[next_id] = [id_mask], [abs_offset]
            except Exception:
                debug_out_fname = "{}/{}_{}_{}_{}.npy".format(
                    sd_syn_ssv.so_storage_path, next_id, abs_offset[0],
                    abs_offset[1], abs_offset[2])
                msg = "Saving syn_ssv {} failed. Debug file at {}." \
                      "".format(item, debug_out_fname)
                log_extraction.error(msg)
                np.save(debug_out_fname, this_vx)
                raise ValueError(msg)
            # aggregate syn properties:
            # ['sj_id', 'cs_id', 'id_sj_ratio', 'id_cs_ratio', 'background_overlap_ratio',
            #  'cs_size', 'sj_size_pseudo']
            syn_props_agg = {}
            for dc in this_attr:
                for k in ['background_overlap_ratio', 'id_cs_ratio', 'id_sj_ratio', 'cs_id',
                          'sj_id', 'sj_size_pseudo', 'cs_size']:
                    syn_props_agg.setdefault(k, []).append(dc[k])
            # store cs and sj IDs
            syn_props_agg['sj_ids'] = syn_props_agg['sj_id']
            del syn_props_agg['sj_id']
            syn_props_agg['cs_ids'] = syn_props_agg['cs_id']
            del syn_props_agg['cs_id']
            # calculate weighted mean of sj, cs and background ratios
            syn_props_agg['sj_size_pseudo'] = np.array(syn_props_agg['sj_size_pseudo'])
            syn_props_agg['cs_size'] = np.array(syn_props_agg['cs_size'])
            syn_props_agg['id_sj_ratio'] = np.array(syn_props_agg['id_sj_ratio'])
            syn_props_agg['id_cs_ratio'] = np.array(syn_props_agg['id_cs_ratio'])
            syn_props_agg['background_overlap_ratio'] = np.array(syn_props_agg['background_overlap_ratio'])
            sj_size_pseudo_norm = np.sum(syn_props_agg['sj_size_pseudo'])
            cs_size_norm = np.sum(syn_props_agg['cs_size'])
            sj_s_w = syn_props_agg['id_sj_ratio'] * syn_props_agg['sj_size_pseudo']
            syn_props_agg['id_sj_ratio'] = np.sum(sj_s_w) / sj_size_pseudo_norm
            back_s_w = syn_props_agg['background_overlap_ratio'] * syn_props_agg['sj_size_pseudo']
            syn_props_agg['background_overlap_ratio'] = np.sum(back_s_w) / sj_size_pseudo_norm
            cs_s_w = syn_props_agg['id_cs_ratio'] * syn_props_agg['cs_size']
            syn_props_agg['id_cs_ratio'] = np.sum(cs_s_w) / cs_size_norm
            del syn_props_agg['cs_size']
            del syn_props_agg['sj_size_pseudo']
            # add syn_ssv dict to AttributeStorage
            this_attr_dc = dict(neuron_partners=ssv_ids)
            this_attr_dc.update(syn_props_agg)
            attr_dc[next_id] = this_attr_dc
            next_id += 100000

        if n_items_for_path > n_per_voxel_path:
            # TODO: passing explicit dest_path might not be required here
            voxel_dc.push(sd_syn_ssv.so_storage_path + voxel_rel_paths[cur_path_id] +
                              "/voxel.pkl")
            attr_dc.push(sd_syn_ssv.so_storage_path + voxel_rel_paths[cur_path_id] +
                             "/attr_dict.pkl")

            cur_path_id += 1
            n_items_for_path = 0
            p_parts = voxel_rel_paths[cur_path_id].strip("/").split("/")

            next_id = int("%.2d%.2d%d" % (int(p_parts[0]), int(p_parts[1]),
                                          int(p_parts[2])))

            base_dir = sd_syn_ssv.so_storage_path + voxel_rel_paths[cur_path_id]
            os.makedirs(base_dir, exist_ok=True)
            voxel_dc = VoxelStorage(base_dir + "voxel.pkl", read_only=False)
            attr_dc = AttributeDict(base_dir + "attr_dict.pkl", read_only=False)

    if n_items_for_path > 0:
        # TODO: passing explicit dest_path might not be required here
        voxel_dc.push(sd_syn_ssv.so_storage_path + voxel_rel_paths[cur_path_id] +
                          "/voxel.pkl")
        attr_dc.push(sd_syn_ssv.so_storage_path + voxel_rel_paths[cur_path_id] +
                         "/attr_dict.pkl")


def filter_relevant_cs_agg(cs_agg, ssd):
    """
    This function filters (likely ;) ) the intra-ssv contact
    sites (inside of a ssv, not between ssvs) that do not need to be agglomerated.

    :param cs_agg:
    :param ssd:
    :return:
    """
    sv_ids = ch.sv_id_to_partner_ids_vec(cs_agg.ids)

    cs_agg_ids = cs_agg.ids.copy()

    # this might mean that all cs between svs with IDs>max(np.uint32) are discarded
    sv_ids[sv_ids >= len(ssd.id_changer)] = -1
    mapped_sv_ids = ssd.id_changer[sv_ids]

    mask = np.all(mapped_sv_ids > 0, axis=1)
    cs_agg_ids = cs_agg_ids[mask]
    filtered_mapped_sv_ids = mapped_sv_ids[mask]

    # this identifies all inter-ssv contact sites
    mask = filtered_mapped_sv_ids[:, 0] - filtered_mapped_sv_ids[:, 1] != 0
    cs_agg_ids = cs_agg_ids[mask]
    relevant_cs_agg = filtered_mapped_sv_ids[mask]

    relevant_cs_ids = np.left_shift(np.max(relevant_cs_agg, axis=1), 32) + np.min(relevant_cs_agg, axis=1)

    rel_cs_to_cs_agg_ids = defaultdict(list)
    for i_entry in range(len(relevant_cs_ids)):
        rel_cs_to_cs_agg_ids[relevant_cs_ids[i_entry]].\
            append(cs_agg_ids[i_entry])

    return rel_cs_to_cs_agg_ids


# TODO: Use this in case contact objects are required
def combine_and_split_cs_agg(wd, cs_gap_nm=300, ssd_version=None,
                             cs_agg_version=None,
                             stride=1000, qsub_pe=None, qsub_queue=None,
                             nb_cpus=None, n_max_co_processes=None):

    ssd = super_segmentation.SuperSegmentationDataset(wd, version=ssd_version)
    cs_agg = segmentation.SegmentationDataset("cs_agg", working_dir=wd,
                                              version=cs_agg_version)

    rel_cs_to_cs_agg_ids = filter_relevant_cs_agg(cs_agg, ssd)

    voxel_rel_paths_2stage = np.unique([subfold_from_ix(ix, 100000)[:-2]
                                        for ix in range(100000)])

    voxel_rel_paths = [subfold_from_ix(ix, 100000) for ix in range(100000)]
    block_steps = np.linspace(0, len(voxel_rel_paths),
                              int(np.ceil(float(len(rel_cs_to_cs_agg_ids)) / stride)) + 1).astype(np.int)

    cs = segmentation.SegmentationDataset("cs_ssv", working_dir=wd, version="new",
                                          create=True, n_folders_fs=100000)

    for p in voxel_rel_paths_2stage:
        os.makedirs(cs.so_storage_path + p)

    rel_cs_to_cs_agg_ids_items = list(rel_cs_to_cs_agg_ids.items())
    i_block = 0
    multi_params = []
    for block in [rel_cs_to_cs_agg_ids_items[i:i + stride]
                  for i in range(0, len(rel_cs_to_cs_agg_ids_items), stride)]:
        multi_params.append([wd, block,
                             voxel_rel_paths[block_steps[i_block]: block_steps[i_block+1]],
                             cs_agg.version, cs.version, ssd.scaling, cs_gap_nm])
        i_block += 1

    if (qsub_pe is None and qsub_queue is None) or not qu.batchjob_enabled():
        results = sm.start_multiprocess(_combine_and_split_cs_agg_thread,
                                        multi_params, nb_cpus=nb_cpus)

    elif qu.batchjob_enabled():
        path_to_out = qu.QSUB_script(multi_params,
                                     "combine_and_split_cs_agg",
                                     pe=qsub_pe, queue=qsub_queue,
                                     script_folder=None,
                                     n_max_co_processes=n_max_co_processes)
    else:
        raise Exception("QSUB not available")

    return cs


# TODO: Use this in case contact objects are required
def _combine_and_split_cs_agg_thread(args):
    wd = args[0]
    rel_cs_to_cs_agg_ids_items = args[1]
    voxel_rel_paths = args[2]
    cs_agg_version = args[3]
    cs_version = args[4]
    scaling = args[5]
    cs_gap_nm = args[6]

    # TODO: changed cs type to 'cs_ssv', check if that is adapted everywhere
    cs = segmentation.SegmentationDataset("cs_ssv", working_dir=wd,
                                          version=cs_version)

    cs_agg = segmentation.SegmentationDataset("cs_agg", working_dir=wd,
                                              version=cs_agg_version)

    n_per_voxel_path = np.ceil(float(len(rel_cs_to_cs_agg_ids_items)) / len(voxel_rel_paths))

    n_items_for_path = 0
    cur_path_id = 0

    try:
        os.makedirs(cs.so_storage_path + voxel_rel_paths[cur_path_id])
    except:
        pass
    voxel_dc = VoxelStorage(cs.so_storage_path + voxel_rel_paths[cur_path_id] +
                         "/voxel.pkl", read_only=False)
    attr_dc = AttributeDict(cs.so_storage_path + voxel_rel_paths[cur_path_id] +
                            "/attr_dict.pkl", read_only=False)

    p_parts = voxel_rel_paths[cur_path_id].strip("/").split("/")
    next_id = int("%.2d%.2d%d" % (int(p_parts[0]), int(p_parts[1]),
                                  int(p_parts[2])))

    for item in rel_cs_to_cs_agg_ids_items:
        n_items_for_path += 1

        ssv_ids = ch.sv_id_to_partner_ids_vec([item[0]])[0]

        voxel_list = cs_agg.get_segmentation_object(item[1][0]).voxel_list
        for cs_agg_id in item[1][1:]:
            cs_agg_object = cs_agg.get_segmentation_object(cs_agg_id)
            voxel_list = np.concatenate([voxel_list, cs_agg_object.voxel_list])

        # if len(voxel_list) < 1e4:
        #     kdtree = spatial.cKDTree(voxel_list * scaling)
        #     pairs = kdtree.query_pairs(r=cs_gap_nm)
        #     graph = nx.from_edgelist(pairs)
        #     ccs = list(nx.connected_components(graph))
        # else:
        ccs = cc_large_voxel_lists(voxel_list * scaling, cs_gap_nm)

        i_cc = 0
        for this_cc in ccs:
            this_vx = voxel_list[np.array(list(this_cc))]
            abs_offset = np.min(this_vx, axis=0)
            this_vx -= abs_offset

            id_mask = np.zeros(np.max(this_vx, axis=0) + 1, dtype=np.bool)
            id_mask[this_vx[:, 0], this_vx[:, 1], this_vx[:, 2]] = True

            try:
                voxel_dc[next_id] = [id_mask], [abs_offset]
            except:
                log_extraction.error("failed {}".format(item))
                np.save(cs.so_storage_path + "/%d_%d_%d_%d.npy" %
                        (next_id, abs_offset[0], abs_offset[1], abs_offset[2]), this_vx)

            attr_dc[next_id] = dict(neuron_partners=ssv_ids)
            next_id += 100000

        if n_items_for_path > n_per_voxel_path:
            voxel_dc.push(cs.so_storage_path + voxel_rel_paths[cur_path_id] +
                              "/voxel.pkl")
            attr_dc.push(cs.so_storage_path + voxel_rel_paths[cur_path_id] +
                             "/attr_dict.pkl")

            cur_path_id += 1
            n_items_for_path = 0
            p_parts = voxel_rel_paths[cur_path_id].strip("/").split("/")

            next_id = int("%.2d%.2d%d" % (int(p_parts[0]), int(p_parts[1]),
                                          int(p_parts[2])))

            try:
                os.makedirs(cs.so_storage_path + voxel_rel_paths[cur_path_id])
            except:
                pass

            voxel_dc = VoxelStorage(cs.so_storage_path + voxel_rel_paths[cur_path_id] +
                                 "voxel.pkl", read_only=False)
            attr_dc = AttributeDict(cs.so_storage_path +
                                    voxel_rel_paths[cur_path_id] + "attr_dict.pkl",
                                    read_only=False)

    if n_items_for_path > 0:
        voxel_dc.push(cs.so_storage_path + voxel_rel_paths[cur_path_id] +
                          "/voxel.pkl")
        attr_dc.push(cs.so_storage_path + voxel_rel_paths[cur_path_id] +
                         "/attr_dict.pkl")


def cc_large_voxel_lists(voxel_list, cs_gap_nm, max_concurrent_nodes=5000,
                         verbose=False):
    kdtree = spatial.cKDTree(voxel_list)

    checked_ids = np.array([], dtype=np.int)
    next_ids = np.array([0])
    ccs = [set(next_ids)]

    current_ccs = 0
    vx_ids = np.arange(len(voxel_list), dtype=np.int)

    while True:
        if verbose:
            log_extraction.debug("NEXT - %d - %d" % (len(next_ids),
                                                     len(checked_ids)))
            for cc in ccs:
                log_extraction.debug("N voxels in cc: %d" % (len(cc)))

        if len(next_ids) == 0:
            p_ids = vx_ids[~np.in1d(vx_ids, checked_ids)]
            if len(p_ids) == 0:
                break
            else:
                current_ccs += 1
                ccs.append(set([p_ids[0]]))
                next_ids = p_ids[:1]

        q_ids = kdtree.query_ball_point(voxel_list[next_ids], r=cs_gap_nm, )
        checked_ids = np.concatenate([checked_ids, next_ids])

        for q_id in q_ids:
            ccs[current_ccs].update(q_id)

        cc_ids = np.array(list(ccs[current_ccs]))
        next_ids = vx_ids[cc_ids[~np.in1d(cc_ids, checked_ids)][:max_concurrent_nodes]]
    return ccs


# Code for mapping SJ to CS, three different ways: via ChunkDataset (currently used), KnossosDataset, SegmentationDataset+
# TODO: SegmentationDataset version of below, probably not necessary anymore
def overlap_mapping_sj_to_cs(cs_sd, sj_sd, rep_coord_dist_nm=2000,
                             n_folders_fs=10000,
                             stride=20, qsub_pe=None, qsub_queue=None,
                             nb_cpus=None, n_max_co_processes=None):
    assert n_folders_fs % stride == 0

    wd = cs_sd.working_dir

    voxel_rel_paths = [subfold_from_ix(ix, n_folders_fs) for ix in range(n_folders_fs)]
    conn_sd = segmentation.SegmentationDataset("conn", working_dir=wd, version="new",
                                               create=True, n_folders_fs=n_folders_fs)

    for p in voxel_rel_paths:
        os.makedirs(conn_sd.so_storage_path + p)

    multi_params = []
    for block_bs in [[i, i+stride] for i in range(0, n_folders_fs, stride)]:
        multi_params.append([wd, block_bs[0], block_bs[1], conn_sd.version,
                             sj_sd.version, cs_sd.version,
                             rep_coord_dist_nm])

    if (qsub_pe is None and qsub_queue is None) or not qu.batchjob_enabled():
        results = sm.start_multiprocess(_overlap_mapping_sj_to_cs_thread,
                                        multi_params, nb_cpus=nb_cpus)

    elif qu.batchjob_enabled():
        path_to_out = qu.QSUB_script(multi_params,
                                     "overlap_mapping_sj_to_cs",
                                     pe=qsub_pe, queue=qsub_queue,
                                     script_folder=None,
                                     n_max_co_processes=n_max_co_processes)
    else:
        raise Exception("QSUB not available")


# TODO: SegmentationDataset version of below, probably not necessary anymore
def _overlap_mapping_sj_to_cs_thread(args):
    wd, block_start, block_end, conn_sd_version, sj_sd_version, cs_sd_version, \
        rep_coord_dist_nm = args

    conn_sd = segmentation.SegmentationDataset("conn", working_dir=wd,
                                               version=conn_sd_version,
                                               create=False)
    sj_sd = segmentation.SegmentationDataset("sj", working_dir=wd,
                                             version=sj_sd_version,
                                             create=False)
    cs_sd = segmentation.SegmentationDataset("cs_ssv", working_dir=wd,
                                             version=cs_sd_version,
                                             create=False)

    cs_id_assignment = np.linspace(0, len(cs_sd.ids), conn_sd.n_folders_fs+1).astype(np.int)

    sj_kdtree = spatial.cKDTree(sj_sd.rep_coords[sj_sd.sizes > sj_sd.config.entries['Sizethresholds']['sj']] * sj_sd.scaling)

    for i_cs_start_id, cs_start_id in enumerate(cs_id_assignment[block_start: block_end]):

        rel_path = subfold_from_ix(i_cs_start_id + block_start, conn_sd.n_folders_fs)

        voxel_dc = VoxelStorage(conn_sd.so_storage_path + rel_path + "/voxel.pkl",
                                read_only=False)
        attr_dc = AttributeDict(conn_sd.so_storage_path + rel_path + "/attr_dict.pkl",
                                read_only=False)

        next_conn_id = i_cs_start_id + block_start
        n_items_for_path = 0
        for cs_list_id in range(cs_start_id, cs_id_assignment[block_start + i_cs_start_id + 1]):
            cs_id = cs_sd.ids[cs_list_id]

            log_extraction.debug('CS ID: %d' % cs_id)

            cs = cs_sd.get_segmentation_object(cs_id)

            overlap_vx_l = overlap_mapping_sj_to_cs_single(cs, sj_sd,
                                                           sj_kdtree=sj_kdtree,
                                                           rep_coord_dist_nm=rep_coord_dist_nm)

            for l in overlap_vx_l:
                sj_id, overlap_vx = l

                bounding_box = [np.min(overlap_vx, axis=0),
                                np.max(overlap_vx, axis=0) + 1]

                vx = np.zeros(bounding_box[1] - bounding_box[0], dtype=np.bool)
                overlap_vx -= bounding_box[0]
                vx[overlap_vx[:, 0], overlap_vx[:, 1], overlap_vx[:, 2]] = True

                voxel_dc[next_conn_id] = [vx], [bounding_box[0]]

                attr_dc[next_conn_id] = {'sj_id': sj_id, 'cs_id': cs_id,
                                         'neuron_partners': cs.lookup_in_attribute_dict('neuron_partners')}

                next_conn_id += conn_sd.n_folders_fs
                n_items_for_path += 1

        if n_items_for_path > 0:
            voxel_dc.push(conn_sd.so_storage_path + rel_path + "/voxel.pkl")
            attr_dc.push(conn_sd.so_storage_path + rel_path + "/attr_dict.pkl")


# TODO: SegmentationDataset version of below, probably not necessary anymore
def overlap_mapping_sj_to_cs_single(cs, sj_sd, sj_kdtree=None, rep_coord_dist_nm=2000):
    cs_kdtree = spatial.cKDTree(cs.voxel_list * cs.scaling)

    if sj_kdtree is None:
        sj_kdtree = spatial.cKDTree(sj_sd.rep_coords[sj_sd.sizes > sj_sd.config.entries['Sizethresholds']['sj']] * sj_sd.scaling)

    cand_sj_ids_l = sj_kdtree.query_ball_point(cs.voxel_list * cs.scaling,
                                               r=rep_coord_dist_nm)
    u_cand_sj_ids = set()
    for l in cand_sj_ids_l:
        u_cand_sj_ids.update(l)

    if len(u_cand_sj_ids) == 0:
        return []

    u_cand_sj_ids = sj_sd.ids[sj_sd.sizes > sj_sd.config.entries['Sizethresholds']['sj']][np.array(list(u_cand_sj_ids))]

    # log_extraction.debug("%d candidate sjs" % len(u_cand_sj_ids))

    overlap_vx_l = []
    for sj_id in u_cand_sj_ids:
        sj = sj_sd.get_segmentation_object(sj_id, create=False)
        dists, _ = cs_kdtree.query(sj.voxel_list * sj.scaling,
                                   distance_upper_bound=1)

        overlap_vx = sj.voxel_list[dists == 0]
        if len(overlap_vx) > 0:
            overlap_vx_l.append([sj_id, overlap_vx])

    # log_extraction.debug("%d candidate sjs overlap" % len(overlap_vx_l))

    return overlap_vx_l


def syn_gen_via_cset(cs_sd, sj_sd, cs_cset, n_folders_fs=10000,
                     n_job_chunks=1000, qsub_pe=None, qsub_queue=None,
                     resume_job=False, nb_cpus=None, n_max_co_processes=None):
    """
    Creates SegmentationDataset of 'syn' objects from ChunkDataset of 'cs_agg'
    (result of contact_site extraction, does NOT require object extraction of
     'cs_agg' only the chunkdataset) and 'sj' dataset.
    Syn objects have the following attributes:
    ['sj_id', 'cs_id', 'id_sj_ratio', 'id_cs_ratio', 'background_overlap_ratio',
    'cs_size', 'sj_size_pseudo']

    Parameters
    ----------
    cs_sd :
    sj_sd :
    cs_cset :
    n_folders_fs :
    n_job_chunks :
    qsub_pe :
    qsub_queue :
    resume_job :
    nb_cpus :
    n_max_co_processes :

    Returns
    -------

    """
    wd = cs_sd.working_dir

    rel_sj_ids = sj_sd.ids[sj_sd.sizes > sj_sd.config.entries['Sizethresholds']['sj']]

    voxel_rel_paths = [subfold_from_ix(ix, n_folders_fs) for ix in range(n_folders_fs)]
    sd_syn = segmentation.SegmentationDataset("syn", working_dir=wd, version="0",
                                               create=True, n_folders_fs=n_folders_fs)

    for p in voxel_rel_paths:
        try:
            os.makedirs(sd_syn.so_storage_path + p)
        except FileExistsError:
            msg = 'SegmentationDataset of type "syn" already exists. ' \
                  '"syn_gen_via_cset" is only executed once, please make sure '\
                  'that no previous SegmentationDataset exists/is overwritten.'
            log_extraction.critical(msg)
            raise FileExistsError(msg)

    sj_id_blocks = np.array_split(rel_sj_ids, n_job_chunks)
    voxel_rel_path_blocks = np.array_split(voxel_rel_paths, n_job_chunks)

    multi_params = []
    for i_block in range(n_job_chunks):
        multi_params.append([wd, sj_id_blocks[i_block],
                             voxel_rel_path_blocks[i_block], sd_syn.version,
                             sj_sd.version, cs_sd.version, cs_cset.path_head_folder])

    if (qsub_pe is None and qsub_queue is None) or not qu.batchjob_enabled():
        _ = sm.start_multiprocess_imap(syn_gen_via_cset_thread,
                                       multi_params, nb_cpus=nb_cpus)

    elif qu.batchjob_enabled():
        _ = qu.QSUB_script(multi_params, "syn_gen_via_cset", pe=qsub_pe,
                           queue=qsub_queue, resume_job=resume_job,
                           script_folder=None, n_cores=nb_cpus,
                            n_max_co_processes=n_max_co_processes)
    else:
        raise Exception("QSUB not available")
    return sd_syn


def syn_gen_via_cset_thread(args):
    wd, sj_ids, voxel_rel_paths, syn_sd_version, sj_sd_version, \
        cs_sd_version, cset_path = args

    sd_syn = segmentation.SegmentationDataset("syn", working_dir=wd,
                                              version=syn_sd_version,
                                              create=False)
    sj_sd = segmentation.SegmentationDataset("sj", working_dir=wd,
                                             version=sj_sd_version,
                                             create=False)
    cs_sd = segmentation.SegmentationDataset("cs_agg", working_dir=wd,
                                             version=cs_sd_version,
                                             create=False)

    cs_cset = load_dataset(cset_path, update_paths=True)

    sj_id_blocks = np.array_split(sj_ids, len(voxel_rel_paths))

    for i_sj_id_block, sj_id_block in enumerate(sj_id_blocks):
        rel_path = voxel_rel_paths[i_sj_id_block]

        voxel_dc = VoxelStorage(sd_syn.so_storage_path + rel_path +
                                "/voxel.pkl", read_only=False)
        attr_dc = AttributeDict(sd_syn.so_storage_path + rel_path +
                                "/attr_dict.pkl", read_only=False)

        next_syn_id = ix_from_subfold(rel_path, sd_syn.n_folders_fs)

        for sj_id in sj_id_block:
            sj = sj_sd.get_segmentation_object(sj_id)
            bb = sj.bounding_box
            #  this SJ-CS overlap method will be outdated very soon
            if np.any(np.linalg.norm((bb[1] - bb[0]) * sd_syn.scaling) >
                      global_params.thresh_sj_bbd_syngen):
                log_extraction.debug(
                    'Skipped huge SJ with size: {}, offset: {}, sj_id: {}, sj_c'
                    'oord: {}'.format(sj.size, sj.bounding_box[0], sj_id,
                                      sj.rep_coord))
                continue
            vxl_sj = sj.voxels
            offset, size = bb[0], bb[1] - bb[0]
            # log_extraction.info('Loading CS chunk data of size {}'.format(size))
            cs_ids = cs_cset.from_chunky_to_matrix(size, offset, 'cs', ['cs'],
                                                   dtype=np.uint64)['cs']
            u_cs_ids, c_cs_ids = np.unique(cs_ids, return_counts=True)
            n_vxs_in_sjbb = float(np.sum(c_cs_ids))  # equivalent to np.prod(size) which is the volume spanned by the SJ bounding box
            zero_ratio = c_cs_ids[u_cs_ids == 0] / n_vxs_in_sjbb
            for cs_id in u_cs_ids:
                if cs_id == 0:
                    continue

                cs = cs_sd.get_segmentation_object(cs_id)

                id_ratio = c_cs_ids[u_cs_ids == cs_id] / n_vxs_in_sjbb
                overlap_vx = np.transpose(np.nonzero((cs_ids == cs_id) & vxl_sj)) + offset
                cs_ratio = float(len(overlap_vx)) / cs.size
                if len(overlap_vx) == 0:
                    continue

                bounding_box = [np.min(overlap_vx, axis=0),
                                np.max(overlap_vx, axis=0) + 1]
                vx_block = np.zeros(bounding_box[1] - bounding_box[0], dtype=np.bool)
                overlap_vx -= bounding_box[0]
                vx_block[overlap_vx[:, 0], overlap_vx[:, 1], overlap_vx[:, 2]] = True

                voxel_dc[next_syn_id] = [vx_block], [bounding_box[0]]
                # also store cs size and sj bounding box (equivalent to the sum of all cs voxels including background...)
                # for faster calculation  of aggregated syn properties during 'syn_ssv' generation ('combine_and_split_syn')
                attr_dc[next_syn_id] = {'sj_id': sj_id, 'cs_id': cs_id,
                                        'id_sj_ratio': id_ratio,
                                        'sj_size_pseudo': n_vxs_in_sjbb,  # TODO: still unclear why not use sj.size, PS. This would be more intuitive.
                                        'id_cs_ratio': cs_ratio,
                                        'cs_size': cs.size,
                                        'background_overlap_ratio': zero_ratio}
                next_syn_id += sd_syn.n_folders_fs

        voxel_dc.push(sd_syn.so_storage_path + rel_path + "/voxel.pkl")
        attr_dc.push(sd_syn.so_storage_path + rel_path + "/attr_dict.pkl")


# TODO: KD version of above, probably not necessary anymore
def overlap_mapping_sj_to_cs_via_kd(cs_sd, sj_sd, cs_kd,
                                    n_folders_fs=10000, n_job_chunks=1000,
                                    qsub_pe=None, qsub_queue=None,
                                    nb_cpus=None, n_max_co_processes=None):

    wd = cs_sd.working_dir

    rel_sj_ids = sj_sd.ids[sj_sd.sizes > sj_sd.config.entries['Sizethresholds']['sj']]

    voxel_rel_paths = [subfold_from_ix(ix, n_folders_fs) for ix in range(n_folders_fs)]
    conn_sd = segmentation.SegmentationDataset("conn", working_dir=wd, version="new",
                                               create=True, n_folders_fs=n_folders_fs)

    for p in voxel_rel_paths:
        os.makedirs(conn_sd.so_storage_path + p)

    sj_id_blocks = np.array_split(rel_sj_ids, n_job_chunks)
    voxel_rel_path_blocks = np.array_split(voxel_rel_paths, n_job_chunks)

    multi_params = []
    for i_block in range(n_job_chunks):
        multi_params.append([wd, sj_id_blocks[i_block],
                             voxel_rel_path_blocks[i_block], conn_sd.version,
                             sj_sd.version, cs_sd.version, cs_kd.knossos_path])

    if (qsub_pe is None and qsub_queue is None) or not qu.batchjob_enabled():
        results = sm.start_multiprocess(_overlap_mapping_sj_to_cs_via_kd_thread,
                                        multi_params, nb_cpus=nb_cpus)

    elif qu.batchjob_enabled():
        path_to_out = qu.QSUB_script(multi_params,
                                     "overlap_mapping_sj_to_cs_via_kd",
                                     pe=qsub_pe, queue=qsub_queue,
                                     script_folder=None,
                                     n_max_co_processes=n_max_co_processes)
    else:
        raise Exception("QSUB not available")
    return conn_sd


# TODO: KD version of above, probably not necessary anymore
def _overlap_mapping_sj_to_cs_via_kd_thread(args):
    wd, sj_ids, voxel_rel_paths, conn_sd_version, sj_sd_version, \
        cs_sd_version, cs_kd_path = args

    conn_sd = segmentation.SegmentationDataset("conn", working_dir=wd,
                                               version=conn_sd_version,
                                               create=False)
    sj_sd = segmentation.SegmentationDataset("sj", working_dir=wd,
                                             version=sj_sd_version,
                                             create=False)
    cs_sd = segmentation.SegmentationDataset("cs_ssv", working_dir=wd,
                                             version=cs_sd_version,
                                             create=False)

    cs_kd = knossosdataset.KnossosDataset()
    cs_kd.initialize_from_knossos_path(cs_kd_path)

    sj_id_blocks = np.array_split(sj_ids, len(voxel_rel_paths))

    for i_sj_id_block, sj_id_block in enumerate(sj_id_blocks):
        rel_path = voxel_rel_paths[i_sj_id_block]

        voxel_dc = VoxelStorage(conn_sd.so_storage_path + rel_path + "/voxel.pkl",
                                read_only=False)
        attr_dc = AttributeDict(conn_sd.so_storage_path + rel_path + "/attr_dict.pkl",
                                read_only=False)

        next_conn_id = ix_from_subfold(rel_path,
                                       conn_sd.n_folders_fs)

        for sj_id in sj_id_block:
            sj = sj_sd.get_segmentation_object(sj_id)
            vxl = sj.voxel_list

            cs_ids = cs_kd.from_overlaycubes_to_list(vxl, datatype=np.uint64)
            u_cs_ids, c_cs_ids = np.unique(cs_ids, return_counts=True)

            zero_ratio = c_cs_ids[u_cs_ids == 0] / np.sum(c_cs_ids)

            for cs_id in u_cs_ids:
                if cs_id == 0:
                    continue

                cs = cs_sd.get_segmentation_object(cs_id)
                id_ratio = c_cs_ids[u_cs_ids == cs_id] / float(np.sum(c_cs_ids))
                overlap_vx = vxl[cs_ids == cs_id]
                cs_ratio = float(len(overlap_vx)) / cs.size

                bounding_box = [np.min(overlap_vx, axis=0),
                                np.max(overlap_vx, axis=0) + 1]

                vx_block = np.zeros(bounding_box[1] - bounding_box[0], dtype=np.bool)
                overlap_vx -= bounding_box[0]
                vx_block[overlap_vx[:, 0], overlap_vx[:, 1], overlap_vx[:, 2]] = True

                voxel_dc[next_conn_id] = [vx_block], [bounding_box[0]]
                attr_dc[next_conn_id] = {'sj_id': sj_id,
                                         'cs_id': cs_id,
                                         'id_sj_ratio': id_ratio,
                                         'id_cs_ratio': cs_ratio,
                                         'background_overlap_ratio': zero_ratio}

                next_conn_id += conn_sd.n_folders_fs

        voxel_dc.push(conn_sd.so_storage_path + rel_path + "/voxel.pkl")
        attr_dc.push(conn_sd.so_storage_path + rel_path + "/attr_dict.pkl")


# Code for property extraction of contact sites (syn_ssv)

def write_conn_gt_kzips(conn, n_objects, folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

    conn_ids = conn.ids[np.random.choice(len(conn.ids), n_objects, replace=False)]

    for conn_id in conn_ids:
        obj = conn.get_segmentation_object(conn_id)
        p = folder + "/obj_%d.k.zip" % conn_id

        obj.save_kzip(p)
        obj.mesh2kzip(p)

        a = skeleton.SkeletonAnnotation()
        a.scaling = obj.scaling
        a.comment = "rep coord - %d" % obj.size

        a.addNode(skeleton.SkeletonNode().from_scratch(a, obj.rep_coord[0],
                                                       obj.rep_coord[1],
                                                       obj.rep_coord[2],
                                                       radius=1))
        skeleton_utils.write_skeleton(folder + "/obj_%d.k.zip" % conn_id, [a])


def create_syn_gt(conn, path_kzip):
    annos = skeleton_utils.loadj0126NML(path_kzip)

    label_coords = []
    labels = []
    for anno in annos:
        node = list(anno.getNodes())[0]
        labels.append(node.getComment())
        label_coords.append(np.array(node.getCoordinate()))

    labels = np.array(labels)
    label_coords = np.array(label_coords)

    conn_kdtree = spatial.cKDTree(conn.rep_coords * conn.scaling)
    ds, list_ids = conn_kdtree.query(label_coords * conn.scaling)

    synssv_ids = conn.ids[list_ids]
    mapped_synssv_objects_kzip = os.path.split(path_kzip)[0] + '/mapped_synssv.k.zip'
    # for label_id in np.where(ds > 0)[0]:
    #     dists, close_ids = conn_kdtree.query(label_coords[label_id] * conn.scaling,
    #                                          k=100)
    #
    #     for close_id in close_ids[np.argsort(dists)]:
    #         conn_o = conn.get_segmentation_object(conn.ids[close_id])
    #         if len(conn_o.mesh[1]) == 0:  # only big synapses..
    #             continue
    #         vx_ds = np.sum(np.abs(conn_o.voxel_list - label_coords[label_id]),
    #                        axis=-1)
    #         print('voxel dist:', np.min(vx_ds))
    #         synssv_ids[label_id] = conn.ids[close_id]
    #         break
    #         # if np.min(vx_ds) == 0:
    #         #     synssv_ids[label_id] = conn.ids[close_id]
    #         #     break
    #
    #     # assert 0 in vx_ds

    features = []
    skel = skeleton.Skeleton()
    anno = skeleton.SkeletonAnnotation()
    anno.scaling = conn.scaling
    for kk, synssv_id in enumerate(synssv_ids):
        synssv_o = conn.get_segmentation_object(synssv_id)
        synssv_o.mesh2kzip(mapped_synssv_objects_kzip, ext_color=None, ply_name='{}.ply'.format(synssv_id))
        n = skeleton.SkeletonNode().from_scratch(anno, synssv_o.rep_coord[0], synssv_o.rep_coord[1],
                                                 synssv_o.rep_coord[2])
        n.setComment('{}'.format(labels[kk]))
        anno.addNode(n)
        features.append(synssv_o_features(synssv_o))
    skel.add_annotation(anno)
    skel.to_kzip(mapped_synssv_objects_kzip)
    features = np.array(features)
    rfc = ensemble.RandomForestClassifier(n_estimators=200, max_features='sqrt',
                                          n_jobs=-1)
    v_features = features[labels != "ambiguous"]
    v_labels = labels[labels != "ambiguous"]
    v_labels = v_labels == "synaptic"
    v_labels = v_labels.astype(np.int)
    score = cross_val_score(rfc, v_features, v_labels, cv=10)
    log_extraction.info('Mean score +- std: {} +- {}'.format(np.mean(score),
                                                             np.std(score)))

    rfc.fit(v_features, v_labels)
    log_extraction.info('RFC importances:' + str(rfc.feature_importances_))

    model_base_dir = os.path.split(global_params.config.mpath_syn_rfc)[0]
    os.makedirs(model_base_dir, exist_ok=True)

    # unclear why there is 'rfc' after it
    externals.joblib.dump(rfc, global_params.config.mpath_syn_rfc)

    return rfc, v_features, v_labels


def synssv_o_features(synssv_o):
    """
    Collects syn_ssv feature for synapse prediction using an RFC.

    Parameters
    ----------
    synssv_o : SegmentationObject

    Returns
    -------
    List
    """
    synssv_o.load_attr_dict()

    features = [synssv_o.size,
                synssv_o.attr_dict["id_sj_ratio"],
                synssv_o.attr_dict["id_cs_ratio"]]

    partner_ids = synssv_o.lookup_in_attribute_dict("neuron_partners")
    for i_partner_id, partner_id in enumerate(partner_ids):
        features.append(synssv_o.attr_dict["n_mi_objs_%d" % i_partner_id])
        features.append(synssv_o.attr_dict["n_mi_vxs_%d" % i_partner_id])
        features.append(synssv_o.attr_dict["n_vc_objs_%d" % i_partner_id])
        features.append(synssv_o.attr_dict["n_vc_vxs_%d" % i_partner_id])
    return features


def map_objects_to_synssv(wd, obj_version=None, ssd_version=None,
                          mi_version=None, vc_version=None, max_vx_dist_nm=None,
                          max_rep_coord_dist_nm=None, qsub_pe=None,
                          qsub_queue=None, nb_cpus=None, n_max_co_processes=None):
    """
    Maps cellular organelles to syn_ssv objects. Needed for the RFC model which
    is executed in 'classify_synssv_objects'.

    Parameters
    ----------
    wd : str
    obj_version : str
    ssd_version : str
    mi_version : str
    vc_version : str
    max_vx_dist_nm : float
    max_rep_coord_dist_nm : float
    qsub_pe : str
    qsub_queue : str
    nb_cpus : int
    n_max_co_processes : int
    """
    if max_rep_coord_dist_nm is None:
        max_rep_coord_dist_nm = global_params.max_rep_coord_dist_nm
    if max_vx_dist_nm is None:
        max_vx_dist_nm = global_params.max_vx_dist_nm
    sd_syn_ssv = segmentation.SegmentationDataset("syn_ssv", working_dir=wd,
                                                  version=obj_version)

    # chunk params
    multi_params = chunkify(sd_syn_ssv.so_dir_paths, 1500)
    multi_params = [(so_dir_paths, wd, obj_version, mi_version, vc_version, ssd_version, max_vx_dist_nm,
                     max_rep_coord_dist_nm) for so_dir_paths in multi_params]

    if (qsub_pe is None and qsub_queue is None) or not qu.batchjob_enabled():
        results = sm.start_multiprocess(_map_objects_to_synssv_thread,
                                        multi_params, nb_cpus=nb_cpus)

    elif qu.batchjob_enabled():
        path_to_out = qu.QSUB_script(multi_params,
                                     "map_objects_to_synssv",
                                     pe=qsub_pe, queue=qsub_queue,
                                     script_folder=None,
                                     n_max_co_processes=n_max_co_processes)
    else:
        raise Exception("QSUB not available")


def _map_objects_to_synssv_thread(args):
    """
    Helper function of 'map_objects_to_synssv'.

    Parameters
    ----------
    args : Tuple
        see 'map_objects_to_synssv'
    """
    so_dir_paths, wd, obj_version, mi_version, vc_version, ssd_version, \
        max_vx_dist_nm, max_rep_coord_dist_nm = args
    global_params.wd = wd

    ssv = super_segmentation.SuperSegmentationDataset(working_dir=wd,
                                                      version=ssd_version)
    sd_syn_ssv = segmentation.SegmentationDataset(obj_type="syn_ssv",
                                                  working_dir=wd,
                                                  version=obj_version)
    mi_sd = segmentation.SegmentationDataset(obj_type="mi",
                                             working_dir=wd,
                                             version=mi_version)
    vc_sd = segmentation.SegmentationDataset(obj_type="vc",
                                             working_dir=wd,
                                             version=vc_version)
    for so_dir_path in so_dir_paths:
        this_attr_dc = AttributeDict(so_dir_path + "/attr_dict.pkl",
                                     read_only=False)

        for synssv_id in this_attr_dc.keys():
            synssv_o = sd_syn_ssv.get_segmentation_object(synssv_id)
            synssv_o.load_attr_dict()

            for k in list(synssv_o.attr_dict.keys()):
                if k.startswith("n_mi_"):
                    del(synssv_o.attr_dict[k])
                if k.startswith("n_vc_"):
                    del(synssv_o.attr_dict[k])

            synssv_feats = objects_to_single_synssv(
                synssv_o, ssv, mi_sd, vc_sd, max_vx_dist_nm=max_vx_dist_nm,
                max_rep_coord_dist_nm=max_rep_coord_dist_nm)

            synssv_o.attr_dict.update(synssv_feats)
            this_attr_dc[synssv_id] = synssv_o.attr_dict

        this_attr_dc.push()


def objects_to_single_synssv(synssv_o, ssv, mi_sd, vc_sd, max_vx_dist_nm=2000,
                             max_rep_coord_dist_nm=4000):
    """
    Maps cellular organelles to syn_ssv objects. Needed for the RFC model which
    is executed in 'classify_synssv_objects'.

    Parameters
    ----------
    synssv_o : SegmentationObject
    ssv : SuperSegmentationObject
    mi_sd :
    vc_sd :
    max_vx_dist_nm :
    max_rep_coord_dist_nm :

    Returns
    -------

    """
    feats = {}
    partner_ids = synssv_o.lookup_in_attribute_dict("neuron_partners")
    for i_partner_id, partner_id in enumerate(partner_ids):
        ssv_o = ssv.get_super_segmentation_object(partner_id)

        # log_extraction.debug(len(ssv_o.mi_ids))
        n_mi_objs, n_mi_vxs = map_objects_from_ssv(synssv_o, mi_sd, ssv_o.mi_ids,
                                                   max_vx_dist_nm,
                                                   max_rep_coord_dist_nm)

        # log_extraction.debug(len(ssv_o.vc_ids))
        n_vc_objs, n_vc_vxs = map_objects_from_ssv(synssv_o, vc_sd, ssv_o.vc_ids,
                                                   max_vx_dist_nm,
                                                   max_rep_coord_dist_nm)

        feats["n_mi_objs_%d" % i_partner_id] = n_mi_objs
        feats["n_mi_vxs_%d" % i_partner_id] = n_mi_vxs
        feats["n_vc_objs_%d" % i_partner_id] = n_vc_objs
        feats["n_vc_vxs_%d" % i_partner_id] = n_vc_vxs

    return feats


def map_objects_from_ssv(synssv_o, sd_obj, obj_ids, max_vx_dist_nm,
                         max_rep_coord_dist_nm):
    """
    Maps cellular organelles to syn_ssv objects. Needed for the RFC model which
    is executed in 'classify_synssv_objects'.

    Parameters
    ----------
    synssv_o : SegmentationObject
        Contact site object of SSV
    sd_obj : SegmentationObject
        Dataset of cellular object to map
    obj_ids : List[int]
        IDs of cellular objects in question
    max_vx_dist_nm : float
    max_rep_coord_dist_nm : float

    Returns
    -------

    """
    obj_mask = np.in1d(sd_obj.ids, obj_ids)

    if np.sum(obj_mask) == 0:
        return 0, 0

    obj_rep_coords = sd_obj.load_cached_data("rep_coord")[obj_mask] * \
                     sd_obj.scaling

    obj_kdtree = spatial.cKDTree(obj_rep_coords)

    close_obj_ids = sd_obj.ids[obj_mask][obj_kdtree.query_ball_point(
        synssv_o.rep_coord * synssv_o.scaling, r=max_rep_coord_dist_nm)]

    synssv_vx_kdtree = spatial.cKDTree(synssv_o.voxel_list * synssv_o.scaling)

    # log_extraction.debug(len(close_obj_ids))

    n_obj_vxs = []
    for close_obj_id in close_obj_ids:
        obj = sd_obj.get_segmentation_object(close_obj_id)
        obj_vxs = obj.voxel_list * obj.scaling

        ds, _ = synssv_vx_kdtree.query(obj_vxs,
                                       distance_upper_bound=max_vx_dist_nm)

        n_obj_vxs.append(np.sum(ds < np.inf))

    n_obj_vxs = np.array(n_obj_vxs)

    # log_extraction.debug(n_obj_vxs)
    n_objects = np.sum(n_obj_vxs > 0)
    n_vxs = np.sum(n_obj_vxs)

    return n_objects, n_vxs


def classify_synssv_objects(wd, obj_version=None, qsub_pe=None,
                            qsub_queue=None, nb_cpus=None, n_max_co_processes=None):
    """
    Classifiy SSV contact sites into snaptic or non-synaptic using an RFC model
    and stores the result in the attribute dict of the syn_ssv objects.

    Parameters
    ----------
    wd : str
    obj_version : str
    qsub_pe : str
    qsub_queue : str
    nb_cpus : int
    n_max_co_processes : int
    """
    sd_syn_ssv = segmentation.SegmentationDataset("syn_ssv", working_dir=wd,
                                                  version=obj_version)

    multi_params = chunkify(sd_syn_ssv.so_dir_paths, 1500)
    multi_params = [(so_dir_paths, wd, obj_version) for so_dir_paths in
                    multi_params]

    if (qsub_pe is None and qsub_queue is None) or not qu.batchjob_enabled():
        results = sm.start_multiprocess(_classify_synssv_objects_thread,
                                        multi_params, nb_cpus=nb_cpus)

    elif qu.batchjob_enabled():
        path_to_out = qu.QSUB_script(multi_params,
                                     "classify_synssv_objects",
                                     pe=qsub_pe, queue=qsub_queue,
                                     script_folder=None,
                                     n_max_co_processes=n_max_co_processes)
    else:
        raise Exception("QSUB not available")


def _classify_synssv_objects_thread(args):
    """
    Helper function of 'classify_synssv_objects'.

    Parameters
    ----------
    args : Tuple
        see 'classify_synssv_objects'
    """
    so_dir_paths, wd, obj_version = args

    sd_syn_ssv = segmentation.SegmentationDataset(obj_type="syn_ssv",
                                                  working_dir=wd,
                                                  version=obj_version)
    rfc = externals.joblib.load(global_params.config.mpath_syn_rfc)

    for so_dir_path in so_dir_paths:
        this_attr_dc = AttributeDict(so_dir_path + "/attr_dict.pkl",
                                     read_only=False)

        for synssv_id in this_attr_dc.keys():
            synssv_o = sd_syn_ssv.get_segmentation_object(synssv_id)
            synssv_o.load_attr_dict()

            feats = synssv_o_features(synssv_o)
            syn_prob = rfc.predict_proba([feats])[0][1]

            synssv_o.attr_dict.update({"syn_prob": syn_prob})
            this_attr_dc[synssv_id] = synssv_o.attr_dict

        this_attr_dc.push()


def collect_properties_from_ssv_partners(wd, obj_version=None, ssd_version=None,
                                         qsub_pe=None, qsub_queue=None,
                                         nb_cpus=None, n_max_co_processes=None):
    """
    Collect axoness, cell types and spiness from synaptic partners and stores
    them in syn_ssv objects. Also maps syn_type_sym_ratio to the synaptic sign
    (-1 for asym., 1 for sym. synapses).

    Parameters
    ----------
    wd : str
    obj_version : str
    ssd_version : int
    qsub_pe : str
    qsub_queue : str
    nb_cpus : int
        CPUS per job
    n_max_co_processes : int
        Number of parallel jobs
    """
    sd_syn_ssv = segmentation.SegmentationDataset("syn_ssv", working_dir=wd,
                                                  version=obj_version)

    multi_params = []
    for so_dir_paths in chunkify(sd_syn_ssv.so_dir_paths, 2000):
        multi_params.append([so_dir_paths, wd, obj_version,
                             ssd_version])
    if (qsub_pe is None and qsub_queue is None) or not qu.batchjob_enabled():
        _ = sm.start_multiprocess_imap(
            _collect_properties_from_ssv_partners_thread, multi_params,
            nb_cpus=nb_cpus)
    elif qu.batchjob_enabled():
        _ = qu.QSUB_script(
            multi_params, "collect_properties_from_ssv_partners", pe=qsub_pe,
            queue=qsub_queue, script_folder=None,
            n_max_co_processes=n_max_co_processes)
    else:
        raise Exception("QSUB not available")


def _collect_properties_from_ssv_partners_thread(args):
    """
    Helper function of 'collect_properties_from_ssv_partners'.

    Parameters
    ----------
    args : Tuple
        see 'collect_properties_from_ssv_partners'
    """
    so_dir_paths, wd, obj_version, ssd_version = args

    ssd = super_segmentation.SuperSegmentationDataset(working_dir=wd,
                                                      version=ssd_version)
    sd_syn_ssv = segmentation.SegmentationDataset(obj_type="syn_ssv",
                                                  working_dir=wd,
                                                  version=obj_version)
    for so_dir_path in so_dir_paths:
        this_attr_dc = AttributeDict(so_dir_path + "/attr_dict.pkl",
                                     read_only=False)

        for synssv_id in this_attr_dc.keys():
            synssv_o = sd_syn_ssv.get_segmentation_object(synssv_id)
            synssv_o.load_attr_dict()

            axoness = []
            spiness = []
            celltypes = []
            for ssv_partner_id in synssv_o.attr_dict["neuron_partners"]:
                ssv_o = ssd.get_super_segmentation_object(ssv_partner_id)
                ssv_o.load_attr_dict()
                # add pred_type key to global_params?
                curr_ax = ssv_o.axoness_for_coords([synssv_o.rep_coord],
                                                   pred_type='axoness_avg10000')
                axoness.append(curr_ax[0])
                # TODO: maybe use more than only a single rep_coord
                curr_sp = ssv_o.semseg_for_coords([synssv_o.rep_coord],
                                                  'spiness')
                spiness.append(curr_sp)
                celltypes.append(ssv_o.attr_dict['celltype_cnn'])
            sym_asym_ratio = synssv_o.attr_dict['syn_type_sym_ratio']
            syn_sign = -1 if sym_asym_ratio > global_params.sym_thresh else 1
            synssv_o.attr_dict.update({'partner_axoness': axoness, 'partner_spiness': spiness,
                                       'partner_celltypes': celltypes, 'syn_sign': syn_sign})
            this_attr_dc[synssv_id] = synssv_o.attr_dict

        this_attr_dc.push()


def export_matrix(obj_version=None, dest_folder=None, syn_prob_t=.5):
    """
    Writes .csv and .kzip summary file of connectivity matrix.

    Parameters
    ----------
    wd : str
    obj_version : str
    dest_folder : str
        Path to csv file
    syn_prob_t :
    """
    if dest_folder is None:
        dest_folder = global_params.config.working_dir + '/connectivity_matrix/'
    os.makedirs(os.path.split(dest_folder)[0], exist_ok=True)
    dest_name = dest_folder + '/conn_mat'
    sd_syn_ssv = segmentation.SegmentationDataset("syn_ssv", working_dir=global_params.config.working_dir,
                                                  version=obj_version)

    syn_prob = sd_syn_ssv.load_cached_data("syn_prob")

    m = syn_prob > syn_prob_t
    m_axs = sd_syn_ssv.load_cached_data("partner_axoness")[m]
    m_cts = sd_syn_ssv.load_cached_data("partner_celltypes")[m]
    m_sp = sd_syn_ssv.load_cached_data("partner_spiness")[m]
    m_coords = sd_syn_ssv.rep_coords[m]
    # m_sizes = sd_syn_ssv.sizes[m]
    m_sizes = sd_syn_ssv.load_cached_data("mesh_area")[m] / 2
    m_ssv_partners = sd_syn_ssv.load_cached_data("neuron_partners")[m]
    m_syn_prob = syn_prob[m]
    m_syn_sign = sd_syn_ssv.load_cached_data("syn_sign")[m]
    m_syn_asym_ratio = sd_syn_ssv.load_cached_data("syn_type_sym_ratio")[m]

    # (loop of skeleton node generation)
    # make sure cache-arrays have ndim == 2
    m_sizes = np.multiply(m_sizes, m_syn_sign).squeeze()[:, None]  # N, 1
    m_sp = m_sp.squeeze()  # N, 2
    m_syn_prob = m_syn_prob.squeeze()[:, None]  # N, 1
    table = np.concatenate([m_coords, m_ssv_partners, m_sizes, m_axs, m_cts,
                            m_sp, m_syn_prob], axis=1)

    # do not overwrite previous files
    if os.path.isfile(dest_name + '.csv'):
        st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
        os.rename(dest_name + '.csv', '{}_{}.csv'.format(dest_name, st))

    np.savetxt(dest_name + ".csv", table, delimiter="\t",
               header="x\ty\tz\tssv1\tssv2\tsize\tcomp1\tcomp2"
                      "\tcelltype1\tcelltype2\tspiness1\tspiness2\tsynprob")

    ax_labels = np.array(["N/A", "D", "A", "S"])
    ax_label_ids = np.array([-1, 0, 1, 2])
    # Documentation of prediction labels, maybe add somewhere to .k.zip or .csv
    ct_labels = ['N/A', 'EA', 'MSN', 'GP', 'INT']
    ct_label_ids = np.array([-1, 0, 1, 2, 3])
    sp_labels = ['N/A', 'neck', 'head', 'shaft', 'other']
    sp_label_ids = np.array([-1, 0, 1, 2, 3])

    annotations = []
    m_sizes = np.abs(m_sizes)

    ms_axs = np.sort(m_axs, axis=1)
    # vigra currently requires numpy==1.11.1
    try:
        u_axs = np.unique(ms_axs, axis=0)
    except TypeError:  # in case numpy < 1.13
        u_axs = np.vstack({tuple(row) for row in ms_axs})
    for u_ax in u_axs:
        anno = skeleton.SkeletonAnnotation()
        anno.scaling = sd_syn_ssv.scaling
        cmt = "{} - {}".format(ax_labels[ax_label_ids == u_ax[0]][0],
                               ax_labels[ax_label_ids == u_ax[1]][0])
        anno.comment = cmt
        for i_syn in np.where(np.sum(np.abs(ms_axs - u_ax), axis=1) == 0)[0]:
            c = m_coords[i_syn]
            # somewhat approximated from sphere volume:
            r = np.power(m_sizes[i_syn] / 3., 1 / 3.)
            #    r = m_sizes[i_syn]
            skel_node = skeleton.SkeletonNode(). \
            from_scratch(anno, c[0], c[1], c[2], radius=r)
            skel_node.data["ids"] = m_ssv_partners[i_syn]
            skel_node.data["size"] = m_sizes[i_syn]
            skel_node.data["syn_prob"] = m_syn_prob[i_syn]
            skel_node.data["sign"] = m_syn_sign[i_syn]
            skel_node.data["in_ex_frac"] = m_syn_asym_ratio[i_syn]
            skel_node.data['sp'] = m_sp[i_syn]
            skel_node.data['ct'] = m_cts[i_syn]
            skel_node.data['ax'] = m_axs[i_syn]
            anno.addNode(skel_node)
        annotations.append(anno)

    # do not overwrite previous files
    if os.path.isfile(dest_name + '.k.zip'):
        st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
        os.rename(dest_name + '.k.zip', '{}_{}.k.zip'.format(dest_name, st))
    skeleton_utils.write_skeleton(dest_name + ".k.zip", annotations)

