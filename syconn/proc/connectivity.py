# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Sven Dorkenwald, Philipp Schubert, Joergen Kornfeld

from collections import defaultdict
import glob
import numpy as np
import os

from ..reps import segmentation, connectivity_helper as cph, \
    super_segmentation_object as ss, rep_helper

from ..mp import qsub_utils as qu
from ..mp import shared_mem as sm

script_folder = os.path.abspath(os.path.dirname(__file__) + "/../QSUB_scripts/")


def filter_relevant_cs_agg(cs_agg, ssd):
    sv_ids = ch.sv_id_to_partner_ids_vec(cs_agg.ids)

    cs_agg_ids = cs_agg.ids.copy()

    sv_ids[sv_ids >= len(ssd.id_changer)] = -1
    mapped_sv_ids = ssd.id_changer[sv_ids]

    mask = np.all(mapped_sv_ids > 0, axis=1)
    cs_agg_ids = cs_agg_ids[mask]
    filtered_mapped_sv_ids = mapped_sv_ids[mask]

    mask = filtered_mapped_sv_ids[:, 0] - filtered_mapped_sv_ids[:, 1] != 0
    cs_agg_ids = cs_agg_ids[mask]
    relevant_cs_agg = filtered_mapped_sv_ids[mask]

    relevant_cs_ids = np.left_shift(np.max(relevant_cs_agg, axis=1), 32) + np.min(relevant_cs_agg, axis=1)

    rel_cs_to_cs_agg_ids = defaultdict(list)
    for i_entry in range(len(relevant_cs_ids)):
        rel_cs_to_cs_agg_ids[relevant_cs_ids[i_entry]].\
            append(cs_agg_ids[i_entry])

    return rel_cs_to_cs_agg_ids


def combine_and_split_cs_agg(wd, cs_gap_nm=300, ssd_version=None,
                             cs_agg_version=None,
                             stride=1000, qsub_pe=None, qsub_queue=None,
                             nb_cpus=None, n_max_co_processes=None):

    ssd = ss.SuperSegmentationDataset(wd, version=ssd_version)
    cs_agg = segmentation.SegmentationDataset("cs_agg", working_dir=wd,
                                              version=cs_agg_version)

    rel_cs_to_cs_agg_ids = filter_relevant_cs_agg(cs_agg, ssd)

    voxel_rel_paths_2stage = np.unique([rep_helper.subfold_from_ix(ix)[:-2]
                                        for ix in range(100000)])

    voxel_rel_paths = [rep_helper.subfold_from_ix(ix) for ix in range(100000)]
    block_steps = np.linspace(0, len(voxel_rel_paths),
                              int(np.ceil(float(len(rel_cs_to_cs_agg_ids)) / stride)) + 1).astype(np.int)

    cs = segmentation.SegmentationDataset("cs", working_dir=wd, version="new",
                                          create=True)

    for p in voxel_rel_paths_2stage:
        os.makedirs(cs.so_storage_path + p)

    rel_cs_to_cs_agg_ids_items = rel_cs_to_cs_agg_ids.items()
    i_block = 0
    multi_params = []
    for block in [rel_cs_to_cs_agg_ids_items[i:i + stride]
                  for i in range(0, len(rel_cs_to_cs_agg_ids_items), stride)]:
        multi_params.append([wd, block,
                             voxel_rel_paths[block_steps[i_block]: block_steps[i_block+1]],
                             cs_agg.version, cs.version, ssd.scaling, cs_gap_nm])
        i_block += 1

    if qsub_pe is None and qsub_queue is None:
        results = sm.start_multiprocess(cph.combine_and_split_cs_agg_helper,
                                        multi_params, nb_cpus=nb_cpus)

    elif qu.__QSUB__:
        path_to_out = qu.QSUB_script(multi_params,
                                     "combine_and_split_cs_agg",
                                     pe=qsub_pe, queue=qsub_queue,
                                     script_folder=script_folder,
                                     n_max_co_processes=n_max_co_processes)
    else:
        raise Exception("QSUB not available")


def map_objects_to_cs(wd, cs_version=None, ssd_version=None, max_map_dist_nm=2000,
                      obj_types=("sj", "mi", "vc"), stride=1000, qsub_pe=None,
                      qsub_queue=None, nb_cpus=1, n_max_co_processes=100):
    cs_dataset = segmentation.SegmentationDataset("cs", version=cs_version,
                                                  working_dir=wd)
    paths = glob.glob(cs_dataset.so_storage_path + "/*/*/*")

    multi_params = []
    for path_block in [paths[i:i + stride]
                       for i in xrange(0, len(paths), stride)]:
        multi_params.append([path_block, obj_types, cs_version, ssd_version,
                             wd, max_map_dist_nm])

    if qsub_pe is None and qsub_queue is None:
        results = sm.start_multiprocess(cph.map_objects_to_cs_thread,
                                        multi_params, nb_cpus=nb_cpus)

    elif qu.__QSUB__:
        path_to_out = qu.QSUB_script(multi_params,
                                     "map_objects_to_cs",
                                     pe=qsub_pe, queue=qsub_queue,
                                     script_folder=script_folder,
                                     n_max_co_processes=n_max_co_processes)
    else:
        raise Exception("QSUB not available")
