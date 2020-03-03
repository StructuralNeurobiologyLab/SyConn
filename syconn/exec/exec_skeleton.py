# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max-Planck-Institute of Neurobiology, Munich, Germany
# Authors: Philipp Schubert, Joergen Kornfeld

import numpy as np
import os
from typing import Optional
from syconn.mp import batchjob_utils as qu
from syconn.reps.super_segmentation_dataset import SuperSegmentationDataset
from syconn.handler.basics import chunkify, chunkify_weighted
from syconn.handler.config import initialize_logging
from syconn.proc.skel_based_classifier import SkelClassifier
from syconn import global_params


def run_skeleton_generation(max_n_jobs: Optional[int] = None,
                            map_myelin: Optional[bool] = None):
    """
    Generate the cell reconstruction skeletons.

    Args:
        max_n_jobs: Number of parallel jobs.
        map_myelin: Map myelin predictions at every ``skeleton['nodes']`` in
        :py:attr:`~syconn.reps.super_segmentation_object.SuperSegmentationObject.skeleton`.

    """
    if map_myelin is None:
        map_myelin = os.path.isdir(global_params.config.working_dir +
                                   '/knossosdatasets/myelin/')
    if max_n_jobs is None:
        max_n_jobs = global_params.config.ncore_total * 2
    log = initialize_logging('skeleton_generation',
                             global_params.config.working_dir + '/logs/',
                             overwrite=False)
    ssd = SuperSegmentationDataset(working_dir=global_params.config.working_dir)

    # list of SSV IDs and SSD parameters need to be given to a single QSUB job
    multi_params = ssd.ssv_ids
    nb_svs_per_ssv = np.array([len(ssd.mapping_dict[ssv_id])
                               for ssv_id in ssd.ssv_ids])
    ordering = np.argsort(nb_svs_per_ssv)
    multi_params = multi_params[ordering[::-1]]
    multi_params = chunkify(multi_params, max_n_jobs)

    # add ssd parameters
    multi_params = [(ssv_ids, ssd.version, ssd.version_dict, ssd.working_dir,
                     map_myelin) for ssv_ids in multi_params]

    # create SSV skeletons, requires SV skeletons!
    log.info('Started skeleton generation of {} SSVs.'.format(
        len(ssd.ssv_ids)))
    qu.batchjob_script(multi_params, "export_skeletons_new", log=log,
                       n_max_co_processes=global_params.config.ncore_total,
                       remove_jobfolder=True, n_cores=2)

    log.info('Finished skeleton generation.')
    # # run skeleton feature extraction # Not needed anymore, will be kept in
    # case skeleton features should remain a feature of SyConn
    # qu.batchjob_script(multi_params, "preproc_skelfeature",
    #                    n_max_co_processes=global_params.config.ncore_total,
    #                    remove_jobfolder=True)


def map_myelin_global(max_n_jobs: Optional[int] = None):
    """
    Stand-alone myelin mapping to cell reconstruction skeletons. See kwarg ``map_myelin``
    in :func:`run_skeleton_generation` for a mapping right after skeleton generation.

    Args:
        max_n_jobs: Number of parallel jobs.

    """
    if max_n_jobs is None:
        max_n_jobs = global_params.config.ncore_total * 2
    log = initialize_logging('myelin_mapping',
                             global_params.config.working_dir + '/logs/',
                             overwrite=False)
    ssd = SuperSegmentationDataset(working_dir=global_params.config.working_dir)

    # list of SSV IDs and SSD parameters need to be given to a single QSUB job
    multi_params = ssd.ssv_ids
    nb_svs_per_ssv = np.array([len(ssd.mapping_dict[ssv_id])
                               for ssv_id in ssd.ssv_ids])
    ordering = np.argsort(nb_svs_per_ssv)
    multi_params = multi_params[ordering[::-1]]
    multi_params = chunkify(multi_params, max_n_jobs)

    # add ssd parameters
    multi_params = [(ssv_ids, ssd.version, ssd.version_dict, ssd.working_dir)
                    for ssv_ids in multi_params]

    # create SSV skeletons, requires SV skeletons!
    log.info('Starting myelin mapping of {} SSVs.'.format(
        len(ssd.ssv_ids)))
    qu.batchjob_script(multi_params, "map_myelin2skel", log=log,
                       n_max_co_processes=global_params.config.ncore_total,
                       remove_jobfolder=True, n_cores=2)

    log.info('Finished myelin mapping.')


def run_skeleton_axoness():
    """
    Prepares the RFC models for skeleton-based axon inference.
    """
    # # run skeleton feature extraction # Not needed anymore, will be kept in
    # case skeleton features should remain a feature of SyConn
    sbc = SkelClassifier("axoness", working_dir=global_params.config.working_dir)
    ft_context = [1000, 2000, 4000, 8000, 12000]
    sbc.generate_data(feature_contexts_nm=ft_context, nb_cpus=global_params.config['ncores_per_node'])
    sbc.classifier_production(ft_context, nb_cpus=global_params.config['ncores_per_node'])



def run_kimimaro_skelgen(curr_dir, max_n_jobs: Optional[int] = None , map_myelin: bool = True):
    """
    Generate the cell reconstruction skeletons with the kimimaro tool. functions are in proc.sekelton, GSUB_kimimaromerge, QSUB_kimimaroskelgen

    Args:
        max_n_jobs: Number of parallel jobs.
        map_myelin: Map myelin predictions at every ``skeleton['nodes']`` in
        :py:attr:`~syconn.reps.super_segmentation_object.SuperSegmentationObject.skeleton`.
        kd_path: path to knossos dataset

    """
    if max_n_jobs is None:
        max_n_jobs = global_params.config.ncore_total * 2
    log = initialize_logging('skeleton_generation',
                             global_params.config.working_dir + '/logs/',
                             overwrite=False)

    from knossos_utils.chunky import ChunkDataset
    from knossos_utils import knossosdataset
    kd = knossosdataset.KnossosDataset()
    kd.initialize_from_knossos_path(global_params.config['paths']['kd_seg'])
    cube_size = np.array([1024, 1024, 512])
    cd = ChunkDataset()
    overlap = np.array([100, 100, 50])
    boundary = (kd.boundary/2).astype(int) #if later working on mag=2
    cd.initialize(kd, boundary, cube_size, '~/cd_tmp/',
                  box_coords=[0, 0, 0],
                  fit_box_size=True)
    multi_params = [(cube_size, offset, overlap) for offset in cd.coord_dict]
    out_dir = qu.batchjob_script(multi_params, "kimimaroskelgen", log=log,
                   n_max_co_processes=global_params.config.ncore_total, remove_jobfolder=False)

    import os
    try:
        import cPickle as pkl
    except ImportError:
        import pickle as pkl
    from syconn.handler.basics import load_pkl2obj, write_obj2pkl
    import shutil
    from syconn.reps.super_segmentation_object import SuperSegmentationObject

    ssd = SuperSegmentationDataset(working_dir=global_params.config.working_dir)

    # list of SSV IDs and SSD parameters need to be given to a single QSUB job

    path_dic = {ssv_id: [] for ssv_id in ssd.ssv_ids}
    for f in os.listdir(out_dir):
        partial_skels = load_pkl2obj(out_dir + "/" + f)
        for cell_id in partial_skels:
            path_dic[cell_id].append(out_dir + "/" + f)
    pathdict_filepath = ("%s/excube1_path_dict.pkl" % global_params.config.working_dir)
    write_obj2pkl(pathdict_filepath, path_dic)
    multi_params = ssd.ssv_ids
    ssv_sizes = np.array([ssv.size for ssv in ssd.ssvs])
    multi_params = chunkify_weighted(multi_params, max_n_jobs, ssv_sizes)

    # add ssd parameters needed for merging of skeleton, ssv_ids, path to folder for kzip files
    zipname = ("%s/excube1_kimimaro_skels_binaryfillingc100dps4/" % curr_dir)
    if not os.path.exists(zipname):
        os.mkdir(zipname)
    multi_params = [(pathdict_filepath, ssv_id, zipname) for ssv_id in multi_params]
    # create SSV skeletons, requires SV skeletons!
    log.info('Starting skeleton generation of {} SSVs.'.format(
        len(ssd.ssv_ids)))
    qu.batchjob_script(multi_params, "kimimaromerge", log=log,
                   n_max_co_processes=global_params.config.ncore_total,
                   remove_jobfolder=True, n_cores=2)
    shutil.rmtree(out_dir + "/../")

    if map_myelin:
        map_myelin_global()
    log.info('Finished skeleton generation.')



