# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max-Planck-Institute of Neurobiology, Munich, Germany
# Authors: Philipp Schubert, Joergen Kornfeld

import numpy as np
import shutil
import os
from typing import Optional, Union

from knossos_utils.chunky import ChunkDataset
from knossos_utils import knossosdataset

from syconn.reps.super_segmentation_dataset import SuperSegmentationDataset
from syconn.handler.basics import chunkify, chunkify_weighted
from syconn.handler.config import initialize_logging
from syconn.mp import batchjob_utils as qu
from syconn.proc.skel_based_classifier import SkelClassifier
from syconn import global_params
from syconn.handler.basics import load_pkl2obj, write_obj2pkl


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
        map_myelin = os.path.isdir(global_params.config.working_dir + '/knossosdatasets/myelin/')
    if max_n_jobs is None:
        max_n_jobs = global_params.config.ncore_total * 2
    log = initialize_logging('ssd_generation', global_params.config.working_dir + '/logs/', overwrite=False)
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
                       remove_jobfolder=True, n_cores=2)

    log.info('Finished skeleton generation.')
    # # run skeleton feature extraction # Not needed anymore, will be kept in
    # case skeleton features should remain a feature of SyConn
    # qu.batchjob_script(multi_params, "preproc_skelfeature",
    #                    remove_jobfolder=True)


def map_myelin_global(max_n_jobs: Optional[int] = None,
                      cube_of_interest_bb: Union[Optional[tuple], np.ndarray] = None):
    """
    Stand-alone myelin mapping to cell reconstruction skeletons. See kwarg ``map_myelin``
    in :func:`run_skeleton_generation` for a mapping right after skeleton generation.

    Args:
        max_n_jobs: Number of parallel jobs.
        cube_of_interest_bb: Optional bounding box (in mag 1 voxel coordinates). If given,
            translates the skeleton nodes coordinates by the offset ``cube_of_interest_bb[0]`` to
            match the coordinate frame of the complete data set volume.

    """
    if max_n_jobs is None:
        max_n_jobs = global_params.config.ncore_total * 2
    log = initialize_logging('myelin_mapping', global_params.config.working_dir + '/logs/', overwrite=False)
    ssd = SuperSegmentationDataset(working_dir=global_params.config.working_dir)

    # list of SSV IDs and SSD parameters need to be given to a single QSUB job
    multi_params = ssd.ssv_ids
    nb_svs_per_ssv = np.array([len(ssd.mapping_dict[ssv_id]) for ssv_id in ssd.ssv_ids])
    ordering = np.argsort(nb_svs_per_ssv)
    multi_params = multi_params[ordering[::-1]]
    multi_params = chunkify(multi_params, max_n_jobs)

    # add ssd parameters
    multi_params = [(ssv_ids, ssd.version, ssd.version_dict, ssd.working_dir, cube_of_interest_bb)
                    for ssv_ids in multi_params]

    # create SSV skeletons, requires SV skeletons!
    log.info('Starting myelin mapping of {} SSVs.'.format(len(ssd.ssv_ids)))
    qu.batchjob_script(multi_params, "map_myelin2skel", log=log, remove_jobfolder=True, n_cores=2)

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


def run_kimimaro_skelgen(max_n_jobs: Optional[int] = None, map_myelin: bool = True,
                         cube_size: np.ndarray = None, cube_of_interest_bb: Optional[tuple] = None):
    """
    Generate the cell reconstruction skeletons with the kimimaro tool. functions are in
    proc.sekelton, GSUB_kimimaromerge, QSUB_kimimaroskelgen

    Args:
        max_n_jobs: Number of parallel jobs.
        map_myelin: Map myelin predictions at every ``skeleton['nodes']`` in
            :py:attr:`~syconn.reps.super_segmentation_object.SuperSegmentationObject.skeleton`.
        cube_size: Cube size used within each worker. This should be as big as possible to prevent
            un-centered skeletons in cell compartments with big diameters.
        cube_of_interest_bb: Partial volume of the data set. Bounding box in mag 1 voxels: (lower coord, upper coord)

    """
    if not os.path.exists(global_params.config.temp_path):
        os.mkdir(global_params.config.temp_path)
    tmp_dir = global_params.config.temp_path + '/skel_gen/'
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)
    if max_n_jobs is None:
        max_n_jobs = global_params.config.ncore_total * 2
    log = initialize_logging('skeleton_generation', global_params.config.working_dir + '/logs/',
                             overwrite=False)

    kd = knossosdataset.KnossosDataset()
    kd.initialize_from_knossos_path(global_params.config['paths']['kd_seg'])
    cd = ChunkDataset()
    if cube_size is None:
        cube_size = np.array([1024, 1024, 512])
    overlap = np.array([100, 100, 50])
    if cube_of_interest_bb is not None:
        cube_of_interest_bb = np.array(cube_of_interest_bb, dtype=np.int)
    else:
        cube_of_interest_bb = np.array([[0, 0, 0], kd.boundary], dtype=np.int)

    # TODO: factor 1/2 must be adapted if anisotropic downsampling is used in KD!
    dataset_size = (cube_of_interest_bb[1] - cube_of_interest_bb[0]) // 2
    # if later working on mag=2
    if np.all(cube_size > dataset_size):
        cube_size = dataset_size

    # TODO: factor 1/2 in box_coords must be adapted if anisotropic downsampling is used in KD!
    cd.initialize(kd, dataset_size, cube_size, f'{tmp_dir}/cd_tmp_skel/',
                  box_coords=cube_of_interest_bb[0] // 2, fit_box_size=True)
    multi_params = [(cube_size, off, overlap, cube_of_interest_bb) for off in cd.coord_dict]
    out_dir = qu.batchjob_script(multi_params, "kimimaroskelgen", log=log, remove_jobfolder=False, n_cores=4)

    ssd = SuperSegmentationDataset(working_dir=global_params.config.working_dir)

    # list of SSV IDs and SSD parameters need to be given to each batch job
    path_dic = {ssv_id: [] for ssv_id in ssd.ssv_ids}
    for f in os.listdir(out_dir):
        partial_skels = load_pkl2obj(out_dir + "/" + f)
        for cell_id in partial_skels:
            path_dic[cell_id].append(out_dir + "/" + f)
    pathdict_filepath = ("%s/excube1_path_dict.pkl" % tmp_dir)
    write_obj2pkl(pathdict_filepath, path_dic)
    multi_params = ssd.ssv_ids
    ssv_sizes = np.array([ssv.size for ssv in ssd.ssvs])
    multi_params = chunkify_weighted(multi_params, max_n_jobs, ssv_sizes)

    # add ssd parameters needed for merging of skeleton, ssv_ids, path to folder for kzip files
    zipname = ("%s/excube1_kimimaro_skels_binaryfillingc100dps4/" % tmp_dir)
    if not os.path.exists(zipname):
        os.mkdir(zipname)
    multi_params = [(pathdict_filepath, ssv_id, zipname) for ssv_id in multi_params]
    # create SSV skeletons, requires SV skeletons!
    log.info('Starting skeleton generation of {} SSVs.'.format(
        len(ssd.ssv_ids)))
    qu.batchjob_script(multi_params, "kimimaromerge", log=log, remove_jobfolder=True, n_cores=2)

    if map_myelin:
        map_myelin_global(cube_of_interest_bb=cube_of_interest_bb)

    shutil.rmtree(tmp_dir)
    shutil.rmtree(out_dir)

    log.info('Finished skeleton generation.')
