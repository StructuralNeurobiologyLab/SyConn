# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max-Planck-Institute of Neurobiology, Munich, Germany
# Authors: Philipp Schubert, Alexandra Rother, Joergen Kornfeld

import shutil
import os
import glob
from typing import Optional, Union

import numpy as np

from knossos_utils.chunky import ChunkDataset
from knossos_utils import knossosdataset

from syconn.reps.super_segmentation_dataset import SuperSegmentationDataset
from syconn.handler.basics import chunkify, chunkify_weighted
from syconn.handler.config import initialize_logging
from syconn.mp import batchjob_utils as qu
from syconn.proc.skel_based_classifier import SkelClassifier
from syconn import global_params
from syconn.mp.mp_utils import start_multiprocess_imap
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
    multi_params = multi_params[np.argsort(ssd.load_cached_data('size'))[::-1]]
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


def map_myelin_global(max_n_jobs: Optional[int] = None):
    """
    Stand-alone myelin mapping to cell reconstruction skeletons. See kwarg ``map_myelin``
    in :func:`run_skeleton_generation` for a mapping right after skeleton generation.

    Args:
        max_n_jobs: Number of parallel jobs.

    """
    if max_n_jobs is None:
        max_n_jobs = global_params.config.ncore_total * 2
    log = initialize_logging('myelin_mapping', global_params.config.working_dir + '/logs/', overwrite=False)
    ssd = SuperSegmentationDataset(working_dir=global_params.config.working_dir)

    # list of SSV IDs and SSD parameters need to be given to a single QSUB job
    multi_params = ssd.ssv_ids
    multi_params = multi_params[np.argsort(ssd.load_cached_data('size'))[::-1]]
    multi_params = chunkify(multi_params, max_n_jobs)

    # add ssd parameters
    multi_params = [(ssv_ids, ssd.version, ssd.version_dict, ssd.working_dir)
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
                         cube_size: np.ndarray = None, cube_of_interest_bb: Optional[tuple] = None,
                         ds: Optional[np.ndarray] = None):
    """
    Generate the cell reconstruction skeletons with the kimimaro tool. functions are in
    proc.sekelton, GSUB_kimimaromerge, QSUB_kimimaroskelgen

    Args:
        max_n_jobs: Number of parallel jobs.
        map_myelin: Map myelin predictions at every ``skeleton['nodes']`` in
            :py:attr:`~syconn.reps.super_segmentation_object.SuperSegmentationObject.skeleton`.
        cube_size: Cube size used within each worker. This should be as big as possible to prevent
            un-centered skeletons in cell compartments with big diameters. In mag 1 voxels.
        cube_of_interest_bb: Partial volume of the data set. Bounding box in mag 1 voxels: (lower coord, upper coord)
        ds: Downsampling.
    """
    if not os.path.exists(global_params.config.temp_path):
        os.mkdir(global_params.config.temp_path)
    tmp_dir = global_params.config.temp_path + '/skel_gen/'
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)
    if max_n_jobs is None:
        max_n_jobs = global_params.config.ncore_total * 2
    if ds is None:
        ds = global_params.config['scaling'][2] // np.array(global_params.config['scaling'])
        assert np.all(ds > 0)
    log = initialize_logging('skeleton_generation', global_params.config.working_dir + '/logs/',
                             overwrite=False)

    kd = knossosdataset.KnossosDataset()
    kd.initialize_from_knossos_path(global_params.config['paths']['kd_seg'])
    cd = ChunkDataset()
    # TODO: cube_size should be voxel size dependent
    if cube_size is None:
        cube_size = np.array([1024, 1024, 512])  # this is in mag1
    if cube_of_interest_bb is not None:
        cube_of_interest_bb = np.array(cube_of_interest_bb, dtype=np.int)
    else:
        cube_of_interest_bb = np.array([[0, 0, 0], kd.boundary], dtype=np.int)

    dataset_size = (cube_of_interest_bb[1] - cube_of_interest_bb[0])

    if np.all(cube_size > dataset_size):
        cube_size = dataset_size

    cd.initialize(kd, dataset_size, cube_size, f'{tmp_dir}/cd_tmp_skel/',
                  box_coords=cube_of_interest_bb[0], fit_box_size=True)
    multi_params = [(cube_size, offs, cube_of_interest_bb, ds) for offs in
                    chunkify(list(cd.coord_dict.keys()), max_n_jobs)]
    # high memory load
    out_dir = qu.batchjob_script(multi_params, "kimimaroskelgen", log=log, remove_jobfolder=False,
                                 n_cores=2, max_iterations=10)

    ssd = SuperSegmentationDataset(working_dir=global_params.config.working_dir)

    # list of SSV IDs and SSD parameters need to be given to each batch job
    path_dc = {ssv_id: [] for ssv_id in ssd.ssv_ids}
    log.info('Cube-wise skeleton generation finished. Generating cells-to-cubes dict.')
    res = start_multiprocess_imap(_collect_paths, glob.glob(out_dir + '*_ids.pkl'), nb_cpus=None)
    for dc in res:
        for k, v in dc.items():
            path_dc[k].append(v[:-8] + '.pkl')
    pathdict_filepath = f"{tmp_dir}/excube1_path_dict.pkl"
    write_obj2pkl(pathdict_filepath, path_dc)
    del path_dc

    multi_params = chunkify_weighted(ssd.ssv_ids, max_n_jobs, ssd.load_cached_data('size'))

    # add ssd parameters needed for merging of skeleton, ssv_ids, path to folder for kzip files
    zipname = ("%s/excube1_kimimaro_skels_binaryfillingc100dps4/" % tmp_dir)
    if not os.path.exists(zipname):
        os.mkdir(zipname)
    multi_params = [(pathdict_filepath, ssv_id, zipname) for ssv_id in multi_params]
    # create SSV skeletons, requires SV skeletons!
    log.info('Merging cube-wise skeletons of {} SSVs.'.format(len(ssd.ssv_ids)))
    # high memory load
    qu.batchjob_script(multi_params, "kimimaromerge", log=log, remove_jobfolder=True, n_cores=1,
                       max_iterations=10)

    if map_myelin:
        map_myelin_global()

    shutil.rmtree(tmp_dir)
    shutil.rmtree(os.path.abspath(out_dir + '/../'))

    log.info('Finished skeleton generation.')


def _collect_paths(p: str) -> dict:
    partial_res = load_pkl2obj(p)
    res = {cellid: p for cellid in partial_res}
    return res
