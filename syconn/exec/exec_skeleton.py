# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max-Planck-Institute of Neurobiology, Munich, Germany
# Authors: Philipp Schubert, Alexandra Rother, Joergen Kornfeld

import shutil
import os
import glob
from typing import Optional, Union, Tuple

import numpy as np

from knossos_utils.chunky import ChunkDataset
from knossos_utils import knossosdataset

from syconn.reps.super_segmentation_dataset import SuperSegmentationDataset
from syconn.handler.basics import chunkify, chunkify_weighted, chunkify_successive
from syconn.handler.config import initialize_logging
from syconn.mp import batchjob_utils as qu
from syconn.proc.skel_based_classifier import SkelClassifier
from syconn import global_params
from syconn.mp.mp_utils import start_multiprocess_imap
from syconn.handler.basics import load_pkl2obj, write_obj2pkl, cset_cube_of_interest_parms, kd_factory


def run_skeleton_generation(map_myelin: Optional[bool] = None, ncores_skelgen: int = 2,
                            cube_size: Optional[np.ndarray] = None):
    """

    Args:
        map_myelin: Map myelin predictions at every ``skeleton['nodes']`` in
            :py:attr:`~syconn.reps.super_segmentation_object.SuperSegmentationObject.skeleton`.
        ncores_skelgen: Number of cores used during skeleton generation.
    """
    if global_params.config.use_kimimaro:
        # volume-based
        run_kimimaro_skeletonization(map_myelin=map_myelin, ncores_skelgen=ncores_skelgen, cube_size=cube_size)
    else:
        # SSV-based skeletonization on mesh vertices, not centered. Does not require cube_of_interest_bb
        run_skeleton_generation_fallback(map_myelin=map_myelin)


def run_skeleton_generation_fallback(max_n_jobs: Optional[int] = None, map_myelin: Optional[bool] = None):
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
    multi_params = multi_params[np.argsort(ssd.load_numpy_data('size'))[::-1]]
    multi_params = chunkify(multi_params, max_n_jobs)

    # add ssd parameters
    multi_params = [(ssv_ids, ssd.version, ssd.version_dict, ssd.working_dir,
                     map_myelin) for ssv_ids in multi_params]

    # create SSV skeletons, requires SV skeletons!
    log.info('Started skeleton generation of {} SSVs.'.format(
        len(ssd.ssv_ids)))
    qu.batchjob_script(multi_params, "export_skeletons_fallback", log=log,
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
    myelin_kd_p = global_params.config.working_dir + '/knossosdatasets/myelin/'
    if not os.path.isdir(myelin_kd_p):
        raise ValueError(f'Could not find myelin KnossosDataset at {myelin_kd_p}.')
    log = initialize_logging('myelin_mapping', global_params.config.working_dir + '/logs/', overwrite=False)
    ssd = SuperSegmentationDataset(working_dir=global_params.config.working_dir)

    # list of SSV IDs and SSD parameters need to be given to a single QSUB job
    multi_params = ssd.ssv_ids
    multi_params = multi_params[np.argsort(ssd.load_numpy_data('size'))[::-1]]
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


def run_kimimaro_skeletonization(max_n_jobs: Optional[int] = None, map_myelin: Optional[bool] = None,
                                 cube_size: np.ndarray = None, ds: Optional[np.ndarray] = None,
                                 ncores_skelgen: int = 2):
    """
    Generate the cell reconstruction skeletons with the kimimaro tool. functions are in
    proc.sekelton, GSUB_kimimaromerge, QSUB_kimimaroskelgen

    Args:
        max_n_jobs: Number of parallel jobs.
        map_myelin: Map myelin predictions at every ``skeleton['nodes']`` in
            :py:attr:`~syconn.reps.super_segmentation_object.SuperSegmentationObject.skeleton`.
        cube_size: Cube size used within each worker. This should be as big as possible to prevent
            un-centered skeletons in cell compartments with big diameters. In mag 1 voxels.
        ds: Downsampling.
        ncores_skelgen: Number of cores used during skeleton generation.
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
    if map_myelin is None:
        map_myelin = os.path.isdir(global_params.config.working_dir + '/knossosdatasets/myelin/')
    log = initialize_logging('skeleton_generation', global_params.config.working_dir + '/logs/',
                             overwrite=False)

    kd = kd_factory(global_params.config['paths']['kd_seg'])
    cd = ChunkDataset()
    # TODO: cube_size should be voxel size dependent
    if cube_size is None:
        cube_size = np.array([1024, 1024, 512])  # this is in mag1

    offset, size, mask_arr, mask_mag = cset_cube_of_interest_parms(
        kd,
        global_params.config.cube_of_interest_bb,
        global_params.config.cube_of_interest_mask_fname,
        global_params.config.cube_of_interest_mask_mag)

    # todo was this necessary? doesn't work like this with new chunking model
    #if np.all(cube_size > size):
    #    cube_size = size

    cd.initialize(kd, size, cube_size, f'{tmp_dir}/cd_tmp_skel/',
                  box_coords=offset, fit_box_size=True, mask_arr=mask_arr,
                  mask_mag=mask_mag)
    multi_params = [(cube_size, offs, ds) for offs in chunkify_successive(
        list(cd.coord_dict.keys()), max(1, len(cd.coord_dict) // max_n_jobs))]

    out_dir = qu.batchjob_script(multi_params, "kimimaroskelgen", log=log, remove_jobfolder=False,
                                 n_cores=ncores_skelgen)

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

    multi_params = chunkify_weighted(ssd.ssv_ids, max_n_jobs * 2, ssd.load_numpy_data('size'))

    multi_params = [(pathdict_filepath, ssv_ids) for ssv_ids in multi_params]
    # create SSV skeletons, requires SV skeletons!
    log.info('Merging cube-wise skeletons of {} SSVs.'.format(len(ssd.ssv_ids)))
    # high memory load
    qu.batchjob_script(multi_params, "kimimaromerge", log=log, remove_jobfolder=True, n_cores=1)

    if map_myelin:
        map_myelin_global()

    shutil.rmtree(tmp_dir)
    shutil.rmtree(os.path.abspath(out_dir + '/../'))

    log.info('Finished skeleton generation.')


def _collect_paths(p: str) -> dict:
    partial_res = load_pkl2obj(p)
    res = {cellid: p for cellid in partial_res}
    return res
