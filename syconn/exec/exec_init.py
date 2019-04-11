# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max-Planck-Institute of Neurobiology, Munich, Germany
# Authors: Philipp Schubert, Joergen Kornfeld

from knossos_utils import knossosdataset
import time
import os
import shutil
import numpy as np
knossosdataset._set_noprint(True)
from knossos_utils import chunky
from syconn import global_params
from syconn.extraction import object_extraction_wrapper as oew
from syconn.proc import sd_proc
from syconn.reps.segmentation import SegmentationDataset
from syconn.handler.config import initialize_logging
from syconn.mp import batchjob_utils as qu
from syconn.handler.basics import chunkify, kd_factory


# TODO: make it work with new SyConn
def run_create_sds(chunk_size=None, n_folders_fs=10000, max_n_jobs=None,
                   generate_sv_meshes=False, load_cellorganelles_from_kd_overlaycubes=False,
                   transf_func_kd_overlay=None, cube_of_interest_bb=None):
    """

    Parameters
    ----------
    chunk_size :
    max_n_jobs : int
    n_folders_fs :
    generate_sv_meshes :
    load_cellorganelles_from_kd_overlaycubes : bool
        Load cell orgenelle prob/seg data from overlaycubes instead of raw cubes.
    transf_func_kd_overlay : Dict[callable]
        Method which is to applied to cube data if `load_from_kd_overlaycubes`
        is True. Must be a dictionary with keys `global_params.existing_cell_organelles`.
    cube_of_interest_bb : Tuple[np.ndarray]
        Defines the bounding box of the cube to process. By default this is
        set to (np.zoers(3); kd.boundary).


    Returns
    -------

    """
    if chunk_size is None:
        chunk_size = [512, 512, 512]
    if max_n_jobs is None:
        max_n_jobs = global_params.NCORE_TOTAL * 3
    log = initialize_logging('create_sds', global_params.config.working_dir +
                             '/logs/', overwrite=True)
    # Sets initial values of object
    kd = kd_factory(global_params.config.kd_seg_path)
    if cube_of_interest_bb is None:
        cube_of_interest_bb = [np.zeros(3, dtype=np.int), kd.boundary]
    size = cube_of_interest_bb[1] - cube_of_interest_bb[0] + 1
    offset = cube_of_interest_bb[0]
    # TODO: get rid of explicit voxel extraction, all info necessary should be extracted
    #  at the beginning, e.g. size, bounding box etc and then refactor to only use those cached attributes!
    # resulting ChunkDataset, required for SV extraction --
    # Object extraction - 2h, the same has to be done for all cell organelles
    cd_dir = global_params.config.working_dir + "chunkdatasets/sv/"
    # Class that contains a dict of chunks (with coordinates) after initializing it
    cd = chunky.ChunkDataset()
    cd.initialize(kd, kd.boundary, chunk_size, cd_dir,
                  box_coords=[0, 0, 0], fit_box_size=True)
    log.info('Generating SegmentationDatasets for cell and cell '
             'organelle supervoxels.')
    # oew.from_ids_to_objects(cd, "sv", overlaydataset_path=global_params.config.kd_seg_path,
    #                         n_chunk_jobs=max_n_jobs, hdf5names=["sv"], n_max_co_processes=None,
    #                         n_folders_fs=n_folders_fs, use_combined_extraction=True, size=size,
    #                         offset=offset, log=log)

    # Object Processing -- Perform after mapping to also cache mapping ratios
    sd = SegmentationDataset("sv", working_dir=global_params.config.working_dir)
    # sd_proc.dataset_analysis(sd, recompute=True, compute_meshprops=False)

    log.info("Extracted {} cell SVs. Preparing rendering locations "
             "(and meshes if not provided).".format(len(sd.ids)))
    start = time.time()
    # chunk them
    multi_params = chunkify(sd.so_dir_paths, max_n_jobs)
    # all other kwargs like obj_type='sv' and version are the current SV SegmentationDataset by default
    so_kwargs = dict(working_dir=global_params.config.working_dir, obj_type='sv')
    multi_params = [[par, so_kwargs] for par in multi_params]
    # if generate_sv_meshes:
    #     _ = qu.QSUB_script(multi_params, "mesh_caching",
    #                        n_max_co_processes=global_params.NCORE_TOTAL)
    # _ = qu.QSUB_script(multi_params, "sample_location_caching",
    #                    n_max_co_processes=global_params.NCORE_TOTAL)
    # # recompute=False: only collect new sample_location property
    # sd_proc.dataset_analysis(sd, compute_meshprops=True, recompute=False)
    log.info('Finished preparation of cell SVs after {:.0f}s.'.format(time.time() - start))
    # create SegmentationDataset for each cell organelle
    if transf_func_kd_overlay is None:
        transf_func_kd_overlay = {k: None for k in global_params.existing_cell_organelles}
    for co in global_params.existing_cell_organelles:
        start = time.time()
        cd_dir = global_params.config.working_dir + "chunkdatasets/{}/".format(co)
        cd.initialize(kd, kd.boundary, chunk_size, cd_dir,
                      box_coords=[0, 0, 0], fit_box_size=True)
        log.info('Started object extraction of cellular organelles "{}" from '
                 '{} chunks.'.format(co, len(cd.chunk_dict)))
        prob_kd_path_dict = {co: getattr(global_params.config, 'kd_{}_path'.format(co))}
        # This creates a SegmentationDataset of type 'co'
        prob_thresh = global_params.config.entries["Probathresholds"][co]  # get probability threshold

        # `from_probabilities_to_objects` will export a KD at `path`, remove if already existing
        path = "{}/knossosdatasets/{}_seg/".format(global_params.config.working_dir, co)
        if os.path.isdir(path):
            shutil.rmtree(path)
        target_kd = knossosdataset.KnossosDataset()
        target_kd.initialize_without_conf(path, kd.boundary, kd.scale, kd.experiment_name, mags=[1, ])
        target_kd = knossosdataset.KnossosDataset()
        target_kd.initialize_from_knossos_path(path)
        oew.from_probabilities_to_objects(cd, co, # membrane_kd_path=global_params.config.kd_barrier_path,  # TODO: currently does not exist
                                          prob_kd_path_dict=prob_kd_path_dict, thresholds=[prob_thresh],
                                          workfolder=global_params.config.working_dir,
                                          hdf5names=[co], n_max_co_processes=None, target_kd=target_kd,
                                          n_folders_fs=n_folders_fs, debug=False, size=size, offset=offset,
                                          load_from_kd_overlaycubes=load_cellorganelles_from_kd_overlaycubes,
                                          transf_func_kd_overlay=transf_func_kd_overlay[co], log=log)
        sd_co = SegmentationDataset(obj_type=co, working_dir=global_params.config.working_dir)

        # TODO: check if this is faster then the alternative below
        sd_proc.dataset_analysis(sd_co, recompute=True, compute_meshprops=False)
        multi_params = chunkify(sd_co.so_dir_paths, max_n_jobs)
        so_kwargs = dict(working_dir=global_params.config.working_dir, obj_type=co)
        multi_params = [[par, so_kwargs] for par in multi_params]
        _ = qu.QSUB_script(multi_params, "mesh_caching",
                           n_max_co_processes=global_params.NCORE_TOTAL)
        sd_proc.dataset_analysis(sd_co, recompute=False, compute_meshprops=True)
        # # Old alternative, requires much more reads/writes then above solution
        # sd_proc.dataset_analysis(sd_co, recompute=True, compute_meshprops=True)

        # About 0.2 h per object class
        log.info('Started mapping of {} cellular organelles of type "{}" to '
                 'cell SVs.'.format(len(sd_co.ids), co))

        sd_proc.map_objects_to_sv(sd, co, global_params.config.kd_seg_path,
                                  n_jobs=max_n_jobs)
        log.info('Finished preparation of {} "{}"-SVs after {:.0f}s.'
                 ''.format(len(sd_co.ids), co, time.time() - start))



