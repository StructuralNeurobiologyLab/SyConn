# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Philipp Schubert, Joergen Kornfeld

import os
import shutil
import time
from logging import Logger
from typing import Optional, Dict, List, Tuple, Union, Callable

import numpy as np
from knossos_utils import chunky, knossosdataset

from . import object_extraction_steps as oes
from .. import global_params
from ..extraction import log_extraction
from ..handler import basics


def calculate_chunk_numbers_for_box(cset, offset, size):
    """
    Calculates the chunk ids that are (partly) contained it the defined volume

    Args:
        cset : ChunkDataset
        offset(np.array):
            offset of the volume to the origin
        size(np.array):
            size of the volume

    Returns:
        chunk_list(list):
            chunk ids
        dictionary(dict):
            with reverse mapping

    """

    for dim in range(3):
        offset_overlap = offset[dim] % cset.chunk_size[dim]
        offset[dim] -= offset_overlap
        size[dim] += offset_overlap
        size[dim] += (cset.chunk_size[dim] - size[dim]) % cset.chunk_size[dim]

    chunk_list = []
    translator = {}
    for x in range(offset[0], offset[0] + size[0], cset.chunk_size[0]):
        for y in range(offset[1], offset[1] + size[1], cset.chunk_size[1]):
            for z in range(offset[2], offset[2] + size[2], cset.chunk_size[2]):
                chunk_list.append(cset.coord_dict[tuple([x, y, z])])
                translator[chunk_list[-1]] = len(chunk_list) - 1
    return chunk_list, translator


def generate_subcell_kd_from_proba(
        subcell_names: List[str], chunk_size: Optional[Union[list, tuple]] = None,
        transf_func_kd_overlay: Optional[Dict[str, Callable]] = None,
        load_cellorganelles_from_kd_overlaycubes: bool = False,
        cube_of_interest_bb: Optional[Tuple[np.ndarray]] = None,
        cube_shape: Optional[Tuple[int]] = None,
        log: Logger = None, overwrite=False, **kwargs):
    """
    Generates a connected components segmentation for the given the sub-cellular
    structures (e.g. ['mi', 'sj', 'vc]) as KnossosDatasets.
    The data format of the source data is KnossosDataset which path(s) is defined
    in ``global_params.config['paths']`` (e.g. key ``kd_mi_path`` for mitochondria).
    The resulting KDs will be stored at (for each ``co in subcell_names``)
    ``"{}/knossosdatasets/{}_seg/".format(global_params.config.working_dir, co)``.
    See :func:`~syconn.extraction.object_extraction_wrapper.from_probabilities_to_kd` for details of
    the conversion process from the initial probability map to the SV segmentation. Default:
    thresholding and connected components, thresholds are set via the `config.yml` file, check
    ``syconn.global_params.config['cell_objects']["probathresholds"]`` of an initialized
    :class:`~syconn.handler.config.DynConfig` object.

    Args:
        subcell_names:
        chunk_size:
        transf_func_kd_overlay:
        load_cellorganelles_from_kd_overlaycubes:
        cube_of_interest_bb:
        cube_shape:
        log:
        overwrite:
        **kwargs:

    Returns:

    """
    if chunk_size is None:
        chunk_size = [512, 512, 512]
    if log is None:
        log = log_extraction
    if cube_shape is None:
        cube_shape = (256, 256, 256)
    kd = basics.kd_factory(global_params.config.kd_seg_path)
    if cube_of_interest_bb is None:
        cube_of_interest_bb = [np.zeros(3, dtype=np.int32), kd.boundary]
    size = cube_of_interest_bb[1] - cube_of_interest_bb[0] + 1
    offset = cube_of_interest_bb[0]
    cd_dir = "{}/chunkdatasets/{}/".format(global_params.config.working_dir, "_".join(subcell_names))
    if os.path.isdir(cd_dir):
        if not overwrite:
            msg = f'Could not start generation of sub-cellular objects ' \
                  f'"{subcell_names}" ChunkDataset because it already exists at "{cd_dir}" ' \
                  f'and overwrite was not set to True.'
            log_extraction.error(msg)
            raise FileExistsError(msg)
        log.debug('Found existing ChunkDataset at {}. Removing it now.'.format(cd_dir))
        shutil.rmtree(cd_dir)
    cd = chunky.ChunkDataset()
    # TODO: possible to restrict ChunkDataset here already to report correct number of processed chunks? Check
    #  coordinate framework compatibility downstream in `from_probabilities_to_kd`
    cd.initialize(kd, kd.boundary, chunk_size, cd_dir,
                  box_coords=[0, 0, 0], fit_box_size=True,
                  list_of_coords=[])
    log.info('Started object extraction of cellular organelles "{}" from '
             '{} chunks.'.format(", ".join(subcell_names), len(cd.chunk_dict)))
    prob_kd_path_dict = {co: getattr(global_params.config, 'kd_{}_path'.format(co)) for co in subcell_names}
    prob_threshs = []  # get probability threshold
    for co in subcell_names:
        prob_threshs.append(global_params.config['cell_objects']["probathresholds"][co])
        path = global_params.config.kd_organelle_seg_paths[co]
        if os.path.isdir(path):
            if not overwrite:
                msg = f'Could not start generation of sub-cellular object ' \
                      f'"{co}" KnossosDataset because it already exists at "{path}" and overwrite ' \
                      f'was not set to True.'
                log_extraction.error(msg)
                raise FileExistsError(msg)
            log.debug('Found existing KD at {}. Removing it now.'.format(path))
            shutil.rmtree(path)
        target_kd = knossosdataset.KnossosDataset()
        scale = np.array(global_params.config['scaling'], dtype=np.float32)
        target_kd._cube_shape = cube_shape
        target_kd.scales = [scale, ]
        target_kd.initialize_without_conf(path, kd.boundary, scale, kd.experiment_name, mags=[1, ],
                                          create_pyk_conf=True, create_knossos_conf=False)
    if load_cellorganelles_from_kd_overlaycubes:  # no thresholds needed
        prob_threshs = None
    from_probabilities_to_kd(global_params.config.kd_organelle_seg_paths, cd,
                             "_".join(subcell_names),
                             # membrane_kd_path=global_params.config.kd_barrier_path,  # TODO: currently does not exist
                             prob_kd_path_dict=prob_kd_path_dict, thresholds=prob_threshs,
                             hdf5names=subcell_names, size=size, offset=offset,
                             load_from_kd_overlaycubes=load_cellorganelles_from_kd_overlaycubes,
                             transf_func_kd_overlay=transf_func_kd_overlay, log=log, **kwargs)
    shutil.rmtree(cd_dir, ignore_errors=True)


def from_probabilities_to_kd(
        target_kd_paths: Optional[Dict[str, str]],
        cset: 'chunky.ChunkDataset', filename: str,
        hdf5names: List[str], prob_kd_path_dict: Optional[Dict[str, str]] = None,
        load_from_kd_overlaycubes: bool = False,
        transf_func_kd_overlay: Optional[Dict[str, Callable]] = None,
        log: Optional[Logger] = None, overlap: str = "auto",
        sigmas: Optional[list] = None, thresholds: Optional[list] = None,
        debug: bool = False, swapdata: bool = False,
        offset: Optional[np.ndarray] = None, size: Optional[np.ndarray] = None,
        suffix: str = "", transform_func: Optional[Callable] = None,
        func_kwargs: Optional[dict] = None, n_cores: Optional[int] = None,
        overlap_thresh: Optional[int] = 0,
        stitch_overlap: Optional[int] = None, membrane_filename: str = None,
        membrane_kd_path: str = None, hdf5_name_membrane: str = None,
        n_chunk_jobs: int = None):
    """
    Method for the conversion of classified (hard labels, e.g. 0, 1, 2; see
    `load_from_kd_overlaycubes` and `transf_func_kd_overlay` parameters)
    or predicted (probability maps, e.g. 0 .. 1 or 0 .. 255, see `thresholds`
    parameter).
    Original data can be provided as ChunkDataset `cset` or via KnossosDataset(s)
    `prob_kd_path_dict`. The ChunkDataset will be used in any case for storing
    the intermediate extraction results (per-cube segmentation, stitched results,
    globally unique segmentation).

    Notes:
        * KnossosDatasets given by `target_kd_paths` need to be initialized
          prior to this function call.

    Args:
        target_kd_paths: Paths to (already initialized) output KnossosDatasets.
          See ``KnossosDataset.initialize_without_conf``.
        cset: ChunkDataset which is used for the object extraction process and
          which may additionally contain the source data. The latter can be
          provided as KnossosDataset(s) (see `prob_kd_path_dict`).
        filename: The base name used to store the extracted in `cset`.
        hdf5names: Keys used to store the intermediate extraction results.
        prob_kd_path_dict: Paths to source KnossosDatasets
        load_from_kd_overlaycubes: Load prob/seg data from overlaycubes instead
          of raw cubes.
        transf_func_kd_overlay: Method which is to applied to cube data if
          `load_from_kd_overlaycubes` is True.
        log: TODO: pass log to all methods called
        overlap: Defines the overlap with neighbouring chunks that is left for
          later processing steps; if 'auto' the overlap is calculated from the
          sigma and the stitch_overlap (here: [1., 1., 1.]).
        sigmas: Defines the sigmas of the Gaussian filters applied to the
          probability maps. Has to be the same length as hdf5names. If None,
          no Gaussian filter is applied.
        thresholds: Threshold for cutting the probability map. Has to be the
          same length as hdf5names. If None, zeros are used instead (not recommended!)
        debug: If True, multiprocessing steps only operate on one core using 'map'
          which allows for better error messages.
        swapdata: If true an x-z swap is applied to the data prior to processing.
        offset: Offset of the processed volume.
        size: Size of the processed volume of the dataset starting at `offset`.
        suffix: Suffix used for the intermediate processing steps.
        transform_func: [WIP] Segmentation method which is applied, currently
          only func:`~syconn.extraction.object_extraction_steps.
          _object_segmentation_thread`
          is supported for batch jobs.
        func_kwargs: keyword arguments for `transform_func`.
        n_chunk_jobs: Number of jobs.
        n_cores: Number of cores used for each job in
          :func:`syconn.extraction.object_extraction_steps.object_segmentation`
          if batch jobs is enabled.
        overlap_thresh: Overlap fraction of object in different chunks to be
          considered stitched. If zero this behavior is disabled.
        stitch_overlap: Volume evaluated during stitching procedure.
        membrane_filename: Experimental. One way to allow access to a membrane
          segmentation when processing vesicle clouds. Filename of the
          prediction in the chunkdataset. The threshold is currently set at 0.4.
        membrane_kd_path: Experimental. One way to allow access to a membrane
          segmentation when processing vesicle clouds. Path to the
          knossosdataset containing a membrane segmentation. The threshold
          is currently set at 0.4.
        hdf5_name_membrane: Experimental. When `membrane_filename` is set this
          key has to be given to access the data in the saved chunk.

    Returns:

    """
    if log is None:
        log = log_extraction
    all_times = []
    step_names = []

    if prob_kd_path_dict is not None:
        kd_keys = list(prob_kd_path_dict.keys())
        assert len(kd_keys) == len(hdf5names)
        for kd_key in kd_keys:
            assert kd_key in hdf5names

    if size is not None and offset is not None:
        chunk_list, chunk_translator = \
            calculate_chunk_numbers_for_box(cset, offset, size)
    else:
        chunk_translator = {}
        chunk_list = [ii for ii in range(len(cset.chunk_dict))]
        for ii in range(len(cset.chunk_dict)):
            chunk_translator[ii] = ii

    if thresholds is not None and thresholds[0] <= 1.:
        thresholds = np.array(thresholds)
        thresholds *= 255

    if sigmas is not None and swapdata == 1:
        for nb_sigma in range(len(sigmas)):
            if len(sigmas[nb_sigma]) == 3:
                sigmas[nb_sigma] = \
                    basics.switch_array_entries(sigmas[nb_sigma], [0, 2])

    # # --------------------------------------------------------------------------

    time_start = time.time()
    cc_info_list, overlap_info = oes.object_segmentation(
        cset, filename, hdf5names, overlap=overlap, sigmas=sigmas,
        thresholds=thresholds, chunk_list=chunk_list, debug=debug,
        swapdata=swapdata, prob_kd_path_dict=prob_kd_path_dict,
        membrane_filename=membrane_filename, membrane_kd_path=membrane_kd_path,
        hdf5_name_membrane=hdf5_name_membrane, fast_load=True,
        suffix=suffix, transform_func=transform_func,
        transform_func_kwargs=func_kwargs,
        nb_cpus=n_cores, load_from_kd_overlaycubes=load_from_kd_overlaycubes,
        transf_func_kd_overlay=transf_func_kd_overlay, n_chunk_jobs=n_chunk_jobs)
    if stitch_overlap is None:
        stitch_overlap = overlap_info[1]
    else:
        overlap_info[1] = stitch_overlap
    if not np.all(stitch_overlap <= overlap_info[0]):
        msg = "Stitch overlap ({}) has to be <= than chunk overlap ({})." \
              "".format(overlap_info[1], overlap_info[0])
        log.error(msg)
        raise ValueError(msg)
    overlap = overlap_info[0]
    all_times.append(time.time() - time_start)
    step_names.append("conneceted components")
    basics.write_obj2pkl(cset.path_head_folder.rstrip("/") +
                         "/connected_components.pkl",
                         [cc_info_list, overlap_info])

    # # # ------------------------------------------------------------------------

    time_start = time.time()
    nb_cc_dict = {}
    max_nb_dict = {}
    max_labels = {}
    for hdf5_name in hdf5names:
        nb_cc_dict[hdf5_name] = np.zeros(len(chunk_list), dtype=np.int32)
        max_nb_dict[hdf5_name] = np.zeros(len(chunk_list), dtype=np.int32)
    for cc_info in cc_info_list:
        nb_cc_dict[cc_info[1]][chunk_translator[cc_info[0]]] = cc_info[2]
    for hdf5_name in hdf5names:
        max_nb_dict[hdf5_name][0] = 0
        for nb_chunk in range(1, len(chunk_list)):
            max_nb_dict[hdf5_name][nb_chunk] = \
                max_nb_dict[hdf5_name][nb_chunk - 1] + nb_cc_dict[hdf5_name][nb_chunk - 1]
        max_labels[hdf5_name] = int(max_nb_dict[hdf5_name][-1] + nb_cc_dict[hdf5_name][-1])
    all_times.append(time.time() - time_start)
    step_names.append("max labels")
    basics.write_obj2pkl(cset.path_head_folder.rstrip("/") + "/max_labels.pkl",
                         max_labels)
    # # ------------------------------------------------------------------------

    time_start = time.time()
    oes.make_unique_labels(cset, filename, hdf5names, chunk_list, max_nb_dict,
                           chunk_translator, debug, suffix=suffix,
                           n_chunk_jobs=n_chunk_jobs, nb_cpus=n_cores)
    all_times.append(time.time() - time_start)
    step_names.append("unique labels")

    # # ------------------------------------------------------------------------

    chunky.save_dataset(cset)  # save dataset to be able to load it during make_stitch_list (this
    # allows to load the ChunkDataset inside the worker instead of pickling it for each, which
    # slows down the submission process.

    time_start = time.time()
    stitch_list = oes.make_stitch_list(cset, filename, hdf5names, chunk_list,
                                       stitch_overlap, overlap, debug,
                                       suffix=suffix,
                                       overlap_thresh=overlap_thresh,
                                       n_chunk_jobs=n_chunk_jobs, nb_cpus=n_cores)
    all_times.append(time.time() - time_start)
    step_names.append("stitch list")
    basics.write_obj2pkl(cset.path_head_folder.rstrip("/") + "/stitch_list.pkl",
                         stitch_list)
    #
    # # ------------------------------------------------------------------------
    #
    time_start = time.time()
    merge_dict, merge_list_dict = oes.make_merge_list(hdf5names, stitch_list,
                                                      max_labels)
    all_times.append(time.time() - time_start)
    step_names.append("merge list")
    basics.write_obj2pkl(cset.path_head_folder.rstrip("/") + "/merge_list.pkl",
                         [merge_dict, merge_list_dict])

    # --------------------------------------------------------------------------

    time_start = time.time()
    oes.apply_merge_list(cset, chunk_list, filename, hdf5names, merge_list_dict,
                         debug, suffix=suffix,
                         n_chunk_jobs=n_chunk_jobs, nb_cpus=n_cores)
    all_times.append(time.time() - time_start)
    step_names.append("apply merge list")

    time_start = time.time()
    chunky.save_dataset(cset)
    oes.export_cset_to_kd_batchjob(
        target_kd_paths, cset, '{}_stitched_components'.format(filename),
        hdf5names, offset=offset, size=size, stride=cset.chunk_size,
        as_raw=False, orig_dtype=np.uint64, unified_labels=False, log=log,
        n_max_job=n_chunk_jobs, n_cores=n_cores)
    all_times.append(time.time() - time_start)
    step_names.append("export KD")

    # --------------------------------------------------------------------------
    log.debug("Time overview [from_probabilities_to_kd]:")
    for ii in range(len(all_times)):
        log.debug("%s: %.3fs" % (step_names[ii], all_times[ii]))
    log.debug("--------------------------")
    log.debug("Total Time: %.1f min" % (np.sum(all_times) / 60))
    log.debug("--------------------------")
