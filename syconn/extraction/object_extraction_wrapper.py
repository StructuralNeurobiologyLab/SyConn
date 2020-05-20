# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Philipp Schubert, Joergen Kornfeld

from .. import global_params
from ..extraction import log_extraction
from ..handler import basics
from . import object_extraction_steps as oes

import time
import os
import shutil
import numpy as np
from typing import Optional, Dict, List, Tuple, Union, Callable
from logging import Logger
from knossos_utils import chunky, knossosdataset


def calculate_chunk_numbers_for_box(cset, offset, size):
    """
    Calculates the chunk ids that are (partly) contained it the defined volume

    Parameters
    ----------
    cset : ChunkDataset
    offset : np.array
        offset of the volume to the origin
    size: np.array
        size of the volume

    Returns
    -------
    chunk_list: List
        chunk ids
    dictionary: dict
        with reverse mapping

    """

    for dim in range(3):
        offset_overlap = offset[dim] % cset.chunk_size[dim]
        offset[dim] -= offset_overlap
        size[dim] += offset_overlap
        size[dim] += (cset.chunk_size[dim] - size[dim]) % cset.chunk_size[dim]

    chunk_list = []
    translator = {}
    for x in range(offset[0], offset[0]+size[0], cset.chunk_size[0]):
        for y in range(offset[1], offset[1]+size[1], cset.chunk_size[1]):
            for z in range(offset[2], offset[2]+size[2], cset.chunk_size[2]):
                chunk_list.append(cset.coord_dict[tuple([x, y, z])])
                translator[chunk_list[-1]] = len(chunk_list)-1
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
        cube_of_interest_bb = [np.zeros(3, dtype=np.int), kd.boundary]
    size = cube_of_interest_bb[1] - cube_of_interest_bb[0] + 1
    offset = cube_of_interest_bb[0]
    cd_dir = "{}/chunkdatasets/{}/".format(global_params.config.working_dir, "_".join(subcell_names))
    if os.path.isdir(cd_dir):
        if not overwrite:
            msg = f'Could not start generation of sub-cellular objects ' \
                  f'"{subcell_names}" ChunkDataset because it already exists ' \
                  f'and overwrite was not set to True.'
            log_extraction.error(msg)
            raise FileExistsError(msg)
        log.debug('Found existing ChunkDataset at {}. Removing it now.'.format(cd_dir))
        shutil.rmtree(cd_dir)
    cd = chunky.ChunkDataset()
    cd.initialize(kd, kd.boundary, chunk_size, cd_dir,
                  box_coords=[0, 0, 0], fit_box_size=True,
                  list_of_coords=[])
    log.info('Started object extraction of cellular organelles "{}" from '
             '{} chunks.'.format(", ".join(subcell_names), len(cd.chunk_dict)))
    prob_kd_path_dict = {co: getattr(global_params.config, 'kd_{}_path'.format(co)) for co in subcell_names}
    prob_threshs = []  # get probability threshold
    # `from_probabilities_to_objects` will export a KD at `path`, remove if already existing
    for co in subcell_names:
        prob_threshs.append(global_params.config['cell_objects']["probathresholds"][co])
        path = global_params.config.kd_organelle_seg_paths[co]
        if os.path.isdir(path):
            if not overwrite:
                msg = f'Could not start generation of sub-cellular object ' \
                      f'"{co}" KnossosDataset because it already exists and overwrite ' \
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
                             "_".join(subcell_names),  # membrane_kd_path=global_params.config.kd_barrier_path,  # TODO: currently does not exist
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
          _gauss_threshold_connected_components_thread`
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
                max_nb_dict[hdf5_name][nb_chunk - 1] + \
                nb_cc_dict[hdf5_name][nb_chunk - 1]
        max_labels[hdf5_name] = int(max_nb_dict[hdf5_name][-1] + \
                                    nb_cc_dict[hdf5_name][-1])
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


def from_probabilities_to_objects(cset, filename, hdf5names,
                                  overlap="auto", sigmas=None, thresholds=None,
                                  debug=False, swapdata=0, target_kd=None,
                                  offset=None, size=None, prob_kd_path_dict=None,
                                  membrane_filename=None, membrane_kd_path=None,
                                  hdf5_name_membrane=None, n_folders_fs=1000,
                                  suffix="", transform_func=None, func_kwargs=None,
                                  nb_cpus=None, workfolder=None, n_erosion=0,
                                  overlap_thresh=0, stitch_overlap=None,
                                  load_from_kd_overlaycubes=False,
                                  transf_func_kd_overlay=None, log=None):
    """
    # TODO: Merge this method with mapping (e.g. iterate over chunks of cell SV segm. and over all
            objects to extract bounding boxes and overlap (i.e. mapping) at the same time
    Main function for the object extraction step; combines all needed steps

    Parameters
    ----------
    cset : chunkdataset instance
    filename : str
        Filename of the prediction in the ChunkDataset.
    hdf5names: List[str]
        List of names/ labels to be extracted and processed from the prediction
        file
    object_names : list of str
        list of names used for 'object_type' when creating SegmentationDataset.
        Must have same length as 'hdf5_names'.
    overlap: str or np.array
        Defines the overlap with neighbouring chunks that is left for later
        processing steps; if 'auto' the overlap is calculated from the sigma and
        the stitch_overlap (here: [1., 1., 1.])
    sigmas: List[List] or None
        Defines the sigmas of the gaussian filters applied to the probability
        maps. Has to be the same length as hdf5names. If None no gaussian filter
        is applied
    thresholds: list of float
        Threshold for cutting the probability map. Has to be the same length as
        hdf5names. If None zeros are used instead (not recommended!)
    target_kd:
    debug: boolean
        If true multiprocessed steps only operate on one core using 'map' which
        allows for better error messages
    swapdata: boolean
        If true an x-z swap is applied to the data prior to processing
    label_density: np.array
        Defines the density of the data. If the data was downsampled prior to
        saving; it has to be interpolated first before processing due to
        alignment issues with the coordinate system. Two-times downsampled
        data would have a label_density of [2, 2, 2]
    offset : np.array
        offset of the volume to the origin
    size: np.array
        size of the volume
    membrane_filename: str
        One way to allow access to a membrane segmentation when processing
        vesicle clouds. Filename of the prediction in the chunkdataset. The
        threshold is currently set at 0.4.
    membrane_kd_path: str
        One way to allow access to a membrane segmentation when processing
        vesicle clouds. Path to the knossosdataset containing a membrane
        segmentation. The threshold is currently set at 0.4.
    hdf5_name_membrane: str
        When using the membrane_filename this key has to be given to access the
        data in the saved chunk
    suffix: str
        Suffix for the intermediate results
    transform_func: callable
        Segmentation method which is applied
    func_kwargs : dict
        key word arguments for transform_func
    nb_cpus : int
        Number of cpus used if QSUB is disabled
    workfolder : str
        destination where SegmentationDataset will be stored
    n_erosion : int
        Number of erosions applied to the segmentation of unique_components0 to avoid
        segmentation artefacts caused by start location dependency in chunk data array.
    overlap_thresh : float
        Overlap fraction of object in different chunks to be considered stitched.
        If zero this behavior is disabled.
    stitch_overlap : np.array
        volume evaluated during stitching procedure
    load_from_kd_overlaycubes : bool
        Load prob/seg data from overlaycubes instead of raw cubes.
    transf_func_kd_overlay : callable
        Method which is to applied to cube data if `load_from_kd_overlaycubes`
        is True.
    log : logging.logger


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

    # --------------------------------------------------------------------------
    #
    time_start = time.time()
    cc_info_list, overlap_info = oes.object_segmentation(
        cset, filename, hdf5names, overlap=overlap, sigmas=sigmas,
        thresholds=thresholds, chunk_list=chunk_list, debug=debug,
        swapdata=swapdata, prob_kd_path_dict=prob_kd_path_dict,
        membrane_filename=membrane_filename, membrane_kd_path=membrane_kd_path,
        hdf5_name_membrane=hdf5_name_membrane, fast_load=True,
        suffix=suffix, transform_func=transform_func, transform_func_kwargs=func_kwargs,
        nb_cpus=nb_cpus, load_from_kd_overlaycubes=load_from_kd_overlaycubes,
        transf_func_kd_overlay=transf_func_kd_overlay)
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

    #
    # # ------------------------------------------------------------------------
    #
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
                max_nb_dict[hdf5_name][nb_chunk - 1] + \
                nb_cc_dict[hdf5_name][nb_chunk - 1]
        max_labels[hdf5_name] = int(max_nb_dict[hdf5_name][-1] + \
                                    nb_cc_dict[hdf5_name][-1])
    all_times.append(time.time() - time_start)
    step_names.append("extracting max labels")
    basics.write_obj2pkl(cset.path_head_folder.rstrip("/") + "/max_labels.pkl",
                         [max_labels])
    #
    # # ------------------------------------------------------------------------
    #
    time_start = time.time()
    oes.make_unique_labels(cset, filename, hdf5names, chunk_list, max_nb_dict,
                           chunk_translator, debug, suffix=suffix)
    all_times.append(time.time() - time_start)
    step_names.append("unique labels")
    #
    # # ------------------------------------------------------------------------
    #
    chunky.save_dataset(cset)  # save dataset to be able to load it during make_stitch_list (this
    # allows to load the ChunkDataset inside the worker instead of pickling it for each which
    # slows down the submission process.
    time_start = time.time()
    stitch_list = oes.make_stitch_list(cset, filename, hdf5names, chunk_list,
                                       stitch_overlap, overlap, debug,
                                       suffix=suffix,
                                       overlap_thresh=overlap_thresh)
    all_times.append(time.time() - time_start)
    step_names.append("stitch list")
    basics.write_obj2pkl(cset.path_head_folder.rstrip("/") + "/stitch_list.pkl",
                         [stitch_list])
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
                         debug, suffix=suffix)
    all_times.append(time.time() - time_start)
    step_names.append("apply merge list")

    if target_kd is not None:
        assert len(hdf5names) == 1, "This method does not yet support the export " \
                                    "of multiple KnossosDatasets."
        time_start = time.time()
        chunky.save_dataset(cset)
        oes.export_cset_to_kd_batchjob({hdf5names[0]: target_kd.conf_path},
            cset, '{}_stitched_components'.format(filename), hdf5names,
            offset=offset, size=size, stride=[4 * 128, 4 * 128, 4 * 128], as_raw=False,
            orig_dtype=np.uint64, unified_labels=False)
        all_times.append(time.time() - time_start)
        step_names.append("export KD")
    # --------------------------------------------------------------------------
        time_start = time.time()
        oes.extract_voxels_combined(cset, filename, hdf5names, n_folders_fs=n_folders_fs,
                                    chunk_list=chunk_list, suffix=suffix, workfolder=workfolder,
                                    overlaydataset_path=target_kd.conf_path)
        all_times.append(time.time() - time_start)
        step_names.append("extract and combine voxels")
    else:
        time_start = time.time()
        oes.extract_voxels(cset, filename, hdf5names, chunk_list=chunk_list, suffix=suffix,
                           workfolder=global_params.config.working_dir, n_folders_fs=n_folders_fs)
        all_times.append(time.time() - time_start)
        step_names.append("extract voxels")

        # --------------------------------------------------------------------------

        time_start = time.time()
        oes.combine_voxels(global_params.config.working_dir, hdf5names,
                           n_folders_fs=n_folders_fs, n_chunk_jobs=5000)

        all_times.append(time.time() - time_start)
        step_names.append("combine voxels")

    # --------------------------------------------------------------------------
    log.debug("Time overview [from_probabilities_to_objects]:")
    for ii in range(len(all_times)):
        log.debug("%s: %.3fs" % (step_names[ii], all_times[ii]))
    log.debug("--------------------------")
    log.debug("Total Time: %.1f min" % (np.sum(all_times) / 60))
    log.debug("--------------------------")


def from_probabilities_to_objects_parameter_sweeping(
        cset, filename, hdf5names, nb_thresholds, overlap="auto", sigmas=None,
        chunk_list=None, swapdata=0, label_density=np.ones(3), offset=None,
        size=None, membrane_filename=None, membrane_kd_path=None,
        hdf5_name_membrane=None):
    """
    Sweeps over different thresholds. Each objectextraction resutls are saved in
    a seperate folder, all intermediate steps are saved with a different suffix
    Parameters
    ----------
    cset : chunkdataset instance
    filename : str
        Filename of the prediction in the chunkdataset
    hdf5names: list of str
        List of names/ labels to be extracted and processed from the prediction
        file
    nb_thresholds: integer
        number of thresholds and therefore runs of objectextractions to do;
        the actual thresholds are equally spaced
    overlap: str or np.array
        Defines the overlap with neighbouring chunks that is left for later
        processing steps; if 'auto' the overlap is calculated from the sigma and
        the stitch_overlap (here: [1., 1., 1.])
    sigmas: list of lists or None
        Defines the sigmas of the gaussian filters applied to the probability
        maps. Has to be the same length as hdf5names. If None no gaussian filter
        is applied
    chunk_list: list of int
        Selective list of chunks for which this function should work on. If None
        all chunks are used.
    swapdata: boolean
        If true an x-z swap is applied to the data prior to processing
    label_density: np.array
        Defines the density of the data. If the data was downsampled prior to
        saving; it has to be interpolated first before processing due to
        alignment issues with the coordinate system. Two-times downsampled
        data would have a label_density of [2, 2, 2]
    offset : np.array
        offset of the volume to the origin
    size: np.array
        size of the volume
    membrane_filename: str
        One way to allow access to a membrane segmentation when processing
        vesicle clouds. Filename of the prediction in the chunkdataset. The
        threshold is currently set at 0.4.
    membrane_kd_path: str
        One way to allow access to a membrane segmentation when processing
        vesicle clouds. Path to the knossosdataset containing a membrane
        segmentation. The threshold is currently set at 0.4.
    hdf5_name_membrane: str
        When using the membrane_filename this key has to be given to access the
        data in the saved chunk
    suffix: str
        Suffix for the intermediate results
    """
    # TODO: currently not used and needs to be refactored
    thresholds = np.array(
        255. / (nb_thresholds + 1) * np.array(range(1, nb_thresholds + 1)),
        dtype=np.uint8)

    all_times = []
    for nb, t in enumerate(thresholds):
        log_extraction.info("\n\n ======= t = %.2f =======" % t)
        time_start = time.time()
        from_probabilities_to_objects(cset, filename, hdf5names,
                                      overlap=overlap, sigmas=sigmas,
                                      thresholds=[t] * len(hdf5names),
                                      chunk_list=chunk_list,
                                      swapdata=swapdata,
                                      label_density=label_density,
                                      offset=offset,
                                      size=size,
                                      membrane_filename=membrane_filename,
                                      membrane_kd_path=membrane_kd_path,
                                      hdf5_name_membrane=hdf5_name_membrane,
                                      suffix=str(nb),
                                      debug=False)
        all_times.append(time.time() - time_start)

    log_extraction.debug("\n\nTime overview:")
    for ii in range(len(all_times)):
        log_extraction.debug("t = %.2f: %.1f min" % (thresholds[ii], all_times[ii] / 60))
    log_extraction.debug("--------------------------")
    log_extraction.debug("Total Time: %.1f min" % (np.sum(all_times) / 60))
    log_extraction.debug("--------------------------\n")


def from_ids_to_objects(cset, filename, hdf5names=None, n_folders_fs=10000, dataset_names=None,
                        overlaydataset_path=None, chunk_list=None, offset=None, log=None,
                        size=None, suffix="", workfolder=None,
                        n_chunk_jobs=5000, use_combined_extraction=True):
    """
    # TODO: add SegmentationDataset initialization (-> `dataset_analysis` etc.)
    Main function for the object extraction step; combines all needed steps
    Parameters
    ----------
    cset : chunkdataset instance
    filename : str
        Filename of the prediction in the chunkdataset
    hdf5names: list of str
        List of names/ labels to be extracted and processed from the prediction
        file
    chunk_list: List[int]
        Selective list of chunks for which this function should work on. If None
        all chunks are used.
    debug: boolean
        If true multiprocessed steps only operate on one core using 'map' which
        allows for better error messages
    offset : np.array
        offset of the volume to the origin
    size: np.array
        size of the volume
    suffix: str
        Suffix for the intermediate results
    workfolder : str
        Directory in which to store results. By default this is set to
        `global_params.config.working_dir`.
    n_chunk_jobs: int
    use_combined_extraction : bool


    """
    if log is None:
        log = log_extraction
    if workfolder is None:
        workfolder = global_params.config.working_dir
    assert overlaydataset_path is not None or hdf5names is not None

    all_times = []
    step_names = []
    if size is not None and offset is not None:
        chunk_list, _ = calculate_chunk_numbers_for_box(cset, offset, size)
    else:
        if chunk_list is None:
            chunk_list = [ii for ii in range(len(cset.chunk_dict))]

    if not use_combined_extraction or overlaydataset_path is None:
        # # --------------------------------------------------------------------------
        #
        time_start = time.time()
        oes.extract_voxels(cset, filename, hdf5names, dataset_names=dataset_names,
                           overlaydataset_path=overlaydataset_path,
                           chunk_list=chunk_list, suffix=suffix, workfolder=workfolder,
                           n_folders_fs=n_folders_fs, n_chunk_jobs=n_chunk_jobs)
        all_times.append(time.time() - time_start)
        step_names.append("voxel extraction")
        #
        # # --------------------------------------------------------------------------
        #
        time_start = time.time()
        oes.combine_voxels(workfolder,
                           hdf5names, dataset_names=dataset_names,
                           n_folders_fs=n_folders_fs, n_chunk_jobs=n_chunk_jobs)
        all_times.append(time.time() - time_start)
        step_names.append("combine voxels")
    else:
        time_start = time.time()
        oes.extract_voxels_combined(cset, filename, hdf5names, overlaydataset_path=overlaydataset_path,
                                    dataset_names=dataset_names, chunk_list=chunk_list, suffix=suffix,
                                    workfolder=workfolder, n_folders_fs=n_folders_fs,
                                    n_chunk_jobs=n_chunk_jobs)
        all_times.append(time.time() - time_start)
        step_names.append("extract voxels combined")

    # --------------------------------------------------------------------------

    log.debug("Time overview [from_ids_to_objects]:")
    for ii in range(len(all_times)):
        log.debug("%s: %.3fs" % (step_names[ii], all_times[ii]))
    log.debug("--------------------------")
    log.debug("Total Time: %.1f min" % (np.sum(all_times) / 60))
    log.debug("--------------------------")
