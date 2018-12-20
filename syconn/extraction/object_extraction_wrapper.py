# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Philipp Schubert, Joergen Kornfeld
import numpy as np
import time
import os

from ..extraction import log_extraction
from ..handler import basics
from . import object_extraction_steps as oes


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
    chunk_list: list
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
    print("Chunk List contains %d elements." % len(chunk_list))
    return chunk_list, translator


def from_probabilities_to_objects(cset, filename, hdf5names, object_names=None,
                                  overlap="auto", sigmas=None,
                                  thresholds=None,
                                  chunk_list=None,
                                  debug=False,
                                  swapdata=0,
                                  offset=None,
                                  size=None,
                                  prob_kd_path_dict=None,
                                  membrane_filename=None,
                                  membrane_kd_path=None,
                                  hdf5_name_membrane=None,
                                  n_folders_fs=1000,
                                  suffix="",
                                  qsub_pe=None,
                                  qsub_queue=None,
                                  n_max_co_processes=None,
                                  transform_func=None,
                                  func_kwargs=None,
                                  nb_cpus=1,
                                  workfolder=None,
                                  n_erosion=0,
                                  overlap_thresh=0,
                                  stitch_overlap=None):
    """
    Main function for the object extraction step; combines all needed steps
    # TODO: change object_names to dataset_names as in other methods

    Parameters
    ----------
    cset : chunkdataset instance
    filename : str
        Filename of the prediction in the chunkdataset
    hdf5names: list of str
        List of names/ labels to be extracted and processed from the prediction
        file
    object_names : list of str
        list of names used for 'object_type' when creating SegmentationDataset.
        Must have same length as 'hdf5_names'.
    overlap: str or np.array
        Defines the overlap with neighbouring chunks that is left for later
        processing steps; if 'auto' the overlap is calculated from the sigma and
        the stitch_overlap (here: [1., 1., 1.])
    sigmas: list of lists or None
        Defines the sigmas of the gaussian filters applied to the probability
        maps. Has to be the same length as hdf5names. If None no gaussian filter
        is applied
    thresholds: list of float
        Threshold for cutting the probability map. Has to be the same length as
        hdf5names. If None zeros are used instead (not recommended!)
    chunk_list: list of int
        Selective list of chunks for which this function should work on. If None
        all chunks are used.
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
    qsub_pe: str or None
        qsub parallel environment
    qsub_queue: str or None
        qsub queue
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

    """
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
        if chunk_list is None:
            chunk_list = [ii for ii in range(len(cset.chunk_dict))]
            for ii in range(len(cset.chunk_dict)):
                chunk_translator[ii] = ii
        else:
            for ii in range(len(chunk_list)):
                chunk_translator[chunk_list[ii]] = ii

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
        swapdata=swapdata,
        prob_kd_path_dict=prob_kd_path_dict,
        membrane_filename=membrane_filename,
        membrane_kd_path=membrane_kd_path,
        hdf5_name_membrane=hdf5_name_membrane,
        fast_load=True, suffix=suffix,
        qsub_pe=qsub_pe, transform_func=transform_func, transform_func_kwargs=func_kwargs,
        qsub_queue=qsub_queue,
        n_max_co_processes=n_max_co_processes, nb_cpus=nb_cpus)
    if stitch_overlap is None:
        stitch_overlap = overlap_info[1]
    else:
        overlap_info[1] = stitch_overlap
    if not np.all(stitch_overlap < overlap_info[0]):
        msg = "Stitch overlap ({}) has to be smaller than chunk overlap ({})." \
              "".format(overlap_info[1], overlap_info[0])
        log_extraction.error(msg)
        raise ValueError(msg)
    overlap = overlap_info[0]
    all_times.append(time.time() - time_start)
    step_names.append("conneceted components")
    log_extraction.info(
        "Time needed for connected components: %.3fs" % all_times[-1])
    basics.write_obj2pkl(cset.path_head_folder.rstrip("/") + "/connected_components.pkl",
                         [cc_info_list, overlap_info])

    #
    # # --------------------------------------------------------------------------
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
    log_extraction.info("Time needed for extracting max labels: %.6fs" % all_times[-1])
    log_extraction.info("Max labels: {}".format(max_labels))
    basics.write_obj2pkl(cset.path_head_folder.rstrip("/") + "/max_labels.pkl",
                         [max_labels])
    #
    # # --------------------------------------------------------------------------
    #
    time_start = time.time()
    oes.make_unique_labels(cset, filename, hdf5names, chunk_list, max_nb_dict,
                           chunk_translator, debug, suffix=suffix,
                           qsub_pe=qsub_pe, qsub_queue=qsub_queue,
                           n_max_co_processes=n_max_co_processes, nb_cpus=nb_cpus)
    all_times.append(time.time() - time_start)
    step_names.append("unique labels")
    log_extraction.info("Time needed for unique labels: %.3fs" % all_times[-1])
    #
    # # --------------------------------------------------------------------------
    #
    time_start = time.time()
    stitch_list = oes.make_stitch_list(cset, filename, hdf5names, chunk_list,
                                       stitch_overlap, overlap, debug,
                                       suffix=suffix, qsub_pe=qsub_pe,
                                       qsub_queue=qsub_queue, n_erosion=n_erosion,
                                       n_max_co_processes=n_max_co_processes,
                                       overlap_thresh=overlap_thresh)
    all_times.append(time.time() - time_start)
    step_names.append("stitch list")
    log_extraction.info(
        "Time needed for stitch list: {:.3f}s.\nLength of stitch-lists for"
        " hdf5-names {}: {}".format(all_times[-1], hdf5names, [
            len(stitch_list[key]) for key in hdf5names]))
    basics.write_obj2pkl(cset.path_head_folder.rstrip("/") + "/stitch_list.pkl",
                         [stitch_list])
    #
    # # --------------------------------------------------------------------------
    #
    time_start = time.time()
    merge_dict, merge_list_dict = oes.make_merge_list(hdf5names, stitch_list,
                                                      max_labels)
    all_times.append(time.time() - time_start)
    step_names.append("merge list")
    log_extraction.info("Time needed for merge list: %.3fs" % all_times[-1])
    basics.write_obj2pkl(cset.path_head_folder.rstrip("/") + "/merge_list.pkl",
                         [merge_dict, merge_list_dict])
    # if all_times[-1] < 0.01:
    #     raise Exception("That was too fast!")

    # -------------------------------------------------------------------------

    time_start = time.time()
    oes.apply_merge_list(cset, chunk_list, filename, hdf5names, merge_list_dict,
                         debug, suffix=suffix, qsub_pe=qsub_pe, nb_cpus=nb_cpus,
                         qsub_queue=qsub_queue, n_max_co_processes=n_max_co_processes)
    all_times.append(time.time() - time_start)
    step_names.append("apply merge list")
    log_extraction.info("Time needed for applying merge list: %.3fs" % all_times[-1])

    # --------------------------------------------------------------------------

    time_start = time.time()
    oes.extract_voxels_combined(cset, filename, hdf5names, n_folders_fs=n_folders_fs,
                       chunk_list=chunk_list, suffix=suffix, workfolder=workfolder,
                       use_work_dir=True, qsub_pe=qsub_pe,
                       qsub_queue=qsub_queue, object_names=object_names,
                       n_max_co_processes=n_max_co_processes, nb_cpus=nb_cpus)
    all_times.append(time.time() - time_start)
    step_names.append("voxel extraction")
    log_extraction.info("Time needed for extracting voxels: %.3fs" % all_times[-1])
    # TODO: Remove map-reduce procedure or make it optional with kwarg
    # # --------------------------------------------------------------------------
    #
    # time_start = time.time()
    # oes.combine_voxels(os.path.dirname(cset.path_head_folder.rstrip("/")),
    #                    hdf5names, n_folders_fs=n_folders_fs, qsub_pe=qsub_pe,
    #                    qsub_queue=qsub_queue,
    #                    n_max_co_processes=n_max_co_processes, nb_cpus=nb_cpus)
    # all_times.append(time.time() - time_start)
    # step_names.append("combine voxels")
    # print("\nTime needed for combining voxels: %.3fs" % all_times[-1])

    # --------------------------------------------------------------------------
    log_extraction.info("Time overview:")
    for ii in range(len(all_times)):
        log_extraction.info("%s: %.3fs" % (step_names[ii], all_times[ii]))
    log_extraction.info("--------------------------")
    log_extraction.info("Total Time: %.1f min" % (np.sum(all_times) / 60))
    log_extraction.info("--------------------------")


def from_probabilities_to_objects_parameter_sweeping(cset,
                                                     filename,
                                                     hdf5names,
                                                     nb_thresholds,
                                                     overlap="auto",
                                                     sigmas=None,
                                                     chunk_list=None,
                                                     swapdata=0,
                                                     label_density=np.ones(3),
                                                     offset=None,
                                                     size=None,
                                                     membrane_filename=None,
                                                     membrane_kd_path=None,
                                                     hdf5_name_membrane=None,
                                                     qsub_pe=None,
                                                     qsub_queue=None):
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
    qsub_pe: str
        qsub parallel environment name
    qsub_queue: str or None
        qsub queue name
    """

    thresholds = np.array(
        255. / (nb_thresholds + 1) * np.array(range(1, nb_thresholds + 1)),
        dtype=np.uint8)

    all_times = []
    for nb, t in enumerate(thresholds):
        print("\n\n ======= t = %.2f =======" % t)
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
                                      qsub_pe=qsub_pe,
                                      qsub_queue=qsub_queue,
                                      debug=False)
        all_times.append(time.time() - time_start)

    print("\n\nTime overview:")
    for ii in range(len(all_times)):
        print("t = %.2f: %.1f min" % (thresholds[ii], all_times[ii] / 60))
    print("--------------------------")
    print("Total Time: %.1f min" % (np.sum(all_times) / 60))
    print("--------------------------\n")


def from_ids_to_objects(cset, filename, hdf5names=None, n_folders_fs=10000, dataset_names=None,
                        overlaydataset_path=None, chunk_list=None, offset=None,
                        size=None, suffix="", qsub_pe=None, qsub_queue=None, qsub_slots=None,
                        n_max_co_processes=None, n_chunk_jobs=5000, transform_func=None,
                        transform_func_kwargs=None):
    """
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
    qsub_pe: str or None
        qsub parallel environment
    qsub_queue: str or None
        qsub queue
    n_max_co_processes: int or None
        Total number of parallel processes that should be running on the cluster.
    n_chunk_jobs: int
    transform_func : callable
    transform_func_kwargs : dict


    """
    assert overlaydataset_path is not None or hdf5names is not None

    all_times = []
    step_names = []
    if size is not None and offset is not None:
        chunk_list, chunk_translator = \
            calculate_chunk_numbers_for_box(cset, offset, size)
    else:
        chunk_translator = {}
        if chunk_list is None:
            chunk_list = [ii for ii in range(len(cset.chunk_dict))]
            for ii in range(len(cset.chunk_dict)):
                chunk_translator[ii] = ii
        else:
            for ii in range(len(chunk_list)):
                chunk_translator[chunk_list[ii]] = ii
    # TODO: Remove or make optional
    # # --------------------------------------------------------------------------
    #
    time_start = time.time()
    oes.extract_voxels(cset, filename, hdf5names, dataset_names=dataset_names,
                       overlaydataset_path=overlaydataset_path,
                       chunk_list=chunk_list, suffix=suffix, qsub_pe=qsub_pe,
                       qsub_queue=qsub_queue,
                       n_folders_fs=n_folders_fs, n_chunk_jobs=n_chunk_jobs,
                       n_max_co_processes=n_max_co_processes, transform_func=transform_func,
                       transform_func_kwargs=transform_func_kwargs)
    all_times.append(time.time() - time_start)
    step_names.append("voxel extraction")
    print("\nTime needed for extracting voxels: %.3fs" % all_times[-1])
    #
    # # --------------------------------------------------------------------------
    #
    time_start = time.time()
    oes.combine_voxels(os.path.dirname(cset.path_head_folder.rstrip("/")),
                       hdf5names, dataset_names=dataset_names,
                       qsub_pe=qsub_pe, qsub_queue=qsub_queue,
                       n_folders_fs=n_folders_fs,
                       n_max_co_processes=n_max_co_processes)
    all_times.append(time.time() - time_start)
    step_names.append("combine voxels")
    print("\nTime needed for combining voxels: %.3fs" % all_times[-1])
    #
    # # --------------------------------------------------------------------------

    # time_start = time.time()
    # oes.extract_voxels_combined(cset, filename, hdf5names,
    #                    overlaydataset_path=overlaydataset_path,
    #                    chunk_list=chunk_list, suffix=suffix, qsub_pe=qsub_pe,
    #                    qsub_queue=qsub_queue, qsub_slots=qsub_slots,
    #                    n_folders_fs=n_folders_fs, n_chunk_jobs=n_chunk_jobs,
    #                    n_max_co_processes=n_max_co_processes)
    # all_times.append(time.time() - time_start)
    # step_names.append("extract voxels combined")
    # print("\nTime needed for extracting voxels combined: %.3fs" % all_times[-1])

    # --------------------------------------------------------------------------

    print("\nTime overview:")
    for ii in range(len(all_times)):
        print("%s: %.3fs" % (step_names[ii], all_times[ii]))
    print("--------------------------")
    print("Total Time: %.1f min" % (np.sum(all_times) / 60))
    print("--------------------------\n\n")
