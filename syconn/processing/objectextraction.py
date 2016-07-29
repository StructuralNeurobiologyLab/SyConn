# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max-Planck-Institute for Medical Research, Heidelberg, Germany
# Authors: Sven Dorkenwald, Philipp Schubert, Jörgen Kornfeld

import numpy as np
import os
import time
import cPickle as pickle
import networkx as nx
import glob

from syconn.processing import objectextraction_helper as oeh
from syconn.multi_proc import multi_proc_main as mpm
from syconn.utils import datahandler, segmentationdataset


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
    print "Chunk List contains %d elements." % len(chunk_list)
    return chunk_list, translator


def gauss_threshold_connected_components(cset, filename, hdf5names,
                                         overlap="auto", sigmas=None,
                                         thresholds=None,
                                         chunk_list=None,
                                         debug=False,
                                         swapdata=False,
                                         label_density=np.ones(3),
                                         membrane_filename=None,
                                         membrane_kd_path=None,
                                         hdf5_name_membrane=None,
                                         fast_load=False,
                                         suffix="",
                                         use_qsub=False):
    """
    Extracts connected component from probability maps
    1. Gaussian filter (defined by sigma)
    2. Thresholding (defined by threshold)
    3. Connected components analysis

    In case of vesicle clouds (hdf5_name in ["p4", "vc"]) the membrane
    segmentation is used to cut connected vesicle clouds across cells
    apart (only if membrane segmentation is provided).

    Parameters
    ----------
    cset : chunkdataset instance
    filename : str
        Filename of the prediction in the chunkdataset
    hdf5names: list of str
        List of names/ labels to be extracted and processed from the prediction
        file
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
    fast_load: boolean
        If true the data of chunk is blindly loaded without checking for enough
        offset to compute the overlap area. Faster, because no neighbouring
        chunk has to be accessed since the default case loads th overlap area
        from them.
    suffix: str
        Suffix for the intermediate results
    use_qsub: boolean
        Whether or not to use qsub

    Returns
    -------
    results_as_list: list
        list containing information about the number of connected components
        in each chunk
    overlap: np.array
    stitch overlap: np.array
    """

    label_density = np.array(label_density)
    if thresholds is None:
        thresholds = np.zeros(len(hdf5names))
    if sigmas is None:
        sigmas = np.zeros(len(hdf5names))
    if not len(sigmas) == len(thresholds) == len(hdf5names):
        raise Exception("Number of thresholds, sigmas and HDF5 names does not "
                        "match!")

    stitch_overlap = np.array([1, 1, 1])
    if overlap == "auto":
        # Truncation of gaussian kernel is 4 per standard deviation
        # (per default). One overlap for matching of connected components
        if sigmas is None:
            max_sigma = np.zeros(3)
        else:
            max_sigma = np.array([np.max(sigmas)] * 3)

        overlap = np.ceil(max_sigma * 4) + stitch_overlap

    print "overlap:", overlap

    print "thresholds:", thresholds

    multi_params = []
    for nb_chunk in chunk_list:
        multi_params.append(
            [cset.chunk_dict[nb_chunk], cset.path_head_folder, filename,
             hdf5names, overlap,
             sigmas, thresholds, swapdata,
             label_density, membrane_filename, membrane_kd_path,
             hdf5_name_membrane, fast_load, suffix])

    if not use_qsub:
        results = mpm.start_multiprocess(oeh.gauss_threshold_connected_components_thread,
                                         multi_params, debug=debug)

        results_as_list = []
        for result in results:
            for entry in result:
                results_as_list.append(entry)

    elif mpm.__QSUB__:
        path_to_out = mpm.QSUB_script(multi_params,
                                      "gauss_threshold_connected_components",
                                      queue="full")

        out_files = glob.glob(path_to_out + "/*")
        results_as_list = []
        for out_file in out_files:
            with open(out_file) as f:
                for entry in pickle.load(f):
                    results_as_list.append(entry)
    else:
        raise Exception("QSUB not available")

    return results_as_list, [overlap, stitch_overlap]


def make_unique_labels(cset, filename, hdf5names, chunk_list, max_nb_dict,
                       chunk_translator, debug, suffix="",
                       use_qsub=False):
    """
    Makes labels unique across chunks

    Parameters
    ----------
    cset : chunkdataset instance
    filename : str
        Filename of the prediction in the chunkdataset
    hdf5names: list of str
        List of names/ labels to be extracted and processed from the prediction
        file
    chunk_list: list of int
        Selective list of chunks for which this function should work on. If None
        all chunks are used.
    max_nb_dict: dictionary
        Maps each chunk id to a integer describing which needs to be added to
        all its entries
    chunk_translator: boolean
        Remapping from chunk ids to position in chunk_list
    debug: boolean
        If true multiprocessed steps only operate on one core using 'map' which
        allows for better error messages
    suffix: str
        Suffix for the intermediate results
    use_qsub: boolean
        Whether or not to use qsub

    """

    multi_params = []
    for nb_chunk in chunk_list:
        this_max_nb_dict = {}
        for hdf5_name in hdf5names:
            this_max_nb_dict[hdf5_name] = max_nb_dict[hdf5_name][
                chunk_translator[nb_chunk]]

        multi_params.append([cset.chunk_dict[nb_chunk], filename, hdf5names,
                             this_max_nb_dict, suffix])

    if not use_qsub:
        results = mpm.start_multiprocess(oeh.make_unique_labels_thread,
                                         multi_params, debug=debug)

    elif mpm.__QSUB__:
        path_to_out = mpm.QSUB_script(multi_params,
                                      "make_unqiue_labels",
                                      queue="full")
    else:
        raise Exception("QSUB not available")


def make_stitch_list(cset, filename, hdf5names, chunk_list, stitch_overlap,
                     overlap, debug, suffix="", use_qsub=False):
    """
    Creates a stitch list for the overlap region between chunks

    Parameters
    ----------
    cset : chunkdataset instance
    filename : str
        Filename of the prediction in the chunkdataset
    hdf5names: list of str
        List of names/ labels to be extracted and processed from the prediction
        file
    chunk_list: list of int
        Selective list of chunks for which this function should work on. If None
        all chunks are used.
    overlap: np.array
        Defines the overlap with neighbouring chunks that is left for later
        processing steps
    stitch_overlap: np.array
        Defines the overlap with neighbouring chunks that is left for stitching
    debug: boolean
        If true multiprocessed steps only operate on one core using 'map' which
        allows for better error messages
    suffix: str
        Suffix for the intermediate results
    use_qsub: boolean
        Whether or not to use qsub

    Returns:
    --------
    stitch_list: list
        list of overlapping component ids
    """

    multi_params = []
    for nb_chunk in chunk_list:
        multi_params.append([cset, nb_chunk, filename, hdf5names,
                             stitch_overlap, overlap, suffix, chunk_list])

    if not use_qsub:
        results = mpm.start_multiprocess(oeh.make_stitch_list_thread,
                                         multi_params, debug=debug)

        stitch_list = {}
        for hdf5_name in hdf5names:
            stitch_list[hdf5_name] = []

        for result in results:
            for hdf5_name in hdf5names:
                elems = result[hdf5_name]
                for elem in elems:
                    stitch_list[hdf5_name].append(elem)

    elif mpm.__QSUB__:
        path_to_out = mpm.QSUB_script(multi_params,
                                      "make_stitch_list_thread",
                                      queue="full")

        out_files = glob.glob(path_to_out + "/*")

        stitch_list = {}
        for hdf5_name in hdf5names:
            stitch_list[hdf5_name] = []

        for out_file in out_files:
            with open(out_file) as f:
                result = pickle.load(f)
                for hdf5_name in hdf5names:
                    elems = result[hdf5_name]
                    for elem in elems:
                        stitch_list[hdf5_name].append(elem)
    else:
        raise Exception("QSUB not available")

    return stitch_list


def make_merge_list(hdf5names, stitch_list, max_labels):
    """
    Creates a merge list from a stitch list by mapping all connected ids to
    one id

    Parameters
    ----------
    hdf5names: list of str
        List of names/ labels to be extracted and processed from the prediction
        file
    stitch_list: dictionary
        Contains pairs of overlapping component ids for each hdf5name
    max_labels dictionary
        Contains the number of different component ids for each hdf5name

    Returns
    -------
    merge_dict: dictionary
        mergelist for each hdf5name
    merge_list_dict: dictionary
        mergedict for each hdf5name
    """

    merge_dict = {}
    merge_list_dict = {}
    for hdf5_name in hdf5names:
        this_stitch_list = stitch_list[hdf5_name]
        max_label = max_labels[hdf5_name]
        graph = nx.from_edgelist(this_stitch_list)
        cc = nx.connected_components(graph)
        merge_dict[hdf5_name] = {}
        merge_list_dict[hdf5_name] = np.arange(max_label + 1)
        for this_cc in cc:
            this_cc = list(this_cc)
            for id in this_cc:
                merge_dict[hdf5_name][id] = this_cc[0]
                merge_list_dict[hdf5_name][id] = this_cc[0]

    return merge_dict, merge_list_dict


def apply_merge_list(cset, chunk_list, filename, hdf5names, merge_list_dict,
                     debug, suffix="", use_qsub=False):
    """
    Applies merge list to all chunks

    Parameters
    ----------
    cset : chunkdataset instance
    chunk_list: list of int
        Selective list of chunks for which this function should work on. If None
        all chunks are used.
    filename : str
        Filename of the prediction in the chunkdataset
    hdf5names: list of str
        List of names/ labels to be extracted and processed from the prediction
        file
    merge_list_dict: dictionary
        mergedict for each hdf5name
    debug: boolean
        If true multiprocessed steps only operate on one core using 'map' which
        allows for better error messages
    suffix: str
        Suffix for the intermediate results
    use_qsub: boolean
        Whether or not to use qsub
    """

    multi_params = []
    merge_list_dict_path = cset.path_head_folder + "merge_list_dict.pkl"

    f = open(merge_list_dict_path, "w")
    pickle.dump(merge_list_dict, f)
    f.close()

    for nb_chunk in chunk_list:
        multi_params.append([cset.chunk_dict[nb_chunk], filename, hdf5names,
                             merge_list_dict_path, suffix])

    if not use_qsub:
        results = mpm.start_multiprocess(oeh.apply_merge_list_thread,
                                         multi_params, debug=debug)

    elif mpm.__QSUB__:
        raise NotImplementedError()
        # path_to_out = mpm.QSUB_script(multi_params,
        #                               "apply_merge_list",
        #                               cset, chunk_list, queue="red3somaq")

    else:
        raise Exception("QSUB not available")


def extract_voxels(cset, filename, hdf5names, debug=False, chunk_list=None,
                   suffix="", use_qsub=False):
    """
    Extracts voxels for each component id

    Parameters
    ----------
    cset : chunkdataset instance
    filename : str
        Filename of the prediction in the chunkdataset
    hdf5names: list of str
        List of names/ labels to be extracted and processed from the prediction
        file
    chunk_list: list of int
        Selective list of chunks for which this function should work on. If None
        all chunks are used.
    debug: boolean
        If true multiprocessed steps only operate on one core using 'map' which
        allows for better error messages
    suffix: str
        Suffix for the intermediate results
    use_qsub: boolean
        Whether or not to use qsub

    """

    if chunk_list is None:
        chunk_list = [ii for ii in range(len(cset.chunk_dict))]

    multi_params = []
    for nb_chunk in chunk_list:
        multi_params.append([cset.chunk_dict[nb_chunk], cset.path_head_folder,
                             filename, hdf5names, suffix])

    if not use_qsub:
        results = mpm.start_multiprocess(oeh.extract_voxels_thread,
                                         multi_params, debug=debug)

    elif mpm.__QSUB__:
        path_to_out = mpm.QSUB_script(multi_params,
                                      "extract_voxels",
                                      queue="full")

    else:
        raise Exception("QSUB not available")


def concatenate_mappings(cset, filename, hdf5names, debug=False, suffix=""):
    """
    Combines all map dicts

    Parameters
    ----------
    cset : chunkdataset instance
    filename : str
        Filename of the prediction in the chunkdataset
    hdf5names: list of str
        List of names/ labels to be extracted and processed from the prediction
        file
    debug: boolean
        If true multiprocessed steps only operate on one core using 'map' which
        allows for better error messages
    suffix: str
        Suffix for the intermediate results

    """

    multi_params = []
    for hdf5_name in hdf5names:
        rel_path = segmentationdataset.get_rel_path(hdf5_name, filename, suffix)
        map_dict_paths = glob.glob(cset.path_head_folder + rel_path +
                                   "/map_dicts/*")
        multi_params.append([cset.path_head_folder, rel_path, map_dict_paths])

    mpm.start_multiprocess(oeh.concatenate_mappings_thread,
                           multi_params, debug=debug)


def create_objects_from_voxels(cset, filename, hdf5names, granularity=15,
                               debug=False, suffix="", use_qsub=False):
    """
    Creates object instances from extracted voxels

    Parameters
    ----------
    cset : chunkdataset instance
    filename : str
        Filename of the prediction in the chunkdataset
    hdf5names: list of str
        List of names/ labels to be extracted and processed from the prediction
        file
    granularity: int
        Defines granularity for partitioning data for multiprocessing
    debug: boolean
        If true multiprocessed steps only operate on one core using 'map' which
        allows for better error messages
    suffix: str
        Suffix for the intermediate results
    use_qsub: boolean
        Whether or not to use qsub

    """

    multi_params = []
    for nb_hdf5_name in range(len(hdf5names)):
        counter = 0
        hdf5_name = hdf5names[nb_hdf5_name]
        path_dataset = cset.path_head_folder + \
                       segmentationdataset.get_rel_path(hdf5_name, filename, suffix)
        if not os.path.exists(path_dataset + "/object_dicts/"):
            os.makedirs(path_dataset + "/object_dicts/")

        map_dict_paths = glob.glob(path_dataset + "/map_dicts/map_*")

        for step in range(
                int(np.ceil(len(map_dict_paths) / float(granularity)))):
            this_map_dict_paths = map_dict_paths[step * granularity:
                                                    (step + 1) * granularity]
            save_path = path_dataset + "/object_dicts/dict_%d.pkl" % counter
            multi_params.append([cset.path_head_folder, this_map_dict_paths,
                                 filename, hdf5_name, save_path, suffix,
                                 counter])
            counter += 1

    if not use_qsub:
        mpm.start_multiprocess(oeh.create_objects_from_voxels_thread,
                               multi_params, debug=debug)

    elif mpm.__QSUB__:
        path_to_out = mpm.QSUB_script(multi_params,
                                      "create_objects_from_voxels",
                                      queue="full")

    else:
        raise Exception("QSUB not available")


def create_datasets_from_objects(cset, filename, hdf5names,
                                 debug=False, suffix="", use_qsub=False):
    """
    Create dataset instance from objects

    Parameters
    ----------
    cset : chunkdataset instance
    filename : str
        Filename of the prediction in the chunkdataset
    hdf5names: list of str
        List of names/ labels to be extracted and processed from the prediction
        file
    debug: boolean
        If true multiprocessed steps only operate on one core using 'map' which
        allows for better error messages
    suffix: str
        Suffix for the intermediate results
    use_qsub: boolean
        Whether or not to use qsub

    """
    multi_params = []
    for hdf5_name in hdf5names:
        multi_params.append(
            [cset.path_head_folder, hdf5_name, filename, suffix])

    if not use_qsub:
        mpm.start_multiprocess(oeh.create_datasets_from_objects_thread,
                               multi_params, debug=debug)

    elif mpm.__QSUB__:
        path_to_out = mpm.QSUB_script(multi_params,
                                      "create_datasets_from_objects",
                                      queue="full")

    else:
        raise Exception("QSUB not available")


def from_probabilities_to_objects(cset, filename, hdf5names,
                                  overlap="auto", sigmas=None,
                                  thresholds=None,
                                  chunk_list=None,
                                  debug=False,
                                  swapdata=0,
                                  label_density=np.ones(3),
                                  offset=None,
                                  size=None,
                                  membrane_filename=None,
                                  membrane_kd_path=None,
                                  hdf5_name_membrane=None,
                                  suffix="",
                                  use_qsub=False):
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
    use_qsub: boolean
        Whether or not to use qsub

    """
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

    if thresholds is not None and thresholds[0] <= 1.:
        thresholds = np.array(thresholds)
        thresholds *= 255

    if sigmas is not None and swapdata == 1:
        for nb_sigma in range(len(sigmas)):
            if len(sigmas[nb_sigma]) == 3:
                sigmas[nb_sigma] = datahandler.switch_array_entries(sigmas[nb_sigma],
                                                           [0, 2])

    time_start = time.time()
    cc_info_list, overlap_info = gauss_threshold_connected_components(
        cset, filename,
        hdf5names, overlap, sigmas, thresholds,
        chunk_list, debug,
        swapdata, label_density=label_density,
        membrane_filename=membrane_filename,
        membrane_kd_path=membrane_kd_path,
        hdf5_name_membrane=hdf5_name_membrane,
        fast_load=True, suffix=suffix,
        use_qsub=use_qsub)

    stitch_overlap = overlap_info[1]
    overlap = overlap_info[0]
    all_times.append(time.time() - time_start)
    step_names.append("conneceted components")
    print "\nTime needed for connected components: %.3fs" % all_times[-1]

    # --------------------------------------------------------------------------

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
    print "\nTime needed for extracting max labels: %.6fs" % all_times[-1]
    print "Max labels: ", max_labels

    # --------------------------------------------------------------------------

    time_start = time.time()
    make_unique_labels(cset, filename, hdf5names, chunk_list, max_nb_dict,
                       chunk_translator, debug, suffix=suffix,
                       use_qsub=use_qsub)
    all_times.append(time.time() - time_start)
    step_names.append("unique labels")
    print "\nTime needed for unique labels: %.3fs" % all_times[-1]

    # --------------------------------------------------------------------------

    time_start = time.time()
    stitch_list = make_stitch_list(cset, filename, hdf5names, chunk_list,
                                   stitch_overlap, overlap, debug,
                                   suffix=suffix, use_qsub=use_qsub)
    all_times.append(time.time() - time_start)
    step_names.append("stitch list")
    print "\nTime needed for stitch list: %.3fs" % all_times[-1]

    # --------------------------------------------------------------------------

    time_start = time.time()
    merge_dict, merge_list_dict = make_merge_list(hdf5names, stitch_list,
                                                  max_labels)
    all_times.append(time.time() - time_start)
    step_names.append("merge list")
    print "\nTime needed for merge list: %.3fs" % all_times[-1]
    # if all_times[-1] < 0.01:
    #     raise Exception("That was too fast!")

    # -------------------------------------------------------------------------

    time_start = time.time()
    apply_merge_list(cset, chunk_list, filename, hdf5names, merge_list_dict,
                     debug, suffix=suffix, use_qsub=use_qsub)
    all_times.append(time.time() - time_start)
    step_names.append("apply merge list")
    print "\nTime needed for applying merge list: %.3fs" % all_times[-1]

    # --------------------------------------------------------------------------

    for hdf5_name in hdf5names:
        path = cset.path_head_folder + "/" + \
               segmentationdataset.get_rel_path(hdf5_name, filename, suffix)
        if not os.path.exists(path + "/map_dicts/"):
            os.makedirs(path + "/map_dicts/")
        if not os.path.exists(path + "/voxels/"):
            os.makedirs(path + "/voxels/")
        if not os.path.exists(path + "/hull_voxels/"):
            os.makedirs(path + "/hull_voxels/")

    # --------------------------------------------------------------------------

    time_start = time.time()
    extract_voxels(cset, filename, hdf5names, debug=debug,
                   chunk_list=chunk_list, suffix=suffix, use_qsub=use_qsub)
    all_times.append(time.time() - time_start)
    step_names.append("voxel extraction")
    print "\nTime needed for extracting voxels: %.3fs" % all_times[-1]

    # --------------------------------------------------------------------------

    time_start = time.time()
    concatenate_mappings(cset, filename, hdf5names, debug=debug, suffix=suffix)
    all_times.append(time.time() - time_start)
    step_names.append("concatenate mappings")
    print "\nTime needed for concatenating mappings: %.3fs" % all_times[-1]

    # --------------------------------------------------------------------------

    time_start = time.time()
    create_objects_from_voxels(cset, filename, hdf5names, granularity=15,
                               debug=debug, suffix=suffix, use_qsub=use_qsub)
    all_times.append(time.time() - time_start)
    step_names.append("create objects from voxels")
    print "\nTime needed for creating objects: %.3fs" % all_times[-1]

    # --------------------------------------------------------------------------

    time_start = time.time()
    create_datasets_from_objects(cset, filename, hdf5names,
                                 debug=debug, suffix=suffix, use_qsub=use_qsub)
    all_times.append(time.time() - time_start)
    step_names.append("create datasets from objects")
    print "\nTime needed for creating datasets: %.3fs" % all_times[-1]

    # --------------------------------------------------------------------------

    print "\nTime overview:"
    for ii in range(len(all_times)):
        print "%s: %.3fs" % (step_names[ii], all_times[ii])
    print "--------------------------"
    print "Total Time: %.1f min" % (np.sum(all_times) / 60)


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
                                                     use_qsub=False):
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
    use_qsub: boolean
        Whether or not to use qsub

    """

    thresholds = np.array(
        255. / (nb_thresholds + 1) * np.array(range(1, nb_thresholds + 1)),
        dtype=np.uint8)

    all_times = []
    for nb, t in enumerate(thresholds):
        print "\n\n ======= t = %.2f =======" % t
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
                                      use_qsub=use_qsub,
                                      debug=False)
        all_times.append(time.time() - time_start)

    print "\n\nTime overview:"
    for ii in range(len(all_times)):
        print "t = %.2f: %.1f min" % (thresholds[ii], all_times[ii] / 60)
    print "--------------------------"
    print "Total Time: %.1f min" % (np.sum(all_times) / 60)
