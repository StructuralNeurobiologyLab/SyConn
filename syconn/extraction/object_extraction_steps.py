# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Philipp Schubert, Joergen Kornfeld
try:
    import cPickle as pkl
except ImportError:
    import pickle as pkl
import glob
import networkx as nx
import numpy as np
import os
import scipy.ndimage
import shutil
import itertools
from collections import defaultdict
from knossos_utils import knossosdataset, chunky
knossosdataset._set_noprint(True)

from ..handler import log_handler
from ..mp import batchjob_utils as qu, mp_utils as sm
from ..proc.general import cut_array_in_one_dim
from ..reps import segmentation, rep_helper as rh
from ..handler import basics
from ..backend.storage import VoxelStorageL, VoxelStorage
from ..proc.image import multi_mop


def gauss_threshold_connected_components(*args, **kwargs):
    # alias
    return object_segmentation(*args, **kwargs)


def object_segmentation(cset, filename, hdf5names, overlap="auto", sigmas=None,
                        thresholds=None, chunk_list=None, debug=False,
                        swapdata=False, prob_kd_path_dict=None,
                        membrane_filename=None, membrane_kd_path=None,
                        hdf5_name_membrane=None, fast_load=False, suffix="",
                        qsub_pe=None, qsub_queue=None, nb_cpus=None,
                        n_max_co_processes=None, transform_func=None,
                        transform_func_kwargs=None):
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
    chunk_list: list of
        Selective list of chunks for which this function should work on. If None
        all chunks are used.
    debug: boolean
        If true multiprocessed steps only operate on one core using 'map' which
        allows for better error messages
    swapdata: boolean
        If true an x-z swap is applied to the data prior to processing
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
    qsub_pe: str or None
        qsub parallel environment
    qsub_queue: str or None
        qsub queue
    transform_func: callable
        Segmentation method which is applied
    transform_func_kwargs : dict
        key word arguments for transform_func

    Returns
    -------
    results_as_list: list
        list containing information about the number of connected components
        in each chunk
    overlap: np.array
    stitch overlap: np.array
    """
    if transform_func is None:
        transform_func = _gauss_threshold_connected_components_thread

    if thresholds is None:
        thresholds = np.zeros(len(hdf5names))
    if sigmas is None:
        sigmas = np.zeros(len(hdf5names))
    if not len(sigmas) == len(thresholds) == len(hdf5names):
        raise Exception("Number of thresholds, sigmas and HDF5 names does not "
                        "match!")

    stitch_overlap = np.array([1, 1, 1])
    if overlap is "auto":
        # # TODO: Check if overlap kwarg can actually be set independent of cset.overlap
        # if np.any(np.array(overlap) > np.array(cset.overlap)):
        #     msg = "Untested behavior for overlap kwarg being bigger than cset.overlap." \
        #           " This might lead to stitching failures."
        #     log_handler.critical(msg)
        #     raise ValueError(msg)
        # Truncation of gaussian kernel is 4 per standard deviation
        # (per default). One overlap for matching of connected components
        if sigmas is None:
            max_sigma = np.zeros(3)
        else:
            max_sigma = np.array([np.max(sigmas)] * 3)
        overlap = np.ceil(max_sigma * 4) + stitch_overlap

    multi_params = []
    for nb_chunk in chunk_list:
        multi_params.append(
            [cset.chunk_dict[nb_chunk], cset.path_head_folder, filename,
             hdf5names, overlap,
             sigmas, thresholds, swapdata, prob_kd_path_dict,
             membrane_filename, membrane_kd_path,
             hdf5_name_membrane, fast_load, suffix, transform_func_kwargs])

    if (qsub_pe is None and qsub_queue is None) or not qu.batchjob_enabled():
        results = sm.start_multiprocess_imap(transform_func,
                                             multi_params, debug=debug)

        results_as_list = []
        for result in results:
            for entry in result:
                results_as_list.append(entry)

    elif qu.batchjob_enabled():
        assert transform_func == _gauss_threshold_connected_components_thread,\
            "QSUB currently only supported for gaussian threshold CC."
        path_to_out = qu.QSUB_script(multi_params,
                                     "gauss_threshold_connected_components",
                                     pe=qsub_pe, queue=qsub_queue,
                                     n_cores=nb_cpus, script_folder=None,
                                     n_max_co_processes=n_max_co_processes)
        out_files = glob.glob(path_to_out + "/*")
        results_as_list = []
        for out_file in out_files:
            with open(out_file, 'rb') as f:
                for entry in pkl.load(f):
                    results_as_list.append(entry)
    else:
        raise Exception("QSUB not available")

    return results_as_list, [overlap, stitch_overlap]


def _gauss_threshold_connected_components_thread(args):
    """
    Default worker of object_segmentation. Performs a gaussian blur with
     subsequent thresholding to extract connected components of a probability
     map. Result summaries are returned and connected components are stored as
     .h5 files.
     TODO: Add generic '_segmentation_thread' to enable a clean support of
     custom-made segmentation functions passed to 'object_segmentation' via
     'transform_func'-kwargs

    Parameters
    ----------
    args : list

    Returns
    -------
    list of lists
        Results of connected component analysis
    """
    chunk = args[0]
    path_head_folder = args[1]
    filename = args[2]
    hdf5names = args[3]
    overlap = args[4]
    sigmas = args[5]
    thresholds = args[6]
    swapdata = args[7]
    prob_kd_path_dict = args[8]
    membrane_filename = args[9]
    membrane_kd_path = args[10]
    hdf5_name_membrane = args[11]
    fast_load = args[12]
    suffix = args[13]

    box_offset = np.array(chunk.coordinates) - np.array(overlap)
    size = np.array(chunk.size) + 2*np.array(overlap)

    if swapdata:
        size = basics.switch_array_entries(size, [0, 2])

    if prob_kd_path_dict is not None:
        bin_data_dict = {}
        for kd_key in prob_kd_path_dict.keys():
            kd = knossosdataset.KnossosDataset()
            kd.initialize_from_knossos_path(prob_kd_path_dict[kd_key])
            bin_data_dict[kd_key] = kd.from_raw_cubes_to_matrix(size,
                                                                box_offset)
    else:
        if not fast_load:
            cset = chunky.load_dataset(path_head_folder)
            bin_data_dict = cset.from_chunky_to_matrix(size, box_offset,
                                                       filename, hdf5names)
        else:
            bin_data_dict = basics.load_from_h5py(chunk.folder + filename + ".h5",
                                                  hdf5names, as_dict=True)

    labels_data = []
    nb_cc_list = []
    for nb_hdf5_name in range(len(hdf5names)):
        hdf5_name = hdf5names[nb_hdf5_name]
        tmp_data = np.copy(bin_data_dict[hdf5_name])

        tmp_data_shape = tmp_data.shape
        offset = (np.array(tmp_data_shape) - np.array(chunk.size) -
                  2 * np.array(overlap)) / 2
        offset = offset.astype(np.int)
        if np.any(offset < 0):
            offset = np.array([0, 0, 0])
        tmp_data = tmp_data[offset[0]: tmp_data_shape[0]-offset[0],
                            offset[1]: tmp_data_shape[1]-offset[1],
                            offset[2]: tmp_data_shape[2]-offset[2]]

        if np.sum(sigmas[nb_hdf5_name]) != 0:
            tmp_data = scipy.ndimage.gaussian_filter(tmp_data, sigmas[nb_hdf5_name])

        if hdf5_name in ["p4", "vc"] and membrane_filename is not None and hdf5_name_membrane is not None:
            membrane_data = basics.load_from_h5py(chunk.folder + membrane_filename + ".h5",
                                                  hdf5_name_membrane)[0]
            membrane_data_shape = membrane_data.shape
            offset = (np.array(membrane_data_shape) - np.array(tmp_data.shape)) / 2
            membrane_data = membrane_data[offset[0]: membrane_data_shape[0]-offset[0],
                                          offset[1]: membrane_data_shape[1]-offset[1],
                                          offset[2]: membrane_data_shape[2]-offset[2]]
            tmp_data[membrane_data > 255*.4] = 0
        elif hdf5_name in ["p4", "vc"] and membrane_kd_path is not None:
            kd_bar = knossosdataset.KnossosDataset()
            kd_bar.initialize_from_knossos_path(membrane_kd_path)
            membrane_data = kd_bar.from_raw_cubes_to_matrix(size, box_offset)
            tmp_data[membrane_data > 255*.4] = 0

        if thresholds[nb_hdf5_name] != 0:
            tmp_data = np.array(tmp_data > thresholds[nb_hdf5_name],
                                dtype=np.uint8)

        this_labels_data, nb_cc = scipy.ndimage.label(tmp_data)
        nb_cc_list.append([chunk.number, hdf5_name, nb_cc])
        labels_data.append(this_labels_data)

    h5_fname = chunk.folder + filename + "_connected_components%s.h5" % suffix
    os.makedirs(os.path.split(h5_fname)[0], exist_ok=True)
    basics.save_to_h5py(labels_data, h5_fname, hdf5names)
    return nb_cc_list


def make_unique_labels(cset, filename, hdf5names, chunk_list, max_nb_dict,
                       chunk_translator, debug, suffix="", nb_cpus=None,
                       qsub_pe=None, qsub_queue=None, n_max_co_processes=100):
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
    chunk_translator: Dict
        Remapping from chunk ids to position in chunk_list
    debug: boolean
        If true multiprocessed steps only operate on one core using 'map' which
        allows for better error messages
    suffix: str
        Suffix for the intermediate results
    qsub_pe: str or None
        qsub parallel environment
    qsub_queue: str or None
        qsub queue

    """

    multi_params = []
    for nb_chunk in chunk_list:
        this_max_nb_dict = {}
        for hdf5_name in hdf5names:
            this_max_nb_dict[hdf5_name] = max_nb_dict[hdf5_name][
                chunk_translator[nb_chunk]]

        multi_params.append([cset.chunk_dict[nb_chunk], filename, hdf5names,
                             this_max_nb_dict, suffix])

    if (qsub_pe is None and qsub_queue is None) or not qu.batchjob_enabled():
        results = sm.start_multiprocess_imap(_make_unique_labels_thread,
                                         multi_params, debug=debug, nb_cpus=nb_cpus)

    elif qu.batchjob_enabled():
        path_to_out = qu.QSUB_script(multi_params,
                                     "make_unique_labels",
                                     pe=qsub_pe, queue=qsub_queue,
                                     script_folder=None,
                                     n_max_co_processes=n_max_co_processes)
    else:
        raise Exception("QSUB not available")


def _make_unique_labels_thread(args):
    chunk = args[0]
    filename = args[1]
    hdf5names = args[2]
    this_max_nb_dict = args[3]
    suffix = args[4]

    cc_data_list = basics.load_from_h5py(chunk.folder + filename +
                                         "_connected_components%s.h5"
                                         % suffix, hdf5names)

    for nb_hdf5_name in range(len(hdf5names)):
        hdf5_name = hdf5names[nb_hdf5_name]
        matrix = cc_data_list[nb_hdf5_name]
        matrix[matrix > 0] += this_max_nb_dict[hdf5_name]

    basics.save_to_h5py(cc_data_list,
                        chunk.folder + filename +
                        "_unique_components%s.h5"
                        % suffix, hdf5names)


def make_stitch_list(cset, filename, hdf5names, chunk_list, stitch_overlap,
                     overlap, debug, suffix="", qsub_pe=None, qsub_queue=None,
                     n_max_co_processes=100, nb_cpus=None, n_erosion=0,
                     overlap_thresh=0):
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
    qsub_pe: str or None
        qsub parallel environment
    qsub_queue: str or None
        qsub queue
    n_erosion : int
        Number of erosions applied to the segmentation of unique_components0 to avoid
        segmentation artefacts caused by start location dependency in chunk data array.
    overlap_thresh : float
                Overlap fraction of object in different chunks to be considered stitched.
                If zero this behavior is disabled.

    Returns
    -------
    stitch_list: Dict
        Dictionary of overlapping component ids
    """

    multi_params = []
    for nb_chunk in chunk_list:
        multi_params.append([cset, nb_chunk, filename, hdf5names, stitch_overlap, overlap,
                             suffix, chunk_list, n_erosion, overlap_thresh])

    if (qsub_pe is None and qsub_queue is None) or not qu.batchjob_enabled():
        results = sm.start_multiprocess_imap(_make_stitch_list_thread,
                                         multi_params, debug=debug, nb_cpus=nb_cpus)

        stitch_list = {}
        for hdf5_name in hdf5names:
            stitch_list[hdf5_name] = []
        for result in results:
            for hdf5_name in hdf5names:
                elems = result[hdf5_name]
                for elem in elems:
                    stitch_list[hdf5_name].append(elem)

    elif qu.batchjob_enabled():
        path_to_out = qu.QSUB_script(multi_params,
                                     "make_stitch_list",
                                     pe=qsub_pe, queue=qsub_queue,
                                     script_folder=None,
                                     n_max_co_processes=n_max_co_processes)

        out_files = glob.glob(path_to_out + "/*")

        stitch_list = {}
        for hdf5_name in hdf5names:
            stitch_list[hdf5_name] = []

        for out_file in out_files:
            with open(out_file, 'rb') as f:
                result = pkl.load(f)
                for hdf5_name in hdf5names:
                    elems = result[hdf5_name]
                    for elem in elems:
                        stitch_list[hdf5_name].append(elem)
    else:
        raise Exception("QSUB not available")

    return stitch_list


def _make_stitch_list_thread(args):
    cset = args[0]
    nb_chunk = args[1]
    filename = args[2]
    hdf5names = args[3]
    stitch_overlap = args[4]
    overlap = args[5]
    suffix = args[6]
    chunk_list = args[7]
    n_erosion = args[8]
    overlap_thresh = args[9]

    chunk = cset.chunk_dict[nb_chunk]
    cc_data_list = basics.load_from_h5py(chunk.folder + filename +
                                         "_unique_components%s.h5"
                                         % suffix, hdf5names)
    # erode segmentation once to avoid start location dependent segmentation artefacts
    struct = np.zeros((3, 3, 3)).astype(np.bool)
    mask = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]).astype(np.bool)
    struct[:, :, 1] = mask  # only perform erosion in xy plane
    for kk in range(len(cc_data_list)):
        cc_data_list[kk] = multi_mop(scipy.ndimage.binary_erosion, cc_data_list[kk], n_iters=n_erosion, background_only=True,
                                     use_find_objects=True, mop_kwargs={'structure': struct}, verbose=False)
    # TODO: opztimize get_neighbouring_chunks
    neighbours, pos = cset.get_neighbouring_chunks(chunk, chunklist=chunk_list,
                                                   con_mode=7)

    #  Compare only upper half of 6-neighborhood for every chunk
    neighbours = neighbours[np.any(pos > 0, axis=1)]
    pos = pos[np.any(pos > 0, axis=1)]

    map_dict = {}
    for nb_hdf5_name in range(len(hdf5names)):
        map_dict[hdf5names[nb_hdf5_name]] = set()
    # Compare only half of 6-neighborhood for every chunk which suffices to cover all overlap areas. Checking all
    # neighbors for every chunk would lead to twice and redundant computational load
    for ii in range(3):
        if neighbours[ii] != -1:
            compare_chunk = cset.chunk_dict[neighbours[ii]]
            cc_data_list_to_compare = \
                basics.load_from_h5py(compare_chunk.folder + filename +
                                      "_unique_components%s.h5"
                                      % suffix, hdf5names)
            # erode segmentaiton once to avoid start location dependent segmentation artefacts
            for kk in range(len(cc_data_list_to_compare)):
                cc_data_list_to_compare[kk] = multi_mop(scipy.ndimage.binary_erosion, cc_data_list_to_compare[kk],
                                                        n_iters=n_erosion, use_find_objects=True,
                                                        mop_kwargs={'structure': struct}, verbose=False)

            cc_area = {}
            cc_area_to_compare = {}

            id = np.argmax(pos[ii])  # get contact dimension (perpendicular to contact plane)
            for nb_hdf5_name in range(len(hdf5names)):
                this_cc_data = cc_data_list[nb_hdf5_name]
                this_cc_data_to_compare = \
                    cc_data_list_to_compare[nb_hdf5_name]

                cc_area[nb_hdf5_name] = \
                    cut_array_in_one_dim(
                        this_cc_data,
                        -overlap[id] - stitch_overlap[id],
                        -overlap[id] + stitch_overlap[id], id)

                cc_area_to_compare[nb_hdf5_name] = \
                    cut_array_in_one_dim(
                        this_cc_data_to_compare,
                        overlap[id] - stitch_overlap[id],
                        overlap[id] + stitch_overlap[id], id)
            for nb_hdf5_name in range(len(hdf5names)):
                hdf5_name = hdf5names[nb_hdf5_name]
                stitch_ixs = np.transpose(np.nonzero((cc_area[nb_hdf5_name] != 0) &
                                                     (cc_area_to_compare[nb_hdf5_name] != 0)))
                ignore_ids = set()  # if already inspected and overlap is insufficient
                for stitch_pos in stitch_ixs:
                    stitch_pos = tuple(stitch_pos)
                    this_id = cc_area[nb_hdf5_name][stitch_pos]
                    compare_id = cc_area_to_compare[nb_hdf5_name][stitch_pos]
                    pair = tuple(sorted([this_id, compare_id]))
                    if (pair not in map_dict[hdf5_name]) and (pair not in ignore_ids):
                        if overlap_thresh > 0:
                            obj_coord_intern = np.transpose(np.nonzero(cc_data_list[nb_hdf5_name] == this_id))
                            obj_coord_intern_compare = np.transpose(np.nonzero(cc_data_list_to_compare[nb_hdf5_name] == compare_id))
                            c1 = chunk.coordinates - chunk.overlap + obj_coord_intern + np.array([1, 1, 1])
                            c2 = compare_chunk.coordinates - compare_chunk.overlap + obj_coord_intern_compare + np.array([1, 1, 1])
                            from scipy import spatial
                            kdt = spatial.cKDTree(c1)
                            dists, ixs = kdt.query(c2)
                            match_vx = np.sum(dists == 0)
                            match_vx_rel = 2 * float(match_vx) / (len(c1) + len(c2))
                            if match_vx_rel > 0.1:
                                map_dict[hdf5_name].add(pair)
                            else:
                                ignore_ids.add(pair)
                            msg = "(stitch ID 1, stitch ID 2), stitch location 1, stitch location 2:" \
                                  " {}. Matching vx {} (abs:) {:.2f}%".format((pair, c1[0], c2[0]), match_vx, match_vx_rel)
                            log_handler.debug(msg)
                        else:
                            map_dict[hdf5_name].add(pair)
    for k, v in map_dict.items():
        map_dict[k] = list(v)
    return map_dict


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
    max_labels : dictionary
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
                     debug, suffix="", qsub_pe=None, qsub_queue=None,
                     n_max_co_processes=100, nb_cpus=None):
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
    qsub_pe: str or None
        qsub parallel environment
    qsub_queue: str or None
        qsub queue
    """

    multi_params = []
    merge_list_dict_path = cset.path_head_folder + "merge_list_dict.pkl"

    f = open(merge_list_dict_path, "wb")
    pkl.dump(merge_list_dict, f)
    f.close()

    for nb_chunk in chunk_list:
        multi_params.append([cset.chunk_dict[nb_chunk], filename, hdf5names,
                             merge_list_dict_path, suffix])

    if (qsub_pe is None and qsub_queue is None) or not qu.batchjob_enabled():
        results = sm.start_multiprocess_imap(_apply_merge_list_thread,
                                         multi_params, debug=debug, nb_cpus=nb_cpus)

    elif qu.batchjob_enabled():
        path_to_out = qu.QSUB_script(multi_params,
                                     "apply_merge_list",
                                     pe=qsub_pe, queue=qsub_queue,
                                     script_folder=None,
                                     n_max_co_processes=n_max_co_processes)

    else:
        raise Exception("QSUB not available")


def _apply_merge_list_thread(args):
    chunk = args[0]
    filename = args[1]
    hdf5names = args[2]
    merge_list_dict_path = args[3]
    postfix = args[4]

    cc_data_list = basics.load_from_h5py(chunk.folder + filename +
                                         "_unique_components%s.h5"
                                         % postfix, hdf5names)

    merge_list_dict = pkl.load(open(merge_list_dict_path, 'rb'))

    for nb_hdf5_name in range(len(hdf5names)):
        hdf5_name = hdf5names[nb_hdf5_name]
        this_cc = cc_data_list[nb_hdf5_name]
        id_changer = merge_list_dict[hdf5_name]
        this_shape = this_cc.shape
        offset = (np.array(this_shape) - chunk.size) // 2  # offset needs to be integer

        this_cc = this_cc[offset[0]: this_shape[0] - offset[0],
                          offset[1]: this_shape[1] - offset[1],
                          offset[2]: this_shape[2] - offset[2]]
        this_cc = id_changer[this_cc]
        cc_data_list[nb_hdf5_name] = np.array(this_cc, dtype=np.uint32)

    basics.save_to_h5py(cc_data_list,
                        chunk.folder + filename +
                        "_stitched_components%s.h5" % postfix,
                        hdf5names)


def extract_voxels(cset, filename, hdf5names=None, dataset_names=None,
                   n_folders_fs=10000, debug=False,
                   workfolder=None, overlaydataset_path=None,
                   chunk_list=None, suffix="", n_chunk_jobs=5000,
                   use_work_dir=True, qsub_pe=None, qsub_queue=None,
                   n_max_co_processes=None, nb_cpus=None, transform_func=None,
                   transform_func_kwargs=None):  # TODO: nb_cpus=1 when memory consumption fixed
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
    chunk_list: List[int] or None
        Selective list of chunks for which this function should work on. If None
        all chunks are used.
    debug: boolean
        If true multiprocessed steps only operate on one core using 'map' which
        allows for better error messages
    suffix: str
        Suffix for the intermediate results
    qsub_pe: str or None
        qsub parallel environment
    qsub_queue: str or None
        qsub queue
    nb_cpus : int
        number of parallel jobs
    transform_func_kwargs : dict
    n_chunk_jobs : int
        Total number of jobs
    n_max_co_processes : int
        Jobs run in parallel
    use_work_dir : bool
        Unclear what this is for
    workfolder : str
        Working directory of SyConn

    """

    if chunk_list is None:
        chunk_list = [ii for ii in range(len(cset.chunk_dict))]
    if use_work_dir:
        if workfolder is None:
            workfolder = os.path.dirname(cset.path_head_folder.rstrip("/"))
    else:
        workfolder = cset.path_head_folder
    voxel_rel_paths = [rh.subfold_from_ix(ix, n_folders_fs) for ix in range(n_folders_fs)]
    if dataset_names is not None:
        for dataset_name in dataset_names:
            dataset_path = workfolder + "/%s_temp/" % dataset_name
            if os.path.exists(dataset_path):
                shutil.rmtree(dataset_path)
    else:
        dataset_names = hdf5names

        for hdf5_name in hdf5names:
            dataset_path = workfolder + "/%s_temp/" % hdf5_name
            if os.path.exists(dataset_path):
                shutil.rmtree(dataset_path)
    multi_params = []
    if n_chunk_jobs > len(chunk_list):
        n_chunk_jobs = len(chunk_list)

    if n_chunk_jobs > len(voxel_rel_paths):
        n_chunk_jobs = len(voxel_rel_paths)
    chunk_blocks = np.array_split(np.array(chunk_list), n_chunk_jobs)
    path_blocks = np.array_split(np.array(voxel_rel_paths), n_chunk_jobs)
    for i_job in range(n_chunk_jobs):
        multi_params.append([[cset.chunk_dict[nb_chunk] for nb_chunk in chunk_blocks[i_job]], workfolder,
                             filename, hdf5names, dataset_names, overlaydataset_path,
                             suffix, path_blocks[i_job], n_folders_fs, transform_func, transform_func_kwargs])
    if (qsub_pe is None and qsub_queue is None) or not qu.batchjob_enabled():
        results = sm.start_multiprocess_imap(_extract_voxels_thread, multi_params,
                                             nb_cpus=nb_cpus, verbose=debug, debug=debug)
    elif qu.batchjob_enabled():
        path_to_out = qu.QSUB_script(multi_params,
                                     "extract_voxels",
                                     pe=qsub_pe, queue=qsub_queue,
                                     script_folder=None,
                                     n_max_co_processes=n_max_co_processes,
                                     n_cores=nb_cpus)
        out_files = glob.glob(path_to_out + "/*")
        results = []
        for out_file in out_files:
            with open(out_file, 'rb') as f:
                results.append(pkl.load(f))
    else:
        raise Exception("QSUB not available")
    for i_hdf5_name, hdf5_name in enumerate(hdf5names):
        dataset_path = workfolder + "/%s_temp/" % dataset_names[i_hdf5_name]
        remap_dict = defaultdict(list)
        for result in results:
            for key, value in result[hdf5_name].items():
                remap_dict[key].append(value)
        with open("%s/remapping_dict.pkl" % dataset_path, "wb") as f:
            pkl.dump(remap_dict, f)


def _extract_voxels_thread(args):
    chunks = args[0]
    workfolder = args[1]
    filename = args[2]
    hdf5names = args[3]
    dataset_names = args[4]
    overlaydataset_path = args[5]
    suffix = args[6]
    voxel_paths = args[7]
    n_folders_fs = args[8]
    transform_func = args[9]
    transform_func_kwargs = args[10]
    # TODO: finish support for syn object extraction -- do not extract all voxels, instead compute bounding box
    # TODO: and enable dynamic queries via VoxelStorage only if neccessary

    map_dict = {}
    for hdf5_name in hdf5names:
        map_dict[hdf5_name] = {}

    for nb_hdf5_name in range(len(hdf5names)):
        hdf5_name = hdf5names[nb_hdf5_name]
        dataset_name = dataset_names[nb_hdf5_name]
        dataset_path = workfolder + "/%s_temp/" % dataset_name

        cur_path_id = 0
        voxel_dc = VoxelStorage(
            dataset_path + voxel_paths[cur_path_id] + "/voxel.pkl",
            read_only=False, disable_locking=True)

        p_parts = voxel_paths[cur_path_id].strip("/").split("/")
        next_id = int("".join(p_parts))

        id_count = 0

        for i_chunk, chunk in enumerate(chunks):

            if overlaydataset_path is None:
                path = chunk.folder + filename + "_stitched_components%s.h5" % suffix

                if not os.path.exists(path):
                    path = chunk.folder + filename + ".h5"
                this_segmentation = basics.load_from_h5py(path, [hdf5_name])[0]
            else:
                kd = knossosdataset.KnossosDataset()
                kd.initialize_from_knossos_path(overlaydataset_path)

                try:
                    this_segmentation = kd.from_overlaycubes_to_matrix(chunk.size,
                                                                       chunk.coordinates)
                except:
                    this_segmentation = kd.from_overlaycubes_to_matrix(chunk.size,
                                                                       chunk.coordinates,
                                                                       datatype=np.uint32)

            uniqueID_coords_dict = defaultdict(list)  # {sv_id: [(x0,y0,z0),(x1,y1,z1),...]}

            dims = this_segmentation.shape
            indices = itertools.product(range(dims[0]), range(dims[1]), range(dims[2]))
            for idx in indices:
                sv_id = this_segmentation[idx]
                uniqueID_coords_dict[sv_id].append(idx)

            if i_chunk == 0:
                n_per_voxel_path = np.ceil(float(len(uniqueID_coords_dict) * len(chunks)) / len(voxel_paths))

            for sv_id in uniqueID_coords_dict:
                if sv_id == 0:
                    continue
                sv_coords = uniqueID_coords_dict[sv_id]
                id_mask_offset = np.min(sv_coords, axis=0)
                abs_offset = chunk.coordinates + id_mask_offset
                id_mask_coords = sv_coords - id_mask_offset
                size = np.max(sv_coords, axis=0) - id_mask_offset + (1, 1, 1)
                id_mask_coords = np.transpose(id_mask_coords)
                id_mask = np.zeros(tuple(size), dtype=bool)
                id_mask[id_mask_coords[0, :], id_mask_coords[1, :], id_mask_coords[2, :]] = True

                if next_id in voxel_dc:
                    voxel_dc.append(next_id, id_mask, abs_offset)
                else:
                    voxel_dc[next_id] = [id_mask], [abs_offset]
                if not sv_id in map_dict[hdf5_name]:
                    map_dict[hdf5_name][sv_id] = [next_id]
                else:
                    map_dict[hdf5_name][sv_id].append(next_id)

                id_count += 1

                if id_count > (cur_path_id + 1) * n_per_voxel_path and \
                        cur_path_id + 1 < len(voxel_paths):
                    voxel_dc.push(dataset_path + voxel_paths[cur_path_id] + "/voxel.pkl")
                    cur_path_id += 1
                    voxel_dc = VoxelStorage(dataset_path + voxel_paths[cur_path_id],
                                            read_only=False, disable_locking=True)
                    p_parts = voxel_paths[cur_path_id].strip("/").split("/")
                    next_id = int("".join(p_parts))
                else:
                    next_id += n_folders_fs

            voxel_dc.push(dataset_path + voxel_paths[cur_path_id] + "/voxel.pkl")

    return map_dict


def combine_voxels(workfolder, hdf5names, dataset_names=None,
                   n_folders_fs=10000, stride=10, nb_cpus=None, sd_version=0,
                   qsub_pe=None, qsub_queue=None, n_max_co_processes=None):
    """
    Extracts voxels for each component id and ceates a SegmentationDataset of type hdf5names.
    SegmentationDataset will always have version 0!

    Parameters
    ----------
    workfolder : str
    hdf5names: List[str]
        Names/labels to be extracted and processed from the prediction
        file
    sd_version : int or str
        0 by default
    n_folders_fs : int
    stride : int
    nb_cpus : int
    qsub_pe: str or None
        qsub parallel environment
    qsub_queue: str or None
        qsub queue
    n_max_co_processes : int

    """
    if dataset_names is None:
        dataset_names = hdf5names
    for ii in range(len(hdf5names)):
        hdf5_name = hdf5names[ii]
        voxel_rel_paths = [rh.subfold_from_ix(ix, n_folders_fs) for ix in
                           range(n_folders_fs)]

        segdataset = segmentation.SegmentationDataset(obj_type=dataset_names[ii],
                                                      working_dir=workfolder,
                                                      version=sd_version,
                                                      create=True, n_folders_fs=n_folders_fs)

        multi_params = []
        path_blocks = np.array_split(np.array(voxel_rel_paths),
                                     int(len(voxel_rel_paths) / stride))

        dataset_temp_path = workfolder + "/%s_temp/" % hdf5_name
        with open(dataset_temp_path + "/remapping_dict.pkl", "rb") as f:
            mapping_dict = pkl.load(f)

        voxel_rel_path_dict = {}
        for voxel_rel_path in voxel_rel_paths:

            end_id = "00000" + "".join(voxel_rel_path.strip('/').split('/'))
            end_id = end_id[-int(np.log10(n_folders_fs)):]
            voxel_rel_path_dict[end_id] = {}

        for so_id in list(mapping_dict.keys()):
            end_id = "00000%d" % so_id
            end_id = end_id[-int(np.log10(n_folders_fs)):]

            voxel_rel_path_dict[end_id][so_id] = mapping_dict[so_id]

        for path_block in path_blocks:
            path_block_dicts = []

            for voxel_rel_path in path_block:
                end_id = "00000" + "".join(voxel_rel_path.strip('/').split('/'))
                end_id = end_id[-int(np.log10(n_folders_fs)):]
                path_block_dicts.append(voxel_rel_path_dict[end_id])

            multi_params.append([workfolder, hdf5_name, dataset_names[ii], path_block,
                                 path_block_dicts, segdataset.version,
                                 n_folders_fs])

        if (qsub_pe is None and qsub_queue is None) or not qu.batchjob_enabled():
            results = sm.start_multiprocess_imap(_combine_voxels_thread,
                                            multi_params, nb_cpus=nb_cpus)

        elif qu.batchjob_enabled():
            path_to_out = qu.QSUB_script(multi_params,
                                         "combine_voxels",
                                         pe=qsub_pe, queue=qsub_queue,
                                         script_folder=None,
                                         n_max_co_processes=n_max_co_processes,
                                         n_cores=nb_cpus)

        else:
            raise Exception("QSUB not available")


def _combine_voxels_thread(args):
    workfolder = args[0]
    hdf5_name = args[1]
    dataset_name = args[2]
    voxel_rel_paths = args[3]
    path_block_dicts = args[4]
    dataset_version = args[5]
    n_folders_fs = args[6]

    dataset_temp_path = workfolder + "/%s_temp/" % hdf5_name

    segdataset = segmentation.SegmentationDataset(obj_type=dataset_name,  working_dir=workfolder,
                                                  version=dataset_version, n_folders_fs=n_folders_fs)

    for i_voxel_rel_path, voxel_rel_path in enumerate(voxel_rel_paths):
        voxel_dc = VoxelStorage(segdataset.so_storage_path + voxel_rel_path +
                                "/voxel.pkl", read_only=False,
                                disable_locking=True)
        for so_id in path_block_dicts[i_voxel_rel_path]:
            fragments = path_block_dicts[i_voxel_rel_path][so_id]
            fragments = [item for sublist in fragments for item in sublist]
            for i_fragment_id, fragment_id in enumerate(fragments):
                voxel_dc_read = VoxelStorage(dataset_temp_path +
                                             rh.subfold_from_ix(fragment_id, n_folders_fs) + "/voxel.pkl",
                                             read_only=True, disable_locking=True)
                bin_arrs, block_offsets = voxel_dc_read[fragment_id]
                if i_fragment_id == 0:
                    voxel_dc[so_id] = bin_arrs, block_offsets
                else:
                    voxel_dc.append(so_id, bin_arrs[0], block_offsets[0])
        voxel_dc.push(segdataset.so_storage_path + voxel_rel_path + "/voxel.pkl")


def extract_voxels_combined(cset, filename, hdf5names=None, dataset_names=None,
                   n_folders_fs=10000, workfolder=None, overlaydataset_path=None,
                   chunk_list=None, suffix="", n_chunk_jobs=5000, sd_version=0,
                   use_work_dir=True, qsub_pe=None, qsub_queue=None, qsub_slots=1,
                   n_max_co_processes=None, nb_cpus=None, object_names=None):
    """
    Creates a SegmentationDataset of type object_names/dataset_names/hdf5names  # TODO fix this redundancy once and for all


    Parameters
    ----------
    cset :
    filename :
    hdf5names :
    dataset_names :
    sd_version : int or str
    n_folders_fs :
    workfolder :
    overlaydataset_path :
    chunk_list :
    suffix :
    n_chunk_jobs :
    use_work_dir :
    qsub_pe :
    qsub_queue :
    qsub_slots :
    n_max_co_processes :
    nb_cpus :
    object_names :

    Returns
    -------

    """
    if object_names is None:
        object_names = hdf5names

    if chunk_list is None:
        chunk_list = [ii for ii in range(len(cset.chunk_dict))]

    if use_work_dir:
        if workfolder is None:
            workfolder = os.path.dirname(cset.path_head_folder.rstrip("/"))
    else:
        workfolder = cset.path_head_folder

    voxel_rel_paths_2stage = []
    for ix in range(n_folders_fs):
        vp = ""
        for p in rh.subfold_from_ix(ix, n_folders_fs).strip('/').split('/')[:-1]:
            vp += p + "/"
        voxel_rel_paths_2stage.append(vp)

    voxel_rel_paths_2stage = np.unique(voxel_rel_paths_2stage)

    if dataset_names is not None:
        # for dataset_name in dataset_names:    TODO: handle
        raise NotImplementedError
        #     dataset_path = workfolder + "/%s_temp/" % dataset_name
        #     if os.path.exists(dataset_path):
        #         shutil.rmtree(dataset_path)
        #
        #    for p in voxel_rel_paths_2stage:
        #         os.makedirs(dataset_path + p)
        pass    # TODO: del
    else:
        dataset_names = hdf5names
        if object_names is not None and len(object_names) != len(hdf5names):
            raise ValueError('object_names were specified but did not match length of "dataset_names"/"hdf5names"')
        for kk, hdf5_name in enumerate(hdf5names):
            object_name = object_names[kk]
            segdataset = segmentation.SegmentationDataset(version=sd_version,
                obj_type=object_name, working_dir=workfolder, create=True, n_folders_fs=n_folders_fs)
            dataset_path = segdataset.so_storage_path
            if os.path.exists(dataset_path):
                shutil.rmtree(dataset_path)

            # TODO: can probably be removed
            # for p in voxel_rel_paths_2stage:
            #     os.makedirs(dataset_path + p, exist_ok=True)

    multi_params = []
    if n_chunk_jobs > len(chunk_list):
        n_chunk_jobs = len(chunk_list)

    chunk_blocks = np.array_split(np.array(chunk_list), n_chunk_jobs)

    for i_job in range(n_chunk_jobs):
        multi_params.append([[cset.chunk_dict[nb_chunk] for nb_chunk in chunk_blocks[i_job]], workfolder,
                             filename, hdf5names, dataset_names, overlaydataset_path,
                             suffix, n_folders_fs, object_names, sd_version])

    if (qsub_pe is None and qsub_queue is None) or not qu.batchjob_enabled():
        results = sm.start_multiprocess_imap(_extract_voxels_combined_thread,
                                        multi_params, nb_cpus=nb_cpus)

    elif qu.batchjob_enabled():
        path_to_out = qu.QSUB_script(multi_params, "extract_voxels_combined",
                                     pe=qsub_pe, queue=qsub_queue, n_cores=qsub_slots,
                                     script_folder=None, n_max_co_processes=n_max_co_processes)


def _extract_voxels_combined_thread(args):
    chunks = args[0]
    workfolder = args[1]
    filename = args[2]
    hdf5names = args[3]
    dataset_names = args[4]
    overlaydataset_path = args[5]
    suffix = args[6]
    n_folders_fs = args[7]
    object_names = args[8]
    sd_version = args[9]
    if object_names is None:
        object_names = hdf5names
    for nb_hdf5_name in range(len(hdf5names)):
        hdf5_name = hdf5names[nb_hdf5_name]
        object_name = object_names[nb_hdf5_name]
        segdataset = segmentation.SegmentationDataset(obj_type=object_name,
                                                      working_dir=workfolder, create=True,
                                                      version=sd_version,
                                                      n_folders_fs=n_folders_fs)

        for i_chunk, chunk in enumerate(chunks):

            if overlaydataset_path is None:
                path = chunk.folder + filename + "_stitched_components%s.h5" % suffix

                if not os.path.exists(path):
                    path = chunk.folder + filename + ".h5"
                this_segmentation = basics.load_from_h5py(path, [hdf5_name])[0]
            else:
                kd = knossosdataset.KnossosDataset()
                kd.initialize_from_knossos_path(overlaydataset_path)

                try:
                    this_segmentation = kd.from_overlaycubes_to_matrix(chunk.size,
                                                                       chunk.coordinates)
                except:
                    this_segmentation = kd.from_overlaycubes_to_matrix(chunk.size,
                                                                       chunk.coordinates,
                                                                       datatype=np.uint32)

            uniqueID_coords_dict = defaultdict(list)  # {sv_id: [(x0,y0,z0),(x1,y1,z1),...]}

            dims = this_segmentation.shape
            indices = itertools.product(range(dims[0]), range(dims[1]), range(dims[2]))
            for idx in indices:
                sv_id = this_segmentation[idx]
                uniqueID_coords_dict[sv_id].append(idx)

            for sv_id in uniqueID_coords_dict:
                if sv_id == 0:
                    continue
                sv_coords = uniqueID_coords_dict[sv_id]
                id_mask_offset = np.min(sv_coords, axis=0)
                abs_offset = chunk.coordinates + id_mask_offset
                id_mask_coords = sv_coords - id_mask_offset
                size = np.max(sv_coords, axis=0) - id_mask_offset + (1, 1, 1)
                id_mask_coords = np.transpose(id_mask_coords)
                id_mask = np.zeros(tuple(size), dtype=bool)
                id_mask[id_mask_coords[0, :], id_mask_coords[1, :], id_mask_coords[2, :]] = True
                voxel_rel_path = rh.subfold_from_ix(sv_id, n_folders_fs)
                voxel_dc = VoxelStorageL(
                    segdataset.so_storage_path + voxel_rel_path + "/voxel.pkl",
                    read_only=False, disable_locking=False)

                if sv_id in voxel_dc:
                    voxel_dc.append(sv_id, id_mask, abs_offset)
                else:
                    voxel_dc[sv_id] = [id_mask], [abs_offset]

                voxel_dc.push(segdataset.so_storage_path + voxel_rel_path + "/voxel.pkl")
