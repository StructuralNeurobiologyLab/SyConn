# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Philipp Schubert, Joergen Kornfeld

import glob
import os
import pickle as pkl
import shutil

import networkx as nx
import numpy as np
import scipy.ndimage
import skimage.segmentation
from knossos_utils import chunky

from .block_processing_C import relabel_vol
from .. import global_params
from ..handler import basics, log_handler, compression
from ..handler.basics import kd_factory
from ..mp import batchjob_utils as qu, mp_utils as sm
from ..proc.general import cut_array_in_one_dim
from ..proc.image import apply_morphological_operations, get_aniso_struct

try:
    import vigra
    from vigra.filters import gaussianSmoothing, distanceTransform
    from vigra.analysis import watershedsNew
except ImportError as e:
    gaussianSmoothing = None
    log_handler.error('ImportError. Could not import VIGRA. '
                      '`object_segmentation` will not be possible. {}'.format(e))


def gauss_threshold_connected_components(*args, **kwargs):
    # alias
    return object_segmentation(*args, **kwargs)


def object_segmentation(cset, filename, hdf5names, overlap="auto", sigmas=None,
                        thresholds=None, chunk_list=None, debug=False,
                        swapdata=False, prob_kd_path_dict=None,
                        membrane_filename=None, membrane_kd_path=None,
                        hdf5_name_membrane=None, fast_load=False, suffix="",
                        nb_cpus=None, transform_func=None,
                        transform_func_kwargs=None, transf_func_kd_overlay=None,
                        load_from_kd_overlaycubes=False, n_chunk_jobs=None):
    """
    Extracts connected component from probability maps.

    By default the following procedure is used:
    1. Gaussian filter (defined by sigma)
    2. Thresholding (defined by threshold)
    3. Connected components analysis

    If `transform_func` is set, the specified method will be applied by every
    worker on the chunk's probability map to generate the segmentation instead.
    Add `transform_func_kwargs` in case `transform_func` specific arguments.

    In case of vesicle clouds (hdf5_name in ["p4", "vc"]) the membrane
    segmentation is used to cut connected vesicle clouds across cells
    apart (only if membrane segmentation is provided).

    Args:
    cset : chunkdataset instance
    filename (str) : Filename of the prediction in the ChunkDataset.
    hdf5names (list): list of strings
        List of names/ labels to be extracted and processed from the prediction
        file.
    overlap (str): str or np.array
        Defines the overlap with neighbouring chunks that is left for later
        processing steps; if 'auto' the overlap is calculated from the sigma and
        the stitch_overlap (here: [1., 1., 1.]) and the number of binary erosion
        in global_params.config['cell_objects']['extract_morph_op'].
    sigmas (list): list of lists or None
        Defines the sigmas of the gaussian filters applied to the probability
        maps. Has to be the same length as hdf5names. If None no gaussian filter
        is applied.
    thresholds(list of float or np.ndarray):
        Threshold for cutting the probability map. Has to be the same length as
        hdf5names. If None zeros are used instead (not recommended!)
    chunk_list(list): 
        Selective list of chunks for which this function should work on. If None
        all chunks are used.
    debug(bool): 
        If true multiprocessed steps only operate on one core using 'map' which
        allows for better error messages.
    swapdata(bool):
        If true an x-z swap is applied to the data prior to processing.
    prob_kd_path_dict:
    membrane_filename(str):
        One way to allow access to a membrane segmentation when processing
        vesicle clouds. Filename of the prediction in the chunkdataset. The
        threshold is currently set at 0.4.
    membrane_kd_path(str):
        One way to allow access to a membrane segmentation when processing
        vesicle clouds. Path to the knossosdataset containing a membrane
        segmentation. The threshold is currently set at 0.4.
    hdf5_name_membrane(str):
        When using the membrane_filename this key has to be given to access the
        data in the saved chunk.
    fast_load(bool):
        If true the data of chunk is blindly loaded without checking for enough
        offset to compute the overlap area. Faster, because no neighbouring
        chunk has to be accessed since the default case loads th overlap area
        from them.
    suffix(str):
        Suffix for the intermediate results.
    nb_cpus:
    transform_func(callable):
        Segmentation method which is applied.
    transform_func_kwargs(dict) :
        key word arguments for transform_func
    load_from_kd_overlaycubes(bool) :
        Load prob/seg data from overlaycubes instead of raw cubes.
    transf_func_kd_overlay :
        Method which is to applied to cube data if `load_from_kd_overlaycubes`
        is True.
    n_chunk_jobs:

    Returns:
    results_as_list(list):
        list containing information about the number of connected components
        in each chunk
    overlap(np.array):
    stitch overlap(np.array):
    """
    if transform_func is None:
        transform_func = _object_segmentation_thread

    if thresholds is None:
        thresholds = np.zeros(len(hdf5names))
    if sigmas is None:
        sigmas = np.zeros(len(hdf5names))
    if not len(sigmas) == len(thresholds) == len(hdf5names):
        raise Exception("Number of thresholds, sigmas and HDF5 names does not "
                        "match!")
    if n_chunk_jobs is None:
        n_chunk_jobs = global_params.config.ncore_total * 4

    if chunk_list is None:
        chunk_list = [ii for ii in range(len(cset.chunk_dict))]

    rand_ixs = np.arange(len(chunk_list))
    np.random.seed(0)
    np.random.shuffle(rand_ixs)
    chunk_list = np.array(chunk_list)[rand_ixs]
    chunk_blocks = basics.chunkify(chunk_list, n_chunk_jobs)

    if overlap is "auto":
        if sigmas is None:
            max_sigma = np.zeros(3)
        else:
            max_sigma = np.array([np.max(sigmas)] * 3)
        overlap = np.ceil(max_sigma * 4)
        morph_ops = global_params.config['cell_objects']['extract_morph_op']
        scaling = global_params.config['scaling']
        aniso = scaling[2] // scaling[0]
        n_erosions = 0
        for k, v in morph_ops.items():
            v = np.array(v)
            # factor 2: erodes both sides; aniso: morphological operation kernel is laterally increased by this factor
            n_erosions = max(n_erosions, 2 * aniso * np.sum(v == 'binary_erosion'))
        overlap = np.max([overlap, [n_erosions, n_erosions, n_erosions // aniso]], axis=0).astype(np.int32)

    stitch_overlap = np.max([overlap.copy(), [1, 1, 1]], axis=0)

    multi_params = []
    for chunk_sub in chunk_blocks:
        multi_params.append(
            [[cset.chunk_dict[nb_chunk] for nb_chunk in chunk_sub],
             cset.path_head_folder, filename, hdf5names, overlap,
             sigmas, thresholds, swapdata, prob_kd_path_dict,
             membrane_filename, membrane_kd_path,
             hdf5_name_membrane, fast_load, suffix, transform_func_kwargs,
             load_from_kd_overlaycubes, transf_func_kd_overlay])

    if not qu.batchjob_enabled():
        results = sm.start_multiprocess_imap(transform_func, multi_params, nb_cpus=nb_cpus, debug=False,
                                             use_dill=True)

        results_as_list = []
        for result in results:
            for entry in result:
                results_as_list.append(entry)

    else:
        assert transform_func == _object_segmentation_thread, "batch jobs currently only supported for " \
                                                              "`_object_segmentation_thread`."
        path_to_out = qu.batchjob_script(
            multi_params, "object_segmentation", n_cores=nb_cpus, use_dill=True, suffix=filename)
        out_files = glob.glob(path_to_out + "/*")
        results_as_list = []
        for out_file in out_files:
            with open(out_file, 'rb') as f:
                for entry in pkl.load(f):
                    results_as_list.append(entry)
        shutil.rmtree(os.path.abspath(path_to_out + "/../"), ignore_errors=True)
    return results_as_list, [overlap, stitch_overlap]


def _object_segmentation_thread(args):
    """
    Default worker of object_segmentation. Performs a gaussian blur with
     subsequent thresholding to extract connected components of a probability
     map. Result summaries are returned and connected components are stored as
     .h5 files.
     TODO: Add generic '_segmentation_thread' to enable a clean support of
     custom-made segmentation functions passed to 'object_segmentation' via
     'transform_func'-kwargs

    Args:
        args(list) :

    Returns:
        list of lists: Results of connected component analysis
    """
    chunks = args[0]
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
    transform_func_kwargs = args[14]
    load_from_kd_overlaycubes = args[15]
    transf_func_kd_overlay = args[16]

    # e.g. {'sj': ['binary_closing', 'binary_opening'], 'mi': [], 'cell': []}
    morph_ops = global_params.config['cell_objects']['extract_morph_op']
    min_seed_vx = global_params.config['cell_objects']['min_seed_vx']
    scaling = np.array(global_params.config['scaling'])
    struct = get_aniso_struct(scaling)
    nb_cc_list = []

    for chunk in chunks:
        box_offset = np.array(chunk.coordinates) - np.array(overlap)
        size = np.array(chunk.size) + 2 * np.array(overlap)

        if swapdata:
            size = basics.switch_array_entries(size, [0, 2])
        if prob_kd_path_dict is not None:
            bin_data_dict = {}
            if load_from_kd_overlaycubes:  # enable possibility to load from overlay cubes as well
                data_k = None
                exp_value = next(iter(prob_kd_path_dict.values()))
                all_equal = all(v == exp_value for v in prob_kd_path_dict.values())
                if all_equal:
                    kd = kd_factory(prob_kd_path_dict[hdf5names[0]])
                    data_k = kd.load_seg(size=size, offset=box_offset, mag=1).swapaxes(0, 2)
                for kd_key in hdf5names:
                    if not all_equal:
                        kd = kd_factory(prob_kd_path_dict[kd_key])
                        data_k = kd.load_seg(size=size, offset=box_offset, mag=1).swapaxes(0, 2)
                    if transf_func_kd_overlay is not None:
                        bin_data_dict[kd_key] = transf_func_kd_overlay[kd_key](data_k)
                    else:
                        bin_data_dict[kd_key] = data_k
            else:  # load raw
                for kd_key in prob_kd_path_dict.keys():
                    kd = kd_factory(prob_kd_path_dict[kd_key])
                    bin_data_dict[kd_key] = kd.load_raw(size=size, offset=box_offset,
                                                        mag=1).swapaxes(0, 2)
        else:
            if not fast_load:
                cset = chunky.load_dataset(path_head_folder)
                bin_data_dict = cset.from_chunky_to_matrix(size, box_offset,
                                                           filename, hdf5names)
            else:
                bin_data_dict = compression.load_from_h5py(chunk.folder + filename + ".h5",
                                                           hdf5names, as_dict=True)

        labels_data = []
        for nb_hdf5_name in range(len(hdf5names)):
            hdf5_name = hdf5names[nb_hdf5_name]
            tmp_data = bin_data_dict[hdf5_name]

            tmp_data_shape = tmp_data.shape
            offset = (np.array(tmp_data_shape) - np.array(chunk.size) -
                      2 * np.array(overlap)) / 2
            offset = offset.astype(np.int32)
            if np.any(offset < 0):
                offset = np.array([0, 0, 0])
            tmp_data = tmp_data[offset[0]: tmp_data_shape[0] - offset[0],
                       offset[1]: tmp_data_shape[1] - offset[1],
                       offset[2]: tmp_data_shape[2] - offset[2]]

            if np.sum(sigmas[nb_hdf5_name]) != 0:
                tmp_data = gaussianSmoothing(tmp_data, sigmas[nb_hdf5_name])

            if hdf5_name in ["p4", "vc"] and membrane_filename is not None and hdf5_name_membrane is not None:
                membrane_data = compression.load_from_h5py(chunk.folder + membrane_filename + ".h5",
                                                           hdf5_name_membrane)[0]
                membrane_data_shape = membrane_data.shape
                offset = (np.array(membrane_data_shape) - np.array(tmp_data.shape)) / 2
                membrane_data = membrane_data[offset[0]: membrane_data_shape[0] - offset[0],
                                offset[1]: membrane_data_shape[1] - offset[1],
                                offset[2]: membrane_data_shape[2] - offset[2]]
                tmp_data[membrane_data > 255 * .4] = 0
                del membrane_data
            elif hdf5_name in ["p4", "vc"] and membrane_kd_path is not None:
                kd_bar = kd_factory(membrane_kd_path)
                membrane_data = kd_bar.load_raw(size=size, offset=box_offset,
                                                mag=1).swapaxes(0, 2)
                tmp_data[membrane_data > 255 * .4] = 0
                del membrane_data
            if thresholds[nb_hdf5_name] != 0 and not load_from_kd_overlaycubes:
                tmp_data = np.array(tmp_data > thresholds[nb_hdf5_name], dtype=np.uint8)

            if hdf5_name in morph_ops:
                if 'binary_erosion' in morph_ops[hdf5_name]:
                    first_erosion_ix = morph_ops[hdf5_name].index('binary_erosion')
                    tmp_data = apply_morphological_operations(tmp_data.copy(), morph_ops[hdf5_name][:first_erosion_ix],
                                                              mop_kwargs=dict(structure=struct))
                    # apply erosion operations to generate watershed seeds
                    markers = apply_morphological_operations(tmp_data.copy(),
                                                             morph_ops[hdf5_name][first_erosion_ix:],
                                                             mop_kwargs=dict(structure=struct))
                    markers = scipy.ndimage.label(markers)[0].astype(np.uint32)
                    # remove small fragments and 0; this might also delete objects bigger than min_size as
                    # this threshold is applied after N binary erosion!
                    if hdf5_name in min_seed_vx and min_seed_vx[hdf5_name] > 1:
                        min_size = min_seed_vx[hdf5_name]
                        ixs, cnt = np.unique(markers, return_counts=True)
                        m = (ixs != 0) & (cnt < min_size)
                        ixs_del = np.sort(ixs[m])
                        ixs_keep = np.sort(ixs[~m])
                        # set small objects to 0
                        label_m = {ix_del: 0 for ix_del in ixs_del}
                        # fill "holes" in ID space with to-be-kept object IDs
                        ii = len(ixs_keep) - 1
                        for ix_del in ixs_del:
                            if (ix_del > ixs_keep[ii]) or (ixs_keep[ii] == 0) or (ii < 0):
                                break
                            label_m[ixs_keep[ii]] = ix_del
                            ii -= 1
                        # in-place modification of markers array
                        relabel_vol(markers, label_m)

                    distance = distanceTransform(tmp_data.astype(np.uint32, copy=False), background=False,
                                                 pixel_pitch=scaling.astype(np.uint32))
                    this_labels_data = skimage.segmentation.watershed(-distance, markers, mask=tmp_data)
                    max_label = np.max(this_labels_data)
                else:
                    mop_data = apply_morphological_operations(tmp_data.copy(), morph_ops[hdf5_name],
                                                              mop_kwargs=dict(structure=struct))
                    this_labels_data, max_label = scipy.ndimage.label(mop_data)
            else:
                this_labels_data, max_label = scipy.ndimage.label(tmp_data)
            nb_cc_list.append([chunk.number, hdf5_name, max_label])
            labels_data.append(this_labels_data)

        h5_fname = chunk.folder + filename + "_connected_components%s.h5" % suffix
        os.makedirs(os.path.split(h5_fname)[0], exist_ok=True)
        compression.save_to_h5py(labels_data, h5_fname, hdf5names)
        del labels_data
    return nb_cc_list


def make_unique_labels(cset, filename, hdf5names, chunk_list, max_nb_dict,
                       chunk_translator, debug, suffix="",
                       n_chunk_jobs=None, nb_cpus=1):
    """
    Makes labels unique across chunks

    Args:
        cset : chunkdataset instance
        filename(str) :
            Filename of the prediction in the chunkdataset
        hdf5names(list): list of str
            List of names/ labels to be extracted and processed from the prediction
            file
        chunk_list(list): list of int
            Selective list of chunks for which this function should work on. If None
            all chunks are used.
        max_nb_dict(dict):
            Maps each chunk id to a integer describing which needs to be added to
            all its entries
        chunk_translator(dict):
            Remapping from chunk ids to position in chunk_list
        debug(bool):
            If true multiprocessed steps only operate on one core using 'map' which
            allows for better error messages
        suffix: str
            Suffix for the intermediate results
        n_chunk_jobs: int
            Number of total jobs.
        nb_cpus: int
    """

    if n_chunk_jobs is None:
        n_chunk_jobs = global_params.config.ncore_total
    chunk_blocks = basics.chunkify(chunk_list, n_chunk_jobs)
    multi_params_glob = []
    for chunk_sub in chunk_blocks:
        multi_params = []
        for nb_chunk in chunk_sub:
            this_max_nb_dict = {}
            for hdf5_name in hdf5names:
                this_max_nb_dict[hdf5_name] = max_nb_dict[hdf5_name][
                    chunk_translator[nb_chunk]]

            multi_params.append([cset.chunk_dict[nb_chunk], filename, hdf5names,
                                 this_max_nb_dict, suffix])
        multi_params_glob.append(multi_params)

    if not qu.batchjob_enabled():
        _ = sm.start_multiprocess_imap(_make_unique_labels_thread,
                                       multi_params_glob, debug=debug)

    else:
        _ = qu.batchjob_script(
            multi_params_glob, "make_unique_labels", suffix=filename,
            remove_jobfolder=True, n_cores=nb_cpus)


def _make_unique_labels_thread(func_args):
    for args in func_args:
        chunk = args[0]
        filename = args[1]
        hdf5names = args[2]
        this_max_nb_dict = args[3]
        suffix = args[4]

        cc_data_list = compression.load_from_h5py(
            chunk.folder + filename + "_connected_components%s.h5" % suffix, hdf5names)

        for nb_hdf5_name in range(len(hdf5names)):
            hdf5_name = hdf5names[nb_hdf5_name]
            cc_data_list[nb_hdf5_name] = cc_data_list[nb_hdf5_name].astype(np.uint64)
            matrix = cc_data_list[nb_hdf5_name]
            matrix[matrix > 0] += this_max_nb_dict[hdf5_name]

        compression.save_to_h5py(cc_data_list, chunk.folder + filename + "_unique_components%s.h5" % suffix, hdf5names)


def make_stitch_list(cset, filename, hdf5names, chunk_list, stitch_overlap,
                     overlap, debug, suffix="", nb_cpus=None,
                     overlap_thresh=0, n_chunk_jobs=None):
    """
    Creates a stitch list for the overlap region between chunks

    Args:
        cset : chunkdataset instance
        filename(str):
            Filename of the prediction in the chunkdataset
        hdf5names(list): list of str
            List of names/ labels to be extracted and processed from the prediction
            file
        chunk_list(list): list of int
            Selective list of chunks for which this function should work on. If None
            all chunks are used.
        overlap(np.array): np.array
            Defines the overlap with neighbouring chunks that is left for later
            processing steps
        stitch_overlap: np.array
            Defines the overlap with neighbouring chunks that is left for stitching
        debug: boolean
            If true multiprocessed steps only operate on one core using 'map' which
            allows for better error messages
        suffix: str
            Suffix for the intermediate results
        nb_cpus: int
            Number of cores used per worker.
        n_chunk_jobs: int
            Number of total jobs.
        overlap_thresh : float
                    Overlap fraction of object in different chunks to be considered stitched.
                    If zero this behavior is disabled.

    Returns:
        stitch_list(dict):
            Dictionary of overlapping component ids
    """
    if n_chunk_jobs is None:
        n_chunk_jobs = global_params.config.ncore_total
    chunk_blocks = basics.chunkify(chunk_list, n_chunk_jobs)
    multi_params = []

    for i_job in range(len(chunk_blocks)):
        multi_params.append([cset.path_head_folder, chunk_blocks[i_job], filename, hdf5names,
                             stitch_overlap, overlap,
                             suffix, chunk_list, overlap_thresh])

    if not qu.batchjob_enabled():
        results = sm.start_multiprocess_imap(_make_stitch_list_thread,
                                             multi_params, debug=debug)

        stitch_list = {}
        for hdf5_name in hdf5names:
            stitch_list[hdf5_name] = []
        for result in results:
            for hdf5_name in hdf5names:
                elems = result[hdf5_name]
                for elem in elems:
                    stitch_list[hdf5_name].append(elem)

    else:
        path_to_out = qu.batchjob_script(multi_params, "make_stitch_list",
                                         suffix=filename, n_cores=nb_cpus)

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
        shutil.rmtree(os.path.abspath(path_to_out + "/../"), ignore_errors=True)

    return stitch_list


def _make_stitch_list_thread(args):
    cpath_head_folder = args[0]
    nb_chunks = args[1]
    filename = args[2]
    hdf5names = args[3]
    stitch_overlap = args[4]
    overlap = args[5]
    suffix = args[6]
    chunk_list = args[7]
    overlap_thresh = args[8]

    map_dict = {}
    for nb_hdf5_name in range(len(hdf5names)):
        map_dict[hdf5names[nb_hdf5_name]] = set()
    cset = chunky.load_dataset(cpath_head_folder)
    for nb_chunk in nb_chunks:
        chunk = cset.chunk_dict[nb_chunk]
        cc_data_list = compression.load_from_h5py(chunk.folder + filename +
                                                  "_unique_components%s.h5" % suffix, hdf5names)

        # TODO: optimize get_neighbouring_chunks
        neighbours, pos = cset.get_neighbouring_chunks(chunk, chunklist=chunk_list,
                                                       con_mode=7)

        #  Compare only upper half of 6-neighborhood for every chunk
        neighbours = neighbours[np.any(pos > 0, axis=1)]
        pos = pos[np.any(pos > 0, axis=1)]

        # Compare only half of 6-neighborhood for every chunk which suffices to cover all overlap areas. Checking all
        # neighbors for every chunk would lead to twice and redundant computational load
        for ii in range(3):
            if neighbours[ii] != -1:
                compare_chunk = cset.chunk_dict[neighbours[ii]]
                cc_data_list_to_compare = \
                    compression.load_from_h5py(
                        compare_chunk.folder + filename + "_unique_components%s.h5" % suffix, hdf5names)

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
                                obj_coord_intern_compare = np.transpose(
                                    np.nonzero(cc_data_list_to_compare[nb_hdf5_name] == compare_id))
                                c1 = chunk.coordinates - chunk.overlap + obj_coord_intern + np.array([1, 1, 1])
                                c2 = compare_chunk.coordinates - compare_chunk.overlap + obj_coord_intern_compare + np.array(
                                    [1, 1, 1])
                                from scipy import spatial
                                kdt = spatial.cKDTree(c1)
                                dists, ixs = kdt.query(c2)
                                match_vx = np.sum(dists == 0)
                                match_vx_rel = 2 * float(match_vx) / (len(c1) + len(c2))
                                if match_vx_rel > 0.1:
                                    map_dict[hdf5_name].add(pair)
                                else:
                                    ignore_ids.add(pair)
                            else:
                                map_dict[hdf5_name].add(pair)
    for k, v in map_dict.items():
        map_dict[k] = list(v)
    return map_dict


def make_merge_list(hdf5names, stitch_list, max_labels):
    """
    Creates a merge list from a stitch list by mapping all connected ids to
    one id

    Args:
        hdf5names (list): list of str
            List of names/ labels to be extracted and processed from the prediction
            file
        stitch_list (dict):
            Contains pairs of overlapping component ids for each hdf5name
        max_labels (dict): dictionary
            Contains the number of different component ids for each hdf5name

    Returns:
        merge_dict (dict):
            mergelist for each hdf5name
        merge_list_dict (dict):
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
                     debug, suffix="", n_chunk_jobs=None, nb_cpus=1):
    """
    Applies merge list to all chunks

    Args:
        cset : chunkdataset instance
        chunk_list (list): list of int
            Selective list of chunks for which this function should work on. If None
            all chunks are used.
        filename (str):
            Filename of the prediction in the chunkdataset
        hdf5names (list): list of str
            List of names/ labels to be extracted and processed from the prediction
            file
        merge_list_dict (dict):
            mergedict for each hdf5name
        debug (bool):
            If true multiprocessed steps only operate on one core using 'map' which
            allows for better error messages
        suffix (str):
            Suffix for the intermediate results
        n_chunk_jobs (int):
            Number of total jobs.
        nb_cpus:
    """

    multi_params = []
    merge_list_dict_path = cset.path_head_folder + "merge_list_dict.pkl"

    f = open(merge_list_dict_path, "wb")
    pkl.dump(merge_list_dict, f, protocol=4)
    f.close()
    if n_chunk_jobs is None:
        n_chunk_jobs = global_params.config.ncore_total * 2
    chunk_blocks = basics.chunkify(chunk_list, n_chunk_jobs)

    for i_job in range(len(chunk_blocks)):
        multi_params.append([[cset.chunk_dict[nb_chunk] for nb_chunk in chunk_blocks[i_job]],
                             filename, hdf5names, merge_list_dict_path, suffix])

    if not qu.batchjob_enabled():
        sm.start_multiprocess_imap(_apply_merge_list_thread, multi_params)

    else:
        qu.batchjob_script(
            multi_params, "apply_merge_list", suffix=filename, n_cores=nb_cpus,
            remove_jobfolder=True)


def _apply_merge_list_thread(args):
    chunks = args[0]
    filename = args[1]
    hdf5names = args[2]
    merge_list_dict_path = args[3]
    postfix = args[4]

    merge_list_dict = pkl.load(open(merge_list_dict_path, 'rb'))
    for chunk in chunks:
        cc_data_list = compression.load_from_h5py(
            chunk.folder + filename + "_unique_components%s.h5" % postfix, hdf5names)
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
            cc_data_list[nb_hdf5_name] = this_cc

        compression.save_to_h5py(cc_data_list,
                                 chunk.folder + filename +
                                 "_stitched_components%s.h5" % postfix,
                                 hdf5names)


def export_cset_to_kd_batchjob(target_kd_paths, cset, name, hdf5names, n_cores=1,
                               offset=None, size=None, stride=(4 * 128, 4 * 128, 4 * 128),
                               overwrite=False, as_raw=False, fast_downsampling=False,
                               n_max_job=None, unified_labels=False, orig_dtype=np.uint8, log=None,
                               compresslevel=None):
    """
    Batchjob version of :class:`knossos_utils.chunky.ChunkDataset.export_cset_to_kd`
    method, see ``knossos_utils.chunky`` for details.

    Notes:
        * KnossosDataset needs to be initialized beforehand (see
          :func:`~KnossosDataset.initialize_without_conf`).
        * Only works if data mag = 1.

    Args:
        target_kd_paths: Target KnossosDatasets.
        cset: Source ChunkDataset.
        name:
        hdf5names:
        n_cores:
        offset:
        size:
        stride:
        overwrite:
        as_raw:
        fast_downsampling:
        n_max_job:
        unified_labels:
        orig_dtype:
        log:
        compresslevel: Compression level in case segmentation data is written for (seg.sz.zip files).

    Returns:

    """
    if n_max_job is None:
        n_max_job = global_params.config.ncore_total

    target_kds = {}
    for hdf5name in hdf5names:
        path = target_kd_paths[hdf5name]
        target_kd = kd_factory(path)
        target_kds[hdf5name] = target_kd

    for hdf5name in hdf5names[1:]:
        assert np.all(target_kds[hdf5names[0]].boundary ==
                      target_kds[hdf5name].boundary), \
            "KnossosDataset boundaries differ."

    if offset is None or size is None:
        offset = np.zeros(3, dtype=np.int32)
        # use any KD to infere the boundary
        size = np.copy(target_kds[hdf5names[0]].boundary)

    multi_params = []
    for coordx in range(offset[0], offset[0] + size[0],
                        stride[0]):
        for coordy in range(offset[1], offset[1] + size[1],
                            stride[1]):
            for coordz in range(offset[2], offset[2] + size[2],
                                stride[2]):
                coords = np.array([coordx, coordy, coordz])
                multi_params.append(coords)
    multi_params = basics.chunkify(multi_params, n_max_job)
    multi_params = [[coords, stride, cset.path_head_folder, target_kd_paths, name,
                     hdf5names, as_raw, unified_labels, 1, orig_dtype,
                     fast_downsampling, overwrite,
                     compresslevel] for coords in multi_params]

    job_suffix = "_" + "_".join(hdf5names)
    qu.batchjob_script(
        multi_params, "export_cset_to_kds", n_cores=n_cores, remove_jobfolder=True,
        suffix=job_suffix, log=log)


def _export_cset_as_kds_thread(args):
    """Helper function.
    TODO: refactor.
    """
    coords = args[0]
    size = np.array(args[1])
    cset_path = args[2]
    target_kd_paths = args[3]
    name = args[4]
    hdf5names = args[5]
    as_raw = args[6]
    unified_labels = args[7]
    nb_threads = args[8]
    orig_dtype = args[9]
    fast_downsampling = args[10]
    overwrite = args[11]
    compresslevel = args[12]

    cset = chunky.load_dataset(cset_path, update_paths=True)

    # Backwards compatibility
    if type(target_kd_paths) is str and len(hdf5names) == 1:
        kd = kd_factory(target_kd_paths)
        target_kds = {hdf5names[0]: kd}
    else:  # Add proper case handling for incorrect arguments
        target_kds = {}
        for hdf5name in hdf5names:
            path = target_kd_paths[hdf5name]
            target_kd = kd_factory(path)
            target_kds[hdf5name] = target_kd

    for dim in range(3):
        if coords[dim] + size[dim] > cset.box_size[dim]:
            size[dim] = cset.box_size[dim] - coords[dim]

    data_dict = cset.from_chunky_to_matrix(size, coords, name, hdf5names,
                                           dtype=orig_dtype)
    for hdf5name in hdf5names:
        curr_d = data_dict[hdf5name]
        if (curr_d.dtype.kind not in ("u", "i")) and (0 < np.max(curr_d) <= 1.0):
            curr_d = (curr_d * 255).astype(np.uint8)
        data_dict[hdf5name] = []
        data_list = curr_d
        # make it ZYX
        data_list = np.swapaxes(data_list, 0, 2)
        kd = target_kds[hdf5name]
        if as_raw:
            kd.save_raw(offset=coords, mags=kd.available_mags, data=data_list, data_mag=1,
                        fast_resampling=fast_downsampling)
        else:
            kd.save_seg(offset=coords, mags=kd.available_mags, data=data_list, data_mag=1,
                        fast_resampling=fast_downsampling, compresslevel=compresslevel)
