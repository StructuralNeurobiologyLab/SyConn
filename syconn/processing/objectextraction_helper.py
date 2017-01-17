# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max-Planck-Institute for Medical Research, Heidelberg, Germany
# Authors: Sven Dorkenwald, Philipp Schubert, Jörgen Kornfeld

from ..utils import datahandler, basics, segmentationdataset
from knossos_utils import chunky, knossosdataset

import cPickle as pkl
import glob
import numpy as np
from scipy import ndimage


def gauss_threshold_connected_components_thread(args):
    chunk = args[0]
    path_head_folder = args[1]
    filename = args[2]
    hdf5names = args[3]
    overlap = args[4]
    sigmas = args[5]
    thresholds = args[6]
    swapdata = args[7]
    label_density = args[8]
    membrane_filename = args[9]
    membrane_kd_path = args[10]
    hdf5_name_membrane = args[11]
    fast_load = args[12]
    suffix = args[13]

    box_offset = np.array(chunk.coordinates) - np.array(overlap)
    size = np.array(chunk.size) + 2*np.array(overlap)

    if swapdata:
        size = basics.switch_array_entries(size, [0, 2])

    if not fast_load:
        cset = chunky.load_dataset(path_head_folder)
        bin_data_dict = cset.from_chunky_to_matrix(size, box_offset,
                                                   filename, hdf5names)
    else:
        bin_data_dict = datahandler.load_from_h5py(chunk.folder + filename + ".h5",
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
            tmp_data = \
                ndimage.gaussian_filter(tmp_data, sigmas[nb_hdf5_name])

        if hdf5_name in ["vc", "vc"] and membrane_filename is not None and \
                        hdf5_name_membrane is not None:
            membrane_data = datahandler.load_from_h5py(chunk.folder+membrane_filename+".h5",
                                                       hdf5_name_membrane)[0]
            membrane_data_shape = membrane_data.shape
            offset = (np.array(membrane_data_shape) - np.array(tmp_data.shape)) / 2
            membrane_data = membrane_data[offset[0]: membrane_data_shape[0]-offset[0],
                                          offset[1]: membrane_data_shape[1]-offset[1],
                                          offset[2]: membrane_data_shape[2]-offset[2]]
            tmp_data[membrane_data > 255*.4] = 0
        elif hdf5_name == ["vc", "vc"] and membrane_kd_path is not None:
            kd_bar = knossosdataset.KnossosDataset()
            kd_bar.initialize_from_knossos_path(membrane_kd_path)
            membrane_data = kd_bar.from_raw_cubes_to_matrix(size, box_offset)
            tmp_data[membrane_data > 255*.4] = 0

        if thresholds[nb_hdf5_name] != 0:
            tmp_data = np.array(tmp_data > thresholds[nb_hdf5_name],
                                dtype=np.uint8)

        this_labels_data, nb_cc = ndimage.label(tmp_data)
        nb_cc_list.append([chunk.number, hdf5_name, nb_cc])
        labels_data.append(this_labels_data)

    datahandler.save_to_h5py(labels_data,
                             chunk.folder + filename +
                             "_connected_components%s.h5" % suffix,
                             hdf5names)

    return nb_cc_list


def make_unique_labels_thread(args):
    chunk = args[0]
    filename = args[1]
    hdf5names = args[2]
    this_max_nb_dict = args[3]
    suffix = args[4]

    cc_data_list = datahandler.load_from_h5py(chunk.folder + filename +
                                              "_connected_components%s.h5"
                                              % suffix, hdf5names)

    for nb_hdf5_name in range(len(hdf5names)):
        hdf5_name = hdf5names[nb_hdf5_name]
        matrix = cc_data_list[nb_hdf5_name]
        matrix[matrix > 0] += this_max_nb_dict[hdf5_name]

    datahandler.save_to_h5py(cc_data_list,
                             chunk.folder + filename +
                             "_unique_components%s.h5"
                             % suffix, hdf5names)


def get_neighbouring_chunks(cset, chunk, chunk_list=None, diagonal=False):
    """ Returns the numbers of all neighbours

    Parameters:
    -----------
    cset: chunkDataset object
    chunk: chunk object

    Returns:
    --------
    list of int
        -1 if neighbour does not exist, else number
         order: [< x, < y, < z, > x, > y, > z]

    """
    coordinate = np.array(chunk.coordinates)
    neighbours = []
    for dim in range(3):
        try:
            this_coordinate = \
                coordinate - \
                datahandler.switch_array_entries(np.array([chunk.size[dim], 0, 0]),
                                        [0, dim])
            neighbour = cset.coord_dict[tuple(this_coordinate)]
            if not chunk_list is None:
                if neighbour in chunk_list:
                    neighbours.append(neighbour)
                else:
                    neighbours.append(-1)
            else:
                neighbours.append(neighbour)
        except:
            neighbours.append(-1)

    for dim in range(3):
        try:
            this_coordinate = \
                coordinate + \
                datahandler.switch_array_entries(np.array([chunk.size[dim], 0, 0]), [0, dim])
            neighbour = cset.coord_dict[tuple(this_coordinate)]
            if not chunk_list is None:
                if neighbour in chunk_list:
                    neighbours.append(neighbour)
                else:
                    neighbours.append(-1)
            else:
                neighbours.append(neighbour)
        except:
            neighbours.append(-1)
    return neighbours


def make_stitch_list_thread(args):
    cset = args[0]
    nb_chunk = args[1]
    filename = args[2]
    hdf5names = args[3]
    stitch_overlap = args[4]
    overlap = args[5]
    suffix = args[6]
    chunk_list = args[7]

    chunk = cset.chunk_dict[nb_chunk]
    cc_data_list = datahandler.load_from_h5py(chunk.folder + filename +
                                     "_unique_components%s.h5"
                                     % suffix, hdf5names)
    neighbours = get_neighbouring_chunks(cset, chunk, chunk_list)

    map_dict = {}
    for nb_hdf5_name in range(len(hdf5names)):
        map_dict[hdf5names[nb_hdf5_name]] = []

    for ii in range(3):
        if neighbours[ii] != -1:
            compare_chunk = cset.chunk_dict[neighbours[ii]]
            cc_data_list_to_compare = \
                datahandler.load_from_h5py(compare_chunk.folder + filename +
                                  "_unique_components%s.h5"
                                  % suffix, hdf5names)

            cc_area = {}
            cc_area_to_compare = {}
            if ii < 3:
                for nb_hdf5_name in range(len(hdf5names)):
                    this_cc_data = cc_data_list[nb_hdf5_name]
                    this_cc_data_to_compare = \
                        cc_data_list_to_compare[nb_hdf5_name]

                    cc_area[nb_hdf5_name] = \
                        datahandler.cut_array_in_one_dim(
                            this_cc_data,
                            overlap[ii] - stitch_overlap[ii],
                            overlap[ii] + stitch_overlap[ii], ii)

                    cc_area_to_compare[nb_hdf5_name] = \
                        datahandler.cut_array_in_one_dim(
                            this_cc_data_to_compare,
                            this_cc_data_to_compare.shape[ii] - overlap[ii] -
                            stitch_overlap[ii],
                            this_cc_data_to_compare.shape[ii] - overlap[ii] +
                            stitch_overlap[ii], ii)

            else:
                id = ii - 3
                for nb_hdf5_name in range(len(hdf5names)):
                    this_cc_data = cc_data_list[nb_hdf5_name]
                    this_cc_data_to_compare = \
                        cc_data_list_to_compare[nb_hdf5_name]

                    cc_area[nb_hdf5_name] = \
                        datahandler.cut_array_in_one_dim(
                            this_cc_data,
                            -overlap[id] - stitch_overlap[id],
                            -overlap[id] + stitch_overlap[id], id)

                    cc_area_to_compare[nb_hdf5_name] = \
                        datahandler.cut_array_in_one_dim(
                            this_cc_data_to_compare,
                            overlap[id] - stitch_overlap[id],
                            overlap[id] + stitch_overlap[id], id)

            this_shape = cc_area[0].shape
            for x in range(this_shape[0]):
                for y in range(this_shape[1]):
                    for z in range(this_shape[2]):
                        for nb_hdf5_name in range(len(hdf5names)):
                            hdf5_name = hdf5names[nb_hdf5_name]
                            this_id = cc_area[nb_hdf5_name][x, y, z]
                            compare_id = \
                                cc_area_to_compare[nb_hdf5_name][x, y, z]
                            if this_id != 0:
                                if compare_id != 0:
                                    try:
                                        if map_dict[hdf5_name][-1] != \
                                                (this_id, compare_id):
                                            map_dict[hdf5_name].append(
                                                (this_id, compare_id))
                                    except:
                                        map_dict[hdf5_name].append(
                                            (this_id, compare_id))
    return map_dict


def apply_merge_list_thread(args):
    chunk = args[0]
    filename = args[1]
    hdf5names = args[2]
    merge_list_dict_path = args[3]
    postfix = args[4]

    cc_data_list = datahandler.load_from_h5py(chunk.folder + filename +
                                              "_unique_components%s.h5"
                                              % postfix, hdf5names)

    merge_list_dict = pkl.load(open(merge_list_dict_path))

    for nb_hdf5_name in range(len(hdf5names)):
        hdf5_name = hdf5names[nb_hdf5_name]
        this_cc = cc_data_list[nb_hdf5_name]
        id_changer = merge_list_dict[hdf5_name]
        this_shape = this_cc.shape
        offset = (np.array(this_shape) - chunk.size) / 2
        this_cc = this_cc[offset[0]: this_shape[0] - offset[0],
                          offset[1]: this_shape[1] - offset[1],
                          offset[2]: this_shape[2] - offset[2]]
        this_cc = id_changer[this_cc]
        cc_data_list[nb_hdf5_name] = np.array(this_cc, dtype=np.uint32)

    datahandler.save_to_h5py(cc_data_list,
                             chunk.folder + filename +
                             "_stitched_components%s.h5" % postfix,
                             hdf5names)


def extract_voxels_thread(args):
    chunk = args[0]
    path_head_folder = args[1]
    filename = args[2]
    hdf5names = args[3]
    suffix = args[4]

    for nb_hdf5_name in range(len(hdf5names)):
        # print "Extracting"
        object_dataset = {}
        hdf5_name = hdf5names[nb_hdf5_name]
        this_segmentation = datahandler.load_from_h5py(chunk.folder + filename +
                                              "_stitched_components%s.h5" %
                                              suffix,
                                              [hdf5_name])[0]

        nonzero = np.nonzero(this_segmentation)

        for index in range(len(nonzero[0])):
            this_x = nonzero[0][index]
            this_y = nonzero[1][index]
            this_z = nonzero[2][index]
            value = this_segmentation[this_x, this_y, this_z]
            if value in object_dataset:
                object_dataset[value].append(
                    np.array([this_x, this_y, this_z]) + chunk.coordinates)
            else:
                object_dataset[value] = \
                    [np.array([this_x, this_y, this_z]) + chunk.coordinates]

        set_dict = {}
        map_dict = {}
        set_cnt = 0
        for id in object_dataset.keys():
            set_dict[str(id)] = object_dataset[id]
            map_dict[id] = [chunk.number, set_cnt]

            if len(set_dict) == 1000:
                np.savez_compressed(path_head_folder +
                                    segmentationdataset.get_rel_path(hdf5_name, filename, suffix=suffix) +
                                    "/voxels/%d_%d" % (chunk.number, set_cnt),
                                    **set_dict)
                set_dict = {}
                set_cnt += 1

        if len(set_dict) > 0:
            np.savez_compressed(path_head_folder + "/" +
                                segmentationdataset.get_rel_path(hdf5_name, filename, suffix=suffix) +
                                "/voxels/%d_%d" % (chunk.number, set_cnt),
                                **set_dict)

        f = open(path_head_folder + "/" +
                 segmentationdataset.get_rel_path(hdf5_name, filename, suffix=suffix) +
                 "/map_dicts/map_%d.pkl" % chunk.number, "w")
        pkl.dump(map_dict, f, pkl.HIGHEST_PROTOCOL)
        f.close()

        this_segmentation = []


def concatenate_mappings_thread(args):
    path_head_folder = args[0]
    rel_path = args[1]
    map_dict_paths = args[2]

    map_dict = {}
    for this_path in map_dict_paths:
        f = open(this_path, "r")
        this_map_dict = pkl.load(f)
        f.close()

        for this_key in this_map_dict.keys():
            if map_dict.has_key(this_key):
                map_dict[this_key].append(this_map_dict[this_key])
            else:
                map_dict[this_key] = [this_map_dict[this_key]]

    f = open(path_head_folder + rel_path + "/direct_map.pkl", "w")
    pkl.dump(map_dict, f, pkl.HIGHEST_PROTOCOL)
    f.close()


def create_objects_from_voxels_thread(args):
    path_head_folder = args[0]
    map_dict_paths = args[1]
    filename = args[2]
    hdf5_name = args[3]
    save_path = args[4]
    suffix = args[5]

    path_dataset = path_head_folder + \
                   segmentationdataset.get_rel_path(hdf5_name, filename, suffix)

    f = open(path_dataset + "/direct_map.pkl", "r")
    map_dict = pkl.load(f)
    f.close()

    object_dict = {}
    for map_dict_path in map_dict_paths:

        f = open(map_dict_path, "r")
        chunk_map_dict = pkl.load(f)
        f.close()

        if len(chunk_map_dict) > 0:
            for this_key in chunk_map_dict.keys():
                paths = []
                if chunk_map_dict[this_key] == map_dict[this_key][0]:
                    for dest in map_dict[this_key]:
                        paths.append(path_dataset + "/voxels/%d_%d.npz" %
                                     (dest[0], dest[1]))
                    object_dict[this_key] = segmentationdataset.SegmentationObject(this_key,
                                                                                   path_dataset,
                                                                                   paths)
                    object_dict[this_key].calculate_rep_coord(
                        calculate_size=True)

    f = open(save_path, "w")
    pkl.dump(object_dict, f, pkl.HIGHEST_PROTOCOL)
    f.close()


def create_datasets_from_objects_thread(args):
    path_head_folder = args[0]
    hdf5_name = args[1]
    filename = args[2]
    suffix = args[3]

    rel_path = segmentationdataset.get_rel_path(hdf5_name, filename, suffix)
    sset = segmentationdataset.SegmentationDataset(hdf5_name, rel_path,
                                                   path_head_folder)

    for obj_dict_path in glob.glob(path_head_folder + rel_path +
                                           "/object_dicts/*"):
        f = open(obj_dict_path, "r")
        this_obj_dict = pkl.load(f)
        f.close()

        sset.object_dict.update(this_obj_dict)

    segmentationdataset.save_dataset(sset)
