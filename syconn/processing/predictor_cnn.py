# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max-Planck-Institute for Medical Research, Heidelberg, Germany
# Authors: Sven Dorkenwald, Philipp Schubert, Jörgen Kornfeld

from elektronn.training import predictor, trainutils
from ..processing import initialization
from knossos_utils import knossosdataset
from knossos_utils import chunky
import glob
import h5py
import numpy as np
import os
import re
import sys
import time


def interpolate(data, mag=2):
    """
    Applies a naive interpolation to the data by replicating entries

    Parameters
    ----------
    data: np.array
    mag: int
        defines downsampling rate

    Returns
    -------
    new_data: np.array
        interpolated data
    """
    ds = data.shape
    new_data = np.zeros([ds[0], ds[1]*mag, ds[2]*mag, ds[3]*mag])
    for x in range(0, mag):
        for y in range(0, mag):
            for z in range(0, mag):
                new_data[:, x::mag, y::mag, z::mag] = data
    return new_data


def create_chunk_checklist(head_path, names):
    """
    Checks which chunks have already been already processed

    Parameters
    ----------
    head_path: str
        path to chunkdataset folder
    names: list
        list of hdf5names

    Returns
    -------
    checklist: np.array

    """
    folders_in_path = glob.glob(head_path+"/chunky_*")
    checklist = np.zeros((len(folders_in_path), len(names)), dtype=np.uint8)
    for folder in folders_in_path:
        if len(re.findall('[\d]+', folder)) > 0:
            chunk_nb = int(re.findall('[\d]+', folder)[-1])
            existing_files = glob.glob(folder+"/*.h5")
            for file in existing_files:
                for name_nb in range(len(names)):
                    if names[name_nb] in file:
                        checklist[chunk_nb, name_nb] = 1

    return checklist


def search_for_chunk(head_path, name, max_age_min=100):
    """
    Finds a chunk that has to be processed

    Parameters
    ----------
    head_path: str
        path to chunkdataset folder
    name: str
        hdf5name
    max_age_min: int
        maximum allowed age of a mutex in minutes

    Returns
    -------
    left_chunk: int
        chunk id of chunk that should be processed. Returns -1 if no chunk is
        left
    """
    folders_in_path = glob.glob(head_path + "/chunky_*")
    checklist = create_chunk_checklist(head_path, [name])
    left_chunks = []
    for chunk_nb in range(len(folders_in_path)):
        if checklist[chunk_nb] == 0:
            left_chunks.append(chunk_nb)
    if len(left_chunks) > 0:
        for ii in range(4):
            this_max_age_min = max_age_min/(ii+1)
            np.random.shuffle(left_chunks)
            for left_chunk in left_chunks:
                folder_path = head_path + "chunky_%d/mutex_%s" % (left_chunk, name)
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)
                    return left_chunk
                else:
                    if time.time()-os.stat(folder_path).st_mtime > this_max_age_min*60:
                        os.rmdir(folder_path)
                        os.makedirs(folder_path)
                        return left_chunk
    return -1


def create_recursive_data(labels, labels_data=None, labels_path="",
                          raw_path="", raw_data=None, use_labels=[]):
    """
    Creates input data for the recursive CNN

    Parameters
    ----------
    labels: list
        hdf5names
    labels_data: np.array
    labels_path: str
    raw_path: str
    raw_data: np.array
    use_labels: list
        determines which labels are used

    Returns
    -------
    recursive_data: np.array
    """
    try:
        len(labels)
    except:
        raise Exception("labels has to be a list")

    if len(use_labels) == 0:
        use_labels = np.ones(len(labels))
    else:
        use_labels = np.array(use_labels)

    if labels_data is None and len(labels_path) == 0:
        raise Exception("No labels_data or labels_path given")

    labels_dict = {}
    new_labels = []
    if len(labels_path) > 0:
        f = h5py.File(labels_path)
        for nb_label in range(len(labels)):
            if use_labels[nb_label]:
                label = labels[nb_label]
                labels_dict[label] = f[label].value
                new_labels.append(label)
        f.close()
    else:
        for nb_label in range(len(labels)):
            if use_labels[nb_label]:
                label = labels[nb_label]
                labels_dict[label] = labels_data[nb_label]
                new_labels.append(label)

    labels = new_labels

    recursive_data = []
    if len(raw_path) > 1 or not raw_data is None:
        if raw_data is None:
            f = h5py.File(raw_path)
            raw_data = f["raw"].value
            f.close()
            print "Raw data is none"

        labels_shape = np.array(labels_dict[labels[0]].shape)
        raw_shape = np.array(raw_data.shape)
        offset = (raw_shape - labels_shape) / 2
        recursive_data.append(raw_data[offset[0]: raw_shape[0] - offset[0],
                              offset[1]: raw_shape[1] - offset[1],
                              offset[2]: raw_shape[2] - offset[2]])

    for nb_label in range(len(labels)):
        label = labels[nb_label]
        if not label == "none":
            print label
            recursive_data.append(labels_dict[label])

    print len(recursive_data)
    return np.array(recursive_data)


def join_chunky_inference_thread(args):
    kd_raw = knossosdataset.KnossosDataset()

    cset = args[0]
    config_path = args[1]
    param_path = args[2]
    names = args[3]
    labels = args[4]
    offset = args[5]
    batch_size = args[6]
    kd_raw.initialize_from_knossos_path(args[7])
    join_chunky_inference(cset, config_path, param_path, names, labels, offset,
                          batch_size, kd=kd_raw)


def join_chunky_inference(cset, config_path, param_path, names,
                          labels, offset, desired_input, gpu=None, MFP=True,
                          invert_data=False, kd=None, mag=1):
    """
    Main predictor function. Handles parallel inference with mutexes; can
    be called multiple times from different independent processes


    Parameters
    ----------
    cset: ChunkDataset
    config_path: str
        path to CNN config file
    param_path: str
        path to param file from CNN training
    names: list of str
        if len == 1 this is just the savename; if len==2 the first name is
        used for creating the recursive data; the second entry is the the
        savename
    labels: list of str
        hdf5names
    offset: np.array
        Defines the extra space around the chunk which should be used to make
        up for the CNN offset
    desired_input: np.array
        desired batch_size
    gpu: int
        gpu number
    MFP: boolean
        whether or not to use max fragment pooling (recommended)
    invert_data: boolean
        if True, data gets inverted before inference
    kd: KnossosDataset
        if None, the linked KnossosDataset from cset will be used
    mag: int
        on which magnite the inference should be carried out

    Returns
    -------
    nothing

    """

    sys.setrecursionlimit(10000)

    offset = np.array(offset)

    if kd is None:
        kd = cset.dataset

    print "Number of chunks:", len(cset.chunk_dict)

    if len(names) > 1:
        n_ch = len(labels)
    else:
        n_ch = 1

    cnn = None
    name = names[0]

    while True:
        try:
            print "Time per chunk: %.3f" % (time.time() - time_start)
        except:
            pass
        time_start = time.time()

        while True:
            nb_chunk = search_for_chunk(cset.path_head_folder, name)
            if nb_chunk == -1:
                break
            chunk = cset.chunk_dict[nb_chunk]
            if len(names) == 1:
                break
            if os.path.exists(chunk.folder + names[1] + ".h5"):
                break
            time.sleep(2)

        if nb_chunk != -1:
            if not os.path.exists(chunk.folder):
                os.makedirs(chunk.folder)

            if not cnn:
                if not gpu:
                    gpu = trainutils.get_free_gpu(wait=0)
                    while gpu == -1:
                        time.sleep(120)
                        gpu = trainutils.get_free_gpu(wait=0)
                    trainutils.initGPU(gpu)

                cnn = predictor.create_predncnn(config_path, n_ch, len(labels),
                                                gpu=gpu,
                                                imposed_input_size=desired_input,
                                                override_MFP_to_active=MFP,
                                                param_file=param_path)

            out_path = chunk.folder + name + ".h5"
            print "Processing Chunk: %d" % nb_chunk
            if len(names) == 1:
                raw_data = kd.from_raw_cubes_to_matrix(
                    (np.array(chunk.size) + 2 * offset) / mag,
                    (chunk.coordinates - offset) / mag,
                    mag=mag,
                    invert_data=invert_data,
                    mirror_oob=True)
                raw_data = raw_data[None, :, :, :]
            else:
                raw_data = kd.from_raw_cubes_to_matrix(
                    (np.array(chunk.size) + 2 * offset) / mag,
                    (chunk.coordinates - offset) / mag,
                    mag=mag,
                    invert_data=invert_data,
                    mirror_oob=True)
                time_rec = time.time()
                rec_labels = []
                for label in labels:
                    if label != "none":
                        rec_labels.append(label)
                raw_data = create_recursive_data(rec_labels,
                                                 labels_path=chunk.folder +
                                                             names[1] + ".h5",
                                                 raw_data=raw_data)
                print "Time for creating recursive data: %.3f" % (
                time.time() - time_rec)

            inference_data = cnn.predictDense(raw_data, as_uint8=True)

            if mag > 1:
                inference_data = interpolate(inference_data, mag=mag)

            f = h5py.File(out_path, "w")
            for ii in range(len(labels)):
                if not labels[ii] == "none":
                    f.create_dataset(labels[ii],
                                     data=inference_data[ii],
                                     compression="gzip")
            f.close()

            folder_path = chunk.folder + "/mutex_%s" % name
            try:
                os.rmdir(folder_path)
            except:
                pass
        else:
            break


def correct_padding_thread(args):
    chunk = args[0]
    filename = args[1]
    offset = args[2]

    data_dict = {}
    changed = False

    with h5py.File("%s%s.h5" % (chunk.folder, filename), "r") as f:
        for hdf5_name in f.keys():
            data = f[hdf5_name].value
            for dim in range(3):
                if data.shape[dim] != chunk.size[dim] + offset[dim] * 2:
                    changed = True
                    padding = np.zeros((3, 2), dtype=np.int)
                    if chunk.coordinates[dim] == 0:
                        padding[dim, 0] = chunk.size[dim] + offset[dim] * 2 - data.shape[dim]
                    else:
                        padding[dim, 1] = chunk.size[dim] + offset[dim] * 2 - data.shape[dim]
                    data = np.pad(data, padding, mode="constant",
                                  constant_values=0)
            data_dict[hdf5_name] = data

    if changed:
        os.rename("%s%s.h5" % (chunk.folder, filename),
                  "%s%s_broken.h5" % (chunk.folder, filename))

    with h5py.File("%s%s_corrected.h5" % (chunk.folder, filename), "w") as f:
        for hdf5_name in data_dict.keys():
            f[hdf5_name] = data_dict[hdf5_name]




