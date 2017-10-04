# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max-Planck-Institute for Medical Research, Heidelberg, Germany
# Authors: Sven Dorkenwald, Philipp Schubert, Joergen Kornfeld

from ..multi_proc import multi_proc_main as mpm

from knossos_utils import chunky, knossosdataset
import numpy as np
import scipy.misc
import time


def export_dense_segmentation_to_cset_thread(args):
    chunk_keys = args[0]
    cset = chunky.load_dataset(args[1])
    kd = knossosdataset.KnossosDataset()
    kd.initialize_from_knossos_path(args[2])
    datatype = args[3]

    for k_chunk in chunk_keys:
        time_start = time.time()
        chunk = cset.chunk_dict[k_chunk]
        print chunk.coordinates
        segmentation_data = kd.from_overlaycubes_to_matrix(chunk.size,
                                                           chunk.coordinates,
                                                           datatype=np.uint32,
                                                           mag=1,
                                                           mirror_oob=False,
                                                           verbose=False,
                                                           nb_threads=40)
        cset.from_matrix_to_chunky(chunk.coordinates, np.zeros(3, dtype=np.int),
                                   segmentation_data, "dense_segmentation",
                                   "sv", datatype=datatype, n_threads=1)
        print "Took %.3f" % (time.time() - time_start)


def export_dense_segmentation_to_cset(cset, kd, datatype=None, batch_size=None,
                                      nb_cpus=1, queue=None, pe=None):
    if batch_size is None:
        batch_size = int(len(cset.chunk_dict.keys()) / 3. / nb_cpus)

    multi_params = []
    c_keys = cset.chunk_dict.keys()

    for i in range(0, len(c_keys), batch_size):
        multi_params.append([c_keys[i:i + batch_size], cset.path_head_folder,
                             kd.conf_path, datatype])

    if (queue or pe) and mpm.__QSUB__:
        mpm.QSUB_script(multi_params, "export_dense_segmentation_to_cset",
                        queue=queue, pe=pe, n_cores=nb_cpus)
    else:
        mpm.start_multiprocess(export_dense_segmentation_to_cset_thread,
                               multi_params, nb_cpus=nb_cpus, debug=False)


def extract_knossosdataset_slice_thread(args):
    kd_path = args[0]
    z = args[1]
    save_path = args[2]
    use_raw = args[3]
    datatype = args[4]

    kd = knossosdataset.KnossosDataset()
    kd.initialize_from_knossos_path(kd_path)

    if use_raw:
        img_slice = kd.from_raw_cubes_to_matrix(
            [kd.boundary[0], kd.boundary[1], 1], [0, 0, z], nb_threads=10)
    else:
        img_slice = kd.from_overlaycubes_to_matrix(
            [kd.boundary[0], kd.boundary[1], 1], [0, 0, z],
            datatype=datatype, nb_threads=10)

    print("Saving")
    scipy.misc.imsave(save_path, img_slice[..., 0])


def extract_knossosdataset_slices(kd_path, save_folder, use_raw=False,
                                  datatype=np.uint32, nb_cpus=1, queue=None,
                                  pe=None):
    kd = knossosdataset.KnossosDataset()
    kd.initialize_from_knossos_path(kd_path)

    multi_params = []

    for z in range(60, kd.boundary[2], 128):
        multi_params.append([kd_path, z, save_folder + "/z_%d.png" % z,
                             use_raw, datatype])

    if (queue or pe) and mpm.__QSUB__:
        mpm.QSUB_script(multi_params, "extract_knossosdataset_slice",
                        queue=queue, pe=pe, n_cores=nb_cpus)
    else:
        mpm.start_multiprocess(extract_knossosdataset_slice_thread,
                               multi_params, nb_cpus=nb_cpus, debug=False)


def extract_chunky_slice_thread(args):
    chunk_dataset_path = args[0]
    z = args[1]
    save_path = args[2]
    filename = args[3]
    hdf5names = args[4]
    datatype = args[5]

    chunk_dataset = chunky.load_dataset(chunk_dataset_path, True)

    img_slice = chunk_dataset.from_chunky_to_matrix([chunk_dataset.box_size[0],
                                                    chunk_dataset.box_size[1], 1],
                                                    [0, 0, z], filename, hdf5names,
                                                    dtype=datatype)

    for hdf5_name in hdf5names:
        scipy.misc.imsave(save_path % hdf5_name, img_slice[hdf5_name][..., 0])


def extract_chunky_slice(chunk_dataset_path, save_folder, filename, hdf5names,
                         datatype=np.uint32, nb_cpus=1, queue=None, pe=None):
    chunk_dataset = chunky.load_dataset(chunk_dataset_path, True)

    multi_params = []

    for z in range(60, chunk_dataset.box_size[2], chunk_dataset.chunk_size[2]):
        multi_params.append([chunk_dataset_path, z,
                             save_folder + "/%s_z_" + "%d.png" % z,
                             filename, hdf5names, datatype])

    if (queue or pe) and mpm.__QSUB__:
        mpm.QSUB_script(multi_params, "extract_chunky_slice",
                        queue=queue, pe=pe, n_cores=nb_cpus)
    else:
        mpm.start_multiprocess(extract_chunky_slice_thread,
                               multi_params, nb_cpus=nb_cpus, debug=False)

