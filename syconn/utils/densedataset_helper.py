# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max-Planck-Institute for Medical Research, Heidelberg, Germany
# Authors: Sven Dorkenwald, Philipp Schubert, Joergen Kornfeld


from knossos_utils import chunky, knossosdataset
import numpy as np
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
                                                           mag=1,
                                                           mirror_oob=False,
                                                           verbose=False,
                                                           nb_threads=40)
        cset.from_matrix_to_chunky(chunk.coordinates, np.zeros(3, dtype=np.int),
                                   segmentation_data, "dense_segmentation",
                                   "sv", datatype=datatype, n_threads=1)
        print "Took %.3f" % (time.time() - time_start)


def export_dense_segmentation_to_cset(cset, kd, datatype=None, batch_size=None,
                                      nb_cpus=16, queue=None, pe=None):
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