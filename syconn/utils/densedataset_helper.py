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
                                   "sv", datatype=datatype, n_threads=40)
        print "Took %.3f" % (time.time() - time_start)
