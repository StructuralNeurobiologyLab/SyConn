# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max-Planck-Institute for Medical Research, Heidelberg, Germany
# Authors: Sven Dorkenwald, Philipp Schubert, Joergen Kornfeld

import numpy as np

from ..multi_proc.multi_proc_main import start_multiprocess
import segmentationdataset
import time


def export_dense_segmentation_to_cset_thread(args):
    chunk_keys = args[0]
    cset = args[1]
    kd = args[2]

    for k_chunk in chunk_keys:
        time_start = time.time()
        chunk = cset.chunk_dict[k_chunk]
        print chunk.coordinates
        segmentation_data = kd.from_overlaycubes_to_matrix(chunk.size,
                                                           chunk.coordinates,
                                                           mag=1,
                                                           mirror_oob=False,
                                                           verbose=False)
        cset.from_matrix_to_chunky(chunk.coordinates, np.zeros(3),
                                   segmentation_data, "dense_segmentation", "sv",
                                   n_threads=1)
        print "Took %.3f" % (time.time() - time_start)


def export_dense_segmentation_to_cset(cset, kd, batch_size=None, nb_cpus=16):
    if batch_size is None:
        batch_size = int(len(cset.chunk_dict.keys()) / 3. / nb_cpus)

    multi_params = []
    c_keys = cset.chunk_dict.keys()

    for i in range(0, len(c_keys), batch_size):
        multi_params.append([c_keys[i:i + batch_size], cset, kd])

    start_multiprocess(export_dense_segmentation_to_cset_thread,
                       multi_params, nb_cpus=nb_cpus)


def apply_rallignment():
    pass




class DenseDataset():
    def __init__(self, path, path_segmentation):
        self._path = path
        self._supersupervoxels = {}


class SuperSuperVoxelObject():
    # mapping
    pass

class SuperVoxelObject(segmentationdataset.SegmentationObject):
    pass