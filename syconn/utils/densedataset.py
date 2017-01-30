# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max-Planck-Institute for Medical Research, Heidelberg, Germany
# Authors: Sven Dorkenwald, Philipp Schubert, Joergen Kornfeld

import numpy as np

import densedataset_helper as ddh
from ..multi_proc import multi_proc_main as mpm
import segmentationdataset
import time


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
        mpm.start_multiprocess(ddh.export_dense_segmentation_to_cset_thread,
                               multi_params, nb_cpus=nb_cpus, debug=False)


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