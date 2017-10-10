# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max-Planck-Institute for Medical Research, Heidelberg, Germany
# Authors: Sven Dorkenwald, Philipp Schubert, Jörgen Kornfeld

from knossos_utils import chunky
import numpy as np


def initialize_cset(kd, path, chunksize):
    """
    Initializes a ChunkDataset

    Parameters
    ----------
    kd: KnossosDataset
        KnossosDataset instance of the corresponding raw data
    path: str
        path to main folder
    chunksize: np.array
        size of each chunk; typically in the order of ~ [1000, 1000, 500]

    Returns
    -------
    cset: ChunkDataset

    """

    cset = chunky.ChunkDataset()
    cset.initialize(kd, kd.boundary.copy(), chunksize, path + "/",
                    box_coords=[0, 0, 0], fit_box_size=True)
    chunky.save_dataset(cset)

    return cset


