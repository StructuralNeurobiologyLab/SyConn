# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Sven Dorkenwald, Philipp Schubert, Joergen Kornfeld

import sys
import time
import numpy as np


def dense_matrix(sv, edge_size):
    """
    Get dense matrix representation of coordinates

    Parameters
    ----------
    sv : np.array
    edge_size : int

    Returns
    -------
    np.array

    """
    mat = np.zeros([edge_size] * 3, dtype=np.uint)
    mat[sv[:, 0], sv[:, 1], sv[:, 2]] = 1
    return mat


def timeit(func):
    def timeit_wrapper(*args, **kwargs):
        start = time.time()
        nb_samples = len(args[0])
        res = func(*args, **kwargs)
        end = time.time()
        sys.stdout.write("\r%0.2f\n" % 1.0)
        sys.stdout.flush()
        print("Prediction of %d samples took %.4gs; %.4gs/sample." % \
              (nb_samples, end - start, (end - start) / nb_samples))
        return res
    return timeit_wrapper


def cut_array_in_one_dim(array, start, end, dim):
    """
    Cuts an array along a dimension

    Parameters
    ----------
    array: np.array
    start: int
    end: int
    dim: int

    Returns
    -------
    array: np.array

    """
    start = int(start)
    end = int(end)
    if start < 0 and end == 0:  # handle case if only last elements should be retrieved, e.g.
        # -2:-0 which only works if second indexing is not used, i.e. [-2:]
        if dim == 0:
            array = array[start:, :, :]
        elif dim == 1:
            array = array[:, start:, :]
        elif dim == 2:
            array = array[:, :, start:]
        else:
            raise NotImplementedError()
    else:
        if dim == 0:
            array = array[start:end, :, :]
        elif dim == 1:
            array = array[:, start:end, :]
        elif dim == 2:
            array = array[:, :, start:end]
        else:
            raise NotImplementedError()
    return array


def crop_bool_array(arr):
    """
    Crops a bool array to its True region

    :param arr: 3d bool array
        array to crop
    :return: 3d bool array, list
        cropped array, offset
    """
    in_mask_indices = [np.flatnonzero(arr.sum(axis=(1, 2))),
                       np.flatnonzero(arr.sum(axis=(0, 2))),
                       np.flatnonzero(arr.sum(axis=(0, 1)))]

    return arr[in_mask_indices[0].min(): in_mask_indices[0].max() + 1,
               in_mask_indices[1].min(): in_mask_indices[1].max() + 1,
               in_mask_indices[2].min(): in_mask_indices[2].max() + 1],\
           [in_mask_indices[0].min(),
            in_mask_indices[1].min(),
            in_mask_indices[2].min()]