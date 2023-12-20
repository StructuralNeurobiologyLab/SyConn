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
    Returns a dense matrix representation of coordinates. The function creates a 3D matrix of zeros
    with dimensions equal to the cube of the edge size. It then sets the value of the matrix at the
    indices specified by the input array to 1.

    Args:
        sv (np.array): A 2D numpy array where each row represents the 3D coordinates of a point.
        edge_size (int): The edge size of the 3D matrix to be created.

    Returns:
        np.array: A 3D numpy array representing the dense matrix.
    """
    mat = np.zeros([edge_size] * 3, dtype=np.uint64)
    mat[sv[:, 0], sv[:, 1], sv[:, 2]] = 1
    return mat


def timeit(func):
    """
    A decorator function that measures the execution time of the decorated function. It prints the
    total execution time and the average time per sample.
    
    Args:
        func (function): The function to be decorated.
    
    Returns:
        function: The decorated function.
    """
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
    Cuts a numpy array along a specified dimension from a start index to an end index. The function
    handles the case where only the last elements should be retrieved.

    Args:
        array (np.array): The input numpy array to be cut.
        start (int): The start index for the cut.
        end (int): The end index for the cut.
        dim (int): The dimension along which the array is to be cut.

    Returns:
        np.array: The cut array.
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
    Crops a 3D boolean array to its True region. The function finds the indices of the True region
    and returns a cropped array along with the offset.

    Args:
        arr (np.array): A 3D boolean array to be cropped.

    Returns:
        Tuple[np.array, List[int]]: A tuple containing the cropped array and the offset.
    """
    in_mask_indices = [np.flatnonzero(arr.sum(axis=(1, 2))),
                       np.flatnonzero(arr.sum(axis=(0, 2))),
                       np.flatnonzero(arr.sum(axis=(0, 1)))]

    return arr[in_mask_indices[0].min(): in_mask_indices[0].max() + 1,
           in_mask_indices[1].min(): in_mask_indices[1].max() + 1,
           in_mask_indices[2].min(): in_mask_indices[2].max() + 1], \
           [in_mask_indices[0].min(),
            in_mask_indices[1].min(),
            in_mask_indices[2].min()]
