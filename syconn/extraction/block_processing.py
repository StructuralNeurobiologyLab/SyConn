# -*- coding: utf-8 -*-
# distutils: language=c++
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Philipp Schubert, Joergen Kornfeld

try:
    import cPickle as pkl
except ImportError:
    import pickle as pkl
import numpy as np


def kernel(chunk, center_id):
    """
    Notes: If there are more than on cell_ids corresponding to maximum count,
        minimum of those cell_id is selected. Resulting in irregular ids corresponding to edge

    Args:
        chunk: Takes array of stencil shape around a point corresponding to center cell_id
        center_id: Cell_id corresponding to center of stencil

    Returns: 64-bit cell id which is made up of cell_id corresponding to
            maximum count in stencil and center cell_id
    """
    unique_ids, counts = np.unique(chunk, return_counts=True)

    counts[unique_ids == 0] = -1
    counts[unique_ids == center_id] = -1

    if np.max(counts) > 0:
        partner_id = unique_ids[np.argmax(counts)]

        if center_id > partner_id:
            return (partner_id << 32) + center_id
        else:
            return (center_id << 32) + partner_id
    else:
        return 0


def process_block(edges, arr, stencil=(7, 7, 3)):
    stencil = np.array(stencil, dtype=np.int32)
    assert np.sum(stencil % 2) == 3

    out = np.zeros_like(arr, dtype=np.uint64)
    offset = stencil / 2
    for x in range(offset[0], arr.shape[0] - offset[0]):
        for y in range(offset[1], arr.shape[1] - offset[1]):
            for z in range(offset[2], arr.shape[2] - offset[2]):
                if edges[x, y, z] == 0:
                    continue

                center_id = arr[x, y, z]
                chunk = arr[x - offset[0]: x + offset[0] + 1,
                        y - offset[1]: y + offset[1],
                        z - offset[2]: z + offset[2]]
                out[x, y, z] = kernel(chunk, center_id)
    return out


def process_block_nonzero(edges, arr, stencil=(7, 7, 3)):
    """

    Args:
        edges: Convolution array for edge detection, standard scipy convolver
        arr: Takes a 3-d array of cell_ids(32-bit)
        stencil: Size of stencil for voxel processing

    Returns: Output 3-d array of 64-bit cell ids where edge is detected.
            64-bit ids has a 32-bit cell-id corresponding to maximum cell
            count and corresponding to center cent_id
    """
    stencil = np.array(stencil, dtype=np.int32)
    assert np.sum(stencil % 2) == 3

    arr_shape = np.array(arr.shape)
    out = np.zeros(arr_shape - stencil + 1, dtype=np.uint64)
    offset = stencil // 2  # int division!
    nze = np.nonzero(edges[offset[0]: -offset[0], offset[1]: -offset[1], offset[2]: -offset[2]])
    for x, y, z in zip(nze[0], nze[1], nze[2]):
        center_id = arr[x + offset[0], y + offset[1], z + offset[2]]
        chunk = arr[x: x + stencil[0], y: y + stencil[1], z: z + stencil[2]]
        out[x, y, z] = kernel(chunk, center_id)
    return out
