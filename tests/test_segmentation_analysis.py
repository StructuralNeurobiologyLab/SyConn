# SyConn
# Copyright (c) 2016 Philipp J. Schubert
# All rights reserved

from syconn.reps.rep_helper import find_object_properties
from syconn.extraction.cs_extraction_steps import detect_cs
import numpy as np
from syconn.global_params import config
import scipy.ndimage
import networkx as nx
from syconn.handler.basics import chunkify_weighted

def test_find_object_properties():
    sample_array = np.array([
            [[0, 1],
             [1, 1]],
            [[5, 2],
             [2, 1]]], np.uint64)
    # func_output[0]=dictionary of id's of unique voxel id, voxel id as key
    # func_output[1]=dictionary of bounding box for voxel ids, voxel id as key
    # func_output[2]=dictionary of count of all voxel ids, voxel id as key
    repcoord_dc, bb_dc, cnt_dc = find_object_properties(sample_array)
    element, count = np.unique(sample_array, return_counts=True)
    assert 0 not in repcoord_dc and 0 not in bb_dc and 0 not in cnt_dc, \
        "Background properties must not be extracted."
    if 0 in element:
        count = count[element != 0]
        element = element[element != 0]
    for i in range(len(element)):
        # testing count of voxel ids
        assert (cnt_dc[element[i]] == count[i].astype(np.uint64)), \
            "Count of the voxels not working."
        # testing unique voxel-id
        ll = repcoord_dc[element[i]]
        check = sample_array[ll[0], ll[1], ll[2]]
        assert element[i].astype(np.uint64) == check.astype(np.uint64), \
            "Object voxel dictionary dosen't match."
    # testing bounding box
    for e in element:
        mask = sample_array == e
        min_bound = np.transpose(np.where(mask)).min(axis=0)
        assert np.all(min_bound == bb_dc[e][0]), \
            "Bounding box dictionary mismatch."
        max_bound = np.transpose(np.where(mask)).max(axis=0) + 1
        assert np.all(max_bound == bb_dc[e][1]), \
            "Bounding box dictionary mismatch."


def test_detect_cs(distance_between_cube, stencil, cube_size):
    """

    Args:
        distance_between_cube: Distance between cubes of two different ids
        stencil: Generic stencil size
        cube_size: Generic cube size of two different ids

    Returns: Assert statement fails if detect_cs() method does not work properly

    """
    assert (np.amax(distance_between_cube) > cube_size), "Distance between cubes should be grater than cube size"
    stencil = stencil
    cube_size = cube_size                                                                #cube size
    distance_between_cube = distance_between_cube                                        #distance between cube
    offset = stencil // 2                                                                #output offset adjustment due to stencil size
    a = np.amax(offset + 1)                                                              #co-ordinate of topmost corner of first cube
    edge_s = np.amax(stencil + distance_between_cube + cube_size)  # data cube size
    sample = np.zeros((edge_s, edge_s, edge_s), dtype=np.uint32)
    c = cube_size
    d = distance_between_cube                                                            #dummy variable
    sample[a:a+c, a:a+c, a:a+c] = 4                                                      #cell_id cube 1
    sample[a+d[0]:a+d[0]+c, a+d[1]:a+d[1]+c, a+d[2]:a+d[2]+c] = 5                        #cell_id cube 2
    edge_id_output_sample = detect_cs(sample)
    higher_id_array = np.asarray(edge_id_output_sample, np.uint32)                       #retracts 32 bit cell id of higher value
    lower_id_array = np.asarray(edge_id_output_sample, np.uint64) // (2 ** 32)           #retracts 32 bit cell id of lower value
    counter = d - offset                                                                 #checks if distance between cubes is longer than stencil size
    output_offset = np.maximum(0, counter)                                               #adjusts output offset accordingly
    output_shape = np.array(sample.shape + np.array([1, 1, 1]) - stencil)
    o_o = output_offset                                                                  #dummy variable for output cube size
    o = offset                                                                           #dummy variable for offset due to stencil size

    output_id = np.zeros((output_shape[0], output_shape[1], output_shape[2]), dtype=np.uint32)

    output_id[a-o[0]+o_o[0]:a+c-o[0], a-o[1]+o_o[1]:a+c-o[1], a-o[2]+o_o[2]:a+c-o[2]] = 1
    output_id[a+d[0]-o[0]:a+d[0]+c-o[0]-o_o[0], a+d[1]-o[1]:a+d[1]+c-o[1]-o_o[1], a+d[2]-o[2]:a+d[2]+c-o[2]-o_o[2]] = 1
    output_id[a-o[0]+1:a+c-o[0]-1, a-o[1]+1:a+c-o[1]-1, a-o[2]+1:a+c-o[2]-1] = 0
    output_id[a+d[0]-o[0]+1:a+d[0]+c-o[0]-1, a+d[1]-o[1]+1:a+d[1]+c-o[1]-1, a+d[2]-o[2]+1:a+d[2]+c-o[2]-1] = 0

    assert np.array_equal(np.array(5*output_id, np.uint32), np.array(higher_id_array, np.uint32)), "higher value cell id array do not match"
    assert np.array_equal(np.array(4*output_id, np.uint32), np.array(lower_id_array, np.uint32)), "lower value cell id array do not match"


def test_config():
    print("None")


def test_chunk_weighted():
    sample_array = np.array([0, 1, 2, 3, 4, 5, 6, 7], np.uint64)
    weights = np.array([3,1,2,7,5,8,0,], np.uint64)
    n = 3                      # number_of_sublists
    output_array = chunkify_weighted(sample_array, n, weights)
    priority = np.argsort(weights)[::-1]
    # print(sample_array[priority[0::n]])
    print(priority)
    for i in range(n):
        assert np.array_equal(np.array(output_array[i], np.uint64), np.array(sample_array[priority[i::n]], np.uint64)), "chunk_weighter() function might have some problem "


if __name__ == '__main__':
    test_chunk_weighted()
    # test_detect_cs(np.array([0, 6, 0]), np.array(config['cell_objects']['cs_filtersize'], dtype=np.int), 5)
    # test_detect_cs(np.array([6, 0, 0]), np.array(config['cell_objects']['cs_filtersize'], dtype=np.int), 5)
    # test_detect_cs(np.array([0, 0, 6]), np.array(config['cell_objects']['cs_filtersize'], dtype=np.int), 5)
