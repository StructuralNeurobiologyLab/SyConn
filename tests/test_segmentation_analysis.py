# SyConn
# Copyright (c) 2016 Philipp J. Schubert
# All rights reserved

from syconn.reps.rep_helper import find_object_properties, find_object_properties_old
from syconn.extraction.cs_extraction_steps import detect_cs
import numpy as np
import time
from syconn.global_params import config
from syconn.handler.basics import chunkify_weighted
from syconn.reps.rep_helper import colorcode_vertices
from scipy import spatial


def test_find_object_properties():
    n = 20
    sample_array = np.zeros(shape=(n, n, n, 2), dtype=np.uint32)
    std_key = b'0000000100000001'
    zero_key = b'0000000000000000'

    repcoord_dc, bb_dc, cnt_dc = find_object_properties(sample_array)

    assert repcoord_dc == {} and bb_dc == {} and cnt_dc == {}, "Must be empty dicts"
    assert zero_key not in repcoord_dc and zero_key not in bb_dc and zero_key not in cnt_dc, \
        "Background properties must not be extracted."

    for ix in range(n):
        for iy in range(n):
            for iz in range(n):
                sample_array[ix, iy, iz] = (1, 1)

    repcoord_dc, bb_dc, cnt_dc = find_object_properties(sample_array)

    assert repcoord_dc == {std_key: [0, 0, 0]}, "Only one ID"
    assert bb_dc == {std_key: [[0, 0, 0], [n, n, n]]}, "Whole chunk is one bounding box"
    assert cnt_dc == {std_key: sample_array.shape[0]*sample_array.shape[1]*sample_array.shape[2]}, \
        "Size must be shape_x*shape_y*shape_z"

    keys_array = np.zeros(shape=(n, n, n), dtype=object)
    for ix in range(n):
        for iy in range(n):
            for iz in range(n):
                sample_array[ix, iy, iz] = (ix, ix + iy + iz)
                keys_array[ix, iy, iz] = bytes(str(ix).zfill(8) + str(ix + iy + iz).zfill(8), 'utf-8)')

    repcoord_dc, bb_dc, cnt_dc = find_object_properties(sample_array)

    assert zero_key not in repcoord_dc and zero_key not in bb_dc and zero_key not in cnt_dc, \
        "Background properties must not be extracted."

    unique_keys, count = np.unique(keys_array[:, :, :], return_counts=True)

    for i, key in enumerate(unique_keys):
        # If key starts with a zero, discard it (runs in real conditions are subject to this anyway
        if key[:8] == b'00000000':
            continue

        # testing count of voxel ids
        assert (cnt_dc[key] == count[i].astype(np.uint64)), "Count of the voxels not working."
        # testing unique voxel-id
        rep_coord = repcoord_dc[key]
        check = sample_array[rep_coord[0], rep_coord[1], rep_coord[2]]
        rhs = bytes(str(check[0]).zfill(8) + str(check[1]).zfill(8), 'utf-8')
        assert key == rhs, f"Object voxel dictionary dosen't match: {key} != {rhs}"

        # testing bounding box
        ix = int(key[:8])
        sigma = int(key[8:]) - ix  # Sigma = iy + iz
        k, small, large = (0, 0, 0)
        found_k = False
        while not found_k and k < n:
            try_bb = [[ix, k, k], [ix + 1, sigma - k + 1, sigma - k + 1]]
            fallback_bb = [[ix, sigma - k, sigma - k], [ix + 1, k + 1, k + 1]]

            try1 = (try_bb == bb_dc[key])
            try2 = (fallback_bb == bb_dc[key])

            if try1 or try2:
                if try1:
                    small = k
                    large = sigma - k
                if try2:
                    small = sigma - k
                    large = k
                found_k = True
            else:
                k += 1

        assert bb_dc[key] == [[ix, small, small], [ix + 1, large + 1, large + 1]]


def test_performance_find_object_properties():

    iterations = 100
    chunk1 = (50, 50, 10)
    chunk2 = (50, 50, 10, 2)

    # Structures
    old_version = []
    new_version = []

    for ii in range(iterations):
        old_version.append(np.random.randint(low=0, high=2 ** 32 - 1, size=chunk1, dtype=np.uint32))
        new_version_to_add = np.random.randint(low=0, high=2 ** 32 - 1, size=chunk2, dtype=np.uint32)

        for x in range(new_version_to_add.shape[0]):
            for y in range(new_version_to_add.shape[1]):
                for z in range(new_version_to_add.shape[2]):
                    if new_version_to_add[x, y, z, 0] > new_version_to_add[x, y, z, 1]:
                        tmp = new_version_to_add[x, y, z, 0]
                        new_version_to_add[x, y, z, 0] = new_version_to_add[x, y, z, 1]
                        new_version_to_add[x, y, z, 1] = tmp

        new_version.append(new_version_to_add)

    print("Begin of actual test is now")

    tic_old = time.perf_counter()
    for ii in range(iterations):
        _, _, _ = find_object_properties_old(old_version[ii])
    toc_old = time.perf_counter()

    tic_new = time.perf_counter()
    for ii in range(iterations):
        _, _, _, = find_object_properties(new_version[ii])
    toc_new = time.perf_counter()

    print(f"For {iterations} iterations, old: {toc_old - tic_old}s and new: {toc_new - tic_new}s")


def _helpertest_detect_cs(distance_between_cube, stencil, cube_size):
    """
    Assert statement fails if detect_cs() method does not work properly

    Args:
        distance_between_cube: Distance between cubes of two different ids
        stencil: Generic stencil size
        cube_size: Generic cube size of two different ids

    Returns:

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

    assert np.array_equal(np.array(5*output_id, np.uint32), np.array(higher_id_array, np.uint32)), \
        "higher value cell id array do not match"
    assert np.array_equal(np.array(4*output_id, np.uint32), np.array(lower_id_array, np.uint32)), \
        "lower value cell id array do not match"


def test_detect_cs():
    _helpertest_detect_cs(np.array([0, 6, 0]), np.array(config['cell_objects']['cs_filtersize'], dtype=np.int), 5)
    _helpertest_detect_cs(np.array([6, 0, 0]), np.array(config['cell_objects']['cs_filtersize'], dtype=np.int), 5)
    _helpertest_detect_cs(np.array([0, 0, 6]), np.array(config['cell_objects']['cs_filtersize'], dtype=np.int), 5)


def test_chunk_weighted():
    sample_array = np.array([0, 1, 2, 3, 4, 5, 6, 7], np.uint64)
    weights = np.array([3, 1, 2, 7, 5, 8, 0, 8], np.uint64)
    n = 3                      # number_of_sublists
    output_array = chunkify_weighted(sample_array, n, weights)
    priority = np.argsort(weights)[::-1]
    for i in range(n):
        assert np.array_equal(np.array(output_array[i], np.uint64), np.array(sample_array[priority[i::n]], np.uint64)),\
            "chunk_weighted() function might have some problem "


def test_colorcode_vertices(grid_size=5, number_of_test_vertices=50):
    """
    Test case fails if colourcode_vertices() is not working
    Args:
        grid_size: basis points for k-d tree
        number_of_test_vertices: Number of vertices to be tested for colorcode_verices()

    Returns:

    """
    n = number_of_test_vertices
    a = grid_size
    rep_values = np.arange(a*a*a)                                    #id values of grid vertices, 125 is number of rep_coords
    rep_coords = np.mgrid[0:a, 0:a, 0:a]                             #grid vertices for k-d tree, 5 is generic length os grid rep_coords
    rep_coords = rep_coords.reshape(3, -1).T                         #grid vertices for k-d tree
    vertices = 5*np.random.rand(n, 3)                                #vertices to be fited to k-d tree, 50 is number of vertices to be tested for
    colors = np.c_[rep_coords, np.ones(a*a*a)]                       #colours of grid vertices of k-d tree
    hull_tree = spatial.cKDTree(rep_coords)                          #processing to get nearest neighbour on grid
    dists, ixs = hull_tree.query(vertices)                           #processing to get nearest neighbour on grid
    output = colorcode_vertices(vertices, rep_coords, rep_values, colors=colors, return_color=False)     #output from colorcode_vertices()
    # vertices_index = [31, 93, 51, 27, 0]
    assert np.array_equal(output, ixs), \
        "colorcode_vertices() function might have some problem"                      #check 1
    output1 = colorcode_vertices(vertices, rep_coords, rep_values, colors=colors, return_color=True)     #output from colorcode_vertices() with color enabled
    assert np.array_equal(output1, colors[ixs]), \
        "colorcode_vertices() function might have some problem with input colors"    #check 2


if __name__ == '__main__':
    # test_chunk_weighted()
    # test_colorcode_vertices(5, 50)
    # test_detect_cs()
    test_find_object_properties()

    print("All module tests passed!")

    test_performance_find_object_properties()

    print("Performance test finished.")