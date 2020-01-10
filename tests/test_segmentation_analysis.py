# SyConn
# Copyright (c) 2016 Philipp J. Schubert
# All rights reserved

from syconn.reps.rep_helper import find_object_properties
from syconn.extraction.cs_extraction_steps import detect_cs
import numpy as np
from syconn.global_params import config
import scipy.ndimage
import networkx as nx


# def test_bfs_smoothing():
#     test_split_subcc()
#     test_create_graph_from_coords()
#     G = nx.Graph()
#     print("Nodes in G: ", G.nodes(data=True))
#     print("Edges in G: ", G.edges(data=True))
#     G.add_weighted_edges_from([(0, 1, 3.0), (1, 2, 7.5)])
#     print("Nodes in G: ", G.nodes(data=True))
#     print("Edges in G: ", G.edges(data=True))
#
# def test_split_subcc():
#     print("Hi")
#
#
# def test_create_graph_from_coords():
#     print("Hi")



# def test_find_object_properties():
#     sample_array = np.array([
#             [[0, 1, 9],
#              [1, 1, 9],
#              [0, 1, 9]],
#             [[5, 2, 4],
#              [2, 1, 3],
#              [0, 1, 9]],
#             [[9, 1, 9],
#              [2, 3, 4],
#              [0, 3, 9]]], np.uint64)
#     func_output = find_object_properties(sample_array)
#     #func_output[0]=dictionary of id's of unique voxel id, voxel id as key
#     #func_output[1]=dictionary of bounding box for voxel ids, voxel id as key
#     #func_output[2]=dictionary of count of all voxel ids, voxel id as key
#     element, count = np.unique(sample_array, return_counts=True)
#     for i in range(len(element)):
#         if element[i] != 0:
#             #testing count of voxel ids
#             assert (func_output[2][element[i]] == count[i].astype(np.uint64)), "Count of the voxels not working" \
#                                                                                " They should be same"
#             # testing unique voxel-id
#             ll = func_output[0][element[i]]
#             check = sample_array[ll[0], ll[1], ll[2]]
#             assert element[i].astype(np.uint64) == check.astype(np.uint64), "object voxel dictionary dosen't match" \
#                              "They should be same"
#     # testing bounding box
#     indices = sample_array.flatten()
#     min_bound = np.full(((int(element[len(element)-1]) + 1), 3), len(indices), dtype=np.uint64)
#     max_bound = np.full(((int(element[len(element)-1]) + 1), 3), 0, dtype=np.uint64)
#     for i in range(len(indices)):
#         if indices[i] != 0:
#             qq = indices[i]
#             index = list(np.unravel_index(i, sample_array.shape))
#             min_bound[qq] = np.minimum(np.array(min_bound[qq]), index)
#             max_bound[qq] = np.maximum(np.array(max_bound[qq]), index)
#     for i in element:
#         if i != 0:
#             assert (np.array_equal(np.array(func_output[1][i][0]), min_bound[i])), \
#                 "Bounding box dictionary mismatch." \
#                 " They should be same"
#             assert np.array_equal(np.array(func_output[1][i][1]), (max_bound[i] + np.ones((3,), dtype=np.uint64))), \
#                 "Bounding box dictionary mismatch." \
#                          " They should be same"
#     print(func_output)
#     print(min_bound)
#     print(max_bound)
#     print(element)
#     print(count)
#

def test_detect_cs(distance_between_cube, stencil, cube_size):
    """

    Returns:

    """
    edge_s = 20                                                                          #data cube size
    sample = np.zeros((edge_s, edge_s, edge_s), dtype=np.uint32)
    stencil = stencil
    cube_size = cube_size                                                                #cube size
    c = cube_size
    a = 7                                                                                #co-ordinate of topmost corner of first cube
    distance_between_cube = distance_between_cube                                        #distance between cube
    d = distance_between_cube                                                            #dummy variable
    sample[a:a+c, a:a+c, a:a+c] = 4                                                      #cell_id cube 1
    sample[a+d[0]:a+d[0]+c, a+d[1]:a+d[1]+c, a+d[2]:a+d[2]+c] = 5                        #cell_id cube 2
    offset = stencil // 2                                                                #output offset adjustment due to stencil size
    edge_id_output_sample = detect_cs(sample)
    higher_id_array = np.asarray(edge_id_output_sample, np.uint32)                       #retracts 32 bit cell id of higher value
    lower_id_array = np.asarray(edge_id_output_sample, np.uint64) // (2 ** 32)           #retracts 32 bit cell id of lower value
    counter = np.array([c, c, c]) + d + np.array([1, 1, 1])                              #checks if distance between cubes is longer than stencil size
    output_offset = np.maximum((0), counter - stencil)                                   #adjusts output offset accordingly
    output_shape = np.array(sample.shape + np.array([1, 1, 1]) - stencil)
    o_o = output_offset                                                                  #dummy variable for output cube size
    o = offset                                                                           #dummy variable for offset due to stencil size

    output_id = np.zeros((output_shape[0],output_shape[1],output_shape[2]), dtype=np.uint32)

    output_id[a-o[0]+o_o[0]:a+c-o[0], a-o[1]+o_o[1]:a+c-o[1], a-o[2]+o_o[2]:a+c-o[2]] = 1
    output_id[a+d[0]-o[0]:a+d[0]+c-o[0]-o_o[0], a+d[1]-o[1]:a+d[1]+c-o[1]-o_o[1], a+d[2]-o[2]:a+d[2]+c-o[2]-o_o[2]] = 1
    output_id[a-o[0]+1:a+c-o[0]-1, a-o[1]+1:a+c-o[1]-1, a-o[2]+1:a+c-o[2]-1] = 0
    output_id[a+d[0]-o[0]+1:a+d[0]+c-o[0]-1, a+d[1]-o[1]+1:a+d[1]+c-o[1]-1, a+d[2]-o[2]+1:a+d[2]+c-o[2]-1] = 0

    assert np.array_equal(np.array(5*output_id, np.uint32), np.array(higher_id_array,np.uint32)), "higher value cell id array do not match"
    assert np.array_equal(np.array(4*output_id, np.uint32), np.array(lower_id_array,np.uint32)), "lower value cell id array do not match"

    # print(5*output_id)
    # print(higher_id_array)
    # print(output_shape)
    # print(output_offset)
    # sample[7:10, 7:10, 7:10] = 4                                                         #cell_id cube 1
    # sample[7:10, 7:10, 11:14] = 5                                                        #cell_id cube 2

def test_process_block_nonzero():
    print("p")


if __name__ == '__main__':
    #test_find_object_properties()
    test_detect_cs(np.array([0, 4, 4]), np.array(config['cell_objects']['cs_filtersize'], dtype=np.int), 3)
    #test_bfs_smoothing()