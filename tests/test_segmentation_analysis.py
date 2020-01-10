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



def test_detect_cs(distance_between_cube, stencil, cube_size):
    """

    Args:
        distance_between_cube: Distance between cubes of two different ids
        stencil: Generic stencil size
        cube_size: Generic cube size of two different ids

    Returns: Assert statement fails if detect_cs() method does not work properly

    """
    stencil = stencil
    cube_size = cube_size                                                                #cube size
    distance_between_cube = distance_between_cube                                        #distance between cube
    offset = stencil // 2                                                                #output offset adjustment due to stencil size
    a = np.amax(offset + 1 )                                                                              #co-ordinate of topmost corner of first cube
    edge_s = np.amax(stencil + distance_between_cube + cube_size)  # data cube size
    sample = np.zeros((edge_s, edge_s, edge_s), dtype=np.uint32)
    c = cube_size
    d = distance_between_cube                                                            #dummy variable
    sample[a:a+c, a:a+c, a:a+c] = 4                                                      #cell_id cube 1
    sample[a+d[0]:a+d[0]+c, a+d[1]:a+d[1]+c, a+d[2]:a+d[2]+c] = 5                        #cell_id cube 2
    edge_id_output_sample = detect_cs(sample)
    higher_id_array = np.asarray(edge_id_output_sample, np.uint32)                       #retracts 32 bit cell id of higher value
    lower_id_array = np.asarray(edge_id_output_sample, np.uint64) // (2 ** 32)           #retracts 32 bit cell id of lower value
    counter = np.array([c, c, c]) + d + np.array([1, 1, 1])                              #checks if distance between cubes is longer than stencil size
    output_offset = np.maximum(0, counter - stencil)                                   #adjusts output offset accordingly
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


if __name__ == '__main__':
    test_detect_cs(np.array([4, 0, 0]), np.array(config['cell_objects']['cs_filtersize'], dtype=np.int), 3)
