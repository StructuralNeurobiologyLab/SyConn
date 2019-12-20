# SyConn
# Copyright (c) 2016 Philipp J. Schubert
# All rights reserved

from syconn.reps.rep_helper import find_object_properties
from syconn.extraction.cs_extraction_steps import detect_cs
import numpy as np
from syconn.global_params import config
import scipy.ndimage
import networkx as nx


def test_bfs_smoothing():
    test_split_subcc()
    test_create_graph_from_coords()
    G = nx.Graph()
    print("Nodes in G: ", G.nodes(data=True))
    print("Edges in G: ", G.edges(data=True))
    G.add_weighted_edges_from([(0, 1, 3.0), (1, 2, 7.5)])
    print("Nodes in G: ", G.nodes(data=True))
    print("Edges in G: ", G.edges(data=True))

def test_split_subcc():
    print("Hi")


def test_create_graph_from_coords():
    print("Hi")



def test_find_object_properties():
    sample_array = np.array([
            [[0, 1, 9],
             [1, 1, 9],
             [0, 1, 9]],
            [[5, 2, 4],
             [2, 1, 3],
             [0, 1, 9]],
            [[9, 1, 9],
             [2, 3, 4],
             [0, 3, 9]]], np.uint64)
    func_output = find_object_properties(sample_array)
    #func_output[0]=dictionary of id's of unique voxel id, voxel id as key
    #func_output[1]=dictionary of bounding box for voxel ids, voxel id as key
    #func_output[2]=dictionary of count of all voxel ids, voxel id as key
    element, count = np.unique(sample_array, return_counts=True)
    for i in range(len(element)):
        if element[i] != 0:
            #testing count of voxel ids
            assert (func_output[2][element[i]] == count[i].astype(np.uint64)), "Count of the voxels not working" \
                                                                               " They should be same"
            # testing unique voxel-id
            ll = func_output[0][element[i]]
            check = sample_array[ll[0], ll[1], ll[2]]
            assert element[i].astype(np.uint64) == check.astype(np.uint64), "object voxel dictionary dosen't match" \
                             "They should be same"
    # testing bounding box
    indices = sample_array.flatten()
    min_bound = np.full(((int(element[len(element)-1]) + 1), 3), len(indices), dtype=np.uint64)
    max_bound = np.full(((int(element[len(element)-1]) + 1), 3), 0, dtype=np.uint64)
    for i in range(len(indices)):
        if indices[i] != 0:
            qq = indices[i]
            index = list(np.unravel_index(i, sample_array.shape))
            min_bound[qq] = np.minimum(np.array(min_bound[qq]), index)
            max_bound[qq] = np.maximum(np.array(max_bound[qq]), index)
    for i in element:
        if i != 0:
            assert (np.array_equal(np.array(func_output[1][i][0]), min_bound[i])), \
                "Bounding box dictionary mismatch." \
                " They should be same"
            assert np.array_equal(np.array(func_output[1][i][1]), (max_bound[i] + np.ones((3,), dtype=np.uint64))), \
                "Bounding box dictionary mismatch." \
                         " They should be same"
    print(func_output)
    print(min_bound)
    print(max_bound)
    print(element)
    print(count)


def test_detect_cs():
    """

    Returns:

    """
    edge_s = 20
    sample_array = np.random.randint(low=0, size=edge_s ** 3, high=1000).reshape((
        edge_s, edge_s, edge_s)).astype(np.uint32)
    # weight = np.array([[[0, 0, 0],
    #                     [0, 1, 0],
    #                     [0, 0, 0]],
    #                    [[0, 1, 0],
    #                     [1, -6, 1],
    #                     [0, 1, 0]],
    #                    [[0, 0, 0],
    #                     [0, 1, 0],
    #                     [0, 0, 0]]], dtype=np.int)
    # sample = np.zeros((edge_s, edge_s, edge_s), dtype=np.uint32)
    # sample[8][8] = 4
    # sample[9][9] = 3
    # sample[10][10] = 5
    # sample[11][11] = 2
    # sample[12][12] = 6
    # aa1 = detect_cs(sample)
    # # offset = np.array(con['cell_objects']['cs_filtersize'], dtype=np.int) // 2
    # edge = scipy.ndimage.convolve(sample.astype(np.int), weight) < 0
    # edge_rel = np.nonzero(edge[offset[0]: -offset[0], offset[1]: -offset[1], offset[2]: -offset[2]])
    sample = np.zeros((edge_s, edge_s, edge_s), dtype=np.uint32)
    # sample[8][8] = 4
    sample[9][9] = 3
    sample[10][10] = 5
    sample[11][11] = 2
    sample[10, :, 12] = 7
    sample[:, 10, 5] = 4
    # sample[12][12] = 6
    # sample[10][10][10] = 7
    # sample[10][10][11] = 7
    # sample[10][10][9] = 7
    #sample[9][9][10] = 8
    offset = np.array(config['cell_objects']['cs_filtersize'], dtype=np.int) // 2
    #offset = stencil mod 2
    #stencil = global_params.config['cell_objects']['cs_filtersize']
    edge_id_output_sample = detect_cs(sample)
    aa = detect_cs(sample_array)
    unique_element, count = np.unique(sample_array, return_counts=True)
    code_64_unique_element = np.unique(aa)
    # id1 = np.asarray(code_64_unique_element // (2 ** 32), dtype=np.uint32)
    # id2 = np.asarray(code_64_unique_element % (2 ** 32), dtype=np.uint32)
    # raise()
    id1 = np.unique(np.concatenate((np.asarray(code_64_unique_element // (2 ** 32), dtype=np.uint32), np.asarray(code_64_unique_element % (2 ** 32), dtype=np.uint32)), axis=0))
    mask = np.isin(id1, unique_element)
    #print(mask)
    assert np.all(mask == True), \
        "Bounding box dictionary mismatch."\


    assert np.all(np.asarray(edge_id_output_sample[9-offset[0], 9-offset[1], :], np.uint32) != 0), " Edge detetion along axis=0 or 1 not working"
    assert np.all(np.asarray(edge_id_output_sample[10-offset[0], 10-offset[1], :], np.uint32) != 0), " Edge detetion along axis=0 or 1 not working"
    assert np.all(np.asarray(edge_id_output_sample[11-offset[0], 11-offset[1], :], np.uint32) != 0), " Edge detetion along axis=0 or 1 not working"

    assert np.all(np.asarray(edge_id_output_sample[10-offset[0], :, 12-offset[2]], np.uint32) != 0), " Edge detetion along axis=2 or 0 not working"

    assert np.all(np.asarray(edge_id_output_sample[:, 10-offset[1], 5-offset[2]], np.uint32) != 0), " Edge detetion along axis=1 or 2 not working"

    # assert np.all(np.asarray([2,5,5,1]) != 0), " Edge detetion in not working"
    print(offset)
    print(np.asarray(edge_id_output_sample[3, 3, :], np.uint32))
    # print(id1)
    # print(unique_element)
    # print(sample)
    print(np.asarray(edge_id_output_sample, np.uint32))
    print("seperation")
    print(np.asarray(edge_id_output_sample, np.uint64) // (2 ** 32))

# def test_detect_cs():
#     edge_s = 15
#     pp = np.random.randint(low=0, size=edge_s**3, high=3).reshape((
#         edge_s, edge_s, edge_s)).astype(np.uint32)
#     # pp[1][1][1] = 1
#     # pp[0][1][1] = 1
#     # pp[2][1][1] = 1
#     # pp[1][0][1] = 1
#     # pp[1][2][1] = 1
#     # pp[1][1][0] = 1
#     # pp[1][1][2] = 1
#     # #pp[1][1][1] = 1
#     weight = np.array([[[0, 0, 0],
#                [0, 1, 0],
#                [0, 0, 0]],
#               [[0, 1, 0],
#                [1, -6, 1],
#                [0, 1, 0]],
#               [[0, 0, 0],
#                [0, 1, 0],
#                [0, 0, 0]]], dtype=np.int)
#     sample = np.zeros((edge_s, edge_s, edge_s), dtype=np.uint32)
#     sample[3][3] = 2
#     # sample[2][3] = 0
#     # sample[5][6] = 0
#     offset = np.array(con['cell_objects']['cs_filtersize'], dtype=np.int) // 2
#     edge = scipy.ndimage.convolve(pp.astype(np.int), weight) < 0
#     edge_rel = np.nonzero(edge[offset[0]: -offset[0], offset[1]: -offset[1], offset[2]: -offset[2]])
#
#     # print(edge[0][0], edge[1][0], edge[2][0])
#     aa = detect_cs(pp)
#     unique_element, count = np.unique(sample_array, return_counts=True)
#     code_64_unique_element = np.unique(aa)
#     id1 = code_64_unique_element // (2**32)
#     id2 = code_64_unique_element % (2**32)
#     id1 = np.unique(np.concatenate(id1,id2))
#     mask = np.isin(id1, unique_element)
#     for e in id1:
#         if not e in unique_element:
#
#
#     # for code_64_bit in code_64_unique_element:
#     #     id1 = code_64_bit // (2**32)
#     #     id2 = code_64_bit % (2**32)
#     #     if [id1 is in unique_element]:
#
#     for i, j, k in zip(edge_rel[0], edge_rel[1], edge_rel[2]):
#         print(aa[i][j][k] % (2**32))
#         print(aa[i][j][k] // (2 ** 32))
#         print("id1")
#         print(pp[i][j][k])
#         print("test")
#         # print(aa[i][j][k] // (2 ** 32))
#         # print("id2")
#
#
#     # aa = np.array([[],
#     #                [],
#     #                []], np.uint64)
#
#     print(con['cell_objects']['cs_filtersize'])
#     print(np.asarray(aa))
#     print(edge[0].shape)
#     # print(type(aa[0]))
#     # print(len(aa[0]))
#     # print(np.asarray(aa[0]))
#     # print(pp[7])
#     # stencil = (7, 7, 3)
#     # print(stencil)
#     # a=[0,1,2,3,4,5]
#     # print(a[1:-1])
#     # print(4294967298 % (2^(32)))


def test_process_block_nonzero():
    print("p")


if __name__ == '__main__':
    #test_find_object_properties()
    test_detect_cs()
    #test_bfs_smoothing()