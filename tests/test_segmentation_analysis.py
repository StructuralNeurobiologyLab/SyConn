# SyConn
# Copyright (c) 2016 Philipp J. Schubert
# All rights reserved

from syconn.reps.rep_helper import find_object_properties
from syconn.extraction.cs_extraction_steps import detect_cs
import numpy as np


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
    aa = detect_cs(np.ones((13,13,13)))
    print(type(aa))


if __name__ == '__main__':
    test_find_object_properties()
    test_detect_cs()