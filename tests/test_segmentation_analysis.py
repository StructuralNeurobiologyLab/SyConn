# SyConn
# Copyright (c) 2016 Philipp J. Schubert
# All rights reserved

from syconn.reps.rep_helper import find_object_properties
import numpy as np


def test_find_objects_properties():
    sample_array = np.array([
            [[0, 1],
             [1, 1]],
            [[5, 2],
             [2, 1]]], np.uint)
    func_output = find_object_properties(sample_array)
    #func_output[0]=dictionary of id's of unique voxel id, voxel id as key
    #func_output[1]=dictionary of bounding box for voxel ids, voxel id as key
    #func_output[2]=dictionary of count of all voxel ids, voxel id as key
    element, count = np.unique(sample_array, return_counts=True)
    for i in range(len(element)):
        if element[i] != 0:
            assert (func_output[2][element[i]] == count[i].astype(np.uint64)), "Count of the voxels not working" \
                                                                               " They should be same"
            ll = func_output[0][element[i]]
            check = sample_array[ll[0], ll[1], ll[2]]
            assert element[i].astype(np.uint64) == check.astype(np.uint64), "object voxel directory dosen't match" \
                             "They should be same"

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
                "Bounding box directory mismatch." \
                " They should be same"
            assert np.array_equal(np.array(func_output[1][i][1]), (max_bound[i] + np.ones((3,), dtype=np.uint64))), \
                "Bounding box directory mismatch." \
                         " They should be same"

if __name__ == '__main__':
    test_find_objects_properties()
