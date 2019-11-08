# SyConn
# Copyright (c) 2016 Philipp J. Schubert
# All rights reserved

import pytest
from syconn.reps.rep_helper import find_object_properties
import os
import logging
import numpy as np


# test_dir = os.path.dirname(os.path.realpath(__file__))
# logging.basicConfig(filename=test_dir + '/test_config.log',
#                     level=logging.DEBUG, filemode='w')

def test_find_objects_properties():
    sample_array=np.array([
            [[0, 1],
             [1, 1]],
            [[5, 2],
             [2, 1]]],np.uint)
    func_output=find_object_properties(sample_array)
    element, count=np.unique(sample_array,return_counts=True)
    for i in range(len(element)):
        if(element[i] != 0 and func_output[2][element[i]] != count[i]):
            assert True, "Count of the voxels not working." \
                         " They should be same"
        elif i != 0:
            ll = func_output[0][element[i]]
            check = sample_array[ll[0],ll[1],ll[2]]
            if (element[i] != check):
                assert True, "object voxel directory dosen't match" \
                             "They should be same"
        # elif(i != 0 and (list(np.unravel_index(first_occurence[i], sample_array.shape)) != func_output[0][element[i]])):
        elif():
            continue
    indices= sample_array.flatten()
    min_bound = []
    max_bound = []
    a = len(indices)
    for i in range(int(element[len(element)-1]) + 1):
        min_bound.append([a, a, a])
        max_bound.append([0, 0, 0])
    for i in range(len(indices)):
        # if(element[indices[i]] != 0):
        if (indices[i] != 0):
            qq=indices[i]
            index=list(np.unravel_index(i, sample_array.shape))
            min_bound[qq] = np.minimum(min_bound[qq],index)
            max_bound[qq] = np.maximum(max_bound[qq],index)
    for i in element:
        if i != 0 :
            if not np.array_equal(func_output[1][i][0], min_bound[i]):
                if not np.array_equal(func_output[1][i][1], (max_bound[i] + [1, 1, 1])):
                    assert True, "Bounding box directory mismatch." \
                                 " They should be same"
                    print("Bounding box error")


if __name__ == '__main__':
    #raise()
    test_find_objects_properties()
