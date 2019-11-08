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
    print(func_output)
    print(func_output[0])
    for i in range(len(element)):
        if(element[i] != 0 and func_output[2][element[i]] != count[i]):
            print("Count function not working")
        elif i != 0:
            ll = func_output[0][element[i]]
            check = sample_array[ll[0],ll[1],ll[2]]
            if (element[i] != check):
                print("object voxel dosen't match")
        # elif(i != 0 and (list(np.unravel_index(first_occurence[i], sample_array.shape)) != func_output[0][element[i]])):
            # print(list(np.unravel_index(first_occurence[i], sample_array.shape)))
            # print(func_output[0][element[i]])
        elif():
            continue
        # print("pp")
    indices= sample_array.flatten()
    min_bound = []
    max_bound = []
    a=len(indices)
    for i in range(int(element[len(element)-1]) + 1):
        min_bound.append([a,a,a])
        max_bound.append([0, 0, 0])
    print(min_bound)
    print(max_bound)
    for i in range(len(indices)):
        # if(element[indices[i]] != 0):
        if (indices[i] != 0):
            qq=indices[i]
            index=list(np.unravel_index(i, sample_array.shape))
            min_bound[qq] = np.minimum(min_bound[qq],index)
            max_bound[qq] = np.maximum(max_bound[qq],index)
            # print("okay")
    for i in element:
        if i != 0 :
            if not np.array_equal(func_output[1][i][0], min_bound[i]):
                if not np.array_equal(func_output[1][i][1], (max_bound[i] + [1, 1, 1])):
                    print("Bounding box error")
    print(min_bound)
    print(max_bound)
    # for i in range(len(element)):
    #
    #     if(i!=0 and (list(np.unravel_index(first_occurence[i],sample_array.shape))!=func_output[0][element[i]])):
    #         print(list(np.unravel_index(first_occurence[i], sample_array.shape)))
    #         print(func_output[0][element[i]])
    #         print("first occurence don't match")
    #     else:
    #         continue
    # print(func_output)
    # generate_default_conf(test_dir, scaling=(1, 1, 1), force_overwrite=True)
    # assert os.path.isfile(test_dir + '/config.yml'), "Error creating config file."
    # with pytest.raises(ValueError):
    #     generate_default_conf(test_dir, scaling=(1, 1, 1))
    # conf = Config(test_dir)
    # conf2 = Config(test_dir)
    # assert conf == conf2, "Mismatch between config objects."
    # entries = conf.entries
    # entries['scaling'] = (2, 2, 2)
    # assert conf != conf2, "Expected mismatch between config objects but " \
    #                       "they are the same."
    # conf.write_config(test_dir)
    # conf2 = Config(test_dir)
    # assert conf == conf2, "Mismatch between config objects after re-loading " \
    #                       "modified config file."
    # os.remove(conf.path_config)
    # logging.info('PASSED: Load and write `Config` class.')


if __name__ == '__main__':
    #raise()
    test_find_objects_properties()
