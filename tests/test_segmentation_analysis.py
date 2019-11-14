# SyConn
# Copyright (c) 2016 Philipp J. Schubert
# All rights reserved

from syconn.reps.rep_helper import find_object_properties
import numpy as np


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
