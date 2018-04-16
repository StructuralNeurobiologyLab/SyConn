# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max-Planck-Institute for Medical Research, Heidelberg, Germany
# Authors: Sven Dorkenwald, Philipp Schubert, Jörgen Kornfeld

import sys

try:
    import cPickle as pkl
# TODO: switch to Python3 at some point and remove above
except Exception:
    import pickle as pkl
from syconn.reps.super_segmentation_helper import create_sso_skeleton
from syconn.reps.super_segmentation_object import SuperSegmentationObject

path_storage_file = sys.argv[1]
path_out_file = sys.argv[2]

with open(path_storage_file) as f:
    args = []
    while True:
        try:
            args.append(pkl.load(f))
        except:
            break


ssv_ixs = args
print(ssv_ixs)
for ix in ssv_ixs:
    sso = SuperSegmentationObject(ix, version="0", working_dir="/wholebrain/scratch/areaxfs3/")
    create_sso_skeleton(sso)
    sso.save_skeleton()
    print(ix)
    # generate and cache features at different contexts (e.g. spiness/axoness):
    for ctx in [2000, 8000]:
        _ = sso.skel_features(ctx)
