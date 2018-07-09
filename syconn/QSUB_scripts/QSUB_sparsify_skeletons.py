# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max-Planck-Institute for Medical Research, Heidelberg, Germany
# Authors: Sven Dorkenwald, Philipp Schubert, Jörgen Kornfeld

import sys

try:
    import cPickle as pkl
except ImportError:
    import pickle as pkl
from syconn.reps.super_segmentation_helper import sparsify_skeleton, create_sso_skeleton
from syconn.reps.super_segmentation_object import SuperSegmentationObject

path_storage_file = sys.argv[1]
path_out_file = sys.argv[2]

with open(path_storage_file, 'rb') as f:
    args = []
    while True:
        try:
            args.append(pkl.load(f))
        except EOFError:
            break


ssv_ixs = args
for ix in ssv_ixs:
    sso = SuperSegmentationObject(ix, version="0", working_dir="/wholebrain/scratch/areaxfs3/")
    if sso.load_skeleton():
        continue
    create_sso_skeleton(sso)
    if len(sso.skeleton["nodes"]) == 0:
        print("Skeleton of SSV %d has zero nodes." % ix)
        continue
    sparsify_skeleton(sso)
    sso.save_skeleton()
    print("Created stitched, pruned and sparsed skeleton for SSV", ix)

with open(path_out_file, "wb") as f:
    pkl.dump("0", f)
