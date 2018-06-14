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
from syconn.reps.super_segmentation_helper import sparsify_skeleton, create_sso_skeleton
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
for ix in ssv_ixs:
    sso = SuperSegmentationObject(ix, version="0", working_dir="/wholebrain/scratch/areaxfs3/")
    sso.load_skeleton()
    if sso.skeleton is None or len(sso.skeleton["nodes"]) <= 1:
        print("Skeleton of SSV %d has zero nodes." % ix)
        continue
    for feat_ctx_nm in [500, 1000, 2000, 4000, 8000]:
        try:
            _ = sso.skel_features(feat_ctx_nm)
        except IndexError as e:
            print("Error at SSO %d (context: %d).\n%s" % (sso.id, feat_ctx_nm, e))

with open(path_out_file, "wb") as f:
    pkl.dump("0", f)