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
from syconn.reps.super_segmentation_helper import create_sso_skeleton, sparsify_skeleton, extract_skel_features, associate_objs_with_skel_nodes
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
    sso.load_skeleton()
    create_sso_skeleton(sso)
    if sso.skeleton is None or len(sso.skeleton["nodes"]) == 0:
        continue
    sso.save_skeleton()
    # generate and cache features at different contexts (e.g. spiness/axoness):
    try:
        if len(sso.skeleton["nodes"]) > 0:
            if not "assoc_sj" in sso.skeleton:
                associate_objs_with_skel_nodes(sso)
            for ctx in [2000, 8000]:
                features = extract_skel_features(sso, feature_context_nm=ctx)
                sso._save_skelfeatures(ctx, features, overwrite=True)
    except Exception as e:
        print("Error occurred with SSO ", sso.id, str(e))

with open(path_out_file, "wb") as f:
    pkl.dump("0", f)
