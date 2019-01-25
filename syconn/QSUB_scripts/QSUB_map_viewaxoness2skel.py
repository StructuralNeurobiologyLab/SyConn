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
from syconn.reps.super_segmentation_helper import majority_vote_compartments
from syconn.reps.super_segmentation_object import SuperSegmentationObject
from syconn import global_params

path_storage_file = sys.argv[1]
path_out_file = sys.argv[2]

with open(path_storage_file, 'rb') as f:
    args = []
    while True:
        try:
            args.append(pkl.load(f))
        except EOFError:
            break

ssv_ixs = args[0]
pred_key_appendix = args[1]
max_dist = global_params.DIST_AXONESS_AVERAGING
for ix in ssv_ixs:
    sso = SuperSegmentationObject(ix, working_dir=global_params.config.working_dir)
    sso.load_skeleton()
    if sso.skeleton is None or len(sso.skeleton["nodes"]) < 2:
        print("Skeleton of SSV %d has zero or less than two nodes." % ix)
        continue
    try:
        for k in [1]:
            sso.cnn_axoness2skel(pred_key_appendix=pred_key_appendix,
                                 force_reload=True, k=k)
    except Exception as e:
        print(str(e) + " SSV mapping error " + str(sso.id))
        continue
    try:
        sso.average_node_axoness_views(pred_key_appendix=pred_key_appendix,
                                       max_dist=max_dist)
    except Exception as e:
        print(str(e) + " SSV averaging error " + str(sso.id) )
        continue
    pred_key = "axoness{}_avg{}".format(pred_key_appendix, max_dist)
    majority_vote_compartments(sso, pred_key)


with open(path_out_file, "wb") as f:
    pkl.dump("0", f)
