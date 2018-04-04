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

    if sso.skeleton is None or len(sso.skeleton["nodes"]) == 0:
        print "Skeleton of SSV %d has zero nodes." % ix
        continue
    # currently uncommented to just do window averaging
    # for k in [1, 20]:
    #     try:
    #         sso.cnn_axoness_2_skel(k=k)
    #     except Exception as e:
    #         print ("Couldn't map axoness (k=%d) for SSO %d.\n%s" % (k, ix, e))
    sso.average_node_axoness(5000)
    sso.average_node_axoness(10000)
    sso.average_node_axoness(15000)
