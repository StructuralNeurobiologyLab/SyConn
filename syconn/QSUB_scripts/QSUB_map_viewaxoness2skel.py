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
        print("Skeleton of SSV %d has zero nodes." % ix)
        continue
    if "axoness" in sso.skeleton:
        pass
        # print("Axoness of SSV %d already exists." % sso.id)
    else:
        try:
            for k in [1, 20]:
                sso.cnn_axoness_2_skel(k=k)
        except Exception as e:
            print("\n------------------------\n" + str(e) +
                  "\nSSV: " + str(sso.id) +
                  "\n------------------------\n")
    try:
        if not "axoness_pred_avg15000" in sso.skeleton:
            sso.average_node_axoness(avg_window=5000)
            sso.average_node_axoness(avg_window=10000)
            sso.average_node_axoness(avg_window=15000)
        # else:
        #     print("Smoothed axoness prediction already exists for SSV %d." % sso.id)
    except Exception as e:
        print("\n------------------------\n" + str(e) +
              "\nSSV: " + str(sso.id) +
              "\n------------------------\n")
