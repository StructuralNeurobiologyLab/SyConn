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
map_properties = global_params.map_properties_semsegax
for ix in ssv_ixs:
    sso = SuperSegmentationObject(ix, working_dir=global_params.config.working_dir)
    sso.load_skeleton()
    if sso.skeleton is None or len(sso.skeleton["nodes"]) < 2:
        print("Skeleton of SSV %d has zero or less than two nodes." % ix)
        continue
    # background label is 5 -> unpredicted is background_label + 1 = 6, average over 50 nearest
    # vertex predictions
    print(ix, len(sso.skeleton['nodes']), len(sso.mesh[1].reshape((-1, 3))))
    node_preds = sso.semseg_for_coords(sso.skeleton['nodes'],
                                       semseg_key=global_params.view_properties_semsegax['semseg_key'],
                                       **map_properties)
    pred_key = "axoness{}_k{}".format(pred_key_appendix, map_properties['k'])
    sso.skeleton[pred_key] = node_preds
    # this will save sso.skeleton, -> also saves `sso.skeleton[pred_key]`
    # TODO: perform this only on 0, 1, 2 -> axon, dendrite and soma, not on bouton predictions
    majority_vote_compartments(sso, pred_key)


with open(path_out_file, "wb") as f:
    pkl.dump("0", f)
