# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max-Planck-Institute for Medical Research, Heidelberg, Germany
# Authors: Sven Dorkenwald, Philipp Schubert, Jörgen Kornfeld

import sys
import numpy as np
try:
    import cPickle as pkl
except ImportError:
    import pickle as pkl
from syconn.reps.super_segmentation_helper import majority_vote_compartments, \
    majorityvote_skeleton_property
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
pred_key = global_params.view_properties_semsegax['semseg_key']
max_dist = global_params.DIST_AXONESS_AVERAGING
for ix in ssv_ixs:
    sso = SuperSegmentationObject(ix, working_dir=global_params.config.working_dir)
    sso.load_skeleton()
    if sso.skeleton is None or len(sso.skeleton["nodes"]) < 2:
        print("Skeleton of SSV %d has zero or less than two nodes." % ix)
        continue
    # vertex predictions
    node_preds = sso.semseg_for_coords(sso.skeleton['nodes'],
                                       semseg_key=global_params.view_properties_semsegax['semseg_key'],
                                       **map_properties)

    # perform average only on axon dendrite and soma predictions
    nodes_ax_den_so = np.array(node_preds, dtype=np.int)
    # set en-passant and terminal boutons to axon class.
    nodes_ax_den_so[nodes_ax_den_so == 3] = 1
    nodes_ax_den_so[nodes_ax_den_so == 4] = 1
    sso.skeleton[pred_key] = node_preds

    # average along skeleton, stored as: "{}_avg{}".format(pred_key, max_dist)
    majorityvote_skeleton_property(sso, prop_key=pred_key,
                                   max_dist=max_dist)
    # suffix '_avg{}' is added by `_average_node_axoness_views`
    nodes_ax_den_so = sso.skeleton["{}_avg{}".format(pred_key, max_dist)]
    # recover bouton predictions within axons and store smoothed result
    nodes_ax_den_so[(node_preds == 3) & (nodes_ax_den_so == 1)] = 3
    nodes_ax_den_so[(node_preds == 4) & (nodes_ax_den_so == 1)] = 4
    sso.skeleton["{}_avg{}".format(pred_key, max_dist)] = nodes_ax_den_so

    # will create a compartment majority voting after removing all soma nodes
    # the restul will be written to: ``ax_pred_key + "_comp_maj"``
    majority_vote_compartments(sso, "{}_avg{}".format(pred_key, max_dist))
    nodes_ax_den_so = sso.skeleton["{}_avg{}_comp_maj".format(pred_key, max_dist)]
    # recover bouton predictions within axons and store majority result
    nodes_ax_den_so[(node_preds == 3) & (nodes_ax_den_so == 1)] = 3
    nodes_ax_den_so[(node_preds == 4) & (nodes_ax_den_so == 1)] = 4
    sso.skeleton["{}_avg{}_comp_maj".format(pred_key, max_dist)] = nodes_ax_den_so
    sso.save_skeleton()


with open(path_out_file, "wb") as f:
    pkl.dump("0", f)
