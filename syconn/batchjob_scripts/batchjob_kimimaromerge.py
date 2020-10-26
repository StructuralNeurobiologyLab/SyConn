# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Philipp Schubert, Alexandra Rother
import sys
import pickle as pkl
import numpy as np
from syconn.handler.basics import load_pkl2obj
from syconn.proc.skeleton import kimimaro_mergeskels, kimimaro_skels_tokzip
from syconn import global_params
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

working_dir = global_params.config.working_dir
scaling = global_params.config["scaling"]
path2results_dc, ssv_ids = args
results_dc = load_pkl2obj(path2results_dc)

for ssv_id in ssv_ids:
    ssv_id = int(ssv_id)
    combined_skel = kimimaro_mergeskels(results_dc[ssv_id], ssv_id)
    sso = SuperSegmentationObject(ssv_id, working_dir=working_dir)

    sso.skeleton = dict()
    if combined_skel.vertices.size > 0:
        sso.skeleton["nodes"] = combined_skel.vertices / scaling  # to fit voxel coordinates
        # get radius in pseudo-voxel units (used by Knossos)
        sso.skeleton["diameters"] = (combined_skel.radii / scaling[0]) * 2  # divide by x scale
        sso.skeleton["edges"] = combined_skel.edges
    else:
        sso.skeleton["nodes"] = np.array([sso.rep_coord], dtype=np.float32)
        sso.skeleton["diameters"] = np.zeros((1, ), dtype=np.float32)
        sso.skeleton["edges"] = np.array([[0, 0], ], dtype=np.int)
    # TODO: apply sparsify_skeleton_fast to remove more nodes
    sso.save_skeleton()

with open(path_out_file, "wb") as f:
    pkl.dump("0", f)

