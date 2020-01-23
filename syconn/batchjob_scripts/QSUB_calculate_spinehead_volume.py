# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max-Planck-Institute for Medical Research, Heidelberg, Germany
# Authors: Sven Dorkenwald, Philipp Schubert, Jörgen Kornfeld

import sys

try:
    import cPickle as pkl
except Exception:
    import pickle as pkl
from syconn.reps.super_segmentation_helper import extract_spinehead_volume_mesh
from syconn.reps.super_segmentation import *

path_storage_file = sys.argv[1]
path_out_file = sys.argv[2]

with open(path_storage_file, 'rb') as f:
    args = []
    while True:
        try:
            args.append(pkl.load(f))
        except EOFError:
            break

sso_ids = args[0]

ssd = SuperSegmentationDataset()
for sso in ssd.get_super_segmentation_object(sso_ids):
    assert sso.load_skeleton(), f"Skeleton of SSO {sso.id} does not exist."
    # if 'spinehead_vol' in sso.skeleton:
    #     continue
    extract_spinehead_volume_mesh(sso)
    sso.save_skeleton()

with open(path_out_file, "wb") as f:
    pkl.dump("", f)
