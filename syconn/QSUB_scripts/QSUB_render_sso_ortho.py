
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
# TODO: switch to Python3 at some point and remove above
except Exception:
    import pickle as pkl
from syconn.reps.super_segmentation_object import SuperSegmentationObject
from syconn.proc.rendering import render_sso_ortho_views

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
    if sso.size > 1e5:
        try:
             _ = sso.load_views(view_key="ortho")
        except KeyError:
            print("Rendering missing SSO %d." % sso.id)
            views = render_sso_ortho_views(sso)
            sso.save_views(views, view_key="ortho")

with open(path_out_file, "wb") as f:
    pkl.dump("0", f)