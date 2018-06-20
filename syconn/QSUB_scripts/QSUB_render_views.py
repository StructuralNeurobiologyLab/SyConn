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
from syconn.reps.super_segmentation import SuperSegmentationObject

path_storage_file = sys.argv[1]
path_out_file = sys.argv[2]

with open(path_storage_file, 'rb') as f:
    args = []
    while True:
        try:
            args.append(pkl.load(f))
        except EOFError:
            break


ch = args[0]
wd = args[1]
for ssv_ix in ch:
    sso = SuperSegmentationObject(ssv_ix, working_dir=wd)
    sso.render_views(add_cellobjects=True, woglia=True, overwrite=True)


with open(path_out_file, "wb") as f:
    pkl.dump("0", f)
