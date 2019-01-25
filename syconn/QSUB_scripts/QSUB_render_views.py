# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max-Planck-Institute for Medical Research, Heidelberg, Germany
# Authors: Sven Dorkenwald, Philipp Schubert, Jörgen Kornfeld

import sys
# TODO: This will be used if PYOPENGL PLATFORM is egl

try:
    import cPickle as pkl
except ImportError:
    import pickle as pkl
from syconn.reps.super_segmentation import SuperSegmentationObject
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


ch = args[0]
wd = args[1]
# global_params.wd = wd
for ssv_ix in ch:
    print(global_params.config.working_dir)
    sso = SuperSegmentationObject(ssv_ix, working_dir=wd,
                                  enable_locking_so=True)
    sso.load_attr_dict()
    sso.render_views(add_cellobjects=True, woglia=True, overwrite=True)

with open(path_out_file, "wb") as f:
    pkl.dump("0", f)
