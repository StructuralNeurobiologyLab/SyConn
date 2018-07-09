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
version = args[2]
for sv_ixs in ch:
    # only generate SSV temporarily but do not write it to FS
    # corresponding SVs are parsed explicitly ('sv_ids=sv_ixs')
    sso = SuperSegmentationObject(sv_ixs[0], working_dir=wd, version=version,
                                  create=False, sv_ids=sv_ixs)
    sso.render_views(add_cellobjects=False, woglia=False, overwrite=True)

with open(path_out_file, "wb") as f:
    pkl.dump("0", f)
