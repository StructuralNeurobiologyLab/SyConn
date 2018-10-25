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
from syconn.reps.segmentation import SegmentationObject

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
for sv_ix in ch:
    # only generate SSV temporarily but do not write it to FS
    # corresponding SVs are parsed explicitly ('sv_ids=sv_ixs')
    sv = SegmentationObject(sv_ix, working_dir=wd)

with open(path_out_file, "wb") as f:
    pkl.dump("0", f)
