# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max-Planck-Institute for Medical Research, Heidelberg, Germany
# Authors: Sven Dorkenwald, Philipp Schubert, Jörgen Kornfeld

import sys
from syconn.reps.super_segmentation_object import celltype_predictor
try:
    import cPickle as pkl
except ImportError:
    import pickle as pkl

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
missing = celltype_predictor(ch)

with open(path_out_file, "wb") as f:
    pkl.dump(missing, f)
