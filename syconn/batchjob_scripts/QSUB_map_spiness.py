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
from syconn.reps.super_segmentation_dataset import map_ssv_semseg

path_storage_file = sys.argv[1]
path_out_file = sys.argv[2]

with open(path_storage_file, 'rb') as f:
    args = []
    while True:
        try:
            args.append(pkl.load(f))
        except EOFError:
            break

map_ssv_semseg(args)

with open(path_out_file, "wb") as f:
    pkl.dump("0", f)
