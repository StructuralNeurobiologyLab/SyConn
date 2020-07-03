# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max-Planck-Institute for Medical Research, Heidelberg, Germany
# Authors: Sven Dorkenwald, Philipp Schubert, Jörgen Kornfeld

try:
    import cPickle as pkl
except ImportError:
    import pickle as pkl
import sys
from syconn.proc import ssd_proc

path_storage_file = sys.argv[1]
path_out_file = sys.argv[2]

with open(path_storage_file, 'rb') as f:
    args = []
    while True:
        try:
            args.append(pkl.load(f))
        except EOFError:
            break

out = ssd_proc.map_synssv_objects_thread(args)

with open(path_out_file, "wb") as f:
    pkl.dump(out, f)
