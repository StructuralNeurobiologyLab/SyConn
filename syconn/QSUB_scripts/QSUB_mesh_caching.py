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
except ImportError:
    import pickle as pkl
from syconn.proc.sd_proc import mesh_chunk
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

so_chunk_paths = args[0]
so_kwargs = args[1]

working_dir = so_kwargs['working_dir']
obj_type = so_kwargs['working_dir']
global_params.wd = working_dir
for path in so_chunk_paths:
    mesh_chunk((path, obj_type))

with open(path_out_file, "wb") as f:
    pkl.dump("0", f)
