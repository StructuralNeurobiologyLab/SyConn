# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max-Planck-Institute for Medical Research, Heidelberg, Germany
# Authors: Sven Dorkenwald, Philipp Schubert, Jörgen Kornfeld

import sys
import dill
import numpy as np  # needed for transfer function
import pickle as pkl
from syconn.extraction import object_extraction_steps as oes

path_storage_file = sys.argv[1]
path_out_file = sys.argv[2]

with open(path_storage_file, 'rb') as f:
    args = []
    while True:
        try:
            args.append(dill.load(f))
        except EOFError:
            break

out = oes._object_segmentation_thread(args)

with open(path_out_file, "wb") as f:
    pkl.dump(out, f)
