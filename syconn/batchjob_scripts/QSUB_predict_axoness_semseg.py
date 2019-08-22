# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max-Planck-Institute for Medical Research, Heidelberg, Germany
# Authors: Sven Dorkenwald, Philipp Schubert, Jörgen Kornfeld

import sys
from syconn.reps.super_segmentation_object import semsegaxoness_predictor
from syconn import global_params
import numpy as np
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
missing = semsegaxoness_predictor([ch, global_params.NCORES_PER_NODE //
                                   global_params.NGPUS_PER_NODE])
if len(missing) > 0:
    print('WARNING: Restarting sem. seg. prediction for {} SSVs ({}).'.format(
        len(missing), str(missing)))
    missing = semsegaxoness_predictor([[m[0] for m in missing], global_params.NCORES_PER_NODE //
                                       global_params.NGPUS_PER_NODE])
if len(missing) > 0:
    print('ERROR: Sem. seg. prediction of {} SSVs ({}) failed.'.format(
        len(missing), str(missing)))
with open(path_out_file, "wb") as f:
    pkl.dump(None, f)
