# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max-Planck-Institute for Medical Research, Heidelberg, Germany
# Authors: Philipp Schubert, Jörgen Kornfeld

import sys
from syconn.reps.super_segmentation_object import celltype_predictor
from syconn.handler import basics
from syconn.mp.mp_utils import start_multiprocess_imap
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

ncpus = global_params.config['ncores_per_node'] // global_params.config['ngpus_per_node']
model_props = global_params.config['celltypes']
# TODO: work-in
# global_params.config.use_point_models
n_worker = 2
params = basics.chunkify(ch, n_worker * 4)
res = start_multiprocess_imap(celltype_predictor, [(p, ncpus, model_props) for p in params],
                              nb_cpus=n_worker)
missing = np.concatenate(res)
if len(missing) > 0:
    missing = celltype_predictor((missing, model_props))
if len(missing) > 0:
    raise ValueError(f'Missing SSV IDs: {missing}')

with open(path_out_file, "wb") as f:
    pkl.dump(missing, f)
