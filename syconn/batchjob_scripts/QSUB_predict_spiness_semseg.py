# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max-Planck-Institute for Medical Research, Heidelberg, Germany
# Authors: Philipp Schubert, Jörgen Kornfeld

import sys
from syconn.reps.super_segmentation_object import semsegspiness_predictor
from syconn import global_params
from syconn.handler import basics
from syconn.mp.mp_utils import start_multiprocess_imap
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
kwargs_semseg2mesh = global_params.config['spines']['semseg2mesh_spines']
kwargs_semsegforcoords = global_params.config['spines']['semseg2coords_spines']
view_props = global_params.config['views']['view_properties']
n_worker = 4
params = [(ch_sub, view_props, ncpus, kwargs_semseg2mesh, kwargs_semsegforcoords) for ch_sub in
          basics.chunkify(ch, n_worker * 2)]
res = start_multiprocess_imap(semsegspiness_predictor, params, nb_cpus=n_worker)
missing = np.concatenate(res)
if len(missing) > 0:
    raise ValueError('Sem. seg. prediction of {} SSVs ({}) failed.'.format(
        len(missing), str(missing)))
with open(path_out_file, "wb") as f:
    pkl.dump(None, f)
