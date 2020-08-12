# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max-Planck-Institute for Medical Research, Heidelberg, Germany
# Authors: Philipp Schubert, Jörgen Kornfeld

import sys
from syconn.reps.super_segmentation_object import semsegaxoness_predictor
from syconn import global_params
from syconn.handler import basics
from syconn.handler.prediction_pts import predict_cmpt_ssd
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
map_properties = global_params.config['compartments']['map_properties_semsegax']
pred_key = global_params.config['compartments']['view_properties_semsegax']['semseg_key']
max_dist = global_params.config['compartments']['dist_axoness_averaging']
ncpus = global_params.config['ncores_per_node'] // global_params.config['ngpus_per_node']
view_props = global_params.config['compartments']['view_properties_semsegax']

if global_params.config.use_point_models:
    # TODO: move mpath to global params
    ssd_kwargs = dict(working_dir=global_params.config.working_dir)
    predict_cmpt_ssd(ssd_kwargs=ssd_kwargs, ssv_ids=ch, mpath='~/thesis/current_work/paper/test_models/', bs=2)
else:
    n_worker = 2
    params = [(ch_sub, view_props, ncpus, map_properties,
               pred_key, max_dist) for ch_sub in basics.chunkify(ch, n_worker * 2)]
    res = start_multiprocess_imap(semsegaxoness_predictor, params, nb_cpus=n_worker)
    missing = np.concatenate(res)
    if len(missing) > 0:
        missing = semsegaxoness_predictor((missing, view_props, ncpus, map_properties))
    if len(missing) > 0:
        print('ERROR: Sem. seg. prediction of {} SSVs ({}) failed.'.format(
            len(missing), str(missing)))
with open(path_out_file, "wb") as f:
    pkl.dump(None, f)
