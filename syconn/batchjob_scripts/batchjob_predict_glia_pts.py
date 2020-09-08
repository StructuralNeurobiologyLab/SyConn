# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Philipp Schubert
import sys
from syconn.handler.prediction_pts import predict_glia_ssv
from syconn import global_params
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


ch, pred_key = args
assert global_params.config.use_point_models

ncpus = global_params.config['ncores_per_node'] // global_params.config['ngpus_per_node']

n_worker = 2

lo_first_n = global_params.config['glia']['subcc_chunk_size_big_ssv']
working_dir = global_params.config.working_dir
ssv_params = []
partitioned = dict()
for sv_ids, g, was_partitioned in ch:
    ssv_params.append(dict(ssv_id=sv_ids[0], sv_ids=sv_ids, working_dir=working_dir, sv_graph=g, version='tmp'))
    partitioned[sv_ids[0]] = was_partitioned
postproc_kwargs = dict(pred_key=pred_key, lo_first_n=lo_first_n, partitioned=partitioned)
predict_glia_ssv(ssv_params, postproc_kwargs=postproc_kwargs, show_progress=False)

with open(path_out_file, "wb") as f:
    pkl.dump(None, f)
