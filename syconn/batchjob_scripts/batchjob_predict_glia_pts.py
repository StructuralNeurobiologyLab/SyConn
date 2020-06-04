# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Philipp Schubert
import sys
from syconn.handler.prediction_pts import predict_glia_ssv
from syconn import global_params
from syconn.handler import basics
from syconn.mp.mp_utils import start_multiprocess_imap
import numpy as np
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


def _predict_glia_ssv(*args, **kwargs):
    return predict_glia_ssv(*args, **kwargs)


ch, pred_key = args
assert global_params.config.use_point_models

ncpus = global_params.config['ncores_per_node'] // global_params.config['ngpus_per_node']

n_worker = 2

kwargs_semseg2mesh = global_params.config['spines']['semseg2mesh_spines']
lo_first_n = global_params.config['glia']['subcc_chunk_size_big_ssv']
working_dir = global_params.config.working_dir
ssv_params = []
partitioned = dict()
for sv_ids, g, was_partitioned in ch:
    ssv_params.append(dict(ssv_id=sv_ids[0], sv_ids=sv_ids, working_dir=working_dir, sv_graph=g))
    partitioned[sv_ids[0]] = was_partitioned
postproc_kwargs = dict(pred_key=pred_key, lo_first_n=lo_first_n, partitioned=partitioned)
params = [(el, dict(postproc_kwargs=postproc_kwargs)) for el in basics.chunkify(ch, n_worker * 2)]
res = start_multiprocess_imap(predict_glia_ssv, params, nb_cpus=n_worker)
missing = np.concatenate(res)
if len(missing) > 0:
    raise ValueError('Sem. seg. prediction of {} SSVs ({}) failed.'.format(
        len(missing), str(missing)))
with open(path_out_file, "wb") as f:
    pkl.dump(None, f)
