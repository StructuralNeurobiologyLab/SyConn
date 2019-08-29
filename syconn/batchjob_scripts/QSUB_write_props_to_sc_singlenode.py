# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Philipp Schubert, Jörgen Kornfeld

import sys

try:
    import cPickle as pkl
except ImportError:
    import pickle as pkl
from syconn.proc import sd_proc
from syconn.mp.mp_utils import start_multiprocess
from syconn import global_params
from syconn.handler.basics import chunkify_successive

path_storage_file = sys.argv[1]
path_out_file = sys.argv[2]

with open(path_storage_file, 'rb') as f:
    args = []
    while True:
        try:
            args.append(pkl.load(f))
        except EOFError:
            break

n_cores = min(global_params.config['ncores_per_node'] // 2, 10)
obj_id_chs = args[0]
params = args[1:]
n_elements_per_job = min(len(obj_id_chs) // n_cores, 15)
if n_elements_per_job != 0:
    multi_params = [[obj_ch, ] + params for obj_ch in chunkify_successive(obj_id_chs,
                                                                          n_elements_per_job)][::-1]
    print("Starting {} worker for {} jobs with each {} elements.".format(
        n_cores, len(multi_params), n_elements_per_job))
    out = start_multiprocess(sd_proc._write_props_to_sc_thread, multi_params, nb_cpus=n_cores)

with open(path_out_file, "wb") as f:
    pkl.dump(None, f)
