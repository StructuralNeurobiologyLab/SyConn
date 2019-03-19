# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Sven Dorkenwald, Philipp Schubert, Joergen Kornfeld

import sys
try:
    import cPickle as pkl
except ImportError:
    import pickle as pkl
from syconn.proc.sd_proc import sos_dict_fact, init_sos, predict_sos_views
from syconn.handler.prediction import NeuralNetworkInterface
path_storage_file = sys.argv[1]
path_out_file = sys.argv[2]

with open(path_storage_file, 'rb') as f:
    args = []
    while True:
        try:
            args.append(pkl.load(f))
        except EOFError:
            break

svixs = args[0]
model_kwargs = args[1]
so_kwargs = args[2]
pred_kwargs = args[3]

model = NeuralNetworkInterface(**model_kwargs)
sd = sos_dict_fact(svixs, **so_kwargs)
sos = init_sos(sd)
out = predict_sos_views(model, sos, **pred_kwargs)

with open(path_out_file, "wb") as f:
    pkl.dump(out, f)
