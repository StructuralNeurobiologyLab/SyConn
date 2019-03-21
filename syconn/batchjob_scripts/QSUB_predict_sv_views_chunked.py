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
from syconn.proc.sd_proc import sos_dict_fact, init_sos, predict_views
from syconn.handler.prediction import NeuralNetworkInterface
from syconn.backend.storage import AttributeDict, CompressedStorage
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
model_kwargs = args[1]
so_kwargs = args[2]
pred_kwargs = args[3]

working_dir = so_kwargs['working_dir']
global_params.wd = working_dir  # adapt working dir

woglia = pred_kwargs["woglia"]
del pred_kwargs["woglia"]
pred_key = pred_kwargs["pred_key"]
if 'raw_only' in pred_kwargs:
    raw_only = pred_kwargs['raw_only']
    del pred_kwargs['raw_only']
else:
    raw_only = False

model = NeuralNetworkInterface(**model_kwargs)
for p in so_chunk_paths:
    view_dc_p = p + "/views_woglia.pkl" if woglia else p + "/views.pkl"
    view_dc = CompressedStorage(view_dc_p, disable_locking=True)
    svixs = list(view_dc.keys())
    if len(svixs) == 0:
        continue
    views = list(view_dc.values())
    if raw_only and views[0].shape[1] != 1:
        for ii in range(len(views)):
            views[ii] = views[ii][:, 1]
    sd = sos_dict_fact(svixs, **so_kwargs)
    sos = init_sos(sd)
    probas = predict_views(model, views, sos, return_proba=True,
                           **pred_kwargs)
    attr_dc_p = p + "/attr_dict.pkl"
    ad = AttributeDict(attr_dc_p, disable_locking=True)
    for ii in range(len(sos)):
        ad[sos[ii].id][pred_key] = probas[ii]
    ad.push()

with open(path_out_file, "wb") as f:
    pkl.dump("0", f)
