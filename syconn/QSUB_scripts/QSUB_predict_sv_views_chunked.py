import os

import sys
import numpy as np
try:
    import cPickle as pkl
except ImportError:
    import pickle as pkl
from syconn.reps.super_segmentation import render_sampled_sos_cc
from syconn.proc.sd_proc import sos_dict_fact, init_sos, predict_views
from syconn.handler.prediction import NeuralNetworkInterface
from syconn.handler.compression import LZ4Dict, AttributeDict
path_storage_file = sys.argv[1]
path_out_file = sys.argv[2]

with open(path_storage_file) as f:
    args = []
    while True:
        try:
            args.append(pkl.load(f))
        except:
            break

so_chunk_paths = args[0]
model_kwargs = args[1]
so_kwargs = args[2]
pred_kwargs = args[3]

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
    view_dc = LZ4Dict(view_dc_p, disable_locking=True)
    svixs = list(view_dc.keys())
    views = list(view_dc.values())
    if raw_only:
        views = views[:, :1]
    sd = sos_dict_fact(svixs, **so_kwargs)
    sos = init_sos(sd)
    probas = predict_views(model, views, sos, return_proba=True, **pred_kwargs)
    attr_dc_p = p + "/attr_dict.pkl"
    ad = AttributeDict(attr_dc_p, disable_locking=True)
    for ii in range(len(sos)):
        ad[sos[ii].id][pred_key] = probas[ii]
    ad.save2pkl()
with open(path_out_file, "wb") as f:
    pkl.dump("0", f)