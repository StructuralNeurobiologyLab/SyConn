import os

import sys
import numpy as np
try:
    import cPickle as pkl
except ImportError:
    import pickle as pkl
from syconn.reps.super_segmentation_helper import pred_and_save_semseg_svs
from syconn.proc.sd_proc import sos_dict_fact, init_sos
from syconn.handler.prediction import NeuralNetworkInterface
from syconn.backend.storage import AttributeDict, CompressedStorage

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

# By default use views after glia removal
if 'woglia' in pred_kwargs:
    woglia = pred_kwargs["woglia"]
    del pred_kwargs["woglia"]
else:
    woglia = True
pred_key = pred_kwargs["pred_key"]
if 'raw_only' in pred_kwargs:
    raw_only = pred_kwargs['raw_only']
    del pred_kwargs['raw_only']
else:
    raw_only = False

model = NeuralNetworkInterface(**model_kwargs)
for p in so_chunk_paths:
    # get raw views
    view_dc_p = p + "/views_woglia.pkl" if woglia else p + "/views.pkl"
    view_dc = CompressedStorage(view_dc_p, disable_locking=True)
    svixs = list(view_dc.keys())
    views = list(view_dc.values())
    if raw_only:
        views = views[:, :1]
    sd = sos_dict_fact(svixs, **so_kwargs)
    svs = init_sos(sd)
    label_views = pred_and_save_semseg_svs(model, views, svs)
    # choose any SV to get a path constructor for the view storage (is the same for all SVs of this chunk)
    lview_dc_p = svs[0].view_path(woglia, view_key=pred_key)
    label_vd = CompressedStorage(lview_dc_p, disable_locking=True)
    for ii in range(len(svs)):
        label_vd[svs[ii].id] = label_views[ii]
    label_vd.push()

with open(path_out_file, "wb") as f:
    pkl.dump("0", f)
