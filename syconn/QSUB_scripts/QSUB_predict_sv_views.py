import os
os.environ["PATH"] = os.environ["PATH"] + ":/u/pschuber/cuda-8.0/bin"
try:
    os.environ["LD_LIBRARY_PATH"] = os.environ["LD_LIBRARY_PATH"] + ":/u/pschuber/cuda-8.0/lib64"
except KeyError:
    os.environ["LD_LIBRARY_PATH"] = "/u/pschuber/cuda-8.0/lib64"
try:
    os.environ["LIBRARY_PATH"] = os.environ["LIBRARY_PATH"] + ":/u/pschuber/cuda-8.0/lib64/"
except KeyError:
    os.environ["LIBRARY_PATH"] = "/u/pschuber/cuda-8.0/lib64/"
try:
    os.environ["CUDA_HOME"] = os.environ["CUDA_HOME"] + ":/u/pschuber/cuda-8.0"
except KeyError:
    os.environ["CUDA_HOME"] = "/u/pschuber/cuda-8.0"


import sys
import numpy as np
try:
    import cPickle as pkl
# TODO: switch to Python3 at some point and remove above
except Exception:
    import pickle as pkl
from syconn.reps.super_segmentation import render_sampled_sos_cc
from syconn.proc.sd_proc import sos_dict_fact, init_sos, predict_sos_views
from syconn.handler.prediction import NeuralNetworkInterface
path_storage_file = sys.argv[1]
path_out_file = sys.argv[2]

with open(path_storage_file) as f:
    args = []
    while True:
        try:
            args.append(pkl.load(f))
        except:
            break

svixs = args[0]
model_kwargs = args[1]
so_kwargs = args[2]
pred_kwargs = args[3]

model = NeuralNetworkInterface(**model_kwargs)
sd = sos_dict_fact(svixs, **so_kwargs)
sos = init_sos(sd)
print(svixs)
predict_sos_views(model, sos, **pred_kwargs)