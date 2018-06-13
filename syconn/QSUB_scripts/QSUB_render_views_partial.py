# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max-Planck-Institute for Medical Research, Heidelberg, Germany
# Authors: Sven Dorkenwald, Philipp Schubert, Jörgen Kornfeld

import sys
import numpy as np
try:
    import cPickle as pkl
# TODO: switch to Python3 at some point and remove above
except Exception:
    import pickle as pkl
from syconn.reps.super_segmentation import render_sampled_sos_cc
from syconn.proc.sd_proc import sos_dict_fact, init_sos

path_storage_file = sys.argv[1]
path_out_file = sys.argv[2]

with open(path_storage_file) as f:
    args = []
    while True:
        try:
            args.append(pkl.load(f))
        except:
            break


ch = args[0]
so_kwargs = args[1]
kwargs = args[2]
print kwargs
for svixs in ch:
    sd = sos_dict_fact(svixs, **so_kwargs)
    sos = init_sos(sd)
    render_sampled_sos_cc(sos, **kwargs)
