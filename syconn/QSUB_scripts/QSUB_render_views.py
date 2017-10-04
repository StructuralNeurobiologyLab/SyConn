# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max-Planck-Institute for Medical Research, Heidelberg, Germany
# Authors: Sven Dorkenwald, Philipp Schubert, Jörgen Kornfeld

import sys
import numpy as np
import cPickle as pkl
from syconnfs.representations import super_segmentation_helper as ssh
from syconnfs.representations.super_segmentation import render_sampled_sos_cc
from syconnfs.representations.segmentation_helper import sos_dict_fact, init_sos

path_storage_file = sys.argv[1]
path_out_file = sys.argv[2]

with open(path_storage_file) as f:
    args = []
    while True:
        try:
            args.append(pkl.load(f))
        except:
            break


so_kwargs = {}  # <-- change SO kwargs here
ch = args[0]
kwargs = args[1]
for svixs in ch:
    sd = sos_dict_fact(svixs, **so_kwargs)
    sos = init_sos(sd)
    render_sampled_sos_cc(sos, **kwargs)
