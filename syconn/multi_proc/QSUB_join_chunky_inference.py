# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max-Planck-Institute for Medical Research, Heidelberg, Germany
# Authors: Sven Dorkenwald, Philipp Schubert, Jörgen Kornfeld

import sys

import cPickle as pkl
from syconn.processing import predictor_cnn as pc

path_storage_file = sys.argv[1]
path_out_file = sys.argv[2]

with open(path_storage_file) as f:
    args = []
    while True:
        try:
            args.append(pkl.load(f))
        except:
            break

cset = args[0]
config_path = args[1]
param_path = args[2]
names = args[3]
labels = args[4]
offset = args[5]
batch_size = args[6]
kd_raw = args[7]
gpu = args[8]

pc.join_chunky_inference(cset, config_path, param_path, names, labels, offset,
                         batch_size, kd=kd_raw, gpu=gpu)
