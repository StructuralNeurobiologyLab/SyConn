# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max-Planck-Institute for Medical Research, Heidelberg, Germany
# Authors: Sven Dorkenwald, Philipp Schubert, Jörgen Kornfeld

import sys

try:
    import cPickle as pkl
except ImportError:
    import pickle as pkl
from syconn.reps.super_segmentation import SuperSegmentationObject
from syconn.proc.rendering import render_sso_coords_multiprocessing
from syconn.mp.mp_utils import start_multiprocess_obj
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

ch = args[0]
wd = args[1]
render_kwargs = args[2]
ssvs_large = []
ssvs_small = []
for ssv_ix in ch:
    sso = SuperSegmentationObject(ssv_ix, working_dir=wd,
                                  enable_locking_so=True)
    if len(sso.sample_locations()) > 1000:
        ssvs_large.append(ssvs_small)

# render huge SSVs in parallel, multiple jobs per SSV
n_parallel_jobs = global_params.NCORES_PER_NODE // global_params.NGPUS_PER_NODE
for ssv in ssvs_large:
    render_sso_coords_multiprocessing(ssvs_large, wd, n_parallel_jobs,
                                      render_indexviews=False,
                                      render_kwargs=render_kwargs)
    render_sso_coords_multiprocessing(ssvs_large, wd, n_parallel_jobs,
                                      render_indexviews=True,
                                      render_kwargs=render_kwargs)

# render small SSVs in parallel, one job per SSV
start_multiprocess_obj('render_views', [[ssv, render_kwargs] for ssv in ssvs_small],
                       nb_cpus=n_parallel_jobs)
with open(path_out_file, "wb") as f:
    pkl.dump("0", f)
