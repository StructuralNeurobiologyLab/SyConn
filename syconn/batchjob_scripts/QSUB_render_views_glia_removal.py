# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max-Planck-Institute for Medical Research, Heidelberg, Germany
# Authors: Sven Dorkenwald, Philipp Schubert, Jörgen Kornfeld

import sys
import os
import numpy as np
import networkx as nx
try:
    import cPickle as pkl
except ImportError:
    import pickle as pkl
from syconn import global_params
from syconn.reps.super_segmentation import SuperSegmentationObject
from syconn.proc.rendering import render_sso_coords_multiprocessing
from syconn.handler.basics import chunkify
from syconn.mp.batchjob_utils import batchjob_script
import shutil

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
version = args[2]

ssvs_large = []
ssvs_small = []
for g in ch:
    # only generate SSV temporarily but do not write it to the file system
    # corresponding SVs are parsed explicitly ('sv_ids=sv_ixs')
    sv_ixs = np.sort(list(g.nodes()))
    sso = SuperSegmentationObject(sv_ixs[0], working_dir=wd, version=version,
                                  create=False, sv_ids=sv_ixs, mesh_caching=True)
    # nodes of sso._rag need to be SV
    new_G = nx.Graph()
    for e in g.edges():
        new_G.add_edge(sso.get_seg_obj("sv", e[0]),
                       sso.get_seg_obj("sv", e[1]))
    sso._rag = new_G
    if len(sso.sample_locations()) > 1000:
        #  TODO: Add as parameter to global_params.py
        ssvs_large.append(sso)
    else:
        ssvs_small.append(sso)
# render huge SSVs in parallel, multiple jobs per SSV, use more threads than cores -> increase
# GPU load
render_kwargs = dict(add_cellobjects=False, woglia=False, overwrite=True)
n_parallel_jobs = global_params.config['ncores_per_node']
for ssv in ssvs_large:
    render_sso_coords_multiprocessing(ssv, n_parallel_jobs,
                                      render_indexviews=False, return_views=False,
                                      render_kwargs=render_kwargs)
    ssv.clear_cache()
render_kwargs = dict(add_cellobjects=False, woglia=False, overwrite=True,
                     skip_indexviews=True)
# render small SSVs in parallel, one job per SSV
sso_kwargs = dict(version=version, create=False, working_dir=wd)
if len(ssvs_small) != 0:
    multi_params = [[ssv.id, ssv.sv_ids] for ssv in ssvs_small]
    print([len(el[1]) for el in multi_params])
    multi_params = chunkify(multi_params, n_parallel_jobs)
    # list of SSV IDs and SSD parameters need to be given to a single QSUB job
    multi_params = [(ixs, sso_kwargs, render_kwargs) for ixs in multi_params]
    path_out = batchjob_script(
        multi_params, "render_views", suffix="_SSV{}".format(ssvs_small[0].id),
        n_cores=1, disable_batchjob=True, overwrite=True)
    folder_del = os.path.abspath(path_out + "/../")
    shutil.rmtree(folder_del, ignore_errors=True)
with open(path_out_file, "wb") as f:
    pkl.dump("0", f)
