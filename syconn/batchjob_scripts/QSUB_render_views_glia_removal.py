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
from syconn.mp.batchjob_utils import QSUB_script
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
    # only generate SSV temporarily but do not write it to FS
    # corresponding SVs are parsed explicitly ('sv_ids=sv_ixs')
    sv_ixs = np.sort(list(g.nodes()))
    sso = SuperSegmentationObject(sv_ixs[0], working_dir=wd, version=version,
                                  create=False, sv_ids=sv_ixs,
                                  enable_locking_so=True)
    # nodes of sso._rag need to be SV
    new_G = nx.Graph()
    for e in g.edges():
        new_G.add_edge(sso.get_seg_obj("sv", e[0]),
                       sso.get_seg_obj("sv", e[1]))
    sso._rag = new_G
    if len(sso.sample_locations()) > np.inf:  # TODO: adapt as soon as
        # `render_sso_coords_multiprocessing` has correct sorting of returned views
        ssvs_large.append(sso)
    else:
        ssvs_small.append(sso)

render_kwargs = dict(add_cellobjects=False, woglia=False, overwrite=True,
                     skip_indexviews=True)
# render huge SSVs in parallel, multiple jobs per SSV
n_parallel_jobs = global_params.NCORES_PER_NODE // global_params.NGPUS_PER_NODE
for ssv in ssvs_large:
    render_sso_coords_multiprocessing(ssvs_large, wd, n_parallel_jobs,
                                      render_indexviews=False, return_views=False,
                                      render_kwargs=render_kwargs)

# render small SSVs in parallel, one job per SSV
sso_kwargs = dict(version=version, create=False, working_dir=wd)
if len(ssvs_small) != 0:
    multi_params = [[ssv.id, ssv.sv_ids] for ssv in ssvs_small]
    multi_params = chunkify(multi_params, n_parallel_jobs)
    # list of SSV IDs and SSD parameters need to be given to a single QSUB job
    multi_params = [(ixs, sso_kwargs, render_kwargs) for ixs in multi_params]
    path_out = QSUB_script(
        multi_params, "render_views", suffix="_SSV{}".format(ssvs_small[0].id),
        n_cores=1, disable_batchjob=True,
        n_max_co_processes=n_parallel_jobs)
    folder_del = os.path.abspath(path_out + "/../")
    shutil.rmtree(folder_del, ignore_errors=True)
    # TODO: this call leads to an error -> investigate further
    # start_multiprocess_obj('render_views', [[ssv, render_kwargs] for ssv in ssvs_small],
    #                        nb_cpus=n_parallel_jobs)

with open(path_out_file, "wb") as f:
    pkl.dump("0", f)
