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
from syconn.proc.sd_proc import sos_dict_fact, init_sos
from syconn import global_params
import networkx as nx
import numpy as np


path_storage_file = sys.argv[1]
path_out_file = sys.argv[2]

with open(path_storage_file, 'rb') as f:
    args = []
    while True:
        try:
            args.append(pkl.load(f))
        except EOFError:
            break

scaling = global_params.config['scaling']
# TODO: This coulb be cunked by loading `mesh_bb` and glia prob. prediction cache arrays
#  (might have to be create via `dataset_analysis`)
for cc in args:
    svixs = list(cc.nodes())
    cc_ix = np.min(svixs)
    sso = SuperSegmentationObject(cc_ix, version="gliaremoval", nb_cpus=1,
                                  working_dir=global_params.config.working_dir,
                                  create=True, scaling=scaling,
                                  sv_ids=svixs)
    so_cc = nx.Graph()
    for e in cc.edges():
        so_cc.add_edge(sso.get_seg_obj("sv", e[0]),
                       sso.get_seg_obj("sv", e[1]))
    sso._rag = so_cc
    sd = sos_dict_fact(svixs)
    sos = init_sos(sd)
    sso._objects["sv"] = sos
    sso.load_attr_dict()
    sso.gliasplit(verbose=False, recompute=False)

with open(path_out_file, "wb") as f:
    pkl.dump("0", f)
