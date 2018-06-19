# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max-Planck-Institute for Medical Research, Heidelberg, Germany
# Authors: Sven Dorkenwald, Philipp Schubert, Jörgen Kornfeld

import sys
try:
    import cPickle as pkl
# TODO: switch to Python3 at some point and remove above
except Exception:
    import pickle as pkl
from syconn.reps.super_segmentation import SuperSegmentationObject
from syconn.proc.sd_proc import sos_dict_fact, init_sos
from syconn.config.global_params import wd, get_dataset_scaling
import networkx as nx
import numpy as np
import warnings

path_storage_file = sys.argv[1]
path_out_file = sys.argv[2]

with open(path_storage_file, 'rb') as f:
    args = []
    while True:
        try:
            args.append(pkl.load(f))
        except EOFError:
            break

for cc in args:
    svixs = cc.nodes()
    cc_ix = np.min(svixs)
    sso = SuperSegmentationObject(cc_ix, version="gliaremoval", nb_cpus=2,
                                  working_dir=wd, create=True,
                                  scaling=get_dataset_scaling(), sv_ids=svixs)
    so_cc = nx.Graph()
    for e in cc.edges_iter():
        so_cc.add_edge(sso.get_seg_obj("sv", e[0]),
                       sso.get_seg_obj("sv", e[1]))
    sso._rag = so_cc
    sd = sos_dict_fact(svixs)
    sos = init_sos(sd)
    sso._objects["sv"] = sos
    if len(sso.svs) > 3e5:
        warnings.warn("Skipped huge SSV %d." % sso.id)
    try:
        sso.gliasplit(verbose=False)
    except Exception as e:
        print("\n--------------------------------------------------------\n"
              "Splitting of SSV %d failed with %s."
              "\n--------------------------------------------------------\n" % (
              cc_ix, e))
