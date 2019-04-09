# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max-Planck-Institute for Medical Research, Heidelberg, Germany
# Authors: Sven Dorkenwald, Philipp Schubert, Jörgen Kornfeld

import sys
import numpy as np
import networkx as nx
try:
    import cPickle as pkl
except ImportError:
    import pickle as pkl
from syconn.reps.super_segmentation import SuperSegmentationObject

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
for g in ch:
    # only generate SSV temporarily but do not write it to FS
    # corresponding SVs are parsed explicitly ('sv_ids=sv_ixs')
    sv_ixs = np.sort(list(g.nodes()))
    sso = SuperSegmentationObject(sv_ixs[0], working_dir=wd, version=version,
                                  create=False, sv_ids=sv_ixs,
                                  enable_locking_so=True)
    sso.load_attr_dict()
    # nodes of sso._rag need to be SV
    new_G = nx.Graph()
    for e in g.edges():
        new_G.add_edge(sso.get_seg_obj("sv", e[0]),
                       sso.get_seg_obj("sv", e[1]))
    sso._rag = new_G
    sso.render_views(add_cellobjects=False, woglia=False, overwrite=True,
                     skip_indexviews=True)

with open(path_out_file, "wb") as f:
    pkl.dump("0", f)
