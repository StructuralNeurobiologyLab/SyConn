# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max-Planck-Institute for Medical Research, Heidelberg, Germany
# Authors: Sven Dorkenwald, Philipp Schubert, Jörgen Kornfeld

import sys
import cPickle as pkl
from syconnfs.representations.super_segmentation import SuperSegmentationObject
from syconnfs.representations.segmentation_helper import sos_dict_fact, init_sos
import networkx as nx
import numpy as np

path_storage_file = sys.argv[1]
path_out_file = sys.argv[2]

with open(path_storage_file) as f:
    args = []
    while True:
        try:
            args.append(pkl.load(f))
        except:
            break

for cc in args:
    svixs = cc.nodes()
    cc_ix = np.min(svixs)
    # try:
    sso = SuperSegmentationObject(cc_ix, version="tmp", nb_cpus=20,
                                  working_dir="/wholebrain/scratch/areaxfs/",
                                  create=False, scaling=(10, 10, 20))
    sso.version_dict["sv"] = "0"
    # if sso.load_attr_dict() == -1:
    sso.attr_dict["sv"] = svixs
    so_cc = nx.Graph()
    for e in cc.edges_iter():
        so_cc.add_edge(sso.get_seg_obj("sv", e[0]),
                       sso.get_seg_obj("sv", e[1]))
    sso._edge_graph = so_cc
    # sso.save_attributes(["sv"], [svixs])
    sd = sos_dict_fact(svixs)
    sos = init_sos(sd)
    sso._objects["sv"] = sos
    if len(sso.svs) > 1e5:
        print "Skipped huge SSV %d." % sso.id
        continue
    try:
        sso.gliasplit(thresh=0.161489, verbose=False)
    except Exception, e:
        print "\n--------------------------------------------------------\n" \
              "Splitting of SSV %d failed with %s." \
              "\n--------------------------------------------------------\n" % (
              cc_ix, e)
