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
from syconnfs.representations import super_segmentation_helper as ssh

path_storage_file = sys.argv[1]
path_out_file = sys.argv[2]

with open(path_storage_file) as f:
    args = []
    while True:
        try:
            args.append(pkl.load(f))
        except:
            break

cs_list = args
for cs in cs_list:
    try:
        ssh.map_cs_properties(cs)
    except KeyError as e:
        print "Error %s during cs property mapping with key %d." % (e, cs.id)
        pass
print "Finished mapping of %d CS." % len(cs_list)

