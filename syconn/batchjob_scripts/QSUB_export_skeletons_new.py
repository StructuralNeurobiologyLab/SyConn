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
from syconn.reps.super_segmentation_helper import create_sso_skeletons_wrapper
from syconn.reps.super_segmentation import SuperSegmentationDataset

path_storage_file = sys.argv[1]
path_out_file = sys.argv[2]

with open(path_storage_file, 'rb') as f:
    args = []
    while True:
        try:
            args.append(pkl.load(f))
        except EOFError:
            break

ssv_ids = args[0]
version = args[1]
version_dict = args[2]
working_dir = args[3]
map_myelin = args[4]

ssd = SuperSegmentationDataset(working_dir=working_dir, version=version,
                               version_dict=version_dict)
ssvs = ssd.get_super_segmentation_object(ssv_ids)
create_sso_skeletons_wrapper(ssvs, map_myelin=map_myelin)


with open(path_out_file, "wb") as f:
    pkl.dump("0", f)
