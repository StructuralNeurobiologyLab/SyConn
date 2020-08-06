# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max-Planck-Institute for Medical Research, Heidelberg, Germany
# Authors: Philipp Schubert, Jörgen Kornfeld

import sys
try:
    import cPickle as pkl
except ImportError:
    import pickle as pkl
from syconn.proc.rendering import _render_views_multiproc
from syconn import global_params
from syconn.reps.super_segmentation_object import SuperSegmentationObject

path_storage_file = sys.argv[1]
path_out_file = sys.argv[2]

with open(path_storage_file, 'rb') as f:
    args = []
    while True:
        try:
            args.append(pkl.load(f))
        except EOFError:
            break

coords = args[0]
sso_kwargs = args[1]
working_dir = sso_kwargs['working_dir']
global_params.wd = working_dir
kwargs = args[2]

sso = SuperSegmentationObject(**sso_kwargs)

args = [coords, sso, kwargs]

views = _render_views_multiproc(args)

with open(path_out_file, "wb") as f:
    pkl.dump(views, f)
