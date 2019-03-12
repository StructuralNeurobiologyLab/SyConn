# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max-Planck-Institute for Medical Research, Heidelberg, Germany
# Authors: Sven Dorkenwald, Philipp Schubert, Jörgen Kornfeld

import sys
import numpy as np
try:
    import cPickle as pkl
except ImportError:
    import pickle as pkl
from syconn.reps.super_segmentation import SuperSegmentationObject
from syconn.proc.sd_proc import sos_dict_fact, init_sos
from syconn.proc.rendering import render_sso_coords, render_sso_coords_index_views
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
sso_kwargs = args[1]
working_dir = so_kwargs['working_dir']
ssv_id = so_kwargs['ssv_id']
global_params.wd = working_dir
kwargs = args[2]
render_indexviews = kwargs['render_indexviews']
del kwargs['render_indexviews']
sso = SuperSegmentationObject(working_dir)
ssv = sso.get_super_segmentation_object(ssv_id)
for coords in ch:
    if render_indexviews:
        views = render_sso_coords_index_views(sso, coords, **kwargs)
    else:
        views = render_sso_coords(sso, coords, **kwargs)

with open(path_out_file, "wb") as f:
    pkl.dump(views, f)
