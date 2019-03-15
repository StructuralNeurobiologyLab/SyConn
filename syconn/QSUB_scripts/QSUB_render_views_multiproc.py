# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max-Planck-Institute for Medical Research, Heidelberg, Germany
# Authors: Sven Dorkenwald, Philipp Schubert, Jörgen Kornfeld

import sys
import numpy as np
import time
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
#params = args[0]
#coords = params['coords']
coords = args[0]
sso_kwargs = args[1]
working_dir = sso_kwargs['working_dir']
global_params.wd = working_dir
kwargs = args[2]
render_indexviews = kwargs['render_indexviews']
del kwargs['render_indexviews']
sso = SuperSegmentationObject(**sso_kwargs)
views = 0  # in case no rendering locations are given
#index = args[3]
print("In QSUB script")
print(len(sso_kwargs))
if render_indexviews:
    print("In Render_index")
    del kwargs['add_cellobjects']
    del kwargs['clahe']
    del kwargs['wire_frame']
    del kwargs['cellobjects_only']
    del kwargs['return_rot_mat']
    views = render_sso_coords_index_views(sso, coords, **kwargs)
else:
    views = render_sso_coords(sso, coords, **kwargs)

with open(path_out_file, "wb") as f:
    pkl.dump(views, f)
