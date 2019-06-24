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
if type(sso_kwargs) is str:  # only the working directory is given
    sso_kwargs = dict(working_dir=sso_kwargs, enable_locking_so=False)
if len(args) == 3:
    render_kwargs = args[2]
else:
    render_kwargs = dict(add_cellobjects=True, woglia=True, overwrite=True)
for ssv_ix in ch:
    if not np.isscalar(ssv_ix):
        sv_ids = ssv_ix[1]
        ssv_ix = ssv_ix[0]
    else:
        sv_ids = None  # will be loaded from attribute dict
    sso = SuperSegmentationObject(ssv_ix, sv_ids=sv_ids, **sso_kwargs)
    sso.load_attr_dict()
    sso.render_views(**render_kwargs)

with open(path_out_file, "wb") as f:
    pkl.dump("0", f)
