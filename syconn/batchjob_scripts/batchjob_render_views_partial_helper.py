# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max-Planck-Institute for Medical Research, Heidelberg, Germany
# Authors: Sven Dorkenwald, Philipp Schubert, Jörgen Kornfeld

import sys
import pickle as pkl
from syconn.reps.super_segmentation import render_sampled_sos_cc
from syconn.proc.sd_proc import sos_dict_fact, init_sos
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

svixs = args[0]
so_kwargs = args[1]
working_dir = so_kwargs['working_dir']
global_params.wd = working_dir
kwargs = args[2]
skip_indexviews = kwargs['skip_indexviews']
del kwargs['skip_indexviews']

# svixs is a list of SV IDs, i.e. a list of multiple SSVs
for svix in svixs:
    sd = sos_dict_fact(svix, **so_kwargs)
    sd['enable_locking'] = False  # locking is explicitly enabled when saving views
    sos = init_sos(sd)
    # render raw views
    render_sampled_sos_cc(sos, index_views=False, **kwargs)
    # now render with index views True.
    if not skip_indexviews:
        render_sampled_sos_cc(sos, index_views=True, **kwargs)

with open(path_out_file, "wb") as f:
    pkl.dump("0", f)
