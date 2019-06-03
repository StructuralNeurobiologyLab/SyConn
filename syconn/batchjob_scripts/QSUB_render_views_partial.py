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
import shutil
import os
from syconn import global_params
from syconn.handler.basics import chunkify
from syconn.mp.batchjob_utils import QSUB_script

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
so_kwargs = args[1]
working_dir = so_kwargs['working_dir']
global_params.wd = working_dir
kwargs = args[2]

n_parallel_jobs = global_params.NCORES_PER_NODE  # -> uses more threads than available (increase
# usage of GPU)
multi_params = ch
# this creates a list of lists of SV IDs
multi_params = chunkify(multi_params, n_parallel_jobs)
# list of SSV IDs and SSD parameters need to be given to a single QSUB job
multi_params = [(ixs, so_kwargs, kwargs) for ixs in multi_params]
# first SV should always be unique
path_out = QSUB_script(
    multi_params, "render_views_partial_helper", suffix="_SSV{}".format(ch[0][0]),
    n_cores=1, disable_batchjob=True, remove_jobfolder=True,
    n_max_co_processes=n_parallel_jobs, show_progress=False)
folder_del = os.path.abspath(path_out + "/../")
shutil.rmtree(folder_del, ignore_errors=True)

with open(path_out_file, "wb") as f:
    pkl.dump("0", f)
