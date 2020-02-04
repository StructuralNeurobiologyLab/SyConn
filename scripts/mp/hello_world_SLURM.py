# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Philipp Schubert, Joergen Kornfeld
import os
import numpy as np
from syconn import global_params
global_params.wd = os.path.expanduser('~/SyConn/batchjob_test/')
from syconn.mp import batchjob_utils as bu
from syconn.handler.basics import chunkify, load_pkl2obj
import glob

# path to the folder containing the scripts
# "/your/qsub_script_folder/"
script_folder = os.path.abspath(os.path.dirname(__file__))

# get example arguments for our jobs (600 arrays of size 10)
params = np.arange(6000).reshape((-1, 10))
# Create a list of arguments; each element is input for an executed script.
# We have created 300 jobs, each with 2 arrays
params = chunkify(params, 30)
out_path = bu.batchjob_script([(ii, par) for ii, par in enumerate(params)], "print",
                              script_folder=script_folder,
                              n_max_co_processes=10)

res = np.zeros((len(params), ))
for ii, fname in enumerate(glob.glob(out_path)):
    out = load_pkl2obj(fname)
    res[ii] = out

assert np.all(res == np.arange(len(params)))
