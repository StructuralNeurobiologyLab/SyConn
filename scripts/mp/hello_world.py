# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Philipp Schubert, Joergen Kornfeld
import os
import numpy as np
from syconn.mp import qsub_utils as mu
from syconn.handler.basics import chunkify


# path to the folder containing the scripts
# "/your/qsub_script_folder/"
script_folder = os.path.abspath(os.path.dirname(__file__))

# get example arguments for our jobs (600 arrays of size 10)
params = np.arange(6000).reshape((-1, 10))
# Create a list of arguments; each element is input for an executed script.
# We have created 300 jobs, each with 2 arrays
params = chunkify(params, 300)
mu.QSUB_script(params, "print", pe="openmp", queue=None,
               script_folder=script_folder, n_max_co_processes=40)
