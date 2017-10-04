# SyConnMP
# Copyright (c) 2016 Philipp J. Schubert
# All rights reserved
import os
import numpy as np
from syconnmp import qsub_utils as qu
from syconn.handler.basics import chunkify


# path to the folder containing the scripts
# "/your/qsub_script_folder/"
script_folder = os.path.abspath(os.path.dirname(__file__))

# get example arguments for our jobs (20 arrays of size 10)
params = np.arange(200).reshape((-1, 10))
# Create a list of arguments; each element is input for an executed script.
# We have created 10 jobs, each with 2 arrays
params = chunkify(params, 10)
print params[:2]
qu.QSUB_script(params, "print", pe="openmp", queue=None,
               script_folder=script_folder, n_max_co_processes=10)