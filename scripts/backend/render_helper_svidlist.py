# SyConnFS
# Copyright (c) 2016 Philipp J. Schubert
# All rights reserved
import time
import sys
from syconn.config.global_params import wd
from syconn.proc.sd_proc import sos_dict_fact, init_sos
from syconn.reps.super_segmentation import render_sampled_sos_cc
import numpy as np
import warnings
##########################################################################
# Helper script for rendering views of SV mesh locations multiprocessed  #
# Change SOs kwargs accordingly                                          #
##########################################################################
# try:
args = sys.argv
so_version = args[1]
so_kwargs = {'version': so_version, 'working_dir': wd}
svixs = np.array(args[2:], dtype=np.uint)
sd = sos_dict_fact(svixs, **so_kwargs)
sos = init_sos(sd)
render_sampled_sos_cc(sos, render_first_only=True, woglia=True,
                      verbose=False, add_cellobjects=True, overwrite=False,
                      cellobjects_only=False)
# except Exception as e:
#     print "Error occured:", e

