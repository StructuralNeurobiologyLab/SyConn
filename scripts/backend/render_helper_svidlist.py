# SyConnFS
# Copyright (c) 2016 Philipp J. Schubert
# All rights reserved
import time
while True:
    # import error might occur when saving project...
    try:
        import sys
        from syconnfs.representations.segmentation_helper import init_sos, sos_dict_fact
        from syconnfs.representations.super_segmentation import render_sampled_sos_cc
        import numpy as np
        import warnings
        break
    except ImportError:
        time.sleep(0.2)

##########################################################################
# Helper script for rendering views of SV mesh locations multiprocessed  #
# Change SOs kwargs accordingly                                          #
##########################################################################
try:
    so_kwargs = {}  # <-- change SO kwargs here
    args = sys.argv
    svixs = np.array(args[1:], dtype=np.int)
    sd = sos_dict_fact(svixs, **so_kwargs)
    sos = init_sos(sd)
    render_sampled_sos_cc(sos, render_first_only=True, woglia=True,
                          verbose=False, add_cellobjects=True, overwrite=False,
                          cellobjects_only=False)
except Exception as e:
    print "Error occured:", e
    raise(e)

