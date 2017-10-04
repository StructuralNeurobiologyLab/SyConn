# SyConnFS
# Copyright (c) 2016 Philipp J. Schubert
# All rights reserved
import time
while True:
    # import error might occur when saving project...
    try:
        import sys
        from syconnfs.representations.super_segmentation import \
            SuperSegmentationObject
        from syconnfs.processing.rendering import render_sampled_sso
        import numpy as np
        break
    except ImportError:
        time.sleep(0.2)

##########################################################################
# Helper script for rendering views of SSO mesh locations multiprocessed #
# Change SSO kwargs accordingly                                          #
##########################################################################
args = sys.argv
cc_ix = np.array(args[1], dtype=np.int)
sso = SuperSegmentationObject(cc_ix, working_dir="/wholebrain/scratch/areaxfs/",
                              version="axgt", create=False, scaling=(10, 10, 20))
render_sampled_sso(sso, verbose=False)
