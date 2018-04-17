# SyConnFS
# Copyright (c) 2016 Philipp J. Schubert
# All rights reserved
import time
import sys
from syconn.reps.super_segmentation import \
    SuperSegmentationObject
from syconn.proc.rendering import render_sampled_sso
from syconn.config.global_params import get_dataset_scaling
import numpy as np

##########################################################################
# Helper script for rendering views of SSO mesh locations multiprocessed #
# Change SSO kwargs accordingly                                          #
##########################################################################
raise DeprecationWarning("SSO parameters have to be set automatically. Use"
                         " QSUB PE instead.")
args = sys.argv
cc_ix = np.array(args[1], dtype=np.int)
sso = SuperSegmentationObject(cc_ix, working_dir="/wholebrain/scratch/areaxfs/",
                              version="axgt", create=False,
                              scaling=get_dataset_scaling())
render_sampled_sso(sso, verbose=False)
