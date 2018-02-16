# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max-Planck-Institute for Medical Research, Heidelberg, Germany
# Authors: Sven Dorkenwald, Philipp Schubert, Jörgen Kornfeld

import sys
import numpy as np
import cPickle as pkl
from syconn.reps.super_segmentation_helper import sparsify_skeleton, create_sso_skeleton
from syconn.reps.super_segmentation_object import SuperSegmentationObject
from scipy.misc import imsave

path_storage_file = sys.argv[1]
path_out_file = sys.argv[2]

with open(path_storage_file) as f:
    args = []
    while True:
        try:
            args.append(pkl.load(f))
        except:
            break

ssv_ixs = args
for ix in ssv_ixs:
    sso = SuperSegmentationObject(ix, version="0", working_dir="/wholebrain/scratch/areaxfs3/")

    if sso.size > 1e5:
        filename = '/wholebrain/scratch/jkornfeld/ssv_gallery/size_{0}_ssv_{1}.png'.format(sso.size, ix)
        views = sso.render_ortho_views(dest_folder='')
        # todo: combine views in a single image file
        imsave(filename, np.hstack(views))
    print("Rendered ortho views for SSV", ix)