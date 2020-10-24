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
from syconn.reps.super_segmentation_dataset import SuperSegmentationDataset
from syconn.handler.prediction_pts import infere_cell_morphology_ssd
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

ssv_ids = args[0]
pred_key_appendix = args[1]

ssd = SuperSegmentationDataset()
ssd_kwargs = dict(working_dir=ssd.working_dir, config=ssd.config)

mpath = global_params.config.mpath_tnet_pts_wholecell

if global_params.config.use_point_models:
    ssv_params = [dict(ssv_id=ssv_id, **ssd_kwargs) for ssv_id in ssv_ids]
    infere_cell_morphology_ssd(ssv_params, mpath=mpath, pred_key_appendix=pred_key_appendix)
else:
    raise NotImplementedError('Whole cell embeddings are not available with multi-view approach.')

with open(path_out_file, "wb") as f:
    pkl.dump("0", f)
