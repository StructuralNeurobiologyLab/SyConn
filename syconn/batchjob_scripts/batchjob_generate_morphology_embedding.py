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
from syconn.reps.super_segmentation_dataset import exctract_ssv_morphology_embedding, SuperSegmentationDataset
from syconn.handler.prediction_pts import infere_cell_morphology_ssd
from syconn import global_params
from syconn.handler import basics
from syconn.mp.mp_utils import start_multiprocess_imap

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
if global_params.config.use_point_models:
    ssd_kwargs = dict(working_dir=ssd.working_dir, config=ssd.config)
    ssv_params = [dict(ssv_id=ssv_id, **ssd_kwargs) for ssv_id in ssv_ids]
    infere_cell_morphology_ssd(ssv_params)
else:
    ncpus = global_params.config['ncores_per_node'] // global_params.config['ngpus_per_node']
    n_worker = 2
    params = basics.chunkify(ssv_ids, n_worker * 4)
    res = start_multiprocess_imap(exctract_ssv_morphology_embedding,
                                  [(p, ncpus, pred_key_appendix) for p in params],
                                  nb_cpus=n_worker)

with open(path_out_file, "wb") as f:
    pkl.dump("0", f)
