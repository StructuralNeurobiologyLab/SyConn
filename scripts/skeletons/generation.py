# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Philipp Schubert, Sven Dorkenwald, Joergen Kornfeld

import numpy as np
import os
from syconn.mp import qsub_utils as qu
from syconn.reps.super_segmentation_dataset import SuperSegmentationDataset
from syconn.handler.basics import chunkify
from syconn.config import global_params

if __name__ == "__main__":
    ssd = SuperSegmentationDataset(working_dir=global_params.wd)
    # TODO: Use this as template for SSD based QSUB jobs
    multi_params = ssd.ssv_ids
    np.random.shuffle(multi_params)
    multi_params = chunkify(multi_params, 4000)
    # add ssd parameters
    multi_params = [(ssv_ids, ssd.version, ssd.version_dict, ssd.working_dir)
                    for ssv_ids in multi_params]
    script_folder = os.path.dirname(os.path.abspath(__file__)) + \
                    "/../../syconn/QSUB_scripts/"
    kwargs = dict(n_max_co_processes=200, pe="openmp", queue=None,
                  script_folder=script_folder, suffix="")
    # create SSV skeletons, requires SV skeletons!
    qu.QSUB_script(multi_params, "export_skeletons_new", **kwargs)

    # run skeleton feature extraction
    qu.QSUB_script(multi_params, "preproc_skelfeature", **kwargs)
