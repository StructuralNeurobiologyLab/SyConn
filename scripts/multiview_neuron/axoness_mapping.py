# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Philipp Schubert, Sven Dorkenwald, Joergen Kornfeld
import os
import numpy as np

from syconn.mp import qsub_utils as qu
from syconn.reps.super_segmentation_dataset import SuperSegmentationDataset
from syconn.handler.basics import chunkify
from syconn.config import global_params
from syconn.handler.logger import initialize_logging


if __name__ == "__main__":
    """Maps axon prediction of rendering locations onto SSV skeletons"""
    log = initialize_logging('axon_mapping', global_params.wd + '/logs/')
    # TODO: currently working directory has to be set globally in global_params and is not adjustable here because all qsub jobs will start a script referring to 'global_params.wd'
    pred_key_appendix = ""
    script_folder = os.path.dirname(os.path.abspath(__file__)) + "/../../syconn/QSUB_scripts/"
    ssds = SuperSegmentationDataset(working_dir=global_params.wd)
    multi_params = ssds.ssv_ids
    np.random.shuffle(multi_params)
    multi_params = chunkify(multi_params, 2000)
    multi_params = [(par, pred_key_appendix) for par in multi_params]
    log('Starting axoness mapping.')
    path_to_out = qu.QSUB_script(multi_params, "map_viewaxoness2skel",
                                 n_max_co_processes=280, pe="openmp", queue=None,
                                 script_folder=script_folder, suffix="", n_cores=1)
    # TODO: perform completeness check
    log.info('Finished axoness mapping.')