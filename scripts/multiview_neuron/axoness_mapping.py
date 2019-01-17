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
    log = initialize_logging('axon_mapping', global_params.wd + '/logs/',
                             overwrite=False)
    pred_key_appendix = ""
    # Working directory has to be changed globally in global_params
    ssd = SuperSegmentationDataset(working_dir=global_params.wd)

    multi_params = np.array(ssd.ssv_ids, dtype=np.uint)
    # sort ssv ids according to their number of SVs (descending)
    nb_svs_per_ssv = np.array([len(ssd.mapping_dict[ssv_id]) for ssv_id
                               in ssd.ssv_ids])
    multi_params = multi_params[np.argsort(nb_svs_per_ssv)[::-1]]
    multi_params = chunkify(multi_params, 2000)

    multi_params = [(par, pred_key_appendix) for par in multi_params]
    log.info('Starting axoness mapping.')
    path_to_out = qu.QSUB_script(multi_params, "map_viewaxoness2skel",
                                 n_max_co_processes=global_params.NCORE_TOTAL,
                                 pe="openmp", queue=None, suffix="", n_cores=1)
    # TODO: perform completeness check
    log.info('Finished axoness mapping.')
