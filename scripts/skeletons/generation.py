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
from syconn.handler.logger import initialize_logging


# ~1h for >90%, >15h for all
if __name__ == "__main__":
    log = initialize_logging('skeleton_generation', global_params.wd + '/logs/')
    ssd = SuperSegmentationDataset(working_dir=global_params.wd)

    # list of SSV IDs and SSD parameters need to be given to a single QSUB job
    multi_params = ssd.ssv_ids
    nb_svs_per_ssv = np.array([len(ssd.mapping_dict[ssv_id])
                               for ssv_id in ssd.ssv_ids])
    ordering = np.argsort(nb_svs_per_ssv)
    multi_params = multi_params[ordering[::-1]]
    multi_params = chunkify(multi_params, 4000)

    # add ssd parameters
    multi_params = [(ssv_ids, ssd.version, ssd.version_dict, ssd.working_dir)
                    for ssv_ids in multi_params]
    script_folder = os.path.dirname(os.path.abspath(__file__)) + \
                    "/../../syconn/QSUB_scripts/"
    kwargs = dict(n_max_co_processes=340, pe="openmp", queue=None,
                  script_folder=script_folder, suffix="")
    # create SSV skeletons, requires SV skeletons!
    log.info('Starting skeleton generation of {} SSVs.'.format(
        len(ssd.ssv_ids)))
    qu.QSUB_script(multi_params, "export_skeletons_new", **kwargs)

    log.info('Finished skeleton generation.')
    # # run skeleton feature extraction # Not needed anymore, will be kept in
    # case skeleton features should remain a feature of SyConn
    # qu.QSUB_script(multi_params, "preproc_skelfeature", **kwargs)
