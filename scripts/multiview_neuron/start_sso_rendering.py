# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Philipp Schubert, Sven Dorkenwald, Joergen Kornfeld
import os
import numpy as np

from syconn.config import global_params
from syconn.mp import qsub_utils as qu
from syconn.reps.super_segmentation_dataset import SuperSegmentationDataset
from syconn.reps.super_segmentation_helper import find_incomplete_ssv_views
from syconn.handler.basics import chunkify
from syconn.handler.logger import initialize_logging

# Using OSMesa:
# ~5h for 'small' SSV (<5000)
# ~24h for one huge SSV (14k SV, most likely including huge SV)
if __name__ == "__main__":
    log = initialize_logging('neuron_view_rendering',
                             global_params.wd + '/logs/')
    # TODO: currently working directory has to be set globally in global_params
    #  and is not adjustable here because all qsub jobs will start a script
    #  referring to 'global_params.wd'
    # view rendering prior to glia removal, choose SSD accordingly
    ssd = SuperSegmentationDataset(working_dir=global_params.wd)

    #  TODO: use actual size criteria, e.g. number of sampling locations
    nb_svs_per_ssv = np.array([len(ssd.mapping_dict[ssv_id])
                               for ssv_id in ssd.ssv_ids])

    # render normal size SSVs
    size_mask = nb_svs_per_ssv <= global_params.RENDERING_MAX_NB_SV
    multi_params = ssd.ssv_ids[size_mask]

    # TODO: Currently slow if SSV contains very large SV(s)
    # sort ssv ids according to their number of SVs (descending)
    ordering = np.argsort(nb_svs_per_ssv[size_mask])
    multi_params = multi_params[ordering[::-1]]
    multi_params = chunkify(multi_params, 2000)
    # list of SSV IDs and SSD parameters need to be given to a single QSUB job
    multi_params = [(ixs, global_params.wd) for ixs in multi_params]
    log.info('Start rendering of {} SSVs. {} huge SSVs will be rendered '
             'afterwards using the whole cluster.'.format(np.sum(size_mask),
                                                          np.sum(~size_mask)))
    # generic
    path_to_out = qu.QSUB_script(multi_params, "render_views", pe="openmp",
                                 n_max_co_processes=global_params.NCORE_TOTAL,
                                 script_folder=None, suffix="", queue=None)
    log.info('Finished rendering of {}/{} SSVs.'.format(len(ordering),
                                                        len(nb_svs_per_ssv)))
    # identify huge SSVs and process them individually on whole cluster
    big_ssv = ssd.ssv_ids[~size_mask]
    for kk, ssv_id in enumerate(big_ssv):
        ssv = ssd.get_super_segmentation_object(ssv_id)
        log.info("Processing SSV [{}/{}] with {} SVs on whole cluster.".format(
            kk+1, len(big_ssv), len(ssv.sv_ids)))
        ssv.render_views(add_cellobjects=True, cellobjects_only=False,
                         woglia=True, qsub_pe="openmp", overwrite=True,
                         qsub_co_jobs=global_params.NCORE_TOTAL,
                         skip_indexviews=False, resume_job=False)
    log.info('Finished rendering of all SSVs. Checking completeness.')
    res = find_incomplete_ssv_views(ssd, woglia=True, n_cores=10)
    if len(res) != 0:
        msg = "Not all SSVs were rendered completely! Missing:\n{}".format(res)
        log.error(msg)
        raise RuntimeError(msg)
    else:
        log.info('Success.')
