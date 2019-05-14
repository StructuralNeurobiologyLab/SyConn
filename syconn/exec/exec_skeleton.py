# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max-Planck-Institute of Neurobiology, Munich, Germany
# Authors: Philipp Schubert, Joergen Kornfeld

import numpy as np
from syconn.mp import batchjob_utils as qu
from syconn.reps.super_segmentation_dataset import SuperSegmentationDataset
from syconn.handler.basics import chunkify
from syconn.handler.config import initialize_logging
from syconn.proc.skel_based_classifier import SkelClassifier
from syconn.global_params import NCORES_PER_NODE
from syconn import global_params


def run_skeleton_generation(max_n_jobs=None):
    if max_n_jobs is None:
        max_n_jobs = global_params.NCORE_TOTAL * 2
    log = initialize_logging('skeleton_generation', global_params.config.working_dir + '/logs/',
                             overwrite=False)
    ssd = SuperSegmentationDataset(working_dir=global_params.config.working_dir)

    # TODO: think about using create_sso_skeleton_fast if underlying RAG
    #  obeys spatial correctness (> 10x faster)
    # list of SSV IDs and SSD parameters need to be given to a single QSUB job
    multi_params = ssd.ssv_ids
    nb_svs_per_ssv = np.array([len(ssd.mapping_dict[ssv_id])
                               for ssv_id in ssd.ssv_ids])
    ordering = np.argsort(nb_svs_per_ssv)
    multi_params = multi_params[ordering[::-1]]
    multi_params = chunkify(multi_params, max_n_jobs)

    # add ssd parameters
    multi_params = [(ssv_ids, ssd.version, ssd.version_dict, ssd.working_dir)
                    for ssv_ids in multi_params]

    # create SSV skeletons, requires SV skeletons!
    log.info('Starting skeleton generation of {} SSVs.'.format(
        len(ssd.ssv_ids)))
    qu.QSUB_script(multi_params, "export_skeletons_new",
                   n_max_co_processes=global_params.NCORE_TOTAL,
                   remove_jobfolder=True)

    log.info('Finished skeleton generation.')
    # # run skeleton feature extraction # Not needed anymore, will be kept in
    # case skeleton features should remain a feature of SyConn
    # qu.QSUB_script(multi_params, "preproc_skelfeature",
    #                    n_max_co_processes=global_params.NCORE_TOTAL,
    #                    remove_jobfolder=True)


def run_skeleton_axoness():
    # # run skeleton feature extraction # Not needed anymore, will be kept in
    # case skeleton features should remain a feature of SyConn
    sbc = SkelClassifier("axoness", working_dir=global_params.config.working_dir)
    ft_context = [1000, 2000, 4000, 8000, 12000]
    sbc.generate_data(feature_contexts_nm=ft_context, nb_cpus=NCORES_PER_NODE)
    sbc.classifier_production(ft_context, nb_cpus=NCORES_PER_NODE)

