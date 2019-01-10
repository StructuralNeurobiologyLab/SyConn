# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max-Planck-Institute of Neurobiology, Munich, Germany
# Authors: Philipp Schubert, Joergen Kornfeld

from syconn.config import global_params
from syconn.reps.segmentation import SegmentationDataset
from syconn.proc.sd_proc import dataset_analysis, extract_synapse_type
from syconn.extraction import cs_processing_steps as cps
from syconn.handler.logger import initialize_logging


if __name__ == '__main__':
    log = initialize_logging('synapse_analysis', global_params.wd + '/logs/')

    kd_sym_path = global_params.kd_sym_path
    kd_asym_path = global_params.kd_asym_path

    sd_syn_ssv = SegmentationDataset(working_dir=global_params.wd,
                                     obj_type='syn_ssv')
    # # This will be replaced by the new method for syn_ssv generation
    extract_synapse_type(sd_syn_ssv, kd_sym_path=kd_sym_path, stride=100,
                         kd_asym_path=kd_asym_path, qsub_pe='openmp')
    log.info('Synapse type was mapped to "syn_ssv".')

    cps.map_objects_to_synssv(global_params.wd)
    log.info('Cellular organelles were mapped to "syn_ssv".')

    cps.classify_synssv_objects(global_params.wd,)# qsub_pe='openmp')
    log.info('Synapse property prediction finished.')

    # # as an alternative to the skeletons, use vertex predictions for that
    cps.collect_properties_from_ssv_partners(global_params.wd, qsub_pe='openmp')

    # collect new object attributes like partner axoness, celltypes,
    # synapse probabilities etc, -> recompute=False
    dataset_analysis(sd_syn_ssv, qsub_pe='openmp', compute_meshprops=True,
                     stride=100, recompute=False)
    log.info('Synapse property collection from SSVs finished.')

    # export_matrix
    dest_p = global_params.wd + '/connectivity_matrix/conn_mat'
    cps.export_matrix(global_params.wd, dest_p)
    log.info('Connectivity matrix was epxorted to "{}".'.format(dest_p))
