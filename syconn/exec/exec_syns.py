# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max-Planck-Institute of Neurobiology, Munich, Germany
# Authors: Philipp Schubert, Joergen Kornfeld

from knossos_utils import knossosdataset
import numpy as np
knossosdataset._set_noprint(True)
from knossos_utils import chunky
from syconn.extraction import cs_extraction_steps as ces
from syconn import global_params
from syconn.reps.segmentation import SegmentationDataset
from syconn.proc.sd_proc import dataset_analysis
from syconn.proc.ssd_proc import map_synssv_objects
from syconn.extraction import cs_processing_steps as cps
from syconn.handler.config import initialize_logging
from syconn.handler.basics import kd_factory


def run_matrix_export():
    log = initialize_logging('synapse_analysis', global_params.config.working_dir + '/logs/',
                             overwrite=True)
    sd_syn_ssv = SegmentationDataset(working_dir=global_params.config.working_dir,
                                     obj_type='syn_ssv')

    # as an alternative to the skeletons, use vertex predictions or
    # sample_locations, ~3.5h @ 300 cpus
    # TODO: requires speed-up; one could collect properties only for synapses >
    #  probability threshold
    #     synssv_ids = synssv_ids[syn_prob > .5]
    #     ssv_partners = ssv_partners[syn_prob > .5]
    # One could also re-use the cached synssv IDs (computed during mapping of
    # synssv to SSVs) -> saves finding SSV ID indices in synapse arrays (->
    # slow for many synapses)
    cps.collect_properties_from_ssv_partners(global_params.config.working_dir,
                                             debug=True)
    #
    # collect new object attributes collected above partner axoness, celltypes,
    # synapse probabilities etc, no need to compute size/rep_coord etc. ->
    # recompute=False
    dataset_analysis(sd_syn_ssv, compute_meshprops=False,
                     recompute=False)
    log.info('Synapse property collection from SSVs finished.')

    # export_matrix
    log.info('Exporting connectivity matrix now.')
    dest_folder = global_params.config.working_dir + '/connectivity_matrix/'
    cps.export_matrix(dest_folder=dest_folder)
    log.info('Connectivity matrix was epxorted to "{}".'.format(dest_folder))


def run_syn_generation(chunk_size=(512, 512, 512), n_folders_fs=10000,
                       max_n_jobs=None, cube_of_interest_bb=None):
    """

    Parameters
    ----------
    chunk_size :
    n_folders_fs :
    max_n_jobs :
    cube_of_interest_bb : Tuple[np.ndarray]
        Defines the bounding box of the cube to process. By default this is
        set to (np.zoers(3); kd.boundary).

    Returns
    -------

    """
    if max_n_jobs is None:
        max_n_jobs = global_params.NCORE_TOTAL * 2

    log = initialize_logging('synapse_generation', global_params.config.working_dir + '/logs/',
                             overwrite=True)

    kd_seg_path = global_params.config.kd_seg_path
    kd = kd_factory(kd_seg_path)

    if cube_of_interest_bb is None:
        cube_of_interest_bb = [np.zeros(3, dtype=np.int), kd.boundary]

    ces.extract_contact_sites(chunk_size=chunk_size, log=log, max_n_jobs=max_n_jobs,
                              cube_of_interest_bb=cube_of_interest_bb,
                              n_folders_fs=n_folders_fs)
    log.info('SegmentationDataset of type "cs" and "syn" was generated.')

    # TODO: add check for SSD existence, which is required at this point
    # This creates an SD of type 'syn_ssv'
    cps.combine_and_split_syn(global_params.config.working_dir, resume_job=False,
                              cs_gap_nm=global_params.cs_gap_nm, log=log,
                              n_folders_fs=n_folders_fs)
    log.info('Synapse objects were created.')
    #
    sd_syn_ssv = SegmentationDataset(working_dir=global_params.config.working_dir,
                                     obj_type='syn_ssv')

    dataset_analysis(sd_syn_ssv, compute_meshprops=True)
    log.info('SegmentationDataset of type "syn_ssv" was generated.')

    cps.map_objects_to_synssv(global_params.config.working_dir, log=log)
    log.info('Cellular organelles were mapped to "syn_ssv".')

    cps.classify_synssv_objects(global_params.config.working_dir, log=log)
    log.info('Synapse property prediction finished.')

    log.info('Collecting and writing syn-ssv objects to SSV attribute '
             'dictionary.')
    # This needs to be run after `classify_synssv_objects` and before
    # `map_synssv_objects` if the latter uses thresholding for synaptic objects
    dataset_analysis(sd_syn_ssv, compute_meshprops=False, recompute=False)  # just collect new data
    # TODO: decide whether this should happen after prob thresholding or not
    map_synssv_objects(log=log)
    log.info('Finished.')
