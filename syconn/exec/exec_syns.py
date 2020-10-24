# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max-Planck-Institute of Neurobiology, Munich, Germany
# Authors: Philipp Schubert, Joergen Kornfeld

from typing import Tuple, Optional, Union

import numpy as np

from syconn import global_params
from syconn.extraction import cs_extraction_steps as ces
from syconn.extraction import cs_processing_steps as cps
from syconn.handler.basics import kd_factory, chunkify
from syconn.handler.config import initialize_logging
from syconn.mp.batchjob_utils import batchjob_script
from syconn.proc.sd_proc import dataset_analysis
from syconn.proc.ssd_proc import map_synssv_objects
from syconn.reps.segmentation import SegmentationDataset
from syconn.reps.super_segmentation import SuperSegmentationDataset


def run_matrix_export():
    """
    Export the matrix as a ``.csv`` file at the ``connectivity_matrix`` folder
    of the currently active working directory.
    Also collects the following synapse properties from prior analysis steps:

        * 'partner_axoness': Cell compartment type (axon: 1, dendrite: 0, soma: 2,
          en-passant bouton: 3, terminal bouton: 4) of the partner neurons.
        * 'partner_spiness': Spine compartment predictions (0: dendritic shaft,
          1: spine head, 2: spine neck, 3: other) of both neurons.
        * 'partner_celltypes': Celltype of the both neurons.
        * 'latent_morph': Local morphology embeddings of the pre- and post-
          synaptic partners.

    Examples:
        See :class:`~syconn.reps.segmentation.SegmentationDataset` for examples.
    """
    # cache cell attributes
    ssd = SuperSegmentationDataset(working_dir=global_params.config.working_dir)
    ssd.save_dataset_deep()
    log = initialize_logging('matrix_export', global_params.config.working_dir + '/logs/', overwrite=True)

    sd_syn_ssv = SegmentationDataset(working_dir=global_params.config.working_dir, obj_type='syn_ssv')

    cps.collect_properties_from_ssv_partners(global_params.config.working_dir, debug=False)
    #
    # collect new object attributes collected above partner axoness, celltypes, synapse probabilities etc,
    # no need to compute size/rep_coord etc. -> recompute=False
    dataset_analysis(sd_syn_ssv, compute_meshprops=False, recompute=False)
    log.info('Synapse property collection from SSVs finished.')

    # export_matrix
    log.info('Exporting connectivity matrix now.')
    dest_folder = global_params.config.working_dir + '/connectivity_matrix/'
    cps.export_matrix(log=log, dest_folder=dest_folder)
    log.info('Connectivity matrix was exported to "{}".'.format(dest_folder))


def run_syn_generation(chunk_size: Optional[Tuple[int, int, int]] = (512, 512, 512), n_folders_fs: int = 10000,
                       max_n_jobs: Optional[int] = None,
                       cube_of_interest_bb: Union[Optional[np.ndarray], tuple] = None):
    """
    Run the synapse generation. Will create
    :class:`~syconn.reps.segmentation.SegmentationDataset` objects with
    the following versions:

        * 'cs': Contact site objects between supervoxels.
        * 'syn': Objects representing the overlap between 'cs' and the initial
          synaptic junction predictions. Note: These objects effectively represent
          synapse fragments between supervoxels.
        * 'syn_ssv': Agglomerated 'syn' objects based on the supervoxel graph.

    Args:
        chunk_size: The size of processed cubes.
        n_folders_fs: Number of folders used to create the folder structure in
            each :class:`~syconn.reps.segmentation.SegmentationDataset`.
        max_n_jobs: Number of parallel jobs.
        cube_of_interest_bb: Defines the bounding box of the cube to process.
            By default this is set to (np.zoers(3); kd.boundary).
    """
    if max_n_jobs is None:
        max_n_jobs = global_params.config.ncore_total * 2

    log = initialize_logging('synapse_detection', global_params.config.working_dir + '/logs/',
                             overwrite=True)

    kd_seg_path = global_params.config.kd_seg_path
    kd = kd_factory(kd_seg_path)

    if cube_of_interest_bb is None:
        cube_of_interest_bb = [np.zeros(3, dtype=np.int), kd.boundary]

    # create KDs and SDs for syn and cs
    ces.extract_contact_sites(chunk_size=chunk_size, log=log, max_n_jobs=max_n_jobs,
                              cube_of_interest_bb=cube_of_interest_bb,
                              n_folders_fs=n_folders_fs)
    log.info('SegmentationDataset of type "cs" and "syn" was generated.')

    # create SD of type 'syn_ssv'
    cps.combine_and_split_syn(global_params.config.working_dir,
                              cs_gap_nm=global_params.config['cell_objects']['cs_gap_nm'],
                              log=log, n_folders_fs=n_folders_fs)

    sd_syn_ssv = SegmentationDataset(working_dir=global_params.config.working_dir,
                                     obj_type='syn_ssv')

    # recompute=False: size, bounding box, rep_coord and mesh properties
    # have already been processed in combine_and_split_syn
    dataset_analysis(sd_syn_ssv, compute_meshprops=False, recompute=False)
    syn_sign = sd_syn_ssv.load_cached_data('syn_sign')
    n_sym = np.sum(syn_sign == -1)
    n_asym = np.sum(syn_sign == 1)
    del syn_sign
    log.info(f'SegmentationDataset of type "syn_ssv" was generated with {len(sd_syn_ssv.ids)} '
             f'objects, {n_sym} symmetric and {n_asym} asymmetric.')
    assert n_sym + n_asym == len(sd_syn_ssv.ids)

    cps.map_objects_from_synssv_partners(global_params.config.working_dir, log=log)
    log.info('Cellular organelles were mapped to "syn_ssv".')

    cps.classify_synssv_objects(global_params.config.working_dir, log=log)
    log.info('Synapse prediction finished.')

    log.info('Collecting and writing syn_ssv objects to SSV attribute '
             'dictionary.')
    # This needs to be run after `classify_synssv_objects` and before
    # `map_synssv_objects` if the latter uses thresholding for synaptic objects
    # just collect new data: ``recompute=False``
    dataset_analysis(sd_syn_ssv, compute_meshprops=False, recompute=False)
    map_synssv_objects(log=log)
    log.info('Finished.')


def run_spinehead_volume_calc():
    """
    Calculate spine head volumes based on a watershed segmentation which is run on 3D spine label masks propagated
    from cell surface predictions.
    """
    log = initialize_logging('compartment_prediction', global_params.config.working_dir + '/logs/',
                             overwrite=False)
    ssd = SuperSegmentationDataset(working_dir=global_params.config.working_dir)
    log.info('Starting spine head volume calculation.')
    multi_params = ssd.ssv_ids[np.argsort(ssd.load_cached_data('size'))[::-1]]
    multi_params = chunkify(multi_params, global_params.config.ncore_total * 4)
    multi_params = [(ixs,) for ixs in multi_params]

    batchjob_script(multi_params, "calculate_spinehead_volume", log=log, remove_jobfolder=True)
    log.info(f'Finished processing of {len(ssd.ssv_ids)} SSVs.')
