# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max-Planck-Institute of Neurobiology, Munich, Germany
# Authors: Philipp Schubert, Joergen Kornfeld

from knossos_utils import knossosdataset
knossosdataset._set_noprint(True)
from knossos_utils import chunky
from syconn.extraction import object_extraction_wrapper as oew
from syconn.extraction import cs_processing_steps
from syconn.extraction import cs_extraction_steps as ces
from syconn import global_params
from syconn.reps.segmentation import SegmentationDataset
from syconn.proc.sd_proc import dataset_analysis, extract_synapse_type
from syconn.proc.ssd_proc import map_synssv_objects
from syconn.extraction import cs_processing_steps as cps
from syconn.handler.logger import initialize_logging


def run_syn_analysis():
    log = initialize_logging('synapse_analysis', global_params.config.working_dir + '/logs/',
                             overwrite=False)

    sd_syn_ssv = SegmentationDataset(working_dir=global_params.config.working_dir,
                                     obj_type='syn_ssv')

    # This will be replaced by the new method for the 'syn_ssv' generation, ~80 min @ 340 cpus
    extract_synapse_type(sd_syn_ssv, kd_sym_path=global_params.config.kd_sym_path, stride=100,
                         kd_asym_path=global_params.config.kd_asym_path, qsub_pe='openmp')
    log.info('Synapse type was mapped to "syn_ssv".')

    # ~1h
    cps.map_objects_to_synssv(global_params.config.working_dir, qsub_pe='openmp')
    log.info('Cellular organelles were mapped to "syn_ssv".')

    cps.classify_synssv_objects(global_params.config.working_dir, qsub_pe='openmp')
    log.info('Synapse property prediction finished.')

    # as an alternative to the skeletons, use vertex predictions or sample_locations, ~3.5h @ 300 cpus
    # TODO: requires speed-up; one could collect properties only for synapses > probability threshold
    cps.collect_properties_from_ssv_partners(global_params.config.working_dir, qsub_pe='openmp')

    # collect new object attributes collected above partner axoness, celltypes,
    # synapse probabilities etc, no need to compute size/rep_coord etc. -> recompute=False
    dataset_analysis(sd_syn_ssv, qsub_pe='openmp', compute_meshprops=True,
                     recompute=False)
    log.info('Synapse property collection from SSVs finished.')

    log.info('Collecting and writing syn-ssv objects to SSV attribute dictionary.')
    map_synssv_objects(qsub_pe='openmp')
    log.info('Finished.')

    # export_matrix
    log.info('Exporting connectivity matrix now.')
    dest_folder = global_params.config.working_dir + '/connectivity_matrix/'
    cps.export_matrix(dest_folder=dest_folder)
    log.info('Connectivity matrix was epxorted to "{}".'.format(dest_folder))


def run_syn_generation(chunk_size=(512, 512, 512)):
    log = initialize_logging('synapse_analysis', global_params.config.working_dir + '/logs/',
                             overwrite=False)

    kd_seg_path = global_params.config.kd_seg_path
    kd = knossosdataset.KnossosDataset()  # Sets initial values of object
    # Initializes the dataset by parsing the knossos.conf in path + "mag1"
    kd.initialize_from_knossos_path(kd_seg_path)

    # Initital contact site extraction
    cd_dir = global_params.config.working_dir + "/chunkdatasets/cs/"
    # Class that contains a dict of chunks (with coordinates) after initializing it
    cd = chunky.ChunkDataset()
    cd.initialize(kd, kd.boundary, chunk_size, cd_dir,
                  box_coords=[0, 0, 0], fit_box_size=True)

    # POPULATES CS CD with SV contacts
    ces.find_contact_sites(cd, kd_seg_path, n_max_co_processes=5000,
                          qsub_pe='default', qsub_queue='all.q')
    ces.extract_agg_contact_sites(cd, global_params.config.working_dir, n_folders_fs=10000, suffix="",
                                  qsub_queue='all.q', n_max_co_processes=5000,
                                  qsub_pe='default')
    log.info('CS extraction finished.')

    # create overlap dataset between SJ and CS: SegmentationDataset of type 'syn'
    # TODO: write new method which iterates over sj prob. map (KD), CS ChunkDataset / KD and (optionally) synapse type in parallel and to create a syn segmentation within from_probmaps_to_objects
    # TODO: SD for cs_agg and sj will not be needed anymore
    cs_sd = SegmentationDataset('cs_agg', working_dir=global_params.config.working_dir,
                                version=0)  # version hard coded
    sj_sd = SegmentationDataset('sj', working_dir=global_params.config.working_dir)
    cs_cset = chunky.load_dataset(cd_dir, update_paths=True)

    # # This creates an SD of type 'syn', currently ~6h, will hopefully be sped up after refactoring
    cs_processing_steps.syn_gen_via_cset(cs_sd, sj_sd, cs_cset, resume_job=False,
                                         nb_cpus=2, qsub_pe='openmp')
    sd = SegmentationDataset("syn", working_dir=global_params.config.working_dir, version="0")
    dataset_analysis(sd, qsub_pe='openmp', compute_meshprops=False)
    log.info('SegmentationDataset of type "syn" was generated.')

    # This creates an SD of type 'syn_ssv', ~15 min
    cps.combine_and_split_syn(global_params.config.working_dir, resume_job=False,
                              stride=250, qsub_pe='default', qsub_queue='all.q',
                              cs_gap_nm=global_params.cs_gap_nm,
                              n_max_co_processes=global_params.NCORE_TOTAL)
    sd_syn_ssv = SegmentationDataset(working_dir=global_params.config.working_dir,
                                     obj_type='syn_ssv')
    dataset_analysis(sd_syn_ssv, qsub_pe='openmp', compute_meshprops=True)
    log.info('SegmentationDataset of type "syn_ssv" was generated.')
