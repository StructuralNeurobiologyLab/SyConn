# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max-Planck-Institute of Neurobiology, Munich, Germany
# Authors: Philipp Schubert, Joergen Kornfeld

from knossos_utils import knossosdataset
from knossos_utils import chunky
from syconn.config import global_params
from syconn.extraction import object_extraction_wrapper as oew
from syconn.extraction import cs_processing_steps
from syconn.reps.super_segmentation import SuperSegmentationDataset
from syconn.reps.segmentation import SegmentationDataset
from syconn.proc.sd_proc import dataset_analysis, extract_synapse_type
from syconn.extraction import cs_extraction_steps as ces
from syconn.extraction import cs_processing_steps as cps
from syconn.handler.logger import initialize_logging


if __name__ == "__main__":
    log = initialize_logging('synapse_analysis', global_params.wd + '/logs/',
                             overwrite=False)

    kd_seg_path = global_params.kd_seg_path
    # kd = knossosdataset.KnossosDataset()  # Sets initial values of object
    # # Initializes the dataset by parsing the knossos.conf in path + "mag1"
    # kd.initialize_from_knossos_path(kd_seg_path)

    # TODO: change path of CS chunkdataset
    # Initital contact site extraction
    cd_dir = global_params.wd + "/chunkdatasets/"
    # Class that contains a dict of chunks (with coordinates) after initializing it
    cd = chunky.ChunkDataset()
    cd.initialize(kd, kd.boundary, [512, 512, 512], cd_dir,
                  box_coords=[0, 0, 0], fit_box_size=True)
    oew.from_ids_to_objects(cd, 'cs', n_chunk_jobs=2000, dataset_names=['syn'],
                            hdf5names=["cs"], n_max_co_processes=300,
                            n_folders_fs=100000)

    # POPULATES CS CD with SV contacts
    ces.find_contact_sites(cd, kd_seg_path, n_max_co_processes=5000,
                          qsub_pe='default', qsub_queue='all.q')
    ces.extract_agg_contact_sites(cd, wd, n_folders_fs=10000, suffix="",
                                  qsub_queue='all.q', n_max_co_processes=5000,
                                  qsub_pe='default')
    log.info('CS extraction finished.')

    # create overlap dataset between SJ and CS: SegmentationDataset of type 'syn'
    # TODO: write new method which iterates over sj prob. map (KD), CS ChunkDataset / KD and (optionally) synapse type in parallel and to create a syn segmentation within from_probmaps_to_objects
    # TODO: SD for cs_agg and sj will not be needed anymore
    cs_sd = SegmentationDataset('cs_agg', working_dir=global_params.wd,
                                version=0)  # version hard coded
    sj_sd = SegmentationDataset('sj', working_dir=global_params.wd)
    cs_cset = chunky.load_dataset(cd_dir, update_paths=True)

    # # This creates an SD of type 'syn', currently ~6h, will hopefully be sped up after refactoring
    cs_processing_steps.syn_gen_via_cset(cs_sd, sj_sd, cs_cset, resume_job=False,
                                         nb_cpus=2, qsub_pe='openmp')
    sd = SegmentationDataset("syn", working_dir=global_params.wd, version="0")
    dataset_analysis(sd, qsub_pe='openmp', compute_meshprops=False)
    log.info('SegmentationDataset of type "syn" was generated.')

    # This creates an SD of type 'syn_ssv', ~15 min
    cps.combine_and_split_syn(global_params.wd, resume_job=False,
                              stride=250, qsub_pe='default', qsub_queue='all.q',
                              cs_gap_nm=global_params.cs_gap_nm,
                              n_max_co_processes=global_params.NCORE_TOTAL)
    sd_syn_ssv = SegmentationDataset(working_dir=global_params.wd,
                                     obj_type='syn_ssv')
    dataset_analysis(sd_syn_ssv, qsub_pe='openmp', compute_meshprops=True,
                     stride=100)
    log.info('SegmentationDataset of type "syn_ssv" was generated.')

    # # TODO add syn_ssv object mapping to pipeline

