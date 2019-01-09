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


if __name__ == "__main__":
    kd_sym_path = global_params.kd_sym_path
    kd_asym_path = global_params.kd_asym_path
    kd_seg_path = global_params.kd_seg_path
    # kd = knossosdataset.KnossosDataset()  # Sets initial values of object
    # # Initializes the dataset by parsing the knossos.conf in path + "mag1"
    # kd.initialize_from_knossos_path(kd_seg_path)

    # TODO: change path of CS chunkdataset
    cd_dir = global_params.wd + "/chunkdatasets/"
    # # Class that contains a dict of chunks (with coordinates) after initializing it
    # cd = chunky.ChunkDataset()
    # cd.initialize(kd, kd.boundary, [512, 512, 512], cd_dir,
    #               box_coords=[0, 0, 0], fit_box_size=True)
    # oew.from_ids_to_objects(cd, 'cs', n_chunk_jobs=2000, dataset_names=['syn'],
    #                         hdf5names=["cs"], n_max_co_processes=300,
    #                         n_folders_fs=100000)
    #
    # # POPULATES CS CD with SV contacts
    # ces.find_contact_sites(cd, kd_seg_path, n_max_co_processes=5000,
    #                       qsub_pe='default', qsub_queue='all.q')

    # # create overlap dataset between SJ and CS
    # # TODO: write new method which iterates over sj prob. map (KD), CS ChunkDataset / KD and (optionally) synapse type in parallel and to create a syn segmentation within from_probmaps_to_objects
    # # TODO: SD for cs_agg and sj will not be needed anymore
    # cs_sd = SegmentationDataset('cs_agg', working_dir=global_params.wd, version=0)  # version hard coded, TODO: Change before next use to default
    # sj_sd = SegmentationDataset('sj', working_dir=global_params.wd)
    # cs_cset = chunky.load_dataset(cd_dir, update_paths=True)

    # # This creates an SD of type 'syn'
    # cs_processing_steps.overlap_mapping_sj_to_cs_via_cset(cs_sd, sj_sd, cs_cset, resume_job=True,
    #                                                       nb_cpus=4, qsub_pe='openmp')
    # sd = SegmentationDataset("syn", working_dir=global_params.wd, version="0")
    # dataset_analysis(sd, qsub_pe='openmp', compute_meshprops=False)

    # # TODO: merge syn objects according to RAG/mergelist/SSVs and build syn_ssv dataset
    # ssd = SuperSegmentationDataset(global_params.wd)
    # # This creates an SD of type 'syn_ssv'
    cps.combine_and_split_syn(global_params.wd, cs_gap_nm=global_params.cs_gap_nm,
                                 stride=100, qsub_pe='default',
                                 qsub_queue='all.q', resume_job=False,
                                 n_max_co_processes=global_params.NCORE_TOTAL)
    sd_syn_ssv = SegmentationDataset(working_dir=global_params.wd,
                                     obj_type='syn_ssv')
    # dataset_analysis(sd_syn_ssv, qsub_pe='openmp', compute_meshprops=True,
    #                  stride=100)
    # # TODO add syn_ssv object mapping to pipeline

    # # This will be replaced with new method for syn_ssv generation
    # extract_synapse_type(sd_syn_ssv, kd_sym_path=kd_sym_path, stride=100,
    #                      kd_asym_path=kd_asym_path, qsub_pe='openmp')

    cps.map_objects_to_synssv(global_params.wd)

    # cps.classify_synssv_objects(global_params.wd,)# qsub_pe='openmp')

    # # # as an alternative to the skeletons, use vertex predictions for that
    # cps.collect_properties_from_ssv_partners(global_params.wd, qsub_pe='openmp')
    #
    # # collect new object attributes like partner axoness, celltypes,
    # # synapse probabilities etc, -> recompute=False
    # dataset_analysis(sd_syn_ssv, qsub_pe='openmp', compute_meshprops=True,
    #                  stride=100, recompute=False)
    #
    # # export_matrix
    # cps.export_matrix(global_params.wd, global_params.wd +
    #                   '/connectivity_matrix/conn_mat')
