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
from syconn.reps.segmentation import SegmentationDataset
from syconn.proc.sd_proc import dataset_analysis


if __name__ == "__main__":
    # kd_seg_path = "/mnt/j0126_cubed/"
    # kd = knossosdataset.KnossosDataset()  # Sets initial values of object
    # kd.initialize_from_knossos_path(kd_seg_path)  # Initializes the dataset by parsing the knossos.conf in path + "mag1"
    #
    # cd = chunky.ChunkDataset()  # Class that contains a dict of chunks (with coordinates) after initializing it
    # cd.initialize(kd, kd.boundary, [512, 512, 512], cd_dir,
    #               box_coords=[0, 0, 0], fit_box_size=True)
    # oew.from_ids_to_objects(cd, 'cs', n_chunk_jobs=2000, dataset_names=['syn'],
    #                         hdf5names=["cs"], n_max_co_processes=300, n_folders_fs=100000)

    # TODO: change path of CS chunkdataset
    cd_dir = global_params.wd + "/chunkdatasets/"
    # TODO: SD for cs_agg and sj will not be needed anymore
    cs_sd = SegmentationDataset('cs_agg', working_dir=global_params.wd, version=0)  # version hard coded, TODO: Change before next use to default
    sj_sd = SegmentationDataset('sj', working_dir=global_params.wd)
    cs_cset = chunky.load_dataset(cd_dir, update_paths=True)

    # TODO: write new method which iterates over sj prob. map (KD), CS ChunkDataset / KD and (optionally) synapse type in parallel and to create a syn segmentation within from_probmaps_to_objects
    cs_processing_steps.overlap_mapping_sj_to_cs_via_cset(cs_sd, sj_sd, cs_cset,
                                                          n_max_co_processes=30,
                                                          nb_cpus=10, qsub_pe='openmp')
    sd = SegmentationDataset("syn", working_dir=global_params.wd, version="0")
    dataset_analysis(sd, qsub_pe='openmp', compute_meshprops=False)
    # TODO: merge syn objects according to RAG/mergelist/SSVs and build syn_ssv dataset
