# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max-Planck-Institute of Neurobiology, Munich, Germany
# Authors: Philipp Schubert, Joergen Kornfeld

from knossos_utils import knossosdataset
from knossos_utils import chunky
from syconn import global_params
from syconn.extraction import object_extraction_wrapper as oew
from syconn.proc import sd_proc
from syconn.reps.segmentation import SegmentationDataset


# TODO: make it work with new SyConn
def run_create_sds():
    # path to KnossosDataset of EM segmentation
    kd_seg_path = global_params.kd_seg_path
    wd = global_params.wd  # "/mnt/j0126/areaxfs_v10/"
    # resulting ChunkDataset, required for SV extraction -- TODO: get rid of explicit voxel extraction, all info necessary should be extracted at the beginning, e.g. size, bounding box etc and then refactor to only use those cached attributes!
    cd_dir = wd + "chunkdatasets/"

    #### initializing and loading and chunk dataset from knossosdataset
    kd = knossosdataset.KnossosDataset()  # Sets initial values of object
    kd.initialize_from_knossos_path(kd_seg_path)  # Initializes the dataset by parsing the knossos.conf in path + "mag1"

    cd = chunky.ChunkDataset()  # Class that contains a dict of chunks (with coordinates) after initializing it
    cd.initialize(kd, kd.boundary, [512, 512, 512], cd_dir,
                  box_coords=[0, 0, 0], fit_box_size=True)

    # Object extraction - 2h, the same has to be done for all cell organelles
    oew.from_ids_to_objects(cd, None, overlaydataset_path=kd_seg_path, n_chunk_jobs=5000,
                            hdf5names=["sv"], n_max_co_processes=5000, qsub_pe='default',
                            qsub_queue='all.q', qsub_slots=1, n_folders_fs=10000)

    # Object Processing - 0.5h
    sd = SegmentationDataset("sv", working_dir=wd)
    sd_proc.dataset_analysis(sd, qsub_pe="default", qsub_queue='all.q',
                             stride=10)

    # Map objects to sv's # TODO: make dependent on global_params.existing_cell_organelles
    # About 0.2 h per object class
    sd_proc.map_objects_to_sv(sd, "sj", kd_seg_path, qsub_pe='default',
                              qsub_queue='all.q', stride=20)

    sd_proc.map_objects_to_sv(sd, "vc", kd_seg_path, qsub_pe='default',
                              qsub_queue='all.q', stride=20)

    sd_proc.map_objects_to_sv(sd, "mi", kd_seg_path, qsub_pe='default',
                              qsub_queue='all.q', stride=20)
