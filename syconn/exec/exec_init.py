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
from syconn.handler.logger import initialize_logging


# TODO: make it work with new SyConn
def run_create_sds():
    log = initialize_logging('create_sds', global_params.paths.working_dir + '/logs/',
                             overwrite=False)
    log.info('Generating SegmentationDatasets for cell and cell organelle supervoxels.')
    # TODO: get rid of explicit voxel extraction, all info necessary should be extracted at the beginning, e.g. size, bounding box etc and then refactor to only use those cached attributes!
    # resulting ChunkDataset, required for SV extraction --

    # Sets initial values of object
    kd = knossosdataset.KnossosDataset()
    # Initializes the dataset by parsing the knossos.conf in path + "mag1"
    kd.initialize_from_knossos_path(global_params.paths.kd_seg_path)

    # Object extraction - 2h, the same has to be done for all cell organelles
    cd_dir = global_params.paths.working_dir + "chunkdatasets/sv/"
    # Class that contains a dict of chunks (with coordinates) after initializing it
    cd = chunky.ChunkDataset()
    cd.initialize(kd, kd.boundary, [512, 512, 512], cd_dir,
                  box_coords=[0, 0, 0], fit_box_size=True)
    oew.from_ids_to_objects(cd, "sv", overlaydataset_path=global_params.paths.kd_seg_path, n_chunk_jobs=5000,
                            hdf5names=["sv"], n_max_co_processes=None, qsub_pe='default',
                            qsub_queue='all.q', qsub_slots=1, n_folders_fs=10000)
    log.info('Finished object extraction for cell SVs.')

    # Object Processing - 0.5h
    sd = SegmentationDataset("sv", working_dir=global_params.paths.working_dir)
    sd_proc.dataset_analysis(sd, qsub_pe="default", qsub_queue='all.q',
                             stride=10)
    log.info('Generating SegmentationDatasets for cell and cell organelle supervoxels.')
    # create SegmentationDataset for each cell organelle
    for co in global_params.existing_cell_organelles:
        cd_dir = global_params.paths.working_dir + "chunkdatasets/{}/".format(co)
        # Class that contains a dict of chunks (with coordinates) after initializing it
        cd = chunky.ChunkDataset()
        cd.initialize(kd, kd.boundary, [512, 512, 512], cd_dir,
                      box_coords=[0, 0, 0], fit_box_size=True)

        oew.from_probabilities_to_objects(cd, co, membrane_kd_path=getattr(global_params, 'kd_{}_path'.format(co)),
                                          hdf5names=[co], n_max_co_processes=None, qsub_pe='default',
                                          qsub_queue='all.q', n_folders_fs=10000)
        # About 0.2 h per object class
        sd_proc.map_objects_to_sv(sd, co, global_params.paths.kd_seg_path, qsub_pe='default',
                                  qsub_queue='all.q', stride=20)
        log.info('Finished object extraction for {} SVs.'.format(co))

