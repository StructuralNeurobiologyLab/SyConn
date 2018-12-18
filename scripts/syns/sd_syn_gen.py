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

    cd_dir = global_params.wd + "/chunkdatasets/"
    cs_sd = SegmentationDataset('cs', working_dir=global_params.wd)
    sj_sd = SegmentationDataset('sj', working_dir=global_params.wd)
    cs_cset = chunky.load_dataset(cd_dir, update_paths=True)

    cs_processing_steps.overlap_mapping_sj_to_cs_via_cset(cs_sd, sj_sd, cs_cset,
                                                          n_max_co_processes=340,
                                                          nb_cpus=1)

    # TODO: merge syn objects according to RAG/mergelist/SSVs and build syn_ssv dataset
