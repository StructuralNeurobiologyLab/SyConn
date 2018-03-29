# SyConnFS
# Copyright (c) 2016 Philipp J. Schubert
# All rights reserved
import os
from syconnmp import qsub_utils as qu
from syconnfs.representations.segmentation import SegmentationDataset
from syconnfs.handler.basics import chunkify


if __name__ == "__main__":
    script_folder = os.path.abspath(os.path.dirname(__file__) + "/../qsub_scripts/")
    sds = SegmentationDataset("cs", version="33",
                              working_dir="/wholebrain/scratch/areaxfs/")
    multi_params = chunkify(list(sds.sos), 1000)
    path_to_out = qu.QSUB_script(multi_params, "map_cs_properties",
                                 n_max_co_processes=40, pe="openmp", queue=None,
                                 script_folder=script_folder)