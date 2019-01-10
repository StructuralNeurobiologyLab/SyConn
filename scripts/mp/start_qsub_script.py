# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Philipp Schubert, Joergen Kornfeld
import os
from syconn.mp import qsub_utils as mu
from syconn.reps.segmentation import SegmentationDataset
from syconn.handler.basics import chunkify


if __name__ == "__main__":
    script_folder = os.path.abspath(os.path.dirname(__file__) + "/../qsub_scripts/")
    sds = SegmentationDataset("cs", version="33",
                              working_dir="/wholebrain/scratch/areaxfs/")
    multi_params = chunkify(list(sds.sos), 1000)
    path_to_out = mu.QSUB_script(multi_params, "map_cs_properties",
                                 n_max_co_processes=40, pe="openmp", queue=None,
                                 script_folder=script_folder)