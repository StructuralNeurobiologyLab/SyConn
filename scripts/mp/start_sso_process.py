# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Philipp Schubert, Joergen Kornfeld
import os
from syconn.mp import batchjob_utils as qu
from syconn.mp.mp_utils import start_multiprocess
from syconn.reps.super_segmentation_dataset import SuperSegmentationDataset
from syconn.handler.basics import chunkify
from syconn.proc.mapping import map_glia_fraction
import numpy as np
import itertools


if __name__ == "__main__":
    script_folder = os.path.dirname(os.path.abspath(__file__)) + "/../../syconn/QSUB_scripts/"
    print(script_folder)
    ssds = SuperSegmentationDataset(working_dir="/wholebrain/scratch/areaxfs3/", version="0")
    multi_params = ssds.ssv_ids
    np.random.shuffle(multi_params)
    multi_params = chunkify(multi_params, 2000)
    path_to_out = qu.QSUB_script(multi_params, "render_sso_ortho",#"export_skeletons_new", #"map_viewaxoness2skel",
                                 n_max_co_processes=100, pe="openmp", queue=None,
                                 script_folder=script_folder, suffix="", n_cores=1)