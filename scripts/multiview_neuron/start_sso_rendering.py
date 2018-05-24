# SyConn
# Copyright (c) 2018 Philipp J. Schubert, J. Kornfeld
# All rights reserved
import os
from syconn.config.global_params import wd
from syconn.mp import qsub_utils as qu
from syconn.reps.super_segmentation_dataset import SuperSegmentationDataset
from syconn.handler.basics import chunkify
import numpy as np


if __name__ == "__main__":
    # view rendering prior to glia removal, choose SSD accordingly
    ssd = SuperSegmentationDataset(working_dir=wd)
    multi_params = ssd.ssv_ids
    np.random.shuffle(multi_params)
    multi_params = chunkify(multi_params, 2000)
    # list of SSV IDs and SSD parameters need to be given to a single QSUB job
    multi_params = [(ixs, wd) for ixs in multi_params]

    # generic
    script_folder = os.path.dirname(os.path.abspath(__file__)) + "/../../syconn/QSUB_scripts/"
    path_to_out = qu.QSUB_script(multi_params, "render_views",
                                 n_max_co_processes=100, pe="openmp", queue=None,
                                 script_folder=script_folder, suffix="")