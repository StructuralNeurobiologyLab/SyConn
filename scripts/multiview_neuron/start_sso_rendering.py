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
    # identify huge SSVs and process them individually on whole cluster
    nb_svs = np.array([ssv.sv_ids for ssv in ssd.ssvs])
    big_ssv = multi_params[nb_svs > 5e3]
    for ssv_id in big_ssv:
        ssv = ssd.get_super_segmentation_object(ssv_id)
        ssv.render_views(add_cellobjects=True, cellobjects_only=False,
                         woglia=True, qsub_pe="openmp")
    multi_params = multi_params[nb_svs <= 5e3]
    np.random.shuffle(multi_params)
    multi_params = chunkify(multi_params, 2000)
    # list of SSV IDs and SSD parameters need to be given to a single QSUB job
    multi_params = [(ixs, wd) for ixs in multi_params]

    # generic
    script_folder = os.path.dirname(os.path.abspath(__file__)) + "/../../syconn/QSUB_scripts/"
    path_to_out = qu.QSUB_script(multi_params, "render_views",
                                 n_max_co_processes=200, pe="openmp", queue=None,
                                 script_folder=script_folder, suffix="")