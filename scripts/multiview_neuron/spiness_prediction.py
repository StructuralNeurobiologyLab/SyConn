# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Philipp Schubert, Sven Dorkenwald, Joergen Kornfeld
import os
import numpy as np
from syconn.config.global_params import wd, mpath_spiness
from syconn.handler.basics import chunkify
from syconn.reps.super_segmentation import SuperSegmentationDataset
from syconn.mp import qsub_utils as qu


if __name__ == "__main__":
    # TODO: currently working directory has to be set globally in global_params and is not adjustable here because all qsub jobs will start a script referring to 'global_params.wd'
    ssd = SuperSegmentationDataset(working_dir=wd)

    # run semantic spine segmentation on multi views
    sd = ssd.get_segmentationdataset("sv")
    # chunk them
    multi_params = chunkify(sd.so_dir_paths, 75)
    pred_key = "spiness"
    # set model properties
    model_kwargs = dict(src=mpath_spiness, multi_gpu=True)
    so_kwargs = dict(working_dir=wd)
    pred_kwargs = dict(pred_key=pred_key)
    multi_params = [[par, model_kwargs, so_kwargs, pred_kwargs]
                    for par in multi_params]
    script_folder = os.path.dirname(
        os.path.abspath(__file__)) + "/../../syconn/QSUB_scripts/"
    qu.QSUB_script(multi_params, "predict_spiness_chunked",
                   n_max_co_processes=25, pe="openmp", queue=None,
                   script_folder=script_folder, n_cores=10,
                   suffix="",  sge_additional_flags="-V")

    # map semantic spine segmentation of multi views on SSV mesh
    multi_params = ssd.ssv_ids
    np.random.shuffle(multi_params)
    multi_params = chunkify(multi_params, 4000)
    # add ssd parameters
    kwargs_semseg2mesh = dict(semseg_key=pred_key)
    multi_params = [(ssv_ids, ssd.version, ssd.version_dict, ssd.working_dir,
                     kwargs_semseg2mesh) for ssv_ids in multi_params]

    qu.QSUB_script(multi_params, "map_spiness", n_max_co_processes=200,
                   pe="openmp", queue=None, script_folder=script_folder,
                   n_cores=10, suffix="", sge_additional_flags="-V")