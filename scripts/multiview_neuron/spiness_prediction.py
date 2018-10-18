# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Philipp Schubert, Sven Dorkenwald, Joergen Kornfeld
from syconn.config.global_params import wd
from syconn.handler.prediction import get_semseg_spiness_model
from syconn.handler.basics import chunkify
from syconn.reps.super_segmentation import SuperSegmentationDataset
from syconn.reps.super_segmentation_helper import find_missing_sv_attributes_in_ssv
from syconn.mp import qsub_utils as qu
import numpy as np
import os


if __name__ == "__main__":
    ssd = SuperSegmentationDataset(working_dir=wd)
    sd = ssd.get_segmentationdataset("sv")
    # sv_ids = ssd.sv_ids
    # np.random.shuffle(sv_ids)
    # chunk them
    multi_params = chunkify(sd.so_dir_paths, 75)
    pred_key = "spiness"
    # get model properties
    m = get_semseg_spiness_model()
    # TODO: get model paths from config file, also for other predictions!
    model_kwargs = dict(model_path=m._path)
    # all other kwargs like obj_type='sv' and version are the current SV SegmentationDataset by default
    so_kwargs = dict(working_dir=wd)
    # for axoness views set woglia to True (because glia were removed beforehand),
    #  raw_only to False
    pred_kwargs = dict(pred_key=pred_key)
    multi_params = [[par, so_kwargs, pred_kwargs] for par in multi_params]
    script_folder = os.path.dirname(
        os.path.abspath(__file__)) + "/../../syconn/QSUB_scripts/"
    path_to_out = qu.QSUB_script(multi_params, "predict_spiness_chunked",
                                 n_max_co_processes=25, pe="openmp", queue=None,
                                 script_folder=script_folder, n_cores=10,
                                 suffix="_spiness", sge_additional_flags="-V")

    # TODO: map predictions onto vertices!
