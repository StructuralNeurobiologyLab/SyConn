# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Philipp Schubert, Sven Dorkenwald, Joergen Kornfeld
from syconn.config.global_params import wd
from syconn.handler.prediction import get_axoness_model_V2
from syconn.handler.basics import chunkify
from syconn.reps.super_segmentation import SuperSegmentationDataset
from syconn.reps.super_segmentation_helper import find_missing_sv_attributes_in_ssv
from syconn.mp import qsub_utils as qu
import numpy as np
import os


def axoness_pred_exists(sv):
    sv.load_attr_dict()
    return 'axoness_probas_v2' in sv.attr_dict


if __name__ == "__main__":
    ssd = SuperSegmentationDataset(working_dir=wd)
    sd = ssd.get_segmentationdataset("sv")
    # sv_ids = ssd.sv_ids
    # np.random.shuffle(sv_ids)
    # chunk them
    multi_params = chunkify(sd.so_dir_paths, 75)
    pred_key = "axoness_probas_v2"
    # get model properties
    m = get_axoness_model_V2()
    model_kwargs = dict(model_path=m._path, normalize_data=m.normalize_data,
                        imposed_batch_size=m.imposed_batch_size, nb_labels=m.nb_labels,
                        channels_to_load=m.channels_to_load)
    # all other kwargs like obj_type='sv' and version are the current SV SegmentationDataset by default
    so_kwargs = dict(working_dir=wd)
    # for axoness views set woglia to True (because glia were removed beforehand),
    #  raw_only to False
    pred_kwargs = dict(woglia=True, pred_key=pred_key, verbose=False,
                       raw_only=False)
    multi_params = [[par, model_kwargs, so_kwargs, pred_kwargs] for par in multi_params]
    # randomly assign to gpu 0 or 1
    for par in multi_params:
        mk = par[1]
        mk["init_gpu"] = np.random.rand(0, 2)
    script_folder = os.path.dirname(
        os.path.abspath(__file__)) + "/../../syconn/QSUB_scripts/"
    path_to_out = qu.QSUB_script(multi_params, "predict_sv_views_chunked",
                                 n_max_co_processes=25, pe="openmp", queue=None,
                                 script_folder=script_folder, n_cores=10,
                                 suffix="_axoness", sge_additional_flags="-V")
    res = find_missing_sv_attributes_in_ssv(ssd, pred_key, n_cores=10)
    if len(res) > 0:
        print("Attribute '{}' missing for follwing"
              " SVs:\n{}".format(pred_key, res))
