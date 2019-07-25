# SyConn
# Copyright (c) 2016 Philipp J. Schubert
# All rights reserved
from syconn import global_params
from syconn.handler.prediction import predict_dense_to_kd


def predict_myelin():
    """
    Generates a probability map for myelinated neuron voxels at
    ``global_params.config.working_dir + '/knossosdatasets/myelin'``.
    """
    predict_dense_to_kd(global_params.config.kd_seg_path,
                        global_params.config.working_dir + '/knossosdatasets/',
                        global_params.config.mpath_myelin, n_channel=2, mag=4,
                        target_channels=[(1, )], target_names=['myelin'])
