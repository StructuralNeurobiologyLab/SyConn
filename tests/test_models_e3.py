# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max-Planck-Institute of Neurobiology, Munich, Germany
# Authors: Philipp Schubert

import os
from syconn.handler.prediction import *
from syconn import global_params


def test_model_load_e3():
    for curr_dir in [os.path.dirname(os.path.realpath(__file__)) + '/',
                     os.path.abspath(os.path.curdir) + '/',
                     os.path.expanduser('~/SyConn/')]:
        m_dir = curr_dir + '/models/'
        if os.path.isdir(m_dir):
            break
    if not os.path.isdir(m_dir):
        raise FileNotFoundError(f'Example data folder could not be found'
                                f' at "{curr_dir}".')

    global_params.wd = curr_dir  # that is where the models folder is placed
    _ = get_celltype_model_e3()
    _ = get_glia_model_e3()
    _ = get_semseg_axon_model()
    _ = get_semseg_spiness_model()
    _ = get_tripletnet_model_e3()
    _ = get_myelin_cnn()
