# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max-Planck-Institute of Neurobiology, Munich, Germany
# Authors: Philipp Schubert
from syconn.handler.prediction import *
from syconn.handler.prediction_pts import *
import pytest
pytest.mark.filterwarnings("ignore:Initialized working directory without existing config file at")


# TODO: loading pytorch3 models currently lets gitlab-runners get stuck after running tests
def _setup_working_dir():
    """
    This function sets up the working directory for the SyConn toolkit. It iterates through a list of 
    potential directories and sets the global working directory to the first one that contains a 
    'models' subdirectory. If no such directory is found, it raises a FileNotFoundError.
    """
    for curr_dir in [os.path.dirname(os.path.realpath(__file__)) + '/',
                    os.path.abspath(os.path.curdir) + '/',
                    os.path.abspath(os.path.curdir) + '/SyConnData/',
                    os.path.abspath(os.path.curdir) + '/SyConn/',
                    os.path.expanduser('~/SyConnData/'),
                    os.path.expanduser('~/SyConn/'),
                    os.path.expanduser('~/scratch/SyConnData/'),
                    os.path.expanduser('~/scratch/SyConn/')]:

        m_dir = curr_dir + '/models/'
        if os.path.isdir(m_dir):
            break
    if not os.path.isdir(m_dir):
        raise FileNotFoundError(f'Example data folder could not be found'
                                f' at "{curr_dir}".')
    global_params.wd = curr_dir  # that is where the models folder is placed


# dense prediction
def test_load_cnn_myelin():
    """
    This function tests the loading of the myelin Convolutional Neural Network (CNN) model in the 
    SyConn toolkit. It first sets up the working directory and then attempts to load the model.
    """
    _setup_working_dir()
    _ = get_myelin_cnn()


# multi-view models
def test_load_cnn_triplet():
    """
    This function tests the loading of the triplet Convolutional Neural Network (CNN) model in the 
    SyConn toolkit. It first sets up the working directory and then attempts to load the model.
    """
    _setup_working_dir()
    _ = get_tripletnet_model_e3()


def test_load_cnn_spines():
    """
    This function tests the loading of the spines Convolutional Neural Network (CNN) model in the 
    SyConn toolkit. It first sets up the working directory and then attempts to load the model.
    """
    _setup_working_dir()
    _ = get_semseg_spiness_model()


def test_load_cnn_axons():
    """
    This function tests the loading of the axons Convolutional Neural Network (CNN) model in the 
    SyConn toolkit. It first sets up the working directory and then attempts to load the model.
    """
    _setup_working_dir()
    _ = get_semseg_axon_model()


def test_load_cnn_glia():
    """
    This function tests the loading of the glia Convolutional Neural Network (CNN) model in the 
    SyConn toolkit. It first sets up the working directory and then attempts to load the model.
    """
    _setup_working_dir()
    _ = get_glia_model_e3()


def test_load_cnn_celltype():
    """
    This function tests the loading of the cell type Convolutional Neural Network (CNN) model in the 
    SyConn toolkit. It first sets up the working directory and then attempts to load the model.
    """
    _setup_working_dir()
    _ = get_celltype_model_e3()


# point based models
def test_load_cnn_compartment_pt():
    """
    This function tests the loading of the compartment point-based Convolutional Neural Network (CNN) 
    model in the SyConn toolkit. It first sets up the working directory and then attempts to load the 
    model.
    """
    _setup_working_dir()
    _ = get_cmpt_model_pts()


def test_load_cnn_glia_pt():
    """
    This function tests the loading of the glia point-based Convolutional Neural Network (CNN) model 
    in the SyConn toolkit. It first sets up the working directory and then attempts to load the model.
    """
    _setup_working_dir()
    _ = get_glia_model_pts()


def test_load_cnn_celltype_pt():
    """
    This function tests the loading of the cell type point-based Convolutional Neural Network (CNN) 
    model in the SyConn toolkit. It first sets up the working directory and then attempts to load the 
    model.
    """
    _setup_working_dir()
    _ = get_celltype_model_pts()


def test_load_cnn_triplet_pt():
    """
    This function tests the loading of the triplet point-based Convolutional Neural Network (CNN) 
    model in the SyConn toolkit. It first sets up the working directory and then attempts to load the 
    model.
    """
    _setup_working_dir()
    _ = get_tnet_model_pts()

