# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Sven Dorkenwald, Philipp Schubert, Joergen Kornfeld

# # define global working directory
# wd = "/wholebrain/songbird/j0126/areaxfs_v5"
wd = "/wholebrain/scratch/areaxfs3/"

# --------------------------------------------------------------- GLIA PARAMETER
# min. connected component size of glia nodes/SV after thresholding glia proba
min_cc_size_glia = 8e3  # in nm; L1-norm on vertex bounding box
# min. connected component size of neuron nodes/SV after thresholding glia proba
min_cc_size_neuron = 8e3  # in nm; L1-norm on vertex bounding box

min_single_sv_size = 30000  # in number of voxels

# Threshold for glia classification
glia_thresh = 0.161489  #

MESH_DOWNSAMPLING = {"sv": (8, 8, 4), "sj": (2, 2, 1), "vc": (4, 4, 2),
                     "mi": (8, 8, 4), "cs": (2, 2, 1), "conn": (2, 2, 1)}
MESH_CLOSING = {"sv": 0, "sj": 0, "vc": 0, "mi": 0, "cs": 0, "conn": 4}

SKEL_FEATURE_CONTEXT = {"axoness": 8000, "spiness": 1000} # in nm


def get_dataset_scaling():
    """
    Helper method to get dataset scaling.

    Returns
    -------
    tuple of float
        (X, Y, Z)
    """
    from .parser import Config
    cfg = Config(wd)
    return cfg.entries["Dataset"]["scaling"]
