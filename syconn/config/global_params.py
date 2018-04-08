# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Sven Dorkenwald, Philipp Schubert, Joergen Kornfeld

# # define global working directory
wd = "/wholebrain/scratch/areaxfs3/"

# --------------------------------------------------------------- GLIA PARAMETER
# min. connected component size of glia nodes/SV after thresholding glia proba
min_cc_size_glia = 8e3 # in nm; L1-norm on vertex bounding box
# min. connected component size of neuron nodes/SV after thresholding glia proba
min_cc_size_neuron = 8e3 # in nm; L1-norm on vertex bounding box

MESH_DOWNSAMPLING = {"sv": (8, 8, 4), "sj": (2, 2, 1), "vc": (4, 4, 2),
                     "mi": (8, 8, 4), "cs": (2, 2, 1), "conn": (2, 2, 1)}
MESH_CLOSING = {"sv": 0, "sj": 0, "vc": 0, "mi": 0, "cs": 0, "conn": 4}

SKEL_FEATURE_CONTEXT = {"axoness": 8000, "spiness": 1000} # in nm