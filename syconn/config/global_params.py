# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Sven Dorkenwald, Philipp Schubert, Joergen Kornfeld

# # define global working directory
wd = "/wholebrain/scratch/areaxfs/"

# --------------------------------------------------------------- GLIA PARAMETER
# min. connected component size of glia nodes/SV after thresholding glia proba
min_cc_size_glia = 8e3 # in nm; L1-norm on vertex bounding box
# min. connected component size of neuron nodes/SV after thresholding glia proba
min_cc_size_neuron = 8e3 # in nm; L1-norm on vertex bounding box
