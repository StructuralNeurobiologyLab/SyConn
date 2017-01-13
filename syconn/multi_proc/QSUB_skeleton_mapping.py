# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max-Planck-Institute for Medical Research, Heidelberg, Germany
# Authors: Sven Dorkenwald, Philipp Schubert, Jörgen Kornfeld

"""Executable file for QSUB job to calculate cell hull and map cell objects to
 tracings

QSUB wrapper for cell object mapping using enrich_tracings from brainqueries.
"""
import sys
from ..brainqueries import enrich_tracings
from ..utils.datahandler import DataHandler
import cPickle as pickle

if __name__ == '__main__':

    path_storage_file = sys.argv[1]
    path_out_file = sys.argv[2]

    with open(path_storage_file) as f:
        nml_list = pickle.load(f)
        wd = pickle.load(f)
        overwrite = pickle.load(f)
    dh = DataHandler(wd)
    dh.nb_cpus = 16
    enrich_tracings(nml_list, overwrite=overwrite)
