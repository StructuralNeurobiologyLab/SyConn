# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max-Planck-Institute for Medical Research, Heidelberg, Germany
# Authors: Sven Dorkenwald, Philipp Schubert, Jörgen Kornfeld

"""Executable file for QSUB job to recalculate cell hull and remap cell objects
 to tracings

QSUB wrapper for cell object mapping using remap_tracings from brainqueries.
"""
import sys
from ..brainqueries import remap_tracings
from ..multi_proc.multi_proc_main import start_multiprocess
import cPickle as pickle


def multi_helper_remap(para):
    para = {'mapped_skel_paths': [para[0]], 'output_dir': para[1],
            'recalc_prop_only': para[2], 'method': para[3],
            'context_range': para[4]}
    remap_tracings(**para)

if __name__ == '__main__':

    path_storage_file = sys.argv[1]
    path_out_file = sys.argv[2]

    with open(path_storage_file) as f:
        nml_list = pickle.load(f)
        output_dir = pickle.load(f)
        recalc_prop_only = pickle.load(f)
        method = pickle.load(f)
        dist = pickle.load(f)

    params = [[nml_list[i], output_dir, recalc_prop_only, method, dist] for i in
              range(len(nml_list))]

    start_multiprocess(multi_helper_remap, params, debug=True, nb_cpus=10)


