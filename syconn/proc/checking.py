# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Philipp Schubert, Joergen Kornfeld
try:
    import cPickle as pkl
except ImportError:
    import pickle as pkl
import glob
import os

from ..mp import qsub_utils as qu
from ..mp import mp_utils as sm
from.checking_helper import find_missing_overlaycubes_thread


def find_missing_overlaycubes(path, stride=100, qsub_pe=None, qsub_queue=None,
                              nb_cpus=1, n_max_co_processes=100):
    if "mag" in path:
        paths = glob.glob(path + "/*/*/*/")
    else:
        paths = glob.glob(path + "/*/*/*/*/")

    multi_params = []
    for path_block in [paths[i:i + stride]
                       for i in range(0, len(paths), stride)]:
        multi_params.append([path_block])

    if qsub_pe is None and qsub_queue is None:
        results = sm.start_multiprocess(find_missing_overlaycubes_thread,
                                        multi_params, nb_cpus=nb_cpus)

    elif qu.__BATCHJOB__:
        path_to_out = qu.QSUB_script(multi_params,
                                     "find_missing_overlaycubes",
                                     pe=qsub_pe, queue=qsub_queue,
                                     script_folder=None,
                                     n_max_co_processes=n_max_co_processes)

        out_files = glob.glob(path_to_out + "/*")
        results = []
        for out_file in out_files:
            with open(out_file, 'rb') as f:
                results.append(pkl.load(f))

    else:
        raise Exception("QSUB not available")

    m_paths = []
    for result in results:
        m_paths += result

    print(m_paths)
    return m_paths
