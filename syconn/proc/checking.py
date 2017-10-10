import cPickle as pkl
import glob
import os

from syconnmp import qsub_utils as qu
from syconnmp import shared_mem as sm
script_folder = os.path.abspath(os.path.dirname(__file__) + "/../multi_proc/")

import checking_helper as ch


def find_missing_overlaycubes(path, stride=100, qsub_pe=None, qsub_queue=None,
                              nb_cpus=1, n_max_co_processes=100):
    if "mag" in path:
        paths = glob.glob(path + "/*/*/*/")
    else:
        paths = glob.glob(path + "/*/*/*/*/")

    multi_params = []
    for path_block in [paths[i:i + stride]
                       for i in xrange(0, len(paths), stride)]:
        multi_params.append([path_block])

    if qsub_pe is None and qsub_queue is None:
        results = sm.start_multiprocess(cv.find_missing_overlaycubes_thread,
                                        multi_params, nb_cpus=nb_cpus)

    elif qu.__QSUB__:
        path_to_out = qu.QSUB_script(multi_params,
                                     "find_missing_overlaycubes",
                                     pe=qsub_pe, queue=qsub_queue,
                                     script_folder=script_folder,
                                     n_max_co_processes=n_max_co_processes)

        out_files = glob.glob(path_to_out + "/*")
        results = []
        for out_file in out_files:
            with open(out_file) as f:
                results.append(pkl.load(f))

    else:
        raise Exception("QSUB not available")

    m_paths = []
    for result in results:
        m_paths += result

    print m_paths
    return m_paths
