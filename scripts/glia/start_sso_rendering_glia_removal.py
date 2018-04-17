import os
from syconn.config.global_params import wd
from syconn.mp import qsub_utils as qu
from syconn.handler.basics import chunkify
from syconn.reps.rep_helper import parse_cc_dict_from_kml
import numpy as np


if __name__ == "__main__":
    # view rendering prior to glia removal, choose SSD accordingly
    version = "tmp"  # glia removal is based on the initial RAG and does not require explicitly stored SSVs
    init_rag_p = wd + "initial_rag.txt"
    assert os.path.isfile(init_rag_p), "Initial RAG could not be found at %s."\
                                       % init_rag_p
    init_rag = parse_cc_dict_from_kml(init_rag_p)
    # get SVs of every connected component
    multi_params = init_rag.values()
    # shuffle connected components
    np.random.shuffle(multi_params)
    # chunk them
    multi_params = chunkify(multi_params, 2000)
    # list of SSV IDs and SSD parameters need to be given to a single QSUB job
    multi_params = [(ixs, wd, version) for ixs in multi_params]

    # generic
    script_folder = os.path.dirname(os.path.abspath(__file__)) + "/../../syconn/QSUB_scripts/"
    path_to_out = qu.QSUB_script(multi_params, "render_views_glia_removal",
                                 n_max_co_processes=200, pe="openmp", queue=None,
                                 script_folder=script_folder, suffix="")