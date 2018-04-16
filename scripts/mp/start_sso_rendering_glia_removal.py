import os
from syconn.mp import qsub_utils as qu
from syconn.mp.shared_mem import start_multiprocess
from syconn.reps.super_segmentation_dataset import SuperSegmentationDataset
from syconn.handler.basics import chunkify
from syconn.proc.mapping import map_glia_fraction
import numpy as np
import itertools
import numpy as np


if __name__ == "__main__":
    # view rendering prior to glia removal, choose SSD accordingly
    wd = "/wholebrain/scratch/areaxfs3/"
    version = "0"
    ssd = SuperSegmentationDataset(working_dir=wd, version=version)
    multi_params = ssd.ssv_ids
    np.random.shuffle(multi_params)
    multi_params = chunkify(multi_params, 2000)
    # list of SSV IDs and SSD parameters need to be given to a single QSUB job
    multi_params = [(ixs, wd, version) for ixs in multi_params]

    # generic
    script_folder = os.path.dirname(os.path.abspath(__file__)) + "/../../syconn/QSUB_scripts/"
    path_to_out = qu.QSUB_script(multi_params, "render_views_glia_removal",
                                 n_max_co_processes=200, pe="openmp", queue=None,
                                 script_folder=script_folder, suffix="")