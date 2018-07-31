import os
from syconn.mp import qsub_utils as qu
from syconn.mp.mp_utils import start_multiprocess
from syconn.reps.super_segmentation_dataset import SuperSegmentationDataset
from syconn.handler.basics import chunkify
from syconn.proc.mapping import map_glia_fraction
import numpy as np
import itertools


if __name__ == "__main__":
    pred_key_appendix = "_v2"
    script_folder = os.path.dirname(os.path.abspath(__file__)) + "/../../syconn/QSUB_scripts/"
    print(script_folder)
    ssds = SuperSegmentationDataset(working_dir="/wholebrain/scratch/areaxfs3/")
    multi_params = ssds.ssv_ids
    np.random.shuffle(multi_params)
    multi_params = chunkify(multi_params, 2000)
    multi_params = [(par, pred_key_appendix) for par in multi_params]
    path_to_out = qu.QSUB_script(multi_params, "map_viewaxoness2skel",
                                 n_max_co_processes=280, pe="openmp", queue=None,
                                 script_folder=script_folder, suffix="", n_cores=1)