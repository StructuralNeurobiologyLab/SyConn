import os
from syconn.mp import qsub_utils as qu
from syconn.mp.shared_mem import start_multiprocess
from syconn.reps.super_segmentation_dataset import SuperSegmentationDataset
from syconn.handler.basics import chunkify
from syconn.proc.mapping import map_glia_fraction
import numpy as np
import itertools


if __name__ == "__main__":
    script_folder = os.path.dirname(os.path.abspath(__file__)) + "/../../syconn/QSUB_scripts/"
    print(script_folder)
    ssds = SuperSegmentationDataset(working_dir="/wholebrain/scratch/areaxfs3/", version="0")
    multi_params = ssds.ssv_ids
    np.random.shuffle(multi_params)
    multi_params = chunkify(multi_params, 4000)
    path_to_out = qu.QSUB_script(multi_params, "map_viewaxoness2skel",#"export_skeletons_new", #"map_viewaxoness2skel",
                                 n_max_co_processes=20, pe="openmp", queue=None,
                                 script_folder=script_folder, suffix="", n_cores=10)