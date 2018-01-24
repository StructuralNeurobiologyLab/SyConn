import os
from syconn.mp import qsub_utils as qu
from syconn.mp.shared_mem import start_multiprocess
from syconn.reps.super_segmentation_dataset import SuperSegmentationDataset
from syconn.handler.basics import chunkify
from syconn.proc.mapping import map_glia_fraction
import numpy as np
import itertools


if __name__ == "__main__":
    script_folder = os.path.abspath(os.path.dirname(__file__) + "/../../syconn/QSUB_scripts/")
    ssds = SuperSegmentationDataset(working_dir="/wholebrain/scratch/areaxfs3/", version="0")
    multi_params = ssds.ssv_ids
    multi_params = chunkify(multi_params, 3000)
    path_to_out = qu.QSUB_script(multi_params, "sparsify_skeletons",
                                 n_max_co_processes=240, pe="openmp", queue=None,
                                 script_folder=script_folder, suffix="")