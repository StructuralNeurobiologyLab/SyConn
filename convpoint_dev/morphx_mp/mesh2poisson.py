import os
import glob
from syconn import global_params
from syconn.mp import batchjob_utils as qu


def process_dataset(input_path: str, output_path: str):
    """ Converts all HybridMeshs, saved as pickle files at input_path, into poisson disk sampled HybridClouds and
        saves them at output_path with the same names.

    Args:
        input_path: Path to pickle files with HybridMeshs.
        output_path: Path to folder in which results should be stored.
    """

    files = glob.glob(input_path + '*.pkl')
    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    multi_params = []
    for file in files:
        multi_params.append([file, output_path])

    _ = qu.QSUB_script(multi_params, "mx_poisson_preprocessing", n_cores=10, remove_jobfolder=True)


if __name__ == '__main__':
    global_params.wd = "/u/jklimesch/mp/"
    process_dataset('/u/jklimesch/gt/gt_all/', '/u/jklimesch/gt/gt_all/test/')
