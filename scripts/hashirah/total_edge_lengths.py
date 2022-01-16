import argparse
import numpy as np

from syconn import global_params
from syconn.reps.super_segmentation import SuperSegmentationDataset, get_total_edge_lengths
from syconn.handler import basics
from syconn.mp.mp_utils import start_multiprocess_imap

def store_total_edge_lengths(ssd: SuperSegmentationDataset, nb_cpus: int = None):
    """
    Store total edge lengths of all cells in a super-segmentation dataset.
    """
    if nb_cpus is None:
        import multiprocessing
        nb_cpus = multiprocessing.cpu_count()

    total_edge_lengths = np.concatenate(start_multiprocess_imap(get_total_edge_lengths, params=list(basics.chunkify_successive(ssd.ssv_ids, 500)), nb_cpus=nb_cpus), axis=0)

    np.save("/wholebrain/songbird/j0126/total_edge_lengths.npy", total_edge_lengths)


parser = argparse.ArgumentParser(description='Store total path lengths for all the ssv ids.')
parser.add_argument(
    '--wd', type=str, default="/ssdscratch/songbird/j0251/j0251_72_seg_20210127_agglo2", help='path to the working directory'
)
parser.add_argument(
    '--ax-pred-key', type=str, dest='ax_pred_key', default='axoness_avg10000', help='Key of compartment prediction stored in skeleton'
)
parser.add_argument(
    '--nb-cpus', '-n', type=int, dest='nb_cpus', default=None, help='Number of CPUs per worker to use'
)

args = parser.parse_args()

global_params.wd = args.wd
ssd = SuperSegmentationDataset(working_dir=global_params.config.working_dir)
store_total_edge_lengths(ssd, args.nb_cpus)