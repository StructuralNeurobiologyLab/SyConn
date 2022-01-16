import argparse
import numpy as np

from syconn import global_params
from syconn.reps.super_segmentation import SuperSegmentationDataset
from syconn.analysis.utils import get_mean_mesh_areas
from syconn.handler import basics
from syconn.mp.mp_utils import start_multiprocess_imap


def store_mean_mesh_areas(ssd: SuperSegmentationDataset, nb_cpus: int = None):

    if nb_cpus is None:
        import multiprocessing
        nb_cpus = multiprocessing.cpu_count()

    mean_mesh_areas = np.concatenate(start_multiprocess_imap(get_mean_mesh_areas, params=list(basics.chunkify_successive(ssd.ssv_ids, 500)), nb_cpus=nb_cpus), axis=0)

    np.save("/home/hashir/j0251/j0251_72_seg_20210127_agglo2/mean_mesh_areas.npy", mean_mesh_areas)

parser = argparse.ArgumentParser(description='Store mean synaptic mesh areas for all the ssv ids.')
parser.add_argument(
    '--wd', type=str, default="/ssdscratch/songbird/j0251/j0251_72_seg_20210127_agglo2", help='path to the working directory'
)
parser.add_argument(
    '--nb-cpus', '-n', type=int, dest='nb_cpus', default=None, help='Number of CPUs per worker to use'
)

args = parser.parse_args()

global_params.wd = args.wd
ssd = SuperSegmentationDataset(working_dir=global_params.config.working_dir)
store_mean_mesh_areas(ssd, args.nb_cpus)