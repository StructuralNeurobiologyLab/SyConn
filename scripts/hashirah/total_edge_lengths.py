import os
import argparse
import numpy as np
import multiprocessing
from multiprocessing import Process, Manager, Pool

from syconn import global_params
from syconn.reps.super_segmentation import SuperSegmentationDataset
from syconn.mp.mp_utils import start_multiprocess_imap


def get_edge_length(ssv_id):
    """Loads the skeleton of the cell and returns the total length
    of the edges.

    Args:
        ssv_id (int): cell id

    Returns:
        float: total edge length of the cell in nm
    """    
    ssv = ssd.get_super_segmentation_object(ssv_id)
    ssv.load_skeleton()
    return ssv.total_edge_length(compartments_of_interest=[0,1,2,3,4], ax_pred_key='axoness_avg10000')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Store total path lengths for all the ssv ids.')
    parser.add_argument(
        '--wd', type=str, default="/ssdscratch/songbird/j0251/j0251_72_seg_20210127_agglo2", help='path to the working directory'
    )
    parser.add_argument(
        '--ax-pred-key', type=str, dest='ax_pred_key', default='axoness_avg10000', help='Key of compartment prediction stored in skeleton'
    )
    parser.add_argument(
        '--nb-cpus', '-n', type=int, dest='nb_cpus', default=multiprocessing.cpu_count(), help='Number of CPUs per worker to use'
    )

    args = parser.parse_args()

    global_params.wd = args.wd
    ssd = SuperSegmentationDataset(working_dir=global_params.config.working_dir)

    r = start_multiprocess_imap(get_edge_length, ssd.ssv_ids, \
                                nb_cpus=os.cpu_count()*20, desc="Computing edge lengths")
    # with Pool(processes=args.nb_cpus) as p:
    #     # imap is slower than map but works with tqdm
    #     r = list(tqdm(p.imap(get_edge_length, ssd.ssv_ids), total=len(ssd.ssv_ids), desc=''))

    total_edge_lengths = np.array(r)

    # save_path = os.path.join(args.wd, "ssv_0")
    # assert os.path.exists(save_path) and os.path.isdir(save_path), "Path does not exist or is not a directory"

    np.save(os.path.join(f"/home/hashir/total_edge_lengths.npy"), total_edge_lengths)