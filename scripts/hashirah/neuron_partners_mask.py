import argparse
import numpy as np
import multiprocessing
from multiprocessing import Pool
from tqdm import tqdm

from syconn import global_params
from syconn.reps.super_segmentation import SuperSegmentationDataset
from syconn.reps.segmentation import SegmentationDataset

def compute_partner_mask(neuron_partners):
    ssv_0, ssv_1 = neuron_partners[0], neuron_partners[1]
    if ssv_0 not in ssv_ids_150 or ssv_1 not in ssv_ids_150:
        return False
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Store neuron partner mask based on the total edge lengths of ssvs.')
    parser.add_argument(
        '--wd', type=str, default="/ssdscratch/songbird/j0251/j0251_72_seg_20210127_agglo2", help='path to the working directory'
    )
    parser.add_argument(
        '--nb-cpus', '-n', type=int, dest='nb_cpus', default=multiprocessing.cpu_count(), help='Number of CPUs per worker to use'
    )
    
    args = parser.parse_args()

    global_params.wd = args.wd
    ssd = SuperSegmentationDataset(working_dir=global_params.config.working_dir)
    sd = SegmentationDataset('syn_ssv', working_dir=global_params.config.working_dir)

    try:
        tpl = ssd.load_numpy_data('total_edge_length') / 1000  # convert to um
        mask = tpl > 150
    except Exception as e:
        print(e)

    ssv_ids_150 = ssd.ssv_ids[mask]
    neuron_partners = sd.load_numpy_data('neuron_partners')

    with Pool(processes=args.nb_cpus) as p:
        r = list(tqdm(p.imap(compute_partner_mask, neuron_partners), total=len(neuron_partners), desc='Computing neuron partners mask'))

    areaxfs_v10 = np.array(r, dtype=bool)
    np.save('/wholebrain/scratch/hashirah/agglo2_tpl_mask.npy', areaxfs_v10)