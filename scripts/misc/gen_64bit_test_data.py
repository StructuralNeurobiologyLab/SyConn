from typing import List, Dict

import argparse
from math import ceil
import pickle as pkl
from glob import glob
import shutil
from pathlib import Path

import networkx as nx
import h5py
import numpy as np
from tqdm import tqdm


def load_h5(f: str, dtype=np.uint32) -> np.ndarray:
    with h5py.File(f,'r') as h:
        keys = list(h.keys())
        if len(keys) != 1:
            raise Exception(f'Make sure {f} contains only one key. Found: {keys}')
        return np.array(h[keys[0]], dtype=dtype)


def remap_64bit(in_ids: List[np.uint32], map: str) -> Dict[np.uint64, np.uint64]:
    """
    Remap in_ids to IDs where about half are below 2**32 and the other half are between 2**32 and 2**64.
    """

    if map == 'half_low_half_high':
        new_ids = []
        new_ids.extend(range(1, 1 + ceil(len(in_ids) / 2)))
        new_ids.extend(range(2**32, 2**32 + ceil(len(in_ids) / 2)))

        remap = {old_id: new_id for old_id, new_id in zip(in_ids, new_ids)}
    elif map == 'leftshift':
        remap = {old_id: old_id << 32 for old_id in in_ids}
    else:
        assert False

    return remap


def convert_data_folder(in_dir: str, map: str = 'half_low_half_high'):
    in_dir_p = Path(in_dir)

    out_dir = f'{in_dir_p.parents[0]}/{in_dir_p.name}_64/'
    if Path(out_dir).exists():
        raise Exception(f'Output dir {out_dir} exists already.')
    Path(out_dir).mkdir()

    other_h5s = [xx for xx in glob(f'{in_dir}/*.h5') if xx != 'seg.h5']
    for cur_h5 in other_h5s:
        shutil.copy(cur_h5, out_dir)

    seg = load_h5(f'{in_dir}/seg.h5', dtype=np.uint32)
    ids = [xx for xx in np.unique(seg) if xx != 0]

    id_map = remap_64bit(ids, map=map)

    with open(f'{out_dir}/id_map.pkl', 'wb') as fp:
        fp.write(pkl.dumps(id_map))

    print('Remapping IDs')
    seg_remapped = np.zeros_like(seg, dtype=np.uint64)
    for old_id, new_id in tqdm(id_map.items()):
        seg_remapped[seg == old_id] = new_id

    with h5py.File(f'{out_dir}/seg.h5', 'w') as h:
        h.create_dataset('seg', data=seg_remapped)

    # The graph does not actually need to be directed, but we are using directed graphs here to ensure that the
    # ordering of the IDs in the edges doesn't change, in order to make checking the correctness of the remapped data
    # easier.
    g_old = nx.read_edgelist(f'{in_dir}/neuron_rag.bz2', create_using=nx.DiGraph, nodetype=np.uint64)
    g_new = nx.DiGraph()

    for n_1, n_2 in g_old.edges():
        g_new.add_edge(id_map[n_1], id_map[n_2])

    nx.write_edgelist(g_new, f'{out_dir}/neuron_rag.bz2')


def main():
    parser = argparse.ArgumentParser(description='Convert data to 64 bit for testing')
    parser.add_argument('--in_dir', type=str, required=True,
                        help='Input directory. Should contain h5 data and edge lists.')
    args = parser.parse_args()
    convert_data_folder(args.in_dir, map='leftshift')


if __name__ == '__main__':
    main()
