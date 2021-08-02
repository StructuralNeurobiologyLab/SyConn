from typing import List, Tuple, Dict, Set, Callable

import argparse

import numpy as np
import pickle as pkl


def load_svagg(in_fname: str) -> List[Set[np.uint64]]:
    ccs = []
    with open(in_fname, 'r') as fp:
        for cur_l in fp.readlines():
            ccs.append(set(np.uint64(xx.strip()) for xx in cur_l.split(',')))
    return ccs


def compare_svagg_lists(in_svagg_1: str, in_svagg_2: str, id_map: Callable[[np.uint64], np.uint64]):
    ccs_1 = load_svagg(in_svagg_1)
    ccs_2 = load_svagg(in_svagg_2)

    # Transforming the inner sets to strings of sorted IDs that are hashable, so that the lists can be converted to
    # sets and easily compared
    ccs_1 = set(','.join([str(zz) for zz in sorted(id_map(yy) for yy in xx)]) for xx in ccs_1)
    ccs_2 = set(','.join([str(zz) for zz in sorted(yy for yy in xx)]) for xx in ccs_2)

    print(f'{len(ccs_1)} components in run 1')
    print(f'{len(ccs_2)} components in run 2')
    print(f'Components in run 1 but not in 2: {ccs_1 - ccs_2}')
    print(f'Components in run 2 but not in 1: {ccs_2 - ccs_1}')


def compare_connectivities(in_csv_1: str, in_csv_2: str, id_map: Callable[[np.uint64], np.uint64]):
    """
    This connectivity comparison method relies on the id mapping preserving the ordering of IDs. This will be the case
    with the remapping methods in gen_64bit_test_data.py

    :param in_csv_1: Path to connectivity csv 1
    :param in_csv_2: Path to connectivity csv 2
    :param id_map: Mapping from ids in run 1 to run 2

    :return:
    """

    conn_1 = set(tuple(xx)
              for xx in np.loadtxt(in_csv_1)[:, 3:5].astype(np.uint64))  # type: Set[Tuple[np.uint64, np.uint64]]
    conn_2 = set(tuple(xx)
              for xx in np.loadtxt(in_csv_2)[:, 3:5].astype(np.uint64))  # type: Set[Tuple[np.uint64, np.uint64]]

    conn_1_remap = set(tuple(id_map(yy) for yy in xx) for xx in conn_1)

    print(f'{len(conn_1_remap)} connections in run 1')
    print(f'{len(conn_2)} connections in run 2')
    print(f'Connections in run 1 but not in 2: {conn_1_remap - conn_2}')
    print(f'Connections in run 2 but not in 1: {conn_2 - conn_1_remap}')

    return


def main():
    parser = argparse.ArgumentParser(description='Compare results of different SyConn runs that differ only in IDs '
                                                 '(for use with gen_64bit_test_data.py)')
    parser.add_argument('--run_dir_1', type=str, required=True,
                        help='SyConn run directory 1')
    parser.add_argument('--run_dir_2', type=str, required=True,
                        help='SyConn run directory 2')
    parser.add_argument('--id_map_pickle', type=str, required=False,
                        help='Path to pickle file mapping IDs from run 1 to run 2. Must preserve ordering.')
    args = parser.parse_args()

    if args.id_map_pickle is not None:
        print(f'Loading mapping pickle from {args.id_map_pickle}')
        with open(args.id_map_pickle, 'rb') as fp:
            id_map_dict = pkl.load(fp)

        def id_map(in_id):
            return id_map_dict[in_id]
    else:
        print(f'No mapping pickle provided, assuming no ID translation required.')

        def id_map(in_id):
            return in_id

    compare_svagg_lists(
        f'{args.run_dir_1}/pruned_svagg_list.txt',
        f'{args.run_dir_2}/pruned_svagg_list.txt',
        id_map)

    compare_connectivities(
        f'{args.run_dir_1}/connectivity_matrix/conn_mat.csv',
        f'{args.run_dir_2}/connectivity_matrix/conn_mat.csv',
        id_map)


if __name__ == '__main__':
    main()
