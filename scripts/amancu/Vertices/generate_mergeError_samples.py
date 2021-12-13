# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max-Planck-Institute of Neurobiology, Munich, Germany
# Authors: Andrei Mancu, Philipp Schubert, Joergen Kornfeld

import numpy as np
import os
import torch
import timeit
import argparse
import gc
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import multiprocessing as mp
import threading as th
from concurrent.futures import ThreadPoolExecutor
from scipy.spatial import cKDTree
from syconn import global_params
from syconn.handler.config import initialize_logging
from syconn.handler.basics import write_obj2pkl, load_pkl2obj
from syconn.reps.segmentation import SegmentationDataset
from syconn.reps.super_segmentation_dataset import SuperSegmentationDataset
from syconn.proc.meshes import calc_contact_syn_mesh, mesh2obj_file_colors
from syconn.proc.ssd_proc import merge_ssv
from morphx.classes.hybridmesh import HybridCloud

# paths
filtered_cs_ids_path = os.path.expanduser('~/mergeError/filtered_cs_ids_5000_100000.npy')
lookup_cellpair2cs_path = os.path.expanduser('~/mergeError/lookup_cellpair2cs.pkl')

#####################################################################
''' Change pt radius here before running '''
#####################################################################
# cs_ptMerger_radius = 100.0
no_Skelmerger_radius = 20e3
skelmerger_radius = 2e3

# CHANGE
nr_samples = 6000
# colors for labels
RED = np.array([255., 50., 50., 255.])
GREY = np.array([180., 180., 180., 255.])


def find_vertNearestNeighbor(merged_cell, cs_verts: np.ndarray, radius):
    '''
    Finds neighboring points of cell cloud to contact site cloud and labels them with 1, else 0

    :param merged_cell: (SegmentationObject) Merged cell object from where to extrect the point cloouds
    :param cs_verts: Mesh vertices of contact site
    :return: point labels for each vertex (1 - in cs, 0 - else), colors for each vertex
    '''
    # combine cell vertices and build mesh
    cell_mesh = merged_cell.mesh
    cell_vertices = cell_mesh[1].reshape(-1, 3)

    # initialize cKTDTree with the combined cell vertices
    vert_NN = cKDTree(data=cell_vertices, )
    # find nearest neighbor of cell vertices to cs vertices
    vert_neighbors = vert_NN.query_ball_point(x=cs_verts, r=radius, workers=2)
    # create single set of point neighbors
    vert_neighbors = np.unique(np.concatenate(vert_neighbors)).astype(int)

    # create labels and the corresponding colors
    # for type compatibility
    one = np.uint(1)
    vertex_labels = np.zeros(shape=(len(cell_vertices),), dtype=np.uint)
    np.put(vertex_labels, vert_neighbors, one)

    colors = np.full(shape=(len(cell_vertices), 4,), fill_value=GREY)
    vert_neighbors = np.array([[x] for x in vert_neighbors])
    np.put_along_axis(colors, vert_neighbors, RED, axis=0)

    return cell_vertices, vertex_labels, colors


def find_nodeNearestNeighbor(merged_cell, cs_coord_list):
    merged_cell_nodes = merged_cell.skeleton['nodes'] * merged_cell.scaling  # coordinates of all nodes

    # labels:
    # 1 for false-merger, 0 for true merger, -1 for ignore
    node_labels = np.zeros((len(merged_cell_nodes),)) - 1

    # find medium cube around artificial merger and set it to 0 (no-merger/cell_body)
    kdtree = cKDTree(data=merged_cell_nodes)

    # find all skeleton nodes which are close to all contact-sites (r=20e3)
    for cs_coord in cs_coord_list:
        ixs = kdtree.query_ball_point(cs_coord, r=no_Skelmerger_radius, workers=2)
        ixs = np.array(ixs[0])
        node_labels[ixs] = int(0)

    # find small circle around artificial merger and set it to 1 (true merger)
    for cs_coord in cs_coord_list:
        # ixs = kdtree.query_ball_point(cs_coord, r=2e3)
        ixs = kdtree.query_ball_point(cs_coord, r=skelmerger_radius, workers=2)
        ixs = np.array(ixs[0])
        if len(ixs)==0:
            continue
        node_labels[ixs] = int(1)

    return merged_cell_nodes, node_labels


def create_labeled_points(cell_pair2cs_ids, cell_pairs, slice, cs_dataset, ssv_set, radii):
    # process every cell pair
    for cellpair in tqdm(cell_pairs[slice], desc='Sample gen'):
        cell1 = cellpair[0]
        cell2 = cellpair[1]
        # if cells are same, skip
        if cell1 == cell2:
            print(f'Dict cells are the same for: {cell1}')
            continue

        if dataset == 'test':
            truth_array = [os.path.exists(os.path.expanduser(
                f'/wholebrain/scratch/amancu/mergeError/test_dataset/R{int(radius)}/sso_{cell1}_{cell2}.pkl'))
                for radius in radii]
        else:
            truth_array = [os.path.exists(os.path.expanduser(
                f'/wholebrain/scratch/amancu/mergeError/ptclouds/R{int(radius)}/Hybridcloud/sso_{cell1}_{cell2}.pkl'))
                for radius in radii]
        if np.all(truth_array):
            continue

        # get partner cells, merge them, and get contact site mesh
        fstObj = ssd.get_super_segmentation_object(cell1)
        sndObj = ssd.get_super_segmentation_object(cell2)

        # cs coords for skeleton nodes
        cs_coord_list = []
        try:
            for cs_id in cell_pair2cs_ids[(cell1, cell2)]:
                cs = cs_dataset.get_segmentation_object(cs_id)
                cs_mesh = calc_contact_syn_mesh(cs, vertex_size=10)
                for mesh in cs_mesh:
                    area_mesh = mesh[1].reshape(-1, 3)
                    # choose random vertex as representative of the contact area (may be multiple for 1 CS)
                    idx = np.random.randint(len(area_mesh), size=1)
                    cs_coord_list.append(area_mesh[idx])
        except Exception as e:
            print(f'[EXCEPTION]: {e}')
            continue
        if len(cs_coord_list) == 0:
            log.info(f'No cs found for given cell pair {cell1} and {cell2}')
            continue

        # pass one vertex from each contact site mess too, so that the skeleton concatenation is done correctly
        merged_cell = merge_ssv(fstObj, sndObj, cs_coord_list)

        # look for nearby skeleton nodes
        merged_cell_nodes, node_labels = find_nodeNearestNeighbor(merged_cell, cs_coord_list)

        features = np.zeros(shape=(len(merged_cell.mesh[1].reshape(-1, 3)),), dtype=np.int32)

        for radius in radii:
            # look for nearest neighbors, merge cell meshes and label
            cell_vertices, vertex_labels, colors = find_vertNearestNeighbor(merged_cell, cs_verts, radius)

            # save mesh to .ply and mesh+skeleton with labels as HybridCloud .pkl
            hc = HybridCloud(vertices=cell_vertices, labels=vertex_labels, features=features,
                             nodes=merged_cell_nodes, node_labels=node_labels,
                             edges=merged_cell.skeleton['edges'])

            if dataset == 'test':
                if hc.save2pkl(os.path.expanduser(
                        f'/wholebrain/scratch/amancu/mergeError/test_dataset/R{int(radius)}/sso_{cell1}_{cell2}.pkl')):
                    log.info('HybridCloud not written')
            else:
                if hc.save2pkl(os.path.expanduser(
                        f'/wholebrain/scratch/amancu/mergeError/ptclouds/R{int(radius)}/Hybridcloud/sso_{cell1}_{cell2}.pkl')):
                    log.info('HybridCloud not written')
        del hc
        gc.collect()


def create_lookup_table(filtered_contact_sites_ids, cs_dataset, dict_sv2ssv):
    """Loop through all contact_sites ids and if two corresponding cells are found,
    store the cell_id and corresponding cs_id into dictionary

    Returns
    -------
    cell_pair2cs_ids : dict
    cell_pairs : list
    """
    cell_pair2cs_ids = dict()
    cell_pairs = list()

    for cs_id in tqdm(filtered_contact_sites_ids):
        sv_partner = cs_dataset.get_segmentation_object(cs_id).cs_partner
        if sv_partner[0] in dict_sv2ssv and sv_partner[1] in dict_sv2ssv:
            c1 = dict_sv2ssv[sv_partner[0]]
            c2 = dict_sv2ssv[sv_partner[1]]
            if c1 == c2:
                continue
            if c1 < c2:
                if (c1, c2) in cell_pair2cs_ids:
                    cell_pair2cs_ids[(c1, c2)].append(cs_id)
                else:
                    cell_pair2cs_ids[(c1, c2)] = [cs_id]
                    cell_pairs.append((c1, c2))
            else:
                if (c2, c1) in cell_pair2cs_ids:
                    cell_pair2cs_ids[c2, c1].append(cs_id)
                else:
                    cell_pair2cs_ids[(c2, c1)] = [cs_id]
                    cell_pairs.append((c2, c1))

    log.info("len cell_pairs2cs_ids: {}".format(len(cell_pair2cs_ids)))
    return cell_pair2cs_ids, cell_pairs


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Generate samples for merger error')
    parser.add_argument('--r', nargs='+', help='Radius of merger',
                        default=[1000,2000])
    parser.add_argument('--nproc', type=int, help='Number of processors to use',
                        default=15)
    parser.add_argument('--set', type=str, help='Training or test set generation.', default='training')
    args = parser.parse_args()
    # global cs_ptMerger_radius
    cs_merge_radii = [int(x) for x in args.r]
    # cs_merge_radii = [1000, 2000]
    # cs_merge_radii = [100, 500, 5000]
    n_proc = args.nproc
    dataset = args.set

    experiment_name = 'generate_mergeError_samples'
    log = initialize_logging(experiment_name, log_dir='/wholebrain/scratch/amancu/mergeError/logs/')

    # setup datasets
    global_params.wd = '/ssdscratch/pschuber/songbird/j0251/rag_flat_Jan2019_v3/'
    cs_dataset = SegmentationDataset(obj_type='cs')
    ssd = SuperSegmentationDataset()
    dict_sv2ssv = ssd.mapping_dict_reversed  # dict: {supervoxel : super-supervoxel}
    log.info(f'Datasets loaded')

    log.info(f'Cs merge radii: {cs_merge_radii}')

    # skip small and very large CS. Keep 5000 < size < 100.000 -> 184715199 contact sites (2000 < size < 100.000 -> 349095102)
    if not os.path.exists(filtered_cs_ids_path):
        filtered_cs_ids = cs_dataset.ids[np.where(
            abs(cs_dataset.sizes - 5000 - 47500) <= 47500)]  # cs_dataset.sizes > 1000 and cs_dataset.sizes < 100000
        np.save(filtered_cs_ids_path, filtered_cs_ids)
        log.info(f'Done fetching cs_ids')
    else:
        filtered_cs_ids = np.load(filtered_cs_ids_path)
        log.info('Gotten from file')

    # create lookup table for cell ids to cs ids
    if not os.path.exists(lookup_cellpair2cs_path):
        cell_pair2cs_ids, cell_pairs = create_lookup_table(filtered_cs_ids, cs_dataset, dict_sv2ssv)
        write_obj2pkl(lookup_cellpair2cs_path, cell_pair2cs_ids)
        log.info(f'Cell pairs written to pickle')
    else:
        try:
            cell_pair2cs_ids = load_pkl2obj(lookup_cellpair2cs_path)
            cell_pairs = []
            for key in cell_pair2cs_ids.keys():
                cell_pairs.append(key)
            log.info(f'Cell pairs loaded')
        except:
            log.info(f'Pickle file not found')
            cell_pair2cs_ids, cell_pairs = create_lookup_table(filtered_cs_ids, cs_dataset, dict_sv2ssv)
            write_obj2pkl(lookup_cellpair2cs_path, cell_pair2cs_ids)
            log.info(f'Cell pairs written to pickle')

    del filtered_cs_ids
    gc.collect()

    if dataset == 'test':
        offset = 30e3
        nr_samples = 2000
        log.info(f'Offset and nr_samples adapted to test set.')
    else:
        offset = 0
        nr_samples = 7000

    # setup parallelization parameters
    log.info(f'Using {n_proc} processors')
    chunksize = nr_samples // n_proc
    proc_slices = []

    for i_proc in range(n_proc):
        chunkstart = int(offset + (i_proc * chunksize))
        # make sure to include the division remainder for the last process
        chunkend = int(offset + (i_proc + 1) * chunksize) if i_proc < n_proc - 1 else int(offset + nr_samples)
        proc_slices.append(np.s_[chunkstart:chunkend])

    log.info(proc_slices)
    # set of ssv_ids to find entries faster
    ssv_ids_set = set(ssd.ssv_ids)
    params = [(cell_pair2cs_ids, cell_pairs, slice, cs_dataset, ssv_ids_set, cs_merge_radii) for slice in proc_slices]

    time = timeit.default_timer()
    log.info(f'Sample generation started {time} with {len(params)} tasks')

    # process_map(create_labeled_points, params, max_workers=n_proc)
    running_tasks = [mp.Process(target=create_labeled_points, args=param) for param in params]
    for running_task in running_tasks:
        running_task.start()
    for running_task in running_tasks:
        running_task.join()
