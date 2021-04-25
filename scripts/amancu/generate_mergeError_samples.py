# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max-Planck-Institute of Neurobiology, Munich, Germany
# Authors: Philipp Schubert, Joergen Kornfeld

import numpy as np
import os
import torch
import multiprocessing as mp
from tqdm import tqdm
from scipy.spatial import cKDTree

from syconn import global_params
from syconn.handler.config import initialize_logging
from syconn.reps.segmentation import SegmentationDataset
from syconn.reps.super_segmentation_dataset import SuperSegmentationDataset
from syconn.proc.meshes import calc_contact_syn_mesh, mesh2obj_file_colors
from syconn.cnn.TrainData import CloudDataSemseg
from syconn.mp.mp_utils import start_multiprocess_imap

# colors for labels
RED = np.array([255.,1.,1.,255.])
GREY = np.array([160.,160.,160.,255.])

def find_nearestNeighbor(verts1: np.ndarray, verts2: np.ndarray, cs_verts: np.ndarray):
    '''
    Finds neighboring points of contact site mesh and labels them with 1, else 0

    :param verts1: Mesh vertices of first partner cell (flat array)
    :param verts2: Mesh vertices of second partner cell (flat array)
    :param cs_verts: Mesh vertices of contact site
    :return: combined cells_mesh (dict of [], verts, []), labels (1 - in cs, 0 - else),
    '''
    # combine cell vertices and build mesh
    cell_vertices = np.concatenate((verts1, verts2)).reshape(-1,3)
    cell_mesh = [np.array([]), cell_vertices, np.array([])]

    #initialize cKTDTree with the combined cell vertices
    NN = cKDTree(data=cell_vertices,)

    # find nearest neighbor of cell vertices to cs vertices
    neighbors = NN.query_ball_point(x=cs_verts, r=55.0)

    # create single set of point neighbors
    neighbors = set(np.concatenate(neighbors))

    # create labels and the corresponding colors
    labels = []
    colors = []
    for i, vert in enumerate(cell_vertices):
        if i in neighbors:
            labels.append(1)
            colors.append(RED)
        else:
            labels.append(0)
            colors.append(GREY)

    # convert to np.arrays for convenience in later processing
    labels = np.array(labels)
    colors = np.array(colors)

    return cell_mesh, labels, colors

def create_labeled_points(arr, slice, cs_dataset, ssv_set):
    log.info(f'Array size: {arr.size} and slice {slice} and sliced array size: {arr[slice].size}')

    # process every id
    for id in tqdm(arr[slice]):
        cs = cs_dataset.get_segmentation_object(id)
        partner_sv_ids = cs.cs_partner
        if not all(item in ssv_set for item in partner_sv_ids):
            # for too small cell sizes, also remove cs id from filtered ids
            continue

        # get the vertices of partnet cells and contact site
        verts1 = ssd.get_super_segmentation_object(partner_sv_ids[0]).mesh[1]
        verts2 = ssd.get_super_segmentation_object(partner_sv_ids[1]).mesh[1]
        cs_mesh = calc_contact_syn_mesh(cs, vertex_size=10)[0]
        cs_verts = cs_mesh[1]

        # look for nearest neighbors, merge cell meshes and label
        cells_mesh, labels, colors = find_nearestNeighbor(verts1, verts2, cs_verts, id)

        # save mesh to .ply and labels to .npy with corresponding ids
        mesh2obj_file_colors(os.path.expanduser(f'/wholebrain/scratch/amancu/mergeError/pts/{id}.ply'), cells_mesh, colors)
        np.save(os.path.expanduser(f'/wholebrain/scratch/amancu/mergeError/labels/{id}.npy'), labels)


if __name__ == '__main__':

    experiment_name = 'generate_mergeError_samples'
    log = initialize_logging(experiment_name, log_dir='/wholebrain/scratch/amancu/mergeError/logs/')

    # setup datasets
    global_params.wd = '/ssdscratch/pschuber/songbird/j0251/rag_flat_Jan2019_v3/'
    cs_dataset = SegmentationDataset(obj_type='cs')
    ssd = SuperSegmentationDataset()
    log.info(f'Datasets loaded')

    # skip small and very large CS. Keep 5000 < size < 100.000 -> 184715199 contact sites (2000 < size < 100.000 -> 349095102)
    if not os.path.exists(os.path.expanduser('~/mergeError/filtered_cs_ids_5000_100000.npy')):
        filtered_cs_ids = cs_dataset.ids[np.where(
            abs(cs_dataset.sizes - 5000 - 47500) <= 47500)]  # cs_dataset.sizes > 1000 and cs_dataset.sizes < 100000
        np.save(os.path.expanduser('~/mergeError/filtered_cs_ids_5000_100000.npy'), filtered_cs_ids)
        log.info(f'Done fetching cs_ids')
    else:
        filtered_cs_ids = np.load(os.path.expanduser('~/mergeError/filtered_cs_ids_5000_100000.npy'))
        log.info('Gotten from file')

    # setup parallelization parameters
    n_proc = mp.cpu_count()
    log.info(f'Using {n_proc} processors')
    chunksize = filtered_cs_ids.size // n_proc
    proc_slices = []
    for i_proc in range(n_proc):
        chunkstart = i_proc * chunksize
        # make sure to include the division remainder for the last process
        chunkend = (i_proc + 1) * chunksize if i_proc < n_proc - 1 else None

        proc_slices.append(np.s_[chunkstart:chunkend])

    # set of ssv_ids to find entries faster
    ssv_ids_set = set(ssd.ssv_ids)

    # initialize Pool for multiprocessing
    with mp.Pool(processes=n_proc) as pool:
        # pass the sliced arrays to each worker process
        results = [pool.apply_async(create_labeled_points,
                                         args=(filtered_cs_ids, slice, cs_dataset, ssv_ids_set,))
                   for slice in proc_slices]

        # blocks async processes until result fetching
        result = []
        for job in results:
            result.append(job.get())