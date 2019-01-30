# -*- coding: utf-8 -*-
# distutils: language=c++
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Philipp Schubert, Joergen Kornfeld



try:
    import cPickle as pkl
except ImportError:
    import pickle as pkl
import glob
import numpy as np
import scipy.ndimage
import time
import itertools
from collections import defaultdict
from knossos_utils import knossosdataset
from knossos_utils import chunky
knossosdataset._set_noprint(True)
import os

from ..reps import segmentation
from ..mp import batchjob_utils as qu
from ..mp import mp_utils as sm
from ..handler import compression
from . import object_extraction_steps as oes
from . import log_extraction

from cython.view cimport array as cvarray
from libc.stdint cimport uint64_t, uint32_t
cimport cython
from libc.stdlib cimport rand
from libcpp.map cimport map
from cython.operator import dereference, postincrement


def find_contact_sites(cset, knossos_path, filename='cs', n_max_co_processes=None,
                       qsub_pe=None, qsub_queue=None):
    os.makedirs(cset.path_head_folder, exist_ok=True)
    multi_params = []
    for chunk in cset.chunk_dict.values():
        multi_params.append([chunk, knossos_path, filename])

    if (qsub_pe is None and qsub_queue is None) or not qu.batchjob_enabled():
        results = sm.start_multiprocess_imap(_contact_site_detection_thread, multi_params,
                                             debug=False, nb_cpus=n_max_co_processes)
    elif qu.batchjob_enabled():
        path_to_out = qu.QSUB_script(multi_params,
                                     "contact_site_detection",
                                     script_folder=None,
                                     n_max_co_processes=n_max_co_processes,
                                     pe=qsub_pe, queue=qsub_queue)

        out_files = glob.glob(path_to_out + "/*")
        results = []
        for out_file in out_files:
            with open(out_file, 'rb') as f:
                results.append(pkl.load(f))
    else:
        raise Exception("QSUB not available")
    chunky.save_dataset(cset)


def _contact_site_detection_thread(args):
    chunk = args[0]
    knossos_path = args[1]
    filename = args[2]

    kd = knossosdataset.KnossosDataset()
    kd.initialize_from_knossos_path(knossos_path)

    overlap = np.array([6, 6, 3], dtype=np.int)
    offset = np.array(chunk.coordinates - overlap)
    size = 2 * overlap + np.array(chunk.size)
    data = kd.from_overlaycubes_to_matrix(size, offset, datatype=np.uint64).astype(np.uint32)

    contacts = detect_cs(data)
    os.makedirs(chunk.folder, exist_ok=True)
    compression.save_to_h5py([contacts],
                             chunk.folder + filename +
                             ".h5", ["cs"])


def detect_cs(arr):
    jac = np.zeros([3, 3, 3], dtype=np.int)
    jac[1, 1, 1] = -6
    jac[1, 1, 0] = 1
    jac[1, 1, 2] = 1
    jac[1, 0, 1] = 1
    jac[1, 2, 1] = 1
    jac[2, 1, 1] = 1
    jac[0, 1, 1] = 1

    edges = scipy.ndimage.convolve(arr.astype(np.int), jac) < 0

    edges = edges.astype(np.uint32)
    # edges[arr == 0] = True
    arr = arr.astype(np.uint32)

    # cs_seg = cse.process_chunk(edges, arr, [7, 7, 3])
    cs_seg = process_block_nonzero(edges, arr, [13, 13, 7])

    return cs_seg


def kernel(chunkP, center_idP):
    cdef int [:,:,:] chunk = chunkP
    cdef uint64_t center_id = center_idP

    cdef map[int, int] unique_ids

    for i in range(chunk.shape[0]):
        for j in range(chunk.shape[1]):
            for k in range(chunk.shape[2]):
                unique_ids[chunk[i][j][k]] = unique_ids[chunk[i][j][k]] + 1
    unique_ids[0] = 0
    unique_ids[center_id] = 0
    cdef int theBiggest  = 0
    cdef uint64_t key = 0

    cdef map[int,int].iterator it = unique_ids.begin()
    while it != unique_ids.end():
        if dereference(it).second > theBiggest:
            theBiggest =  dereference(it).second
            key = dereference(it).first
        postincrement(it)

    if theBiggest > 0:
        if center_id > key:
            return (key << 32 ) + center_id
        else:
            return (center_id << 32) + key

    else:
        return key


def process_block(uint32_t[:, :, :] edges, int[:, :, :] arr, stencil1=(7,7,3)):
    #cdef int [:, :, :] stencil = stencil1
    cdef int stencil[3]
    stencil[:] = [stencil1[0], stencil1[1], stencil1[2]]
    assert (stencil[0]%2 + stencil[1]%2 + stencil[2]%2 ) == 3
    cdef uint64_t[:, :, :] out = cvarray(shape = (arr.shape[0], arr.shape[1], arr.shape[2]), itemsize = sizeof(uint64_t), format = 'Q')
    out [:, :, :] = 0
    cdef int offset[3]
    offset[:] = [stencil[0]/2, stencil[1]/2, stencil[2]/2] ### check what ype do you need
    cdef int center_id
    cdef int[:, :, :] chunk = cvarray(shape=(2*offset[0]+2, 2*offset[2]+2, 2*offset[2]+2), itemsize=sizeof(int), format='i')

    for x in range(offset[0], arr.shape[0] - offset[0]):
        for y in range(offset[1], arr.shape[1] - offset[1]):
            for z in range(offset[2], arr.shape[2] - offset[2]):
                if edges[x, y, z] == 0:
                    continue

                center_id = arr[x, y, z] #be sure that it's 32 or 64 bit intiger
                chunk = arr[x - offset[0]: x + offset[0] + 1, y - offset[1]: y + offset[1], z - offset[2]: z + offset[2]]
                out[x, y, z] = kernel(chunk, center_id)

    return np.array(out)


def process_block_nonzero(int[:, :, :] edges, int[:, :, :] arr, stencil1=(7,7,3)):
    cdef int stencil[3]
    stencil[:] = [stencil1[0], stencil1[1], stencil1[2]]
    assert (stencil[0]%2 + stencil[1]%2 + stencil[2]%2 ) == 3

    cdef uint64_t[:, :, :] out = cvarray(shape = (1 + arr.shape[0] - stencil[0], arr.shape[1] - stencil[1] + 1, arr.shape[2] - stencil[2] + 1), itemsize = sizeof(uint64_t), format = 'Q')
    out[:, :, :] = 0
    cdef int center_id
    cdef int offset[3]
    offset [:] = [stencil[0]/2, stencil[1]/2, stencil[2]/2]
    cdef int[:, :, :] chunk = cvarray(shape=(stencil[0]+1, stencil[1]+1, stencil[2]+1), itemsize=sizeof(int), format='i')

    for x in range(0, edges.shape[0]-2*offset[0]):
        for y in range(0, edges.shape[1]-2*offset[1]):
            for z in range(0, edges.shape[2]-2*offset[2]):
                if edges[x+offset[0], y+offset[1], z+offset[2]] == 0:
                    continue
                center_id = arr[x + offset[0], y + offset[1], z + offset[2]]
                chunk = arr[x: x + stencil[0], y: y + stencil[1], z: z + stencil[2]]
                out[x, y, z] = kernel(chunk, center_id)
    return np.array(out)


def extract_agg_contact_sites(cset, working_dir, filename='cs', hdf5name='cs',
                              n_folders_fs=10000, suffix="",
                              n_max_co_processes=None, qsub_pe=None,
                              qsub_queue=None, nb_cpus=1):

    all_times = []
    step_names = []
    time_start = time.time()
    oes.extract_voxels(cset, filename, [hdf5name], dataset_names=['cs_agg'],
                       n_folders_fs=n_folders_fs,
                       chunk_list=None, suffix=suffix, workfolder=working_dir,
                       use_work_dir=True, qsub_pe=qsub_pe,
                       qsub_queue=qsub_queue,
                       n_max_co_processes=n_max_co_processes,
                       nb_cpus=nb_cpus)
    all_times.append(time.time() - time_start)
    step_names.append("voxel extraction")

    # --------------------------------------------------------------------------

    time_start = time.time()
    oes.combine_voxels(working_dir, ['cs_agg'],
                       n_folders_fs=n_folders_fs, qsub_pe=qsub_pe,
                       qsub_queue=qsub_queue,
                       n_max_co_processes=n_max_co_processes,
                       nb_cpus=nb_cpus)
    all_times.append(time.time() - time_start)
    step_names.append("combine voxels")

    log_extraction.debug("Time overview:")
    for ii in range(len(all_times)):
        log_extraction.debug("%s: %.3fs" % (step_names[ii], all_times[ii]))
    log_extraction.debug("--------------------------")
    log_extraction.debug("Total Time: %.1f min" % (np.sum(all_times) / 60.))
    log_extraction.debug("--------------------------")


def _extract_agg_cs_thread(args):
    chunk_block = args[0]
    working_dir = args[1]
    filename = args[2]
    version = args[3]

    segdataset = segmentation.SegmentationDataset("cs_agg", version=version,
                                                  working_dir=working_dir)
    for chunk in chunk_block:
        path = chunk.folder + filename + ".h5"

        this_segmentation = compression.load_from_h5py(path, ["cs"])[0]

        # taken from 'extract_voxels'
        svid_coords_dict = defaultdict(list)  # {id1: [(x0,y0,z0), ..], id2: ..}
        dims = this_segmentation.shape
        indices = itertools.product(range(dims[0]), range(dims[1]),
                                    range(dims[2]))
        # get all SV voxel coords in one pass
        for idx in indices:
            sv_id = this_segmentation[idx]
            svid_coords_dict[sv_id].append(idx)
        # extract bounding boxes
        for sv_id in svid_coords_dict:
            if sv_id == 0:
                continue
            sv_coords = svid_coords_dict[sv_id]
            id_mask_offset = np.min(sv_coords, axis=0)
            abs_offset = chunk.coordinates + id_mask_offset
            id_mask_coords = sv_coords - id_mask_offset
            size = np.max(sv_coords, axis=0) - id_mask_offset + (1, 1, 1)
            id_mask_coords = np.transpose(id_mask_coords)
            id_mask = np.zeros(tuple(size), dtype=bool)
            id_mask[id_mask_coords[0, :], id_mask_coords[1, :],
                    id_mask_coords[2, :]] = True
            segobj = segdataset.get_segmentation_object(sv_id, create=True)
            segobj.save_voxels(id_mask, abs_offset)

        # TODO: PREVIOUS CODE - delete when above was tested
        # unique_ids = np.unique(this_segmentation)
        # for unique_id in unique_ids:
        #     if unique_id == 0:
        #         continue
        #
        #     id_mask = this_segmentation == unique_id
        #     id_mask, in_chunk_offset = crop_bool_array(id_mask)
        #     abs_offset = chunk.coordinates + np.array(in_chunk_offset)
        #     abs_offset = abs_offset.astype(np.int)
        #     segobj = segdataset.get_segmentation_object(unique_id,
        #                                                 create=True)
        #     segobj.save_voxels(id_mask, abs_offset)
        #     print(unique_id)

