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
import shutil
import glob
import numpy as np
import scipy.ndimage
from knossos_utils import knossosdataset
from knossos_utils import chunky
knossosdataset._set_noprint(True)
import os
from ..mp import batchjob_utils as qu
from ..handler import compression
from ..handler.basics import kd_factory
from . object_extraction_steps import export_cset_to_kd_batchjob
from . import log_extraction
from .object_extraction_wrapper import from_ids_to_objects, calculate_chunk_numbers_for_box
from ..mp.mp_utils import start_multiprocess_imap
try:
    from .block_processing_cython import kernel, process_block, process_block_nonzero
except ImportError as e:
    log_extraction.warning('Could not import cython version of `block_processing`.')
    from .block_processing import kernel, process_block, process_block_nonzero
from syconn import global_params


def find_contact_sites(cset, knossos_path, filename='cs', n_max_co_processes=None,
                       size=None, offset=None):
    """

    Parameters
    ----------
    cset :
    knossos_path :
    filename :
    n_max_co_processes :
    qsub_pe :
    qsub_queue :
    size :
    offset :

    Returns
    -------

    """
    if size is not None and offset is not None:
        chunk_list, _ = \
            calculate_chunk_numbers_for_box(cset, offset, size)
    else:
        chunk_list = [ii for ii in range(len(cset.chunk_dict))]

    os.makedirs(cset.path_head_folder, exist_ok=True)
    multi_params = []
    for chunk_k in chunk_list:
        multi_params.append([cset.chunk_dict[chunk_k], knossos_path, filename])

    if not qu.batchjob_enabled():
        _ = start_multiprocess_imap(_contact_site_detection_thread, multi_params,
                                    debug=False, nb_cpus=n_max_co_processes)
    else:
        _ = qu.QSUB_script(multi_params, "contact_site_detection",
                           script_folder=None, n_max_co_processes=n_max_co_processes)
    chunky.save_dataset(cset)


def _contact_site_detection_thread(args):
    chunk = args[0]
    knossos_path = args[1]
    filename = args[2]

    kd = kd_factory(knossos_path)

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


def extract_agg_contact_sites(cset, working_dir, filename='cs', hdf5name='cs',
                              n_folders_fs=10000, suffix="",
                              n_max_co_processes=None, n_chunk_jobs=2000, size=None,
                              offset=None, log=None):
    """

    Parameters
    ----------
    cset :
    working_dir :
    filename :
    hdf5name :
    n_folders_fs :
    suffix :
    n_max_co_processes :
    n_chunk_jobs :
    size :
    offset :

    Returns
    -------

    """
    if log is None:
        log = log_extraction
    chunky.save_dataset(cset)
    # init CS segmentation KD
    kd = kd_factory(global_params.config.kd_seg_path)
    path = "{}/knossosdatasets/{}_seg/".format(global_params.config.working_dir, filename)
    if os.path.isdir(path):
        log.debug('Found existing KD at {}. Removing it now.'.format(path))
        shutil.rmtree(path)
        log.debug('Done')
    target_kd = knossosdataset.KnossosDataset()
    target_kd.initialize_without_conf(path, kd.boundary, kd.scale, kd.experiment_name, mags=[1, ])
    target_kd = knossosdataset.KnossosDataset()
    target_kd.initialize_from_knossos_path(path)

    # convert Chunkdataset to KD
    export_cset_to_kd_batchjob(
        cset, target_kd, '{}'.format(filename), [hdf5name],
        offset=offset, size=size, stride=[4 * 128, 4 * 128, 4 * 128], as_raw=False,
        orig_dtype=np.uint64, unified_labels=False,
        n_max_co_processes=n_max_co_processes)
    log.debug('Finished conversion of ChunkDataset ({}) into KnossosDataset ({})'.format(
        cset.path_head_folder, target_kd.knossos_path))

    # get CS SD
    from_ids_to_objects(cset, filename, overlaydataset_path=target_kd.conf_path,
                        n_chunk_jobs=n_chunk_jobs, hdf5names=[hdf5name],
                        n_max_co_processes=n_max_co_processes, workfolder=working_dir,
                        n_folders_fs=n_folders_fs, use_combined_extraction=True, suffix=suffix,
                        size=size, offset=offset, log=log)
