# -*- coding: utf-8 -*-
# distutils: language=c++
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Philipp Schubert, Joergen Kornfeld

from collections import defaultdict
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
from ..handler import compression, basics
from ..reps import segmentation
from ..handler.basics import kd_factory, chunkify
from . object_extraction_steps import export_cset_to_kd_batchjob
from . import log_extraction
from .object_extraction_wrapper import from_ids_to_objects, calculate_chunk_numbers_for_box
from ..mp.mp_utils import start_multiprocess_imap
try:
    from .block_processing_C import process_block, process_block_nonzero, extract_cs_syntype
except ImportError as e:
    extract_cs_syntype = None
    log_extraction.warning('Could not import cython version of `block_processing`.')
    from .block_processing import process_block, process_block_nonzero
from ..proc.sd_proc import merge_prop_dicts, dataset_analysis
from .. import global_params


def extract_contact_sites(cset, knossos_path, filename='cs', n_max_co_processes=None,
                          size=None, offset=None, log=None, max_n_jobs=None):
    """
    # Replaces `find_contact_sites`, `extract_agg_contact_sites`, `syn_gen_via_cset`
    and `extract_synapse_type`.

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
    if extract_cs_syntype is None:
        raise ImportError('`extract_contact_sites` requires the cythonized method '
                          '`extract_cs_syntype`. Use `find_contact_sites` and others for '
                          'contact site processing.')
    if max_n_jobs is None:
        max_n_jobs = global_params.NCORE_TOTAL * 2
    if log is None:
        log = log_extraction
    if size is not None and offset is not None:
        chunk_list, _ = \
            calculate_chunk_numbers_for_box(cset, offset, size)
    else:
        chunk_list = [ii for ii in range(len(cset.chunk_dict))]

    os.makedirs(cset.path_head_folder, exist_ok=True)
    multi_params = []
    # TODO: currently pickles Chunk objects -> job submission might be slow
    for chunk_k in chunkify(chunk_list, max_n_jobs):
        multi_params.append([[cset.chunk_dict[k] for k in chunk_k],
                             knossos_path, filename])

    if not qu.batchjob_enabled():
        results = start_multiprocess_imap(_contact_site_detection_thread, multi_params,
                                    debug=False, nb_cpus=n_max_co_processes)
    else:
        path_to_out = qu.QSUB_script(multi_params, "contact_site_extraction",
                           n_max_co_processes=n_max_co_processes)
        out_files = glob.glob(path_to_out + "/*")
        results = []
        for out_file in out_files:
            with open(out_file, 'rb') as f:
                results.append(pkl.load(f))

    # reduce step
    cs_props = [{}, defaultdict(list), {}]
    tot_sym_cnt = {}
    tot_asym_cnt = {}
    for curr_props, asym_cnt, sym_cnt in results:
        merge_prop_dicts([cs_props, curr_props])
        merge_type_dicts([tot_asym_cnt, asym_cnt])
        merge_type_dicts([tot_sym_cnt, sym_cnt])

    # TODO: extract syn objects! maybe replace sj_0 Segmentation dataset by the overlapping CS<->
    #  sj objects -> run syn. extraction and sd_generation in parallel and return mi_0, vc_0 and
    #  syn_0 -> use syns as new sjs during rendering!
    dict_paths = []
    # dump intermediate results
    dict_p = "{}/cs_prop_dict.pkl".format(global_params.config.temp_path)
    with open(dict_p, "wb") as f:
        pkl.dump(cs_props, f)
    del cs_props
    dict_paths.append(dict_p)

    # convert counting dicts to store ratio of syn. type voxels
    dict_p = "{}/cs_sym_cnt.pkl".format(global_params.config.temp_path)
    with open(dict_p, "wb") as f:
        pkl.dump(tot_sym_cnt, f)
    del tot_sym_cnt
    dict_paths.append(dict_p)

    dict_p = "{}/cs_asym_cnt.pkl".format(global_params.config.temp_path)
    with open(dict_p, "wb") as f:
        pkl.dump(tot_asym_cnt, f)
    del tot_asym_cnt
    dict_paths.append(dict_p)

    # writing CS properties to SD
    multi_params = [(sv_id_block, n_folders_fs) for sv_id_block in basics.chunkify(np.arange(n_folders_fs), n_chunk_jobs)]
    if not qu.batchjob_enabled():
        sm.start_multiprocess_imap(_write_props_to_cs_thread, multi_params,
                                   nb_cpus=n_max_co_processes, debug=False)
    else:
        qu.QSUB_script(multi_params, "write_props_to_cs", script_folder=None,
                       n_cores=n_cores, n_max_co_processes=n_max_co_processes)
    sd = segmentation.SegmentationDataset(working_dir=global_params.config.working_dir,
                                             obj_type="cs", version=0)
    dataset_analysis(sd, recompute=True, compute_meshprops=False)

    # write CS segmentation to KD
    chunky.save_dataset(cset)
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
        cset, target_kd, '{}'.format(filename), ['cs'],
        offset=offset, size=size, stride=[4 * 128, 4 * 128, 4 * 128], as_raw=False,
        orig_dtype=np.uint64, unified_labels=False,
        n_max_co_processes=n_max_co_processes)
    log.debug('Finished conversion of ChunkDataset ({}) into KnossosDataset ({})'.format(
        cset.path_head_folder, target_kd.knossos_path))
    for p in dict_paths:
        os.remove(p)


def _contact_site_extraction_thread(args):
    chunks = args[0]
    knossos_path = args[1]
    filename = args[2]

    kd = kd_factory(knossos_path)
    kd_syn = kd_factory(global_params.config.kd_sj_path)
    kd_syntype_sym = kd_factory(global_params.config.kd_sym_path)
    kd_syntype_asym = kd_factory(global_params.config.kd_asym_path)
    cs_props = [{}, defaultdict(list), {}]
    tot_sym_cnt = {}
    tot_asym_cnt = {}
    for chunk in chunks:
        overlap = np.array([6, 6, 3], dtype=np.int)  # TODO: check shape after `detect_cs` ->
        # valid convolution, i.e. is `contacts` actually of shape `chunk.size`?
        offset = np.array(chunk.coordinates - overlap)
        size = 2 * overlap + np.array(chunk.size)
        data = kd.from_overlaycubes_to_matrix(size, offset, datatype=np.uint64).astype(np.uint32)
        contacts = detect_cs(data)
        # store syn information as: synaptic voxel (1), symmetric type (2) and asymmetric type (3)
        syn_d = (kd_syn.from_overlaycubes_to_matrix(
            size, offset, datatype=np.uint64) > 0).astype(np.uint8)

        # TODO: adapt in case data format of input changes!
        # get binary mask for symmetric and asymmetric syn. type per voxel
        sym_d = (kd_syntype_sym.from_raw_cubes_to_matrix(size, offset)[..., None] >= 123).astype(
            np.uint8)
        asym_d = (kd_syntype_asym.from_raw_cubes_to_matrix(size, offset)[..., None] >= 123).astype(
            np.uint8)

        curr_props, asym_cnt, sym_cnt = extract_cs_syntype(contacts, syn_d, type_d)  # this
        # counts SJ
        # foreground voxels overlapping with the CS objects and the asym and sym voxels
        raise()
        os.makedirs(chunk.folder, exist_ok=True)
        # TODO: save syn.h5 as well!

        compression.save_to_h5py([contacts],
                                 chunk.folder + filename +
                                 ".h5", [filename])
        merge_prop_dicts([cs_props, curr_props])
        merge_type_dicts([tot_asym_cnt, asym_cnt])
        merge_type_dicts([tot_sym_cnt, sym_cnt])
        del curr_props, asym_cnt, sym_cnt
    return cs_props, tot_asym_cnt, tot_sym_cnt


def convert_nvox2ratio_syntype(syn_cnts, sym_cnts, asym_cnts):
    """get ratio of sym. and asym. voxels to the synaptic foreground voxels of each contact site
    object.
    Sym. and asym. ratios do not necessarily sum to 1 if types are predicted independently
    """
    # TODO consider to implement in cython

    sym_ratio = {}
    asym_ratio = {}
    for cs_id, cnt in syn_cnts.items():
        if cs_id in sym_cnts:
            sym_ratio[cs_id] = sym_cnts[cs_id] / cnt
        else:
            sym_ratio[cs_id] = 0
        if cs_id in asym_cnts:
            asym_ratio[cs_id] = asym_cnts[cs_id] / cnt
        else:
            asym_ratio[cs_id] = 0
    return asym_ratio, sym_ratio


def merge_type_dicts(type_dicts):
    """
    Merge map dictionaries in-place. Values will be stored in first dictionary

    Parameters
    ----------
    type_dicts

    Returns
    -------

    """
    tot_map = type_dicts[0]
    for el in type_dicts[1:]:
        # iterate over subcell. ids with dictionaries as values which store
        # the number of overlap voxels to cell SVs
        for cs_id, cnt in el.items():
            if cs_id in tot_map:
                tot_map[cs_id] += cnt
            else:
                tot_map[cs_id] = cnt


def find_contact_sites(cset, knossos_path, filename='cs', n_max_co_processes=None,
                       size=None, offset=None):
    """
    # TODO: add additional chunk-chunking (less number of submitted jobs)

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
                           n_max_co_processes=n_max_co_processes)
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
