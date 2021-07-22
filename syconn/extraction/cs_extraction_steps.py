# -*- coding: utf-8 -*-
# distutils: language=c++
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Philipp Schubert, Joergen Kornfeld
from typing import Iterable, Dict, List, Sequence, Any

import gc
import glob
import os
import pickle as pkl
import shutil
import time
from collections import defaultdict
from logging import Logger
from typing import Optional, Dict, List, Tuple, Union, Callable

from multiprocessing import Process

import numpy as np
import scipy.ndimage
import tqdm
from knossos_utils import chunky
from knossos_utils import knossosdataset

from . import log_extraction
from .block_processing_C import extract_cs_syntype
from .object_extraction_steps import export_cset_to_kd_batchjob
from .object_extraction_wrapper import calculate_chunk_numbers_for_box
from .. import global_params
from ..backend.storage import AttributeDict, VoxelStorageDyn, VoxelStorage, CompressedStorage
from ..handler import compression, basics
from ..mp import batchjob_utils as qu
from ..mp.mp_utils import start_multiprocess_imap
from ..proc.sd_proc import _cache_storage_paths
from ..proc.sd_proc import merge_prop_dicts, dataset_analysis
from ..proc.image import apply_morphological_operations, get_aniso_struct
from ..reps import rep_helper
from ..reps import segmentation
from .find_object_properties import merge_type_dicts, detect_cs_64bit, detect_cs, find_object_properties, \
    find_object_properties_cs_64bit, merge_voxel_dicts, extract_cs_syntype_64bit


def extract_contact_sites(chunk_size: Optional[Tuple[int, int, int]] = None, log: Optional[Logger] = None,
                          max_n_jobs: Optional[int] = None, cube_of_interest_bb: Optional[np.ndarray] = None,
                          n_folders_fs: int = 1000, cube_shape: Optional[Tuple[int]] = None, overwrite: bool = False,
                          transf_func_sj_seg: Optional[Callable] = None):
    """
    Extracts contact sites and their overlap with ``sj`` objects and stores them in a
    :class:`~syconn.reps.segmentation.SegmentationDataset` of type ``cs`` and ``syn``
    respectively. If synapse type is available, this information will be stored
    as the voxel-ratio per class in the attribute dictionary of the ``syn``
    objects (keys: ``sym_prop``, ``asym_prop``). These properties will further be used by
    :func:`~syconn.extraction.cs_processing_steps.combine_and_split_syn` which aggregates per-SV synapse
    fragments (syn) to per-SSV synapses (syn_ssv).

    Examples:
        The synapse type labels and KnossosDatasets are defined in the `config.yml` file and can be set
        initially by changing the following attributes depending on how the synapse type prediction is stored.

        (i) The type prediction is stored as segmentation in a single data set with three
        labels (0: background, 1: symmetric, 2: asymmetric):

            kd_asym_path = root_dir + 'kd_asym_sym/'
            kd_sym_path = root_dir + 'kd_asym_sym/'
            key_val_pairs_conf = [
                ('cell_objects', {'sym_label': 1, 'asym_label': 2,}
                )
            ]
                generate_default_conf(working_dir, kd_sym=kd_sym_path, kd_asym=kd_asym_path,
                          key_value_pairs=key_val_pairs_conf)


        (ii) The type prediction is stored as segmentation in a two data sets each with two
        labels (0: background, 1: symmetric and 0: background, 1: asymmetric):

            kd_asym_path = root_dir + 'kd_asym/'
            kd_sym_path = root_dir + 'kd_sym/'
            key_val_pairs_conf = [
                ('cell_objects', {'sym_label': 1, 'asym_label': 1,}
                )
            ]
                generate_default_conf(working_dir, kd_sym=kd_sym_path, kd_asym=kd_asym_path,
                          key_value_pairs=key_val_pairs_conf)


        (iii) The type prediction is stored as probability map in the raw channel (uint8, range: 0..255)
        in a data set for each type:

            kd_asym_path = root_dir + 'kd_asym/'
            kd_sym_path = root_dir + 'kd_sym/'
            key_val_pairs_conf = [
                ('cell_objects', {'sym_label': None, 'asym_label': None,}
                )
            ]
                generate_default_conf(working_dir, kd_sym=kd_sym_path, kd_asym=kd_asym_path,
                          key_value_pairs=key_val_pairs_conf)

    Notes:
        * Deletes existing KnossosDataset and SegmentationDataset of type 'syn' and 'cs'!
        * Replaced ``find_contact_sites``, ``extract_agg_contact_sites``, `
          `syn_gen_via_cset`` and ``extract_synapse_type``.

    Args:
        chunk_size: Sub-cube volume which is processed at a time.
        log: Logger.
        max_n_jobs: Maximum number of jobs, only used as a lower bound.
        cube_of_interest_bb: Sub-volume of the data set which is processed.
            Default: Entire data set.
        n_folders_fs: Number of folders used for organizing supervoxel data.
        cube_shape: Cube shape used within 'syn' and 'cs' KnossosDataset.
        overwrite: Overwrite existing cache.
        transf_func_sj_seg: Method that converts the cell organelle segmentation into a binary mask of background vs.
            sj foreground.

    """
    if extract_cs_syntype is None:
        msg = '`extract_contact_sites` requires the cythonized method ' \
              '`extract_cs_syntype`.'
        log_extraction.error(msg)
        raise ImportError(msg)
    kd = basics.kd_factory(global_params.config.kd_seg_path)
    if cube_of_interest_bb is None:
        cube_of_interest_bb = [np.zeros(3, dtype=np.int32), kd.boundary]
    if cube_shape is None:
        cube_shape = (256, 256, 256)
    if chunk_size is None:
        chunk_size = (512, 512, 512)
    if np.any(np.array(chunk_size) % np.array(cube_shape)):
        raise ValueError(f'Chunk size ({chunk_size}) must be divisible by cube shape ({cube_shape}).')
    if max_n_jobs is None:
        max_n_jobs = global_params.config.ncore_total * 4
    size = cube_of_interest_bb[1] - cube_of_interest_bb[0] + 1
    offset = cube_of_interest_bb[0]

    if global_params.config.use_new_subfold:
        target_dir_func = rep_helper.subfold_from_ix_new
    else:
        target_dir_func = rep_helper.subfold_from_ix_OLD

    # Initital contact site extraction
    cd_dir = global_params.config.temp_path + "/chunkdatasets/cs/"
    # Class that contains a dict of chunks (with coordinates) after initializing it
    cset = chunky.ChunkDataset()
    cset.initialize(kd, kd.boundary, chunk_size, cd_dir,
                    box_coords=[0, 0, 0], fit_box_size=True)

    if log is None:
        log = log_extraction
    if size is not None and offset is not None:
        chunk_list, _ = \
            calculate_chunk_numbers_for_box(cset, offset, size)
    else:
        chunk_list = [ii for ii in range(len(cset.chunk_dict))]

    all_times = []
    step_names = []
    dict_paths_tmp = []
    dir_props = f"{global_params.config.temp_path}/tmp_props_cssyn/"

    # remove previous temporary results.
    if os.path.isdir(dir_props):
        if not overwrite:
            msg = f'Could not start extraction of supervoxel objects ' \
                  f'because temporary files already exist at "{dir_props}" ' \
                  f'and overwrite was set to False.'
            log.error(msg)
            raise FileExistsError(msg)
        log.debug(f'Found existing cache folder at {dir_props}. Removing it now.')
        shutil.rmtree(dir_props)
    os.makedirs(dir_props)

    # todo remove?
    # init KD for syn and cs
    #for ot in ['cs', 'syn']:
    #    path_kd = f"{global_params.config.working_dir}/knossosdatasets/{ot}_seg/"
    #    if os.path.isdir(path_kd):
    #        log.debug('Found existing KD at {}. Removing it now.'.format(path_kd))
    #        shutil.rmtree(path_kd)
    #    target_kd = knossosdataset.KnossosDataset()
    #    target_kd._cube_shape = cube_shape
    #    scale = np.array(global_params.config['scaling'])
    #    target_kd.scales = [scale, ]
    #    target_kd.initialize_without_conf(path_kd, kd.boundary, scale, kd.experiment_name,
    #                                      mags=[1, ], create_pyk_conf=True, create_knossos_conf=False)

    multi_params = []
    iter_params = basics.chunkify(chunk_list, max_n_jobs)
    for ii, chunk_k in enumerate(iter_params):
        multi_params.append([[cset.chunk_dict[k] for k in chunk_k],
                             global_params.config.kd_seg_path, ii, dir_props, transf_func_sj_seg])

    # reduce step
    start = time.time()
    cs_worker_dc_fname = f'{global_params.config.temp_path}/cs_worker_dict.pkl'
    dict_paths_tmp += [cs_worker_dc_fname, dir_props]
    syn_ids = []
    cs_ids = []
    # Contact sites cs are a superset of synaptic contacts syn.
    # Maps worker IDs to contact sites encountered by the respective worker (Nx2 ndarray, or empty ndarray)
    cs_worker_mapping = dict()  # type: Dict[int, np.ndarray]
    if qu.batchjob_enabled():
        path_to_out = qu.batchjob_script(multi_params, "contact_site_extraction", log=log, use_dill=True)
        out_files = glob.glob(path_to_out + "/*")

        for out_file in tqdm.tqdm(out_files, leave=False):
            with open(out_file, 'rb') as f:
                worker_nr, worker_res = pkl.load(f)
            syn_ids_curr = np.array(worker_res['syn'], dtype=np.uint64)
            cs_ids_curr = np.array(worker_res['cs'], dtype=np.uint64)
            syn_ids.append(syn_ids_curr)
            cs_ids.append(cs_ids_curr)
            cs_worker_mapping[worker_nr] = cs_ids_curr
    else:
        results = start_multiprocess_imap(_contact_site_extraction_thread,
                                          multi_params, verbose=False, debug=False)
        for worker_nr, worker_res in tqdm.tqdm(results, leave=False):
            syn_ids_curr = np.array(worker_res['syn'], dtype=np.uint64)
            cs_ids_curr = np.array(worker_res['cs'], dtype=np.uint64)
            syn_ids.append(syn_ids_curr)
            cs_ids.append(cs_ids_curr)
            cs_worker_mapping[worker_nr] = cs_ids_curr
        del results
    log_extraction.debug(f'Collected partial results from {len(cs_worker_mapping)} workers.')
    with open(cs_worker_dc_fname, 'wb') as f:
        pkl.dump(cs_worker_mapping, f, protocol=4)
    del cs_worker_mapping, cset

    syn_ids = np.unique(
        np.concatenate([xx for xx in syn_ids if xx.size > 0]), axis=0)  # type: np.ndarray[[Any, 2], np.uint64]
    n_syn = syn_ids.shape[0]
    del syn_ids

    cs_ids = np.unique(
        np.concatenate([xx for xx in cs_ids if xx.size > 0]), axis=0)  # type: np.ndarray[[Any, 2], np.uint64]
    n_cs = cs_ids.shape[0]

    # only required as syn is a subset of cs!
    dest_p = f'{global_params.config.temp_path}/storage_targets_cs.pkl'
    dict_paths_tmp.append(dest_p)

    _cache_storage_paths((dest_p, cs_ids, n_folders_fs))
    del cs_ids

    step_names.append("extract objects and collect properties of cs and syn.")
    all_times.append(time.time() - start)

    log.info(f'Finished extraction of initial contact sites (#objects: {n_cs}) and synapses'
             f' (#objects: {n_syn}).')
    if n_syn == 0:
        log.critical('WARNING: Did not find any synapses during extraction step.')

    # create folders for existing (sub-)cell supervoxels to prevent collisions using makedirs
    for ii, struct in enumerate(['cs', 'syn']):
        sc_sd = segmentation.SegmentationDataset(
            working_dir=global_params.config.working_dir, obj_type=struct,
            version=0, n_folders_fs=n_folders_fs)
        if os.path.isdir(sc_sd.path):
            log.debug('Found existing SD at {}. Removing it now.'.format(sc_sd.path))
            shutil.rmtree(sc_sd.path)
        ids = rep_helper.get_unique_subfold_ixs(n_folders_fs)
        for ix in tqdm.tqdm(ids, leave=False):
            curr_dir = sc_sd.so_storage_path + target_dir_func(
                ix, n_folders_fs)
            os.makedirs(curr_dir, exist_ok=True)

    # Write SD
    # todo remove?
    path = "{}/knossosdatasets/syn_seg/".format(global_params.config.working_dir)
    path_cs = "{}/knossosdatasets/cs_seg/".format(global_params.config.working_dir)
    storage_location_ids = rep_helper.get_unique_subfold_ixs(n_folders_fs)
    max_n_jobs = min(max_n_jobs, len(storage_location_ids))
    n_cores = 2 if qu.batchjob_enabled() else 1  # use additional cores for loading data from disk
    # slightly increase ncores per worker to compensate IO related downtime
    multi_params = [(sv_id_block, n_folders_fs, path, path_cs, dir_props, n_cores)
                    for sv_id_block in basics.chunkify(storage_location_ids, max_n_jobs)]
    if not qu.batchjob_enabled():
        start_multiprocess_imap(_write_props_to_syn_thread, multi_params, debug=False)
    else:
        qu.batchjob_script(multi_params, "write_props_to_syn", log=log,
                           n_cores=1, remove_jobfolder=True)
    # Mesh props are not computed as this is done for the agglomerated versions (only syn_ssv)
    sd_syn = segmentation.SegmentationDataset(working_dir=global_params.config.working_dir,
                                              obj_type='syn', version=0)
    sd_cs = segmentation.SegmentationDataset(working_dir=global_params.config.working_dir,
                                             obj_type='cs', version=0)
    da_kwargs = dict(recompute=False, compute_meshprops=False)
    procs = [Process(target=dataset_analysis, args=(sd_syn,), kwargs=da_kwargs),
             Process(target=dataset_analysis, args=(sd_cs,), kwargs=da_kwargs)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
        if p.exitcode != 0:
            raise Exception(f'Worker {p.name} stopped unexpectedly with exit '
                            f'code {p.exitcode}.')
        p.close()

    # remove temporary files
    for p in dict_paths_tmp:
        if os.path.isfile(p):
            os.remove(p)
        elif os.path.isdir(p):
            shutil.rmtree(p)
    shutil.rmtree(cd_dir, ignore_errors=True)
    if qu.batchjob_enabled():
        jobfolder = os.path.abspath(f'{path_to_out}/../')
        try:
            shutil.rmtree(jobfolder, ignore_errors=False)
        except Exception as e:
            log.error(f'Could not delete job folder at "{jobfolder}". {str(e)}')


def _contact_site_extraction_thread(args: Union[tuple, list]) \
        -> Tuple[int, Dict[str, List[dict]]]:
    """
    Helper function to extract properties of ``cs`` and ``syn`` objects.

    Args:
        args:
            * ``Chunk`` objects
            * Path to KnossosDataset containing the cell supervoxels.

    Todo:
        * Get rid of the second argument -> use config parameter instead.

    Returns:
        Two lists of dictionaries (representative coordinates, bounding box and
        voxel count) for ``cs`` and ``syn`` objects, per-synapse counts of
        symmetric and asymmetric voxels.
    """
    chunks = args[0]
    knossos_path = args[1]
    worker_nr = args[2]
    dir_props = args[3]
    transf_func_sj_seg = args[4]
    worker_dir_props = f"{dir_props}/{worker_nr}/"
    os.makedirs(worker_dir_props, exist_ok=True)

    morph_ops = global_params.config['cell_objects']['extract_morph_op']
    scaling = np.array(global_params.config['scaling'])
    struct = get_aniso_struct(scaling)

    if global_params.config.syntype_available and \
            (global_params.config.sym_label == global_params.config.asym_label) and \
            (global_params.config.kd_sym_path == global_params.config.kd_asym_path):
        raise ValueError('Both KnossosDatasets and labels for symmetric and '
                         'asymmetric synapses are identical. Either one '
                         'must differ.')

    # init target KD for cs and syn segmentation
    # todo remove?
    #kd_cs = basics.kd_factory(f"{global_params.config.working_dir}/knossosdatasets/cs_seg/")
    #kd_syn = basics.kd_factory(f"{global_params.config.working_dir}/knossosdatasets/syn_seg/")

    # init. synaptic junction (sj) KD
    kd_sj = basics.kd_factory(global_params.config.kd_sj_path)
    # init synapse type KD if available
    if global_params.config.syntype_available:
        kd_syntype_sym = basics.kd_factory(global_params.config.kd_sym_path)
        kd_syntype_asym = basics.kd_factory(global_params.config.kd_asym_path)
    else:
        kd_syntype_sym, kd_syntype_asym = None, None

    # cell segmentation
    kd = basics.kd_factory(knossos_path)

    cs_props = [{}, defaultdict(list), {}]
    syn_props = [{}, defaultdict(list), {}]
    syn_voxels = {}
    tot_sym_cnt = {}
    tot_asym_cnt = {}
    cs_filtersize = np.array(global_params.config['cell_objects']['cs_filtersize'])
    cs_dilation = global_params.config['cell_objects']['cs_dilation']
    # stencil_offset is used to load more data as these will be cropped when performing detect_cs(_64bit)
    stencil_offset = cs_filtersize // 2
    # additional overlap, e.g. to prevent boundary artifacts by dilation/closing
    overlap = max(stencil_offset)
    for chunk in chunks:
        offset = np.array(chunk.coordinates - overlap)  # also used for loading synapse data
        size = 2 * overlap + np.array(chunk.size)  # also used for loading synapse data
        data = kd.load_seg(size=size + 2 * stencil_offset,
                           offset=offset - stencil_offset,
                           mag=1, datatype=np.uint64).swapaxes(0, 2)

        # contacts has size as given with `size`, because detect_cs performs valid conv.
        # -> contacts result is cropped by stencil_offset on each side
        contacts = np.asarray(detect_cs_64bit(data))

        if transf_func_sj_seg is None:
            sj_d = (kd_sj.load_raw(size=size, offset=offset, mag=1).swapaxes(0, 2) >
                    255 * global_params.config['cell_objects']["probathresholds"]['sj']).astype(np.uint8, copy=False)
        else:
            sj_d = transf_func_sj_seg(
                kd_sj.load_seg(size=size, offset=offset, mag=1).swapaxes(0, 2)).astype(np.uint8, copy=False)
        # apply mrophological operations on sj binary mask
        if 'sj' in morph_ops:
            sj_d = apply_morphological_operations(
                sj_d.copy(), morph_ops['sj'], mop_kwargs=dict(structure=struct)).astype(np.uint8, copy=False)

        # get binary mask for symmetric and asymmetric syn. type per voxel
        if global_params.config.syntype_available:
            if global_params.config.kd_asym_path != global_params.config.kd_sym_path:
                # TODO: add thresholds to global_params
                if global_params.config.sym_label is None:
                    sym_d = (kd_syntype_sym.load_raw(size=size, offset=offset, mag=1).swapaxes(0, 2)
                             >= 123).astype(np.uint8, copy=False)
                else:
                    sym_d = (kd_syntype_sym.load_seg(size=size, offset=offset, mag=1).swapaxes(0, 2)
                             == global_params.config.sym_label).astype(np.uint8, copy=False)

                if global_params.config.asym_label is None:
                    asym_d = (kd_syntype_asym.load_raw(size=size, offset=offset, mag=1).swapaxes(0, 2)
                              >= 123).astype(np.uint8, copy=False)
                else:
                    asym_d = (kd_syntype_asym.load_seg(size=size, offset=offset, mag=1).swapaxes(0, 2)
                              == global_params.config.asym_label).astype(np.uint8, copy=False)
            else:
                assert global_params.config.asym_label is not None, \
                    'Label of asymmetric synapses is not set.'
                assert global_params.config.sym_label is not None, \
                    'Label of symmetric synapses is not set.'
                # load synapse type classification results stored in the same KD
                sym_d = kd_syntype_sym.load_seg(size=size, offset=offset, mag=1).swapaxes(0, 2)
                # create copy
                asym_d = np.array(sym_d == global_params.config.asym_label, dtype=np.uint8)
                sym_d = np.array(sym_d == global_params.config.sym_label, dtype=np.uint8)
        else:
            sym_d = np.zeros_like(sj_d)
            asym_d = np.zeros_like(sj_d)

        # close gaps of contact sites prior to overlapping synaptic junction map with contact sites

        # returns rep. coords, bounding box and size for every ID in contacts
        # used to get location of every contact site to perform closing operation
        _, bb_dc, _ = find_object_properties_cs_64bit(contacts)
        (bb_dc, ) = _nested_dicts_to_tuple_dicts((bb_dc, ))
        n_closings = overlap
        for ix in bb_dc.keys():
            obj_start, obj_end = np.array(bb_dc[ix])
            obj_start -= n_closings
            obj_start[obj_start < 0] = 0
            obj_end += n_closings
            # create slice obj
            new_obj_slices = tuple(slice(obj_start[ii], obj_end[ii], None) for
                                   ii in range(3))
            sub_vol = contacts[new_obj_slices]
            binary_mask = (sub_vol == ix)[..., 0].astype(np.int8, copy=False)
            res = scipy.ndimage.binary_closing(
                binary_mask, iterations=n_closings)
            # reduce fragmenting of contact sites
            res = scipy.ndimage.binary_dilation(res, iterations=cs_dilation)
            # only update background or the object itself and do not remove object voxels (res == 1), e.g. at boundary
            proc_mask = ((binary_mask == 1) | (sub_vol[..., 0] == 0)) & (res == 1)
            contacts[new_obj_slices][proc_mask] = res[proc_mask, np.newaxis] * ix

        # this counts SJ foreground voxels overlapping with the CS objects
        # and the asym and sym voxels, do not use overlap here!
        curr_cs_p, curr_syn_p, asym_cnt, sym_cnt, curr_syn_vx = extract_cs_syntype_64bit(
            contacts[overlap:-overlap, overlap:-overlap, overlap:-overlap],
            sj_d[overlap:-overlap, overlap:-overlap, overlap:-overlap],
            asym_d[overlap:-overlap, overlap:-overlap, overlap:-overlap],
            # overlap was removed; use correct offset for the analysis of the object properties
            sym_d[overlap:-overlap, overlap:-overlap, overlap:-overlap], offset=offset + overlap)
        # change nested dicts to dict with tuple keys: dict[tuple(id1, id2)] = ...
        tuple_dicts = _nested_dicts_to_tuple_dicts((*curr_cs_p, *curr_syn_p, asym_cnt, sym_cnt, curr_syn_vx))
        curr_cs_p, curr_syn_p, asym_cnt, sym_cnt, curr_syn_vx = (
            tuple_dicts[0:3],
            tuple_dicts[3:6],
            tuple_dicts[6],
            tuple_dicts[7],
            tuple_dicts[8])

        # todo remove
        #kd_cs.save_seg(offset=offset + overlap, mags=[1, ],
        #               data=contacts[overlap:-overlap, overlap:-overlap, overlap:-overlap].swapaxes(0, 2),
        #               data_mag=1)
        ## syn segmentation contains the intersecting voxels between SJ and CS
        #contacts[sj_d == 0] = 0
        #kd_syn.save_seg(offset=offset + overlap, mags=[1, ],
        #                data=contacts[overlap:-overlap, overlap:-overlap, overlap:-overlap].swapaxes(0, 2),
        #                data_mag=1)

        # overlap was removed; use correct offset for the analysis of the object properties
        merge_prop_dicts([cs_props, curr_cs_p], offset=offset + overlap)
        merge_prop_dicts([syn_props, curr_syn_p], offset=offset + overlap)
        merge_voxel_dicts([syn_voxels, curr_syn_vx], key_to_str=True)
        merge_type_dicts([tot_asym_cnt, asym_cnt])
        merge_type_dicts([tot_sym_cnt, sym_cnt])
        del curr_cs_p, curr_syn_p, asym_cnt, sym_cnt

    # Ugly - the numba typed.List(s) in this dict can't be pickled, so converting them to pure python here
    syn_voxels = {k: [[i for i in vv] for vv in v] for k, v in syn_voxels.items()}
    basics.write_obj2pkl(f'{worker_dir_props}/cs_props_{worker_nr}.pkl', cs_props)
    basics.write_obj2pkl(f'{worker_dir_props}/syn_props_{worker_nr}.pkl', syn_props)
    np.savez(f'{worker_dir_props}/syn_voxels_{worker_nr}.npz', **syn_voxels)
    basics.write_obj2pkl(f'{worker_dir_props}/tot_asym_cnt_{worker_nr}.pkl', tot_asym_cnt)
    basics.write_obj2pkl(f'{worker_dir_props}/tot_sym_cnt_{worker_nr}.pkl', tot_sym_cnt)

    return worker_nr, dict(cs=list(cs_props[0].keys()), syn=list(syn_props[0].keys()))


def _nested_dicts_to_tuple_dicts(in_dicts: Sequence[Dict]) -> Tuple[Dict, ...]:
    """
    Re-pack Sequence of dictionaries of the form Dict[k1, Dict[k2, ...]] into Tuple of dictionaries of the form
    Dict[(k1, k2), ...].
    """

    out_dicts = tuple(dict() for _ in range(len(in_dicts)))

    for cur_out_dict, cur_in_dict in zip(out_dicts, in_dicts):
        for k_1, v_1 in cur_in_dict.items():
            for k_2 in v_1.keys():
                tuple_key = (k_1, k_2)
                cur_out_dict[tuple_key] = cur_in_dict[k_1][k_2]

    return out_dicts


def _row_wise_intersect(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    # Credit for this trick to Joe Kington
    # https://stackoverflow.com/questions/8317022/get-intersecting-rows-across-two-2d-numpy-arrays
    #

    _, ncols = A.shape
    dtype = {'names': ['f{}'.format(i) for i in range(ncols)],
             'formats': ncols * [A.dtype]}

    C = np.intersect1d(A.view(dtype), B.view(dtype))
    C = C.view(A.dtype).reshape(-1, ncols)

    return C


# iterate over the subcellular SV ID chunks
def _write_props_to_syn_thread(args):
    cs_ids_ch = args[0]  # type: List[int]
    n_folders_fs = args[1]  # type: int
    knossos_path = args[2]  # type: str
    knossos_path_cs = args[3]  # type: str
    dir_props = args[4]  #type: str
    if len(args) < 6:
        nb_cores = 4
    else:
        nb_cores = args[5]
    min_obj_vx_dc = global_params.config['cell_objects']['min_obj_vx']
    tmp_path = global_params.config.temp_path
    if global_params.config.use_new_subfold:
        target_dir_func = rep_helper.subfold_from_ix_new
    else:
        target_dir_func = rep_helper.subfold_from_ix_OLD

    for obj_id_mod in cs_ids_ch:
        # get destination paths for the current objects
        dest_dc_tmp = CompressedStorage(f'{tmp_path}/storage_targets_cs.pkl', disable_locking=True)
        k = target_dir_func(obj_id_mod, n_folders_fs)
        if k not in dest_dc_tmp or len(dest_dc_tmp[k]) == 0:
            continue

        obj_keys = dest_dc_tmp[k]  # type: np.ndarray[[Any, 2], np.uint64]
        del dest_dc_tmp

        sd = segmentation.SegmentationDataset(
            n_folders_fs=n_folders_fs, obj_type='syn', working_dir=global_params.config.working_dir, version=0)
        sd_cs = segmentation.SegmentationDataset(
            n_folders_fs=n_folders_fs, obj_type='cs',working_dir=global_params.config.working_dir, version=0)

        # The bbox dicts (cs_props[1] and syn_props[1]) here will be Dict[List[List[np.ndarray]]]. Note the
        # nested lists.
        cs_props = [{}, defaultdict(list), {}]
        syn_props = [{}, defaultdict(list), {}]
        cs_sym_cnt = {}
        cs_asym_cnt = {}
        syn_voxels = {}

        # get cached worker lookup
        with open(f'{global_params.config.temp_path}/cs_worker_dict.pkl', "rb") as f:
            cs_workers_tmp = pkl.load(f)  # type: Dict[int, np.ndarray[[Any, 2], np.uint64]]

        params = []
        for worker_id, obj_ids in cs_workers_tmp.items():
            # CS IDs of interest - found by worker worker_id, and belonging to current storage folder (in other words,
            # we're mapping from a chunking in voxel space to a chunking in ID space).
            current_chunk_relevant_ids = [tuple(xx) for xx in _row_wise_intersect(obj_ids, obj_keys)]
            if current_chunk_relevant_ids:
                params.append([dir_props, worker_id, current_chunk_relevant_ids])

        del cs_workers_tmp
        res = start_multiprocess_imap(_write_props_collect_helper, params, nb_cpus=nb_cores, show_progress=False,
                                      debug=False)
        for tmp_dcs_cs, tmp_dcs_syn, tmp_sym_dc, tmp_asym_dc, tmp_syn_vxs in res:
            if len(tmp_dcs_cs) == 0:
                continue
            # cs
            merge_prop_dicts([cs_props, tmp_dcs_cs])
            del tmp_dcs_cs
            # syn
            merge_prop_dicts([syn_props, tmp_dcs_syn])
            del tmp_dcs_syn
            merge_type_dicts([cs_asym_cnt, tmp_asym_dc])
            del tmp_asym_dc
            merge_type_dicts([cs_sym_cnt, tmp_sym_dc])
            del tmp_sym_dc
            merge_voxel_dicts([syn_voxels, tmp_syn_vxs], key_to_str=False)
            del tmp_syn_vxs

        # get dummy segmentation object to fetch attribute dictionary for this batch of object IDs
        dummy_so = sd.get_segmentation_object(obj_id_mod)
        attr_p = dummy_so.attr_dict_path
        vx_p = dummy_so.voxel_path
        this_attr_dc = AttributeDict(attr_p, read_only=False, disable_locking=True)
        # this class is only used to query the voxel data
        voxel_dc = VoxelStorageDyn(vx_p, voxel_mode=False, read_only=False, disable_locking=True)

        # get dummy CS segmentation object to fetch attribute dictionary for this batch of object IDs
        dummy_so_cs = sd_cs.get_segmentation_object(obj_id_mod)
        attr_p_cs = dummy_so_cs.attr_dict_path
        vx_p_cs = dummy_so_cs.voxel_path
        this_attr_dc_cs = AttributeDict(attr_p_cs, read_only=False, disable_locking=True)
        voxel_dc_cs = VoxelStorageDyn(vx_p_cs, voxel_mode=False, read_only=False, disable_locking=True)
        ids_to_load_voxels = []
        for cs_id in obj_keys:
            cs_id = tuple(cs_id)
            # write cs to dict
            size_cs = cs_props[2][cs_id]
            if size_cs < min_obj_vx_dc['cs']:
                continue
            rp_cs = cs_props[0][cs_id]
            # Note the nested list here, this is why the concatenate and min/max goes along the right dimension to find
            # the min / max corners.
            bboxes = cs_props[1][cs_id]  # type: List[List[np.ndarray]]
            bbs_cs = np.concatenate(bboxes)
            this_attr_dc_cs[cs_id]["rep_coord"] = rp_cs
            cur_bbox = np.array([bbs_cs[:, 0].min(axis=0), bbs_cs[:, 1].max(axis=0)])
            this_attr_dc_cs[cs_id]["bounding_box"] = cur_bbox
            this_attr_dc_cs[cs_id]["size"] = size_cs
            voxel_dc_cs[cs_id] = bbs_cs
            voxel_dc_cs.increase_object_size(cs_id, size_cs)
            voxel_dc_cs.set_object_repcoord(cs_id, rp_cs)

            # write syn to dict
            if cs_id not in syn_props[0] or syn_props[2][cs_id] < min_obj_vx_dc['syn']:
                continue
            rp = syn_props[0][cs_id]
            bbs = np.concatenate(syn_props[1][cs_id])
            size = syn_props[2][cs_id]
            this_attr_dc[cs_id]["rep_coord"] = rp
            bb = np.array([bbs[:, 0].min(axis=0), bbs[:, 1].max(axis=0)])
            this_attr_dc[cs_id]["bounding_box"] = bb
            this_attr_dc[cs_id]["size"] = size
            try:
                sym_prop = cs_sym_cnt[cs_id] / size
            except KeyError:
                sym_prop = 0
            try:
                asym_prop = cs_asym_cnt[cs_id] / size
            except KeyError:
                asym_prop = 0
            this_attr_dc[cs_id]["sym_prop"] = sym_prop
            this_attr_dc[cs_id]["asym_prop"] = asym_prop

            cs_ratio_vx = size / size_cs  # number of overlap voxels (syn voxels) divided by cs size
            # inverse 'CS' density: c_cs_ids[u_cs_ids == 0] / n_vxs_in_sjbb  (previous version)
            # cs_id is the same as the syn_id, not necessary to store this
            add_feat_dict = {'cs_id': cs_id,
                             'id_cs_ratio': cs_ratio_vx,
                             'cs_size': size_cs}
            this_attr_dc[cs_id].update(add_feat_dict)
            voxel_dc[cs_id] = bbs
            voxel_dc.increase_object_size(cs_id, size)
            voxel_dc.set_object_repcoord(cs_id, rp)
            ids_to_load_voxels.append(cs_id)
            # # write voxels explicitly - this assumes reasonably sized synapses
            voxel_dc.set_voxel_cache(cs_id, np.array(syn_voxels[cs_id], dtype=np.uint32))
            del syn_voxels[cs_id]

        voxel_dc.push()
        voxel_dc_cs.push()
        this_attr_dc.push()
        this_attr_dc_cs.push()


def _write_props_collect_helper(args) -> Tuple[List[dict], List[dict], dict, dict, dict]:
    dir_props = args[0]  # type: str
    worker_id = args[1]  # type: int
    intersec = args[2]  # type: List[Tuple[int, int]]

    if len(intersec) == 0:
        return [{}, {}, {}], [{}, {}, {}], {}, {}, {}
    worker_dir_props = f'{dir_props}/{worker_id}/'

    # Contact Sites (cs)
    dc = basics.load_pkl2obj(f'{worker_dir_props}/cs_props_{worker_id}.pkl')

    cs_rep_coord = dc[0]  # type: Dict[Tuple[int, int], np.ndarray]
    cs_bbox = dc[1]  # type: Dict[Tuple[int, int], List[np.ndarray]]
    cs_size = dc[2]  # type: Dict[Tuple[int, int], int]

    cs_rep_coord_filtered, cs_bbox_filtered, cs_size_filtered = dict(), defaultdict(list), dict()
    cs_filtered = [cs_rep_coord_filtered, cs_bbox_filtered, cs_size_filtered]
    for k in intersec:
        cs_rep_coord_filtered[k] = cs_rep_coord[k]
        cs_bbox_filtered[k] = cs_bbox[k]
        cs_size_filtered[k] = cs_size[k]
    del dc

    # Synapses (syn)
    dc_syn = basics.load_pkl2obj(f'{worker_dir_props}/syn_props_{worker_id}.pkl')
    curr_sym_cnt = basics.load_pkl2obj(f'{worker_dir_props}/tot_sym_cnt_{worker_id}.pkl')
    curr_asym_cnt = basics.load_pkl2obj(f'{worker_dir_props}/tot_asym_cnt_{worker_id}.pkl')
    curr_syn_vxs = np.load(f'{worker_dir_props}/syn_voxels_{worker_id}.npz')

    syn_rep_coord = dc_syn[0]  # type: Dict[Tuple[int, int], np.ndarray]
    syn_bbox = dc_syn[1]  # type: Dict[Tuple[int, int], List[np.ndarray]]
    syn_size = dc_syn[2]  # type: Dict[Tuple[int, int], int]

    syn_rep_coord_filtered, syn_bbox_filtered, syn_size_filtered = dict(), defaultdict(list), dict()
    syn_filtered = [syn_rep_coord_filtered, syn_bbox_filtered, syn_size_filtered]
    filtered_syn_vxs = dict()

    filtered_sym_dc = dict()
    filtered_asym_dc = dict()
    for k in intersec:
        if k not in syn_rep_coord:
            continue
        syn_rep_coord_filtered[k] = syn_rep_coord[k]
        syn_bbox_filtered[k] = syn_bbox[k]
        syn_size_filtered[k] = syn_size[k]
        filtered_syn_vxs[k] = curr_syn_vxs[str(k)]  # np.savez supports string keys only
        if k in curr_sym_cnt:
            filtered_sym_dc[k] = curr_sym_cnt[k]
        if k in curr_asym_cnt:
            filtered_asym_dc[k] = curr_asym_cnt[k]
    return cs_filtered, syn_filtered, filtered_sym_dc, filtered_asym_dc, filtered_syn_vxs


def _generate_storage_lookup(args):
    """
    Generates a look-up dictionary for given storage destinations to corresponding
    object IDs in `id_chunk` (used for SegmentationObjects) by calling
    `rep_helper.subfold_from_ix`

    Parameters
    ----------
    args : List or Tuple
        id_chunk: SegmentationObject IDs
        req_subfold_keys : keys of requested storages
        n_folders_fs : number of folders

    Returns
    -------
    Dict
        look-up dictionary: [key -> value] storage destination -> list of IDs
    """
    id_chunk, req_subfold_keys, n_folders_fs = args
    dest_dc_tmp = defaultdict(list)
    cs_ids_ch_set = set(req_subfold_keys)
    for obj_id in id_chunk:
        subfold_key = rep_helper.subfold_from_ix(obj_id, n_folders_fs)
        if subfold_key in cs_ids_ch_set:
            dest_dc_tmp[subfold_key].append(obj_id)
    return dest_dc_tmp
