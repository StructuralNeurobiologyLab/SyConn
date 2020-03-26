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
from typing import Optional, Dict, List, Tuple, Union
import time
import shutil
import tqdm
from logging import Logger
import glob
import numpy as np
import scipy.ndimage
from knossos_utils import knossosdataset
from knossos_utils import chunky
knossosdataset._set_noprint(True)
import os
from ..backend.storage import AttributeDict, VoxelStorageDyn, VoxelStorage, CompressedStorage
from ..reps import rep_helper
from ..mp import batchjob_utils as qu
from ..handler import compression, basics
from ..reps import segmentation
from . object_extraction_steps import export_cset_to_kd_batchjob
from . import log_extraction
from .object_extraction_wrapper import from_ids_to_objects, calculate_chunk_numbers_for_box
from ..mp.mp_utils import start_multiprocess_imap
from ..proc.sd_proc import _cache_storage_paths
from ..proc.image import multi_mop_backgroundonly
import multiprocessing
try:
    from .block_processing_C import process_block_nonzero as process_block_nonzero_C
    from .block_processing_C import extract_cs_syntype

    def process_block_nonzero(*args):
        return np.asarray(process_block_nonzero_C(*args))

except ImportError as e:
    extract_cs_syntype = None
    log_extraction.warning('Could not import cython version of `block_processing`. {}'.format(
        str(e)))
    from .block_processing import process_block_nonzero
from ..proc.sd_proc import merge_prop_dicts, dataset_analysis
from .. import global_params


def extract_contact_sites(n_max_co_processes: Optional[int] = None,
                          chunk_size: Optional[Tuple[int, int, int]] = None,
                          log: Optional[Logger] = None,
                          max_n_jobs: Optional[int] = None,
                          cube_of_interest_bb: Optional[np.ndarray] = None,
                          n_folders_fs: int = 1000,
                          cube_shape: Optional[Tuple[int]] = None,
                          overwrite: bool = False):
    """
    Extracts contact sites and their overlap with `sj` objects and stores them in a
    :class:`~syconn.reps.segmentation.SegmentationDataset` of type ``cs`` and ``syn``
    respectively. If synapse type is available, this information will be stored
    as the voxel-ratio per class in the attribute dictionary of the ``syn``
    objects (keys: ``sym_prop``, ``asym_prop``).

    Todo:
        extract syn objects! maybe replace sj_0 Segmentation dataset by the overlapping CS<->
        sj objects -> run syn. extraction and sd_generation in parallel and return mi_0, vc_0 and
        syn_0 -> use syns as new sjs during rendering!
        -> Run CS generation in parallel with mapping to at least get the syn objects before
        rendering the neuron views (which need subcellular structures, there one can then use mi,
        vc and syn (instead of sj)).

    Notes:
        Replaced ``find_contact_sites``, ``extract_agg_contact_sites``, `
        `syn_gen_via_cset`` and ``extract_synapse_type``.

    Args:
        n_max_co_processes: Number of parallel workers.
        chunk_size: Sub-cube volume which is processed at a time.
        log: Logger.
        max_n_jobs: Maximum number of jobs, only used as a lower bound.
        cube_of_interest_bb: Sub-volume of the data set which is processed.
            Default: Entire data set.
        n_folders_fs: Number of folders used for organizing supervoxel data.
        cube_shape: Cube shape used within contact site KnossosDataset.
        overwrite: Overwrite existing cache.

    """
    if extract_cs_syntype is None:
        msg = '`extract_contact_sites` requires the cythonized method ' \
              '`extract_cs_syntype`. Use `find_contact_sites` and others ' \
              'for contact site processing.'
        log_extraction.error(msg)
        raise ImportError(msg)
    kd = basics.kd_factory(global_params.config.kd_seg_path)
    if cube_of_interest_bb is None:
        cube_of_interest_bb = [np.zeros(3, dtype=np.int), kd.boundary]
    if chunk_size is None:
        chunk_size = (512, 512, 512)
    if cube_shape is None:
        cube_shape = (256, 256, 256)
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
    dir_props = f"{global_params.config.temp_path}/tmp_props_cssyn/"

    # remove previous temporary results.
    if os.path.isdir(dir_props):
        if not overwrite:
            msg = f'Could not start extraction of supervoxel objects ' \
                  f'because temporary files already existed at "{dir_props}" ' \
                  f'and overwrite was set to False.'
            log.error(msg)
            raise FileExistsError(msg)
        log.debug(f'Found existing cache folder at {dir_props}. Removing it now.')
        shutil.rmtree(dir_props)
    if os.path.isdir(cset.path_head_folder):
        if not overwrite:
            msg = f'Could not start extraction of supervoxel objects ' \
                  f'because temporary files already existed at "{cset.path_head_folder}" ' \
                  f'and overwrite was set to False.'
            log.error(msg)
            raise FileExistsError(msg)
        log.debug(f'Found existing cache folder at {cset.path_head_folder}. Removing it now.')
        shutil.rmtree(cset.path_head_folder)
    os.makedirs(dir_props)
    os.makedirs(cset.path_head_folder)

    multi_params = []
    # TODO: currently pickles Chunk objects -> job submission might be slow
    if max_n_jobs > len(chunk_list) // 50:
        iter_params = basics.chunkify(chunk_list, max_n_jobs)
    else:
        iter_params = basics.chunkify_successive(chunk_list, 50)
    for ii, chunk_k in enumerate(iter_params):
        multi_params.append([[cset.chunk_dict[k] for k in chunk_k],
                             global_params.config.kd_seg_path, ii, dir_props])

    # reduce step
    start = time.time()
    cs_worker_dc_fname = f'{global_params.config.temp_path}/cs_worker_dict.pkl'
    dict_paths_tmp = [cs_worker_dc_fname]
    syn_ids = []
    cs_ids = []
    cs_worker_mapping = dict()  # cs include syns
    if qu.batchjob_enabled():
        path_to_out = qu.batchjob_script(multi_params, "contact_site_extraction",
                                         n_max_co_processes=n_max_co_processes, log=log)
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
        results = start_multiprocess_imap(
            _contact_site_extraction_thread, multi_params, nb_cpus=n_max_co_processes,
            verbose=False, debug=False)

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
    del cs_worker_mapping

    syn_ids = np.unique(np.concatenate(syn_ids)).astype(np.uint64)
    n_syn = len(syn_ids)
    del syn_ids

    cs_ids = np.unique(np.concatenate(cs_ids)).astype(np.uint64)
    n_cs = len(cs_ids)

    # only required as syn is a subset of cs!
    dest_p = f'{global_params.config.temp_path}/storage_targets_cs.pkl'
    dict_paths_tmp.append(dest_p)

    _ = _cache_storage_paths((dest_p, cs_ids, n_folders_fs))
    del cs_ids

    step_names.append("extract objects and collect properties of cs and syn.")
    all_times.append(time.time() - start)

    # reduce step
    start = time.time()

    all_times.append(time.time() - start)
    step_names.append("conversion of results")

    log.info('Finished extraction of contact sites (#objects: {}) and synapses'
             ' (#objects: {}).'.format(n_cs, n_syn))
    if n_syn == 0:
        log.critical('WARNING: Did not find any synapses during extraction step.')

    # write cs and syn segmentation to KD and SD
    chunky.save_dataset(cset)
    kd = basics.kd_factory(global_params.config.kd_seg_path)
    # convert Chunkdataset to syn and cs KD
    # TODO: spawn in parallel
    for obj_type in ['cs', 'syn']:
        path = "{}/knossosdatasets/{}_seg/".format(
            global_params.config.working_dir, obj_type)
        if os.path.isdir(path):
            log.debug('Found existing KD at {}. Removing it now.'.format(path))
            shutil.rmtree(path)
        target_kd = knossosdataset.KnossosDataset()
        target_kd._cube_shape = cube_shape
        scale = np.array(global_params.config['scaling'])
        target_kd.scales = [scale, ]
        target_kd.initialize_without_conf(path, kd.boundary, scale, kd.experiment_name,
                                          mags=[1, ])
        target_kd = basics.kd_factory(path)
        export_cset_to_kd_batchjob({obj_type: path},
            cset, obj_type, [obj_type],
            offset=offset, size=size, stride=chunk_size, as_raw=False,
            orig_dtype=np.uint64, unified_labels=False,
            n_max_co_processes=n_max_co_processes, log=log)
        log.debug('Finished conversion of ChunkDataset ({}) into KnossosDataset'
                  ' ({})'.format(cset.path_head_folder, target_kd.knossos_path))

    # create folders for existing (sub-)cell supervoxels to prevent concurrent makedirs
    for ii, struct in enumerate(['cs', 'syn']):
        sc_sd = segmentation.SegmentationDataset(
            working_dir=global_params.config.working_dir, obj_type=struct,
            version=0, n_folders_fs=n_folders_fs)
        ids = rep_helper.get_unique_subfold_ixs(n_folders_fs)
        for ix in tqdm.tqdm(ids, leave=False):
            curr_dir = sc_sd.so_storage_path + target_dir_func(
                ix, n_folders_fs)
            os.makedirs(curr_dir, exist_ok=True)

    # Write SD
    path = "{}/knossosdatasets/syn_seg/".format(global_params.config.working_dir)
    path_cs = "{}/knossosdatasets/cs_seg/".format(global_params.config.working_dir)
    storage_location_ids = rep_helper.get_unique_subfold_ixs(n_folders_fs)
    max_n_jobs = min(max_n_jobs, len(storage_location_ids))
    multi_params = [(sv_id_block, n_folders_fs, path, path_cs, dir_props)
                    for sv_id_block in basics.chunkify(storage_location_ids, max_n_jobs)]
    if not qu.batchjob_enabled():
        start_multiprocess_imap(_write_props_to_syn_thread,
                                multi_params, nb_cpus=1, debug=False)
    else:
        qu.batchjob_script(multi_params, "write_props_to_syn", log=log,
                           n_cores=1, n_max_co_processes=max_n_jobs,
                           remove_jobfolder=True)

    sd_syn = segmentation.SegmentationDataset(working_dir=global_params.config.working_dir,
                                              obj_type='syn', version=0)
    dataset_analysis(sd_syn, recompute=True, compute_meshprops=False)
    sd_cs = segmentation.SegmentationDataset(working_dir=global_params.config.working_dir,
                                             obj_type='cs', version=0)
    dataset_analysis(sd_cs, recompute=True, compute_meshprops=False)
    # if len(sd_syn.ids) != n_syn or len(sd_cs.ids) != n_cs:
        # raise ValueError(f'Number mismatch between detected contact sites or synapses (SV level).')
    # for p in dict_paths_tmp:
    #     os.remove(p)
    shutil.rmtree(cd_dir, ignore_errors=True)
    # if qu.batchjob_enabled():
    #     shutil.rmtree(path_to_out, ignore_errors=True)


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
        * using the prob. maps of the initial synaptic junction prediction
          the object extraction of ``sj`` objects can be removed.

    Returns:
        Two lists of dictionaries (representative coordinates, bounding box and
        voxel count) for ``cs`` and ``syn`` objects, per-synapse counts of
        symmetric and asymmetric voxels.
    """
    chunks = args[0]
    knossos_path = args[1]
    worker_nr = args[2]
    dir_props = args[3]
    worker_dir_props = f"{dir_props}/{worker_nr}/"
    os.makedirs(worker_dir_props, exist_ok=True)

    if global_params.config.syntype_available and \
       (global_params.config.sym_label == global_params.config.asym_label) and \
       (global_params.config.kd_sym_path == global_params.config.kd_asym_path):
        raise ValueError('Both KnossosDatasets and labels for symmetric and '
                         'asymmetric synapses are identical. Either one '
                         'must differ.')

    kd = basics.kd_factory(knossos_path)
    # TODO: use prob maps in kd.kd_sj_path (proba maps -> get rid of SJ extraction),
    #  see below.
    kd_syn = basics.kd_factory(global_params.config.kd_organelle_seg_paths['sj'])
    if global_params.config.syntype_available:
        kd_syntype_sym = basics.kd_factory(global_params.config.kd_sym_path)
        kd_syntype_asym = basics.kd_factory(global_params.config.kd_asym_path)
    else:
        kd_syntype_sym, kd_syntype_asym = None, None
    cs_props = [{}, defaultdict(list), {}]
    syn_props = [{}, defaultdict(list), {}]
    tot_sym_cnt = {}
    tot_asym_cnt = {}
    cum_dt_data = 0
    cum_dt_proc = 0
    cum_dt_proc2 = 0
    cs_filtersize = np.array(global_params.config['cell_objects']['cs_filtersize'])
    for chunk in chunks:
        # additional overlap, e.g. to prevent boundary artifacts by dilation/closing
        # TODO: if this is set to 0, `detect_cs` still returns a black boundary region
        #  although it already incorporates an additional offset (-> stencil_offset)
        overlap = max(cs_filtersize) // 2
        offset = np.array(chunk.coordinates - overlap)  # also used for loading synapse data
        size = 2 * overlap + np.array(chunk.size)  # also used for loading synapse data
        start = time.time()

        stencil_offset = (cs_filtersize - 1) // 2

        data = kd.load_seg(size=size + 2 * stencil_offset,
                           offset=offset - stencil_offset,
                           mag=1, datatype=np.uint64).astype(np.uint32, copy=False).swapaxes(0, 2)
        cum_dt_data += time.time() - start
        start = time.time()
        # contacts has size as given with `size`, because it performs valid conv.
        # -> contacts result is cropped by stencil_offset on each side
        contacts = np.asarray(detect_cs(data))
        cum_dt_proc += time.time() - start

        start = time.time()
        # TODO: use prob maps in kd.kd_sj_path (proba maps -> get rid of SJ extraction)
        # syn_d = (kd_syn.from_raw_cubes_to_matrix(size, offset) > 255 * global_params.config[
        # 'cell_objects']["probathresholds"]['sj']).astype(np.uint8)
        syn_d = (kd_syn.load_seg(size=size, offset=offset, mag=1,
                                 datatype=np.uint64) > 0).astype(np.uint8, copy=False).swapaxes(0, 2)
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
                assert global_params.config.asym_label is not None,\
                    'Label of asymmetric synapses is not set.'
                assert global_params.config.sym_label is not None,\
                    'Label of symmetric synapses is not set.'
                # load synapse type classification results stored in the same KD
                sym_d = kd_syntype_sym.load_seg(size=size, offset=offset, mag=1).swapaxes(0, 2)
                # create copy
                asym_d = np.array(sym_d == global_params.config.asym_label, dtype=np.uint8)
                sym_d = np.array(sym_d == global_params.config.sym_label, dtype=np.uint8)
        else:
            sym_d = np.zeros_like(syn_d)
            asym_d = np.zeros_like(syn_d)
        cum_dt_data += time.time() - start

        # close gaps of contact sites prior to overlapping synaptic junction map with contact sites
        start = time.time()
        # returns rep. coords, bounding box and size for every ID in contacts
        # used to get location of every contact site to perform closing operation
        _, bb_dc, _ = rep_helper.find_object_properties(contacts)
        n_closings = min(cs_filtersize)
        for ix in bb_dc.keys():
            obj_start, obj_end = np.array(bb_dc[ix])
            obj_start -= n_closings
            obj_start[obj_start < 0] = 0
            obj_end += n_closings
            # create slice obj
            new_obj_slices = tuple(slice(obj_start[ii], obj_end[ii], None) for
                                   ii in range(3))
            sub_vol = contacts[new_obj_slices]
            binary_mask = (sub_vol == ix).astype(np.int8, copy=False)
            res = scipy.ndimage.binary_closing(
                binary_mask, iterations=n_closings)
            # TODO: add to parameters to config
            res = scipy.ndimage.binary_dilation(
                res, iterations=2)
            # only update background or the objects itself
            proc_mask = (binary_mask == 1) | (sub_vol == 0)
            contacts[new_obj_slices][proc_mask] = res[proc_mask] * ix
        cum_dt_proc2 += time.time() - start

        start = time.time()
        # this counts SJ foreground voxels overlapping with the CS objects
        # and the asym and sym voxels, do not use overlap here!
        curr_cs_p, curr_syn_p, asym_cnt, sym_cnt = extract_cs_syntype(
            contacts[overlap:-overlap, overlap:-overlap, overlap:-overlap],
            syn_d[overlap:-overlap, overlap:-overlap, overlap:-overlap],
            asym_d[overlap:-overlap, overlap:-overlap, overlap:-overlap],
            sym_d[overlap:-overlap, overlap:-overlap, overlap:-overlap])
        cum_dt_proc += time.time() - start
        os.makedirs(chunk.folder, exist_ok=True)
        compression.save_to_h5py([contacts[overlap:-overlap, overlap:-overlap,
                                  overlap:-overlap]], chunk.folder + "cs.h5",
                                 ['cs'], overwrite=True)
        # syn segmentation only contain the overlap voxels between SJ and CS
        contacts[syn_d == 0] = 0
        compression.save_to_h5py([contacts[overlap:-overlap, overlap:-overlap,
                                  overlap:-overlap]], chunk.folder + "syn.h5",
                                 ['syn'], overwrite=True)
        # overlap was removed for the analysis of the object properties
        merge_prop_dicts([cs_props, curr_cs_p], offset=offset + overlap)
        merge_prop_dicts([syn_props, curr_syn_p], offset=offset + overlap)
        merge_type_dicts([tot_asym_cnt, asym_cnt])
        merge_type_dicts([tot_sym_cnt, sym_cnt])
        del curr_cs_p, curr_syn_p, asym_cnt, sym_cnt
    basics.write_obj2pkl(f'{worker_dir_props}/cs_props_{worker_nr}.pkl', cs_props)
    basics.write_obj2pkl(f'{worker_dir_props}/syn_props_{worker_nr}.pkl', syn_props)
    basics.write_obj2pkl(f'{worker_dir_props}/tot_asym_cnt_{worker_nr}.pkl', tot_asym_cnt)
    basics.write_obj2pkl(f'{worker_dir_props}/tot_sym_cnt_{worker_nr}.pkl', tot_sym_cnt)
    # log_extraction.error("Cum. time for loading data: {:.2f} s; for processing: {:.2f} "
    #                      "s for processing2: {:.2f} s. {} cs and {} syn.".format(
    #     cum_dt_data, cum_dt_proc, cum_dt_proc2, len(cs_props[0]), len(syn_props[0])))
    return worker_nr, dict(cs=list(cs_props[0].keys()), syn=list(syn_props[0].keys()))


# iterate over the subcellular SV ID chunks
def _write_props_to_syn_thread(args):
    cs_ids_ch = args[0]
    n_folders_fs = args[1]
    knossos_path = args[2]
    knossos_path_cs = args[3]
    dir_props = args[4]
    min_obj_vx_dc = global_params.config['cell_objects']['min_obj_vx']
    tmp_path = global_params.config.temp_path
    if global_params.config.use_new_subfold:
        target_dir_func = rep_helper.subfold_from_ix_new
    else:
        target_dir_func = rep_helper.subfold_from_ix_OLD

    # get cached worker lookup
    with open(f'{global_params.config.temp_path}/cs_worker_dict.pkl', "rb") as f:
        cs_workers_tmp = pkl.load(f)

    for obj_id_mod in cs_ids_ch:
        # get destination paths for the current objects
        dest_dc_tmp = CompressedStorage(f'{tmp_path}/storage_targets_cs.pkl', disable_locking=True)
        k = target_dir_func(obj_id_mod, n_folders_fs)
        if k not in dest_dc_tmp or len(dest_dc_tmp[k]) == 0:
            continue
        obj_keys = dest_dc_tmp[k]
        del dest_dc_tmp

        sd = segmentation.SegmentationDataset(n_folders_fs=n_folders_fs, obj_type='syn',
                                              working_dir=global_params.config.working_dir,
                                              version=0)

        sd_cs = segmentation.SegmentationDataset(n_folders_fs=n_folders_fs, obj_type='cs',
                                                 working_dir=global_params.config.working_dir,
                                                 version=0)
        cs_props = [{}, defaultdict(list), {}]
        syn_props = [{}, defaultdict(list), {}]
        cs_sym_cnt = {}
        cs_asym_cnt = {}
        for worker_id, obj_ids in cs_workers_tmp.items():
            intersec = set(obj_ids).intersection(obj_keys)
            load_worker = len(intersec) > 0
            if load_worker:
                worker_dir_props = f"{dir_props}/{worker_id}/"
                # cs
                fname = f'{worker_dir_props}/cs_props_{worker_id}.pkl'
                start = time.time()
                dc = basics.load_pkl2obj(fname)
                tmp_dcs = [dict(), defaultdict(list), dict()]
                for k in intersec:
                    tmp_dcs[0][k] = dc[0][k]
                    tmp_dcs[1][k] = dc[1][k]
                    tmp_dcs[2][k] = dc[2][k]
                del dc
                merge_prop_dicts([cs_props, tmp_dcs])
                del tmp_dcs

                # syn
                fname = f'{worker_dir_props}/syn_props_{worker_id}.pkl'
                dc = basics.load_pkl2obj(fname)
                fname = f'{worker_dir_props}/tot_sym_cnt_{worker_id}.pkl'
                curr_sym_cnt = basics.load_pkl2obj(fname)
                fname = f'{worker_dir_props}/tot_asym_cnt_{worker_id}.pkl'
                curr_asym_cnt = basics.load_pkl2obj(fname)
                tmp_dcs = [dict(), defaultdict(list), dict()]
                tmp_sym_dc = dict()
                tmp_asym_dc = dict()
                for k in intersec:
                    if k not in dc[0]:
                        continue
                    tmp_dcs[0][k] = dc[0][k]
                    tmp_dcs[1][k] = dc[1][k]
                    tmp_dcs[2][k] = dc[2][k]
                    if k in curr_sym_cnt:
                        tmp_sym_dc[k] = curr_sym_cnt[k]
                    if k in curr_asym_cnt:
                        tmp_asym_dc[k] = curr_asym_cnt[k]
                del dc, curr_sym_cnt, curr_asym_cnt
                merge_prop_dicts([syn_props, tmp_dcs])
                del tmp_dcs
                merge_type_dicts([cs_asym_cnt, tmp_asym_dc])
                del tmp_asym_dc
                merge_type_dicts([cs_sym_cnt, tmp_sym_dc])
                del tmp_sym_dc
        # get dummy segmentation object to fetch attribute dictionary for this batch of object IDs
        dummy_so = sd.get_segmentation_object(obj_id_mod)
        attr_p = dummy_so.attr_dict_path
        vx_p = dummy_so.voxel_path
        this_attr_dc = AttributeDict(attr_p, read_only=False, disable_locking=True)
        # this class is only used to query the voxel data
        voxel_dc = VoxelStorageDyn(vx_p, voxel_mode=False, voxeldata_path=knossos_path,
                                   read_only=False, disable_locking=True)
        voxel_dc_store = VoxelStorage(vx_p, read_only=False, disable_locking=True)

        # get dummy CS segmentation object to fetch attribute dictionary for this batch of object
        # IDs
        dummy_so_cs = sd_cs.get_segmentation_object(obj_id_mod)
        attr_p_cs = dummy_so_cs.attr_dict_path
        vx_p_cs = dummy_so_cs.voxel_path
        this_attr_dc_cs = AttributeDict(attr_p_cs, read_only=False, disable_locking=True)
        voxel_dc_cs = VoxelStorageDyn(vx_p_cs, voxel_mode=False, voxeldata_path=knossos_path_cs,
                                      read_only=False, disable_locking=True)

        for cs_id in obj_keys:
            # write cs to dict
            if cs_props[2][cs_id] < min_obj_vx_dc['cs']:
                continue
            rp_cs = cs_props[0][cs_id]
            bbs_cs = np.concatenate(cs_props[1][cs_id])
            size_cs = cs_props[2][cs_id]
            this_attr_dc_cs[cs_id]["rep_coord"] = rp_cs
            this_attr_dc_cs[cs_id]["bounding_box"] = np.array(
                [bbs_cs[:, 0].min(axis=0), bbs_cs[:, 1].max(axis=0)])
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
            bb = np.array(
                [bbs[:, 0].min(axis=0), bbs[:, 1].max(axis=0)])
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

            # TODO: should be refactored at some point, changed to bounding box of
            #  the overlap object instead of the SJ bounding box. Also the background ratio was adapted
            n_vxs_in_sjbb = np.prod(bb[1] - bb[0]) # number of CS voxels in syn BB
            id_ratio = size_cs / n_vxs_in_sjbb  # this is the fraction of CS voxels within the syn BB
            cs_ratio = size / size_cs  # number of overlap voxels (syn voxels) divided by cs size
            background_overlap_ratio = 1 - id_ratio  # TODO: not the same as before anymore: local
            # inverse 'CS' density: c_cs_ids[u_cs_ids == 0] / n_vxs_in_sjbb  (previous version)
            add_feat_dict = {'sj_id': cs_id, 'cs_id': cs_id,
                             'id_sj_ratio': id_ratio,
                             'sj_size_pseudo': n_vxs_in_sjbb,
                             'id_cs_ratio': cs_ratio,
                             'cs_size': size_cs,
                             'background_overlap_ratio': background_overlap_ratio}
            this_attr_dc[cs_id].update(add_feat_dict)
            voxel_dc[cs_id] = bbs
            voxel_dc.increase_object_size(cs_id, size)
            voxel_dc.set_object_repcoord(cs_id, rp)
            # write voxels explicitly - this assumes reasonably sized synapses
            voxel_dc_store[cs_id] = voxel_dc.get_voxeldata(cs_id)
        voxel_dc_store.push()
        voxel_dc_cs.push()
        this_attr_dc.push()
        this_attr_dc_cs.push()


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
    size :
    offset :

    Returns
    -------

    """
    log_extraction.warning(DeprecationWarning('"find_contact_sites" was replaced by '
                                              '"extract_contact_sites".'))
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
        _ = qu.batchjob_script(
            multi_params, "contact_site_detection",
            n_max_co_processes=n_max_co_processes)
    chunky.save_dataset(cset)


def _contact_site_detection_thread(args):
    chunk = args[0]
    knossos_path = args[1]
    filename = args[2]

    kd = basics.kd_factory(knossos_path)

    overlap = np.array([6, 6, 3], dtype=np.int)
    offset = np.array(chunk.coordinates - overlap)
    size = 2 * overlap + np.array(chunk.size)
    data = kd.load_seg(size=size, offset=offset, mag=1,
                       datatype=np.uint64).astype(np.uint32, copy=False).swapaxes(0, 2)
    contacts = detect_cs(data)
    os.makedirs(chunk.folder, exist_ok=True)
    compression.save_to_h5py([contacts],
                             chunk.folder + filename +
                             ".h5", ["cs"])


def detect_cs(arr: np.ndarray) -> np.ndarray:
    """

    Args:
        arr: 3D segmentation array (only np.uint32!)

    Returns:
        3D contact site segmentation array (np.uint64).
    """
    jac = np.zeros([3, 3, 3], dtype=np.int)
    jac[1, 1, 1] = -6
    jac[1, 1, 0] = 1
    jac[1, 1, 2] = 1
    jac[1, 0, 1] = 1
    jac[1, 2, 1] = 1
    jac[2, 1, 1] = 1
    jac[0, 1, 1] = 1
    edges = scipy.ndimage.convolve(arr.astype(np.int), jac) < 0
    edges = edges.astype(np.uint32, copy=False)
    arr = arr.astype(np.uint32, copy=False)
    cs_seg = process_block_nonzero(
        edges, arr, global_params.config['cell_objects']['cs_filtersize'])
    return cs_seg


def extract_agg_contact_sites(cset, working_dir, filename='cs', hdf5name='cs',
                              n_folders_fs=10000, suffix="",
                              n_max_co_processes=None, n_chunk_jobs=2000, size=None,
                              offset=None, log=None, cube_shape=None):
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
    cube_shape :

    Returns
    -------

    """
    if log is None:
        log = log_extraction
    if cube_shape is None:
        cube_shape = (256, 256, 256)
    chunky.save_dataset(cset)
    # init CS segmentation KD
    kd = basics.kd_factory(global_params.config.kd_seg_path)
    path = "{}/knossosdatasets/{}_seg/".format(global_params.config.working_dir, filename)
    if os.path.isdir(path):
        log.debug('Found existing KD at {}. Removing it now.'.format(path))
        shutil.rmtree(path)
    target_kd = knossosdataset.KnossosDataset()
    target_kd._cube_shape = cube_shape
    scale = np.array(global_params.config['scaling'])
    target_kd.scales = [scale, ]
    target_kd.initialize_without_conf(path, kd.boundary, scale, kd.experiment_name, mags=[1, ],
                                      create_pyk_conf=True, create_knossos_conf=False)

    # convert Chunkdataset to KD
    export_cset_to_kd_batchjob({hdf5name: path},
        cset, '{}'.format(filename), [hdf5name],
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
