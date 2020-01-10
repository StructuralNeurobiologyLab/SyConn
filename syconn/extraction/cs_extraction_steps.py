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
from logging import Logger
import glob
import numpy as np
import scipy.ndimage
from knossos_utils import knossosdataset
from knossos_utils import chunky
knossosdataset._set_noprint(True)
import os
from ..backend.storage import AttributeDict, VoxelStorageDyn, VoxelStorage
from ..reps import rep_helper
from ..mp import batchjob_utils as qu
from ..handler import compression, basics
from ..reps import segmentation
from ..handler.basics import kd_factory, chunkify
from . object_extraction_steps import export_cset_to_kd_batchjob
from . import log_extraction
from .object_extraction_wrapper import from_ids_to_objects, calculate_chunk_numbers_for_box
from ..mp.mp_utils import start_multiprocess_imap
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
                          cube_shape: Optional[Tuple[int]] = None):
    """
    Extracts contact sites and their overlap with `sj` objects and stores them in a
    :class:`~syconn.reps.segmentation.SegmentationDataset` of type ``cs`` and ``syn``
    respectively. If synapse type is available, this information will be stored
    as the voxel-ratio per class in the attribute dictionary of the ``syn``
    objects (keys: ``sym_prop``, ``asym_prop``).

    Notes:
        Replaced ``find_contact_sites``, ``extract_agg_contact_sites``, `
        `syn_gen_via_cset`` and ``extract_synapse_type``.

    Args:
        n_max_co_processes: Number of parallel workers.
        chunk_size: Sub-cube volume which is processed at a time.
        log: Logger.
        max_n_jobs: Maximum number of jobs.
        cube_of_interest_bb: Sub-volume of the data set which is processed.
            Default: Entire data set.
        n_folders_fs: Number of folders used for organizing supervoxel data.
        cube_shape: Cube shape used within contact site KnossosDataset.

    """
    if extract_cs_syntype is None:
        msg = '`extract_contact_sites` requires the cythonized method ' \
              '`extract_cs_syntype`. Use `find_contact_sites` and others ' \
              'for contact site processing.'
        log_extraction.error(msg)
        raise ImportError(msg)
    kd = kd_factory(global_params.config.kd_seg_path)
    if cube_of_interest_bb is None:
        cube_of_interest_bb = [np.zeros(3, dtype=np.int), kd.boundary]
    if chunk_size is None:
        chunk_size = (512, 512, 512)
    if cube_shape is None:
        cube_shape = (256, 256, 256)
    size = cube_of_interest_bb[1] - cube_of_interest_bb[0] + 1
    offset = cube_of_interest_bb[0]

    # Initital contact site extraction
    cd_dir = global_params.config.temp_path + "/chunkdatasets/cs/"
    # Class that contains a dict of chunks (with coordinates) after initializing it
    cset = chunky.ChunkDataset()
    cset.initialize(kd, kd.boundary, chunk_size, cd_dir,
                    box_coords=[0, 0, 0], fit_box_size=True)

    if max_n_jobs is None:
        max_n_jobs = global_params.config.ncore_total * 2
    if log is None:
        log = log_extraction
    if size is not None and offset is not None:
        chunk_list, _ = \
            calculate_chunk_numbers_for_box(cset, offset, size)
    else:
        chunk_list = [ii for ii in range(len(cset.chunk_dict))]
    # shuffle chunklist to get a more balanced work-load
    rand_ixs = np.arange(len(chunk_list))
    np.random.shuffle(rand_ixs)
    chunk_list = np.array(chunk_list)[rand_ixs]

    os.makedirs(cset.path_head_folder, exist_ok=True)
    multi_params = []
    # TODO: currently pickles Chunk objects -> job submission might be slow
    for chunk_k in chunkify(chunk_list, max_n_jobs):
        multi_params.append([[cset.chunk_dict[k] for k in chunk_k],
                             global_params.config.kd_seg_path])

    if not qu.batchjob_enabled():
        results = start_multiprocess_imap(
            _contact_site_extraction_thread, multi_params, debug=False,
            nb_cpus=n_max_co_processes)
    else:
        path_to_out = qu.QSUB_script(multi_params, "contact_site_extraction",
                                     n_max_co_processes=n_max_co_processes,
                                     log=log)
        out_files = glob.glob(path_to_out + "/*")
        results = []
        for out_file in out_files:
            with open(out_file, 'rb') as f:
                results.append(pkl.load(f))
        shutil.rmtree(os.path.abspath(path_to_out + "/../"), ignore_errors=True)
    # reduce step
    cs_props = [{}, defaultdict(list), {}]
    syn_props = [{}, defaultdict(list), {}]
    tot_sym_cnt = {}
    tot_asym_cnt = {}
    for curr_props, curr_syn_props, asym_cnt, sym_cnt in results:
        merge_prop_dicts([cs_props, curr_props])
        merge_prop_dicts([syn_props, curr_syn_props])
        merge_type_dicts([tot_asym_cnt, asym_cnt])
        merge_type_dicts([tot_sym_cnt, sym_cnt])
    log.info('Finished extraction of contact sites (#objects: {}) and synapses'
             ' (#objects: {}).'.format(len(cs_props[0]), len(syn_props[0])))
    if len(syn_props[0]) == 0:
        log.critical('WARNING: Did not find any synapses during extraction step.')
    # TODO: extract syn objects! maybe replace sj_0 Segmentation dataset by the overlapping CS<->
    #  sj objects -> run syn. extraction and sd_generation in parallel and return mi_0, vc_0 and
    #  syn_0 -> use syns as new sjs during rendering!
    #  -> Run CS generation in parallel with mapping to at least get the syn objects before
    #  rendering the neuron views (which need subcellular structures, there one can then use mi,
    #  vc and syn (instead of sj))
    dict_paths = []
    # dump intermediate results
    # TODO: size filter here or during write-out? TODO: use config parameter
    dict_p = "{}/cs_prop_dict.pkl".format(global_params.config.temp_path)
    with open(dict_p, "wb") as f:
        pkl.dump(cs_props, f, protocol=4)
    del cs_props
    dict_paths.append(dict_p)

    dict_p = "{}/syn_prop_dict.pkl".format(global_params.config.temp_path)
    with open(dict_p, "wb") as f:
        pkl.dump(syn_props, f, protocol=4)
    del syn_props
    dict_paths.append(dict_p)

    # convert counting dicts to store ratio of syn. type voxels
    dict_p = "{}/cs_sym_cnt.pkl".format(global_params.config.temp_path)
    with open(dict_p, "wb") as f:
        pkl.dump(tot_sym_cnt, f, protocol=4)
    del tot_sym_cnt
    dict_paths.append(dict_p)

    dict_p = "{}/cs_asym_cnt.pkl".format(global_params.config.temp_path)
    with open(dict_p, "wb") as f:
        pkl.dump(tot_asym_cnt, f, protocol=4)
    del tot_asym_cnt
    dict_paths.append(dict_p)

    # write cs and syn segmentation to KD and SD
    chunky.save_dataset(cset)
    kd = kd_factory(global_params.config.kd_seg_path)
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
        target_kd = kd_factory(path)
        export_cset_to_kd_batchjob({obj_type: path},
            cset, obj_type, [obj_type],
            offset=offset, size=size, stride=chunk_size, as_raw=False,
            orig_dtype=np.uint64, unified_labels=False,
            n_max_co_processes=n_max_co_processes, log=log)
        log.debug('Finished conversion of ChunkDataset ({}) into KnossosDataset'
                  ' ({})'.format(cset.path_head_folder, target_kd.knossos_path))

    # Write SD
    max_n_jobs = global_params.config['nnodes_total'] * 2
    path = "{}/knossosdatasets/syn_seg/".format(global_params.config.working_dir)
    path_cs = "{}/knossosdatasets/cs_seg/".format(global_params.config.working_dir)
    storage_location_ids = rep_helper.get_unique_subfold_ixs(n_folders_fs)
    multi_params = [(sv_id_block, n_folders_fs, path, path_cs) for sv_id_block in
                    basics.chunkify(storage_location_ids, max_n_jobs)]

    if not qu.batchjob_enabled():
        start_multiprocess_imap(_write_props_to_syn_singlenode_thread,
                                multi_params, nb_cpus=1, debug=False)
    else:
        qu.QSUB_script(multi_params, "write_props_to_syn_singlenode", log=log,
                       n_cores=global_params.config['ncores_per_node'],
                       n_max_co_processes=global_params.config['nnodes_total'],
                       remove_jobfolder=True)

    sd = segmentation.SegmentationDataset(working_dir=global_params.config.working_dir,
                                          obj_type='syn', version=0)
    dataset_analysis(sd, recompute=True, compute_meshprops=False)
    sd = segmentation.SegmentationDataset(working_dir=global_params.config.working_dir,
                                          obj_type='cs', version=0)
    dataset_analysis(sd, recompute=True, compute_meshprops=False)

    for p in dict_paths:
        os.remove(p)
    shutil.rmtree(cd_dir, ignore_errors=True)


def _contact_site_extraction_thread(args: Union[tuple, list]) \
        -> Tuple[List[dict], List[dict], dict, dict]:
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

    if global_params.config.syntype_available and \
       (global_params.config.sym_label == global_params.config.asym_label) and \
       (global_params.config.kd_sym_path == global_params.config.kd_asym_path):
        raise ValueError('Both KnossosDatasets and labels for symmetric and '
                         'asymmetric synapses are identical. Either one '
                         'must differ.')

    kd = kd_factory(knossos_path)
    # TODO: use prob maps in kd.kd_sj_path (proba maps -> get rid of SJ extraction),
    #  see below.
    kd_syn = kd_factory(global_params.config.kd_organelle_seg_paths['sj'])
    if global_params.config.syntype_available:
        kd_syntype_sym = kd_factory(global_params.config.kd_sym_path)
        kd_syntype_asym = kd_factory(global_params.config.kd_asym_path)
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
                           mag=1, datatype=np.uint64).astype(np.uint32).swapaxes(0, 2)
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
                                 datatype=np.uint64) > 0).astype(np.uint8).swapaxes(0, 2)
        # get binary mask for symmetric and asymmetric syn. type per voxel
        if global_params.config.syntype_available:
            if global_params.config.kd_asym_path != global_params.config.kd_sym_path:
                # TODO: add thresholds to global_params
                if global_params.config.sym_label is None:
                    sym_d = (kd_syntype_sym.load_raw(size=size, offset=offset, mag=1).swapaxes(0, 2)
                             >= 123).astype(np.uint8)
                else:
                    sym_d = (kd_syntype_sym.load_seg(size=size, offset=offset, mag=1).swapaxes(0, 2)
                             == global_params.config.sym_label).astype(np.uint8)

                if global_params.config.asym_label is None:
                    asym_d = (kd_syntype_asym.load_raw(size=size, offset=offset, mag=1).swapaxes(0, 2)
                              >= 123).astype(np.uint8)
                else:
                    asym_d = (kd_syntype_asym.load_seg(size=size, offset=offset, mag=1).swapaxes(0, 2)
                              == global_params.config.asym_label).astype(np.uint8)
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
            binary_mask = (sub_vol == ix).astype(np.int)
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
    # log_extraction.error("Cum. time for loading data: {:.2f} s; for processing: {:.2f} "
    #                      "s for processing2: {:.2f} s. {} cs and {} syn.".format(
    #     cum_dt_data, cum_dt_proc, cum_dt_proc2, len(cs_props[0]), len(syn_props[0])))
    return cs_props, syn_props, tot_asym_cnt, tot_sym_cnt


def _write_props_to_syn_thread(args):
    """"""
    # TODO: refactor in the same way as object mapping (reduce dictionaries)
    # TODO: refactor such that voxel data is stored during extraction
    cs_ids_ch = args[0]
    n_folders_fs = args[1]
    knossos_path = args[2]
    knossos_path_cs = args[3]

    # get cached dicts
    dict_p = "{}/cs_prop_dict.pkl".format(global_params.config.temp_path)
    with open(dict_p, "rb") as f:
        cs_props = pkl.load(f)

    dict_p = "{}/syn_prop_dict.pkl".format(global_params.config.temp_path)
    with open(dict_p, "rb") as f:
        syn_props = pkl.load(f)

    dict_p = "{}/cs_sym_cnt.pkl".format(global_params.config.temp_path)
    with open(dict_p, "rb") as f:
        cs_sym_cnt = pkl.load(f)

    dict_p = "{}/cs_asym_cnt.pkl".format(global_params.config.temp_path)
    with open(dict_p, "rb") as f:
        cs_asym_cnt = pkl.load(f)

    # store destinations for each existing obj
    dest_dc = defaultdict(list)
    for cs_id in cs_props[0]:
        dest_dc[rep_helper.subfold_from_ix(cs_id, n_folders_fs)].append(cs_id)

    # get SegmentationDataset of current subcell.
    sd = segmentation.SegmentationDataset(
        n_folders_fs=n_folders_fs, obj_type='syn', version=0,
        working_dir=global_params.config.working_dir)

    sd_cs = segmentation.SegmentationDataset(
        n_folders_fs=n_folders_fs, obj_type='cs', version=0,
        working_dir=global_params.config.working_dir)
    # iterate over the subcellular SV ID chunks
    for obj_id_mod in cs_ids_ch:
        obj_keys = dest_dc[rep_helper.subfold_from_ix(obj_id_mod, n_folders_fs)]
        if len(obj_keys) == 0:
            continue
        # get dummy segmentation object to fetch attribute dictionary for
        # this batch of object IDs
        dummy_so = sd.get_segmentation_object(obj_id_mod)
        attr_p = dummy_so.attr_dict_path
        vx_p = dummy_so.voxel_path
        this_attr_dc = AttributeDict(attr_p, read_only=False,
                                     disable_locking=True)
        # this class is only used to query the voxel data
        voxel_dc = VoxelStorageDyn(
            vx_p, voxel_mode=False, voxeldata_path=knossos_path,
            read_only=False, disable_locking=True)
        voxel_dc_store = VoxelStorage(vx_p, read_only=False, disable_locking=True)

        # get dummy CS segmentation object to fetch attribute dictionary for
        # this batch of object IDs
        dummy_so_cs = sd_cs.get_segmentation_object(obj_id_mod)
        attr_p_cs = dummy_so_cs.attr_dict_path
        vx_p_cs = dummy_so_cs.voxel_path
        this_attr_dc_cs = AttributeDict(attr_p_cs, read_only=False,
                                        disable_locking=True)
        voxel_dc_cs = VoxelStorageDyn(
            vx_p_cs, voxel_mode=False, voxeldata_path=knossos_path_cs,
            read_only=False, disable_locking=True)
        for cs_id in obj_keys:
            # write cs to dict
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

            if cs_id not in syn_props[0]:
                continue
            # write syn to dict
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

            # syn and cs have the same ID
            # TODO: these should be refactored at some point, currently its not the same
            #  as before because the bounding box of the overlap object is used instead of
            #  the SJ bounding box. Also the background ratio was adapted
            # number of CS voxels in syn BB
            n_vxs_in_sjbb = np.prod(bb[1] - bb[0])
            # this is the fraction of CS voxels within the syn BB
            id_ratio = size_cs / n_vxs_in_sjbb
            # number of overlap voxels (syn voxels) divided by cs size
            cs_ratio = size / size_cs
            background_overlap_ratio = 1 - id_ratio
            # inverse 'CS' density:
            # c_cs_ids[u_cs_ids == 0] / n_vxs_in_sjbb  (previous version)
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
            # write voxels explicitly, Assumes, reasonably sized synapses!
            voxel_dc_store[cs_id] = voxel_dc.get_voxeldata(cs_id)
        voxel_dc_store.push()  # write voxel data explicitly
        voxel_dc_cs.push()
        this_attr_dc.push()
        this_attr_dc_cs.push()


def _write_props_to_syn_singlenode_thread(args):
    """"""
    # TODO: refactor in the same way as object mapping (reduce dictionaries)
    # TODO: refactor such that voxel data is stored during extraction
    cs_ids_ch = args[0]
    n_folders_fs = args[1]
    knossos_path = args[2]
    knossos_path_cs = args[3]
    nb_cpus = global_params.config['ncores_per_node']
    # consider use of multiprocessing manager dict
    # get cached dicts
    dict_p = "{}/cs_prop_dict.pkl".format(global_params.config.temp_path)

    with open(dict_p, "rb") as f:
        cs_props_tmp = pkl.load(f)

    # collect destinations of to-be-processed objects
    dest_dc = defaultdict(list)

    params = [(id_ch, [rep_helper.subfold_from_ix(
        store_key, n_folders_fs) for store_key in cs_ids_ch],
               n_folders_fs) for id_ch in chunkify(list(cs_props_tmp[0].keys()), nb_cpus)]
    res = start_multiprocess_imap(_generate_storage_lookup, params,
                                  nb_cpus=nb_cpus)
    for dc in res:
        for k, v in dc.items():
            dest_dc[k].extend(v)
    all_obj_keys = np.concatenate(list(dest_dc.values()))
    if len(all_obj_keys) == 0:
        log_extraction.critical('No object keys found during '
                                '`_write_props_to_syn_singlenode_thread`')
        return
    # keep only relevant data
    cs_props = [{}, {}, {}]
    for k in all_obj_keys:
        cs_props[0][k] = cs_props_tmp[0][k]
        cs_props[1][k] = cs_props_tmp[1][k]
        cs_props[2][k] = cs_props_tmp[2][k]
    del cs_props_tmp

    dict_p = "{}/syn_prop_dict.pkl".format(global_params.config.temp_path)
    with open(dict_p, "rb") as f:
        syn_props_tmp = pkl.load(f)

    syn_props = [{}, {}, {}]
    for k in all_obj_keys:
        try:
            syn_props[0][k] = syn_props_tmp[0][k]
        except KeyError:
            continue
        # fails if only first property of an object exists -> additional validity check
        syn_props[1][k] = syn_props_tmp[1][k]
        syn_props[2][k] = syn_props_tmp[2][k]
    del syn_props_tmp

    dict_p = "{}/cs_sym_cnt.pkl".format(global_params.config.temp_path)
    with open(dict_p, "rb") as f:
        cs_sym_cnt_tmp = pkl.load(f)

    cs_sym_cnt = {}
    for k in all_obj_keys:
        try:
            cs_sym_cnt[k] = cs_sym_cnt_tmp[k]
        except KeyError:  # no type prediction for this contact site
            pass
    del cs_sym_cnt_tmp

    dict_p = "{}/cs_asym_cnt.pkl".format(global_params.config.temp_path)
    with open(dict_p, "rb") as f:
        cs_asym_cnt_tmp = pkl.load(f)

    cs_asym_cnt = {}
    for k in all_obj_keys:
        try:
            cs_asym_cnt[k] = cs_asym_cnt_tmp[k]
        except KeyError:  # no type prediction for this contact site
            pass
    del cs_asym_cnt_tmp
    temp_folder = '{}/write_syn_props/{}_tmp_dcs/'.format(
        global_params.config.temp_path, cs_ids_ch[0])
    os.makedirs(temp_folder, exist_ok=True)
    basics.write_obj2pkl(temp_folder + "dest_dc.pkl", dest_dc)
    basics.write_obj2pkl(temp_folder + "cs_props.pkl", cs_props)
    basics.write_obj2pkl(temp_folder + "syn_props.pkl", syn_props)
    basics.write_obj2pkl(temp_folder + "cs_sym_cnt.pkl", cs_sym_cnt)
    basics.write_obj2pkl(temp_folder + "cs_asym_cnt.pkl", cs_asym_cnt)
    m_params = [(obj_id_mod, temp_folder, n_folders_fs, knossos_path, knossos_path_cs) for
                obj_id_mod in cs_ids_ch]

    start_multiprocess_imap(_helper_func, m_params, nb_cpus=nb_cpus,
                            verbose=False)
    shutil.rmtree(temp_folder, ignore_errors=True)


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


# iterate over the subcellular SV ID chunks
def _helper_func(args):
    obj_id_mod, temp_folder, n_folders_fs, knossos_path, knossos_path_cs = args

    dest_dc = basics.load_pkl2obj(temp_folder + "dest_dc.pkl")
    cs_props = basics.load_pkl2obj(temp_folder + "cs_props.pkl")
    syn_props = basics.load_pkl2obj(temp_folder + "syn_props.pkl")
    cs_sym_cnt = basics.load_pkl2obj(temp_folder + "cs_sym_cnt.pkl")
    cs_asym_cnt = basics.load_pkl2obj(temp_folder + "cs_asym_cnt.pkl")

    sd = segmentation.SegmentationDataset(n_folders_fs=n_folders_fs, obj_type='syn',
                                          working_dir=global_params.config.working_dir,
                                          version=0)

    sd_cs = segmentation.SegmentationDataset(n_folders_fs=n_folders_fs, obj_type='cs',
                                             working_dir=global_params.config.working_dir,
                                             version=0)

    obj_keys = dest_dc[rep_helper.subfold_from_ix(obj_id_mod, n_folders_fs)]
    if len(obj_keys) == 0:
        return
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

        if cs_id not in syn_props[0]:
            continue
        # write syn to dict
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

        # syn and cs have the same ID
        # TODO: these should be refactored at some point, currently its not the same
        #  as before because the bounding box of the overlap object is used instead of
        #  the SJ bounding box. ALso the background ratio was adapted
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

        # write voxels explicitely, Assumes, reasonably sized synapses!
        voxel_dc_store[cs_id] = voxel_dc.get_voxeldata(cs_id)
    voxel_dc_store.push()  # write voxel data explicitly
    voxel_dc_cs.push()
    this_attr_dc.push()
    this_attr_dc_cs.push()


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
    data = kd.load_seg(size=size, offset=offset, mag=1,
                       datatype=np.uint64).astype(np.uint32).swapaxes(0, 2)
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
    edges = edges.astype(np.uint32)
    arr = arr.astype(np.uint32)
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
    kd = kd_factory(global_params.config.kd_seg_path)
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
