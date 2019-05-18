# -*- coding: utf-8 -*-
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
import os
import tqdm
import time
import sys
import shutil
from collections import defaultdict
from knossos_utils import knossosdataset
from knossos_utils import chunky
knossosdataset._set_noprint(True)

from ..global_params import MESH_DOWNSAMPLING,\
    MESH_CLOSING, NCORES_PER_NODE
from .. import global_params
from .image import single_conn_comp_img
from ..mp import batchjob_utils as qu
from ..mp import mp_utils as sm
from ..backend.storage import AttributeDict, VoxelStorage, VoxelStorageDyn, MeshStorage
from ..reps import segmentation, segmentation_helper
from ..reps import rep_helper
from ..handler import basics
from ..proc.meshes import mesh_chunk, write_mesh2kzip
from . import log_proc
from ..extraction import object_extraction_wrapper as oew
from zmesh import Mesher

import sys


def dataset_analysis(sd, recompute=True, n_jobs=None, n_max_co_processes=None,
                     compute_meshprops=False):
    """ Analyze SegmentationDataset and extract and cache SegmentationObjects
    attributes as numpy arrays. Will only recognize dict/storage entries of type int
    for object attribute collection.


    :param sd: SegmentationDataset
    :param recompute: bool
        whether or not to (re-)compute key information of each object
        (rep_coord, bounding_box, size)
    :param n_jobs: int
        number of jobs
    :param qsub_pe: str
        qsub parallel environment
    :param qsub_queue: str
        qsub queue
    :param nb_cpus: int
        number of cores used for multithreading
        number of cores per worker for qsub jobs
    :param n_max_co_processes: int
        max number of workers running at the same time when using qsub
    :param compute_meshprops: bool
    """
    if n_jobs is None:
        n_jobs = global_params.NCORE_TOTAL  # individual tasks are very fast
    paths = sd.so_dir_paths
    if compute_meshprops:
        if not (sd.type in MESH_DOWNSAMPLING and sd.type in MESH_CLOSING):
            msg = 'SegmentationDataset of type "{}" has no configured mesh parameters. ' \
                  'Please add them to global_params.py accordingly.'
            log_proc.error(msg)
            raise ValueError(msg)
    # Partitioning the work
    multi_params = basics.chunkify(paths, n_jobs)
    multi_params = [(mps, sd.type, sd.version, sd.working_dir, recompute,
                     compute_meshprops) for mps in multi_params]

    # Running workers
    if not qu.batchjob_enabled():
        results = sm.start_multiprocess_imap(_dataset_analysis_thread,
                                             multi_params, nb_cpus=n_max_co_processes,
                                             debug=False)

    else:
        path_to_out = qu.QSUB_script(multi_params, "dataset_analysis",
                                     n_max_co_processes=n_max_co_processes,
                                     suffix=sd.type)
        out_files = glob.glob(path_to_out + "/*")
        results = []
        for out_file in out_files:
            with open(out_file, 'rb') as f:
                results.append(pkl.load(f))
        shutil.rmtree(os.path.abspath(path_to_out + "/../"), ignore_errors=True)
    # Creating summaries
    # TODO: This is a potential bottleneck for very large datasets
    # TODO: resulting cache-arrays might have different lengths if attribute is missing in
    # some dictionatries -> add checks!
    attr_dict = {}
    for this_attr_dict in results:
        for attribute in this_attr_dict:
            if not attribute in attr_dict:
                attr_dict[attribute] = []

            attr_dict[attribute] += this_attr_dict[attribute]

    for attribute in attr_dict:
        try:
            np.save(sd.path + "/%ss.npy" % attribute, attr_dict[attribute])
        except ValueError as e:
            log_proc.warn('ValueError {} encountered when writing numpy array caches in'
                          ' "dataset_analysis", this is currently caught by using `dtype=object`'
                          'which is not advised.'.format(e))
            if 'setting an array element with a sequence' in str(e):
                np.save(sd.path + "/%ss.npy" % attribute,
                        np.array(attr_dict[attribute], dtype=np.object))
            else:
                raise ValueError(e)


def _dataset_analysis_thread(args):
    """ Worker of dataset_analysis """

    paths = args[0]
    obj_type = args[1]
    version = args[2]
    working_dir = args[3]
    recompute = args[4]
    compute_meshprops = args[5]
    global_attr_dict = dict(id=[], size=[], bounding_box=[], rep_coord=[],
                            mesh_area=[])
    for p in paths:
        if not len(os.listdir(p)) > 0:
            os.rmdir(p)
        else:
            this_attr_dc = AttributeDict(p + "/attr_dict.pkl",
                                         read_only=not recompute)
            if recompute:
                this_vx_dc = VoxelStorage(p + "/voxel.pkl", read_only=True,
                                          disable_locking=True)
                so_ids = list(this_vx_dc.keys())  # e.g. isinstance(np.array([100, ], dtype=np.uint)[0], int) fails
            else:
                so_ids = list(this_attr_dc.keys())
            for so_id in so_ids:
                global_attr_dict["id"].append(so_id)
                so = segmentation.SegmentationObject(so_id, obj_type,
                                                     version, working_dir)
                so.attr_dict = this_attr_dc[so_id]
                if recompute:
                    # prevent loading voxels in case we use VoxelStorageDyn
                    if not isinstance(this_vx_dc, VoxelStorageDyn):  # use fall-back
                        so.load_voxels(voxel_dc=this_vx_dc)
                        so.calculate_rep_coord(voxel_dc=this_vx_dc)
                    else:
                        so.calculate_bounding_box(this_vx_dc)
                        so.calculate_rep_coord(this_vx_dc)
                        so.calculate_size(this_vx_dc)
                    so.attr_dict["rep_coord"] = so.rep_coord
                    so.attr_dict["bounding_box"] = so.bounding_box
                    so.attr_dict["size"] = so.size
                if compute_meshprops:
                    # if mesh does not exist beforehand, it will be generated
                    so.attr_dict["mesh_bb"] = so.mesh_bb
                    so.attr_dict["mesh_area"] = so.mesh_area
                for attribute in so.attr_dict.keys():
                    if attribute not in global_attr_dict:
                        global_attr_dict[attribute] = []
                    global_attr_dict[attribute].append(so.attr_dict[attribute])
                this_attr_dc[so_id] = so.attr_dict
            if recompute:
                this_attr_dc.push()
    return global_attr_dict


def map_objects_to_sv_multiple(sd, obj_types, kd_path, readonly=False, 
                               n_jobs=1000, qsub_pe=None, qsub_queue=None,
                               nb_cpus=None, n_max_co_processes=None):
    assert isinstance(obj_types, list)  # TODO: probably possible to optimize
    for obj_type in obj_types:
        map_objects_to_sv(sd, obj_type, kd_path, readonly=readonly, n_jobs=n_jobs,
                          nb_cpus=nb_cpus, n_max_co_processes=n_max_co_processes)
        

def map_objects_to_sv(sd, obj_type, kd_path, readonly=False, n_jobs=1000,
                      nb_cpus=None, n_max_co_processes=None):
    """
    TODO: (cython) optimization required! E.g. replace by single iteration over cell segm. and all cell organelle KDs/CDs
    Maps objects to SVs. The segmentation needs to be written to a KnossosDataset before running this

    Parameters
    ----------
    sd : SegmentationDataset
    obj_type : str
    kd_path : str
        path to knossos dataset containing the segmentation
    readonly : bool
        if True the mapping is only read from the segmentation objects and not
        computed. This requires the previous computation of the mapping for the
        mapped segmentation objects.
    n_jobs : int
        total number of jobs
    nb_cpus : int or None
        number of cores used for multithreading
        number of cores per worker for qsub jobs
    n_max_co_processes : int or None
        max number of workers running at the same time when using qsub
    """
    start = time.time()
    if sd.type != "sv":
        raise Exception("You are mapping to a non-sv dataset")
    assert obj_type in sd.version_dict
    seg_dataset = sd.get_segmentationdataset(obj_type)
    paths = seg_dataset.so_dir_paths

    # write cell organell mappings to cell organelle SV attribute dicts
    # Partitioning the work
    multi_params = basics.chunkify(paths, n_jobs)
    multi_params = [(mps, obj_type, sd.version_dict[obj_type], sd.working_dir,
                     kd_path, readonly) for mps in multi_params]
    # Running workers - Extracting mapping
    if qu.batchjob_enabled():
        path_to_out = qu.QSUB_script(multi_params, "map_objects", n_cores=nb_cpus,
                                     n_max_co_processes=n_max_co_processes)
        out_files = glob.glob(path_to_out + "/*")
        results = []
        for out_file in out_files:
            with open(out_file, 'rb') as f:
                results.append(pkl.load(f))
    else:
        results = sm.start_multiprocess_imap(_map_objects_thread, multi_params,
                                             nb_cpus=n_max_co_processes, verbose=True,
                                             debug=False)

    # write cell organell mappings to cell SV attribute dicts
    sv_obj_map_dict = defaultdict(dict)
    for result in results:
        for sv_key, value in result.items():
            sv_obj_map_dict[sv_key].update(value)

    mapping_dict_path = seg_dataset.path + "/sv_%s_mapping_dict.pkl" % sd.version
    with open(mapping_dict_path, "wb") as f:
        pkl.dump(sv_obj_map_dict, f)

    paths = sd.so_dir_paths

    # Partitioning the work
    multi_params = basics.chunkify(paths, n_jobs)
    multi_params = [(path_block, obj_type, mapping_dict_path) for path_block in multi_params]

    # Running workers - Writing mapping to SVs
    if not qu.batchjob_enabled():
        sm.start_multiprocess_imap(_write_mapping_to_sv_thread, multi_params,
                                   nb_cpus=n_max_co_processes, debug=False)
    else:
        qu.QSUB_script(multi_params, "write_mapping_to_sv", script_folder=None,
                       n_cores=nb_cpus, n_max_co_processes=n_max_co_processes)
    log_proc.debug("map_objects_to_sv: %.1f min" % ((time.time() - start) / 60.))


def _map_objects_thread(args):
    """Worker of map_objects_to_sv"""
    # TODO: this needs to be done densely by matching cell organelle segmentation (see corresponding ChunkDataset
    #  which is an intermediate result of 'from_probabilities_to_objects') to SV segmentation

    paths = args[0]
    obj_type = args[1]
    obj_version = args[2]
    working_dir = args[3]
    kd_path = args[4]
    readonly = args[5]
    global_params.wd = working_dir
    if len(args) > 6:
        datatype = args[6]
    else:
        datatype = np.uint64
    kd = knossosdataset.KnossosDataset()
    kd.initialize_from_knossos_path(kd_path)
    seg_dataset = segmentation.SegmentationDataset(obj_type, version=obj_version,
                                                   working_dir=working_dir)
    sv_id_dict = {}
    for p in paths:
        this_attr_dc = AttributeDict(p + "/attr_dict.pkl", read_only=readonly,
                                     disable_locking=True)
        this_vx_dc = VoxelStorage(p + "/voxel.pkl", read_only=True,
                                  disable_locking=True)
        for so_id in this_vx_dc.keys():
            so = seg_dataset.get_segmentation_object(so_id)
            so.attr_dict = this_attr_dc[so_id]
            bb = so.bounding_box
            # check object BB, TODO: HACK because kd.from_overlaycubes_to_list(vx_list,
            #  datatype=datatype) will load the data via the object's BB --> new mapping is
            #  needed asap
            if np.linalg.norm((bb[1] - bb[0]) * seg_dataset.scaling) > global_params.thresh_mi_bbd_mapping:
                log_proc.warn(
                    'Skipped huge MI with size: {}, offset: {}, mi_id: {}, mi_c'
                    'oord: {}'.format(so.size, so.bounding_box[0], so_id, so.rep_coord))
                so.attr_dict["mapping_ids"] = []
                so.attr_dict["mapping_ratios"] = []
                this_attr_dc[so_id] = so.attr_dict
                continue
            so.load_voxels(voxel_dc=this_vx_dc)
            if readonly:
                if "mapping_ids" in so.attr_dict:
                    ids = so.attr_dict["mapping_ids"]
                    id_ratios = so.attr_dict["mapping_ratios"]

                    for i_id in range(len(ids)):
                        if ids[i_id] in sv_id_dict:
                            sv_id_dict[ids[i_id]][so_id] = id_ratios[i_id]
                        else:
                            sv_id_dict[ids[i_id]] = {so_id: id_ratios[i_id]}
            else:
                if np.product(so.shape) > 1e12:  # TODO: Seems hacky
                    continue
                vx_list = np.argwhere(so.voxels) + so.bounding_box[0]
                try:
                    id_list = kd.from_overlaycubes_to_list(vx_list, datatype=datatype)
                except:
                    log_proc.error('Could not load overlaycube '
                                   'during object mapping')
                    continue
                ids, id_counts = np.unique(id_list, return_counts=True)  # cell SV IDs
                id_ratios = id_counts / float(np.sum(id_counts))

                for i_id in range(len(ids)):
                    if ids[i_id] in sv_id_dict:
                        sv_id_dict[ids[i_id]][so_id] = id_ratios[i_id]
                    else:
                        sv_id_dict[ids[i_id]] = {so_id: id_ratios[i_id]}
                so.attr_dict["mapping_ids"] = ids
                so.attr_dict["mapping_ratios"] = id_ratios
                this_attr_dc[so_id] = so.attr_dict
        if not readonly:
            this_attr_dc.push()
    return sv_id_dict


def _write_mapping_to_sv_thread(args):
    """ Worker of map_objects_to_sv """

    paths = args[0]
    obj_type = args[1]
    mapping_dict_path = args[2]

    with open(mapping_dict_path, "rb") as f:
        mapping_dict = pkl.load(f)

    for p in paths:
        this_attr_dc = AttributeDict(p + "/attr_dict.pkl",
                                     read_only=False)
        for sv_id in this_attr_dc.keys():
            this_attr_dc[sv_id]["mapping_%s_ids" % obj_type] = \
                list(mapping_dict[sv_id].keys())
            this_attr_dc[sv_id]["mapping_%s_ratios" % obj_type] = \
                list(mapping_dict[sv_id].values())
        this_attr_dc.push()


def map_subcell_extract_props(kd_seg_path, kd_organelle_paths, n_folders_fs=1000, n_folders_fs_sc=1000,
                              n_chunk_jobs=None, n_cores=1, n_max_co_processes=None,
                              cube_of_interest_bb=None, chunk_size=None, log=None, mesh_provided=False):
    """Replaces `map_objects_to_sv` and parts of `from_ids_to_objects`.

    Extracts segmentation properties for each SV in cell and subcellular segmentation.
    Requires KDs at `kd_seg_path` and `kd_organelle_paths`

    Step 1: Extract properties and overlap voxels locally
    Step 2: Analyze resulting IDs (get result locations of each SV object)
    Step 3: Write out combined results for each SV object according to SegmentationDataset chunking

    Parameters
    ----------
    kd_seg_path : str
    kd_organelle_paths : Dict[int]
    n_folders_fs
    n_folders_fs_sc :
        `n_folders_fs` for subcellular structures
    n_chunk_jobs
    n_cores
    n_max_co_processes
    cube_of_interest_bb
    chunk_size

    Returns
    -------

    """
    kd = basics.kd_factory(kd_seg_path)
    assert sys.version_info >= (3, 6)  # below, dictionaries are unordered!
    for k, v in kd_organelle_paths.items():
        if not np.array_equal(basics.kd_factory(v).boundary, kd.boundary):
            msg = "Data shape of subcellular structures '{}' differs from " \
                  "cell segmentation data. {} vs. {}".format(
                k, basics.kd_factory(v).boundary, kd.boundary)
            log_proc.error(msg)
            raise ValueError(msg)

    # get chunks
    if log is None:
        log = log_proc
    if n_chunk_jobs is None:
        n_chunk_jobs = global_params.NCORE_TOTAL * 2
    if chunk_size is None:
        chunk_size = [512, 512, 512]
    if cube_of_interest_bb is None:
        cube_of_interest_bb = [np.zeros(3, dtype=np.int), kd.boundary]
    size = cube_of_interest_bb[1] - cube_of_interest_bb[0] + 1
    offset = cube_of_interest_bb[0]
    cd_dir = "{}/chunkdatasets/tmp/".format(global_params.config.temp_path)
    cd = chunky.ChunkDataset()
    cd.initialize(kd, kd.boundary, chunk_size, cd_dir,
                  box_coords=[0, 0, 0], fit_box_size=True)
    chunk_list, _ = oew.calculate_chunk_numbers_for_box(cd, offset=offset,
                                                        size=size)

    all_times = []
    step_names = []

    # extracting mapping
    start = time.time()
    # multi_params = basics.chunkify(chunk_list, 1)
    multi_params = basics.chunkify(chunk_list, n_chunk_jobs)
    multi_params = [(chs, chunk_size, kd_seg_path, list(kd_organelle_paths.values()), worker_nr, mesh_provided)
                    for chs, worker_nr in zip(multi_params, range(len(multi_params)))]

    if qu.batchjob_enabled():
        path_to_out = qu.QSUB_script(multi_params, "map_subcell_extract_props", n_cores=n_cores,
                                     n_max_co_processes=n_max_co_processes)
        out_files = glob.glob(path_to_out + "/*")
        results = []
        for out_file in out_files:
            with open(out_file, 'rb') as f:
                results.append(pkl.load(f))
    else:
        results = sm.start_multiprocess_imap(_map_subcell_extract_props_thread, multi_params,
                                             nb_cpus=n_max_co_processes, verbose=True,
                                             debug=False)
    all_times.append(time.time() - start)
    step_names.append("extract and map segmentation objects")

    # reduce step
    start = time.time()
    tot_cp = [{}, defaultdict(list), {}]
    tot_scp = [[{}, defaultdict(list), {}] for _ in range(len(kd_organelle_paths))]
    tot_scm = [{} for _ in range(len(kd_organelle_paths))]

    cell_mesh_workers = defaultdict(list)
    subcell_mesh_workers = [defaultdict(list) for _ in range(len(kd_organelle_paths))]
    list_of_workers = []

    for cp_dc, scp_dcs, scm_dcs, worker_nr, ids_list in results:
        merge_prop_dicts([tot_cp, cp_dc])
        list_of_workers.append(worker_nr)
        for cell_id in ids_list[0]:
            cell_mesh_workers[cell_id].append(worker_nr)
        # iterate over each subcellular structure
        for ii in range(len(kd_organelle_paths)):
            merge_map_dicts([tot_scm[ii], scm_dcs[ii]])
            merge_prop_dicts([tot_scp[ii], scp_dcs[ii]])
            for subcell_id in ids_list[ii+1]:
                subcell_mesh_workers[ii][subcell_id].append(worker_nr)
        del cp_dc, scp_dcs, scm_dcs, ids_list  # will remove redundant data inside results

    # convert mapping dicts to store ratio of number of overlapping voxels
    prop_dict_p = "{}/sv_prop_dict.pkl".format(global_params.config.temp_path)
    with open(prop_dict_p, "wb") as f:
        pkl.dump(tot_cp, f)
    del tot_cp

    dict_paths = [prop_dict_p, ]
    for ii, k in enumerate(global_params.existing_cell_organelles):
        convert_nvox2ratio_mapdict(tot_scm[ii])  # in-place conversion
        # subcell ID -> cell ID -> ratio
        mapping_dict_path = "{}/{}_mapping_dict.pkl".format(global_params.config.temp_path, k)
        with open(mapping_dict_path, "wb") as f:
            pkl.dump(tot_scm[ii], f)

        # cell ID -> subcell ID -> ratio
        tot_scm_inv = invert_mdc(tot_scm[ii])
        mapping_dict_path_inv = "{}_inv.pkl".format(mapping_dict_path[:-4])
        with open(mapping_dict_path_inv, "wb") as f:
            pkl.dump(tot_scm_inv, f)
        tot_scm[ii] = {}  # free space
        del tot_scm_inv

        prop_dict = tot_scp[ii]
        prop_dict_p = "{}/{}_prop_dict.pkl".format(global_params.config.temp_path, k)
        with open(prop_dict_p, "wb") as f:
            pkl.dump(prop_dict, f)
        tot_scp[ii] = []  # free space

        dict_paths += [mapping_dict_path, mapping_dict_path_inv]
    all_times.append(time.time() - start)

    step_names.append("conversion of results")

    # TODO: do write out chunk-wise by analyzing location of intermediate results for each SV and write
    #  them packaged according to the SegmentationDataset chunks
    # # analyze output
    # start = time.time()
    #
    #
    # all_times.append(time.time() - start)
    # step_names.append("analyze output")

    # writing cell SV properties to SD
    start = time.time()
    multi_params = [(sv_id_block, n_folders_fs, cell_mesh_workers, mesh_provided)
                    for sv_id_block in basics.chunkify(np.arange(n_folders_fs), n_chunk_jobs)]
    if not qu.batchjob_enabled():
        sm.start_multiprocess_imap(_write_props_to_sv_thread, multi_params,
                                   nb_cpus=n_max_co_processes, debug=False)
    else:
        qu.QSUB_script(multi_params, "write_props_to_sv", script_folder=None,
                       n_cores=n_cores, n_max_co_processes=n_max_co_processes,
                       remove_jobfolder=True)
    all_times.append(time.time() - start)
    step_names.append("write cell SV dataset")
    sv_sd = segmentation.SegmentationDataset(working_dir=global_params.config.working_dir,
                                             obj_type="sv", version=0)
    dataset_analysis(sv_sd, recompute=True, compute_meshprops=False)

    # write to subcell. SV attribute dicts
    multi_params = [(sv_id_block, n_folders_fs_sc, subcell_mesh_workers)
                    for sv_id_block in basics.chunkify(np.arange(n_folders_fs_sc), n_chunk_jobs)]
    if not qu.batchjob_enabled():
        sm.start_multiprocess_imap(_write_props_to_sc_thread, multi_params,
                                   nb_cpus=n_max_co_processes, debug=False)
    else:
        qu.QSUB_script(multi_params, "write_props_to_sc", script_folder=None,
                       n_cores=n_cores, n_max_co_processes=n_max_co_processes,
                       remove_jobfolder=True)
    step_names.append("write subcellular SV dataset")
    for k in global_params.existing_cell_organelles:
        sc_sd = segmentation.SegmentationDataset(working_dir=global_params.config.working_dir,
                                                 obj_type=k, version=0)
        dataset_analysis(sc_sd, recompute=True, compute_meshprops=False)

    # clear temporary files
    for p in dict_paths:
        os.remove(p)

    for worker_nr in list_of_workers:
        p = global_params.config.temp_path + "/tmp_meshes_worker_" + str(worker_nr) + ".pkl"
        os.remove(p)

    shutil.rmtree(cd_dir, ignore_errors=True)
    if qu.batchjob_enabled():  # remove job directory of `map_subcell_extract_props`
        shutil.rmtree(os.path.abspath(path_to_out + "/../"), ignore_errors=True)
    log.debug("Time overview [from_probabilities_to_objects]:")
    for ii in range(len(all_times)):
        log.debug("%s: %.3fs" % (step_names[ii], all_times[ii]))
    log.debug("--------------------------")
    log.debug("Total Time: %.1f min" % (np.sum(all_times) / 60))
    log.debug("--------------------------")
    print("\n\n KONIEC IMPREZY\n\n\n")
    sys.exit()


def _map_subcell_extract_props_thread(args):
    from syconn.reps.find_object_properties_C import map_subcell_extract_propsC
    chunks = args[0]
    chunk_size = args[1]
    kd_cell_p = args[2]
    kd_subcell_ps = args[3]
    worker_nr = args[4]
    mesh_provided = args[5]

    kd_cell = basics.kd_factory(kd_cell_p)
    cd_dir = "{}/chunkdatasets/tmp/".format(global_params.config.temp_path)
    cd = chunky.ChunkDataset()
    cd.initialize(kd_cell, kd_cell.boundary, chunk_size, cd_dir,
                  box_coords=[0, 0, 0], fit_box_size=True)
    kd_subcells = [basics.kd_factory(kd_subcell_p) for kd_subcell_p in kd_subcell_ps]
    n_subcell = len(kd_subcells)

    cpd_lst = [{}, defaultdict(list), {}]    # cell property dicts
    scpd_lst = [[{}, defaultdict(list), {}] for _ in range(n_subcell)]   # subcell. property dicts
    scmd_lst = [{} for _ in range(n_subcell)]   # subcell. mapping dicts

    big_mesh_dict = {}
    big_mesh_dict['sv'] = defaultdict(list)
    for organelle in global_params.existing_cell_organelles:
        big_mesh_dict[organelle] = defaultdict(list)

    # iterate over chunks and store information in property dicts for subcellular and cellular structures
    for ch_id in chunks:
        ch = cd.chunk_dict[ch_id]
        offset, size = ch.coordinates.astype(np.int), ch.size
        cell_d = kd_cell.from_overlaycubes_to_matrix(size, offset)
        tmp_cell_mesh = defaultdict(list)
        tmp_cell_mesh = find_meshes(cell_d, offset)
        # get all segmentation arrays concatenates as 4D array: [C, X, Y, Z]
        tmp_subcell_meshes = [defaultdict(list) for _ in kd_subcells]
        subcell_d = []
        for kd_sc, i in zip(kd_subcells, range(len(kd_subcells))):
            subc_d = kd_sc.from_overlaycubes_to_matrix(size, offset)
            tmp_subcell_meshes[i] = find_meshes(subc_d, offset)
            subcell_d.append(subc_d[None, ])  # add auxiliary axis
        subcell_d = np.concatenate(subcell_d)
        cell_prop_dicts, subcell_prop_dicts, subcell_mapping_dicts = map_subcell_extract_propsC(cell_d, subcell_d)
        # reorder to match [[rc, bb, size], [rc, bb, size]] for e.g. [mi, vc]
        subcell_prop_dicts = [[subcell_prop_dicts[0][ii], subcell_prop_dicts[1][ii],
                               subcell_prop_dicts[2][ii]] for ii in range(n_subcell)]
        # merge cell properties: list list of dicts
        merge_prop_dicts([cpd_lst, cell_prop_dicts], offset)

        # collect subcell properties: list of list of dicts
        # collect subcell mappings to cell SVs: list of list of dicts and list of dict of dict of int

        merge_meshes_dict(big_mesh_dict['sv'], tmp_cell_mesh)
        for organelle, ii in zip(global_params.existing_cell_organelles,
                                 range(len(global_params.existing_cell_organelles))):
            merge_map_dicts([scmd_lst[ii], subcell_mapping_dicts[ii]])
            merge_prop_dicts([scpd_lst[ii], subcell_prop_dicts[ii]], offset)
            merge_meshes_dict(big_mesh_dict[organelle], tmp_subcell_meshes[ii])

        del subcell_prop_dicts
        del subcell_mapping_dicts
        del cell_prop_dicts
        del tmp_subcell_meshes
        del tmp_cell_mesh

    output_worker = open(global_params.config.temp_path + "/tmp_meshes_worker_" + str(worker_nr) + ".pkl", 'wb')
    pkl.dump(big_mesh_dict, output_worker)
    output_worker.close()
    ids_list = []
    ids_list.append(list(big_mesh_dict['sv'].keys()))
    for organelle in global_params.existing_cell_organelles:
        ids_list.append(list(big_mesh_dict[organelle].keys()))

    return cpd_lst, scpd_lst, scmd_lst, worker_nr, ids_list


def find_meshes(chunk, offset):

    """

    """
    scaling = np.array(global_params.config.entries["Dataset"]["scaling"])
    mesher = Mesher((1, 1, 1))
    mesher.mesh(chunk)

    meshes = {}
    for obj_id in mesher.ids():
        tmp = mesher.get_mesh(obj_id, normals=True, simplification_factor=40, max_simplification_error=0)
        # the values of simplification_factor & max_simplification_error are random

        tmp.vertices[:] = (tmp.vertices[:, ::-1] + offset) * scaling
        meshes[obj_id] = [tmp.faces.flatten(), tmp.vertices.flatten(), tmp.normals.flatten()]
        mesher.erase(obj_id)

    mesher.clear()

    return meshes


def merge_meshes_dict(m_storage, tmp_dict):

    """ Merge meshes dictionaries:

    m_storage: list dictionary
    tmp_dict: list dictionary
    {obj_id: [faces, vertices, normals]}
    """
    for obj_id in tmp_dict:
        merge_meshes_single(m_storage, obj_id, tmp_dict[obj_id])


def merge_meshes_single(m_storage, obj_id, tmp_dict):

    """ Merge meshes dictionaries:
    m_storage: objec of type MeshStorage
    tmp_dict: list dictionary
    """
    if obj_id not in m_storage:
        m_storage[obj_id] = [tmp_dict[0], tmp_dict[1], tmp_dict[2]]

    else:
        n_el = int((len(m_storage[obj_id][1])) / 3)
        tmp_dict[0] += n_el
        m_storage[obj_id][0] = np.concatenate((m_storage[obj_id][0], tmp_dict[0]))
        m_storage[obj_id][1] = np.concatenate((m_storage[obj_id][1], tmp_dict[1]))
        m_storage[obj_id][2] = np.concatenate((m_storage[obj_id][2], tmp_dict[2]))


def merge_prop_dicts(prop_dicts, offset=None):
    """Merge property dicts in-place. All values will be stored in the first dict."""
    tot_rc = prop_dicts[0][0]
    tot_bb = prop_dicts[0][1]
    tot_size = prop_dicts[0][2]
    for el in prop_dicts[1:]:
        if len(el[0]) == 0:
            continue
        if offset is not None:
            for k in el[0]:  # update chunk offset  # TODO: could be done at the end of the map_extract cython code
                el[0][k] += offset
        tot_rc.update(el[0])  # just overwrite existing elements
        for k, v in el[1].items():
            if offset is None:
                bb = v
            else:
                bb = v + offset
            tot_bb[k].append(bb)  # collect all bounding boxes to enable efficient data loading
        for k, v in el[2].items():
            if k in tot_size:
                tot_size[k] += v
            else:
                tot_size[k] = v


def convert_nvox2ratio_mapdict(map_dc):
    """convert number of overlap voxels of each subcellular structure object
     inside the mapping dicts to each cell SV
     (subcell ID -> cell ID -> number overlap vxs) to fraction.
     """
    # TODO consider to implement in cython
    for subcell_id, subcell_dc in map_dc.items():
        s = np.sum(list(subcell_dc.values()))  # total number of overlap voxels
        for k, v in subcell_dc.items():
            map_dc[subcell_id][k] = subcell_dc[k] / s


def invert_mdc(mapping_dict):
    """Inverts mapping dict to: cell ID -> subcell ID -> value (ratio or voxel count)"""
    mdc_inv = {}
    for subcell_id, subcell_dc in mapping_dict.items():
        for cell_id, v in subcell_dc.items():
            if cell_id not in mdc_inv:
                mdc_inv[cell_id] = {subcell_id: v}
            else:
                mdc_inv[cell_id][subcell_id] = v
    return mdc_inv


def merge_map_dicts(map_dicts):
    """
    Merge map dictionaries in-place. Values will be stored in first dictionary

    Parameters
    ----------
    map_dicts

    Returns
    -------

    """
    tot_map = map_dicts[0]
    for el in map_dicts[1:]:
        # iterate over subcell. ids with dictionaries as values which store
        # the number of overlap voxels to cell SVs
        for sc_id, sc_dc in el.items():
            if sc_id in tot_map:
                for cellsv_id, ol_vx_cnt in sc_dc.items():
                    if cellsv_id in tot_map[sc_id]:
                        tot_map[sc_id][cellsv_id] += ol_vx_cnt
                    else:
                        tot_map[sc_id][cellsv_id] = ol_vx_cnt
            else:
                tot_map[sc_id] = sc_dc


def _analyze_res_map_extract():
    """TBD"""
    return


def _write_props_to_sc_thread(args):
    """"""

    obj_id_chs = args[0]
    n_folders_fs = args[1]
    subcell_mesh_workers = args[2]

    # iterate over the subcell structures
    for ii, k in enumerate(global_params.existing_cell_organelles):
        # get cached mapping and property dicts of current subcellular structure
        mapping_dict_path = "{}/{}_mapping_dict.pkl".format(global_params.config.temp_path, k)
        with open(mapping_dict_path, "rb") as f:
            mapping_dict = pkl.load(f)
        prop_dict_p = "{}/{}_prop_dict.pkl".format(global_params.config.temp_path, k)
        with open(prop_dict_p, "rb") as f:
            prop_dict = pkl.load(f)

        # store destinations for each existing obj
        dest_dc = defaultdict(list)
        for subcell_id in mapping_dict:
            dest_dc[rep_helper.subfold_from_ix(subcell_id, n_folders_fs)].append(subcell_id)

        # get SegmentationDataset of current subcell.
        sc_sd = segmentation.SegmentationDataset(n_folders_fs=n_folders_fs, obj_type=k,
                                                 working_dir=global_params.config.working_dir, version=0)

        # iterate over the subcellular SV ID chunks
        for obj_id_mod in obj_id_chs:
            obj_keys = dest_dc[rep_helper.subfold_from_ix(obj_id_mod, n_folders_fs)]
            # get dummy segmentation object to fetch attribute dictionary for this batch of object IDs
            dummy_so = sc_sd.get_segmentation_object(obj_id_mod)
            attr_p = dummy_so.attr_dict_path
            vx_p = dummy_so.voxel_path
            this_attr_dc = AttributeDict(attr_p, read_only=False, disable_locking=True)
            voxel_dc = VoxelStorageDyn(vx_p, voxel_mode=False, voxeldata_path=global_params.config.kd_organelle_seg_paths[k],
                                       read_only=False, disable_locking=True)
            for sc_id in obj_keys:
                if sc_id in subcell_mesh_workers[ii]:
                    mesh_area = save_mesh_and_return_area(sc_id, k, subcell_mesh_workers[ii][sc_id], dummy_so)
                    this_attr_dc[sc_id]["mesh_area"] = mesh_area
                else:
                    obj_mesh = MeshStorage(dummy_so.mesh_path, disable_locking=True, read_only=False)
                    obj_mesh.push()
                    this_attr_dc[sc_id]["mesh_area"] = 0

                this_attr_dc[sc_id]["mapping_ids"] = \
                    list(mapping_dict[sc_id].keys())
                this_attr_dc[sc_id]["mapping_ratios"] = \
                    list(mapping_dict[sc_id].values())
                rp = prop_dict[0][sc_id]
                bbs = np.concatenate(prop_dict[1][sc_id])
                size = prop_dict[2][sc_id]
                this_attr_dc[sc_id]["rep_coord"] = rp
                this_attr_dc[sc_id]["bounding_box"] = np.array(
                    [bbs[:, 0].min(axis=0), bbs[:, 1].max(axis=0)])
                this_attr_dc[sc_id]["size"] = size
                voxel_dc[sc_id] = bbs
                # TODO: make use of these stored properties downstream during the reduce step
                voxel_dc.increase_object_size(sc_id, size)
                voxel_dc.set_object_repcoord(sc_id, rp)
            voxel_dc.push()
            this_attr_dc.push()


def _write_props_to_sv_thread(args):
    """"""

    obj_id_chs = args[0]
    n_folders_fs = args[1]
    cell_mesh_workers = args[2]
    mesh_provided = args[3]

    # iterate over the subcell structures
    # get cached mapping and property dicts of current subcellular structure
    mapping_dicts = {}
    for ii, k in enumerate(global_params.existing_cell_organelles):
        mapping_dict_path = "{}/{}_mapping_dict_inv.pkl".format(global_params.config.temp_path, k)
        with open(mapping_dict_path, "rb") as f:
            mapping_dicts[k] = pkl.load(f)

    prop_dict_p = "{}/sv_prop_dict.pkl".format(global_params.config.temp_path)
    with open(prop_dict_p, "rb") as f:
        prop_dict = pkl.load(f)

    # store destinations for each existing obj
    dest_dc = defaultdict(list)
    for k in prop_dict[0]:  # use dictionary for rep coord (just use any, they all share the same keys)
        dest_dc[rep_helper.subfold_from_ix(k, n_folders_fs)].append(k)

    # get SegmentationDataset of current subcell.
    sv_sd = segmentation.SegmentationDataset(n_folders_fs=n_folders_fs,
                                             obj_type="sv", working_dir=global_params.config.working_dir, version=0)
    # iterate over the subcellular SV ID chunks
    for obj_id_mod in obj_id_chs:
        obj_keys = dest_dc[rep_helper.subfold_from_ix(obj_id_mod, n_folders_fs)]
        # get dummy segmentation object to fetch attribute dictionary for this batch of object IDs
        dummy_so = sv_sd.get_segmentation_object(obj_id_mod)
        attr_p = dummy_so.attr_dict_path
        vx_p = dummy_so.voxel_path
        this_attr_dc = AttributeDict(attr_p, read_only=False, disable_locking=True)
        voxel_dc = VoxelStorageDyn(vx_p, voxel_mode=False, voxeldata_path=global_params.config.kd_seg_path,
                                   read_only=False, disable_locking=True)
        for sv_id in obj_keys:
            if not mesh_provided:
                if sv_id in cell_mesh_workers:
                    mesh_area = save_mesh_and_return_area(sv_id, 'sv', cell_mesh_workers[sv_id], dummy_so)
                    this_attr_dc[sv_id]["mesh_area"] = mesh_area
                else:
                    obj_mesh = MeshStorage(dummy_so.mesh_path, disable_locking=True, read_only=False)
                    obj_mesh.push()
                    this_attr_dc[sv_id]["mesh_area"] = 0

            for ii, k in enumerate(global_params.existing_cell_organelles):
                if sv_id not in mapping_dicts[k]:  # no object of this type mapped to current cell SV
                    continue
                this_attr_dc[sv_id]["mapping_%s_ids" % k] = \
                    list(mapping_dicts[k][sv_id].keys())
                this_attr_dc[sv_id]["mapping_%s_ratios" % k] = \
                    list(mapping_dicts[k][sv_id].values())
            rp = prop_dict[0][sv_id]
            bbs = np.concatenate(prop_dict[1][sv_id])
            size = prop_dict[2][sv_id]
            this_attr_dc[sv_id]["rep_coord"] = rp
            this_attr_dc[sv_id]["bounding_box"] = np.array(
                [bbs[:, 0].min(axis=0), bbs[:, 1].max(axis=0)])
            this_attr_dc[sv_id]["size"] = size
            voxel_dc[sv_id] = bbs
            # TODO: make use of these stored properties downstream during the reduce step
            voxel_dc.increase_object_size(sv_id, size)
            voxel_dc.set_object_repcoord(sv_id, rp)
        voxel_dc.push()
        this_attr_dc.push()


def save_mesh_and_return_area(obj_id, obj_type, workers_list, so):

    obj_mesh = MeshStorage(so.mesh_path, disable_locking=True, read_only=False)
    for worker_nr in workers_list:
        p = global_params.config.temp_path + "/tmp_meshes_worker_" + str(worker_nr) + ".pkl"
        pkl_file = open(p, 'rb')
        single_mesh = pkl.load(pkl_file)[obj_type][obj_id]
        pkl_file.close()
        merge_meshes_single(obj_mesh, obj_id, single_mesh)
    obj_mesh.push()

    return so.mesh_area


def binary_filling_cs(cs_sd, n_iterations=13, stride=1000,
                      qsub_pe=None, qsub_queue=None, nb_cpus=None,
                      n_max_co_processes=None):
    paths = cs_sd.so_dir_paths

    # Partitioning the work
    multi_params = []
    for path_block in [paths[i:i + stride] for i in range(0, len(paths), stride)]:
        multi_params.append([path_block, cs_sd.version, cs_sd.working_dir,
                             n_iterations])

    # Running workers
    if (qsub_pe is None and qsub_queue is None) or not qu.batchjob_enabled():
        results = sm.start_multiprocess(_binary_filling_cs_thread,
                                        multi_params, nb_cpus=nb_cpus)

    elif qu.batchjob_enabled():
        path_to_out = qu.QSUB_script(multi_params,
                                     "binary_filling_cs",
                                     pe=qsub_pe, queue=qsub_queue,
                                     script_folder=None,
                                     n_cores=nb_cpus,
                                     n_max_co_processes=n_max_co_processes,
                                     remove_jobfolder=True)
    else:
        raise Exception("QSUB not available")


def _binary_filling_cs_thread(args):
    paths = args[0]
    obj_version = args[1]
    working_dir = args[2]
    n_iterations = args[3]

    cs_sd = segmentation.SegmentationDataset('cs',
                                             version=obj_version,
                                             working_dir=working_dir)

    for p in paths:
        this_vx_dc = VoxelStorage(p + "/voxel.pkl", read_only=False)

        for so_id in this_vx_dc.keys():
            so = cs_sd.get_segmentation_object(so_id)
            # so.attr_dict = this_attr_dc[so_id]
            so.load_voxels(voxel_dc=this_vx_dc)
            filled_voxels = segmentation_helper.binary_closing(
                so.voxels.copy(), n_iterations=n_iterations)

            this_vx_dc[so_id] = [filled_voxels], [so.bounding_box[0]]

        this_vx_dc.push()


def init_sos(sos_dict):
    loc_dict = sos_dict.copy()
    svixs = loc_dict["svixs"]
    del loc_dict["svixs"]
    sos = [segmentation.SegmentationObject(ix, **loc_dict) for ix in svixs]
    return sos


def sos_dict_fact(svixs, version=None, scaling=None, obj_type="sv",
                  working_dir=None, create=False):
    if working_dir is None:
        working_dir = global_params.config.working_dir
    if scaling is None:
        scaling = global_params.config.entries['Dataset']['scaling']
    sos_dict = {"svixs": svixs, "version": version,
                "working_dir": working_dir, "scaling": scaling,
                "create": create, "obj_type": obj_type}
    return sos_dict


def predict_sos_views(model, sos, pred_key, nb_cpus=1, woglia=True,
                      verbose=False, raw_only=False, single_cc_only=False,
                      return_proba=False):
    """

    Parameters
    ----------
    model :
    sos :
    pred_key :
    nb_cpus :
    woglia :
    verbose :
    raw_only :
    single_cc_only :
    return_proba :

    Returns
    -------

    """
    nb_chunks = np.max([1, len(sos) // 200])
    so_chs = basics.chunkify(sos, nb_chunks)
    all_probas = []
    if verbose:
        pbar = tqdm.tqdm(total=len(sos))
    for ch in so_chs:
        views = sm.start_multiprocess_obj("load_views", [[sv, {"woglia": woglia,
                                          "raw_only": raw_only}]
                                          for sv in ch], nb_cpus=nb_cpus)
        proba = predict_views(model, views, ch, pred_key, verbose=False,
                              single_cc_only=single_cc_only,
                              return_proba=return_proba, nb_cpus=nb_cpus)
        if verbose:
            pbar.update(len(ch))
        if return_proba:
            all_probas.append(np.concatenate(proba))
    if verbose:
        pbar.close()
    if return_proba:
        return np.concatenate(all_probas)


def predict_views(model, views, ch, pred_key, single_cc_only=False,
                  verbose=False, return_proba=False, nb_cpus=1):
    """
    Will not be written to disk if return_proba is True.

    Parameters
    ----------
    model : nn.Model
    views : np.array
    ch : List[SegmentationObject]
    pred_key : str
    single_cc_only : bool
    verbose : bool
    return_proba : bool
    nb_cpus : int

    Returns
    -------

    """
    for kk in range(len(views)):
        data = views[kk]
        for i in range(len(data)):
            if single_cc_only:
                sing_cc = np.concatenate([single_conn_comp_img(data[i, 0, :1]),
                                          single_conn_comp_img(data[i, 0, 1:])])
                data[i, 0] = sing_cc
        views[kk] = data
    part_views = np.cumsum([0] + [len(v) for v in views])
    assert len(part_views) == len(views) + 1
    views = np.concatenate(views)
    probas = model.predict_proba(views, verbose=verbose)
    so_probas = []
    for ii, _ in enumerate(part_views[:-1]):
        sv_probas = probas[part_views[ii]:part_views[ii + 1]]
        so_probas.append(sv_probas)
    assert len(part_views) == len(so_probas) + 1
    if return_proba:
        return so_probas
    if nb_cpus > 1:  # make sure locking is enabled if multiprocessed
        for so in ch:
            so.enable_locking = True
    params = [[so, prob, pred_key] for so, prob in zip(ch, so_probas)]
    sm.start_multiprocess(multi_probas_saver, params, nb_cpus=nb_cpus)


def multi_probas_saver(args):
    so, probas, key = args
    so.save_attributes([key], [probas])


def export_sd_to_knossosdataset(sd, kd, block_edge_length=512,
                                qsub_pe=None, qsub_queue=None, nb_cpus=10,
                                n_max_co_processes=100):

    block_size = np.array([block_edge_length] * 3)

    grid_c = []
    for i_dim in range(3):
        grid_c.append(np.arange(0, kd.boundary[i_dim], block_size[i_dim]))

    bbs_block_range = sd.load_cached_data("bounding_box") / np.array(block_size)
    bbs_block_range = bbs_block_range.astype(np.int)

    kd_block_range = np.array(kd.boundary / block_size + 1, dtype=np.int)

    bbs_job_dict = defaultdict(list)

    for i_so_id, so_id in enumerate(sd.ids):
        for i_b in range(bbs_block_range[i_so_id, 0, 0],
                         bbs_block_range[i_so_id, 1, 0] + 1):
            if i_b < 0 or i_b > kd_block_range[0]:
                continue

            for j_b in range(bbs_block_range[i_so_id, 0, 1],
                             bbs_block_range[i_so_id, 1, 1] + 1):
                if j_b < 0 or j_b > kd_block_range[1]:
                    continue

                for k_b in range(bbs_block_range[i_so_id, 0, 2],
                                 bbs_block_range[i_so_id, 1, 2] + 1):
                    if k_b < 0 or k_b > kd_block_range[2]:
                        continue

                    bbs_job_dict[(i_b, j_b, k_b)].append(so_id)

    multi_params = []

    for grid_loc in bbs_job_dict.keys():
        multi_params.append([np.array(grid_loc), bbs_job_dict[grid_loc], sd.type, sd.version,
                             sd.working_dir, kd.knossos_path, block_edge_length])

    if (qsub_pe is None and qsub_queue is None) or not qu.batchjob_enabled():
        _ = sm.start_multiprocess(_export_sd_to_knossosdataset_thread,
                                  multi_params, nb_cpus=nb_cpus)

    elif qu.batchjob_enabled():
        _ = qu.QSUB_script(multi_params, "export_sd_to_knossosdataset",
                           n_max_co_processes=n_max_co_processes,
                           remove_jobfolder=True)
    else:
        raise Exception("QSUB not available")


def _export_sd_to_knossosdataset_thread(args):
    block_loc = args[0]
    so_ids = args[1]
    obj_type = args[2]
    version = args[3]
    working_dir = args[4]
    kd_path = args[5]
    block_edge_length = args[6]

    block_size = np.array([block_edge_length] * 3, dtype=np.int)

    kd = knossosdataset.KnossosDataset()
    kd.initialize_from_knossos_path(kd_path)

    sd = segmentation.SegmentationDataset(obj_type=obj_type,
                                          working_dir=working_dir,
                                          version=version)

    overlay_block = np.zeros(block_size, dtype=np.uint64)
    block_start = (block_loc * block_size).astype(np.int)

    for so_id in so_ids:
        so = sd.get_segmentation_object(so_id, False)
        vx = so.voxel_list - block_start

        vx = vx[~np.any(vx < 0, axis=1)]
        vx = vx[~np.any(vx >= block_edge_length, axis=1)]

        overlay_block[vx[:, 0], vx[:, 1], vx[:, 2]] = so_id

    kd.from_matrix_to_cubes(block_start,
                            data=overlay_block,
                            overwrite=True,
                            nb_threads=1,
                            verbose=True)


def mesh_proc_chunked(working_dir, obj_type, nb_cpus=NCORES_PER_NODE):
    """
    Caches the meshes for all SegmentationObjects within the SegmentationDataset
     with object type 'obj_type'.

    Parameters
    ----------
    working_dir : str
        Path to working directory
    obj_type : str
        Object type identifier, like 'sj', 'vc' or 'mi'
    nb_cpus : int
        Default is 20.
    """
    sd = segmentation.SegmentationDataset(obj_type, working_dir=working_dir)
    multi_params = sd.so_dir_paths
    print("Processing %d mesh dicts of %s." % (len(multi_params), obj_type))
    sm.start_multiprocess_imap(mesh_chunk, multi_params, nb_cpus=nb_cpus,
                               debug=False)
