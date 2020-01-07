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
from typing import Optional, List, Tuple, Dict, Union
knossosdataset._set_noprint(True)

from .. import global_params
from .image import single_conn_comp_img
from ..mp import batchjob_utils as qu
from ..mp import mp_utils as sm
from ..backend.storage import AttributeDict, VoxelStorage, VoxelStorageDyn, MeshStorage
from ..reps import segmentation, segmentation_helper
from ..reps import rep_helper
from ..handler import basics
from ..proc.meshes import mesh_chunk
from . import log_proc
from ..extraction import object_extraction_wrapper as oew
from .meshes import mesh_area_calc, merge_meshes_incl_norm
from zmesh import Mesher

import sys


def dataset_analysis(sd, recompute=True, n_jobs=None, n_max_co_processes=None,
                     compute_meshprops=False):
    # TODO: refactor s.t. jobs use more than 1 CPU, currently submission times are slow
    """ Analyze SegmentationDataset and extract and cache SegmentationObjects
    attributes as numpy arrays. Will only recognize dict/storage entries of type int
    for object attribute collection.


    :param sd: SegmentationDataset
    :param recompute: bool
        whether or not to (re-)compute key information of each object
        (rep_coord, bounding_box, size)
    :param n_jobs: int
        number of jobs
    :param nb_cpus: int
        number of cores used for multithreading
        number of cores per worker for qsub jobs
    :param n_max_co_processes: int
        max number of workers running at the same time when using qsub
    :param compute_meshprops: bool
    """
    if n_jobs is None:
        n_jobs = global_params.config.ncore_total  # individual tasks are very fast
    paths = sd.so_dir_paths
    if compute_meshprops:
        if not (sd.type in global_params.config['meshes']['downsampling'] and sd.type in
                global_params.config['meshes']['closings']):
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
    #  some dictionaries -> add checks!
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
                # e.g. isinstance(np.array([100, ], dtype=np.uint)[0], int) fails
                so_ids = list(this_vx_dc.keys())
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
                              cube_of_interest_bb=None, chunk_size=None, log=None):
    """Replaces `map_objects_to_sv` and parts of `from_ids_to_objects`.

    Extracts segmentation properties for each SV in cell and subcellular segmentation.
    Requires KDs at `kd_seg_path` and `kd_organelle_paths`.

    Step 1: Extract properties and overlap voxels locally
    Step 2: Analyze resulting IDs (get result locations of each SV object)
    Step 3: Write out combined results for each SV object according to
        SegmentationDataset chunking

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
        n_chunk_jobs = global_params.config.ncore_total * 2
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
    sv_sd = segmentation.SegmentationDataset(
        working_dir=global_params.config.working_dir, obj_type="sv",
        version=0, n_folders_fs=n_folders_fs)

    all_times = []
    step_names = []

    # extract mapping
    start = time.time()
    multi_params = list(basics.chunkify_successive(
        chunk_list, np.max([len(chunk_list) // n_chunk_jobs, 1])))
    multi_params = [(chs, chunk_size, kd_seg_path, list(kd_organelle_paths.values()),
                     worker_nr, global_params.config.allow_mesh_gen_cells)
                    for chs, worker_nr in zip(multi_params, range(len(multi_params)))]

    if qu.batchjob_enabled():
        path_to_out = qu.QSUB_script(
            multi_params, "map_subcell_extract_props", n_cores=n_cores,
            n_max_co_processes=n_max_co_processes)
        out_files = glob.glob(path_to_out + "/*")
        results = []
        for out_file in out_files:
            with open(out_file, 'rb') as f:
                results.append(pkl.load(f))
    else:
        results = sm.start_multiprocess_imap(
            _map_subcell_extract_props_thread, multi_params, nb_cpus=n_max_co_processes,
            verbose=True, debug=False)
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

    for cp_dc, scp_dcs, scm_dcs, worker_nr, ref_mesh_dc in results:
        merge_prop_dicts([tot_cp, cp_dc])
        list_of_workers.append(worker_nr)
        for cell_id, chunk_ids in ref_mesh_dc['sv'].items():
            cell_mesh_workers[cell_id].append((worker_nr, chunk_ids))
        # iterate over each subcellular structure
        for ii in range(len(kd_organelle_paths)):
            organelle = global_params.config['existing_cell_organelles'][ii]
            merge_map_dicts([tot_scm[ii], scm_dcs[ii]])
            merge_prop_dicts([tot_scp[ii], scp_dcs[ii]])
            for subcell_id, chunk_ids in ref_mesh_dc[organelle].items():
                subcell_mesh_workers[ii][subcell_id].append((worker_nr, chunk_ids))
        del cp_dc, scp_dcs, scm_dcs, ref_mesh_dc  # will remove redundant data inside results

    sc_mesh_worker_dc = "{}/sc_mesh_worker_dict.pkl".format(global_params.config.temp_path)
    with open(sc_mesh_worker_dc, 'wb') as f:
        pkl.dump(subcell_mesh_workers, f, protocol=4)
    del subcell_mesh_workers

    c_mesh_worker_dc = "{}/c_mesh_worker_dict.pkl".format(global_params.config.temp_path)
    with open(c_mesh_worker_dc, 'wb') as f:
        pkl.dump(cell_mesh_workers, f, protocol=4)
    del cell_mesh_workers

    # convert mapping dicts to store ratio of number of overlapping voxels
    # TODO: size filter here or during write-out?
    prop_dict_p = "{}/sv_prop_dict.pkl".format(global_params.config.temp_path)
    with open(prop_dict_p, 'wb') as f:
        pkl.dump(tot_cp, f, protocol=4)

    # use dictionary for rep coord (just use any, they all share the same keys)
    for k in tot_cp[0]:
        curr_dir = sv_sd.so_storage_path + rep_helper.subfold_from_ix(k, n_folders_fs)
        os.makedirs(curr_dir, exist_ok=True)
    del tot_cp

    dict_paths = [prop_dict_p, ]
    for ii, k in enumerate(global_params.config['existing_cell_organelles']):
        convert_nvox2ratio_mapdict(tot_scm[ii])  # in-place conversion
        # subcell ID -> cell ID -> ratio
        mapping_dict_path = "{}/{}_mapping_dict.pkl".format(global_params.config.temp_path, k)
        with open(mapping_dict_path, 'wb') as f:
            pkl.dump(tot_scm[ii], f, protocol=4)

        # cell ID -> subcell ID -> ratio
        tot_scm_inv = invert_mdc(tot_scm[ii])
        mapping_dict_path_inv = "{}_inv.pkl".format(mapping_dict_path[:-4])
        with open(mapping_dict_path_inv, 'wb') as f:
            pkl.dump(tot_scm_inv, f, protocol=4)
        tot_scm[ii] = {}  # free space
        del tot_scm_inv

        prop_dict = tot_scp[ii]
        prop_dict_p = "{}/{}_prop_dict.pkl".format(global_params.config.temp_path, k)
        with open(prop_dict_p, 'wb') as f:
            pkl.dump(prop_dict, f, protocol=4)
        sc_sd = segmentation.SegmentationDataset(
            working_dir=global_params.config.working_dir, obj_type=k, version=0,
            n_folders_fs=n_folders_fs_sc)
        # use dictionary for rep coord (just use any, they all share the
        # same keys)
        for el in prop_dict[0]:
            curr_dir = sc_sd.so_storage_path + rep_helper.subfold_from_ix(
                el, sc_sd.n_folders_fs)
            os.makedirs(curr_dir, exist_ok=True)
        tot_scp[ii] = []  # free space

        dict_paths += [mapping_dict_path, mapping_dict_path_inv]
    all_times.append(time.time() - start)

    step_names.append("conversion of results")

    # writing cell SV properties to SD
    start = time.time()
    # create "dummy" IDs which represent each a unique storage path
    storage_location_ids = rep_helper.get_unique_subfold_ixs(n_folders_fs)
    multi_params = [(sv_id_block, n_folders_fs,
                     global_params.config.allow_mesh_gen_cells)
                    for sv_id_block in basics.chunkify(storage_location_ids,
                                                       global_params.config['nnodes_total'] * 2)]
    if not qu.batchjob_enabled():
        sm.start_multiprocess_imap(_write_props_to_sv_thread, multi_params,
                                   nb_cpus=n_max_co_processes, debug=False)
    else:
        qu.QSUB_script(multi_params, "write_props_to_sv_singlenode",
                       n_cores=global_params.config['ncores_per_node'],
                       n_max_co_processes=global_params.config['nnodes_total'],
                       remove_jobfolder=True)
    if global_params.config.use_new_meshing:
        dataset_analysis(sv_sd, recompute=False, compute_meshprops=False)
    all_times.append(time.time() - start)
    step_names.append("write cell SV dataset")

    # write to subcell. SV attribute dicts
    # create "dummy" IDs which represent each a unique storage path
    storage_location_ids = rep_helper.get_unique_subfold_ixs(n_folders_fs_sc)
    multi_params = [(sv_id_block, n_folders_fs_sc)
                    for sv_id_block in basics.chunkify(storage_location_ids,
                                                       global_params.config['nnodes_total'] * 2)]
    if not qu.batchjob_enabled():
        sm.start_multiprocess_imap(_write_props_to_sc_thread, multi_params,
                                   nb_cpus=n_max_co_processes, debug=False)
    else:
        qu.QSUB_script(multi_params, "write_props_to_sc_singlenode", script_folder=None,
                       n_cores=global_params.config['ncores_per_node'],
                       n_max_co_processes=global_params.config['nnodes_total'],
                       remove_jobfolder=True)
    step_names.append("write subcellular SV dataset")

    if global_params.config.use_new_meshing:
        for k in global_params.config['existing_cell_organelles']:
            sc_sd = segmentation.SegmentationDataset(
                working_dir=global_params.config.working_dir, obj_type=k,
                version=0, n_folders_fs=n_folders_fs_sc)
            dataset_analysis(sc_sd, recompute=False, compute_meshprops=False)
        # remove temporary folders
        for worker_nr in list_of_workers:
            p = "{}/tmp_meshes_{}/".format(global_params.config.temp_path,
                                           worker_nr)
            shutil.rmtree(os.path.dirname(p))
    # clear temporary files
    for p in dict_paths:
        os.remove(p)
    shutil.rmtree(cd_dir, ignore_errors=True)
    if qu.batchjob_enabled():  # remove job directory of `map_subcell_extract_props`
        shutil.rmtree(os.path.abspath(path_to_out + "/../"), ignore_errors=True)
    log.debug("Time overview [map_subcell_extract_props]:")
    for ii in range(len(all_times)):
        log.debug("%s: %.3fs" % (step_names[ii], all_times[ii]))
    log.debug("--------------------------")
    log.debug("Total Time: %.1f min" % (np.sum(all_times) / 60))
    log.debug("--------------------------")


def _map_subcell_extract_props_thread(args):
    from syconn.reps.find_object_properties_C import map_subcell_extract_propsC
    chunks = args[0]
    chunk_size = args[1]
    kd_cell_p = args[2]
    kd_subcell_ps = args[3]
    worker_nr = args[4]
    generate_sv_mesh = args[5]

    kd_cell = basics.kd_factory(kd_cell_p)
    cd_dir = "{}/chunkdatasets/tmp/".format(global_params.config.temp_path)
    cd = chunky.ChunkDataset()
    cd.initialize(kd_cell, kd_cell.boundary, chunk_size, cd_dir,
                  box_coords=[0, 0, 0], fit_box_size=True)
    kd_subcells = [basics.kd_factory(kd_subcell_p) for kd_subcell_p in kd_subcell_ps]
    n_subcell = len(kd_subcells)

    # cell property dicts
    cpd_lst = [{}, defaultdict(list), {}]
    # subcell. property dicts
    scpd_lst = [[{}, defaultdict(list), {}] for _ in range(n_subcell)]
    # subcell. mapping dicts
    scmd_lst = [{} for _ in range(n_subcell)]

    existing_oragnelles = global_params.config['existing_cell_organelles']

    ref_mesh_dict = {}
    ref_mesh_dict['sv'] = defaultdict(set)
    for organelle in existing_oragnelles:
        ref_mesh_dict[organelle] = defaultdict(set)

    dt_times_dc = {'find_mesh': 0, 'mesh_io': 0}

    # iterate over chunks and store information in property dicts for
    # subcellular and cellular structures
    for ch_cnt, ch_id in enumerate(chunks):
        ch = cd.chunk_dict[ch_id]
        offset, size = ch.coordinates.astype(np.int), ch.size
        cell_d = kd_cell.load_seg(size=size, offset=offset, mag=1).swapaxes(0, 2)
        # get all segmentation arrays concatenates as 4D array: [C, X, Y, Z]
        subcell_d = []
        for kd_sc, i in zip(kd_subcells, range(len(kd_subcells))):
            subc_d = kd_sc.load_seg(size=size, offset=offset, mag=1).swapaxes(0, 2)
            if global_params.config.use_new_meshing:
                organelle = existing_oragnelles[i]
                start = time.time()
                tmp_subcell_meshes = find_meshes(subc_d, offset)
                # store reference to partial results of each object
                for k in tmp_subcell_meshes.keys():
                    ref_mesh_dict[organelle][k].add(ch_id)
                dt_times_dc['find_mesh'] += time.time() - start
                start = time.time()
                p = "{}/tmp_meshes_{}/{}_{}_ch{}.pkl".format(global_params.config.temp_path,
                                                             worker_nr, organelle, worker_nr,
                                                             ch_id)
                if ch_cnt == 0:
                    os.makedirs(os.path.dirname(p), exist_ok=True)
                output_worker = open(p, 'wb')
                pkl.dump(tmp_subcell_meshes, output_worker, protocol=4)
                output_worker.close()
                dt_times_dc['mesh_io'] += time.time() - start
                del tmp_subcell_meshes
            subcell_d.append(subc_d[None, ])  # add auxiliary axis
        subcell_d = np.concatenate(subcell_d)
        cell_prop_dicts, subcell_prop_dicts, subcell_mapping_dicts = \
            map_subcell_extract_propsC(cell_d, subcell_d)
        # reorder to match [[rc, bb, size], [rc, bb, size]] for e.g. [mi, vc]
        subcell_prop_dicts = [[subcell_prop_dicts[0][ii], subcell_prop_dicts[1][ii],
                               subcell_prop_dicts[2][ii]] for ii in range(n_subcell)]
        # merge cell properties: list list of dicts
        merge_prop_dicts([cpd_lst, cell_prop_dicts], offset)

        # collect subcell properties: list of list of dicts
        # collect subcell mappings to cell SVs: list of list of
        # dicts and list of dict of dict of int
        if generate_sv_mesh and global_params.config.use_new_meshing:
            start = time.time()
            tmp_cell_mesh = find_meshes(cell_d, offset)
            # store reference to partial results of each object
            for k in tmp_cell_mesh.keys():
                ref_mesh_dict['sv'][k].add(ch_id)
            dt_times_dc['find_mesh'] += time.time() - start
            start = time.time()
            p = "{}/tmp_meshes_{}/{}_{}_ch{}.pkl".format(
                global_params.config.temp_path, worker_nr, "sv", worker_nr, ch_id)
            output_worker = open(p, 'wb')
            pkl.dump(tmp_cell_mesh, output_worker, protocol=4)
            output_worker.close()
            dt_times_dc['mesh_io'] += time.time() - start
            del tmp_cell_mesh
        for organelle, ii in zip(global_params.config['existing_cell_organelles'],
                                 range(len(global_params.config['existing_cell_organelles']))):
            merge_map_dicts([scmd_lst[ii], subcell_mapping_dicts[ii]])
            merge_prop_dicts([scpd_lst[ii], subcell_prop_dicts[ii]], offset)

        del subcell_prop_dicts
        del subcell_mapping_dicts
        del cell_prop_dicts

    if global_params.config.use_new_meshing:
        dt_str = ["{}: {:.2f}s\t".format(k, v) for k, v in dt_times_dc.items()]
        log_proc.debug('{}'.format("".join(dt_str)))
    return cpd_lst, scpd_lst, scmd_lst, worker_nr, ref_mesh_dict


def find_meshes(chunk: np.ndarray, offset: np.ndarray)\
        -> Dict[int, List[np.ndarray]]:
    """
    Find meshes within a segmented cube. The offset is given in voxels. Mesh
    vertices are scaled according to
    ``global_params.config['scaling']``.

    Args:
        chunk: Cube which is processed.
        offset: Offset of the cube in voxels.

    Returns:
        The mesh of each segmentation ID in the input `chunk`.
    """
    scaling = np.array(global_params.config['scaling'])
    mesher = Mesher(scaling[::-1])  # xyz -> zyx
    mesher.mesh(chunk)
    offset = offset * scaling
    meshes = {}
    for obj_id in mesher.ids():
        # in nm (after scaling, see top)
        tmp = mesher.get_mesh(obj_id, **global_params.config['meshes']['meshing_props'])
        # the values of simplification_factor & max_simplification_error are random

        tmp.vertices[:] = (tmp.vertices[:, ::-1] + offset)  # zyx -> xyz
        meshes[obj_id] = [tmp.faces[:, ::-1].flatten().astype(np.uint32),
                          tmp.vertices.flatten().astype(np.float32)]
        if tmp.normals is not None:
            meshes[obj_id].append(tmp.normals.flatten().astype(np.float32))
        else:
            meshes[obj_id].append(np.zeros((0, 3), dtype=np.float32))
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
        # TODO: this needs to be a parameter -> add global parameter for face type
        n_el = int((len(m_storage[obj_id][1])) / 3)
        m_storage[obj_id][0] = np.concatenate((m_storage[obj_id][0], tmp_dict[0] + n_el))
        m_storage[obj_id][1] = np.concatenate((m_storage[obj_id][1], tmp_dict[1]))
        m_storage[obj_id][2] = np.concatenate((m_storage[obj_id][2], tmp_dict[2]))


def merge_prop_dicts(prop_dicts: List[dict],
                     offset: Optional[np.ndarray] = None):
    """Merge property dicts in-place. All values will be stored in the first dict."""
    tot_rc = prop_dicts[0][0]
    tot_bb = prop_dicts[0][1]
    tot_size = prop_dicts[0][2]
    for el in prop_dicts[1:]:
        if len(el[0]) == 0:
            continue
        if offset is not None:
            # update chunk offset  # TODO: could be done at the end of the map_extract cython code
            for k in el[0]:
                el[0][k] = (el[0][k] + offset).astype(np.int32)
        tot_rc.update(el[0])  # just overwrite existing elements
        for k, v in el[1].items():
            if offset is None:
                bb = np.array(v, dtype=np.int32)
            else:
                bb = (v + offset).astype(np.int32)
            # collect all bounding boxes to enable efficient data loading
            tot_bb[k].append(bb)
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

    sc_mesh_worker_dc = "{}/sc_mesh_worker_dict.pkl".format(
        global_params.config.temp_path)
    with open(sc_mesh_worker_dc, "rb") as f:
        subcell_mesh_workers = pkl.load(f)

    # iterate over the subcell structures
    for ii, organelle in enumerate(global_params.config['existing_cell_organelles']):
        # get cached mapping and property dicts of current subcellular structure
        prop_dict_p = "{}/{}_prop_dict.pkl".format(global_params.config.temp_path,
                                                   organelle)
        with open(prop_dict_p, "rb") as f:
            prop_dict_tmp = pkl.load(f)

        # store destinations for each existing obj
        # TODO: optimize collection procedure
        dest_dc = defaultdict(list)
        for subcell_id in prop_dict_tmp[0]:
            dest_dc[rep_helper.subfold_from_ix(
                subcell_id, n_folders_fs)].append(subcell_id)

        all_obj_keys = []
        for obj_id_mod in obj_id_chs:
            all_obj_keys += dest_dc[rep_helper.subfold_from_ix(
                obj_id_mod, n_folders_fs)]

        # only keep relevant data
        prop_dict = [{}, {}, {}]
        for k in all_obj_keys:
            prop_dict[0][k] = prop_dict_tmp[0][k]
            prop_dict[1][k] = prop_dict_tmp[1][k]
            prop_dict[2][k] = prop_dict_tmp[2][k]
        del prop_dict_tmp

        mapping_dict_path = "{}/{}_mapping_dict.pkl".format(
            global_params.config.temp_path, organelle)
        with open(mapping_dict_path, "rb") as f:
            mapping_dict_tmp = pkl.load(f)

        mapping_dict = {}
        for k in all_obj_keys:
            try:
                mapping_dict[k] = mapping_dict_tmp[k]
            except KeyError:
                pass  # object was not mapped to any SV
        del mapping_dict_tmp

        # fetch all required mesh data
        if global_params.config.use_new_meshing:
            # get cached mesh dicts for segmentation object 'organelle'
            cached_mesh_dc = defaultdict(list)
            worker_ids = defaultdict(set)
            for k in all_obj_keys:
                if prop_dict[2][k] < global_params.config['meshes']['mesh_min_obj_vx']:
                    # do not load mesh-cache of small objects
                    continue
                for el in subcell_mesh_workers[ii][k]:
                    # el[1] contains the chunks
                    worker_ids[el[0]].update(el[1])
            for worker_nr, chunk_ids in worker_ids.items():
                for ch_id in chunk_ids:
                    p = "{}/tmp_meshes_{}/{}_{}_ch{}.pkl".format(
                        global_params.config.temp_path, worker_nr, organelle, worker_nr, ch_id)
                    pkl_file = open(p, "rb")
                    partial_mesh_dc = pkl.load(pkl_file)
                    pkl_file.close()
                    # only loaded keys which are part of the worker's chunk
                    for el in np.intersect1d(all_obj_keys, list(partial_mesh_dc.keys())):
                        cached_mesh_dc[el].append(partial_mesh_dc[el])

        # get SegmentationDataset of current subcell.
        sc_sd = segmentation.SegmentationDataset(
            n_folders_fs=n_folders_fs, obj_type=organelle,
            working_dir=global_params.config.working_dir, version=0)

        # iterate over the subcellular SV ID chunks
        for obj_id_mod in obj_id_chs:
            obj_keys = dest_dc[rep_helper.subfold_from_ix(obj_id_mod, n_folders_fs)]

            # get dummy segmentation object to fetch attribute
            # dictionary for this batch of object IDs
            dummy_so = sc_sd.get_segmentation_object(obj_id_mod)
            attr_p = dummy_so.attr_dict_path
            vx_p = dummy_so.voxel_path
            this_attr_dc = AttributeDict(attr_p, read_only=False,
                                         disable_locking=True)
            voxel_dc = VoxelStorageDyn(
                vx_p, voxel_mode=False, read_only=False, disable_locking=True,
                voxeldata_path=global_params.config.kd_organelle_seg_paths[organelle])

            if global_params.config.use_new_meshing:
                obj_mesh_dc = MeshStorage(dummy_so.mesh_path,
                                          disable_locking=True, read_only=False)

            for sc_id in obj_keys:
                if sc_id in mapping_dict:
                    this_attr_dc[sc_id]["mapping_ids"] = \
                        list(mapping_dict[sc_id].keys())
                    this_attr_dc[sc_id]["mapping_ratios"] = \
                        list(mapping_dict[sc_id].values())
                else:
                    this_attr_dc[sc_id]["mapping_ids"] = []
                    this_attr_dc[sc_id]["mapping_ratios"] = []
                rp = prop_dict[0][sc_id]
                bbs = np.concatenate(prop_dict[1][sc_id])
                size = prop_dict[2][sc_id]
                this_attr_dc[sc_id]["rep_coord"] = rp
                this_attr_dc[sc_id]["bounding_box"] = np.array(
                    [bbs[:, 0].min(axis=0), bbs[:, 1].max(axis=0)])
                this_attr_dc[sc_id]["size"] = size
                voxel_dc[sc_id] = bbs
                # TODO: make use of these stored properties
                #  downstream during the reduce step
                voxel_dc.increase_object_size(sc_id, size)
                voxel_dc.set_object_repcoord(sc_id, rp)
                if global_params.config.use_new_meshing:
                    try:
                        partial_meshes = cached_mesh_dc[sc_id]
                    except KeyError:  # object has size < 10
                        partial_meshes = []
                    del cached_mesh_dc[sc_id]
                    list_of_ind = []
                    list_of_ver = []
                    list_of_norm = []
                    for single_mesh in partial_meshes:
                        list_of_ind.append(single_mesh[0])
                        list_of_ver.append(single_mesh[1])
                        list_of_norm.append(single_mesh[2])
                    mesh = merge_meshes_incl_norm(list_of_ind, list_of_ver, list_of_norm)
                    obj_mesh_dc[sc_id] = mesh
                    verts = mesh[1].reshape(-1, 3)
                    if len(verts) > 0:
                        mesh_bb = [np.min(verts, axis=0), np.max(verts, axis=0)]
                        del verts
                        this_attr_dc[sc_id]["mesh_bb"] = mesh_bb
                        this_attr_dc[sc_id]["mesh_area"] = mesh_area_calc(mesh)
                    else:
                        this_attr_dc[sc_id]["mesh_bb"] = this_attr_dc[sc_id]["bounding_box"] * \
                                                         dummy_so.scaling
                        this_attr_dc[sc_id]["mesh_area"] = 0
            voxel_dc.push()
            this_attr_dc.push()
            if global_params.config.use_new_meshing:
                obj_mesh_dc.push()


def _write_props_to_sv_thread(args):
    """"""

    obj_id_chs = args[0]
    n_folders_fs = args[1]
    generate_sv_mesh = args[2]
    dt_loading_cache = time.time()
    prop_dict_p = "{}/sv_prop_dict.pkl".format(global_params.config.temp_path)
    prop_dict_tmp = basics.load_pkl2obj(prop_dict_p)

    # store destinations for each existing obj
    dest_dc = defaultdict(list)
    # use dictionary for rep coord (just use any, they all share the same keys)
    for k in prop_dict_tmp[0]:
        dest_dc[rep_helper.subfold_from_ix(k, n_folders_fs)].append(k)
    all_obj_keys = []
    for obj_id_mod in obj_id_chs:
        all_obj_keys += dest_dc[rep_helper.subfold_from_ix(obj_id_mod, n_folders_fs)]
    if len(all_obj_keys) == 0:
        return
    # keep only relevant data
    prop_dict = [{}, {}, {}]
    for k in all_obj_keys:
        prop_dict[0][k] = prop_dict_tmp[0][k]
        prop_dict[1][k] = prop_dict_tmp[1][k]
        prop_dict[2][k] = prop_dict_tmp[2][k]
    del prop_dict_tmp

    c_mesh_worker_dc = "{}/c_mesh_worker_dict.pkl".format(global_params.config.temp_path)
    with open(c_mesh_worker_dc, "rb") as f:
        cell_mesh_workers_tmp = pkl.load(f)

    cell_mesh_workers = {}
    for k in all_obj_keys:
        cell_mesh_workers[k] = cell_mesh_workers_tmp[k]
    del cell_mesh_workers_tmp

    # iterate over the subcell structures
    # get cached mapping and property dicts of current subcellular structure
    mapping_dicts = {k: {} for k in global_params.config['existing_cell_organelles']}
    for ii, k in enumerate(global_params.config['existing_cell_organelles']):
        mapping_dict_path = "{}/{}_mapping_dict_inv.pkl".format(global_params.config.temp_path, k)
        with open(mapping_dict_path, "rb") as f:
            mapping_dict_tmp = pkl.load(f)

        for ix in all_obj_keys:
            try:
                mapping_dicts[k][ix] = mapping_dict_tmp[ix]
            except KeyError:  # no object of current type mapped to cell SV
                pass
        del mapping_dict_tmp

    dt_loading_cache = time.time() - dt_loading_cache

    log_proc.debug('[SV] loaded cache dicts after {:.2f} min'.format(
        dt_loading_cache / 60))

    # fetch all required mesh data
    dt_mesh_merge_io = 0

    if global_params.config.use_new_meshing:
        # get cached mesh dicts for segmentation object k
        cached_mesh_dc = defaultdict(list)
        start = time.time()
        worker_ids = defaultdict(set)
        for k in all_obj_keys:
            # ignore mesh of small objects
            if prop_dict[2][k] < global_params.config['meshes']['mesh_min_obj_vx']:
                continue
            # el is (worker_nr (int), chunk_ids (set))
            for el in cell_mesh_workers[k]:
                worker_ids[el[0]].update(el[1])
        log_proc.debug('Loading meshes of {} SVs from {} worker '
                       'caches.'.format(len(all_obj_keys), len(worker_ids)))
        for worker_nr, chunk_ids in worker_ids.items():
            for ch_id in chunk_ids:
                p = "{}/tmp_meshes_{}/{}_{}_ch{}.pkl".format(
                    global_params.config.temp_path, worker_nr, "sv", worker_nr, ch_id)
                pkl_file = open(p, "rb")
                partial_mesh_dc = pkl.load(pkl_file)
                pkl_file.close()
                # only loaded keys which are part of the worker's chunk
                for el in np.intersect1d(all_obj_keys, list(partial_mesh_dc.keys())):
                    cached_mesh_dc[el].append(partial_mesh_dc[el])
        dt_mesh_merge_io += time.time() - start
        log_proc.debug('Loading meshes from worker caches took'
                       ' {:.2f} min'.format(dt_mesh_merge_io / 60))

    # get SegmentationDataset of cell SV
    sv_sd = segmentation.SegmentationDataset(
        n_folders_fs=n_folders_fs, obj_type="sv",
        working_dir=global_params.config.working_dir, version=0)
    # iterate over the subcellular SV ID chunks
    dt_mesh_area = 0
    dt_mesh_merge = 0  # without io

    for obj_id_mod in obj_id_chs:
        obj_keys = dest_dc[rep_helper.subfold_from_ix(obj_id_mod, n_folders_fs)]

        # get dummy segmentation object to fetch attribute dictionary for this batch of object IDs
        dummy_so = sv_sd.get_segmentation_object(obj_id_mod)
        attr_p = dummy_so.attr_dict_path
        vx_p = dummy_so.voxel_path
        this_attr_dc = AttributeDict(attr_p, read_only=False, disable_locking=True)
        voxel_dc = VoxelStorageDyn(vx_p, voxel_mode=False,
                                   voxeldata_path=global_params.config.kd_seg_path,
                                   read_only=False, disable_locking=True)
        obj_mesh_dc = MeshStorage(dummy_so.mesh_path, disable_locking=True,
                                  read_only=False)

        for sv_id in obj_keys:
            for ii, k in enumerate(global_params.config['existing_cell_organelles']):
                if sv_id not in mapping_dicts[k]:
                    # no object of this type mapped to current cell SV
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
            if generate_sv_mesh and global_params.config.use_new_meshing:
                try:
                    partial_meshes = cached_mesh_dc[sv_id]
                except KeyError:  # object has small number of voxels
                    partial_meshes = []
                del cached_mesh_dc[sv_id]
                list_of_ind = []
                list_of_ver = []
                list_of_norm = []
                for single_mesh in partial_meshes:
                    list_of_ind.append(single_mesh[0])
                    list_of_ver.append(single_mesh[1])
                    list_of_norm.append(single_mesh[2])
                start2 = time.time()
                mesh = merge_meshes_incl_norm(list_of_ind, list_of_ver, list_of_norm)
                dt_mesh_merge += time.time() - start2

                obj_mesh_dc[sv_id] = mesh
                start = time.time()
                verts = mesh[1].reshape(-1, 3)
                if len(verts) > 0:
                    mesh_bb = [np.min(verts, axis=0), np.max(verts, axis=0)]
                    del verts
                    this_attr_dc[sv_id]["mesh_bb"] = mesh_bb
                    this_attr_dc[sv_id]["mesh_area"] = mesh_area_calc(mesh)
                else:
                    this_attr_dc[sv_id]["mesh_bb"] = this_attr_dc[sv_id]["bounding_box"] * \
                                                     dummy_so.scaling
                    this_attr_dc[sv_id]["mesh_area"] = 0
                dt_mesh_area += time.time() - start
        voxel_dc.push()
        this_attr_dc.push()
        if global_params.config.use_new_meshing:
            obj_mesh_dc.push()
    if global_params.config.use_new_meshing:
        log_proc.debug('[SV] dt mesh area {:.2f}s \t dt '
                       'mesh merge {:.2f} s'.format(
            dt_mesh_area, dt_mesh_merge))


def binary_filling_cs(cs_sd, n_iterations=13, stride=1000,
                      nb_cpus=None, n_max_co_processes=None):
    paths = cs_sd.so_dir_paths

    # Partitioning the work
    multi_params = []
    for path_block in [paths[i:i + stride] for i in range(0, len(paths), stride)]:
        multi_params.append([path_block, cs_sd.version, cs_sd.working_dir,
                             n_iterations])

    # Running workers
    if not qu.batchjob_enabled():
        sm.start_multiprocess(_binary_filling_cs_thread,
                              multi_params, nb_cpus=nb_cpus)

    else:
        qu.QSUB_script(multi_params, "binary_filling_cs", n_cores=nb_cpus,
                       n_max_co_processes=n_max_co_processes,
                       remove_jobfolder=True)


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
        scaling = global_params.config['scaling']
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


def export_sd_to_knossosdataset(sd, kd, block_edge_length=512, nb_cpus=10,
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

    if not qu.batchjob_enabled():
        _ = sm.start_multiprocess(_export_sd_to_knossosdataset_thread,
                                  multi_params, nb_cpus=nb_cpus)

    else:
        _ = qu.QSUB_script(multi_params, "export_sd_to_knossosdataset",
                           n_max_co_processes=n_max_co_processes,
                           remove_jobfolder=True)


def _export_sd_to_knossosdataset_thread(args):
    block_loc = args[0]
    so_ids = args[1]
    obj_type = args[2]
    version = args[3]
    working_dir = args[4]
    kd_path = args[5]
    block_edge_length = args[6]

    block_size = np.array([block_edge_length] * 3, dtype=np.int)

    kd = basics.kd_factory(kd_path)

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


def mesh_proc_chunked(working_dir, obj_type, nb_cpus=None):
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
    if nb_cpus is None:
        nb_cpus = global_params.config['ncores_per_node']
    sd = segmentation.SegmentationDataset(obj_type, working_dir=working_dir)
    multi_params = sd.so_dir_paths
    print("Processing %d mesh dicts of %s." % (len(multi_params), obj_type))
    sm.start_multiprocess_imap(mesh_chunk, multi_params, nb_cpus=nb_cpus,
                               debug=False)
