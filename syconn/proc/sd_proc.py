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
from collections import defaultdict
from knossos_utils import knossosdataset
knossosdataset._set_noprint(True)

from ..global_params import MESH_DOWNSAMPLING,\
    MESH_CLOSING, NCORES_PER_NODE
from .. import global_params
from .image import single_conn_comp_img
from ..mp import batchjob_utils as qu
from ..mp import mp_utils as sm
from ..backend.storage import AttributeDict, VoxelStorage
from ..reps import segmentation, segmentation_helper
from ..handler import basics
from ..proc.meshes import mesh_chunk
from . import log_proc


def dataset_analysis(sd, recompute=True, n_jobs=1000, qsub_pe=None,
                     qsub_queue=None, nb_cpus=None, n_max_co_processes=None,
                     compute_meshprops=False):
    """ Analyze SegmentationDataset and extract and cache SegmentationObjects
    attributes as numpy arrays.

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
    if (qsub_pe is None and qsub_queue is None) or not qu.batchjob_enabled():
        results = sm.start_multiprocess_imap(_dataset_analysis_thread,
                                             multi_params, nb_cpus=nb_cpus)

    elif qu.batchjob_enabled():
        path_to_out = qu.QSUB_script(multi_params,
                                     "dataset_analysis",
                                     pe=qsub_pe, queue=qsub_queue,
                                     script_folder=None,
                                     n_cores=nb_cpus,
                                     n_max_co_processes=n_max_co_processes)
        out_files = glob.glob(path_to_out + "/*")
        results = []
        for out_file in out_files:
            with open(out_file, 'rb') as f:
                results.append(pkl.load(f))
    else:
        raise Exception("QSUB not available")
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
        np.save(sd.path + "/%ss.npy" % attribute, attr_dict[attribute])


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
                this_vx_dc = VoxelStorage(p + "/voxel.pkl", read_only=True)
                so_ids = list(this_vx_dc.keys())
            else:
                so_ids = list(this_attr_dc.keys())
            for so_id in so_ids:
                global_attr_dict["id"].append(so_id)
                so = segmentation.SegmentationObject(so_id, obj_type,
                                                     version, working_dir)
                so.attr_dict = this_attr_dc[so_id]
                if recompute:
                    so.load_voxels(voxel_dc=this_vx_dc)
                    so.calculate_rep_coord(voxel_dc=this_vx_dc)
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
                          qsub_pe=qsub_pe, qsub_queue=qsub_queue, nb_cpus=nb_cpus,
                          n_max_co_processes=n_max_co_processes)
        

def map_objects_to_sv(sd, obj_type, kd_path, readonly=False, n_jobs=1000,
                      qsub_pe=None, qsub_queue=None, nb_cpus=None,
                      n_max_co_processes=None):
    """
    TODO: (cython) optimization required! E.g. replace by single iteration over cell segm. and all cell organelle KDs
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
    qsub_pe : str
        qsub parallel environment
    qsub_queue : str
        qsub queue
    nb_cpus : int or None
        number of cores used for multithreading
        number of cores per worker for qsub jobs
    n_max_co_processes : int or None
        max number of workers running at the same time when using qsub
    Returns
    -------

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
    # does not work with imap in python 3.6, jobs do not return # TODO investigate; also: single processing is faster than single-node with 20 cores
    if qu.batchjob_enabled():
        path_to_out = qu.QSUB_script(multi_params, "map_objects",
                                     pe=qsub_pe, queue=qsub_queue,
                                     script_folder=None, n_cores=nb_cpus,
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
    if (qsub_pe is None and qsub_queue is None) or not qu.batchjob_enabled():
        sm.start_multiprocess_imap(_write_mapping_to_sv_thread, multi_params,
                                   nb_cpus=nb_cpus, debug=False)
    elif qu.batchjob_enabled():
        qu.QSUB_script(multi_params, "write_mapping_to_sv", pe=qsub_pe,
                       queue=qsub_queue, script_folder=None,
                       n_cores=nb_cpus, n_max_co_processes=n_max_co_processes)
    else:
        raise Exception("QSUB not available")
    log_proc.debug("map_objects_to_sv: %.1f min" % ((time.time() - start) / 60.))


def _map_objects_thread(args):
    """Worker of map_objects_to_sv"""

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
                ids, id_counts = np.unique(id_list, return_counts=True)
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
                                     n_max_co_processes=n_max_co_processes)
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
    model :
    views :
    ch :
    pred_key :
    single_cc_only :
    verbose :
    return_proba :
    nb_cpus :

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
        results = sm.start_multiprocess(_export_sd_to_knossosdataset_thread,
                                        multi_params, nb_cpus=nb_cpus)

    elif qu.batchjob_enabled():
        path_to_out = qu.QSUB_script(multi_params,
                                     "export_sd_to_knossosdataset",
                                     pe=qsub_pe, queue=qsub_queue,
                                     script_folder=None,
                                     n_max_co_processes=n_max_co_processes)
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


def extract_synapse_type(sj_sd, kd_asym_path, kd_sym_path,
                         trafo_dict_path=None, stride=10,
                         qsub_pe=None, qsub_queue=None, nb_cpus=None,
                         n_max_co_processes=None):
    """TODO: will be refactored into single method when generating syn objects
    Extract synapse type from KnossosDatasets. Stores sym.-asym. ratio in
    SJ object attribute dict.

    Parameters
    ----------
    sj_sd : SegmentationDataset
    kd_asym_path : str
    kd_sym_path : str
    trafo_dict_path : dict
    stride : int
    qsub_pe : str
    qsub_queue : str
    nb_cpus : int
    n_max_co_processes : int
    """
    assert "syn_ssv" in sj_sd.version_dict
    paths = sj_sd.so_dir_paths

    # Partitioning the work
    multi_params = []
    for path_block in [paths[i:i + stride] for i in range(0, len(paths), stride)]:
        multi_params.append([path_block, sj_sd.version, sj_sd.working_dir,
                             kd_asym_path, kd_sym_path, trafo_dict_path])

    # Running workers - Extracting mapping
    if (qsub_pe is None and qsub_queue is None) or not qu.batchjob_enabled():
        results = sm.start_multiprocess(_extract_synapse_type_thread,
                                        multi_params, nb_cpus=nb_cpus)

    elif qu.batchjob_enabled():
        path_to_out = qu.QSUB_script(multi_params,
                                     "extract_synapse_type",
                                     pe=qsub_pe, queue=qsub_queue,
                                     script_folder=None,
                                     n_cores=nb_cpus,
                                     n_max_co_processes=n_max_co_processes)
    else:
        raise Exception("QSUB not available")


def _extract_synapse_type_thread(args):
    paths = args[0]
    obj_version = args[1]
    working_dir = args[2]
    kd_asym_path = args[3]
    kd_sym_path = args[4]
    trafo_dict_path = args[5]

    if trafo_dict_path is not None:
        with open(trafo_dict_path, "rb") as f:
            trafo_dict = pkl.load(f)
    else:
        trafo_dict = None

    kd_asym = knossosdataset.KnossosDataset()
    kd_asym.initialize_from_knossos_path(kd_asym_path)
    kd_sym = knossosdataset.KnossosDataset()
    kd_sym.initialize_from_knossos_path(kd_sym_path)

    seg_dataset = segmentation.SegmentationDataset("syn_ssv",
                                                   version=obj_version,
                                                   working_dir=working_dir)
    for p in paths:
        this_attr_dc = AttributeDict(p + "/attr_dict.pkl",
                                     read_only=False, disable_locking=True)
        for so_id in this_attr_dc.keys():
            so = seg_dataset.get_segmentation_object(so_id)
            so.attr_dict = this_attr_dc[so_id]
            so.load_voxel_list()

            vxl = so.voxel_list

            if trafo_dict is not None:
                vxl -= trafo_dict[so_id]
                vxl = vxl[:, [1, 0, 2]]
            # TODO: remvoe try-except
            try:
                asym_prop = np.mean(kd_asym.from_raw_cubes_to_list(vxl))
                sym_prop = np.mean(kd_sym.from_raw_cubes_to_list(vxl))
            except:
                log_proc.error("Failed to read raw cubes during synapse type "
                               "extraction.")
                sym_prop = 0
                asym_prop = 0

            if sym_prop + asym_prop == 0:
                sym_ratio = -1
            else:
                sym_ratio = sym_prop / float(asym_prop + sym_prop)
            so.attr_dict["syn_type_sym_ratio"] = sym_ratio
            this_attr_dc[so_id] = so.attr_dict
        this_attr_dc.push()


def mesh_proc_chunked(working_dir, obj_type, nb_cpus=NCORES_PER_NODE):
    """TODO: create QSUB solution and run if cell SV meshes do not exist
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
