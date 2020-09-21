# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max-Planck-Institute of Neurobiology, Munich, Germany
# Authors: Philipp Schubert

import os
import shutil
import time
from logging import Logger
from multiprocessing import Process
from typing import Optional, Callable, Tuple, Dict, Any, Union

import networkx as nx
import numpy as np

from syconn import global_params
from syconn.exec import exec_skeleton
from syconn.extraction import object_extraction_wrapper as oew
from syconn.handler.basics import chunkify, kd_factory
from syconn.handler.config import initialize_logging
from syconn.mp import batchjob_utils as qu
from syconn.mp.mp_utils import start_multiprocess_imap
from syconn.proc import sd_proc
from syconn.proc import ssd_proc
from syconn.proc.graphs import create_ccsize_dict
from syconn.reps.segmentation import SegmentationDataset
from syconn.reps.super_segmentation import SuperSegmentationDataset


def run_create_neuron_ssd(apply_ssv_size_threshold: Optional[bool] = None):
    """
    Creates a :class:`~syconn.reps.super_segmentation_dataset.SuperSegmentationDataset` with
    ``version=0`` at the currently active working directory based on the RAG
    at ``/glia/neuron_rag.bz2``. In case glia splitting is active, this will be
    the RAG after glia removal, if it was disabled it is identical to ``pruned_rag.bz2``.

    Args:
        apply_ssv_size_threshold:

    Notes:
        Requires :func:`~syconn.exec_init.init_cell_subcell_sds` and
        optionally :func:`~run_glia_splitting`.

    Returns:

    """
    log = initialize_logging('ssd_generation', global_params.config.working_dir + '/logs/',
                             overwrite=False)
    g_p = "{}/glia/neuron_rag.bz2".format(global_params.config.working_dir)

    rag_g = nx.read_edgelist(g_p, nodetype=np.uint)

    # if rag was not created by glia splitting procedure this filtering is required
    if apply_ssv_size_threshold is None:
        apply_ssv_size_threshold = not global_params.config.prior_glia_removal
    if apply_ssv_size_threshold:
        sd = SegmentationDataset("sv", working_dir=global_params.config.working_dir)

        sv_size_dict = {}
        bbs = sd.load_cached_data('bounding_box') * sd.scaling
        for ii in range(len(sd.ids)):
            sv_size_dict[sd.ids[ii]] = bbs[ii]
        ccsize_dict = create_ccsize_dict(rag_g, sv_size_dict)
        log.debug("Finished preparation of SSV size dictionary based "
                  "on bounding box diagional of corresponding SVs.")
        before_cnt = len(rag_g.nodes())
        for ix in list(rag_g.nodes()):
            if ccsize_dict[ix] < global_params.config['glia']['min_cc_size_ssv']:
                rag_g.remove_node(ix)
        log.debug("Removed %d neuron CCs because of size." %
                  (before_cnt - len(rag_g.nodes())))

    ccs = nx.connected_components(rag_g)
    cc_dict = {}
    for cc in ccs:
        cc_arr = np.array(list(cc))
        cc_dict[np.min(cc_arr)] = cc_arr

    cc_dict_inv = {}
    for ssv_id, cc in cc_dict.items():
        for sv_id in cc:
            cc_dict_inv[sv_id] = ssv_id

    log.info('Parsed RAG from {} with {} SSVs and {} SVs.'.format(
        g_p, len(cc_dict), len(cc_dict_inv)))

    ssd = SuperSegmentationDataset(working_dir=global_params.config.working_dir, version='0',
                                   ssd_type="ssv", sv_mapping=cc_dict_inv)
    # create cache-arrays for frequently used attributes
    # also executes 'ssd.save_dataset_shallow()'
    ssd.save_dataset_deep()

    max_n_jobs = global_params.config['ncores_per_node'] * 2
    # Write SSV RAGs
    multi_params = ssd.ssv_ids[np.argsort(ssd.load_cached_data('size'))[::-1]]
    # split all cells into chunks within upper half and lower half (sorted by size)
    # -> process a balanced load of large cells with the first jobs, and then the other, smaller half
    half_ix = len(multi_params) // 2
    multi_params = chunkify(multi_params[:half_ix], max_n_jobs // 2) + \
                   chunkify(multi_params[half_ix:], max_n_jobs // 2)

    multi_params = [(g_p, ssv_ids) for ssv_ids in multi_params]
    start_multiprocess_imap(_ssv_rag_writer, multi_params,
                            nb_cpus=global_params.config['ncores_per_node'])
    log.info('Finished saving individual SSV RAGs.')

    log.info('Finished SSD initialization. Starting cellular organelle mapping.')
    # map cellular organelles to SSVs
    ssd_proc.aggregate_segmentation_object_mappings(ssd, global_params.config['existing_cell_organelles'])
    ssd_proc.apply_mapping_decisions(ssd, global_params.config['existing_cell_organelles'])
    log.info('Finished mapping of cellular organelles to SSVs. Writing individual SSV graphs.')


def _ssv_rag_writer(args):
    g_p, ssv_ids = args
    ssd = SuperSegmentationDataset(working_dir=global_params.config.working_dir,
                                   version='0', ssd_type="ssv")
    rag_g = nx.read_edgelist(g_p, nodetype=np.uint)
    ccs = nx.connected_components(rag_g)
    cc_dict = {}
    for cc in ccs:
        cc_arr = np.array(list(cc))
        cc_dict[np.min(cc_arr)] = cc_arr
    for ssv in ssd.get_super_segmentation_object(ssv_ids):
        # get all nodes in CC of this SSV
        if len(cc_dict[ssv.id]) > 1:  # CCs with 1 node do not exist in the global RAG
            n_list = nx.node_connected_component(rag_g, ssv.id)
            # get SSV RAG as subgraph
            ssv_rag = nx.subgraph(rag_g, n_list)
        else:
            ssv_rag = nx.Graph()
            # ssv.id is the minimal SV ID, and therefore the only SV in this case
            ssv_rag.add_edge(ssv.id, ssv.id)
        nx.write_edgelist(ssv_rag, ssv.edgelist_path)


def sd_init(co: str, max_n_jobs: int, log: Optional[Logger] = None):
    """
    Initialize :class:`~syconn.reps.segmentation.SegmentationDataset` of given
    supervoxel type `co`.

    Args:
        co: Cellular organelle identifier (e.g. 'mi', 'vc', ...).
        max_n_jobs: Number of parallel jobs.
        log: Logger.
    """
    sd_seg = SegmentationDataset(obj_type=co, working_dir=global_params.config.working_dir,
                                 version="0")
    multi_params = chunkify(sd_seg.so_dir_paths, max_n_jobs)
    so_kwargs = dict(working_dir=global_params.config.working_dir, obj_type=co)
    multi_params = [[par, so_kwargs] for par in multi_params]

    if not global_params.config.use_new_meshing and \
            (co != "sv" or (co == "sv" and global_params.config.allow_mesh_gen_cells)):
        _ = qu.batchjob_script(
            multi_params, 'mesh_caching', suffix=co, remove_jobfolder=False, log=log)

    # write mesh properties to attribute dictionaries if old meshing is active
    if not global_params.config.use_new_meshing:
        sd_proc.dataset_analysis(sd_seg, recompute=False, compute_meshprops=True)


def kd_init(co, chunk_size, transf_func_kd_overlay: Optional[Callable],
            load_cellorganelles_from_kd_overlaycubes: bool,
            cube_of_interest_bb: Tuple[np.ndarray],
            log: Logger):
    """
    Replaced by a single call of :func:`~generate_subcell_kd_from_proba`.

    Initializes a per-object segmentation KnossosDataset for the given supervoxel type
    `co` based on an initial prediction which location has to be defined in the config.yml file
    for the `co` object, e.g. ``'kd_mi'`` for ``co='mi'``
    (see :func:`~syconn.handler.config.generate_default_conf`). Results will be stored as a
    KnossosDataset at `"{}/knossosdatasets/{}_seg/".format(global_params.config.working_dir, co)`.
    Appropriate parameters have to be set inside the config.yml file, see
    :func:`~syconn.extraction.object_extraction_wrapper.generate_subcell_kd_from_proba`
    or :func:`~syconn.handler.config.generate_default_conf` for more details.

    Examples:
        Was used to process sub-cellular structures independently:

                ps = [Process(target=kd_init, args=[co, chunk_size, transf_func_kd_overlay,
                    load_cellorganelles_from_kd_overlaycubes, cube_of_interest_bb, log])
                    for co in global_params.config['existing_cell_organelles']]
                for p in ps:
                    p.start()
                    time.sleep(5)
                for p in ps:
                    p.join()
                    if p.exitcode != 0:
                        raise Exception(f'Worker {p.name} stopped unexpectedly with exit code {p.exitcode}.')

    Args:
        co: Type of cell organelle supervoxels, e.g 'mi' for mitochondria or 'vc' for
            vesicle clouds.
        chunk_size: Size of the cube which are processed by each worker.
        transf_func_kd_overlay: Transformation applied on the prob. map or segmentation
            data.
        load_cellorganelles_from_kd_overlaycubes:
        cube_of_interest_bb: Bounding of the (sub-) volume of the dataset
            which is processed.
        log: Logger.
    """
    oew.generate_subcell_kd_from_proba(
        co, chunk_size=chunk_size, transf_func_kd_overlay=transf_func_kd_overlay,
        load_cellorganelles_from_kd_overlaycubes=load_cellorganelles_from_kd_overlaycubes,
        cube_of_interest_bb=cube_of_interest_bb, log=log)


def init_cell_subcell_sds(chunk_size: Optional[Tuple[int, int, int]] = None,
                          n_folders_fs: int = 10000, n_folders_fs_sc: int = 10000,
                          max_n_jobs: Optional[int] = None,
                          load_cellorganelles_from_kd_overlaycubes: bool = False,
                          transf_func_kd_overlay: Optional[Dict[Any, Callable]] = None,
                          cube_of_interest_bb: Optional[Union[tuple, np.ndarray]] = None,
                          overwrite=False):
    """
    Todo:
        * Don't extract sj objects and replace their use-cases with syn objects (?).

    Args:
        chunk_size: Size of the cube which are processed by each worker.
        n_folders_fs: Number of folders used to create the folder structure in
            the resulting :class:`~syconn.reps.segmentation.SegmentationDataset`
            for the cell supervoxels (``version='sv'``).
        n_folders_fs_sc: Number of folders used to create the folder structure in
            the resulting :class:`~syconn.reps.segmentation.SegmentationDataset`
            for the cell organelle supervxeols (e.g. ``version='mi'``).
        max_n_jobs: Number of parallel jobs.
        load_cellorganelles_from_kd_overlaycubes:
        transf_func_kd_overlay: Transformation applied on the prob. map or segmentation
            data.
        cube_of_interest_bb: Bounding of the (sub-) volume of the dataset
            which is processed (minimum and maximum coordinates in mag1 voxels,
            XYZ).
        overwrite: If True, will overwrite existing data.
    """
    log = initialize_logging('sd_generation', global_params.config.working_dir +
                             '/logs/', overwrite=True)
    if transf_func_kd_overlay is None:
        transf_func_kd_overlay = {k: None for k in global_params.config['existing_cell_organelles']}
    if chunk_size is None:
        chunk_size = [512, 512, 512]
    chunk_size_kdinit = chunk_size
    if max_n_jobs is None:
        max_n_jobs = global_params.config.ncore_total * 4
        # loading cached data or adapt number of jobs/cache size dynamically,
        # dependent on the dataset
    kd = kd_factory(global_params.config.kd_seg_path)
    if cube_of_interest_bb is None:
        cube_of_interest_bb = [np.zeros(3, dtype=np.int), kd.boundary]

    log.info('Converting the predictions of the following cellular organelles to'
             ' KnossosDatasets: {}.'.format(global_params.config['existing_cell_organelles']))
    start = time.time()
    oew.generate_subcell_kd_from_proba(
        global_params.config['existing_cell_organelles'],
        chunk_size=chunk_size_kdinit, transf_func_kd_overlay=transf_func_kd_overlay,
        load_cellorganelles_from_kd_overlaycubes=load_cellorganelles_from_kd_overlaycubes,
        cube_of_interest_bb=cube_of_interest_bb, log=log, n_chunk_jobs=max_n_jobs,
        overwrite=overwrite)
    log.info('Finished KD generation after {:.0f}s.'.format(time.time() - start))

    log.info('Generating SegmentationDatasets for subcellular structures {} and'
             ' cell supervoxels.'.format(global_params.config['existing_cell_organelles']))
    start = time.time()
    sd_proc.map_subcell_extract_props(
        global_params.config.kd_seg_path, global_params.config.kd_organelle_seg_paths,
        n_folders_fs=n_folders_fs, n_folders_fs_sc=n_folders_fs_sc, n_chunk_jobs=max_n_jobs,
        cube_of_interest_bb=cube_of_interest_bb, chunk_size=chunk_size, log=log,
        overwrite=overwrite)
    log.info('Finished extraction and mapping after {:.2f}s.'
             ''.format(time.time() - start))

    log.info('Caching properties of subcellular structures {} and cell'
             ' supervoxels'.format(global_params.config['existing_cell_organelles']))
    start = time.time()
    ps = [Process(target=sd_init, args=(co, max_n_jobs, log))
          for co in ["sv"] + global_params.config['existing_cell_organelles']]
    for p in ps:
        p.start()
        time.sleep(2)
    for p in ps:
        p.join()
        if p.exitcode != 0:
            raise Exception(f'Worker {p.name} stopped unexpectedly with exit '
                            f'code {p.exitcode}.')
        p.close()
    log.info('Finished SD caching after {:.2f}s.'
             ''.format(time.time() - start))


def run_create_rag():
    """
    If ``global_params.config.prior_glia_removal==True``:
        stores pruned RAG at ``global_params.config.pruned_rag_path``, required for all glia
        removal steps. :func:`~syconn.exec.exec_inference.run_glia_splitting`
        will finally store the ``neuron_rag.bz2`` at the currently active working directory.
    else:
        stores pruned RAG at ``global_params.config.working_dir + /glia/neuron_rag.bz2``,
        required by :func:`~syconn.exec.exec_init.run_create_neuron_ssd`.
    """
    log = initialize_logging('sd_generation', global_params.config.working_dir +
                             '/logs/', overwrite=False)
    # Crop RAG according to cell SVs found during SD generation and apply size threshold
    G = nx.read_edgelist(global_params.config.init_rag_path, nodetype=np.uint)
    if 0 in G.nodes():
        G.remove_node(0)
        log.warning('Found background node 0 in original graph. Removing.')
    all_sv_ids_in_rag = np.array(list(G.nodes()), dtype=np.uint)
    log.info("Found {} SVs in initial RAG.".format(len(all_sv_ids_in_rag)))

    # add single SV connected components to initial graph
    sd = SegmentationDataset(obj_type='sv', working_dir=global_params.config.working_dir, cache_properties=['size'])
    diff = np.array(list(set(sd.ids).difference(set(all_sv_ids_in_rag))))
    log.info('Found {} single-element connected component SVs which were missing'
             ' in initial RAG.'.format(len(diff)))

    for ix in diff:
        G.add_edge(ix, ix)

    log.debug("Found {} SVs in initial RAG after adding size-one connected "
              "components.".format(G.number_of_nodes()))

    # remove small connected components
    sv_size_dict = {}
    bbs = sd.load_cached_data('bounding_box') * sd.scaling
    for ii in range(len(sd.ids)):
        sv_size_dict[sd.ids[ii]] = bbs[ii]
    ccsize_dict = create_ccsize_dict(G, sv_size_dict)
    log.debug("Finished preparation of SSV size dictionary based "
              "on bounding box diagonal of corresponding SVs.")
    before_cnt = len(G.nodes())
    for ix in list(G.nodes()):
        if ccsize_dict[ix] < global_params.config['glia']['min_cc_size_ssv']:
            G.remove_node(ix)
    cc_gs = list(nx.connected_component_subgraphs(G))
    total_size = 0
    for n in G.nodes():
        total_size += sd.get_segmentation_object(n).size
    total_size_cmm = np.prod(sd.scaling) * total_size / 1e18
    log.info("Removed {} SVs from RAG because of size. Final RAG contains {} SVs in {} CCs ({} mm^3; {} Gvx).".format(
        before_cnt - G.number_of_nodes(), G.number_of_nodes(), len(cc_gs), total_size_cmm, total_size / 1e9))
    nx.write_edgelist(G, global_params.config.pruned_rag_path)

    if not global_params.config.prior_glia_removal:
        os.makedirs(global_params.config.working_dir + '/glia/', exist_ok=True)
        shutil.copy(global_params.config.pruned_rag_path, global_params.config.working_dir
                    + '/glia/neuron_rag.bz2')
