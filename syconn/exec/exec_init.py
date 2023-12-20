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
import tqdm

from syconn import global_params
from syconn.extraction import object_extraction_wrapper as oew
from syconn.handler.basics import chunkify, kd_factory
from syconn.handler.config import initialize_logging
from syconn.mp import batchjob_utils as qu
from syconn.mp.mp_utils import start_multiprocess_imap
from syconn.proc import sd_proc
from syconn.proc import ssd_proc
from syconn.proc.graphs import create_ccsize_dict
from syconn.reps.segmentation import SegmentationDataset
from syconn.reps.super_segmentation import SuperSegmentationDataset, SuperSegmentationObject


def run_create_neuron_ssd(apply_ssv_size_threshold: bool = False, ncores_per_job: int = 1, overwrite: bool = False):
    """
    This function creates a SuperSegmentationDataset with version 0 at the currently active working directory based on
    the SV graph at the neuron_svgraph_path. If astrocyte splitting is active, this will be the SV graph after
    astrocyte removal, if it was disabled it is identical to pruned_svgraph.bz2.
    
    Args:
        apply_ssv_size_threshold (bool): If True, applies a filter with minimum bounding box diagonal. This is usually
            not needed as the filter is applied either in run_create_rag (prior_astrocyte_removal=False) or during the
            astrocyte separation.
        ncores_per_job (int): Number of cores per worker for SuperSegmentationDataset.save_dataset_deep.
        overwrite (bool): If True, overwrites existing data.
    
    Notes:
        * This is a memory intensive step, consider increasing `ncores_per_job`.
        * Requires init_cell_subcell_sds and optionally run_astrocyte_splitting.
        * Networkx requires a lot of memory for >1e9 edges, graph_tool and igraph are not usable for this either.
          Currently the workaround is to not use the graph information when storing the cell SV graph, but only the
          connected component as a graph with N-1 edges (N nodes). [TODO]
    """
    working_dir = global_params.config.working_dir
    log = initialize_logging('ssd_generation', working_dir + '/logs/', overwrite=False)
    cc_dict = {}
    if apply_ssv_size_threshold:
        g_p = global_params.config.neuron_svgraph_path
        sv_g = nx.read_edgelist(g_p, nodetype=np.uint64)
        sd = SegmentationDataset("sv", working_dir=working_dir)

        sv_size_dict = {}
        bbs = sd.load_numpy_data('bounding_box') * sd.scaling
        for ii in range(len(sd.ids)):
            sv_size_dict[sd.ids[ii]] = bbs[ii]
        ccsize_dict = create_ccsize_dict(sv_g, sv_size_dict)
        log.info("Finished preparation of SSV size dictionary based "
                 "on bounding box diagional of corresponding SVs.")
        before_cnt = len(sv_g.nodes())
        for ix in list(sv_g.nodes()):
            if ccsize_dict[ix] < global_params.config['min_cc_size_ssv']:
                sv_g.remove_node(ix)
        log.info("Removed %d neuron CCs because of size." % (before_cnt - len(sv_g.nodes())))
        for cc in nx.connected_components(sv_g):
            cc_arr = np.array(list(cc), dtype=np.uint64)
            cc_dict[np.min(cc_arr)] = cc_arr
    else:
        g_p = global_params.config.neuron_svagg_list_path
        with open(g_p, 'r') as f:
            for line in f:
                cc = [int(el) for el in line.split(',')]
                cc = np.array(cc, dtype=np.uint64)
                cc_dict[np.min(cc)] = cc

    cc_dict_inv = {}
    for ssv_id, cc in cc_dict.items():
        for sv_id in cc:
            cc_dict_inv[sv_id] = ssv_id

    log.info('Parsed RAG from {} with {} SSVs and {} SVs.'.format(g_p, len(cc_dict), len(cc_dict_inv)))
    ssd = SuperSegmentationDataset(working_dir=working_dir, version='0',
                                   ssd_type="ssv", sv_mapping=cc_dict_inv, create=True,
                                   overwrite=overwrite)
    # create cache-arrays for frequently used attributes
    # also executes 'ssd.save_dataset_shallow()' and populates sv_ids attribute of all SSVs
    ssd.save_dataset_deep(nb_cpus=ncores_per_job)

    max_n_jobs = global_params.config['ncores_per_node'] * 3
    # Write SSV RAGs
    multi_params = ssd.ssv_ids[np.argsort(ssd.load_numpy_data('size'))[::-1]]
    # split all cells into chunks within upper half and lower half (sorted by size)
    # -> process a balanced load of large cells with the first jobs, and then the other, smaller half
    half_ix = len(multi_params) // 2
    njobs_per_half = max(max_n_jobs // 2, 1)
    multi_params = chunkify(multi_params[:half_ix], njobs_per_half) + \
                   chunkify(multi_params[half_ix:], njobs_per_half)

    multi_params = [(ssv_ids, [cc_dict[ssv_id] for ssv_id in ssv_ids]) for ssv_ids in multi_params]
    start_multiprocess_imap(_ssv_svgraph_writer, multi_params, debug=False,
                            nb_cpus=global_params.config['ncores_per_node'])
    log.info('Finished saving cell SV graphs.')

    log.info('Finished SSD initialization. Starting cellular organelle mapping.')
    # map cellular organelles to SSVs
    ssd_proc.aggregate_segmentation_object_mappings(ssd, global_params.config['process_cell_organelles'],
                                                    nb_cpus=ncores_per_job)
    ssd_proc.apply_mapping_decisions(ssd, global_params.config['process_cell_organelles'],
                                     nb_cpus=ncores_per_job)
    log.info('Finished mapping of cellular organelles to SSVs.')


def _ssv_svgraph_writer(args):
    """
    This function writes the SV graph of a SuperSegmentationObject (SSO) to a file. The SV graph is a "dummy" graph
    created from the list of supervoxels (SVs) belonging to the SSO. If the SSO contains only one SV, an edge is added
    between the single SV to itself to form a valid graph. The function also checks if the SV graph is connected and
    issues a warning if it is not. The SV graph is then written to the edgelist_path of the SSO.
    
    Args:
        args (tuple): A tuple containing a list of SSO IDs and a list of lists of SVs belonging to each SSO.
    """
    ssv_ids, sv_lists = args
    config = global_params.config
    for ssv_id, sv_list in zip(ssv_ids, sv_lists):
        ssv = SuperSegmentationObject(ssv_id, config=config)
        # create "dummy" graph
        if len(sv_list) == 1:
            sv_list = [sv_list[0], sv_list[0]]  # this will add an edge between the single supervoxel..
        sv_graph = nx.from_edgelist([(sv_list[ii], sv_list[ii+1]) for ii in range(len(sv_list) - 1)])
        if not nx.is_connected(sv_graph):
            print(f'WARNING: SV graph of SSO with ID={ssv_id} is not connected.')
        nx.write_edgelist(sv_graph, ssv.edgelist_path)


def sd_init(co: str, max_n_jobs: int, log: Optional[Logger] = None):
    """
    This function initializes a SegmentationDataset of a given supervoxel type.
    
    Args:
        co (str): Cellular organelle identifier (e.g. 'mi', 'vc', ...).
        max_n_jobs (int): Number of parallel jobs.
        log (Optional[Logger]): Logger object for logging progress and debug 
        information.
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
    This function initializes a per-object segmentation KnossosDataset for the given 
    supervoxel type based on an initial prediction. The location of the prediction has 
    to be defined in the config.yml file for the object. The results will be stored as 
    a KnossosDataset at "{}/knossosdatasets/{}_seg/".format(global_params.config.working_dir, co).
    
    Examples:
        Was used to process sub-cellular structures independently:
    
                ps = [Process(target=kd_init, args=[co, chunk_size, transf_func_kd_overlay,
                    load_cellorganelles_from_kd_overlaycubes, cube_of_interest_bb, log])
                    for co in global_params.config['process_cell_organelles']]
                for p in ps:
                    p.start()
                    time.sleep(5)
                for p in ps:
                    p.join()
                    if p.exitcode != 0:
                        raise Exception(f'Worker {p.name} stopped unexpectedly with exit code {p.exitcode}.')
    
    Args:
        co: Type of cell organelle supervoxels, e.g 'mi' for mitochondria or 'vc' for vesicle clouds.
        chunk_size: Size of the cube which are processed by each worker.
        transf_func_kd_overlay: Transformation applied on the prob. map or segmentation data.
        load_cellorganelles_from_kd_overlaycubes: If True, loads cell organelles from kd overlay cubes.
        cube_of_interest_bb: Bounding of the (sub-) volume of the dataset which is processed.
        log: Logger object for logging progress and debug information.
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
    Converts a binary class segmentation mask of sub-cellular structure predictions into an instance segmentation
    using connected components / watershed. Then, it extracts the properties of sub-cellular structures (voxel count,
    coordinate, bounding box, mesh, etc.) and their associations with cell fragments by calculating the overlap
    between every sub-cellular structure and cell fragment instance.
    
    Args:
        chunk_size (Optional[Tuple[int, int, int]]): Size of the cube which are processed by each worker.
        n_folders_fs (int): Number of folders used to create the folder structure in the resulting 
            SegmentationDataset for the cell supervoxels.
        n_folders_fs_sc (int): Number of folders used to create the folder structure in the resulting 
            SegmentationDataset for the cell organelle supervoxels.
        max_n_jobs (Optional[int]): Number of parallel jobs.
        load_cellorganelles_from_kd_overlaycubes (bool): If True, cell organelles will be loaded from 
            KnossosDataset overlay cubes.
        transf_func_kd_overlay (Optional[Dict[Any, Callable]]): Transformation function applied on the probability 
            map or segmentation data.
        cube_of_interest_bb (Optional[Union[tuple, np.ndarray]]): Bounding box of the (sub-) volume of the dataset 
            which is processed (minimum and maximum coordinates in mag1 voxels, XYZ).
        overwrite (bool): If True, will overwrite existing data.
    """
    log = initialize_logging('sd_generation', global_params.config.working_dir +
                             '/logs/', overwrite=True)
    if transf_func_kd_overlay is None:
        transf_func_kd_overlay = {k: None for k in global_params.config['process_cell_organelles']}
    if chunk_size is None:
        chunk_size = [512, 512, 512]
    chunk_size_kdinit = chunk_size
    if max_n_jobs is None:
        max_n_jobs = global_params.config.ncore_total * 2
        # loading cached data or adapt number of jobs/cache size dynamically,
        # dependent on the dataset
    kd = kd_factory(global_params.config.kd_seg_path)
    if cube_of_interest_bb is None:
        cube_of_interest_bb = [np.zeros(3, dtype=np.int32), kd.boundary]

    log.info('Converting the predictions of the following cellular organelles to'
             ' KnossosDatasets: {}.'.format(global_params.config['process_cell_organelles']))
    start = time.time()
    oew.generate_subcell_kd_from_proba(
        global_params.config['process_cell_organelles'],
        chunk_size=chunk_size_kdinit, transf_func_kd_overlay=transf_func_kd_overlay,
        load_cellorganelles_from_kd_overlaycubes=load_cellorganelles_from_kd_overlaycubes,
        cube_of_interest_bb=cube_of_interest_bb, log=log, n_chunk_jobs=max_n_jobs,
        overwrite=overwrite)
    log.info('Finished KD generation after {:.0f}s.'.format(time.time() - start))

    log.info('Generating SegmentationDatasets for subcellular structures {} and'
             ' cell supervoxels.'.format(global_params.config['process_cell_organelles']))
    start = time.time()
    sd_proc.map_subcell_extract_props(
        global_params.config.kd_seg_path, global_params.config.kd_organelle_seg_paths,
        n_folders_fs=n_folders_fs, n_folders_fs_sc=n_folders_fs_sc, n_chunk_jobs=max_n_jobs,
        cube_of_interest_bb=cube_of_interest_bb, chunk_size=chunk_size, log=log,
        overwrite=overwrite)
    log.info('Finished extraction and mapping after {:.2f}s.'
             ''.format(time.time() - start))

    log.info('Caching properties of subcellular structures {} and cell'
             ' supervoxels'.format(global_params.config['process_cell_organelles']))
    start = time.time()
    ps = [Process(target=sd_init, args=(co, max_n_jobs, log))
          for co in ["sv"] + global_params.config['process_cell_organelles']]
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


def run_create_rag(graph_node_dtype=None):
    # TODO: use BinarySearchStore
    """
    Stores pruned Region Adjacency Graph (RAG) or pruned supervoxel (SV) graph based on the 
    configuration. If 'prior_astrocyte_removal' is set to True in the configuration, it stores 
    the pruned RAG at `global_params.config.pruned_svgraph_path`, required for all glia removal 
    steps. :func:`~syconn.exec.exec_inference.run_astrocyte_splitting` will finally store the 
    neuron SV graph. Otherwise, it stores the pruned SV graph at 
    :attr:`~syconn.handler.config.DynConfig.pruned_svgraph_path`, required by 
    :func:`~syconn.exec.exec_init.run_create_neuron_ssd`.
    
    Args:
        graph_node_dtype (Optional): Data type of the graph nodes. Defaults to np.uint64.
    """
    log = initialize_logging('sd_generation', global_params.config.working_dir +
                             '/logs/', overwrite=False)
    if graph_node_dtype is None:
        graph_node_dtype = np.uint64
    # Crop RAG according to cell SVs found during SD generation and apply size threshold
    G = nx.read_edgelist(global_params.config.init_svgraph_path, nodetype=graph_node_dtype)
    if 0 in G.nodes():
        G.remove_node(0)
        log.warning('Found background node 0 in original graph. Removing.')
    all_sv_ids_in_rag = np.array(list(G.nodes()), dtype=np.uint64)
    log.info("Found {} SVs in initial RAG.".format(len(all_sv_ids_in_rag)))

    # add single SV connected components to initial graph
    sd = SegmentationDataset(obj_type='sv', working_dir=global_params.config.working_dir, cache_properties=['size'])
    diff = np.setdiff1d(sd.ids, all_sv_ids_in_rag)
    log.info(f'Found {len(diff)} single-element connected component SVs which were missing in initial RAG.')
    for ix in diff:
        G.add_edge(ix, ix)

    log.debug("Found {} SVs in initial RAG after adding size-one connected "
              "components.".format(G.number_of_nodes()))

    # remove small connected components
    sv_size_dict = {}
    bbs = sd.load_numpy_data('bounding_box') * sd.scaling
    for ii in range(len(sd.ids)):
        sv_size_dict[sd.ids[ii]] = bbs[ii]
    try:
        ccsize_dict = create_ccsize_dict(G, sv_size_dict)
    except ValueError as e:
        raise ValueError from e
    log.debug("Finished preparation of SSV size dictionary based "
              "on bounding box diagonal of corresponding SVs.")
    before_cnt = G.number_of_nodes()
    # explicit copy needed, as G is modified in the  loop
    for ix in tqdm.tqdm(list(G.nodes()), total=before_cnt, desc='CC size filter'):
        if ccsize_dict[ix] <= global_params.config['min_cc_size_ssv']:
            G.remove_node(ix)
    # TODO: check if this loop (despite cache_properties=['size'], see above) is limiting speed
    total_size = 0
    for n in tqdm.tqdm(G.nodes(), total=G.number_of_nodes(), desc='Total size'):
        total_size += sd.get_segmentation_object(n).size
    total_size_cmm = np.prod(sd.scaling) * total_size / 1e18
    log.info(f"Removed {before_cnt - G.number_of_nodes()} SVs from RAG because of size (bounding box diagonal <= "
             f"{global_params.config['min_cc_size_ssv']} nm). Final RAG contains {G.number_of_nodes()} SVs in "
             f"{nx.number_connected_components(G)} CCs ({total_size_cmm} mm^3; {total_size / 1e9} Gvx).")
    nx.write_edgelist(G, global_params.config.pruned_svgraph_path)
    with open(global_params.config.pruned_svagg_list_path, 'w') as f:
        for cc in nx.connected_components(G):
            f.write(','.join([str(el) for el in cc]) + '\n')
    log.debug("SV agglomerations have been written to file.")
    if not global_params.config.prior_astrocyte_removal:
        os.makedirs(global_params.config.working_dir + '/glia/', exist_ok=True)
        shutil.copy(global_params.config.pruned_svgraph_path, global_params.config.neuron_svgraph_path)
        shutil.copy(global_params.config.pruned_svagg_list_path, global_params.config.neuron_svagg_list_path)
