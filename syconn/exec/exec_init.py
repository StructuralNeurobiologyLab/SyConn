# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max-Planck-Institute of Neurobiology, Munich, Germany
# Authors: Philipp Schubert, Joergen Kornfeld

from knossos_utils import knossosdataset
import time
from logging import Logger
import os
from multiprocessing import Process
import shutil
import networkx as nx
import numpy as np
from typing import Optional, Callable, Tuple, Dict, Any
from syconn import global_params
from syconn.extraction import object_extraction_wrapper as oew
from syconn.proc import sd_proc
from syconn.reps.segmentation import SegmentationDataset
from syconn.handler.config import initialize_logging
from syconn.mp import batchjob_utils as qu
from syconn.proc.graphs import create_ccsize_dict
from syconn.handler.basics import chunkify, kd_factory
knossosdataset._set_noprint(True)


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

    if not global_params.config.use_new_meshing and (co != "sv" or (co == "sv" and
            global_params.config.allow_mesh_gen_cells)):
        _ = qu.QSUB_script(multi_params, "mesh_caching", suffix=co, remove_jobfolder=False,
                           n_max_co_processes=global_params.config.ncore_total, log=log)

    if co == "sv":
        _ = qu.QSUB_script(multi_params, "sample_location_caching",
                           n_max_co_processes=global_params.config.ncore_total,
                           suffix=co, remove_jobfolder=True, log=log)

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
    for the `co` object, e.g. `kd_mi` for `co='mi'`
    (see :func:`~syconn.handler.config.generate_default_conf`). Results will be stored as a
    KnossosDataset at `"{}/knossosdatasets/{}_seg/".format(global_params.config.working_dir, co)`.
    Appropriate parameters have to be set inside the config.yml file, see
    :func:`~syconn.extraction.object_extraction_wrapper.generate_subcell_kd_from_proba`
    or :func:`~syconn.handler.config.generate_default_conf` for more details.

    Examples:
        Was used to process sub-cellular structures independently:

                ps = [Process(target=kd_init, args=[co, chunk_size, transf_func_kd_overlay,
                      load_cellorganelles_from_kd_overlaycubes,
                      cube_of_interest_bb, log])
                    for co in global_params.config['existing_cell_organelles']]
                for p in ps:
                    p.start()
                    time.sleep(5)
                for p in ps:
                    p.join()
                    if p.exitcode != 0:
                        raise Exception(f'Worker {p.name} stopped unexpectedly with exit '
                                        f'code {p.exitcode}.')

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
                          cube_of_interest_bb: Optional[Tuple[np.ndarray]] = None,
                          n_cores: int = 1):
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
        n_cores: Cores used within :func:`~map_subcell_extract_props`.
    """
    log = initialize_logging('create_sds', global_params.config.working_dir +
                             '/logs/', overwrite=True)
    if transf_func_kd_overlay is None:
        transf_func_kd_overlay = {k: None for k in global_params.config['existing_cell_organelles']}
    if chunk_size is None:
        chunk_size_kdinit = [1024, 1024, 512]
        chunk_size = [512, 512, 512]
    else:
        chunk_size_kdinit = chunk_size
    if max_n_jobs is None:
        max_n_jobs = global_params.config.ncore_total * 4
        # loading cached data or adapt number of jobs/cache size dynamically,
        # dependent on the dataset
    kd = kd_factory(global_params.config.kd_seg_path)
    if cube_of_interest_bb is None:
        cube_of_interest_bb = [np.zeros(3, dtype=np.int), kd.boundary]

    log.info('Converting predictions of cellular organelles to KnossosDatasets for every'
             'type available: {}.'.format(global_params.config['existing_cell_organelles']))
    start = time.time()
    # TODO: process all subcellular structures at the same time if they are stored in the same KD
    # oew.generate_subcell_kd_from_proba(
    #     global_params.config['existing_cell_organelles'],
    #     chunk_size=chunk_size_kdinit, transf_func_kd_overlay=transf_func_kd_overlay,
    #     load_cellorganelles_from_kd_overlaycubes=load_cellorganelles_from_kd_overlaycubes,
    #     cube_of_interest_bb=cube_of_interest_bb, log=log, n_chunk_jobs=max_n_jobs,
    #     n_cores=n_cores)
    log.info('Finished KD generation after {:.0f}s.'.format(time.time() - start))

    log.info('Generating SegmentationDatasets for subcellular structures {} and'
             ' cell supervoxels.'.format(global_params.config['existing_cell_organelles']))
    start = time.time()
    sd_proc.map_subcell_extract_props(
        global_params.config.kd_seg_path, global_params.config.kd_organelle_seg_paths,
        n_folders_fs=n_folders_fs, n_folders_fs_sc=n_folders_fs_sc, n_chunk_jobs=max_n_jobs,
        cube_of_interest_bb=cube_of_interest_bb, chunk_size=chunk_size, log=log,
        n_cores=n_cores)
    log.info('Finished extraction and mapping after {:.2f}s.'
             ''.format(time.time() - start))

    log.info('Caching properties of subcellular structures {} and cell'
             ' supervoxels'.format(global_params.config['existing_cell_organelles']))
    start = time.time()
    ps = [Process(target=sd_init, args=[co, max_n_jobs, log])
          for co in ["sv"] + global_params.config['existing_cell_organelles']]
    for p in ps:
        p.start()
        time.sleep(5)
    for p in ps:
        p.join()
        if p.exitcode != 0:
            raise Exception(f'Worker {p.name} stopped unexpectedly with exit '
                            f'code {p.exitcode}.')
    log.info('Finished SD caching after {:.2f}s.'
             ''.format(time.time() - start))


def run_create_rag():
    """
    If ``global_params.config.prior_glia_removal==True``:
        stores pruned RAG at ``global_params.config.pruned_rag_path``, required for all glia
        removal steps. :func:`~syconn.exec.exec_multiview.run_glia_splitting`
        will finally store the ``neuron_rag.bz2`` at the currently active working directory.
    else:
        stores pruned RAG at ``global_params.config.working_dir + /glia/neuron_rag.bz2``,
        required by :func:`~syconn.exec.exec_multiview.run_create_neuron_ssd`.
    """
    log = initialize_logging('create_rag', global_params.config.working_dir +
                             '/logs/', overwrite=True)
    # Crop RAG according to cell SVs found during SD generation and apply size threshold
    G = nx.read_edgelist(global_params.config.init_rag_path, nodetype=np.uint)
    if 0 in G.nodes():
        G.remove_node(0)
        log.warning('Found background node 0 in original graph. Removing.')
    all_sv_ids_in_rag = np.array(list(G.nodes()), dtype=np.uint)
    log.info("Found {} SVs in initial RAG.".format(len(all_sv_ids_in_rag)))

    # add single SV connected components to initial graph
    sd = SegmentationDataset(obj_type='sv', working_dir=global_params.config.working_dir)
    sv_ids = sd.ids
    diff = np.array(list(set(sv_ids).difference(set(all_sv_ids_in_rag))))
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
    log.info("Removed {} SVs from RAG because of size. Final RAG contains {}"
             " SVs in {} CCs.".format(before_cnt - G.number_of_nodes(),
                                      G.number_of_nodes(), len(cc_gs)))
    nx.write_edgelist(G, global_params.config.pruned_rag_path)

    if not global_params.config.prior_glia_removal:
        os.makedirs(global_params.config.working_dir + '/glia/', exist_ok=True)
        shutil.copy(global_params.config.pruned_rag_path, global_params.config.working_dir
                    + '/glia/neuron_rag.bz2')
