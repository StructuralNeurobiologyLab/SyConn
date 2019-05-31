# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max-Planck-Institute of Neurobiology, Munich, Germany
# Authors: Philipp Schubert, Joergen Kornfeld

from knossos_utils import knossosdataset
import time
import os
import sys
from multiprocessing import Process
import shutil
import networkx as nx
import numpy as np
knossosdataset._set_noprint(True)
from knossos_utils import chunky
from syconn import global_params
from syconn.extraction import object_extraction_wrapper as oew
from syconn.proc import sd_proc
from syconn.reps.segmentation import SegmentationDataset
from syconn.handler.config import initialize_logging
from syconn.mp import batchjob_utils as qu
from syconn.proc.graphs import create_ccsize_dict
from syconn.handler.basics import chunkify, kd_factory


def sd_init(co, max_n_jobs, log=None):
    sd_seg = SegmentationDataset(obj_type=co, working_dir=global_params.config.working_dir,
                                 version="0")
    multi_params = chunkify(sd_seg.so_dir_paths, max_n_jobs)
    so_kwargs = dict(working_dir=global_params.config.working_dir, obj_type=co)
    multi_params = [[par, so_kwargs] for par in multi_params]

    if not global_params.config.use_new_meshing and (co != "sv" or (co == "sv" and
            global_params.config.allow_mesh_gen_cells)):
        _ = qu.QSUB_script(multi_params, "mesh_caching", suffix=co, remove_jobfolder=False,
                           n_max_co_processes=global_params.NCORE_TOTAL, log=log)

    if co == "sv":
        _ = qu.QSUB_script(multi_params, "sample_location_caching",
                           n_max_co_processes=global_params.NCORE_TOTAL,
                           suffix=co, remove_jobfolder=True, log=log)

    # write mesh properties to attribute dictionaries if old meshing is active
    if not global_params.config.use_new_meshing:
        sd_proc.dataset_analysis(sd_seg, recompute=False, compute_meshprops=True)


def kd_init(co, chunk_size, transf_func_kd_overlay, load_cellorganelles_from_kd_overlaycubes,
    cube_of_interest_bb, log):
    oew.generate_subcell_kd_from_proba(
        co, chunk_size=chunk_size, transf_func_kd_overlay=transf_func_kd_overlay,
        load_cellorganelles_from_kd_overlaycubes=load_cellorganelles_from_kd_overlaycubes,
        cube_of_interest_bb=cube_of_interest_bb, log=log)


def init_cell_subcell_sds(chunk_size=None, n_folders_fs=10000, n_folders_fs_sc=10000, max_n_jobs=None,
                          load_cellorganelles_from_kd_overlaycubes=False,
                          transf_func_kd_overlay=None, cube_of_interest_bb=None):
    # TODO: Don't extract sj objects and replace their use-cases with syn objects (?)
    """
    If `global_params.config.prior_glia_removal==True`:
        stores pruned RAG at `global_params.config.pruned_rag_path`, required for all glia
        removal steps. `run_glia_splitting` will finally return `neuron_rag.bz2`
    else:
        stores pruned RAG at `global_params.config.working_dir + /glia/neuron_rag.bz2`, required
        for `run_create_neuron_ssd`.

    Parameters
    ----------
    chunk_size :
    n_folders_fs :
    n_folders_fs_sc :
    max_n_jobs :
    generate_sv_meshes :
    load_cellorganelles_from_kd_overlaycubes :
    transf_func_kd_overlay :
    cube_of_interest_bb :

    Returns
    -------

    """
    log = initialize_logging('create_sds', global_params.config.working_dir +
                             '/logs/', overwrite=True)
    if transf_func_kd_overlay is None:
        transf_func_kd_overlay = {k: None for k in global_params.existing_cell_organelles}
    if chunk_size is None:
        chunk_size = [512, 512, 512]
    if max_n_jobs is None:
        max_n_jobs = global_params.NCORE_TOTAL * 2
        # loading cached data or adapt number of jobs/cache size dynamically, dependent on the
        # dataset
    kd = kd_factory(global_params.config.kd_seg_path)
    if cube_of_interest_bb is None:
        cube_of_interest_bb = [np.zeros(3, dtype=np.int), kd.boundary]

    log.info('Generating KnossosDatasets for subcellular structures {}.'
             ''.format(global_params.existing_cell_organelles))
    start = time.time()
    ps = [Process(target=kd_init, args=[co, chunk_size, transf_func_kd_overlay,
                                        load_cellorganelles_from_kd_overlaycubes,
                                        cube_of_interest_bb, log])
          for co in global_params.existing_cell_organelles]
    for p in ps:
        p.start()
        time.sleep(5)
    for p in ps:
        p.join()
    log.info('Finished KD generation after {:.0f}s.'.format(time.time() - start))

    log.info('Generating SegmentationDatasets for subcellular structures {} and'
             ' cell supervoxels'.format(global_params.existing_cell_organelles))
    start = time.time()
    sd_proc.map_subcell_extract_props(
        global_params.config.kd_seg_path, global_params.config.kd_organelle_seg_paths,
        n_folders_fs=n_folders_fs, n_folders_fs_sc=n_folders_fs_sc, n_chunk_jobs=max_n_jobs,
        cube_of_interest_bb=cube_of_interest_bb, chunk_size=chunk_size, log=log)
    log.info('Finished extraction and mapping after {:.2f}s.'
             ''.format(time.time() - start))

    log.info('Caching properties of subcellular structures {} and cell'
             ' supervoxels'.format(global_params.existing_cell_organelles))
    start = time.time()
    ps = [Process(target=sd_init, args=[co, max_n_jobs, log])
          for co in ["sv"] + global_params.existing_cell_organelles]
    for p in ps:
        p.start()
        time.sleep(5)
    for p in ps:
        p.join()
    log.info('Finished SD caching after {:.2f}s.'
             ''.format(time.time() - start))


def run_create_rag():
    log = initialize_logging('create_rag', global_params.config.working_dir +
                             '/logs/', overwrite=True)
    # Crop RAG according to cell SVs found during SD generation and apply size threshold
    G = nx.read_edgelist(global_params.config.init_rag_path, nodetype=np.uint)

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
        if ccsize_dict[ix] < global_params.min_cc_size_ssv:
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
