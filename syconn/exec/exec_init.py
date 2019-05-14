# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max-Planck-Institute of Neurobiology, Munich, Germany
# Authors: Philipp Schubert, Joergen Kornfeld

from knossos_utils import knossosdataset
import time
import os
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


def sd_init(co, generate_sv_meshes, max_n_jobs):
    sd_seg = SegmentationDataset(obj_type=co, working_dir=global_params.config.working_dir,
                                 version="0")
    multi_params = chunkify(sd_seg.so_dir_paths, max_n_jobs)
    so_kwargs = dict(working_dir=global_params.config.working_dir, obj_type=co)
    if co != "sv" or (co == "sv" and generate_sv_meshes):
        multi_params = [[par, so_kwargs] for par in multi_params]
        _ = qu.QSUB_script(multi_params, "mesh_caching", suffix=co, remove_jobfolder=True,
                           n_max_co_processes=global_params.NCORE_TOTAL)

    if co == "sv":
        _ = qu.QSUB_script(multi_params, "sample_location_caching",
                           n_max_co_processes=global_params.NCORE_TOTAL,
                           suffix=co, remove_jobfolder=True)

    # now cache mesh properties
    sd_proc.dataset_analysis(sd_seg, recompute=False, compute_meshprops=True)


def kd_init(co, chunk_size, transf_func_kd_overlay, load_cellorganelles_from_kd_overlaycubes, \
    cube_of_interest_bb, log):
    oew.generate_subcell_kd_from_proba(
        co, chunk_size=chunk_size, transf_func_kd_overlay=transf_func_kd_overlay,
        load_cellorganelles_from_kd_overlaycubes=load_cellorganelles_from_kd_overlaycubes,
        cube_of_interest_bb=cube_of_interest_bb, log=log)


def init_cell_subcell_sds(chunk_size=None, n_folders_fs=10000, n_folders_fs_sc=10000, max_n_jobs=None,
                          generate_sv_meshes=False, load_cellorganelles_from_kd_overlaycubes=False,
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
        max_n_jobs = global_params.NCORE_TOTAL
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
        time.sleep(10)
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
    log.info('Finished extraction and mapping after {:.0f}s.'
             ''.format(time.time() - start))

    log.info('Caching properties and calculating meshes for subcellular '
             'structures {} and cell supervoxels'.format(global_params.existing_cell_organelles))
    start = time.time()
    ps = [Process(target=sd_init, args=[co, generate_sv_meshes, max_n_jobs])
          for co in ["sv"] + global_params.existing_cell_organelles]
    for p in ps:
        p.start()
        time.sleep(10)
    for p in ps:
        p.join()
    log.info('Finished caching of meshes and rendering locations after {:.0f}s.'
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
    log.info('Found {} single connected component SVs which were missing'
             ' in initial RAG.'.format(len(diff)))

    for ix in diff:
        G.add_edge(ix, ix)

    log.debug("Found {} SVs in initial RAG after adding size-one connected "
              "components. Writing kml text file.".format(G.number_of_nodes()))

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


def run_create_sds(chunk_size=None, n_folders_fs=10000, max_n_jobs=None,
                   generate_sv_meshes=False, load_cellorganelles_from_kd_overlaycubes=False,
                   transf_func_kd_overlay=None, cube_of_interest_bb=None):
    """DEPRECATED

    Parameters
    ----------
    chunk_size :
    max_n_jobs : int
    n_folders_fs :
    generate_sv_meshes :
    load_cellorganelles_from_kd_overlaycubes : bool
        Load cell orgenelle prob/seg data from overlaycubes instead of raw cubes.
    transf_func_kd_overlay : Dict[callable]
        Method which is to applied to cube data if `load_from_kd_overlaycubes`
        is True. Must be a dictionary with keys `global_params.existing_cell_organelles`.
    cube_of_interest_bb : Tuple[np.ndarray]
        Defines the bounding box of the cube to process. By default this is
        set to (np.zoers(3); kd.boundary).


    Returns
    -------

    """
    if chunk_size is None:
        chunk_size = [512, 512, 512]
    if max_n_jobs is None:
        max_n_jobs = global_params.NCORE_TOTAL * 2
    log = initialize_logging('create_sds', global_params.config.working_dir +
                             '/logs/', overwrite=True)
    # Sets initial values of object
    kd = kd_factory(global_params.config.kd_seg_path)
    if cube_of_interest_bb is None:
        cube_of_interest_bb = [np.zeros(3, dtype=np.int), kd.boundary]
    size = cube_of_interest_bb[1] - cube_of_interest_bb[0] + 1
    offset = cube_of_interest_bb[0]
    # resulting ChunkDataset, required for SV extraction --
    # Object extraction - 2h, the same has to be done for all cell organelles
    cd_dir = global_params.config.working_dir + "chunkdatasets/sv/"
    # Class that contains a dict of chunks (with coordinates) after initializing it
    cd = chunky.ChunkDataset()
    cd.initialize(kd, kd.boundary, chunk_size, cd_dir,
                  box_coords=[0, 0, 0], fit_box_size=True)
    log.info('Generating SegmentationDatasets for cell and cell '
             'organelle supervoxels.')
    oew.from_ids_to_objects(cd, "sv", overlaydataset_path=global_params.config.kd_seg_path,
                            n_chunk_jobs=max_n_jobs, hdf5names=["sv"], n_max_co_processes=None,
                            n_folders_fs=n_folders_fs, use_combined_extraction=True, size=size,
                            offset=offset, log=log)

    # Object Processing -- Perform after mapping to also cache mapping ratios
    sd = SegmentationDataset("sv", working_dir=global_params.config.working_dir)
    sd_proc.dataset_analysis(sd, recompute=True, compute_meshprops=False)

    log.info("Extracted {} cell SVs. Preparing rendering locations "
             "(and meshes if not provided).".format(len(sd.ids)))
    # chunk them
    multi_params = chunkify(sd.so_dir_paths, max_n_jobs)
    # all other kwargs like obj_type='sv' and version are the current SV SegmentationDataset by default
    so_kwargs = dict(working_dir=global_params.config.working_dir, obj_type='sv')
    multi_params = [[par, so_kwargs] for par in multi_params]
    if generate_sv_meshes:
        start = time.time()
        _ = qu.QSUB_script(multi_params, "mesh_caching",
                           n_max_co_processes=global_params.NCORE_TOTAL,
                           remove_jobfolder=True)
        log.info('Finished mesh caching of {} "{}"-SVs after {:.0f}s.'
                 ''.format(len(sd.ids), 'cell', time.time() - start))
    start = time.time()
    _ = qu.QSUB_script(multi_params, "sample_location_caching",
                       n_max_co_processes=global_params.NCORE_TOTAL)
    log.info('Finished caching of rendering locations after {:.0f}s.'
             ''.format(time.time() - start))
    # recompute=False: only collect new sample_location property
    sd_proc.dataset_analysis(sd, compute_meshprops=True, recompute=False)
    log.info('Finished preparation of cell SVs after {:.0f}s.'.format(time.time() - start))
    # create SegmentationDataset for each cell organelle
    if transf_func_kd_overlay is None:
        transf_func_kd_overlay = {k: None for k in global_params.existing_cell_organelles}
    for co in global_params.existing_cell_organelles:
        cd = chunky.ChunkDataset()
        cd_dir = global_params.config.working_dir + "chunkdatasets/{}/".format(co)
        cd.initialize(kd, kd.boundary, chunk_size, cd_dir,
                      box_coords=[0, 0, 0], fit_box_size=True)
        log.info('Started object extraction of cellular organelles "{}" from '
                 '{} chunks.'.format(co, len(cd.chunk_dict)))
        prob_kd_path_dict = {co: getattr(global_params.config, 'kd_{}_path'.format(co))}
        # This creates a SegmentationDataset of type 'co'
        prob_thresh = global_params.config.entries["Probathresholds"][co]  # get probability threshold

        # `from_probabilities_to_objects` will export a KD at `path`, remove if already existing
        path = "{}/knossosdatasets/{}_seg/".format(global_params.config.working_dir, co)
        if os.path.isdir(path):
            log.debug('Found existing KD at {}. Removing it now.'.format(path))
            shutil.rmtree(path)
            log.debug('Done')
        target_kd = knossosdataset.KnossosDataset()
        target_kd.initialize_without_conf(path, kd.boundary, kd.scale, kd.experiment_name, mags=[1, ])
        target_kd = knossosdataset.KnossosDataset()
        target_kd.initialize_from_knossos_path(path)
        oew.from_probabilities_to_objects(cd, co, # membrane_kd_path=global_params.config.kd_barrier_path,  # TODO: currently does not exist
                                          prob_kd_path_dict=prob_kd_path_dict, thresholds=[prob_thresh],
                                          workfolder=global_params.config.working_dir,
                                          hdf5names=[co], n_max_co_processes=None, target_kd=target_kd,
                                          n_folders_fs=n_folders_fs, debug=False, size=size, offset=offset,
                                          load_from_kd_overlaycubes=load_cellorganelles_from_kd_overlaycubes,
                                          transf_func_kd_overlay=transf_func_kd_overlay[co], log=log)
        sd_co = SegmentationDataset(obj_type=co, working_dir=global_params.config.working_dir)

        # TODO: check if this is faster then the alternative below
        sd_proc.dataset_analysis(sd_co, recompute=True, compute_meshprops=False)
        multi_params = chunkify(sd_co.so_dir_paths, max_n_jobs)
        so_kwargs = dict(working_dir=global_params.config.working_dir, obj_type=co)
        start = time.time()
        multi_params = [[par, so_kwargs] for par in multi_params]
        _ = qu.QSUB_script(multi_params, "mesh_caching",
                           n_max_co_processes=global_params.NCORE_TOTAL)
        log.info('Finished mesh caching of {} "{}"-SVs after {:.0f}s.'
                 ''.format(len(sd_co.ids), co, time.time() - start))
        # # Old alternative, requires much more reads/writes then above solution
        # sd_proc.dataset_analysis(sd_co, recompute=True, compute_meshprops=True)

        # About 0.2 h per object class
        start = time.time()
        sd_proc.map_objects_to_sv(sd, co, global_params.config.kd_seg_path,
                                  n_jobs=max_n_jobs)
        log.info('Finished mapping of {} cellular organelles of type "{}" to '
                 'cell SVs after {:.0f}s.'.format(len(sd_co.ids), co,
                                                  time.time() - start))
        sd_proc.dataset_analysis(sd_co, recompute=False, compute_meshprops=True)
