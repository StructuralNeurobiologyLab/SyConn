# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max-Planck-Institute of Neurobiology, Munich, Germany
# Authors: Philipp Schubert

import os
import shutil
from typing import Optional

import networkx as nx
import numpy as np

from syconn import global_params
from syconn.handler.basics import chunkify
from syconn.handler.config import initialize_logging
from syconn.handler.prediction_pts import predict_glia_ssv, predict_celltype_ssd, infere_cell_morphology_ssd, \
    predict_cmpt_ssd
from syconn.mp import batchjob_utils as qu
from syconn.proc.glia_splitting import run_glia_splitting, collect_glia_sv, write_astrocyte_svgraph, transform_rag_edgelist2pkl
from syconn.proc.graphs import create_ccsize_dict
from syconn.proc.graphs import split_subcc_join
from syconn.reps.segmentation import SegmentationDataset
from syconn.reps.segmentation_helper import find_missing_sv_views
from syconn.reps.super_segmentation import SuperSegmentationDataset


def run_morphology_embedding(max_n_jobs: Optional[int] = None):
    """
    Infer local morphology embeddings for all neuron reconstructions base on
    triplet-loss trained cellular morphology learning network (tCMN).
    The point based model is trained with the pts_loader_scalar (used for celltypes)

    Args:
        max_n_jobs: Number of parallel jobs.

    Notes:
        Requires :func:`~syconn.exec.exec_init.run_create_neuron_ssd`, :func:`~run_neuron_rendering` and
        :func:`~syconn.exec.skeleton.run_skeleton_generation`.
    """
    if max_n_jobs is None:
        max_n_jobs = global_params.config.ngpu_total * 4
    log = initialize_logging('morphology_embedding', global_params.config.working_dir
                             + '/logs/', overwrite=False)
    ssd = SuperSegmentationDataset(working_dir=global_params.config.working_dir)
    pred_key_appendix = ""
    log.info(f'Starting local morphology generation with {"points" if global_params.config.use_point_models else "views"}.')
    # sort ssv ids according to their number of SVs (descending)
    multi_params = ssd.ssv_ids[np.argsort(ssd.load_numpy_data('size'))[::-1]]
    if not qu.batchjob_enabled() and global_params.config.use_point_models:
        ssd_kwargs = dict(working_dir=ssd.working_dir, config=ssd.config)
        ssv_params = [dict(ssv_id=ssv_id, **ssd_kwargs) for ssv_id in multi_params]
        infere_cell_morphology_ssd(ssv_params)
    else:
        # split all cells into upper half and lower half (sorted by size)
        half_ix = len(multi_params) // 2
        njobs_per_half = max(max_n_jobs // 2, 1)
        multi_params = chunkify(multi_params[:half_ix], njobs_per_half) + \
                       chunkify(multi_params[half_ix:], njobs_per_half)
        # add ssd parameters
        multi_params = [(ssv_ids, pred_key_appendix, global_params.config.use_point_models) for ssv_ids in multi_params]
        qu.batchjob_script(multi_params, "generate_morphology_embedding",
                           n_cores=global_params.config['ncores_per_node'] // global_params.config['ngpus_per_node'],
                           log=log, suffix="", additional_flags="--gres=gpu:1", remove_jobfolder=True)
    log.info('Finished extraction of cell morphology embeddings.')


def run_cell_embedding(max_n_jobs: Optional[int] = None):
    """
    Infer cell embeddings for all neuron reconstructions base on
    triplet-loss trained cellular morphology learning network (tCMN).
    The point based model is trained with the pts_loader_scalar (used for celltypes). Multi-views
    functionality is not implemented.

    Args:
        max_n_jobs: Number of parallel jobs.

    Notes:
        Requires :func:`~syconn.exec.exec_init.run_create_neuron_ssd`, :func:`~run_neuron_rendering` and
        :func:`~syconn.exec.skeleton.run_skeleton_generation`.
    """
    if max_n_jobs is None:
        max_n_jobs = global_params.config.ngpu_total * 4
    log = initialize_logging('morphology_embedding', global_params.config.working_dir
                             + '/logs/', overwrite=False)
    ssd = SuperSegmentationDataset(working_dir=global_params.config.working_dir)
    pred_key_appendix = '_wholecell'

    log.info(f'Starting cell morphology generation with'
             f' {"points" if global_params.config.use_point_models else "views"}.')

    # sort ssv ids according to their number of SVs (descending)
    multi_params = ssd.ssv_ids[np.argsort(ssd.load_numpy_data('size'))[::-1]]
    if not qu.batchjob_enabled() and global_params.config.use_point_models:
        ssd_kwargs = dict(working_dir=ssd.working_dir, config=ssd.config)
        ssv_params = [dict(ssv_id=ssv_id, **ssd_kwargs) for ssv_id in multi_params]
        infere_cell_morphology_ssd(ssv_params, mpath=global_params.config.mpath_tnet_pts_wholecell)
    else:
        # split all cells into upper half and lower half (sorted by size)
        half_ix = len(multi_params) // 2
        njobs_per_half = max(max_n_jobs // 2, 1)
        multi_params = chunkify(multi_params[:half_ix], njobs_per_half) + \
                       chunkify(multi_params[half_ix:], njobs_per_half)
        # add ssd parameters
        multi_params = [(ssv_ids, pred_key_appendix) for ssv_ids in multi_params]
        qu.batchjob_script(multi_params, "generate_cell_embedding",
                           n_cores=global_params.config['ncores_per_node'] // global_params.config['ngpus_per_node'],
                           log=log, suffix="", additional_flags="--gres=gpu:1", remove_jobfolder=True)
    log.info('Finished extraction of whole-cell morphology embeddings.')


def run_celltype_prediction(max_n_jobs_gpu: Optional[int] = None):
    """
    Run the celltype inference based on the ``img2scalar`` CMN.

    Args:
        max_n_jobs_gpu: Number of parallel GPU jobs.

    Notes:
        Requires :func:`~syconn.exec.exec_init.run_create_neuron_ssd` and :func:`~run_neuron_rendering`.
    """
    if max_n_jobs_gpu is None:
        max_n_jobs_gpu = global_params.config.ngpu_total * 4 if qu.batchjob_enabled() else 2
    log = initialize_logging('celltype_prediction', global_params.config.working_dir + '/logs/',
                             overwrite=False)
    ssd = SuperSegmentationDataset(working_dir=global_params.config.working_dir)
    multi_params = ssd.ssv_ids[np.argsort(ssd.load_numpy_data('size'))[::-1]]
    log.info(f'Starting cell type prediction with {"points" if global_params.config.use_point_models else "views"}.')
    if not qu.batchjob_enabled() and global_params.config.use_point_models:
        predict_celltype_ssd(ssd_kwargs=dict(working_dir=global_params.config.working_dir), ssv_ids=multi_params)
    else:
        # split all cells into upper half and lower half (sorted by size)
        half_ix = len(multi_params) // 2
        njobs_per_half = max(max_n_jobs_gpu // 2, 1)
        multi_params = chunkify(multi_params[:half_ix], njobs_per_half) + \
                       chunkify(multi_params[half_ix:], njobs_per_half)
        # job parameter will be read sequentially, i.e. in order to provide only
        # one list as parameter one needs an additonal axis
        multi_params = [(ixs, global_params.config.use_point_models) for ixs in multi_params]
        qu.batchjob_script(multi_params, "predict_cell_type", log=log, suffix="", additional_flags="--gres=gpu:1",
                           n_cores=global_params.config['ncores_per_node'] // global_params.config['ngpus_per_node'],
                           remove_jobfolder=True)
    log.info(f'Finished prediction of {len(ssd.ssv_ids)} SSVs.')


def run_semsegaxoness_prediction(max_n_jobs_gpu: Optional[int] = None):
    """
    Infer and map semantic segmentation of the 2D projections onto the cell reconstruction mesh
    (``ssv.label_dict('vertex')``) via ``semseg_of_sso_nocache``.
    The following skeleton attributes are generated by ``semsegaxoness2skel`` and available in
    :py:attr:`~syconn.reps.super_segmentation_object.SuperSegmentationObject.skeleton`:

        * "axoness": Vertex predictions mapped to skeleton (see
          ``global_params.config['compartments']['map_properties_semsegax']``.
        * "axoness_avg10000": Sliding window average along skeleton (10um traversal length).
        * "axoness_avg10000_comp_maj": Majority vote on connected components after removing the
          soma.

    Args:
        max_n_jobs_gpu: Number of parallel GPU jobs.
    """
    if max_n_jobs_gpu is None:
        max_n_jobs_gpu = global_params.config.ngpu_total * 4 if qu.batchjob_enabled() else 1
    if qu.batchjob_enabled():
        n_cores = global_params.config['ncores_per_node'] // global_params.config['ngpus_per_node']
    else:
        n_cores = global_params.config['ncores_per_node']
    log = initialize_logging('compartment_prediction', global_params.config.working_dir + '/logs/',
                             overwrite=False)
    ssd = SuperSegmentationDataset(working_dir=global_params.config.working_dir)
    multi_params = ssd.ssv_ids[np.argsort(ssd.load_numpy_data('size'))[::-1]]

    if not qu.batchjob_enabled() and global_params.config.use_point_models:
        ssd_kwargs = dict(working_dir=global_params.config.working_dir)
        predict_cmpt_ssd(ssd_kwargs=ssd_kwargs, ssv_ids=multi_params, bs=1)
    else:
        multi_params = chunkify(multi_params, max_n_jobs_gpu)
        # job parameter will be read sequentially, i.e. in order to provide only
        # one list as parameter one needs an additonal axis
        multi_params = [(ixs, global_params.config.use_point_models) for ixs in multi_params]
        path_to_out = qu.batchjob_script(multi_params, 'predict_axoness_semseg', log=log,
                                         suffix="", additional_flags="--gres=gpu:1",
                                         n_cores=n_cores, remove_jobfolder=False)
        shutil.rmtree(os.path.abspath(path_to_out + "/../"), ignore_errors=True)
    log.info(f'Finished compartment prediction of {len(ssd.ssv_ids)} SSVs.')


def run_semsegspiness_prediction(max_n_jobs_gpu: Optional[int] = None):
    """
    Will store semantic spine labels inside``ssv.label_dict('vertex')['spiness]``.

    Args:
        max_n_jobs_gpu: Number of parallel GPU jobs. Used for the inference.
    """
    if max_n_jobs_gpu is None:
        max_n_jobs_gpu = global_params.config.ngpu_total * 4 if qu.batchjob_enabled() else 1
    log = initialize_logging('compartment_prediction', global_params.config.working_dir
                             + '/logs/', overwrite=False)
    ssd = SuperSegmentationDataset(working_dir=global_params.config.working_dir)
    multi_params = ssd.ssv_ids[np.argsort(ssd.load_numpy_data('size'))[::-1]]
    # split all cells into upper half and lower half (sorted by size)
    half_ix = len(multi_params) // 2
    njobs_per_half = max(max_n_jobs_gpu // 2, 1)
    multi_params = chunkify(multi_params[:half_ix], njobs_per_half) + \
                   chunkify(multi_params[half_ix:], njobs_per_half)
    # job parameter will be read sequentially, i.e. in order to provide only
    # one list as parameter one needs an additional axis
    multi_params = [(ixs,) for ixs in multi_params]

    qu.batchjob_script(multi_params, 'predict_spiness_semseg', log=log,
                       n_cores=global_params.config['ncores_per_node'] // global_params.config['ngpus_per_node'],
                       suffix="", additional_flags="--gres=gpu:1", remove_jobfolder=True)
    log.info('Finished spine prediction.')


def run_astrocyte_prediction_pts(max_n_jobs_gpu: Optional[int] = None):
    """
    Predict astrocyte and neuron supervoxels with point cloud based convolutional networks.

    Notes:
        * post-processing currently requires locking. In order to prevent locking, an additional map-reduce step
          is required to write the final probas of all SVs in a "per-storage" (per chunk attribute dict) fashion.

    Args:
        max_n_jobs_gpu:

    Notes:
        Requires :func:`~syconn.exec_init.init_cell_subcell_sds`.
    """
    if max_n_jobs_gpu is None:
        max_n_jobs_gpu = global_params.config.ngpu_total * 4
    log = initialize_logging('glia_separation', global_params.config.working_dir + '/logs/', overwrite=False)
    pred_key = "glia_probas"

    log.info("Preparing RAG.")
    G = nx.read_edgelist(global_params.config.pruned_svgraph_path, nodetype=np.uint64)

    cc_gs = sorted(list((G.subgraph(c) for c in nx.connected_components(G))), key=len, reverse=True)

    # generate parameter for view rendering of individual SSV
    sds = SegmentationDataset("sv", working_dir=global_params.config.working_dir)
    sv_size_dict = {}
    bbs = sds.load_numpy_data('bounding_box') * sds.scaling
    for ii in range(len(sds.ids)):
        sv_size_dict[sds.ids[ii]] = bbs[ii]

    # TODO: can be removed
    ccsize_dict = create_ccsize_dict(cc_gs, sv_size_dict, is_connected_components=True)

    log.info("Preparing cells for glia prediction.")
    lo_first_n = global_params.config['glia']['subcc_chunk_size_big_ssv']
    max_nb_sv = global_params.config['glia']['subcc_size_big_ssv'] + 2 * (lo_first_n - 1)
    multi_params = []
    # Store supervoxels belonging to one cell and whether they have been partitioned or not
    for g in cc_gs:
        if g.number_of_nodes() > global_params.config['glia']['rendering_max_nb_sv']:
            # partition large SSVs into small chunks with overlap
            parts = split_subcc_join(g, max_nb_sv, lo_first_n=lo_first_n)
            multi_params.extend([(p, g.subgraph(p), True) for p in parts])
        # TODO: can be removed
        elif ccsize_dict[list(g.nodes())[0]] < global_params.config['min_cc_size_ssv']:
            raise ValueError(f'Pruned rag did contain SSVs below minimum bounding box size!')
        else:
            multi_params.append((list(g.nodes()), g, False))
    # only append to this key if needed (e.g. different versions)
    # TODO: sort by size!
    np.random.seed(0)
    np.random.shuffle(multi_params)
    # job parameter will be read sequentially, i.e. in order to provide only
    # one list as parameter one needs an additional axis
    if not qu.batchjob_enabled():
        # Default SLURM fallback with Popen keeps freezing.
        working_dir = global_params.config.working_dir
        ssv_params = []
        partitioned = dict()
        for sv_ids, g, was_partitioned in multi_params:
            ssv_params.append(dict(ssv_id=sv_ids[0], sv_ids=sv_ids, working_dir=working_dir, sv_graph=g, version='tmp'))
            partitioned[sv_ids[0]] = was_partitioned
        postproc_kwargs = dict(pred_key=pred_key, lo_first_n=lo_first_n, partitioned=partitioned)
        predict_glia_ssv(ssv_params, postproc_kwargs=postproc_kwargs)
    else:
        multi_params = [(el, pred_key) for el in chunkify(multi_params, max_n_jobs_gpu)]
        qu.batchjob_script(multi_params, 'predict_glia_pts', log=log,
                           n_cores=global_params.config['ncores_per_node'] // global_params.config['ngpus_per_node'],
                           suffix="", additional_flags="--gres=gpu:1", remove_jobfolder=True)
    log.info('Finished glia prediction.')


def run_astrocyte_prediction():
    """
    Predict astrocyte supervoxels based on the ``img2scalar`` CMN.

    Notes:
        Requires :func:`~syconn.exec_init.init_cell_subcell_sds` and
        :func:`~run_astrocyte_rendering`.
    """
    log = initialize_logging('glia_separation', global_params.config.working_dir + '/logs/',
                             overwrite=False)
    # only append to this key if needed (e.g. different versions)
    pred_key = "glia_probas"

    # Load initial RAG from  Knossos mergelist text file.
    g = nx.read_edgelist(global_params.config.pruned_svgraph_path, nodetype=np.uint64)
    all_sv_ids_in_rag = np.array(list(g.nodes()), dtype=np.uint64)

    log.debug('Found {} CCs with a total of {} SVs in inital RAG.'.format(
        nx.number_connected_components(g), g.number_of_nodes()))
    # chunk them
    sd = SegmentationDataset("sv", working_dir=global_params.config.working_dir)
    multi_params = chunkify(sd.so_dir_paths, global_params.config.ngpu_total * 2)
    # get model properties
    model_kwargs = 'get_glia_model_e3'
    # all other kwargs like obj_type='sv' and version are the current SV
    # SegmentationDataset by default
    so_kwargs = dict(working_dir=global_params.config.working_dir)
    # for glia views set woglia to False (because glia are included),
    #  raw_only to True
    pred_kwargs = dict(woglia=False, pred_key=pred_key, verbose=False, raw_only=True)

    multi_params = [[par, model_kwargs, so_kwargs, pred_kwargs] for par in
                    multi_params]
    n_cores = global_params.config['ncores_per_node'] // global_params.config['ngpus_per_node']
    qu.batchjob_script(multi_params, "predict_sv_views_chunked_e3", log=log,
                       script_folder=None, n_cores=n_cores,
                       suffix="_glia", additional_flags="--gres=gpu:1",
                       remove_jobfolder=True)
    log.info('Finished glia prediction. Checking completeness.')
    res = find_missing_sv_views(sd, woglia=False, n_cores=global_params.config['ncores_per_node'])
    missing_contained_in_rag = np.intersect1d(res, all_sv_ids_in_rag)
    if len(missing_contained_in_rag) != 0:
        msg = "Not all SVs were predicted! {}/{} missing:\n" \
              "{}".format(len(missing_contained_in_rag), len(all_sv_ids_in_rag),
                          missing_contained_in_rag[:100])
        log.error(msg)
        raise ValueError(msg)
    else:
        log.info('Success.')


def run_astrocyte_splitting():
    """
    Uses the pruned RAG at ``global_params.config.pruned_svgraph_path`` (stored as edge list .bz2 file)
    which is  computed in :func:`~syconn.exec.exec_init.init_cell_subcell_sds` to split astrocyte
    fragments from neuron reconstructions and separate those and entire glial cells from
    the neuron supervoxel graph.

    Stores neuron SV graph at :attr:`~syconn.handler.config.DynConfig.neuron_svgraph_path`
    which is then used by :func:`~syconn.exec.exec_init.run_create_neuron_ssd`.

    Todo:
        * refactor how splits are stored, currently those are stored at ssv_tmp

    Notes:
        Requires :func:`~syconn.exec_init.init_cell_subcell_sds`,
        :func:`~run_astrocyte_rendering` and :func:`~run_astrocyte_prediction`.
    """
    log = initialize_logging('astrocyte_separation', global_params.config.working_dir + '/logs/',
                             overwrite=False)
    G = nx.read_edgelist(global_params.config.pruned_svgraph_path, nodetype=np.uint64)
    log.debug('Found {} CCs with a total of {} SVs in inital RAG.'.format(
        nx.number_connected_components(G), G.number_of_nodes()))

    if not os.path.isdir(global_params.config.working_dir + "/glia/"):
        os.makedirs(global_params.config.working_dir + "/glia/")
    transform_rag_edgelist2pkl(G)

    # first perform glia splitting based on multi-view predictions, results are
    # stored at SuperSegmentationDataset ssv_gliaremoval
    run_glia_splitting()

    # collect all neuron and glia SVs and store them in numpy array
    collect_glia_sv()

    # use reconnected RAG or initial rag here
    recon_nx = G
    # create glia / neuron RAGs
    write_astrocyte_svgraph(recon_nx, global_params.config['min_cc_size_ssv'], log=log)
    log.info("Finished astrocyte splitting. Resulting neuron and astrocyte SV graphs are stored at {}."
             "".format(global_params.config.working_dir + "/glia/"))
