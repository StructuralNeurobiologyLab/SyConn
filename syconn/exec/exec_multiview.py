# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max-Planck-Institute of Neurobiology, Munich, Germany
# Authors: Philipp Schubert, Joergen Kornfeld

import tqdm
import pickle as pkl
import glob
import os
import numpy as np
import networkx as nx
import re
import shutil

from syconn.reps.rep_helper import knossos_ml_from_ccs
from syconn.reps.segmentation_helper import find_missing_sv_views
from syconn.reps.super_segmentation import SuperSegmentationObject
from syconn.global_params import rag_suffix, RENDERING_MAX_NB_SV
from syconn.proc.glia_splitting import qsub_glia_splitting, collect_glia_sv, \
    write_glia_rag, transform_rag_edgelist2pkl
from syconn.reps.segmentation import SegmentationDataset
from syconn.reps.segmentation_helper import find_missing_sv_attributes
from syconn.handler.prediction import get_glia_model
from syconn.proc.graphs import create_ccsize_dict
from syconn.proc import ssd_proc
from syconn.reps.super_segmentation_helper import find_incomplete_ssv_views
from syconn import global_params
from syconn.handler.prediction import get_axoness_model
from syconn.handler.basics import chunkify
from syconn.reps.super_segmentation import SuperSegmentationDataset
from syconn.reps.super_segmentation_helper import find_missing_sv_attributes_in_ssv
from syconn.handler.config import initialize_logging
from syconn.mp import batchjob_utils as qu
from syconn.exec import exec_skeleton


def run_morphology_embedding(max_n_jobs=None):
    if max_n_jobs is None:
        max_n_jobs = global_params.NGPU_TOTAL * 2
    log = initialize_logging('morphology_embedding', global_params.config.working_dir
                             + '/logs/', overwrite=False)
    ssd = SuperSegmentationDataset(working_dir=global_params.config.working_dir)
    pred_key_appendix = ""

    multi_params = np.array(ssd.ssv_ids, dtype=np.uint)
    nb_svs_per_ssv = np.array([len(ssd.mapping_dict[ssv_id]) for ssv_id
                               in ssd.ssv_ids])
    # sort ssv ids according to their number of SVs (descending)
    multi_params = multi_params[np.argsort(nb_svs_per_ssv)[::-1]]
    multi_params = chunkify(multi_params, max_n_jobs)
    # add ssd parameters
    multi_params = [(ssv_ids, ssd.version, ssd.version_dict, ssd.working_dir,
                     pred_key_appendix) for ssv_ids in multi_params]
    qu.QSUB_script(multi_params, "generate_morphology_embedding", n_max_co_processes=global_params.NGPU_TOTAL,
                   n_cores=global_params.NCORES_PER_NODE // global_params.NGPUS_PER_NODE,
                   suffix="", additional_flags="--gres=gpu:1", remove_jobfolder=True)
    log.info('Finished extraction of cell morphology embedding.')


def run_axoness_mapping(max_n_jobs=None):
    if max_n_jobs is None:
        max_n_jobs = global_params.NCORE_TOTAL * 2
    """Maps axon prediction of rendering locations onto SSV skeletons"""
    log = initialize_logging('axon_mapping', global_params.config.working_dir + '/logs/',
                             overwrite=False)
    pred_key_appendix = ""
    # Working directory has to be changed globally in global_params
    ssd = SuperSegmentationDataset(working_dir=global_params.config.working_dir)

    multi_params = np.array(ssd.ssv_ids, dtype=np.uint)
    # sort ssv ids according to their number of SVs (descending)
    nb_svs_per_ssv = np.array([len(ssd.mapping_dict[ssv_id]) for ssv_id
                               in ssd.ssv_ids])
    multi_params = multi_params[np.argsort(nb_svs_per_ssv)[::-1]]
    multi_params = chunkify(multi_params, max_n_jobs)

    multi_params = [(par, pred_key_appendix) for par in multi_params]
    log.info('Starting axoness mapping.')
    _ = qu.QSUB_script(multi_params, "map_viewaxoness2skel",
                       n_max_co_processes=global_params.NCORE_TOTAL,
                       suffix="", n_cores=1, remove_jobfolder=True)
    # TODO: perform completeness check
    log.info('Finished axoness mapping.')


def run_axoness_prediction(max_n_jobs_gpu=None, e3=False):
    log = initialize_logging('axon_prediction', global_params.config.working_dir + '/logs/',
                             overwrite=False)
    if max_n_jobs_gpu is None:
        max_n_jobs_gpu = global_params.NGPU_TOTAL * 2
    # here because all qsub jobs will start a script referring to 'global_params.config.working_dir'
    ssd = SuperSegmentationDataset(working_dir=global_params.config.working_dir)
    sd = ssd.get_segmentationdataset("sv")
    # chunk them
    multi_params = chunkify(sd.so_dir_paths, max_n_jobs_gpu)
    pred_key = "axoness_probas"  # leave this fixed because it is used all over
    # get model properties
    log.info('Performing axon prediction of neuron views. Labels will be stored '
             'on SV level in the attribute dict with key "{}"'.format(pred_key))
    if e3 is True:
        model_kwargs = 'get_axoness_model_e3'
    else:
        m = get_axoness_model()
        model_kwargs = dict(model_path=m._path, normalize_data=m.normalize_data,
                            imposed_batch_size=m.imposed_batch_size, nb_labels=m.nb_labels,
                            channels_to_load=m.channels_to_load)
    
    #all other kwargs like obj_type='sv' and version are the current SV SegmentationDataset by default
    so_kwargs = dict(working_dir=global_params.config.working_dir)
    # for axoness views set woglia to True (because glia were removed beforehand),
    #  raw_only to False
    pred_kwargs = dict(woglia=True, pred_key=pred_key, verbose=False,
                       raw_only=False)
    multi_params = [[par, model_kwargs, so_kwargs, pred_kwargs] for
                    par in multi_params]

    if e3 is True:
        _ = qu.QSUB_script(multi_params, "predict_sv_views_chunked_e3",
                           n_max_co_processes=global_params.NGPU_TOTAL,
                           n_cores=global_params.NCORES_PER_NODE // global_params.NGPUS_PER_NODE,
                           suffix="_axoness", additional_flags="--gres=gpu:1",
                           remove_jobfolder=True)
    else:
        for par in multi_params:
            mk = par[1]
            # Single GPUs are made available for every job via slurm, no need for random assignments.
            mk["init_gpu"] = 0  # np.random.rand(0, 2)
        _ = qu.QSUB_script(multi_params, "predict_sv_views_chunked",
                           n_max_co_processes=global_params.NGPU_TOTAL // 2,
                           n_cores=global_params.NCORES_PER_NODE, suffix="_axoness",
                           additional_flags="--gres=gpu:1",
                           remove_jobfolder=True)
    log.info('Finished axon prediction. Now checking for missing predictions.')
    res = find_missing_sv_attributes_in_ssv(ssd, pred_key, n_cores=global_params.NCORES_PER_NODE)
    if len(res) > 0:
        log.error("Attribute '{}' missing for follwing"
                  " SVs:\n{}".format(pred_key, res))
    else:
        log.info('Success.')


def run_celltype_prediction(max_n_jobs_gpu=None):
    if max_n_jobs_gpu is None:
        max_n_jobs_gpu = global_params.NGPU_TOTAL * 2
    log = initialize_logging('celltype_prediction', global_params.config.working_dir+ '/logs/',
                             overwrite=False)
    ssd = SuperSegmentationDataset(working_dir=global_params.config.working_dir)
    # shuffle SV IDs
    np.random.seed(0)

    log.info('Starting cell type prediction.')
    nb_svs_per_ssv = np.array([len(ssd.mapping_dict[ssv_id])
                               for ssv_id in ssd.ssv_ids])
    multi_params = ssd.ssv_ids
    ordering = np.argsort(nb_svs_per_ssv)
    multi_params = multi_params[ordering[::-1]]
    max_n_jobs_gpu = np.max([max_n_jobs_gpu, len(multi_params) // 200])  # at most 200 SSV per job
    multi_params = chunkify(multi_params, max_n_jobs_gpu)
    # job parameter will be read sequentially, i.e. in order to provide only
    # one list as parameter one needs an additonal axis
    multi_params = [(ixs, ) for ixs in multi_params]

    # TODO: switch n_max_co_processes to `global_params.NGPUS_TOTAL` as soon as EGL ressource allocation works!
    path_to_out = qu.QSUB_script(multi_params, "predict_cell_type",
                                 n_max_co_processes=global_params.NNODES_TOTAL,
                                 suffix="", additional_flags="--gres=gpu:2",
                                 n_cores=global_params.NCORES_PER_NODE)
    log.info('Finished prediction of {} SSVs. Checking completeness.'
             ''.format(len(ordering)))
    out_files = glob.glob(path_to_out + "*.pkl")
    err = []
    for fp in out_files:
        with open(fp, "rb") as f:
            local_err = pkl.load(f)
        err += list(local_err)
    if len(err) > 0:
        log.error("{} errors occurred for SSVs with ID: "
                  "{}".format(len(err), [el[0] for el in err]))
    else:
        log.info('Success.')


def run_spiness_prediction(max_n_jobs_gpu=None, max_n_jobs=None):
    if max_n_jobs is None:
        max_n_jobs = global_params.NCORE_TOTAL * 2
    if max_n_jobs_gpu is None:
        max_n_jobs_gpu = global_params.NGPU_TOTAL * 2
    log = initialize_logging('spine_identification', global_params.config.working_dir
                             + '/logs/', overwrite=False)
    ssd = SuperSegmentationDataset(working_dir=global_params.config.working_dir)
    pred_key = "spiness"

    # run semantic spine segmentation on multi views
    sd = ssd.get_segmentationdataset("sv")
    # chunk them
    multi_params = chunkify(sd.so_dir_paths, max_n_jobs_gpu)
    # set model properties
    model_kwargs = dict(src=global_params.config.mpath_spiness,
                        multi_gpu=False)
    so_kwargs = dict(working_dir=global_params.config.working_dir)
    pred_kwargs = dict(pred_key=pred_key)
    multi_params = [[par, model_kwargs, so_kwargs, pred_kwargs]
                    for par in multi_params]
    log.info('Starting spine prediction.')
    qu.QSUB_script(multi_params, "predict_spiness_chunked",
                   n_max_co_processes=global_params.NGPU_TOTAL,
                   n_cores=global_params.NCORES_PER_NODE // global_params.NGPUS_PER_NODE,
                   suffix="",  additional_flags="--gres=gpu:1",
                   remove_jobfolder=True)
    log.info('Finished spine prediction.')
    # map semantic spine segmentation of multi views on SSV mesh
    # TODO: CURRENTLY HIGH MEMORY CONSUMPTION
    if not ssd.mapping_dict_exists:
        raise ValueError('Mapping dict does not exist.')
    multi_params = np.array(ssd.ssv_ids, dtype=np.uint)
    nb_svs_per_ssv = np.array([len(ssd.mapping_dict[ssv_id]) for ssv_id
                               in ssd.ssv_ids])
    # sort ssv ids according to their number of SVs (descending)
    multi_params = multi_params[np.argsort(nb_svs_per_ssv)[::-1]]
    multi_params = chunkify(multi_params, max_n_jobs)
    # add ssd parameters
    kwargs_semseg2mesh = dict(semseg_key=pred_key, force_recompute=True)
    multi_params = [(ssv_ids, ssd.version, ssd.version_dict, ssd.working_dir,
                     kwargs_semseg2mesh) for ssv_ids in multi_params]
    log.info('Starting mapping of spine predictions to neurite surfaces.')
    qu.QSUB_script(multi_params, "map_spiness", n_max_co_processes=global_params.NCORE_TOTAL,
                   n_cores=1, suffix="", additional_flags="", remove_jobfolder=True)
    log.info('Finished spine mapping.')


def run_neuron_rendering(max_n_jobs=None):
    if max_n_jobs is None:
        max_n_jobs = global_params.NGPU_TOTAL * 4 if global_params.PYOPENGL_PLATFORM == 'egl' \
            else global_params.NCORE_TOTAL * 4
    log = initialize_logging('neuron_view_rendering',
                             global_params.config.working_dir + '/logs/')
    # view rendering prior to glia removal, choose SSD accordingly
    ssd = SuperSegmentationDataset(working_dir=global_params.config.working_dir)

    #  TODO: use actual size criteria, e.g. number of sampling locations
    nb_svs_per_ssv = np.array([len(ssd.mapping_dict[ssv_id])
                               for ssv_id in ssd.ssv_ids])

    # render normal size SSVs
    size_mask = nb_svs_per_ssv <= global_params.RENDERING_MAX_NB_SV
    multi_params = ssd.ssv_ids[size_mask]
    # sort ssv ids according to their number of SVs (descending)
    ordering = np.argsort(nb_svs_per_ssv[size_mask])
    multi_params = multi_params[ordering[::-1]]
    multi_params = chunkify(multi_params, max_n_jobs)
    # list of SSV IDs and SSD parameters need to be given to a single QSUB job
    multi_params = [(ixs, global_params.config.working_dir) for ixs in multi_params]
    log.info('Started rendering of {} SSVs. '.format(np.sum(size_mask)))
    if np.sum(~size_mask) > 0:
        log.info('{} huge SSVs will be rendered afterwards using the whole'
                 ' cluster.'.format(np.sum(~size_mask)))
    # generic
    if global_params.PYOPENGL_PLATFORM == 'osmesa':  # utilize all CPUs
        path_to_out = qu.QSUB_script(multi_params, "render_views",
                           n_max_co_processes=global_params.NCORE_TOTAL,
                           remove_jobfolder=False)
    elif global_params.PYOPENGL_PLATFORM == 'egl':  # utilize 1 GPU per task
        # run EGL on single node: 20 parallel jobs
        if global_params.config.working_dir is not None and 'example_cube' in \
                global_params.config.working_dir:
            n_cores = 1
            n_parallel_jobs = global_params.NCORES_PER_NODE
            path_to_out = qu.QSUB_script(multi_params, "render_views",
                               n_max_co_processes=n_parallel_jobs,
                               additional_flags="--gres=gpu:2",
                               n_cores=n_cores, remove_jobfolder=False)
        # run on whole cluster
        else:
            n_cores = global_params.NCORES_PER_NODE // global_params.NGPUS_PER_NODE
            n_parallel_jobs = global_params.NGPU_TOTAL
            path_to_out = qu.QSUB_script(multi_params, "render_views_egl",
                               n_max_co_processes=n_parallel_jobs,
                               additional_flags="--gres=gpu:1",
                               n_cores=n_cores, remove_jobfolder=False)
    else:
        raise RuntimeError('Specified OpenGL platform "{}" not supported.'
                           ''.format(global_params.PYOPENGL_PLATFORM))
    if np.sum(~size_mask) > 0:
        log.info('Finished rendering of {}/{} SSVs.'.format(len(ordering),
                                                            len(nb_svs_per_ssv)))
        # identify huge SSVs and process them individually on whole cluster
        big_ssv = ssd.ssv_ids[~size_mask]
        for kk, ssv_id in enumerate(big_ssv):
            ssv = ssd.get_super_segmentation_object(ssv_id)
            log.info("Processing SSV [{}/{}] with {} SVs on whole cluster.".format(
                kk+1, len(big_ssv), len(ssv.sv_ids)))
            ssv.render_views(add_cellobjects=True, cellobjects_only=False,
                             woglia=True, qsub_pe="openmp", overwrite=True,
                             qsub_co_jobs=global_params.NCORE_TOTAL,
                             skip_indexviews=False, resume_job=False)
    log.info('Finished rendering of all SSVs. Checking completeness.')
    res = find_incomplete_ssv_views(ssd, woglia=True, n_cores=global_params.NCORES_PER_NODE)
    if len(res) != 0:
        msg = "Not all SSVs were rendered completely! Missing:\n{}".format(res)
        log.error(msg)
        raise RuntimeError(msg)
    else:
        shutil.rmtree(os.path.abspath(path_to_out + "/../"), ignore_errors=True)
        log.info('Success.')


def run_create_neuron_ssd():
    """
    Creates SuperSegmentationDataset with version 0.

    Parameters
    ----------
    prior_glia_removal : bool
        If False, will apply filtering to create SSO objects above minimum size, see global_params.min_cc_size_ssv
         and cache SV sample locations.

    Returns
    -------

    """
    log = initialize_logging('create_neuron_ssd', global_params.config.working_dir + '/logs/',
                             overwrite=False)
    suffix = global_params.rag_suffix
    # TODO: the following paths currently require prior glia-splitting
    g_p = "{}/glia/neuron_rag{}.bz2".format(global_params.config.working_dir, suffix)
    rag_g = nx.read_edgelist(g_p, nodetype=np.uint)
    # e.g. if rag was not created by glia splitting procedure this filtering is required

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
    ssd.save_dataset_deep(n_max_co_processes=global_params.NCORE_TOTAL)  # also executes 'ssd.save_dataset_shallow()'

    exec_skeleton.run_skeleton_generation()

    log.info('Finished SSD initialization. Starting cellular '
             'organelle mapping.')

    # map cellular organelles to SSVs
    # TODO: sort by SSV size (descending)
    ssd_proc.aggregate_segmentation_object_mappings(
        ssd, global_params.existing_cell_organelles)
    ssd_proc.apply_mapping_decisions(
        ssd, global_params.existing_cell_organelles)
    log.info('Finished mapping of cellular organelles to SSVs. '
             'Writing individual SSV graphs.')

    # Write SSV RAGs
    pbar = tqdm.tqdm(total=len(ssd.ssv_ids), mininterval=0.5)
    for ssv in ssd.ssvs:
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
        pbar.update(1)
    pbar.close()
    log.info('Finished saving individual SSV RAGs.')


def run_glia_prediction(e3=False):
    log = initialize_logging('glia_prediction', global_params.config.working_dir + '/logs/',
                             overwrite=False)
    # only append to this key if needed (for e.g. different versions, change accordingly in 'axoness_mapping.py')
    pred_key = "glia_probas"

    # Load initial RAG from  Knossos mergelist text file.
    g = nx.read_edgelist(global_params.config.pruned_rag_path, nodetype=np.uint)
    all_sv_ids_in_rag = np.array(list(g.nodes()), dtype=np.uint)

    log.debug('Found {} CCs with a total of {} SVs in inital RAG.'.format(
        nx.number_connected_components(g), g.number_of_nodes()))
    # chunk them
    sd = SegmentationDataset("sv", working_dir=global_params.config.working_dir)
    multi_params = chunkify(sd.so_dir_paths, global_params.NGPU_TOTAL)
    # get model properties
    if e3 == True:
        model_kwargs = 'get_glia_model_e3'
    else:
        m = get_glia_model()
        model_kwargs = dict(model_path=m._path,
                            normalize_data=m.normalize_data,
                            imposed_batch_size=m.imposed_batch_size,
                            nb_labels=m.nb_labels,
                            channels_to_load=m.channels_to_load)
    # all other kwargs like obj_type='sv' and version are the current SV SegmentationDataset by default
    so_kwargs = dict(working_dir=global_params.config.working_dir)
    # for glia views set woglia to False (because glia are included),
    #  raw_only to True
    pred_kwargs = dict(woglia=False, pred_key=pred_key, verbose=False,
                       raw_only=True)

    multi_params = [[par, model_kwargs, so_kwargs, pred_kwargs] for par in
                    multi_params]
    if e3 is True:
        qu.QSUB_script(multi_params, "predict_sv_views_chunked_e3",
                       n_max_co_processes=2 * global_params.NNODES_TOTAL,
                       script_folder=None, n_cores=global_params.NCORES_PER_NODE,
                       suffix="_glia", additional_flags="--gres=gpu:1",
                       remove_jobfolder=True)
    else:
        # randomly assign to gpu 0 or 1
        for par in multi_params:
            mk = par[1]
            # GPUs are made available for every job via slurm, no need for random assignments: np.random.rand(0, 2)
            mk["init_gpu"] = 0
        _ = qu.QSUB_script(multi_params, "predict_sv_views_chunked",
                           n_max_co_processes=global_params.NNODES_TOTAL,
                           n_cores=global_params.NCORES_PER_NODE, suffix="_glia",
                           additional_flags="--gres=gpu:1", remove_jobfolder=True)
    log.info('Finished glia prediction. Checking completeness.')
    res = find_missing_sv_views(sd, woglia=False, n_cores=global_params.NCORES_PER_NODE)
    missing_not_contained_in_rag = []
    missing_contained_in_rag = []
    for el in res:
        if el not in all_sv_ids_in_rag:
            missing_not_contained_in_rag.append(el)  # TODO: decide whether to use or not
        else:
            missing_contained_in_rag.append(el)
    if len(missing_contained_in_rag) != 0:
        msg = "Not all SVs were predicted! Missing:\n" \
              "{}".format(missing_contained_in_rag)
        log.error(msg)
        raise ValueError(msg)
    else:
        log.info('Success.')


def run_glia_splitting():
    """
    Uses the pruned RAG (stored as edge list .bz2 file) which is computed
     in `init_cell_subcell_sds`.

    Stores neuron RAG at `"{}/glia/neuron_rag{}.bz2".format(global_params.config.working_dir,
    suffix)` which is then used by `run_create_neuron_ssd`

    Returns
    -------

    """
    log = initialize_logging('glia_splitting', global_params.config.working_dir + '/logs/',
                             overwrite=False)
    G = nx.read_edgelist(global_params.config.pruned_rag_path, nodetype=np.uint)
    log.debug('Found {} CCs with a total of {} SVs in inital RAG.'.format(
        nx.number_connected_components(G), G.number_of_nodes()))

    if not os.path.isdir(global_params.config.working_dir + "/glia/"):
        os.makedirs(global_params.config.working_dir + "/glia/")
    transform_rag_edgelist2pkl(G)

    # first perform glia splitting based on multi-view predictions, results are
    # stored at SuperSegmentationDataset ssv_gliaremoval
    qsub_glia_splitting()

    # collect all neuron and glia SVs and store them in numpy array
    collect_glia_sv()

    # # here use reconnected RAG or initial rag
    recon_nx = G
    # create glia / neuron RAGs
    write_glia_rag(recon_nx, global_params.min_cc_size_ssv, suffix=rag_suffix)
    log.info("Finished glia splitting. Resulting neuron and glia RAGs are stored at {}."
             "".format(global_params.config.working_dir + "/glia/"))


def run_glia_rendering(max_n_jobs=None):
    """
    Uses the pruned RAG (stored as edge list .bz2 file) which is computed
     in `init_cell_subcell_sds`.

    Parameters
    ----------
    max_n_jobs :

    Returns
    -------

    """
    if max_n_jobs is None:
        max_n_jobs = global_params.NGPU_TOTAL * 4 if global_params.PYOPENGL_PLATFORM == 'egl' \
            else global_params.NCORE_TOTAL * 4
    log = initialize_logging('glia_view_rendering', global_params.config.working_dir + '/logs/',
                             overwrite=False)
    np.random.seed(0)

    # view rendering prior to glia removal, choose SSD accordingly
    version = "tmp"  # glia removal is based on the initial RAG and does not require explicitly stored SSVs

    G = nx.read_edgelist(global_params.config.pruned_rag_path, nodetype=np.uint)

    cc_gs = list(nx.connected_component_subgraphs(G))
    all_sv_ids_in_rag = np.array(list(G.nodes()), dtype=np.uint)

    # write out readable format for 'glia_prediction.py'
    ccs = [[n for n in cc] for cc in cc_gs]
    kml = knossos_ml_from_ccs([np.sort(cc)[0] for cc in ccs], ccs)
    with open(global_params.config.working_dir + "initial_rag.txt", 'w') as f:
        f.write(kml)

    # generate parameter for view rendering of individual SSV
    log.info("Starting view rendering.")
    multi_params = cc_gs
    big_ssv = []
    small_ssv = []
    for g in multi_params:
        if g.number_of_nodes() > RENDERING_MAX_NB_SV:
            big_ssv.append(g)
        else:
            small_ssv.append(g)

    # # identify huge SSVs and process them individually on whole cluster
    # nb_svs = np.array([g.number_of_nodes() for g in multi_params])
    # big_ssv = multi_params[nb_svs > RENDERING_MAX_NB_SV]

    for kk, g in enumerate(big_ssv[::-1]):
        # Create SSV object
        sv_ixs = np.sort(list(g.nodes()))
        log.info("Processing SSV [{}/{}] with {} SVs on whole cluster.".format(
            kk+1, len(big_ssv), len(sv_ixs)))
        sso = SuperSegmentationObject(sv_ixs[0], working_dir=global_params.config.working_dir,
                                      version=version, create=False, sv_ids=sv_ixs)
        # nodes of sso._rag need to be SV
        new_G = nx.Graph()
        for e in g.edges():
            new_G.add_edge(sso.get_seg_obj("sv", e[0]),
                           sso.get_seg_obj("sv", e[1]))
        sso._rag = new_G
        sso.render_views(add_cellobjects=False, cellobjects_only=False,
                         skip_indexviews=True, woglia=False, overwrite=True,
                         qsub_co_jobs=global_params.NCORE_TOTAL)

    # render small SSV without overhead and single cpus on whole cluster
    multi_params = small_ssv
    np.random.shuffle(multi_params)
    multi_params = chunkify(multi_params, max_n_jobs)

    # list of SSV IDs and SSD parameters need to be given to a single QSUB job
    multi_params = [(ixs, global_params.config.working_dir, version) for ixs in multi_params]
    _ = qu.QSUB_script(multi_params, "render_views_glia_removal",
                                 n_max_co_processes=global_params.NGPU_TOTAL,
                                 n_cores=global_params.NCORES_PER_NODE // global_params.NGPUS_PER_NODE,
                                 additional_flags="--gres=gpu:1",
                                 remove_jobfolder=True)

    # check completeness
    log.info('Finished view rendering for glia separation. Checking completeness.')
    sd = SegmentationDataset("sv", working_dir=global_params.config.working_dir)
    res = find_missing_sv_views(sd, woglia=False, n_cores=global_params.NCORES_PER_NODE)
    missing_not_contained_in_rag = []
    missing_contained_in_rag = []
    for el in res:
        if el not in all_sv_ids_in_rag:
            missing_not_contained_in_rag.append(el)  # TODO: decide whether to use or not
        else:
            missing_contained_in_rag.append(el)
    if len(missing_contained_in_rag) != 0:
        msg = "Not all SVs were rendered completely! Missing:\n" \
              "{}".format(missing_contained_in_rag)
        log.error(msg)
        raise ValueError(msg)
    else:
        log.info('All SVs now contain views required for glia prediction.')


def axoness_pred_exists(sv):
    sv.load_attr_dict()
    return 'axoness_probas' in sv.attr_dict
