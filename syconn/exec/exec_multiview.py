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
import time
import shutil
from typing import Optional
from multiprocessing import Queue, Process

from syconn.reps.segmentation_helper import find_missing_sv_views
from syconn.reps.super_segmentation import SuperSegmentationObject
from syconn.proc.glia_splitting import qsub_glia_splitting, collect_glia_sv, \
    write_glia_rag, transform_rag_edgelist2pkl
from syconn.reps.segmentation import SegmentationDataset
from syconn.handler.prediction import get_glia_model
from syconn.proc.graphs import create_ccsize_dict
from syconn.proc.rendering import render_sso_coords_multiprocessing
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


def run_morphology_embedding(max_n_jobs: Optional[int] = None):
    """
    Infer local morphology embeddings for all neuron reconstructions base on
    triplet-loss trained cellular morphology learning network (tCMN).

    Args:
        max_n_jobs: Number of parallel jobs.

    Notes:
        Requires :func:`~run_create_neuron_ssd`, :func:`~run_neuron_rendering` and
        :func:`~syconn.exec.skeleton.run_skeleton_generation`.
    """
    if max_n_jobs is None:
        max_n_jobs = global_params.config.ngpu_total * 2
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
    qu.QSUB_script(multi_params, "generate_morphology_embedding",
                   n_max_co_processes=global_params.config.ngpu_total,
                   n_cores=global_params.config['ncores_per_node'] // global_params.config['ngpus_per_node'],
                   log=log, suffix="", additional_flags="--gres=gpu:1",
                   remove_jobfolder=True)
    log.info('Finished extraction of cell morphology embedding.')


def run_axoness_mapping(max_n_jobs: Optional[int] = None):
    """
    Map ``img2scalar`` CMN results of the 2D projections onto the cell
    reconstruction mesh. See :func:`~run_semsegaxoness_mapping` for the
    semantic segmentation approach.

    Args:
        max_n_jobs: Number of parallel jobs.

    Notes:
        Requires :func:`~run_create_neuron_ssd`, :func:`~run_neuron_rendering`,
        :func:`run_axoness_prediction` and
        :func:`~syconn.exec.skeleton.run_skeleton_generation`.
    """
    if max_n_jobs is None:
        max_n_jobs = global_params.config.ncore_total * 2
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
    _ = qu.QSUB_script(multi_params, "map_viewaxoness2skel", log=log,
                       n_max_co_processes=global_params.config.ncore_total,
                       suffix="", n_cores=1, remove_jobfolder=True)
    # TODO: perform completeness check
    log.info('Finished axoness mapping.')


def run_axoness_prediction(max_n_jobs_gpu: Optional[int] = None,
                           e3: bool = True):
    """
    Run the axoness inference based on the ``img2scalar`` CMN. See
    :func:`~run_semsegaxoness_prediction` for the semantic segmentation model.

    Args:
        max_n_jobs_gpu: Number of parallel jobs.
        e3: If True, use elektronn3 models.

    Notes:
        Requires :func:`~run_create_neuron_ssd`, :func:`~run_neuron_rendering` and
        :func:`~syconn.exec.skeleton.run_skeleton_generation`.
    """
    log = initialize_logging('axon_prediction', global_params.config.working_dir + '/logs/',
                             overwrite=False)
    if max_n_jobs_gpu is None:
        max_n_jobs_gpu = global_params.config.ngpu_total * 2
    # here because all qsub jobs will start a script referring to
    # 'global_params.config.working_dir'
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
    
    # all other kwargs like obj_type='sv' and version are the current SV
    # SegmentationDataset by default
    so_kwargs = dict(working_dir=global_params.config.working_dir)
    # for axoness views set woglia to True (because glia were removed beforehand),
    #  raw_only to False
    pred_kwargs = dict(woglia=True, pred_key=pred_key, verbose=False,
                       raw_only=False)
    multi_params = [[par, model_kwargs, so_kwargs, pred_kwargs] for
                    par in multi_params]

    if e3 is True:
        # TODO: using two GPUs on a single node seems to be error-prone
        #  -> wb13 froze when processing example_cube=2
        n_cores = global_params.config['ncores_per_node'] // global_params.config['ngpus_per_node']
        _ = qu.QSUB_script(multi_params, "predict_sv_views_chunked_e3", log=log,
                           n_max_co_processes=global_params.config.ngpu_total,
                           n_cores=n_cores,
                           suffix="_axoness", additional_flags="--gres=gpu:1",
                           remove_jobfolder=True)
    else:
        for par in multi_params:
            mk = par[1]
            # SLURM is GPU aware, no need for random assignments.
            mk["init_gpu"] = 0  # np.random.rand(0, 2)
        _ = qu.QSUB_script(multi_params, "predict_sv_views_chunked", log=log,
                           n_max_co_processes=global_params.config.ngpu_total // 2,
                           n_cores=global_params.config['ncores_per_node'], suffix="_axoness",
                           additional_flags="--gres=gpu:1",
                           remove_jobfolder=True)
    log.info('Finished axon prediction. Now checking for missing predictions.')
    res = find_missing_sv_attributes_in_ssv(ssd, pred_key, n_cores=global_params.config['ncores_per_node'])
    if len(res) > 0:
        log.error("Attribute '{}' missing for follwing"
                  " SVs:\n{}".format(pred_key, res))
    else:
        log.info('Success.')


def run_celltype_prediction(max_n_jobs_gpu: Optional[int] = None):
    """
    Run the celltype inference based on the ``img2scalar`` CMN.

    Args:
        max_n_jobs_gpu: Number of parallel GPU jobs.

    Notes:
        Requires :func:`~run_create_neuron_ssd` and :func:`~run_neuron_rendering`.
    """
    if max_n_jobs_gpu is None:
        max_n_jobs_gpu = global_params.config.ngpu_total * 2
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

    path_to_out = qu.QSUB_script(multi_params, "predict_cell_type", log=log,
                                 n_max_co_processes=global_params.config['nnodes_total'],
                                 suffix="", additional_flags="--gres=gpu:1",
                                 n_cores=global_params.config['ncores_per_node'] // global_params.config['ngpus_per_node'],
                                 remove_jobfolder=True)
    log.info('Finished prediction of {} SSVs. Checking completeness.'
             ''.format(len(ordering)))
    out_files = glob.glob(path_to_out + "*.pkl")
    err = []
    for fp in out_files:
        with open(fp, "rb") as f:
            local_err = pkl.load(f)
        err += list(local_err)
    if len(err) > 0:
        msg = "{} errors occurred for SSVs with ID: " \
              "{}".format(len(err), [el[0] for el in err])
        log.error(msg)
        raise ValueError(msg)
    else:
        log.info('Success.')


def run_semsegaxoness_mapping(max_n_jobs: Optional[int] = None):
    """
    Map semantic segmentation results of the 2D projections onto the cell
    reconstruction mesh.
    Generates the following attributes by default in
    :py:attr:`~syconn.reps.super_segmentation_object.SuperSegmentationObject.skeleton`:
        * "axoness": Vertex predictions mapped to skeleton (see
          ``global_params.config['compartments']['map_properties_semsegax']``.
        * "axoness_avg10000": Sliding window average along skeleton (10um traversal length).
        * "axoness_avg10000_comp_maj": Majority vote on connected components after removing the
          soma.

    Args:
        max_n_jobs: Number of parallel jobs.

    Notes:
        Requires :func:`~run_create_neuron_ssd`, :func:`~run_neuron_rendering`,
        :func:`~run_semsegaxoness_prediction` and
        :func:`~syconn.exec.skeleton.run_skeleton_generation`.
    """
    if max_n_jobs is None:
        max_n_jobs = global_params.config.ncore_total * 2
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
    _ = qu.QSUB_script(multi_params, "map_semsegaxoness2skel", log=log,
                       n_max_co_processes=global_params.config.ncore_total,
                       suffix="", n_cores=1, remove_jobfolder=True)
    # TODO: perform completeness check
    log.info('Finished axoness mapping.')


def run_semsegaxoness_prediction(max_n_jobs_gpu: Optional[int] = None):
    """
    Will store semantic axoness labels as ``view_properties_semsegax['semseg_key']`` inside
    ``ssv.label_dict('vertex')``.

    Todo:
        * run rendering chunk-wise instead of on-the-fly and then perform
          prediction chunk-wise as well, adopt from spiness step.

    Args:
        max_n_jobs_gpu: Number of parallel GPU jobs.

    Returns:

    """
    if max_n_jobs_gpu is None:
        max_n_jobs_gpu = global_params.config.ngpu_total * 2
    log = initialize_logging('axoness_prediction', global_params.config.working_dir+ '/logs/',
                             overwrite=False)
    ssd = SuperSegmentationDataset(working_dir=global_params.config.working_dir)
    # shuffle SV IDs
    np.random.seed(0)

    log.info('Starting axoness prediction.')
    nb_svs_per_ssv = np.array([len(ssd.mapping_dict[ssv_id])
                               for ssv_id in ssd.ssv_ids])
    multi_params = ssd.ssv_ids
    ordering = np.argsort(nb_svs_per_ssv)
    multi_params = multi_params[ordering[::-1]]
    max_n_jobs_gpu = np.max([max_n_jobs_gpu, len(multi_params) // 100])  # at most 100 SSV per job
    multi_params = chunkify(multi_params, max_n_jobs_gpu)
    # job parameter will be read sequentially, i.e. in order to provide only
    # one list as parameter one needs an additonal axis
    multi_params = [(ixs, ) for ixs in multi_params]

    # if not 'example' in global_params.config.working_dir:
    n_cores = global_params.config['ncores_per_node'] // global_params.config['ngpus_per_node']
    # else:
    #     n_cores = global_params.config['ncores_per_node']
    path_to_out = qu.QSUB_script(multi_params, "predict_axoness_semseg", log=log,
                                 n_max_co_processes=global_params.config.ngpu_total,
                                 suffix="", additional_flags="--gres=gpu:1",
                                 n_cores=n_cores,
                                 remove_jobfolder=False)
    log.info('Finished prediction of {} SSVs. Checking completeness.'
             ''.format(len(ordering)))
    out_files = glob.glob(path_to_out + "*.pkl")
    err = []
    for fp in out_files:
        with open(fp, "rb") as f:
            local_err = pkl.load(f)
        if local_err is not None:
            err += list(local_err)
    if len(err) > 0:
        msg = "{} errors occurred for SSVs with ID: " \
              "{}".format(len(err), [el[0] for el in err])
        log.error(msg)
        raise ValueError(msg)
    else:
        log.info('Success.')
    shutil.rmtree(os.path.abspath(path_to_out + "/../"), ignore_errors=True)


def run_spiness_prediction(max_n_jobs_gpu: Optional[int] = None,
                           max_n_jobs: Optional[int] = None):
    """
    Will store semantic spine labels inside``ssv.label_dict('vertex')['spiness]``.

    Todo:
        * run rendering chunk-wise instead of on-the-fly and then perform
          prediction chunk-wise as well, adopt from spiness step.

    Args:
        max_n_jobs_gpu: Number of parallel GPU jobs. Used for the inference.
        max_n_jobs : Number of parallel CPU jobs. Used for the mapping step.
    """
    if max_n_jobs is None:
        max_n_jobs = global_params.config.ncore_total * 2
    if max_n_jobs_gpu is None:
        max_n_jobs_gpu = global_params.config.ngpu_total * 2
    log = initialize_logging('spine_identification', global_params.config.working_dir
                             + '/logs/', overwrite=False)
    ssd = SuperSegmentationDataset(working_dir=global_params.config.working_dir)

    # run semantic spine segmentation on multi views
    sd = ssd.get_segmentationdataset("sv")
    # chunk them
    multi_params = chunkify(sd.so_dir_paths, max_n_jobs_gpu)
    # set model properties
    model_kwargs = dict(src=global_params.config.mpath_spiness,
                        multi_gpu=False)
    so_kwargs = dict(working_dir=global_params.config.working_dir)
    pred_kwargs = dict(pred_key=global_params.config['spines']['semseg2mesh_spines']['semseg_key'])
    multi_params = [[par, model_kwargs, so_kwargs, pred_kwargs]
                    for par in multi_params]
    log.info('Starting spine prediction.')
    qu.QSUB_script(multi_params, "predict_spiness_chunked", log=log,
                   n_max_co_processes=global_params.config.ngpu_total,
                   n_cores=global_params.config['ncores_per_node'] // global_params.config['ngpus_per_node'],
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
    kwargs_semseg2mesh = global_params.config['spines']['semseg2mesh_spines']
    kwargs_semsegforcoords = global_params.config['spines']['semseg2coords_spines']
    multi_params = [(ssv_ids, ssd.version, ssd.version_dict, ssd.working_dir,
                     kwargs_semseg2mesh, kwargs_semsegforcoords) for ssv_ids in multi_params]
    log.info('Started mapping of spine predictions to neurite surfaces.')
    qu.QSUB_script(multi_params, "map_spiness", n_max_co_processes=global_params.config.ncore_total,
                   n_cores=4, suffix="", additional_flags="", remove_jobfolder=True, log=log)
    log.info('Finished spine mapping.')


def _run_neuron_rendering_small_helper(max_n_jobs: Optional[int] = None):
    """
    Render the default views as defined in ``global_params`` [WIP] of small
    neuron reconstructions. Helper method of :func:`~run_neuron_rendering`.

    Args:
        max_n_jobs: Number of parallel jobs.

    Notes:
        Requires :func:`~run_create_neuron_ssd`.
    """

    if max_n_jobs is None:
        max_n_jobs = global_params.config.ngpu_total * 4 if \
            global_params.config['pyopengl_platform'] == 'egl' \
            else global_params.config.ncore_total * 4
    log = initialize_logging('neuron_view_rendering_small',
                             global_params.config.working_dir + '/logs/')
    # view rendering prior to glia removal, choose SSD accordingly
    ssd = SuperSegmentationDataset(working_dir=global_params.config.working_dir)

    #  TODO: use actual size criteria, e.g. number of sampling locations
    nb_svs_per_ssv = np.array([len(ssd.mapping_dict[ssv_id])
                               for ssv_id in ssd.ssv_ids])

    # render normal size SSVs
    size_mask = nb_svs_per_ssv <= global_params.config['glia']['rendering_max_nb_sv']
    if 'example' in global_params.config.working_dir and np.sum(~size_mask) == 0:
        # generate at least one (artificial) huge SSV
        size_mask[:1] = False
        size_mask[1:] = True

    multi_params = ssd.ssv_ids[size_mask]
    # sort ssv ids according to their number of SVs (descending)
    ordering = np.argsort(nb_svs_per_ssv[size_mask])
    multi_params = multi_params[ordering[::-1]]
    multi_params = chunkify(multi_params, max_n_jobs)
    # list of SSV IDs and SSD parameters need to be given to a single QSUB job
    multi_params = [(ixs, global_params.config.working_dir) for ixs in multi_params]
    log.info('Started rendering of {} SSVs. '.format(np.sum(size_mask)))

    if global_params.config['pyopengl_platform'] == 'osmesa':  # utilize all CPUs
        qu.QSUB_script(multi_params, "render_views", log=log, suffix='_small',
                       n_max_co_processes=global_params.config.ncore_total,
                       remove_jobfolder=False)
    elif global_params.config['pyopengl_platform'] == 'egl':  # utilize 1 GPU per task
        # run EGL on single node: 20 parallel jobs
        if not qu.batchjob_enabled():
            n_cores = 1
            n_parallel_jobs = global_params.config['ncores_per_node']
            qu.QSUB_script(multi_params, "render_views", suffix='_small',
                           n_max_co_processes=n_parallel_jobs, log=log,
                           additional_flags="--gres=gpu:2", disable_batchjob=True,
                           n_cores=n_cores, remove_jobfolder=True)
        # run on whole cluster
        else:
            n_cores = global_params.config['ncores_per_node'] // global_params.config['ngpus_per_node']
            n_parallel_jobs = global_params.config.ngpu_total
            qu.QSUB_script(multi_params, "render_views_egl", suffix='_small',
                           n_max_co_processes=n_parallel_jobs, log=log,
                           additional_flags="--gres=gpu:1",
                           n_cores=n_cores, remove_jobfolder=True)
    else:
        raise RuntimeError('Specified OpenGL platform "{}" not supported.'
                           ''.format(global_params.config['pyopengl_platform']))
    log.info('Finished rendering of {}/{} SSVs.'.format(len(ordering),
                                                        len(nb_svs_per_ssv)))


def _run_neuron_rendering_big_helper(max_n_jobs: Optional[int] = None):
    """
    Render the default views as defined in ``global_params`` [WIP] of huge
    neuron reconstructions. Helper method of :func:`~run_neuron_rendering`.

    Args:
        max_n_jobs: Number of parallel jobs.

    Notes:
        Requires :func:`~run_create_neuron_ssd`.
    """
    if max_n_jobs is None:
        max_n_jobs = global_params.config['nnodes_total'] * 2
    log = initialize_logging('neuron_view_rendering_big',
                             global_params.config.working_dir + '/logs/')
    # view rendering prior to glia removal, choose SSD accordingly
    ssd = SuperSegmentationDataset(working_dir=global_params.config.working_dir)

    #  TODO: use actual size criteria, e.g. number of sampling locations
    nb_svs_per_ssv = np.array([len(ssd.mapping_dict[ssv_id])
                               for ssv_id in ssd.ssv_ids])

    # render normal size SSVs
    size_mask = nb_svs_per_ssv <= global_params.config['glia']['rendering_max_nb_sv']
    if 'example' in global_params.config.working_dir and np.sum(~size_mask) == 0:
        # generate at least one (artificial) huge SSV
        size_mask[:1] = False
        size_mask[1:] = True
    # sort ssv ids according to their number of SVs (descending)
    # list of SSV IDs and SSD parameters need to be given to a single QSUB job
    if np.sum(~size_mask) > 0:
        log.info('{} huge SSVs will be rendered on the cluster.'.format(np.sum(~size_mask)))
        # identify huge SSVs and process them individually on whole cluster
        big_ssv = ssd.ssv_ids[~size_mask]

        # # TODO: Currently high memory consumption when rendering index views! take into account
        # #  when multiprocessing
        # # TODO: refactor `render_sso_coords_multiprocessing` and then use `QSUB_render_views_egl`
        # #  here!
        # render normal views only
        n_cores = global_params.config['ncores_per_node'] // global_params.config['ngpus_per_node']
        n_parallel_jobs = global_params.config.ngpu_total
        render_kwargs = dict(add_cellobjects=True, woglia=True, overwrite=True,
                             skip_indexviews=True)
        sso_kwargs = dict(working_dir=global_params.config.working_dir, nb_cpus=n_cores,
                          enable_locking_so=False, enable_locking=False)

        # sort ssv ids according to their number of SVs (descending)
        ordering = np.argsort(nb_svs_per_ssv[~size_mask])
        multi_params = big_ssv[ordering[::-1]]
        multi_params = chunkify(multi_params, max_n_jobs)
        # list of SSV IDs and SSD parameters need to be given to a single QSUB job
        multi_params = [(ixs, sso_kwargs, render_kwargs) for ixs in multi_params]
        qu.QSUB_script(multi_params, "render_views", suffix='_big',
                       n_max_co_processes=n_parallel_jobs, log=log,
                       additional_flags="--gres=gpu:1",
                       n_cores=n_cores, remove_jobfolder=True)
        # # render index-views only
        for ssv_id in big_ssv:
            ssv = SuperSegmentationObject(ssv_id, working_dir=global_params.config.working_dir)
            render_sso_coords_multiprocessing(ssv, global_params.config.working_dir, verbose=True,
                                              return_views=False, disable_batchjob=False,
                                              n_jobs=n_parallel_jobs, n_cores=n_cores,
                                              render_indexviews=True)
        log.info('Finished rendering of {}/{} SSVs.'.format(len(big_ssv),
                                                            len(nb_svs_per_ssv)))


def run_neuron_rendering(max_n_jobs: Optional[int] = None):
    """
    Render the default views as defined in ``global_params`` [WIP].

    Args:
        max_n_jobs: Number of parallel jobs.

    Notes:
        Requires :func:`~run_create_neuron_ssd`.
    """
    log = initialize_logging('neuron_view_rendering',
                             global_params.config.working_dir + '/logs/')
    ps = [Process(target=_run_neuron_rendering_big_helper, args=(max_n_jobs, )),
          Process(target=_run_neuron_rendering_small_helper, args=(max_n_jobs, ))]
    for p in ps:
        p.start()
        time.sleep(10)
    for p in ps:
        p.join()
        if p.exitcode != 0:
            raise Exception(f'Worker {p.name} stopped unexpectedly with exit '
                            f'code {p.exitcode}.')
    log.info('Finished rendering of all SSVs. Checking completeness.')
    ssd = SuperSegmentationDataset(working_dir=global_params.config.working_dir)
    res = find_incomplete_ssv_views(ssd, woglia=True, n_cores=global_params.config['ncores_per_node'])
    if len(res) != 0:
        msg = "Not all SSVs were rendered! {}/{} missing:\n" \
              "{}".format(len(res), len(ssd.ssv_ids),
                          res[:10])
        log.error(msg)
        raise RuntimeError(msg)
    log.info('Success.')


def run_create_neuron_ssd():
    """
    Creates a :class:`~syconn.reps.super_segmentation_dataset.SuperSegmentationDataset` with
    ``version=0`` at the currently active working directory based on the RAG
    at ``/glia/neuron_rag.bz2``. In case glia splitting is active, this will be
    the RAG after glia removal, if it was disabled it is identical to ``pruned_rag.bz2``.

    Notes:
        Requires :func:`~syconn.exec_init.init_cell_subcell_sds` and
        optionally :func:`~run_glia_splitting`.
    """
    log = initialize_logging('create_neuron_ssd', global_params.config.working_dir + '/logs/',
                             overwrite=False)
    g_p = "{}/glia/neuron_rag.bz2".format(global_params.config.working_dir)
    rag_g = nx.read_edgelist(g_p, nodetype=np.uint)

    # e.g. if rag was not created by glia splitting procedure this filtering is required
    if not global_params.config.prior_glia_removal:
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
    ssd.save_dataset_deep(n_max_co_processes=global_params.config.ncore_total)

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

    exec_skeleton.run_skeleton_generation()

    log.info('Finished SSD initialization. Starting cellular '
             'organelle mapping.')

    # map cellular organelles to SSVs
    # TODO: sort by SSV size (descending)
    ssd_proc.aggregate_segmentation_object_mappings(
        ssd, global_params.config['existing_cell_organelles'])
    ssd_proc.apply_mapping_decisions(
        ssd, global_params.config['existing_cell_organelles'])
    log.info('Finished mapping of cellular organelles to SSVs. '
             'Writing individual SSV graphs.')


def run_glia_prediction(e3: bool = False):
    """
    Predict glia supervoxels based on the ``img2scalar`` CMN.

    Args:
        e3: If True, use elektronn3 models.

    Notes:
        Requires :func:`~syconn.exec_init.init_cell_subcell_sds` and
        :func:`~run_glia_rendering`.
    """
    log = initialize_logging('glia_prediction', global_params.config.working_dir + '/logs/',
                             overwrite=False)
    # only append to this key if needed (for e.g. different versions,
    # change accordingly in 'axoness_mapping.py')
    pred_key = "glia_probas"

    # Load initial RAG from  Knossos mergelist text file.
    g = nx.read_edgelist(global_params.config.pruned_rag_path, nodetype=np.uint)
    all_sv_ids_in_rag = np.array(list(g.nodes()), dtype=np.uint)

    log.debug('Found {} CCs with a total of {} SVs in inital RAG.'.format(
        nx.number_connected_components(g), g.number_of_nodes()))
    # chunk them
    sd = SegmentationDataset("sv", working_dir=global_params.config.working_dir)
    multi_params = chunkify(sd.so_dir_paths, global_params.config.ngpu_total * 2)
    # get model properties
    if e3 is True:
        model_kwargs = 'get_glia_model_e3'
    else:
        m = get_glia_model()
        model_kwargs = dict(model_path=m._path,
                            normalize_data=m.normalize_data,
                            imposed_batch_size=m.imposed_batch_size,
                            nb_labels=m.nb_labels,
                            channels_to_load=m.channels_to_load)
    # all other kwargs like obj_type='sv' and version are the current SV
    # SegmentationDataset by default
    so_kwargs = dict(working_dir=global_params.config.working_dir)
    # for glia views set woglia to False (because glia are included),
    #  raw_only to True
    pred_kwargs = dict(woglia=False, pred_key=pred_key, verbose=False,
                       raw_only=True)

    multi_params = [[par, model_kwargs, so_kwargs, pred_kwargs] for par in
                    multi_params]
    if e3 is True:
        # TODO: using two GPUs on a single node seems to be error-prone
        #  -> wb13 froze when processing example_cube=2
        n_cores = global_params.config['ncores_per_node'] // global_params.config['ngpus_per_node']
        qu.QSUB_script(multi_params, "predict_sv_views_chunked_e3", log=log,
                       n_max_co_processes=global_params.config.ngpu_total,
                       script_folder=None, n_cores=n_cores,
                       suffix="_glia", additional_flags="--gres=gpu:1",
                       remove_jobfolder=True)
    else:
        # randomly assign to gpu 0 or 1
        for par in multi_params:
            mk = par[1]
            # GPUs are made available for every job via slurm,
            # no need for random assignments: np.random.rand(0, 2)
            mk["init_gpu"] = 0
        _ = qu.QSUB_script(multi_params, "predict_sv_views_chunked", log=log,
                           n_max_co_processes=global_params.config.ngpu_total,
                           n_cores=global_params.config['ncores_per_node'] // global_params.config['ngpus_per_node'],
                           suffix="_glia",
                           additional_flags="--gres=gpu:1", remove_jobfolder=True)
    log.info('Finished glia prediction. Checking completeness.')
    res = find_missing_sv_views(sd, woglia=False, n_cores=global_params.config['ncores_per_node'])
    missing_not_contained_in_rag = []
    missing_contained_in_rag = []
    for el in res:
        if el not in all_sv_ids_in_rag:
            missing_not_contained_in_rag.append(el)  # TODO: decide whether to use or not
        else:
            missing_contained_in_rag.append(el)
    if len(missing_contained_in_rag) != 0:
        msg = "Not all SVs were predicted! {}/{} missing:\n" \
              "{}".format(len(missing_contained_in_rag), len(all_sv_ids_in_rag),
                          missing_contained_in_rag[:100])
        log.error(msg)
        raise ValueError(msg)
    else:
        log.info('Success.')


def run_glia_splitting():
    """
    Uses the pruned RAG at ``global_params.config.pruned_rag_path`` (stored as edge list .bz2 file)
    which is  computed in :func:`~syconn.exec.exec_init.init_cell_subcell_sds` to split glia
    fragments from neuron reconstructions and separate those and entire glial cells from
    the neuron supervoxel graph.

    Stores neuron RAG at ``"{}/glia/neuron_rag{}.bz2".format(global_params.config.working_dir,
    suffix)`` which is then used by :func:`~run_create_neuron_ssd`.

    Notes:
        Requires :func:`~syconn.exec_init.init_cell_subcell_sds`,
        :func:`~run_glia_rendering` and :func:`~run_glia_prediction`.
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
    write_glia_rag(recon_nx, global_params.config['glia']['min_cc_size_ssv'])
    log.info("Finished glia splitting. Resulting neuron and glia RAGs are stored at {}."
             "".format(global_params.config.working_dir + "/glia/"))


def _run_huge_ssv_render_worker(q: Queue, q_out: Queue):
    """
    Helper method of :func:`~run_glia_rendering`.

    Args:
        q: Input queue.
        q_out: Output queue.

    """
    while True:
        inp = q.get()
        if inp == -1:
            break
        kk, g, version = inp
        # Create SSV object
        sv_ixs = np.sort(list(g.nodes()))
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
                         qsub_co_jobs=global_params.config.ngpu_total)
        q_out.put(0)


def run_glia_rendering(max_n_jobs: Optional[int] = None):
    """
    Uses the pruned RAG at ``global_params.config.pruned_rag_path``
    (stored as edge list .bz2 file) which is computed in
    :func:`~syconn.exec.exec_init.init_cell_subcell_sds` to aggregate the
    rendering context from the underlying supervoxel graph.

    Args:
        max_n_jobs: Number of parallel jobs.
    """
    if max_n_jobs is None:
        max_n_jobs = global_params.config.ngpu_total * 4 if \
            global_params.config['pyopengl_platform'] == 'egl' \
            else global_params.config.ncore_total * 4
    log = initialize_logging('glia_view_rendering', global_params.config.working_dir + '/logs/',
                             overwrite=True)
    log.info("Preparing RAG.")
    np.random.seed(0)

    # view rendering prior to glia removal, choose SSD accordingly
    # glia removal is based on the initial RAG and does not require explicitly stored SSVs
    # TODO: refactor how splits are stored, currently those are stored at ssv_tmp
    version = "tmp"

    G = nx.read_edgelist(global_params.config.pruned_rag_path, nodetype=np.uint)

    cc_gs = sorted(list(nx.connected_component_subgraphs(G)), key=len, reverse=True)
    all_sv_ids_in_rag = np.array(list(G.nodes()), dtype=np.uint)

    # generate parameter for view rendering of individual SSV
    # TODO: remove SVs below minimum size (-> global_params.config['glia']['min_cc_size_ssv'])
    sds = SegmentationDataset("sv", working_dir=global_params.config.working_dir)
    sv_size_dict = {}
    bbs = sds.load_cached_data('bounding_box') * sds.scaling
    for ii in range(len(sds.ids)):
        sv_size_dict[sds.ids[ii]] = bbs[ii]
    ccsize_dict = create_ccsize_dict(cc_gs, sv_size_dict,
                                     is_connected_components=True)

    multi_params = cc_gs
    big_ssv = []
    small_ssv = []
    for g in multi_params:
        if g.number_of_nodes() > global_params.config['glia']['rendering_max_nb_sv']:
            big_ssv.append(g)
        elif ccsize_dict[list(g.nodes())[0]] < global_params.config['glia']['min_cc_size_ssv']:
            pass  # ignore this CC
        else:
            small_ssv.append(g)

    log.info("View rendering for glia separation started.")
    # # identify huge SSVs and process them on the entire cluster
    if len(big_ssv) > 0:
        n_threads = 2
        log.info("Processing {} huge SSVs in {} threads on the entire cluster"
                 ".".format(len(big_ssv), n_threads))
        q_in = Queue()
        q_out = Queue()
        for kk, g in enumerate(big_ssv):
            q_in.put((kk, g, version))
        for _ in range(n_threads):
            q_in.put(-1)
        ps = [Process(target=_run_huge_ssv_render_worker, args=(q_in, q_out))
              for _ in range(n_threads)]
        for p in ps:
            p.start()
            time.sleep(0.5)
        q_in.close()
        q_in.join_thread()
        for p in ps:
            p.join()
            if p.exitcode != 0:
                raise Exception(f'Worker {p.name} stopped unexpectedly with exit '
                                f'code {p.exitcode}.')
        if q_out.qsize() != len(big_ssv):
            msg = 'Not all `_run_huge_ssv_render_worker` jobs completed ' \
                  'successfully.'
            log.error(msg)
            raise ValueError(msg)
    # render small SSV without overhead and single cpus on whole cluster
    multi_params = small_ssv
    np.random.shuffle(multi_params)
    multi_params = chunkify(multi_params, max_n_jobs)
    # list of SSV IDs and SSD parameters need to be given to a single QSUB job
    multi_params = [(ixs, global_params.config.working_dir, version) for ixs in multi_params]
    _ = qu.QSUB_script(multi_params, "render_views_glia_removal", log=log,
                       n_max_co_processes=global_params.config.ngpu_total,
                       n_cores=global_params.config['ncores_per_node'] // global_params.config['ngpus_per_node'],
                       additional_flags="--gres=gpu:1", remove_jobfolder=True)

    # check completeness
    log.info('Finished view rendering for glia separation. Checking completeness.')
    sd = SegmentationDataset("sv", working_dir=global_params.config.working_dir)
    res = find_missing_sv_views(sd, woglia=False, n_cores=global_params.config['ncores_per_node'])
    missing_not_contained_in_rag = []
    missing_contained_in_rag = []
    for el in res:
        if el not in all_sv_ids_in_rag:
            missing_not_contained_in_rag.append(el)  # TODO: decide whether to use or not
        else:
            missing_contained_in_rag.append(el)
    if len(missing_contained_in_rag) != 0:
        msg = "Not all SVs were rendered completely! {}/{} missing:\n" \
              "{}".format(len(missing_contained_in_rag), len(all_sv_ids_in_rag),
                          missing_contained_in_rag[:100])
        log.error(msg)
        raise ValueError(msg)
    else:
        log.info('All SVs now contain views required for glia prediction.')
    # TODO: remove temporary SSV datasets
