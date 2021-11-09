# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max-Planck-Institute of Neurobiology, Munich, Germany
# Authors: Philipp Schubert

import time
from multiprocessing import Queue, Process
from typing import Optional

import networkx as nx
import numpy as np

from syconn import global_params
from syconn.handler.basics import chunkify
from syconn.handler.config import initialize_logging
from syconn.mp import batchjob_utils as qu
from syconn.proc.graphs import create_ccsize_dict
from syconn.reps.segmentation import SegmentationDataset
from syconn.reps.segmentation_helper import find_missing_sv_views
from syconn.reps.super_segmentation import SuperSegmentationDataset
from syconn.reps.super_segmentation import SuperSegmentationObject
from syconn.reps.super_segmentation_helper import find_incomplete_ssv_views


def _run_neuron_rendering_small_helper(max_n_jobs: Optional[int] = None):
    """
    Render the default views as defined in ``global_params`` [WIP] of small
    neuron reconstructions. Helper method of :func:`~run_neuron_rendering`.

    Args:
        max_n_jobs: Number of parallel jobs.

    Notes:
        Requires :func:`~syconn.exec.exec_init.run_create_neuron_ssd`.
    """

    if max_n_jobs is None:
        max_n_jobs = global_params.config.ngpu_total * 10 if \
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
    ordering = np.argsort(ssd.load_numpy_data('size')[size_mask])
    multi_params = multi_params[ordering[::-1]]
    multi_params = chunkify(multi_params, max_n_jobs)
    # list of SSV IDs and SSD parameters need to be given to a single QSUB job
    multi_params = [(ixs, global_params.config.working_dir) for ixs in multi_params]

    log.info('Started rendering of {} SSVs. '.format(np.sum(size_mask)))

    if global_params.config['pyopengl_platform'] == 'osmesa':  # utilize all CPUs
        qu.batchjob_script(multi_params, "render_views", log=log, suffix='_small',
                           remove_jobfolder=True)
    elif global_params.config['pyopengl_platform'] == 'egl':  # utilize 1 GPU per task
        # run EGL on single node: 20 parallel jobs
        if not qu.batchjob_enabled():
            n_cores = 1
            n_parallel_jobs = global_params.config['ncores_per_node']
            qu.batchjob_script(multi_params, "render_views", suffix='_small', log=log,
                               additional_flags="--gres=gpu:2", disable_batchjob=True,
                               n_cores=n_cores, remove_jobfolder=True)
        # run on whole cluster
        else:
            n_cores = global_params.config['ncores_per_node'] // global_params.config['ngpus_per_node']
            n_parallel_jobs = global_params.config.ngpu_total
            qu.batchjob_script(multi_params, "render_views_egl", suffix='_small', log=log,
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
        Requires :py:func:`~syconn.exec.exec_init.run_create_neuron_ssd`.
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

        # render normal views only
        n_cores = global_params.config['ncores_per_node'] // global_params.config['ngpus_per_node']

        # sort ssv ids according to their number of SVs (descending)
        multi_params = big_ssv[np.argsort(ssd.load_numpy_data('size')[~size_mask])[::-1]]
        multi_params = chunkify(multi_params, max_n_jobs)
        # list of SSV IDs and SSD parameters need to be given to a single QSUB job
        multi_params = [(ixs, global_params.config.working_dir) for ixs in multi_params]
        qu.batchjob_script(multi_params, "render_views_egl", suffix='_big', log=log,
                           additional_flags="--gres=gpu:1",
                           n_cores=n_cores, remove_jobfolder=True)
        log.info('Finished rendering of {}/{} SSVs.'.format(len(big_ssv),
                                                            len(nb_svs_per_ssv)))


def run_neuron_rendering(max_n_jobs: Optional[int] = None):
    """
    Render the default views as defined in ``global_params`` [WIP].

    Args:
        max_n_jobs: Number of parallel jobs.

    Notes:
        Requires :func:`~syconn.exec.exec_init.run_create_neuron_ssd`.
    """
    log = initialize_logging('neuron_rendering',
                             global_params.config.working_dir + '/logs/')
    ps = [Process(target=_run_neuron_rendering_big_helper, args=(max_n_jobs,)),
          Process(target=_run_neuron_rendering_small_helper, args=(max_n_jobs,))]
    for p in ps:
        p.start()
        time.sleep(10)
    for p in ps:
        p.join()
        if p.exitcode != 0:
            raise Exception(f'Worker {p.name} stopped unexpectedly with exit '
                            f'code {p.exitcode}.')
        p.close()
    log.info('Finished rendering of all SSVs. Checking completeness.')
    ssd = SuperSegmentationDataset(working_dir=global_params.config.working_dir)
    res = find_incomplete_ssv_views(ssd, woglia=True, n_cores=global_params.config['ncores_per_node'])
    if len(res) != 0:
        msg = "Not all SSVs were rendered! {}/{} missing. Example IDs:\n" \
              "{}".format(len(res), len(ssd.ssv_ids),
                          res[:10])
        log.error(msg)
        raise RuntimeError(msg)
    log.info('Success.')


def _run_huge_ssv_render_worker(q: Queue, q_out: Queue):
    """
    Helper method of :func:`~run_astrocyte_rendering`.

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
                         skip_indexviews=True, woglia=False, overwrite=True)
        q_out.put(0)


def run_astrocyte_rendering(max_n_jobs: Optional[int] = None):
    """
    Uses the pruned RAG at ``global_params.config.pruned_svgraph_path``
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
    log = initialize_logging('glia_separation', global_params.config.working_dir + '/logs/',
                             overwrite=True)

    sds = SegmentationDataset("sv", working_dir=global_params.config.working_dir)

    # precompute rendering locations
    multi_params = chunkify(sds.so_dir_paths, global_params.config.ncore_total * 2)
    so_kwargs = dict(working_dir=global_params.config.working_dir, obj_type='sv')
    multi_params = [[par, so_kwargs] for par in multi_params]
    # TODO: remove comment as soon as astrocyte separation supports on the fly view generation
    # if not global_params.config.use_onthefly_views:
    _ = qu.batchjob_script(multi_params, "sample_location_caching", remove_jobfolder=True, log=log)

    log.info("Preparing RAG.")
    np.random.seed(0)

    # view rendering prior to glia removal, choose SSD accordingly
    # glia removal is based on the initial RAG and does not require explicitly stored SSVs
    version = "tmp"

    G = nx.read_edgelist(global_params.config.pruned_svgraph_path, nodetype=np.uint64)

    cc_gs = sorted(list((G.subgraph(c) for c in nx.connected_components(G))), key=len, reverse=True)
    all_sv_ids_in_rag = np.array(list(G.nodes()), dtype=np.uint64)

    # generate parameter for view rendering of individual SSV
    sv_size_dict = {}
    bbs = sds.load_numpy_data('bounding_box') * sds.scaling
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
        elif ccsize_dict[list(g.nodes())[0]] < global_params.config['min_cc_size_ssv']:
            pass  # ignore this CC
        else:
            small_ssv.append(g)

    log.info("View rendering for astrocyte separation started.")
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
            p.close()
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

    _ = qu.batchjob_script(
        multi_params, "render_views_glia_removal", log=log,
        n_cores=global_params.config['ncores_per_node'] // global_params.config['ngpus_per_node'],
        additional_flags="--gres=gpu:1", remove_jobfolder=True)

    # check completeness
    log.info('Finished view rendering for astrocyte separation. Checking completeness.')
    sd = SegmentationDataset("sv", working_dir=global_params.config.working_dir)
    res = find_missing_sv_views(sd, woglia=False, n_cores=global_params.config['ncores_per_node'])
    missing_not_contained_in_rag = np.setdiff1d(res, all_sv_ids_in_rag)  # TODO: report at least.
    missing_contained_in_rag = np.intersect1d(res, all_sv_ids_in_rag)
    if len(missing_contained_in_rag) != 0:
        msg = "Not all SVs were rendered completely! {}/{} missing:\n" \
              "{}".format(len(missing_contained_in_rag), len(all_sv_ids_in_rag),
                          missing_contained_in_rag[:100])
        log.error(msg)
        raise ValueError(msg)
    else:
        log.info('All SVs now contain views required for glia prediction.')
    # TODO: remove temporary SSV datasets
