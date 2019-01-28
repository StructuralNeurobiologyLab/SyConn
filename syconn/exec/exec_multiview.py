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

from syconn.reps.rep_helper import knossos_ml_from_ccs
from syconn.reps.segmentation_helper import find_missing_sv_views
from syconn.reps.super_segmentation import SuperSegmentationObject
from syconn.global_params import rag_suffix, RENDERING_MAX_NB_SV
from syconn.proc.glia_splitting import qsub_glia_splitting, collect_glia_sv, \
    write_glia_rag, transform_rag_edgelist2pkl
from syconn.reps.segmentation import SegmentationDataset
from syconn.reps.segmentation_helper import find_missing_sv_attributes
from syconn.handler.prediction import get_glia_model
from syconn.handler.basics import parse_cc_dict_from_kml
from syconn.proc.graphs import create_ccsize_dict
from syconn.proc import ssd_proc
from syconn.reps.super_segmentation_helper import find_incomplete_ssv_views
from syconn.global_params import NGPU_TOTAL
from syconn import global_params
from syconn.handler.prediction import get_axoness_model
from syconn.handler.basics import chunkify
from syconn.reps.super_segmentation import SuperSegmentationDataset
from syconn.reps.super_segmentation_helper import find_missing_sv_attributes_in_ssv
from syconn.handler.logger import initialize_logging
from syconn.mp import batchjob_utils as qu


def run_axoness_mapping():
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
    multi_params = chunkify(multi_params, 2000)

    multi_params = [(par, pred_key_appendix) for par in multi_params]
    log.info('Starting axoness mapping.')
    path_to_out = qu.QSUB_script(multi_params, "map_viewaxoness2skel",
                                 n_max_co_processes=global_params.NCORE_TOTAL,
                                 pe="openmp", queue=None, suffix="", n_cores=1)
    # TODO: perform completeness check
    log.info('Finished axoness mapping.')


def run_axoness_prediction(n_jobs=100):
    log = initialize_logging('axon_prediction', global_params.config.working_dir + '/logs/',
                             overwrite=False)
    # TODO: currently working directory has to be set globally in global_params and is not adjustable
    # here because all qsub jobs will start a script referring to 'global_params.config.working_dir'
    ssd = SuperSegmentationDataset(working_dir=global_params.config.working_dir)
    sd = ssd.get_segmentationdataset("sv")
    # chunk them
    multi_params = chunkify(sd.so_dir_paths, n_jobs)
    pred_key = "axoness_probas"  # leave this fixed because it is used all over
    # get model properties
    log.info('Performing axon prediction of neuron views. Labels will be stored '
             'on SV level in the attribute dict with key "{}"'.format(pred_key))

    m = get_axoness_model()
    model_kwargs = dict(model_path=m._path, normalize_data=m.normalize_data,
                        imposed_batch_size=m.imposed_batch_size, nb_labels=m.nb_labels,
                        channels_to_load=m.channels_to_load)
    # all other kwargs like obj_type='sv' and version are the current SV SegmentationDataset by default
    so_kwargs = dict(working_dir=global_params.config.working_dir)
    # for axoness views set woglia to True (because glia were removed beforehand),
    #  raw_only to False
    pred_kwargs = dict(woglia=True, pred_key=pred_key, verbose=False,
                       raw_only=False)
    multi_params = [[par, model_kwargs, so_kwargs, pred_kwargs] for
                    par in multi_params]
    for par in multi_params:
        mk = par[1]
        # Single GPUs are made available for every job via slurm, no need for random assignments.
        mk["init_gpu"] = 0  # np.random.rand(0, 2)

    path_to_out = qu.QSUB_script(multi_params, "predict_sv_views_chunked",
                                 n_max_co_processes=15, pe="openmp", queue=None,
                                 script_folder=None, n_cores=10,
                                 suffix="_axoness", additional_flags="--gres=gpu:1")  # removed -V
    log.info('Finished axon prediction. Now checking for missing predictions.')
    res = find_missing_sv_attributes_in_ssv(ssd, pred_key, n_cores=10)
    if len(res) > 0:
        log.error("Attribute '{}' missing for follwing"
                  " SVs:\n{}".format(pred_key, res))
    else:
        log.info('Success.')


def run_celltype_prediction(n_jobs=100):
    log = initialize_logging('celltype_prediction', global_params.config.working_dir+ '/logs/',
                             overwrite=False)
    ssd = SuperSegmentationDataset(working_dir=global_params.config.working_dir)
    # shuffle SV IDs
    np.random.seed(0)
    ssv_ids = ssd.ssv_ids

    log.info('Starting cell type prediction.')
    nb_svs_per_ssv = np.array([len(ssd.mapping_dict[ssv_id])
                               for ssv_id in ssd.ssv_ids])
    multi_params = ssd.ssv_ids
    ordering = np.argsort(nb_svs_per_ssv)
    multi_params = multi_params[ordering[::-1]]
    multi_params = chunkify(multi_params, n_jobs)
    # job parameter will be read sequentially, i.e. in order to provide only
    # one list as parameter one needs an additonal axis
    multi_params = [(ixs, ) for ixs in multi_params]

    path_to_out = qu.QSUB_script(multi_params, "predict_cell_type", pe="openmp",
                                 n_max_co_processes=34, queue=None,
                                 script_folder=None, suffix="",
                                 n_cores=10, additional_flags="--gres=gpu:1")
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


def run_spiness_prediction(n_jobs=100):
    log = initialize_logging('spine_identification', global_params.config.working_dir
                             + '/logs/', overwrite=False)
    ssd = SuperSegmentationDataset(working_dir=global_params.config.working_dir)
    pred_key = "spiness"

    # run semantic spine segmentation on multi views
    sd = ssd.get_segmentationdataset("sv")
    # chunk them
    multi_params = chunkify(sd.so_dir_paths, n_jobs)
    # set model properties
    model_kwargs = dict(src=global_params.config.mpath_spiness,
                        multi_gpu=False)
    so_kwargs = dict(working_dir=global_params.config.working_dir)
    pred_kwargs = dict(pred_key=pred_key)
    multi_params = [[par, model_kwargs, so_kwargs, pred_kwargs]
                    for par in multi_params]
    log.info('Starting spine prediction.')
    qu.QSUB_script(multi_params, "predict_spiness_chunked",
                   n_max_co_processes=NGPU_TOTAL, pe="openmp", queue=None,
                   n_cores=10, python_path=global_params.config.py36path,  # use python 3.6
                   suffix="",  additional_flags="--gres=gpu:1")   # removed -V (used with QSUB)
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
    multi_params = chunkify(multi_params, 3000)
    # add ssd parameters
    kwargs_semseg2mesh = dict(semseg_key=pred_key, force_overwrite=True)
    multi_params = [(ssv_ids, ssd.version, ssd.version_dict, ssd.working_dir,
                     kwargs_semseg2mesh) for ssv_ids in multi_params]
    log.info('Starting mapping of spine predictions to neurite surfaces.')
    qu.QSUB_script(multi_params, "map_spiness", pe="openmp", queue=None,
                   n_cores=2, suffix="", additional_flags="", resume_job=False)  # removed -V (used with QSUB)
    log.info('Finished spine mapping.')


def run_neuron_rendering():
    log = initialize_logging('neuron_view_rendering',
                             global_params.config.working_dir + '/logs/')
    # TODO: currently working directory has to be set globally in global_params
    #  and is not adjustable here because all qsub jobs will start a script
    #  referring to 'global_params.config.working_dir'
    # view rendering prior to glia removal, choose SSD accordingly
    ssd = SuperSegmentationDataset(working_dir=global_params.config.working_dir)

    #  TODO: use actual size criteria, e.g. number of sampling locations
    nb_svs_per_ssv = np.array([len(ssd.mapping_dict[ssv_id])
                               for ssv_id in ssd.ssv_ids])

    # render normal size SSVs
    size_mask = nb_svs_per_ssv <= global_params.RENDERING_MAX_NB_SV
    multi_params = ssd.ssv_ids[size_mask]

    # TODO: Currently slow if SSV contains very large SV(s)
    # sort ssv ids according to their number of SVs (descending)
    ordering = np.argsort(nb_svs_per_ssv[size_mask])
    multi_params = multi_params[ordering[::-1]]
    multi_params = chunkify(multi_params, 2000)
    # list of SSV IDs and SSD parameters need to be given to a single QSUB job
    multi_params = [(ixs, global_params.config.working_dir) for ixs in multi_params]
    log.info('Start rendering of {} SSVs. '.format(np.sum(size_mask)))
    if np.sum(~size_mask) > 0:
        log.info('{} huge SSVs will be rendered afterwards using the whole'
                 ' cluster.'.format(np.sum(~size_mask)))
    # generic
    path_to_out = qu.QSUB_script(multi_params, "render_views", pe="openmp",
                                 n_max_co_processes=global_params.NCORE_TOTAL,
                                 script_folder=None, suffix="", queue=None)
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
    res = find_incomplete_ssv_views(ssd, woglia=True, n_cores=10)
    if len(res) != 0:
        msg = "Not all SSVs were rendered completely! Missing:\n{}".format(res)
        log.error(msg)
        raise RuntimeError(msg)
    else:
        log.info('Success.')


def run_create_neuron_ssd(prior_glia_removal=True):
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
    if not prior_glia_removal:
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
            if ccsize_dict[ix] < global_params.min_cc_size_ssv:
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
    ssd.save_dataset_shallow()
    ssd.save_dataset_deep(qsub_pe="openmp", n_max_co_processes=200)
    log.info('Finished SSD initialization. Starting cellular '
             'organelle mapping.')

    # map cellular organelles to SSVs
    # TODO: increase number of jobs in the next two QSUB submissions and sort by SSV size (descending)
    ssd_proc.aggregate_segmentation_object_mappings(
        ssd, global_params.existing_cell_organelles, qsub_pe="openmp")
    ssd_proc.apply_mapping_decisions(
        ssd, global_params.existing_cell_organelles, qsub_pe="openmp")
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


def run_glia_prediction():
    log = initialize_logging('glia_prediction', global_params.config.working_dir + '/logs/',
                             overwrite=False)
    # only append to this key if needed (for e.g. different versions, change accordingly in 'axoness_mapping.py')
    pred_key = "glia_probas"
    # Load initial RAG from  Knossos mergelist text file.
    init_rag_p = global_params.config.working_dir + "initial_rag.txt"
    assert os.path.isfile(init_rag_p), "Initial RAG could not be found at %s."\
                                       % init_rag_p
    init_rag = parse_cc_dict_from_kml(init_rag_p)
    log.info('Found {} CCs with a total of {} SVs in inital RAG.'
          ''.format(len(init_rag), np.sum([len(v) for v in init_rag.values()])))
    # chunk them
    sd = SegmentationDataset("sv", working_dir=global_params.config.working_dir)
    multi_params = chunkify(sd.so_dir_paths, 100)
    # get model properties
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
    # randomly assign to gpu 0 or 1
    for par in multi_params:
        mk = par[1]
        # GPUs are made available for every job via slurm, no need for random assignments: np.random.rand(0, 2)
        mk["init_gpu"] = 0
    path_to_out = qu.QSUB_script(multi_params, "predict_sv_views_chunked",
                                 n_max_co_processes=25, pe="openmp",
                                 queue=None, n_cores=10, suffix="_glia",
                                 script_folder=None,
                                 additional_flags="--gres=gpu:1")  # removed -V
    log.info('Finished glia prediction. Checking completeness.')
    res = find_missing_sv_attributes(sd, pred_key, n_cores=10)
    if len(res) > 0:
        log.error("Attribute '{}' missing for follwing"
                  " SVs:\n{}".format(pred_key, res))
    else:
        log.info('Success.')


def run_glia_splitting():
    log = initialize_logging('glia_splitting', global_params.config.working_dir + '/logs/',
                             overwrite=False)
    # path to networkx file containing the initial rag, TODO: create alternative formats
    G = nx.Graph()  # TODO: Make this more general
    with open(global_params.config.init_rag_path, 'r') as f:
        for l in f.readlines():
            edges = [int(v) for v in re.findall('(\d+)', l)]
            G.add_edge(edges[0], edges[1])

    all_sv_ids_in_rag = np.array(list(G.nodes()), dtype=np.uint)
    log.info("Found {} SVs in initial RAG.".format(len(all_sv_ids_in_rag)))

    # add single SV connected components to initial graph
    sd = SegmentationDataset(obj_type='sv', working_dir=global_params.config.working_dir)
    sv_ids = sd.ids
    diff = np.array(list(set(sv_ids).difference(set(all_sv_ids_in_rag))))
    log.info('Found {} single connected component SVs which were'
             ' missing in initial RAG.'.format(len(diff)))

    for ix in diff:
        G.add_node(ix)

    all_sv_ids_in_rag = np.array(list(G.nodes()), dtype=np.uint)
    log.info("Found {} SVs in initial RAG after adding size-one connected "
             "components. Writing RAG to pkl.".format(len(all_sv_ids_in_rag)))

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
    log.info("Finished glia splitting. Resulting RAGs are stored at {}."
             "".format(global_params.config.working_dir + "/glia/"))


def run_glia_rendering():
    log = initialize_logging('glia_view_rendering', global_params.config.working_dir + '/logs/',
                             overwrite=False)
    np.random.seed(0)

    # view rendering prior to glia removal, choose SSD accordingly
    version = "tmp"  # glia removal is based on the initial RAG and does not require explicitly stored SSVs

    G = nx.Graph()  # TODO: Add factory method for initial RAG
    with open(global_params.config.init_rag_path, 'r') as f:
        for l in f.readlines():
            edges = [int(v) for v in re.findall('(\d+)', l)]
            G.add_edge(edges[0], edges[1])

    all_sv_ids_in_rag = np.array(list(G.nodes()), dtype=np.uint)
    log.info("Found {} SVs in initial RAG.".format(len(all_sv_ids_in_rag)))

    # add single SV connected components to initial graph
    sd = SegmentationDataset(obj_type='sv', working_dir=global_params.config.working_dir)
    sv_ids = sd.ids
    diff = np.array(list(set(sv_ids).difference(set(all_sv_ids_in_rag))))
    log.info('Found {} single connected component SVs which were missing'
             ' in initial RAG.'.format(len(diff)))

    for ix in diff:
        G.add_node(ix)

    all_sv_ids_in_rag = np.array(list(G.nodes()), dtype=np.uint)
    log.info("Found {} SVs in initial RAG after adding size-one connected "
             "components. Writing kml text file".format(len(all_sv_ids_in_rag)))

    # write out readable format for 'glia_prediction.py'
    ccs = [[n for n in cc] for cc in nx.connected_component_subgraphs(G)]
    kml = knossos_ml_from_ccs([np.sort(cc)[0] for cc in ccs], ccs)
    with open(global_params.config.working_dir + "initial_rag.txt", 'w') as f:
        f.write(kml)

    # generate parameter for view rendering of individual SSV
    log.info("Starting view rendering.")
    multi_params = []
    for cc in nx.connected_component_subgraphs(G):
        multi_params.append(cc)
    multi_params = np.array(multi_params)

    # identify huge SSVs and process them individually on whole cluster
    nb_svs = np.array([g.number_of_nodes() for g in multi_params])
    big_ssv = multi_params[nb_svs > RENDERING_MAX_NB_SV]

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
                         skip_indexviews=True, woglia=False,
                         qsub_pe="openmp", overwrite=True, qsub_co_jobs=global_params.NCORE_TOTAL)

    # render small SSV without overhead and single cpus on whole cluster
    multi_params = multi_params[nb_svs <= RENDERING_MAX_NB_SV]
    np.random.shuffle(multi_params)
    multi_params = chunkify(multi_params, 2000)

    # list of SSV IDs and SSD parameters need to be given to a single QSUB job
    multi_params = [(ixs, global_params.config.working_dir, version) for ixs in multi_params]
    path_to_out = qu.QSUB_script(multi_params, "render_views_glia_removal",
                                 n_max_co_processes=global_params.NCORE_TOTAL, pe="openmp",
                                 queue=None, script_folder=None, suffix="")

    # check completeness
    sd = SegmentationDataset("sv", working_dir=global_params.config.working_dir)
    res = find_missing_sv_views(sd, woglia=False, n_cores=10)
    missing_not_contained_in_rag = []
    missing_contained_in_rag = []
    for el in res:
        if el not in all_sv_ids_in_rag:
            missing_not_contained_in_rag.append(el)
        else:
            missing_contained_in_rag.append(el)
    if len(missing_not_contained_in_rag):
        log.info("%d SVs were not rendered but also not part of the initial"
                 "RAG: {}".format(missing_not_contained_in_rag))
    if len(missing_contained_in_rag) != 0:
        msg = "Not all SSVs were rendered completely! Missing:\n" \
              "{}".format(missing_contained_in_rag)
        log.error(msg)
        raise RuntimeError(msg)


def axoness_pred_exists(sv):
    sv.load_attr_dict()
    return 'axoness_probas' in sv.attr_dict
