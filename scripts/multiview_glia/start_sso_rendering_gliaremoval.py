# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max-Planck-Institute of Neurobiology, Munich, Germany
# Authors: Philipp Schubert, Joergen Kornfeld
import os
import numpy as np
import networkx as nx
import re

from syconn.config.global_params import wd, RENDERING_MAX_NB_SV, path_initrag
from syconn.mp import qsub_utils as qu
from syconn.handler.basics import chunkify
from syconn.reps.rep_helper import knossos_ml_from_ccs
from syconn.reps.segmentation_helper import find_missing_sv_views
from syconn.reps.segmentation import SegmentationDataset
from syconn.reps.super_segmentation import SuperSegmentationObject
from syconn.handler.logger import initialize_logging


if __name__ == "__main__":
    log = initialize_logging('glia_view_rendering', wd + '/logs/',
                             overwrite=False)
    N_JOBS = 360
    np.random.seed(0)

    # view rendering prior to glia removal, choose SSD accordingly
    version = "tmp"  # glia removal is based on the initial RAG and does not require explicitly stored SSVs
    # init_rag_p = wd + "initial_rag.txt"
    # assert os.path.isfile(init_rag_p), "Initial RAG could not be found at %s."\
    #                                    % init_rag_p
    # init_rag = parse_cc_dict_from_kml(init_rag_p)
    # all_sv_ids_in_rag = np.concatenate(list(init_rag.values()))

    G = nx.Graph()  # TODO: Add factory method for initial RAG
    with open(path_initrag, 'r') as f:
        for l in f.readlines():
            edges = [int(v) for v in re.findall('(\d+)', l)]
            G.add_edge(edges[0], edges[1])

    all_sv_ids_in_rag = np.array(list(G.nodes()), dtype=np.uint)
    log.info("Found {} SVs in initial RAG.".format(len(all_sv_ids_in_rag)))

    # add single SV connected components to initial graph
    sd = SegmentationDataset(obj_type='sv', working_dir=wd)
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
    with open(wd + "initial_rag.txt", 'w') as f:
        f.write(kml)

    # # preprocess sample locations
        log.info("Starting sample location caching.")
    sd = SegmentationDataset("sv", working_dir=wd)
    # chunk them
    multi_params = chunkify(sd.so_dir_paths, 1000)
    # all other kwargs like obj_type='sv' and version are the current SV SegmentationDataset by default
    so_kwargs = dict(working_dir=wd)
    multi_params = [[par, so_kwargs] for par in multi_params]
    path_to_out = qu.QSUB_script(multi_params, "sample_location_caching",
                                 n_max_co_processes=300, pe="openmp", queue=None,
                                 script_folder=None, suffix="")

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
        sso = SuperSegmentationObject(sv_ixs[0], working_dir=wd, version=version,
                                      create=False, sv_ids=sv_ixs)
        # nodes of sso._rag need to be SV
        new_G = nx.Graph()
        for e in g.edges():
            new_G.add_edge(sso.get_seg_obj("sv", e[0]),
                           sso.get_seg_obj("sv", e[1]))
        sso._rag = new_G
        sso.render_views(add_cellobjects=False, cellobjects_only=False,
                         skip_indexviews=True, woglia=False,
                         qsub_pe="openmp", overwrite=True, qsub_co_jobs=N_JOBS)

    # render small SSV without overhead and single cpus on whole cluster
    multi_params = multi_params[nb_svs <= RENDERING_MAX_NB_SV]
    np.random.shuffle(multi_params)
    multi_params = chunkify(multi_params, 2000)

    # list of SSV IDs and SSD parameters need to be given to a single QSUB job
    multi_params = [(ixs, wd, version) for ixs in multi_params]
    path_to_out = qu.QSUB_script(multi_params, "render_views_glia_removal",
                                 n_max_co_processes=N_JOBS, pe="openmp", queue=None,
                                 script_folder=None, suffix="")

    # check completeness
    sd = SegmentationDataset("sv", working_dir=wd)
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

