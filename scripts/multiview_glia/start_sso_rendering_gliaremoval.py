# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max-Planck-Institute of Neurobiology, Munich, Germany
# Authors: Philipp Schubert, Joergen Kornfeld
import os
from syconn.config.global_params import wd
from syconn.mp import qsub_utils as qu
from syconn.handler.basics import chunkify, parse_cc_dict_from_kml
from syconn.reps.segmentation_helper import find_missing_sv_views
from syconn.reps.segmentation import SegmentationDataset
from syconn.reps.super_segmentation_object import SuperSegmentationObject
import numpy as np
import networkx as nx
import re


if __name__ == "__main__":
    np.random.seed(0)
    # generic QSUB script folder
    script_folder = os.path.dirname(os.path.abspath(__file__)) + "/../../syconn/QSUB_scripts/"
    # view rendering prior to glia removal, choose SSD accordingly
    version = "tmp"  # glia removal is based on the initial RAG and does not require explicitly stored SSVs
    # init_rag_p = wd + "initial_rag.txt"
    # assert os.path.isfile(init_rag_p), "Initial RAG could not be found at %s."\
    #                                    % init_rag_p
    # init_rag = parse_cc_dict_from_kml(init_rag_p)
    # all_sv_ids_in_rag = np.concatenate(list(init_rag.values()))

    G = nx.Graph()  # TODO: del and uncomment lines above
    with open('/wholebrain/songbird/j0126/RAGs/v4b_20180407_v4b_20180407_merges_newcb_ids_cbsplits.txt', 'r') as f:
        for l in f.readlines():
            edges = [int(v) for v in re.findall('(\d+)', l)]
            G.add_edge(edges[0], edges[1])
    all_sv_ids_in_rag = np.array(list(G.nodes()), dtype=np.uint)
    print("Found {} SVs in initial RAG. Starting view rendering.".format(len(all_sv_ids_in_rag)))
    # # preprocess sample locations
    # multi_params = chunkify(all_sv_ids_in_rag, 1000)
    # multi_params = [(sv_ixs, wd) for sv_ixs in multi_params]
    # path_to_out = qu.QSUB_script(multi_params, "sample_location_caching",
    #                              n_max_co_processes=100, pe="openmp", queue=None,   # TODO: n_max_co_processes=200
    #                              script_folder=script_folder, suffix="")

    # generate parameter for view rendering of individual SSV
    multi_params = []
    for cc in nx.connected_component_subgraphs(G):
        multi_params.append(cc)
    multi_params = np.array(multi_params)

    # identify huge SSVs and process them individually on whole cluster
    nb_svs = np.array([g.number_of_nodes() for g in multi_params])
    big_ssv = multi_params[nb_svs > 5e3]
    for kk, g in enumerate(big_ssv):
        # Create SSV object
        sv_ixs = np.sort(list(g.nodes()))
        print("Processing SSV [{}/{}] with {} SVs on whole cluster.".format(kk, len(big_ssv), len(sv_ixs)))
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
                         qsub_pe="openmp", overwrite=True)

    # render small SSV without overhead and single cpus on whole cluster
    multi_params = multi_params[nb_svs <= 5e3]
    np.random.shuffle(multi_params)
    multi_params = chunkify(multi_params, 2000)

    # list of SSV IDs and SSD parameters need to be given to a single QSUB job
    multi_params = [(ixs, wd, version) for ixs in multi_params]
    path_to_out = qu.QSUB_script(multi_params, "render_views_glia_removal",
                                 n_max_co_processes=100, pe="openmp", queue=None,   # TODO: n_max_co_processes=200
                                 script_folder=script_folder, suffix="")

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
        print("%d SVs were not rendered but also not part of the initial"
              "RAG: {}".format(missing_not_contained_in_rag))
    if len(missing_contained_in_rag) != 0:
        raise RuntimeError("Not all SSVs were rendered completely! Missing:\n"
                           "{}".format(missing_contained_in_rag))
