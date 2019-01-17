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

from syconn.config.global_params import wd, min_cc_size_neuron, path_initrag,\
    rag_suffix
from syconn.proc.glia_splitting import qsub_glia_splitting, collect_glia_sv, \
    write_glia_rag, transform_rag_edgelist2pkl
from syconn.reps.segmentation import SegmentationDataset
from syconn.handler.logger import initialize_logging


if __name__ == "__main__":
    log = initialize_logging('glia_splitting', wd + '/logs/',
                             overwrite=False)
    suffix = rag_suffix
    # path to networkx file containing the initial rag, TODO: create alternative formats
    G = nx.Graph()  # TODO: Make this more general
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
    log.info('Found {} single connected component SVs which were'
             ' missing in initial RAG.'.format(len(diff)))

    for ix in diff:
        G.add_node(ix)

    all_sv_ids_in_rag = np.array(list(G.nodes()), dtype=np.uint)
    log.info("Found {} SVs in initial RAG after adding size-one connected "
             "components. Writing RAG to pkl.".format(len(all_sv_ids_in_rag)))

    if not os.path.isdir(wd + "/glia/"):
        os.makedirs(wd + "/glia/")
    transform_rag_edgelist2pkl(G)

    # first perform glia splitting based on multi-view predictions, results are
    # stored at SuperSegmentationDataset ssv_gliaremoval
    qsub_glia_splitting()

    # collect all neuron and glia SVs and store them in numpy array
    collect_glia_sv()

    # # here use reconnected RAG or initial rag
    recon_nx = G
    # create glia / neuron RAGs
    write_glia_rag(recon_nx, min_cc_size_neuron, suffix=rag_suffix)
    log.info("Finished glia splitting. Resulting RAGs are stored at {}."
             "".format(wd + "/glia/"))

