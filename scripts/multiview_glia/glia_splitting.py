# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max-Planck-Institute of Neurobiology, Munich, Germany
# Authors: Philipp Schubert, Joergen Kornfeld
import os
from syconn.config.global_params import wd, glia_thresh, min_single_sv_size
from syconn.proc.graphs import transform_rag_edgelist2pkl, qsub_glia_splitting,\
    write_glia_rag, collect_glia_sv
import networkx as nx


if __name__ == "__main__":
    # path to networkx file containing the initial rag, create alternative formats
    rag_fname = wd + "/rag.nx"
    G = nx.read_edgelist(rag_fname, nodetype=int, delimiter=',')
    if not os.path.isdir(wd + "/glia/"):
        os.makedirs(wd + "/glia/")
    transform_rag_edgelist2pkl(G)

    # first perform glia splitting based on multi-view predictions, results are
    # stored at SuperSegmentationDataset ssv_gliaremoval
    qsub_glia_splitting()

    # collect all neuron and glia SVs and store them in numpy array
    collect_glia_sv()

    # here use reconnected RAG or initial rag
    recon_nx = wd + "/reconnect_rag.nx"
    # create glia / neuron RAGs
    write_glia_rag(recon_nx)