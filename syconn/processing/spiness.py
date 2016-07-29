# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max-Planck-Institute for Medical Research, Heidelberg, Germany
# Authors: Sven Dorkenwald, Philipp Schubert, Jörgen Kornfeld

import numpy as np
import networkx as nx
import syconn.utils.skeleton_utils as su
from features import assign_property2node, node_branch_end_distance


def collect_spineheads(anno, dist=6000):
    """
    Searches nodes in annotation for nodes with spinehead prediciton
    and returns them as list (no copy!).
    """
    nodes = anno.getNodes()
    # get distances to endpoints, stored as first
    _ = node_branch_end_distance(anno, dist)
    spineheads = []
    for node in nodes:
        if int(node.data["spiness_pred"]) == 1 and (node.degree() == 1):
            spineheads.append(node)
    return spineheads


def assign_neck(anno, max_head2endpoint_dist=600, max_neck2endpoint_dist=3000):
    """Assign nodes between spine head node and first node with degree 2 as
    spine necks inplace.
    head (1) and shaft (0). Key for prediction is "spiness_pred"

    Parameters
    ----------
    anno : SkeletonAnnotation
        mapped cell tracing
    max_head2endpoint_dist : int
        maximum distance between spine head and endpoint on graph
    max_neck2endpoint_dist : int
        maximum distance between spine neck and endpoint on graph

    """
    headnodes = collect_spineheads(anno, dist=np.inf)
    hn_ids = [n.ID for n in headnodes]
    for node in anno.getNodes():
        if node.ID in hn_ids:
            continue
        assign_property2node(node, '0', 'spiness')
    graph = su.annotation_to_nx_graph(anno)
    for hn in headnodes:
        for node in nx.dfs_preorder_nodes(graph, hn):
            # if branch point stop
            if node.ID in hn_ids:
                continue
            if node.degree() >= 3:
                break
            if node.data["endpointdistance"] < max_head2endpoint_dist:
                assign_property2node(node, '1', 'spiness')
            # otherwise set node as spine neck
            else:
                if node.data["spiness_pred"] == '2':
                    continue
                if node.data["endpointdistance"] > max_neck2endpoint_dist:
                    break
                assign_property2node(node, '2', 'spiness')
