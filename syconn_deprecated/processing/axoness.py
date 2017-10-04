# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max-Planck-Institute for Medical Research, Heidelberg, Germany
# Authors: Sven Dorkenwald, Philipp Schubert, Jörgen Kornfeld

import networkx as nx
import numpy as np
from numpy import array as arr
import re
from features import assign_property2node, majority_vote
from learning_rfc import cell_classification
from knossos_utils import skeleton_utils as su
from knossos_utils.skeleton import SkeletonAnnotation


def predict_axoness_from_node_comments(anno):
    """Exctracts axoness prediction from node comment for given contact site
    annotation.

    Parameters
    ----------
    anno : SkeletonAnnotation
        Contact site

    Returns
    -------
    numpy.array, numpy.array
        skeleton IDS, skeleton axoness
    """
    axoness = [[], []]
    cs_comment = anno.getComment()
    try:
        ids = re.findall('skel_(\d+)_(\d+)', cs_comment)[0]
    except IndexError:
        ids = re.findall('syn(\d+).k.zip_give_syn(\d+)', cs_comment)[0]
    for node in list(anno.getNodes()):
        n_comment = node.getComment()
        if 'skelnode' in n_comment:
            axoness_class = re.findall('axoness(\d+)', n_comment)[0]
            try:
                skel_id = re.findall('cs\d+_(\d+)_', n_comment)[0]
            except IndexError:
                skel_id = re.findall('syn(\d+).k.zip', n_comment)[0]
            axoness[ids.index(skel_id)] += [int(axoness_class)]
    axoness_0 = cell_classification(arr(axoness[0]))
    axoness_1 = cell_classification(arr(axoness[1]))
    axoness_comment = ids[0]+'axoness'+str(axoness_0) \
    + '_' + ids[1]+'axoness'+str(axoness_1)
    anno.appendComment(axoness_comment)
    return arr([int(ix) for ix in ids]), arr([axoness_0, axoness_1])


def predict_axoness_from_nodes(anno):
    """Exctracts axoness prediction from nodes for given contact site annotation

    Parameters
    ----------
    anno : SkeletonAnnotation
        contact site.

    Returns
    -------
    numpy.array, numpy.array
        skeleton IDS, skeleton axoness
    """
    axoness = [[], []]
    ids = []
    for node in list(anno.getNodes()):
        n_comment = node.getComment()
        if '_center' in n_comment:
            ids = [int(node.data['adj_skel1'])]
            ids.append(int(node.data['adj_skel2']))
            break
    try:
        for node in list(anno.getNodes()):
            n_comment = node.getComment()
            if 'skelnode' in n_comment:
                axoness_class = node.data['axoness_pred']
                try:
                    skel_id = int(re.findall('(\d+)_skelnode', n_comment)[0])
                except IndexError:
                    skel_id = int(re.findall('syn(\d+)', n_comment)[0])
                axoness[ids.index(skel_id)] += [int(axoness_class)]
        axoness_0 = int(np.round(np.mean(axoness[0])))
        axoness_1 = int(np.round(np.mean(axoness[1])))
    except KeyError as e:
        if e.message != 'axoness_pred':
            raise e
        else:
            print "Axoness prediction not available in cs '%s'." % anno.filename
            axoness_0 = -1
            axoness_1 = -1
    axoness_comment = str(ids[0]) + 'axoness' + str(axoness_0) + '_' + \
                      str(ids[1]) + 'axoness' + str(axoness_1)
    anno.appendComment(axoness_comment)
    return arr(ids), arr([axoness_0, axoness_1])


def calc_distance2soma(graph, nodes):
    """Calculates the distance to a soma node for each node and stores it
    inplace in node.data['dist2soma']. Building depth first search graph at
    each node for sorted node ordering.

    Parameters
    ----------
    graph : graph of SKeletonAnnotation
    nodes : SkeletonNodes
    """
    for source in nodes:
        distance = 0
        current_coords = arr(source.getCoordinate_scaled())
        for node in nx.dfs_preorder_nodes(graph, source):
            new_coords = arr(node.getCoordinate_scaled())
            distance += np.linalg.norm(current_coords - new_coords)
            current_coords = new_coords
            if int(node.data['axoness_pred']) == 2:
                source.data['dist2soma'] = distance
                break


