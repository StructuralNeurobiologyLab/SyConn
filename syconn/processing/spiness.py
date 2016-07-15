import numpy as np

import networkx as nx

import syconn.new_skeleton.annotationUtils as au
from features import morphology_feature, assign_property2node, \
    node_branch_end_distance
from learning_rfc import save_train_clf
from syconn.utils.datahandler import get_filepaths_from_dir
from syconn.utils.datahandler import load_ordered_mapped_skeleton

__author__ = 'philipp'


def load_node_gt(path_to_file):
    """
    :param path_to_file: Path to gt file
    :return node labels and node ids
    """
    nml = load_ordered_mapped_skeleton(path_to_file)
    nml = nml[0]
    nodes = nml.getNodes()
    Comment = []
    id_list = []
    for x in nodes:
        node_comment = x.getComment()
        if 'Shaft' in node_comment:
            Comment.append(0)
            id_list.append(x.getID())
        elif 'Spine-Head' in node_comment:
            Comment.append(1)
            id_list.append(x.getID())
        elif 'Spine-Neck' in node_comment:
            Comment.append(2)
            id_list.append(x.getID())
    Y = np.array(Comment)
    id = np.array(id_list)
    return Y, id


def save_spiness_clf(gt_path, recompute=False, clf_used='rf'):
    """
    Save spiness clf specified by clf_used to gt_directory.
    :param gt_path: str to directory of spiness ground truth
    :param clf_used: 'rf' or 'svm'
    """
    X, y = load_spine_gt(gt_path, recompute=recompute)
    save_train_clf(X, y, clf_used, gt_path)


def load_spine_gt(gt_path, recompute=False):
    """
    Load spiness ground truth at given path. Set recompute if precomputed data
    is unwanted.
    """
    if recompute:
        list_path = get_filepaths_from_dir(gt_path)
        print list_path, len(list_path)
        Y = []
        X = []
        for path in list_path:
            all_x, spinehead_feat, idfeat = morphology_feature(path)
            y, idgt = load_node_gt(path)
            # only use shaft (0) and head (1) labels
            idgt = idgt[y != 2]
            y = y[y != 2]
            Y += y.tolist()
            for id in idgt:
                X.append(all_x[idfeat == id])
        X = np.array(X)
        Y = np.array(Y)
        np.save(gt_path + 'spine_feature.npy', X)
        np.save(gt_path + 'spine_label.npy', Y)
    else:
        X = np.load(gt_path+'spine_feature.npy')
        Y = np.load(gt_path+'spine_label.npy')
    if len(X) != len(Y): raise("Inconsistent data dimensions.")
    print "Using %d features for %d different labels (supp:%d, %d) and " \
          "%d samples." % (X.shape[2], len(set(Y)), np.sum(Y == 0),
                           np.sum(Y == 1), len(Y))
    return X[:, 0, :], Y


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
    """
    Assign nodes between spine head node and first node with degree 2 as
    spine necks inplace.
    :param anno: AnnotationObject containing spiness prediction of class
    head (1) and shaft (0). Key for prediction is "spiness_pred"
    """
    headnodes = collect_spineheads(anno, dist=np.inf)
    hn_ids = [n.ID for n in headnodes]
    for node in anno.getNodes():
        if node.ID in hn_ids:
            continue
        assign_property2node(node, '0', 'spiness')
    graph = au.annotation_to_nx_graph(anno)
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
