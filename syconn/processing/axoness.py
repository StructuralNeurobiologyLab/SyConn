import numpy as np
import os
import re
import time
from multiprocessing import Pool, Manager, cpu_count
from numpy import array as arr
from sys import stdout

import networkx as nx
from sklearn.externals import joblib

from features import assign_property2node, majority_vote,\
    update_property_feat_kzip
from learning_rfc import cell_classification, load_csv2feat
from syconn.utils import skeleton_utils as su
from syconn.utils.datahandler import get_filepaths_from_dir, \
    load_ordered_mapped_skeleton, get_skelID_from_path
from syconn.utils.skeleton import Skeleton, SkeletonAnnotation
__author__ = 'philipp'


def predict_axoness_mappedskel(skel_fpaths, recompute_feat=False):
    """
    Predict axoness of each node on list of mapped skeletons.
    skel_fpaths : list of str
    recompute_feat : bool
    :return:
    """
    nb_cpus = cpu_count()
    pool = Pool(processes=nb_cpus)
    m = Manager()
    q = m.Queue()
    params = [(path, q, recompute_feat) for path in skel_fpaths]
    result = map(predict_axoness_of_single_mappedskel, params)
    # monitor loop
    while True:
        if result.ready():
            break
        else:
            size = float(q.qsize())
            stdout.write("\r%0.2f" % (size / len(params)))
            stdout.flush()
            time.sleep(4)
    _ = result.get()
    pool.close()
    pool.join()


def predict_axoness_of_single_mappedskel(args):
    """Predict  axoness of mapped skeleton. Helper function of
    ``predict_axoness_mappedskel``. Saves mapped skeleton with predicted
    axoness at origin.
    """
    path, q, recompute_feat = args
    anno, mitos, p4, az = load_ordered_mapped_skeleton(path)
    rfc_axoness = joblib.load('/lustre/pschuber/gt_axoness/rfc/rfc_axoness.pkl')
    if recompute_feat:
        update_property_feat_kzip(path)
    print "Load feature from file."
    input, header = load_csv2feat(path)
    axoness_feat = input[:, 1:]
    node_ids = input[:, 0].astype(np.int64)
    proba = rfc_axoness.predict_proba(axoness_feat)
    # TODO bug in newskeleton! correct to fix it like that?
    anno_node_ids = [node.getID() for node in anno.getNodes()]
    assert len(node_ids) == len(anno_node_ids), 'Length of stored features and'\
                                                'anno nodes differ!'
    diff = np.abs(np.min(node_ids) - np.min(anno_node_ids))
    print "Difference between node ids and saved node IDS:", diff
    for k, node_id in enumerate(node_ids):
        node = anno.getNodeByID(node_id + diff)
        node_comment = node.getComment()
        ax_ix = node_comment.find('axoness')
        pred = np.argmax(proba[k])
        if ax_ix == -1:
            node.appendComment('axoness%d' % pred)
        else:
            help_list = list(node_comment)
            help_list[ax_ix+7] = str(pred)
            node.setComment("".join(help_list))
        for ii in range(len(proba[k])):
            node.setDataElem('axoness_proba%d' % ii, proba[k, ii])
    majority_vote(anno, 'axoness', 6000)
    grow_out_soma(anno)
    majority_processes(anno)
    dummy_skel = Skeleton()
    dummy_skel.add_annotation(anno)
    dummy_skel.add_annotation(mitos)
    dummy_skel.add_annotation(p4)
    dummy_skel.add_annotation(az)
    dummy_skel.to_kzip(path[:-6] + '_smoothed_process_majority.k.zip')
    if q is not None:
        q.put(1)


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
    axoness_comment = str(ids[0]) + 'axoness' + str(axoness_0) \
    + '_' + str(ids[1]) + 'axoness' + str(axoness_1)
    anno.appendComment(axoness_comment)
    return arr(ids), arr([axoness_0, axoness_1])


def majority_processes(anno):
    """Label processes of cell in anno according to majority of axonoess in
    its nodes. If anno contains soma nodes, a slight smoothing is applied
    and afterwards the soma is grown out, in order to avoid branch points
    near the soma which are false positive axons/dendrite nodes.
    Inplace operation.

    Parameters
    ----------
    anno : SkeletonAnnotation
    """
    hns = []
    soma_node_nb = 0
    soma_node_ids = []
    for node in anno.getNodes():
        if node.degree() == 1:
            hns.append(node)
        if int(node.data["axoness_pred"]) == 2:
            soma_node_nb += 1
            soma_node_ids.append(node.getID())
    if soma_node_nb != 0:
        grow_out_soma(anno)
        majority_vote(anno, 'axoness', 3000)
        used_hn_ids = []
        graph = su.annotation_to_nx_graph(anno)
        calc_distance2soma(graph, hns)
        # reorder head nodes with descending distance to soma
        distances = [node.data['dist2soma'] for node in hns]
        hns = [hns[ii] for ii in np.argsort(distances)[::-1]]
        for hn in hns:
            if hn.getID() in used_hn_ids:
                continue
            else:
                visited_nodes = []
                axoness_found = []
                used_hn_ids.append(hn.getID())
            for node in nx.dfs_preorder_nodes(graph, hn):
                # if branch point stop
                if node.degree() == 1:
                    used_hn_ids.append(node.getID())
                if int(node.data["axoness_pred"]) == 2:
                    if len(axoness_found) == 0:
                        break
                    majority_axoness = cell_classification(arr(axoness_found))
                    for n in visited_nodes:
                        assign_property2node(n, majority_axoness, 'axoness')
                    break
                else:
                    visited_nodes.append(node)
                    axoness_found.append(int(node.data["axoness_pred"]))
        for n_ix in soma_node_ids:
            anno.getNodeByID(n_ix).data['axoness_pred'] = 2
    else:
        print "Process without soma prediction. Using majority vote of cell" \
              "part."
        axoness_found = []
        for node in anno.getNodes():
            axoness_found.append(int(node.data["axoness_pred"]))
        majority_axoness = cell_classification(arr(axoness_found))
        for node in anno.getNodes():
            assign_property2node(node, majority_axoness, 'axoness')


def calc_distance2soma(graph, nodes):
    """ Calculates the distance to a soma node for each node and stores it
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


def grow_out_soma(anno, max_dist=700):
    """
    Grows out soma nodes, in order to overcome false negative soma nodes which
    should have separated axon and dendritic processes.
    Parameters
    ----------
    anno : SkeletonAnnotation
    max_dist : int

    """
    soma_nodes = []
    graph = su.annotation_to_nx_graph(anno)
    for node in anno.getNodes():
        if int(node.data["axoness_pred"]) == 2:
            soma_nodes.append(node)
    for source in soma_nodes:
        distance = 0
        current_coords = arr(source.getCoordinate_scaled())
        for node in nx.dfs_preorder_nodes(graph, source):
            new_coords = arr(node.getCoordinate_scaled())
            distance += np.linalg.norm(current_coords - new_coords)
            if distance > max_dist:
                break
            if int(node.data["axoness_pred"]) != 2:
                assign_property2node(node, 2, 'axoness')
            current_coords = new_coords

