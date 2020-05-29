import os
import re
import glob

import numpy as np
import networkx as nx
from collections import deque
from knossos_utils.skeleton_utils import load_skeleton
from knossos_utils.skeleton import Skeleton, SkeletonAnnotation, SkeletonNode
from syconn.reps.super_segmentation import SuperSegmentationObject
from syconn import global_params


def expand_labels(input_path: str, output_path: str, scaling):
    input_path = os.path.expanduser(input_path)
    output_path = os.path.expanduser(output_path)
    sso_id = int(re.findall(r"/(\d+).", input_path)[0])
    sso = SuperSegmentationObject(sso_id)
    sso.meshes2kzip(output_path)
    # load annotation object and corresponding skeleton
    a_obj = load_skeleton(input_path)
    if len(a_obj) == 1:
        a_obj = list(a_obj.values())[0]
    else:
        a_obj = a_obj["skeleton"]
    a_nodes = list(a_obj.getNodes())
    a_node_coords = np.array([n.getCoordinate() * scaling for n in a_nodes])
    a_node_labels = np.array([comment2int(n.getComment()) for n in a_nodes], dtype=np.int)
    # generate graph from nodes in annotation object
    a_edges = []
    for node in a_nodes:
        ix = a_nodes.index(node)
        neighbors = node.getNeighbors()
        for neighbor in neighbors:
            nix = a_nodes.index(neighbor)
            a_edges.append((ix, nix))
    # propagate labels, nodes with no label get label from nearest node with label
    g = nx.Graph()
    g.add_nodes_from([(i, dict(label=a_node_labels[i])) for i in range(len(a_nodes))])
    g.add_edges_from(a_edges)
    cached_labels = a_node_labels.copy()
    for node in g.nodes:
        if g.nodes[node]['label'] == -1:
            ix = label_search(g, node)
            a_node_labels[node] = cached_labels[ix]
    labels = [int2comment(label) for label in a_node_labels]
    nxGraph2kzip(g, a_node_coords, labels, output_path)


def comment2int(comment: str):
    """ Map comments used during annotation to respective label. """
    if comment == "gt_dendrite" or comment == "shaft":
        return 0
    elif comment == "gt_axon":
        return 1
    elif comment == "gt_soma" or comment == "other":
        return 2
    elif comment == "gt_bouton":
        return 3
    elif comment == "gt_terminal":
        return 4
    elif comment == "gt_neck" or comment == "neck":
        return 5
    elif comment == "gt_head" or comment == "head":
        return 6
    else:
        return -1


def int2comment(label: int):
    li = ["gt_dendrite", "gt_axon", "gt_soma", "gt_bouton", "gt_terminal", "gt_neck", "gt_head"]
    return li[label]


def label_search(g: nx.Graph, source: int) -> int:
    """ Find nearest node to source which has a label. """
    visited = [source]
    neighbors = g.neighbors(source)
    de = deque([i for i in neighbors])
    while de:
        curr = de.pop()
        if g.nodes[curr]['label'] != -1:
            return curr
        if curr not in visited:
            visited.append(curr)
            neighbors = g.neighbors(curr)
            de.extendleft([i for i in neighbors if i not in visited])
    return 0


def nxGraph2kzip(g, coords, labels, kzip_path):
    skel = Skeleton()
    anno = SkeletonAnnotation()
    node_mapping = {}
    for v in g.nodes():
        c = coords[v]
        n = SkeletonNode().from_scratch(anno, c[0], c[1], c[2])
        n.setComment(labels[v])
        node_mapping[v] = n
        anno.addNode(n)
    for e in g.edges():
        anno.addEdge(node_mapping[e[0]], node_mapping[e[1]])
    skel.add_annotation(anno)
    skel.to_kzip(kzip_path)


if __name__ == '__main__':
    data_path = "/wholebrain/u/jklimesch/thesis/tmp/sparse_gt/"
    destination = "/wholebrain/u/jklimesch/thesis/tmp/sparse_gt/expanded/"
    global_params.wd = "/wholebrain/songbird/j0126/areaxfs_v6/"
    file_paths = glob.glob(data_path + '*.k.zip')
    for file in file_paths:
        slashs = [pos for pos, char in enumerate(file) if char == '/']
        name = file[slashs[-1] + 1:]
        expand_labels(file, destination + name, np.array([10, 10, 20]))
