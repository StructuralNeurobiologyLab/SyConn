from syconn.reps.super_segmentation import SuperSegmentationDataset
from tqdm import tqdm
from syconn.reps.super_segmentation_helper import map_myelin2coords, majorityvote_skeleton_property
import numpy as np
import networkx as nx
from collections import deque
from morphx.classes.hybridcloud import HybridCloud
from knossos_utils.skeleton_utils import load_skeleton
from knossos_utils.skeleton import Skeleton, SkeletonAnnotation, SkeletonNode


def map_myelin(sso, hybrid_cloud):
    """
    Adds information about myelin to HybridCloud object. This information gets used by the ChunkHandler of the
    NeuronX training pipeline (see _adapt_obj function) if the features are set accordingly (see train.py in neuronx
    => pipeline).
    """
    sso.skeleton['myelin'] = map_myelin2coords(sso.skeleton["nodes"], mag=4)
    # majorityvote_skeleton_property(sso, 'myelin')
    # myelinated = sso.skeleton['myelin_avg10000']
    myelinated = sso.skeleton['myelin']
    hm_myelin = HybridCloud(vertices=hybrid_cloud.vertices, nodes=sso.skeleton["nodes"]*sso.scaling)
    nodes_idcs = np.arange(len(hm_myelin.nodes))
    myel_nodes = nodes_idcs[myelinated.astype(bool)]
    myel_vertices = []
    for node in myel_nodes:
        myel_vertices.extend(hm_myelin.verts2node[node])
    # myelinated vertices get type 1, not myelinated vertices get type 0
    types = np.zeros(len(hybrid_cloud.vertices))
    types[myel_vertices] = 1
    hybrid_cloud.set_types(types)
    return hybrid_cloud


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
    return -1


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


def comment2int(comment: str, convert_to_morphx: bool = False):
    """ Map comments used during annotation to respective label. """
    comment = comment.strip()
    if comment in ["gt_dendrite", "shaft", "d"]:
        return 0
    elif comment == "d_end":
        return 0 if not convert_to_morphx else 13
    elif comment in ["gt_axon", "a", "axon"]:
        return 1
    elif comment == "a_end":
        return 1 if not convert_to_morphx else 14
    elif comment in ["gt_soma", "other", "s"]:
        return 2
    elif comment in ["gt_bouton", "b", "bouton"]:
        return 3
    elif comment in ["gt_terminal", "t", "terminal"]:
        return 4
    elif comment in ["gt_neck", "neck", "n"]:
        return 5
    elif comment in ["gt_head", "head", "h"]:
        return 6
    elif comment in ["nr"]:
        return 7
    elif comment in ["in"]:
        return 8
    elif comment in ["p"]:
        return 9
    elif comment in ["st"]:
        return 10
    elif comment in ["ignore", "end"]:
        return 11
    elif comment in ["merger"]:
        return 12
    else:
        # unidentified label names get the label from the nearest node with appropriate label
        return -1


def sso2kzip(id: int, ssd: SuperSegmentationDataset, ouput_path: str, skeleton: bool = True):
    sso = ssd.get_super_segmentation_object(id)
    sso.load_attr_dict()
    sso.meshes2kzip(ouput_path, synssv_instead_sj=True)
    if skeleton:
        sso.load_skeleton()
        sso.save_skeleton_to_kzip(ouput_path)


def anno_skeleton2np(kzip, scaling, verbose=False, convert_to_morphx=False):
    a_obj = load_skeleton(kzip)
    try:
        a_obj = dict(skeleton=a_obj['skeleton'])
    except KeyError:
        a_obj = dict(skeleton=a_obj[''])
    a_obj = list(a_obj.values())[0]
    a_nodes = list(a_obj.getNodes())
    a_node_coords = np.array([n.getCoordinate() * scaling for n in a_nodes])
    a_node_labels = np.array([comment2int(n.getComment(), convert_to_morphx) for n in a_nodes], dtype=np.int)
    a_node_labels_raw = np.array([n.getComment() for n in a_nodes])
    # generate graph from nodes in annotation object
    a_edges = []
    if verbose:
        print("Building graph...")
    for node in tqdm(a_nodes, disable=not verbose):
        ix = a_nodes.index(node)
        neighbors = node.getNeighbors()
        for neighbor in neighbors:
            nix = a_nodes.index(neighbor)
            a_edges.append((ix, nix))
    g = nx.Graph()
    g.add_nodes_from([(i, dict(label=a_node_labels[i])) for i in range(len(a_nodes))])
    g.add_edges_from(a_edges)
    a_edges = np.array(g.edges)
    a_node_labels_orig = np.array(a_node_labels)
    # propagate labels, nodes with no label get label from nearest node with label
    if -1 in a_node_labels:
        if verbose:
            print("Propagating labels...")
        nodes = np.array(g.nodes)
        # shuffling to make sure the label searches are not done for consecutive nodes => brings potential speedup
        np.random.shuffle(nodes)
        for node in tqdm(nodes, disable=not verbose):
            if a_node_labels[node] == -1:
                ix = label_search(g, node)
                if ix == -1:
                    a_node_labels[node] = 11
                else:
                    # all nodes between source and first node with label take on that label
                    path = nx.shortest_path(g, node, ix)
                    a_node_labels[path] = a_node_labels[ix]
    return a_node_coords, a_edges, a_node_labels, a_node_labels_raw, g, a_node_labels_orig
