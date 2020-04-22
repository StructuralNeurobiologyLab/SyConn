# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max-Planck-Institute of Neurobiology, Munich, Germany
# Authors: Jonathan Klimesch

import os
import glob
import re
import numpy as np
import networkx as nx
from collections import deque
from sklearn.neighbors import KDTree
from knossos_utils.skeleton_utils import load_skeleton
from syconn.reps.super_segmentation_helper import map_myelin2coords, majorityvote_skeleton_property

from syconn import global_params
from syconn.mp.mp_utils import start_multiprocess_imap
from multiprocessing import cpu_count
from morphx.classes.cloudensemble import CloudEnsemble
from morphx.classes.hybridmesh import HybridMesh
from syconn.reps.super_segmentation import SuperSegmentationObject


def labels2mesh(args):
    """
    Args:
        args: Tuple of the following values:
            kzip_path: path to current sso.
            out_path: path to folder where output should be saved.
    """
    kzip_path, out_path, version = args

    # load and prepare sso
    sso_id = int(re.findall(r"/(\d+).", kzip_path)[0])
    sso = SuperSegmentationObject(sso_id, version=version)
    sso.load_attr_dict()

    # load cell and cell organelles
    meshes = [sso.mesh, sso.mi_mesh, sso.vc_mesh, sso.sj_mesh]
    label_map = [-1, 7, 8, 9]
    hms = []
    for ix, mesh in enumerate(meshes):
        indices, vertices, normals = mesh
        vertices = vertices.reshape((-1, 3))
        labels = np.ones((len(vertices), 1)) * label_map[ix]
        indices = indices.reshape((-1, 3))
        hm = HybridMesh(vertices=vertices, faces=indices, labels=labels)
        hms.append(hm)

    # load annotation object
    a_obj = load_skeleton(kzip_path)
    if len(a_obj) == 1:
        a_obj = list(a_obj.values())[0]
    else:
        a_obj = a_obj["skeleton"]
    a_nodes = list(a_obj.getNodes())

    # extract node coordinates and labels and remove nodes with label -1
    a_node_coords = np.array([n.getCoordinate() * sso.scaling for n in a_nodes])
    a_node_labels = np.array([comment2int(n.getComment()) for n in a_nodes], dtype=np.int)
    a_node_coords = a_node_coords[(a_node_labels != -1)]
    a_node_labels = a_node_labels[(a_node_labels != -1)]

    # load skeleton (skeletons were already generated before)
    sso.load_skeleton()
    skel = sso.skeleton
    nodes = skel['nodes'] * sso.scaling
    edges = skel['edges']
    node_labels = np.ones((len(nodes), 1)) * -1

    # create KD tree for mapping existing labels from annotation skeleton to real skeleton
    tree = KDTree(nodes)
    dist, ind = tree.query(a_node_coords, k=1)
    node_labels[ind.reshape(len(ind))] = a_node_labels.reshape(-1, 1)

    # nodes without label get label from nearest node with label
    g = nx.Graph()
    g.add_nodes_from([(i, dict(label=node_labels[i])) for i in range(len(nodes))])
    g.add_edges_from([(edges[i][0], edges[i][1]) for i in range(len(edges))])
    for node in g.nodes:
        if g.nodes[node]['label'] == -1:
            ix = label_search(g, node)
            node_labels[node] = node_labels[ix]

    # create cloud ensemble
    encoding = {'dendrite': 0, 'axon': 1, 'soma': 2, 'bouton': 3, 'terminal': 4, 'neck': 5, 'head': 6}
    obj_names = ['hc', 'mi', 'vc', 'sy']
    hm = None
    clouds = {}
    for ix, cloud in enumerate(hms):
        if ix == 0:
            vertices = hms[0].vertices
            hm = HybridMesh(vertices=vertices, labels=np.ones(len(vertices))*-1, faces=hms[0].faces, nodes=nodes,
                            edges=edges, encoding=encoding, node_labels=node_labels)
            hm.nodel2vertl()
        else:
            hms[ix].set_encoding({obj_names[ix]: label_map[ix]})
            clouds[obj_names[ix]] = hms[ix]
    ce = CloudEnsemble(clouds, hm, no_pred=['mi', 'vc', 'sy'])

    # add myelin (see docstring of map_myelin2coords)
    sso.skeleton['myelin'] = map_myelin2coords(sso.skeleton["nodes"], mag=4)
    majorityvote_skeleton_property(sso, 'myelin')
    myelinated = sso.skeleton['myelin_avg10000']
    nodes_idcs = np.arange(len(hm.nodes))
    myel_nodes = nodes_idcs[myelinated.astype(bool)]
    myel_vertices = []
    for node in myel_nodes:
        myel_vertices.extend(hm.verts2node[node])
    # myelinated vertices get type 1, not myelinated vertices get type 0
    types = np.zeros(len(hm.vertices))
    types[myel_vertices] = 1
    hm.set_types(types)

    # save generated cloud ensemble to file
    ce.save2pkl(f'{out_path}/sso_{sso.id}.pkl')


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


def gt_generation(kzip_paths, out_path, version: str = None):
    if not os.path.isdir(out_path):
        os.makedirs(out_path)

    params = [(p, out_path, version) for p in kzip_paths]
    # labels2mesh(params[1])
    # start mapping for each kzip in kzip_paths
    start_multiprocess_imap(labels2mesh, params, nb_cpus=cpu_count(), debug=False)


if __name__ == "__main__":
    destination = "/wholebrain/u/jklimesch/thesis/gt/intermediate/"
    data_path = "/wholebrain/u/jklimesch/thesis/gt/annotations/sparse_gt/spgt/"
    file_paths = glob.glob(data_path + '*.k.zip', recursive=False)
    # spine GT
    global_params.wd = "/wholebrain/scratch/areaxfs3/"
    gt_generation(file_paths, destination, version='spgt')
    # axon GT
    # global_params.wd = "/wholebrain/songbird/j0126/areaxfs_v6/"
    # gt_generation(file_paths, destination)
