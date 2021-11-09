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
from .utils import label_search, comment2int
from knossos_utils.skeleton_utils import load_skeleton
from syconn.reps.super_segmentation_helper import map_myelin2coords, majorityvote_skeleton_property

from syconn import global_params
from syconn.mp.mp_utils import start_multiprocess_imap
from multiprocessing import cpu_count
from morphx.classes.cloudensemble import CloudEnsemble
from morphx.classes.hybridmesh import HybridMesh
from morphx.classes.hybridcloud import HybridCloud
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
    meshes = [sso.mesh, sso.mi_mesh, sso.vc_mesh]
    # load new synapse version
    meshes.append(sso._load_obj_mesh('syn_ssv', rewrite=False))
    label_map = [-1, 7, 8, 9]
    hcs = []
    faces = None
    for ix, mesh in enumerate(meshes):
        indices, vertices, normals = mesh
        vertices = vertices.reshape((-1, 3))
        labels = np.ones((len(vertices), 1)) * label_map[ix]
        indices = indices.reshape((-1, 3))
        if ix == 0:
            faces = indices
        hc = HybridCloud(vertices=vertices, labels=labels)
        hcs.append(hc)

    # load annotation object and corresponding skeleton
    a_obj = load_skeleton(kzip_path)
    if len(a_obj) != 1:
        raise ValueError("File contains more or less than one skeleton!")
    a_obj = list(a_obj.values())[0]
    a_nodes = list(a_obj.getNodes())
    a_node_coords = np.array([n.getCoordinate() * sso.scaling for n in a_nodes])
    a_node_labels = np.array([comment2int(n.getComment()) for n in a_nodes], dtype=np.int)
    # generate graph from nodes in annotation object
    a_edges = []
    for node in a_nodes:
        ix = a_nodes.index(node)
        neighbors = node.getNeighbors()
        for neighbor in neighbors:
            nix = a_nodes.index(neighbor)
            a_edges.append((ix, nix))
    g = nx.Graph()
    g.add_nodes_from([(i, dict(label=a_node_labels[i])) for i in range(len(a_nodes))])
    g.add_edges_from(a_edges)
    a_edges = np.array(g.edges)
    # propagate labels, nodes with no label get label from nearest node with label
    if -1 in a_node_labels:
        cached_labels = a_node_labels.copy()
        for node in g.nodes:
            if g.nodes[node]['label'] == -1:
                ix = label_search(g, node)
                a_node_labels[node] = cached_labels[ix]

    # create cloud ensemble
    encoding = {'dendrite': 0, 'axon': 1, 'soma': 2, 'bouton': 3, 'terminal': 4, 'neck': 5, 'head': 6}
    obj_names = ['hc', 'mi', 'vc', 'sy']
    hc = None
    clouds = {}
    for ix, cloud in enumerate(hcs):
        if ix == 0:
            vertices = hcs[0].vertices
            hm = HybridCloud(vertices=vertices, labels=np.ones(len(vertices))*-1, nodes=a_node_coords, edges=a_edges,
                             encoding=encoding, node_labels=a_node_labels)
            # hm.node_sliding_window_bfs(predictions=False)
            hm.nodel2vertl()
            sso.load_skeleton()
            skel = sso.skeleton
            nodes = skel['nodes'] * sso.scaling
            edges = skel['edges']
            hc = HybridMesh(vertices=vertices, labels=hm.labels, nodes=nodes, edges=edges, encoding=hm.encoding,
                            faces=faces)
        else:
            hcs[ix].set_encoding({obj_names[ix]: label_map[ix]})
            clouds[obj_names[ix]] = hcs[ix]

    # add myelin (see docstring of map_myelin2coords)
    sso.load_skeleton()
    sso.skeleton['myelin'] = map_myelin2coords(sso.skeleton["nodes"], mag=4)
    majorityvote_skeleton_property(sso, 'myelin')
    myelinated = sso.skeleton['myelin_avg10000']
    hm_myelin = HybridCloud(vertices=hcs[0].vertices, nodes=sso.skeleton["nodes"]*sso.scaling)
    nodes_idcs = np.arange(len(hm_myelin.nodes))
    myel_nodes = nodes_idcs[myelinated.astype(bool)]
    myel_vertices = []
    for node in myel_nodes:
        myel_vertices.extend(hm_myelin.verts2node[node])
    # myelinated vertices get type 1, not myelinated vertices get type 0
    types = np.zeros(len(hc.vertices))
    types[myel_vertices] = 1
    hc.set_types(types)

    ce = CloudEnsemble(clouds, hc, no_pred=['mi', 'vc', 'sy'])
    # save generated cloud ensemble to file
    ce.save2pkl(f'{out_path}/sso_{sso.id}.pkl')


def gt_generation(kzip_paths, out_path, version: str = None):
    if not os.path.isdir(out_path):
        os.makedirs(out_path)

    params = [(p, out_path, version) for p in kzip_paths]
    start_multiprocess_imap(labels2mesh, params, nb_cpus=cpu_count(), debug=False)


if __name__ == "__main__":
    destination = "/wholebrain/u/jklimesch/thesis/gt/new_GT/"
    data_path = "/wholebrain/u/jklimesch/thesis/gt/new_GT/"
    file_paths = glob.glob(data_path + '*.k.zip', recursive=False)
    # spine GT
    # global_params.wd = "/wholebrain/scratch/areaxfs3/"
    # gt_generation(file_paths, destination, version='spgt')
    # axon GT
    global_params.wd = "/wholebrain/songbird/j0126/areaxfs_v6/"
    gt_generation(file_paths, destination)
