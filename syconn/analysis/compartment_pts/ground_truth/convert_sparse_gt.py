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
from tqdm import tqdm
from knossos_utils.skeleton_utils import load_skeleton
from scipy.spatial import cKDTree
from syconn import global_params
from syconn.mp.mp_utils import start_multiprocess_imap
from multiprocessing import cpu_count
from morphx.classes.cloudensemble import CloudEnsemble
from morphx.classes.hybridcloud import HybridCloud
from syconn.reps.super_segmentation import SuperSegmentationObject


def labels2mesh(args):
    kzip_path, out_path, version = args

    # load and prepare sso
    sso_id = int(re.findall(r"/(\d+).", kzip_path)[0])
    sso = SuperSegmentationObject(sso_id, version=version)
    sso.load_attr_dict()

    # load cell and cell organelles
    meshes = [sso.mesh, sso.mi_mesh, sso.vc_mesh]
    # load new synapse version
    meshes.append(sso._load_obj_mesh('sj', rewrite=False))
    label_map = [-1, 7, 8, 9]
    hcs = []
    for ix, mesh in enumerate(meshes):
        indices, vertices, normals = mesh
        vertices = vertices.reshape((-1, 3))
        labels = np.ones((len(vertices), 1)) * label_map[ix]
        hc = HybridCloud(vertices=vertices, labels=labels)
        hcs.append(hc)

    # load annotation object and corresponding skeleton
    a_obj = load_skeleton(kzip_path)
    if len(a_obj) != 1:
        raise ValueError("File contains more or less than one skeleton!")
    a_obj = list(a_obj.values())[0]
    a_nodes = list(a_obj.getNodes())
    a_node_coords = np.array([n.getCoordinate() * sso.scaling for n in a_nodes]) + sso.scaling
    a_node_labels = np.array([comment2int(n.getComment()) for n in a_nodes], dtype=np.int)

    sso.load_skeleton()
    skel = sso.skeleton
    nodes = skel['nodes'] * sso.scaling
    edges = skel['edges']

    tree = cKDTree(a_node_coords)
    dists, idcs = tree.query(nodes, k=1)
    node_labels = a_node_labels[idcs]

    # create cloud ensemble
    encoding = {'dendrite': 0, 'axon': 1, 'soma': 2, 'bouton': 3, 'terminal': 4, 'neck': 5, 'head': 6}
    obj_names = ['sv', 'mi', 'vc', 'sy']
    hc = None
    clouds = {}
    for ix, cloud in enumerate(hcs):
        if ix == 0:
            vertices = hcs[0].vertices
            labels = induce_vertex_label(vertices, nodes, node_labels)
            hc = HybridCloud(vertices=vertices, labels=labels, nodes=nodes, edges=edges, encoding=encoding,
                             node_labels=node_labels)
        else:
            hcs[ix].set_encoding({obj_names[ix]: label_map[ix]})
            clouds[obj_names[ix]] = hcs[ix]
    ce = CloudEnsemble(clouds, hc, no_pred=['mi', 'vc', 'sy'])

    # save generated cloud ensemble to file
    ce.save2pkl(f'{out_path}/{sso.id}.pkl')


def induce_vertex_label(verts: np.ndarray, nodes: np.ndarray, nlabels: np.ndarray) -> np.ndarray:
    tree = cKDTree(nodes[nlabels != -1])
    dist, idcs = tree.query(verts, k=1)
    labels = nlabels[nlabels != -1][idcs]
    return labels


def comment2int(comment: str):
    """ Map comments used during annotation to respective label. """
    if comment == "gt_dendrite" or comment == "shaft" or comment == "s":
        return 0
    elif comment == "gt_axon":
        return 1
    elif comment == "gt_soma" or comment == "other":
        return 2
    elif comment == "gt_bouton":
        return 3
    elif comment == "gt_terminal":
        return 4
    elif comment == "gt_neck" or comment == "neck" or comment == "p" or comment == "nr" or comment == "in":
        return 5
    elif comment == "gt_head" or comment == "head":
        return 6
    else:
        return -1


def gt_generation(kzip_paths, out_path, version: str = None):
    if not os.path.isdir(out_path):
        os.makedirs(out_path)

    params = [(p, out_path, version) for p in kzip_paths]
    # labels2mesh(params[2])
    # start mapping for each kzip in kzip_paths
    start_multiprocess_imap(labels2mesh, params, nb_cpus=cpu_count(), debug=False)


if __name__ == "__main__":
    destination = "/wholebrain/u/jklimesch/thesis/gt/cmn/dnh/sparse/"
    data_path = "/wholebrain/u/jklimesch/thesis/gt/cmn/dnh/annotations/v1/"
    file_paths = glob.glob(data_path + '*.k.zip', recursive=False)
    # spine GT
    global_params.wd = "/wholebrain/scratch/areaxfs3/"
    gt_generation(file_paths, destination, version='spgt')
    # axon GT
    # global_params.wd = "/wholebrain/songbird/j0126/areaxfs_v6/"
    # gt_generation(file_paths, destination)
