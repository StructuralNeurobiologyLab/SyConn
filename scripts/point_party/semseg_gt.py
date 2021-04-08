# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max-Planck-Institute of Neurobiology, Munich, Germany
# Authors: Jonathan Klimesch, Philipp Schubert

import os
import glob
import re
import numpy as np
import networkx as nx
from collections import deque
from syconn.proc.meshes import write_mesh2kzip
from syconn.handler.basics import load_pkl2obj
from sklearn.neighbors import KDTree
from scipy.spatial import cKDTree
from knossos_utils.skeleton_utils import load_skeleton
from syconn.reps.super_segmentation_helper import map_myelin2coords, majorityvote_skeleton_property

from syconn import global_params
from syconn.mp.mp_utils import start_multiprocess_imap
from multiprocessing import cpu_count
from morphx.classes.cloudensemble import CloudEnsemble
from morphx.classes.hybridmesh import HybridMesh, HybridCloud
from syconn.reps.super_segmentation import SuperSegmentationObject
from syconn.handler.prediction_pts import pts_feat_dict


def labels2mesh(args):
    """
    Args:
        args: Tuple of the following values:
            kzip_path: path to current sso.
            out_path: path to folder where output should be saved.
    """
    kzip_path, out_path, version = args
    sso_id = int(re.findall(r"/(\d+)\.", kzip_path)[0])
    # load annotation object
    a_obj = load_skeleton(kzip_path)
    if len(a_obj) == 1:
        a_obj = list(a_obj.values())[0]
    elif str(sso_id) in a_obj:
        a_obj = a_obj[str(sso_id)]
    else:
        a_obj = a_obj["skeleton"]
    a_nodes = list(a_obj.getNodes())

    # load and prepare sso
    sso = SuperSegmentationObject(sso_id, version=version)
    sso.load_attr_dict()
    # load skeleton (skeletons were already generated before)
    sso.load_skeleton()
    skel = sso.skeleton
    nodes = skel['nodes'] * sso.scaling
    edges = skel['edges']

    # extract node coordinates and labels and remove nodes with label -1
    a_node_coords = np.array([n.getCoordinate() * sso.scaling for n in a_nodes])
    a_node_labels = np.array([comment2int(n.getComment()) for n in a_nodes], dtype=np.int32)
    a_node_coords = a_node_coords[(a_node_labels != -1)]
    a_node_labels = a_node_labels[(a_node_labels != -1)]

    # get sso skeleton nodes which are close to GT nodes
    # 0: not close to a labeled GT node, 1: close to an annotated node
    kdt = cKDTree(a_node_coords)
    node_labels = np.ones((len(nodes), 1))
    dists, ixs = kdt.query(nodes, distance_upper_bound=1000)
    node_labels[dists == np.inf] = 0
    dists, ixs = kdt.query(nodes)
    node_labels[a_node_labels[ixs] == 33] = 0

    kdt = cKDTree(a_node_coords)
    # load cell and cell organelles
    # _ = sso._load_obj_mesh('syn_ssv', rewrite=True)
    meshes = [sso.mesh, sso.mi_mesh, sso.vc_mesh, sso.syn_ssv_mesh]
    feature_map = dict(pts_feat_dict)

    # create cloud ensemble
    encoding = {'dendrite': 0, 'axon': 1, 'soma': 2, 'bouton': 3, 'terminal': 4, 'neck': 5, 'head': 6}
    obj_names = ['sv', 'mi', 'vc', 'syn_ssv']
    verts_tot = []
    feats_tot = []
    labels_tot = []
    for ix, mesh in enumerate(meshes):
        indices, vertices, normals = mesh
        vertices = vertices.reshape((-1, 3))
        labels = np.ones((len(vertices), 1)) * -1
        feats = np.ones((len(vertices), 1)) * feature_map[obj_names[ix]]

        # remove ignore and glia vertices
        _, ixs = kdt.query(vertices)
        vert_l = a_node_labels[ixs]

        if ix == 0:
            # save colored mesh
            col_lookup = {0: (125, 125, 125, 255), 1: (255, 125, 125, 255), 2: (125, 255, 125, 255),
                          3: (125, 125, 255, 255),
                          4: (255, 255, 125, 255), 5: (125, 255, 255, 255), 6: (255, 125, 255, 255), 7: (0, 0, 0, 255),
                          8: (255, 0, 0, 255), 33: (238, 238, 0, 255)}
            cols = np.array([col_lookup[el] for el in vert_l.squeeze()], dtype=np.uint8)
            write_mesh2kzip(f'{out_path}/sso_{sso.id}.k.zip', indices.astype(np.float32),
                            vertices.astype(np.float32), None, cols, f'{sso_id}.ply')

        vertices = vertices[vert_l != 33]
        feats = feats[vert_l != 33]
        # only set labels for cell surface points -> subcellular structures will have label -1
        if ix == 0:
            labels = vert_l[vert_l != 33][:, None]
        else:
            labels = labels[vert_l != 33]

        verts_tot.append(vertices)
        feats_tot.append(feats)
        labels_tot.append(labels)

    verts_tot = np.concatenate(verts_tot)
    feats_tot = np.concatenate(feats_tot).astype(np.uint8)
    # labels contain negative integer
    labels_tot = np.concatenate(labels_tot).astype(np.int16)
    # print(sso_id, np.unique(labels_tot, return_counts=True), labels_tot.shape, verts_tot.shape, feats_tot.shape)
    print(sso_id, np.unique(node_labels, return_counts=True))

    hc = HybridCloud(vertices=verts_tot, features=feats_tot, labels=labels_tot,
                     nodes=nodes, node_labels=node_labels, edges=edges)

    # # add myelin (see docstring of map_myelin2coords)
    # sso.skeleton['myelin'] = map_myelin2coords(sso.skeleton["nodes"], mag=4)
    # majorityvote_skeleton_property(sso, 'myelin')
    # myelinated = sso.skeleton['myelin_avg10000']
    # nodes_idcs = np.arange(len(hm.nodes))
    # myel_nodes = nodes_idcs[myelinated.astype(bool)]
    # myel_vertices = []
    # for node in myel_nodes:
    #     myel_vertices.extend(hm.verts2node[node])
    # # myelinated vertices get type 1, not myelinated vertices get type 0
    # types = np.zeros(len(hm.vertices))
    # types[myel_vertices] = 1
    # hm.set_types(types)

    # save generated hybrid cloud to pkl
    hc.save2pkl(f'{out_path}/sso_{sso.id}.pkl')


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
    elif comment in ["gt_neck", 'neck', 'in', 'nr']:  # 'in' and 'nr' neck sub classes
        return 5
    elif comment in ["gt_head", "head", 's', 'gt_protrusion', 'p']:  # 's' for stubby, 'p' for protrusion
        return 6
    elif comment == 'gt_glia' or comment == 'ignore':
        return 33
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
    data_path = "/wholebrain/songbird/j0126/GT/compartment_gt_2020/2020_05/"
    destination = data_path + '/hc_out_2020_08/'
    os.makedirs(destination, exist_ok=True)

    # axon GT
    global_params.wd = "/wholebrain/songbird/j0126/areaxfs_v6/"
    file_paths = glob.glob(data_path + '*.k.zip', recursive=False)
    file_paths += glob.glob(data_path + '/batch2/*.k.zip', recursive=False)

    gt_generation(file_paths, destination)

    # # spine GT
    # global_params.wd = "/wholebrain/scratch/areaxfs3/"
    # data_path = data_path + "/sparse_gt/spgt/"
    # file_paths = glob.glob(data_path + '*.k.zip', recursive=False)
    # gt_generation(file_paths, destination, version='spgt')
