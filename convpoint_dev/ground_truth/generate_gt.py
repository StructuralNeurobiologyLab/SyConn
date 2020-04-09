# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max-Planck-Institute of Neurobiology, Munich, Germany
# Authors: Jonathan Klimesch

import numpy as np
import os
import glob
import re
from knossos_utils.skeleton_utils import load_skeleton
from sklearn.neighbors import KDTree
from syconn.reps.super_segmentation import SuperSegmentationObject
from syconn import global_params
from syconn.mp.mp_utils import start_multiprocess_imap
from multiprocessing import cpu_count
from morphx.classes.hybridmesh import HybridMesh
from morphx.classes.cloudensemble import CloudEnsemble


def comment2int(comment: str):
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


def labels2mesh(args):
    """
    Args:
        args: Tuple of the following values:
            kzip_path: path to current sso.
            out_path: path to folder where output should be saved.
    """

    kzip_path, out_path = args

    # get sso
    sso_id = int(re.findall(r"/(\d+).", kzip_path)[0])
    sso = SuperSegmentationObject(sso_id)
    sso.load_attr_dict()

    # load cell and cell organelles (order of meshes in array is important for later merging process)
    meshes = [sso.mesh, sso.mi_mesh, sso.vc_mesh, sso.syn_ssv_mesh]
    label_map = [-1, 7, 8, 9]

    hms = []
    for ix, mesh in enumerate(meshes):
        indices, vertices, normals = mesh
        vertices = vertices.reshape((-1, 3))
        labels = np.ones((len(vertices), 1)) * label_map[ix]
        indices = indices.reshape((-1, 3))
        hm = HybridMesh(vertices=vertices, faces=indices, labels=labels)
        hms.append(hm)

    # CREATE CELL HYBRIDMESH #

    # load annotation object
    a_obj = load_skeleton(kzip_path)
    if len(a_obj) == 1:
        a_obj = list(a_obj.values())[0]
    else:
        a_obj = a_obj["skeleton"]
    a_nodes = list(a_obj.getNodes())

    # extract node coordinates and labels
    a_node_coords = np.array([n.getCoordinate() * sso.scaling for n in a_nodes])
    a_node_labels = np.array([comment2int(n.getComment()) for n in a_nodes], dtype=np.int)

    # remove nodes where label = -1
    a_node_coords = a_node_coords[(a_node_labels != -1)]
    a_node_labels = a_node_labels[(a_node_labels != -1)]

    # create KD tree from skeleton node coordinates
    tree = KDTree(a_node_coords)

    # transfer labels from skeleton to mesh
    vertices = hms[0].vertices
    dist, ind = tree.query(vertices, k=1)  # k-nearest neighbour
    vertex_labels = a_node_labels[ind]  # retrieving labels of vertices

    # load skeleton (skeletons were already generated before)
    # sso = create_sso_skeleton_fast(sso, max_dist_thresh_iter2=10000)

    sso.load_skeleton()
    skel = sso.skeleton
    nodes = skel['nodes']*sso.scaling
    edges = skel['edges']

    encoding = {'dendrite': 0, 'axon': 1, 'soma': 2, 'bouton': 3, 'terminal': 4, 'neck': 5, 'head': 6}

    # CREATE CLOUD SET #

    obj_names = ['hc', 'mi', 'vc', 'sy']
    hm = None
    clouds = {}
    for ix, cloud in enumerate(hms):
        if ix == 0:
            hm = HybridMesh(vertices=vertices, labels=vertex_labels, faces=hms[0].faces, nodes=nodes,
                            edges=edges, encoding=encoding)
        else:
            hms[ix].set_encoding({obj_names[ix]: label_map[ix]})
            clouds[obj_names[ix]] = hms[ix]

    ce = CloudEnsemble(clouds, hm, no_pred=['mi', 'vc', 'sy'])
    ce.save2pkl(f'{out_path}/sso_{sso.id}.pkl')


def gt_generation(kzip_paths, out_path):
    if not os.path.isdir(out_path):
        os.makedirs(out_path)

    params = [(p, out_path) for p in kzip_paths]

    # start mapping for each kzip in kzip_paths
    start_multiprocess_imap(labels2mesh, params, nb_cpus=cpu_count(), debug=False)


if __name__ == "__main__":
    # set paths
    destination = "/wholebrain/u/jklimesch/thesis/gt/gt_meshsets/"
    global_params.wd = "/wholebrain/songbird/j0126/areaxfs_v6/"
    data_path = "/wholebrain/u/jklimesch/thesis/gt/gt_julian/"

    file_paths = glob.glob(data_path + '*.k.zip', recursive=False)

    # generate ground truth
    gt_generation(file_paths, destination)
