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
import pickle as pkl
from syconn.proc.meshes import mesh_area_calc
from syconn.handler.multiviews import generate_rendering_locs
from syconn.proc.graphs import create_graph_from_coords
from knossos_utils.skeleton_utils import load_skeleton
from sklearn.neighbors import KDTree
from syconn.reps.super_segmentation import SuperSegmentationObject
from syconn.reps.super_segmentation_helper import create_sso_skeleton_fast
from syconn import global_params
from syconn.mp.mp_utils import start_multiprocess_imap
from multiprocessing import cpu_count


def comment2int(comment: str):
    if comment == "gt_dendrite":
        return 0
    elif comment == "gt_axon":
        return 1
    elif comment == "gt_soma":
        return 2
    elif comment == "gt_bouton":
        return 3
    elif comment == "gt_terminal":
        return 4
    elif comment == "gt_neck":
        return 5
    elif comment == "gt_head":
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

    encoding = {0: 'dendrite', 1: 'axon', 2: 'soma', 3: 'bouton', 4: 'terminal',
                5: 'neck', 6: 'head', 7: 'mi', 8: 'vc', 9: 'sj'}

    # load cell and cell organelles (order of meshes in array is important for later merging process)
    meshes = [sso.mesh, sso.mi_mesh, sso.vc_mesh, sso.syn_ssv_mesh]
    mesh_areas = [mesh_area_calc(meshes[0]), mesh_area_calc(meshes[1]),
                  mesh_area_calc(meshes[2]), mesh_area_calc(meshes[3])]
    label_map = [-1, 7, 8, 9]
    vertices_l = 0
    indices_l = 0

    # find sizes
    for mesh in meshes:
        indices, vertices, normals = mesh
        vertices = vertices.reshape((-1, 3))
        indices = indices.reshape((-1, 3))
        vertices_l += len(vertices)
        indices_l += len(indices)

    # prepare merged arrays
    t_vertices = np.zeros((vertices_l, 3))
    t_indices = np.zeros((indices_l, 3))
    t_labels = np.zeros((vertices_l, 1))

    # merge meshes
    vertices_o = 0
    indices_o = 0

    for ix, mesh in enumerate(meshes):
        indices, vertices, normals = mesh
        vertices = vertices.reshape((-1, 3))
        indices = indices.reshape((-1, 3))

        t_vertices[vertices_o:vertices_o+len(vertices)] = vertices
        t_labels[vertices_o:vertices_o+len(vertices)] = label_map[ix]
        vertices_o += len(vertices)
        t_indices[indices_o:indices_o+len(indices)] = indices
        indices_o += len(indices)

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

    # filter nodes where label = -1
    a_node_coords = a_node_coords[(a_node_labels != -1)]
    a_node_labels = a_node_labels[(a_node_labels != -1)]

    # create KD tree from skeleton node coordinates
    tree = KDTree(a_node_coords)

    # transfer labels from skeleton to mesh
    indices, vertices, normals = meshes[0]
    vertices = vertices.reshape((-1, 3))
    dist, ind = tree.query(vertices, k=1)  # k-nearest neighbour
    vertex_labels = a_node_labels[ind]  # retrieving labels of vertices

    # save labels in merged label array
    t_labels[0:len(vertex_labels)] = vertex_labels

    # load skeleton (skeletons were already generated before)
    sso = create_sso_skeleton_fast(sso, max_dist_thresh_iter2=10000)
    skel = sso.skeleton
    skel['nodes'] = skel['nodes']*sso.scaling

    # pack all results into single dict
    gt_dict = {'nodes': skel['nodes'], 'edges': skel['edges'], 'vertices': t_vertices,
               'indices': t_indices, 'normals': np.array([]), 'labels': t_labels, 'encoding': encoding,
               'areas': mesh_areas}

    # save gt as pickle
    with open("{}/sso_{}.pkl".format(out_path, sso.id), 'wb') as f:
        pkl.dump(gt_dict, f)


def gt_generation(kzip_paths, dest_dir=None):
    if not os.path.isdir(dest_dir):
        os.makedirs(dest_dir)

    dest_p_results = "{}/gt_all/".format(dest_dir)
    if not os.path.isdir(dest_p_results):
        os.makedirs(dest_p_results)

    params = [(p, dest_p_results) for p in kzip_paths]

    # start mapping for each kzip in kzip_paths
    start_multiprocess_imap(labels2mesh, params, nb_cpus=cpu_count(), debug=False)


if __name__ == "__main__":
    # set paths
    dest_gt_dir = "/wholebrain/u/jklimesch/gt/"
    global_params.wd = "/wholebrain/songbird/j0126/areaxfs_v6/"
    label_file_folder = "/wholebrain/u/jklimesch/gt/gt_julian/"

    file_paths = glob.glob(label_file_folder + '*.k.zip', recursive=False)

    # generate ground truth
    gt_generation(file_paths, dest_dir=dest_gt_dir)
