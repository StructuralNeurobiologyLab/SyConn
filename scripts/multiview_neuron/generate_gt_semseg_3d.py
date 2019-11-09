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
from knossos_utils.skeleton_utils import load_skeleton
from sklearn.neighbors import KDTree
from syconn.proc.meshes import write_mesh2kzip
from syconn.reps.super_segmentation import SuperSegmentationObject, SuperSegmentationDataset
from syconn import global_params
from syconn.handler.multiviews import str2intconverter
from syconn.mp.mp_utils import start_multiprocess_imap
from multiprocessing import cpu_count
from collections import defaultdict


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
    sso = SuperSegmentationObject(sso_id, version='semsegaxoness')

    # load mesh
    indices, vertices, normals = sso.mesh
    vertices = vertices.reshape((-1, 3))

    # load annotation object
    a_obj = load_skeleton(kzip_path)
    if len(a_obj) == 1:
        a_obj = list(a_obj.values())[0]
    else:
        a_obj = a_obj["skeleton"]
    a_nodes = list(a_obj.getNodes())

    # extract node coordinates and labels
    a_node_coords = np.array([n.getCoordinate() * sso.scaling for n in a_nodes])
    a_node_labels = np.array([str2intconverter(n.getComment(), 'axgt') for n in a_nodes], dtype=np.int)

    # filter nodes where label = -1
    a_node_coords = a_node_coords[(a_node_labels != -1)]
    a_node_labels = a_node_labels[(a_node_labels != -1)]

    # create KD tree from skeleton node coordinates
    tree = KDTree(a_node_coords)

    # transfer labels from skeleton to mesh
    dist, ind = tree.query(vertices, k=1)  # k-nearest neighbour
    vertex_labels = a_node_labels[ind]  # retrieving labels of vertices

    # load skeleton
    sso.load_skeleton()
    skel = sso.skeleton

    tree = KDTree(skel['nodes']*sso.scaling)
    dist, ind = tree.query(vertices, k=1)

    # create mapping array between skeleton nodes and mesh nodes
    skel2mesh_dict = defaultdict(list)
    for vertex_idx, skel_idx in enumerate(ind):
        skel2mesh_dict[skel_idx[0]].append(vertex_idx)

    # pack all results into single dict
    gt_dict = {'skel_nodes': skel['nodes']*sso.scaling, 'skel_edges': skel['edges'], 'mesh_verts': vertices,
               'vert_labels': vertex_labels, 'skel2mesh': skel2mesh_dict}

    # save training info as pickle
    with open("{}/sso_{}_info.pkl".format(out_path, sso.id), 'wb') as f:
        pkl.dump(gt_dict, f)

    if out_path is not None:
        # dendrite, axon, soma, bouton, terminal, background
        colors = [[0.6, 0.6, 0.6, 1], [0.9, 0.2, 0.2, 1], [0.1, 0.1, 0.1, 1],
                  [0.05, 0.6, 0.6, 1], [0.6, 0.05, 0.05, 1], [0.9, 0.9, 0.9, 1]]
        colors = (np.array(colors) * 255).astype(np.uint8)
        color_array_mesh = colors[vertex_labels][:, 0]
        write_mesh2kzip("{}/sso_{}_gtlabels.k.zip".format(out_path, sso.id),
                        sso.mesh[0], sso.mesh[1], sso.mesh[2], color_array_mesh,
                        ply_fname="gtlabels.ply")


def gt_generation(kzip_paths, dest_dir=None):
    # set up destination path
    if dest_dir is None:
        dest_dir = os.path.expanduser("~/semseg/")
    if not os.path.isdir(dest_dir):
        os.makedirs(dest_dir)

    dest_p_results = "{}/gt_results/".format(dest_dir)
    if not os.path.isdir(dest_p_results):
        os.makedirs(dest_p_results)

    params = [(p, dest_p_results) for p in kzip_paths]

    # start mapping for each kzip in kzip_paths
    start_multiprocess_imap(labels2mesh, params, nb_cpus=cpu_count(), debug=False)


if __name__ == "__main__":
    # set paths
    dest_gt_dir = "/wholebrain/u/jklimesch/gt/"
    global_params.wd = "/wholebrain/songbird/j0126/areaxfs_v6/"
    label_file_folder = "/wholebrain/scratch/areaxfs3/ssv_semsegaxoness/gt_axoness_semseg_skeletons" \
                        "/NEW_including_boutons/batch2_results_v2/"

    file_paths = glob.glob(label_file_folder + '*.k.zip', recursive=False)

    # generate ground truth
    gt_generation(file_paths, dest_dir=dest_gt_dir)
