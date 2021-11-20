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

from tqdm import tqdm
from syconn import global_params
from syconn.mp.mp_utils import start_multiprocess_imap
from multiprocessing import cpu_count
from morphx.classes.cloudensemble import CloudEnsemble
from morphx.classes.hybridmesh import HybridMesh, HybridCloud
from syconn.reps.super_segmentation import SuperSegmentationObject
from syconn.handler.prediction_pts import pts_feat_dict


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


def anno_skeleton2np(a_obj, scaling, verbose=False):
    a_nodes = list(a_obj.getNodes())
    a_node_coords = np.array([n.getCoordinate() * scaling for n in a_nodes])
    a_node_labels = np.array([comment2int(n.getComment()) for n in a_nodes], dtype=np.int32)
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
    # remove labels on branches that are only at the soma
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
    # keep 13 and 14 in orig labels (so that they will not be used as starting nodes for context generation)
    a_node_labels_orig = np.array(a_node_labels)
    # use a_end and d_end as axon and dendrite label, but not as starting locations
    a_node_labels[a_node_labels == 13] = 0  # convert to dendrite
    a_node_labels[a_node_labels == 14] = 1  # convert to axon
    return a_node_coords, a_edges, a_node_labels, a_node_labels_raw, g, a_node_labels_orig


def labels2mesh(args):
    """
    Args:
        args: Tuple of the following values:
            kzip_path: path to current sso.
            out_path: path to folder where output should be saved.
    """
    kzip_path, out_path, version, overwrite = args
    if 'areaxfs3' in global_params.wd:
        sso_id = int(re.findall(r"(\d+).\d+.k.zip", os.path.split(kzip_path)[1])[0])
    else:
        sso_id = int(re.findall(r"_(\d+)", os.path.split(kzip_path)[1])[0])
    path2pkl = f'{out_path}/sso_{sso_id}.pkl'
    if os.path.isfile(path2pkl) and not overwrite:
        return
    sso = SuperSegmentationObject(sso_id, version=version)
    assert sso.attr_dict_exists
    # load annotation object
    a_obj = load_skeleton(kzip_path, scaling=sso.scaling)
    if len(a_obj) == 1:
        a_obj = list(a_obj.values())[0]
    elif str(sso_id) in a_obj:
        a_obj = a_obj[str(sso_id)]
    elif 'skeleton' in a_obj:
        a_obj = a_obj["skeleton"]
    elif 1 in a_obj:  # use first annotation object.. OBDA
        a_obj = a_obj[1]
    else:
        raise ValueError(f'Could not find annotation skeleton in "{kzip_path}".')

    label_mapping = label_mappings[TARGET_LABELS]
    num_class = class_nums[TARGET_LABELS]

    # load and prepare sso
    sso.load_attr_dict()

    # extract node coordinates and labels and remove nodes with label 11 (ignore)
    a_node_coords, a_edges, a_node_labels, a_node_labels_raw, g, a_node_labels_orig = \
        anno_skeleton2np(a_obj, scaling=sso.scaling)
    a_node_coords_orig = np.array(a_node_coords)

    if 'areaxfs3' in global_params.wd:
        sso.load_skeleton()
        nodes = sso.skeleton['nodes'] * sso.scaling
        edges = sso.skeleton['edges']
    else:
        # keep the kzip skeleton, as they might contain new edges and nodes
        nodes = np.array(a_node_coords)
        sso.skeleton = dict()
        sso.skeleton['nodes'] = nodes / sso.scaling
        sso.skeleton['edges'] = np.array(a_edges)
        sso.skeleton['diameters'] = np.ones((len(nodes), 1))
        edges = sso.skeleton['edges']

    # remove nodes with ignore labels
    for l_remove in label_remove[TARGET_LABELS]:
        a_node_coords = a_node_coords[(a_node_labels != l_remove)]
        a_node_labels = a_node_labels[(a_node_labels != l_remove)]
        a_node_coords_orig = a_node_coords_orig[(a_node_labels_orig != l_remove)]
        a_node_labels_orig = a_node_labels_orig[(a_node_labels_orig != l_remove)]
    # remap labels
    for orig, new in label_mapping:
        a_node_labels[a_node_labels == orig] = new
        a_node_labels_orig[a_node_labels_orig == orig] = new

    assert np.max(a_node_labels) <= num_class - 1
    assert np.max(a_node_labels_orig) <= num_class - 1

    # get sso skeleton nodes which are close to manually annotated GT nodes (i.e. do not start from nodes that received
    # their label by propagation
    # 0: not close to a labeled GT node, 1: close to an annotated node
    kdt = cKDTree(a_node_coords_orig)
    node_labels = np.ones((len(nodes), 1), dtype=np.int32)
    node_orig_labels = -np.ones((len(nodes), ), dtype=np.int32)
    # TODO: use graph traversal approach
    dists, ixs = kdt.query(nodes, distance_upper_bound=1000)
    node_labels[dists == np.inf] = 0

    node_orig_labels[dists != np.inf] = a_node_labels_orig[ixs[dists != np.inf]]
    kdt = cKDTree(a_node_coords)
    # load cell and cell organelles
    # _ = sso._load_obj_mesh('syn_ssv', rewrite=True)
    if global_params.wd == '/wholebrain/scratch/areaxfs3/':
        # syn_ssv do not exists there yet
        meshes = [sso.mesh, sso.mi_mesh, sso.vc_mesh, sso.sj_mesh]
    else:
        meshes = [sso.mesh, sso.mi_mesh, sso.vc_mesh, sso.syn_ssv_mesh]
    feature_map = dict(pts_feat_dict)

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
                          8: (255, 0, 0, 255)}
            cols = np.array([col_lookup[el] for el in vert_l.squeeze()], dtype=np.uint8)
            write_mesh2kzip(f'{out_path}/sso_{sso.id}.k.zip', indices.astype(np.float32),
                            vertices.astype(np.float32), None, cols, f'{sso_id}.ply')
            sso.skeleton['source'] = node_labels.squeeze()
            sso.skeleton['orig_labels'] = node_orig_labels.squeeze()
            sso.save_skeleton_to_kzip(f'{out_path}/sso_{sso.id}.k.zip', additional_keys=['source', 'orig_labels'])

        # only set labels for cell surface points -> subcellular structures will have label -1
        if ix == 0:
            labels = vert_l[:, None]
        else:
            labels = labels

        verts_tot.append(vertices)
        feats_tot.append(feats)
        labels_tot.append(labels)

    verts_tot = np.concatenate(verts_tot)
    feats_tot = np.concatenate(feats_tot).astype(np.uint8)
    # labels contain negative integer
    labels_tot = np.concatenate(labels_tot).astype(np.int16)
    # print(sso_id, np.unique(labels_tot, return_counts=True), labels_tot.shape, verts_tot.shape, feats_tot.shape)
    assert np.sum(node_labels) > 0, f'No valid no labels found for cell "{kzip_path}".'
    hc = HybridCloud(vertices=verts_tot, features=feats_tot, labels=labels_tot,
                     nodes=nodes, node_labels=node_labels, edges=edges)

    # # add myelin (see docstring of map_myelin2coords)
    # sso.skeleton['myelin'] = map_myelin2coords(sso.skeleton["nodes"], mag=4)
    # majorityvote_skeleton_property(sso, 'myelin')
    # myelinated = sso.skeleton['myelin_avg10000']
    # nodes_idcs = np.arange(len(hc.nodes))
    # myel_nodes = nodes_idcs[myelinated.astype(bool)]
    # myel_vertices = []
    # for node in myel_nodes:
    #     myel_vertices.extend(hc.verts2node[node])
    # # myelinated vertices get type 1, not myelinated vertices get type 0
    # types = np.zeros(len(hc.vertices))
    # types[myel_vertices] = 1
    # hc.set_types(types)

    # save generated hybrid cloud to pkl
    hc.save2pkl(path2pkl)


def comment2int(comment: str, convert_to_morphx: bool = True):
    """ Map comments used during annotation to respective label.

     encoding = {'dendrite': 0, 'axon': 1, 'soma': 2, 'bouton': 3,
            'terminal': 4, 'neck': 5, 'head': 6, 'nr': 7,
            'in': 8, 'p': 9, 'st': 10, 'ignore': 11, 'merger': 12,
            'pure_dendrite': 13, 'pure_axon': 14, 'soma_at_pure_comp': 15}

     """
    comment = comment.strip()
    if comment in ["gt_dendrite", "shaft", "d"]:
        return 0
    elif comment == "d_end":
        return 0 if not convert_to_morphx else 13
    elif comment in ["gt_axon", "a", "axon"]:
        return 1
    elif comment == "a_end":
        return 1 if not convert_to_morphx else 14
    elif comment == "s_end":
        return 1 if not convert_to_morphx else 15
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


"""
Label mappings must always result in a consecutive labeling starting from 0. Otherwise the loss calculation
will throw a CUDA error
j0126: 'dendrite': 0, 'axon': 1, 'soma': 2, 'bouton': 3, 'terminal': 4, 'neck': 5, 'head': 6

j0251: 'dendrite': 0, 'axon': 1, 'soma': 2, 'bouton': 3, 'terminal': 4, 'neck': 5, 'head': 6,
       'nr': 7, 'in': 8, 'p': 9, 'st': 10, 'ignore': 11, 'merger': 12, 'pure_dendrite': 13,
       'pure_axon': 14}
"""
# j0251 ignore labels - is applied before label_mappings from below!
label_remove = dict(
    # ignore "ignore", merger, pure dendrite and pure axon (TODO: what are those?!)
    fine=[11, 12, 13, 14, 15, -1],
    # ignore axon, soma, bouton, terminal
    dnh=[1, 2, 3, 4, 11, 12, 13, 14, 15, -1],
    # ignore dendrite, soma, neck, head
    abt=[0, 2, 5, 6, 11, 12, 13, 14, 15, -1],
    # ignore same as in "fine"
    ads=[11, 12, 13, 14, 15, -1],
)

# j0251 mappings
label_mappings = dict(fine=[(7, 5), (8, 5), (9, 5), (10, 6)],  # st (10; stubby) to "head"
                      # map nr, in, p, neck to "neck" (1) and head, st (10; stubby) to "head" (2)., dendrite stays 0
                      dnh=[(7, 1), (8, 1), (9, 1), (5, 1), (10, 2), (6, 2)],
                      # map axon to 0, bouton to 1 and terminal to 2
                      abt=[(1, 0), (3, 1), (4, 2)],
                      # map all dendritic compartments to dendrite (0) and all axonic to axon (1)
                      ads=[(7, 0), (8, 0), (9, 0), (5, 0), (10, 0), (6, 0), (3, 1), (4, 1)],
                      )

class_nums = dict(fine=7, dnh=3, abt=3, ads=3)
target_names = dict(fine=['dendrite', 'axon', 'soma', 'bouton', 'terminal', 'neck', 'head'],
                    dnh=['dendrite', 'neck', 'head'],
                    abt=['axon', 'bouton', 'terminal'],
                    ads=['dendrite', 'axon', 'soma'])


def gt_generation(kzip_paths, out_path, version: str = None, overwrite=True):
    if not os.path.isdir(out_path):
        os.makedirs(out_path)

    params = [(p, out_path, version, overwrite) for p in kzip_paths]
    # labels2mesh(params[1])
    # start mapping for each kzip in kzip_paths
    start_multiprocess_imap(labels2mesh, params, nb_cpus=10, debug=False)


if __name__ == "__main__":
    TARGET_LABELS = 'fine'  # 'ads'
    # j0251 GT refined (Nov 16, 2021)
    global_params.wd = "/ssdscratch/pschuber/songbird/j0251/rag_flat_Jan2019_v3/"

    data_path = "/wholebrain/songbird/j0251/groundtruth/compartment_gt/2021_11_subset/train/"
    destination = data_path + '/hc_out_2021_11_fine/'
    os.makedirs(destination, exist_ok=True)
    file_paths = glob.glob(data_path + '*.k.zip', recursive=False)

    gt_generation(file_paths, destination, overwrite=False)

    # -------- OLD ------------
    # # axon GT
    # global_params.wd = "/wholebrain/songbird/j0126/areaxfs_v6/"
    # file_paths = glob.glob(data_path + '*.k.zip', recursive=False)
    # file_paths += glob.glob(data_path + '/batch2/*.k.zip', recursive=False)
    #
    # gt_generation(file_paths, destination)

    # # spine GT (CMN paper)
    # data_path = "/wholebrain/songbird/j0126/GT/spgt_semseg/kzips/"
    # destination = data_path + '/pkl_files/'
    # global_params.wd = "/wholebrain/scratch/areaxfs3/"
    # file_paths = glob.glob(data_path + '*.k.zip', recursive=False)
    # gt_generation(file_paths, destination, version='spgt')
