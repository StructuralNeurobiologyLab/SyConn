# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Sven Dorkenwald, Philipp Schubert, Joergen Kornfeld

import numpy as np
from scipy import spatial
import networkx as nx
from knossos_utils.skeleton import Skeleton, SkeletonAnnotation, SkeletonNode
import itertools
import sys
from ..mp.shared_mem import start_multiprocess_obj, start_multiprocess
from ..config.global_params import min_cc_size_glia, min_cc_size_neuron

def split_subcc(g, max_nb, verbose=False, start_nodes=None):
    """
    Creates subgraph for each node consisting of nodes within distance
    threshold.

    Parameters
    ----------
    g : Graph
    max_nb : int
    verbose : bool
    start_nodes : iterable
        node ID's

    Returns
    -------
    dict
    """
    subnodes = {}
    nb_nodes = len(g.nodes())
    cnt = 0
    if start_nodes is None:
        iter_ixs = g.nodes_iter()
    else:
        iter_ixs = start_nodes
    for n in iter_ixs:
        if verbose:
            if cnt % 100 == 0:
                sys.stdout.write("\r%0.6f" % (cnt / float(nb_nodes)))
                sys.stdout.flush()
            cnt += 1
        n_subgraph = [n]
        nb_edges = 0
        for e in nx.bfs_edges(g, n):
            n_subgraph.append(e[1])
            nb_edges += 1
            if nb_edges == max_nb:
                break
        subnodes[n] = n_subgraph
    return subnodes


def split_glia_graph(nx_g, thresh, clahe=False, shortest_paths_dest_dir=None,
                     nb_cpus=1, pred_key_appendix=""):
    """
    Split graph into glia and non-glua CC's.

    Parameters
    ----------
    nx_g : nx.Graph
    thresh : float
    clahe : bool
    shortest_paths_dest_dir : str
        None (default), else path to directory to write shortest paths
        between neuron type SV end nodes
    nb_cpus : int
    pred_key_appendix : str

    Returns
    -------
    list, list
        Neuron, glia connected components
    """
    _ = start_multiprocess_obj("mesh_bb", [[sv, ] for sv in nx_g.nodes()],
                               nb_cpus=nb_cpus)
    glia_key = "glia_probas"
    if clahe:
        glia_key += "_clahe"
    glia_key += pred_key_appendix
    glianess, size = get_glianess_dict(nx_g.nodes(), thresh, glia_key,
                                       nb_cpus=nb_cpus)
    return remove_glia_nodes(nx_g, size, glianess, return_removed_nodes=True,
                             shortest_paths_dest_dir=shortest_paths_dest_dir)


def split_glia(sso, thresh, clahe=False, shortest_paths_dest_dir=None,
               pred_key_appendix=""):
    """
    Split SuperSegmentationObject into glia and non glia
    SegmentationObjects.

    Parameters
    ----------
    sso : SuperSegmentationObject
    thresh : float
    clahe : bool
    shortest_paths_dest_dir : str
        None (default), else path to directory to write shortest paths
        between neuron type SV end nodes
    pred_key_appendix : str
        Defines type of glia predictions

    Returns
    -------
    list, list (of SegmentationObject)
        Neuron, glia nodes
    """
    nx_G = sso.rag
    nonglia_ccs, glia_ccs = split_glia_graph(nx_G, thresh=thresh, clahe=clahe,
                            nb_cpus=sso.nb_cpus, shortest_paths_dest_dir=
                            shortest_paths_dest_dir, pred_key_appendix=pred_key_appendix)
    return nonglia_ccs, glia_ccs


def create_ccsize_dict(g, sizes):
    ccs = nx.connected_components(g)
    node2cssize_dict = {}
    for cc in ccs:
        mesh_bbs = np.concatenate([sizes[n] for n in cc])
        cc_size = np.linalg.norm(np.max(mesh_bbs, axis=0)-
                                 np.min(mesh_bbs, axis=0), ord=2)
        for n in cc:
            node2cssize_dict[n] = cc_size
    return node2cssize_dict


def get_glianess_dict(seg_objs, thresh, glia_key, nb_cpus=1):
    glianess = {}
    sizes = {}
    params = [[so, glia_key, thresh] for so in seg_objs]
    res = start_multiprocess(glia_loader_helper, params, nb_cpus=nb_cpus)
    for ii, el in enumerate(res):
        so = seg_objs[ii]
        glianess[so] = el[0]
        sizes[so] = el[1]
    return glianess, sizes


def glia_loader_helper(args):
    so, glia_key, thresh = args
    if not glia_key in so.attr_dict.keys():
        so.load_attr_dict()
    curr_glianess = so.glia_pred(thresh)
    curr_size = so.mesh_bb
    return curr_glianess, curr_size


def remove_glia_nodes(g, size_dict, glia_dict, return_removed_nodes=False,
                      shortest_paths_dest_dir=None):
    """
    Calculate distance weights for shortest path analysis or similar, based on
    glia and size vertex properties and removes unsupporting glia nodes.

    Parameters
    ----------
    g : Graph
    return_removed_nodes : bool

    Returns
    -------
    list of list of nodes
        Remaining connected components of type neuron
    """
    # set up node weights based on glia prediction and size
    # weights = {}
    # e_weights = {}
    # for n in g.nodes():
    #     weights[n] = np.linalg.norm(size_dict[n][1]-size_dict[n][0], ord=2)\
    #                  * glia_dict[n]
    # # set up edge weights based on sum of node weights
    # for e in g.edges():
    #     e_weights[e] = weights[list(e)[0]] + weights[list(e)[1]]
    # nx.set_node_attributes(g, 'weight', weights)
    # nx.set_edge_attributes(g, 'weights', e_weights)

    # get neuron type connected component sizes
    g_neuron = g.copy()
    for n in g_neuron.nodes():
        if glia_dict[n] != 0:
            g_neuron.remove_node(n)
    neuron2ccsize_dict = create_ccsize_dict(g_neuron, size_dict)
    if np.all(neuron2ccsize_dict.values() <= min_cc_size_neuron): # no significant neuron SV
        if return_removed_nodes:
            return [], g.nodes()
        return []

    # get glia type connected component sizes
    g_glia = g.copy()
    for n in g_glia.nodes():
        if glia_dict[n] == 0:
            g_glia.remove_node(n)
    glia2ccsize_dict = create_ccsize_dict(g_glia, size_dict)
    if np.all(glia2ccsize_dict.values() <= min_cc_size_glia): # no significant glia SV
        if return_removed_nodes:
            return g.nodes(), []
        return []

    support_glia_nodes = set()

    tiny_glia_fragments = []
    for n in g_glia.nodes_iter():
        if glia2ccsize_dict[n] < min_cc_size_glia:
            tiny_glia_fragments += [n]

    # create new neuron graph
    g_neuron = g.copy()
    for n in g.nodes_iter():
        if glia_dict[n] != 0 and n not in tiny_glia_fragments and n not in support_glia_nodes:
            g_neuron.remove_node(n)

    # find orphaned neuron SV's
    neuron2ccsize_dict = create_ccsize_dict(g_neuron, size_dict)
    for n in g_neuron.nodes():
        if neuron2ccsize_dict[n] < min_cc_size_neuron:
            g_neuron.remove_node(n)

    # create new glia graph with remaining nodes
    g_glia = g.copy()
    for n in g_neuron.nodes_iter():
        g_glia.remove_node(n)
    # remove unsupportive glia nodes and small neuron type fragments from neuron graph

    neuron_ccs = list(nx.connected_components(g_neuron))
    if return_removed_nodes:
        glia_ccs = list(nx.connected_components(g_glia))
        assert len(g_glia) + len(g_neuron) == len(g)
        return neuron_ccs, glia_ccs
    return neuron_ccs


def glia_path_length(glia_path, glia_dict, write_paths=None):
    """
    Get the path length of glia SV within glia_path. Assumes single connected
    glia component within this path. Uses the mesh property of each
    SegmentationObject to build a graph from all vertices to find shortest path
    through (or more precise: along the surface of) glia. Edges between non-glia
    vertices have negligible distance (0.0001) to ensure shortest path
    along non-glia surfaces.

    Parameters
    ----------
    glia_path : list of SegmentationObjects
    glia_dict : dict
        Dictionary which keys the SegmentationObjects in glia_path and returns
        their glia prediction

    Returns
    -------
    float
        Shortest path between neuron type nodes in nm
    """
    g = nx.Graph()
    col = {}
    curr_ind = 0
    if write_paths is not None:
        all_vert = np.zeros((0, 3))
    for so in glia_path:
        is_glia_sv = int(glia_dict[so] > 0)
        ind, vert = so.mesh
        # connect meshes of different SV, starts after first SV
        if curr_ind > 0:
            # build kd tree from vertices of SV before
            kd_tree = spatial.cKDTree(vert_resh)
            # get indices of vertives of SV before (= indices of graph nodes)
            ind_offset_before = curr_ind - len(vert_resh)
            # query vertices of current mesh to find close connects
            next_vert_resh = vert.reshape((-1, 3))
            dists, ixs = kd_tree.query(next_vert_resh, distance_upper_bound=500)
            for kk, ix in enumerate(ixs):
                if dists[kk] > 500:
                    continue
                if is_glia_sv:
                    edge_weight = eucl_dist(next_vert_resh[kk], vert_resh[ix])
                else:
                    edge_weight = 0.0001
                g.add_edge(curr_ind + kk, ind_offset_before + ix,
                           weights=edge_weight)
        vert_resh = vert.reshape((-1, 3))
        # save all vertices for writing shortest path skeleton
        if write_paths is not None:
            all_vert = np.concatenate([all_vert, vert_resh])
        # connect fragments of SV mesh
        kd_tree = spatial.cKDTree(vert_resh)
        dists, ixs = kd_tree.query(vert_resh, k=20, distance_upper_bound=500)
        for kk in range(len(ixs)):
            nn_ixs = ixs[kk]
            nn_dists = dists[kk]
            col[curr_ind + kk] = glia_dict[so]
            for curr_ix, curr_dist in zip(nn_ixs, nn_dists):
                col[curr_ind + curr_ix] = glia_dict[so]
                if is_glia_sv:
                    dist = curr_dist
                else:  # only take path through glia into account
                    dist = 0
                g.add_edge(kk+curr_ind, curr_ix+curr_ind, weights=dist)
        curr_ind += len(vert_resh)
    start_ix = 0  # choose any index of the first mesh
    end_ix = curr_ind - 1  # choose any index of the last mesh
    shortest_path_length = nx.dijkstra_path_length(g, start_ix, end_ix, weight="weights")
    if write_paths is not None:
        shortest_path = nx.dijkstra_path(g, start_ix, end_ix, weight="weights")
        anno = coordpath2anno([all_vert[ix] for ix in shortest_path])
        anno.setComment("%0.4f" % shortest_path_length)
        skel = Skeleton()
        skel.add_annotation(anno)
        skel.to_kzip("%s/%0.4f_vertpath.k.zip" % (write_paths, shortest_path_length))
    return shortest_path_length


def eucl_dist(a, b):
    return np.linalg.norm(a-b)


def get_glia_paths(g, glia_dict, node2ccsize_dict, min_cc_size_neuron,
                   node2ccsize_dict_glia, min_cc_size_glia):
    """
    Find paths between neuron type SV grpah nodes which contain glia nodes.

    Parameters
    ----------
    g :
    glia_dict :
    node2ccsize_dict :
    min_cc_size_neuron :
    node2ccsize_dict_glia :
    min_cc_size_glia :

    Returns
    -------

    """
    end_nodes = []
    paths = nx.all_pairs_dijkstra_path(g, weight="weights")
    for n, d in g.degree().items():
        if d == 1 and glia_dict[n] == 0 and node2ccsize_dict[n] > min_cc_size_neuron:
            end_nodes.append(n)

    # find all nodes along these ways and store them as mandatory nodes
    glia_paths = []
    glia_svixs_in_paths = []
    for a, b in itertools.combinations(end_nodes, 2):
        glia_nodes = [n for n in paths[a][b] if glia_dict[n] != 0]
        if len(glia_nodes) == 0:
            continue
        sv_ccsizes = [node2ccsize_dict_glia[n] for n in glia_nodes]
        if np.max(sv_ccsizes) <= min_cc_size_glia:  # check minimum glia size
            continue
        sv_ixs = np.array([n.id for n in glia_nodes])
        glia_nodes_already_exist = False
        for el_ixs in glia_svixs_in_paths:
            if np.all(sv_ixs == el_ixs):
                glia_nodes_already_exist = True
                break
        if glia_nodes_already_exist: # check if same glia path exists already
            continue
        glia_paths.append(paths[a][b])
        glia_svixs_in_paths.append(np.array([so.id for so in glia_nodes]))
    # print glia_svixs_in_paths
    return glia_paths


def write_sopath2skeleton(so_path, dest_path, comment=None):
    """
    Writes very simple skeleton, each node represents the center of mass of a
    SV, and edges are created in list order.

    Parameters
    ----------
    so_path : list of SegmentationObject
    dest_path : str
    """
    skel = Skeleton()
    anno = SkeletonAnnotation()
    anno.scaling = [10, 10, 20]
    rep_nodes = []
    for so in so_path:
        vert = so.mesh[1].reshape((-1, 3))
        com = np.mean(vert, axis=0)
        kd_tree = spatial.cKDTree(vert)
        dist, nn_ix = kd_tree.query([com])
        nn = vert[nn_ix[0]] / np.array([10, 10, 20])
        n = SkeletonNode().from_scratch(anno, nn[0], nn[1], nn[2])
        anno.addNode(n)
        rep_nodes.append(n)
    for i in range(1, len(rep_nodes)):
        anno.addEdge(rep_nodes[i-1], rep_nodes[i])
    if comment is not None:
        anno.setComment(comment)
    skel.add_annotation(anno)
    skel.to_kzip(dest_path)


def coordpath2anno(coords):
    """
    Creates skeleton from scaled coordinates, assume coords are in order for
    edge creation.

    Parameters
    ----------
    coords : np.array

    Returns
    -------
    SkeletonAnnotation
    """
    anno = SkeletonAnnotation()
    anno.scaling = [10, 10, 20]
    rep_nodes = []
    for c in coords:
        n = SkeletonNode().from_scratch(anno, c[0]/10, c[1]/10, c[2]/20)
        anno.addNode(n)
        rep_nodes.append(n)
    for i in range(1, len(rep_nodes)):
        anno.addEdge(rep_nodes[i-1], rep_nodes[i])
    return anno


def create_mst_skeleton(coords, max_dist=6000, force_single_cc=True):
    """
    Generate skeleton from sample locations by adding edges between points
    with a maximum distance and then pruning the skeleton using MST.
    
    Parameters
    ----------
    coords : np.array
    max_dist : float
    force_single_cc : bool
        force that the tree generated from coords is a single connected 
        component

    Returns
    -------
    np. array 
        edge list of nodes (coords) using the ordering of coords, i.e. the
        edge (1, 2) connects coordinate coord[1] and coord[2].
    """
    kd_t = spatial.cKDTree(coords)
    pairs = kd_t.query_pairs(r=max_dist, output_type="ndarray")
    g = nx.Graph()
    weights = np.array([np.linalg.norm(coords[p[0]]-coords[p[1]]) for p in pairs])
    g.add_weighted_edges_from([[pairs[i][0], pairs[i][1], weights[i]] for i in range(len(pairs))])
    while not (nx.is_connected(g) and len(g.nodes()) == len(coords)):
        if not force_single_cc:
            break
        max_dist += 2e3
        print("Generated skeleton is not a single connected component. " \
              "Increasing maximum node distance to %0.0f" % (max_dist))
        pairs = kd_t.query_pairs(r=max_dist, output_type="ndarray")
        g = nx.Graph()
        weights = np.array(
            [np.linalg.norm(coords[p[0]] - coords[p[1]]) for p in pairs])
        g.add_weighted_edges_from(
            [[int(pairs[i][0]), int(pairs[i][1]), weights[i]] for i in range(len(pairs))])
    g = nx.minimum_spanning_tree(g)
    return np.array(g.edges())
