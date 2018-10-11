# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Sven Dorkenwald, Philipp Schubert, Joergen Kornfeld

from scipy import spatial
import networkx as nx
import numpy as np
from knossos_utils.skeleton import Skeleton, SkeletonAnnotation, SkeletonNode
import tqdm
from ..mp.mp_utils import start_multiprocess_obj
from ..config.global_params import min_cc_size_glia, min_cc_size_neuron,\
    get_dataset_scaling, wd, glia_thresh, min_single_sv_size
import os
from ..mp import qsub_utils as qu
from ..mp.mp_utils import start_multiprocess_imap as start_multiprocess
from ..reps.segmentation import SegmentationDataset
from ..reps.super_segmentation import SuperSegmentationObject
from ..handler.basics import *
from ..backend.storage import AttributeDict
from ..reps.rep_helper import knossos_ml_from_ccs
import itertools


def bfs_smoothing(vertices, vertex_labels, max_edge_length=120, n_voting=40):
    """
    Smooth vertex labels by applying a majority vote on a
    BFS subset of nodes for every node in the graph
    Parameters
    ----------
    vertices : np.array
        N, 3
    vertex_labels : np.array
        N, 1
    max_edge_length : float
        maximum distance between vertices to consider them connected in the
        graph
    n_voting : int
        Number of collected nodes during BFS used for majority vote
    Returns
    -------
    np.array
        smoothed vertex labels
    """
    G = create_graph_from_coords(vertices, max_dist=max_edge_length, mst=False,
                                 force_single_cc=False)
    # create BFS subset
    bfs_nn = split_subcc(G, max_nb=n_voting, verbose=False)
    new_vertex_labels = np.zeros_like(vertex_labels)
    for ii in range(len(vertex_labels)):
        curr_labels = vertex_labels[bfs_nn[ii]]
        labels, counts = np.unique(curr_labels, return_counts=True)
        majority_label = labels[np.argmax(counts)]
        new_vertex_labels[ii] = majority_label
    return new_vertex_labels


def split_subcc(g, max_nb, verbose=False, start_nodes=None):
    """
    Creates subgraph for each node consisting of nodes until maximum number of
    nodes is reached.

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
    if verbose:
        nb_nodes = g.number_of_nodes()
        pbar = tqdm.tqdm(total=nb_nodes)
    if start_nodes is None:
        iter_ixs = g.nodes()
    else:
        iter_ixs = start_nodes
    for n in iter_ixs:
        n_subgraph = [n]
        nb_edges = 0
        for e in nx.bfs_edges(g, n):
            n_subgraph.append(e[1])
            nb_edges += 1
            if nb_edges == max_nb:
                break
        subnodes[n] = n_subgraph
        if verbose:
            pbar.update(1)
    if verbose:
        pbar.close()
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


def get_glianess_dict(seg_objs, thresh, glia_key, nb_cpus=1, use_sv_volume=False):
    glianess = {}
    sizes = {}
    params = [[so, glia_key, thresh, use_sv_volume] for so in seg_objs]
    res = start_multiprocess(glia_loader_helper, params, nb_cpus=nb_cpus)
    for ii, el in enumerate(res):
        so = seg_objs[ii]
        glianess[so] = el[0]
        sizes[so] = el[1]
    return glianess, sizes


def glia_loader_helper(args):
    so, glia_key, thresh, use_sv_volume = args
    if not glia_key in so.attr_dict.keys():
        so.load_attr_dict()
    curr_glianess = so.glia_pred(thresh)
    if not use_sv_volume:
        curr_size = so.mesh_bb
    else:
        curr_size = so.size
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
            return [], list(g.nodes())
        return []

    # get glia type connected component sizes
    g_glia = g.copy()
    for n in g_glia.nodes():
        if glia_dict[n] == 0:
            g_glia.remove_node(n)
    glia2ccsize_dict = create_ccsize_dict(g_glia, size_dict)
    if np.all(glia2ccsize_dict.values() <= min_cc_size_glia): # no significant glia SV
        if return_removed_nodes:
            return list(g.nodes()), []
        return []

    tiny_glia_fragments = []
    for n in g_glia.nodes_iter():
        if glia2ccsize_dict[n] < min_cc_size_glia:
            tiny_glia_fragments += [n]

    # create new neuron graph without sufficiently big glia connected components
    g_neuron = g.copy()
    for n in g.nodes_iter():
        if glia_dict[n] != 0 and n not in tiny_glia_fragments:
            g_neuron.remove_node(n)

    # find orphaned neuron SV's and add them to glia graph
    neuron2ccsize_dict = create_ccsize_dict(g_neuron, size_dict)
    for n in g_neuron.nodes():
        if neuron2ccsize_dict[n] < min_cc_size_neuron:
            g_neuron.remove_node(n)

    # create new glia graph with remaining nodes
    # (as the complementary set of sufficiently big neuron connected components)
    g_glia = g.copy()
    for n in g_neuron.nodes_iter():
        g_glia.remove_node(n)

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
        anno.setComment("{0:.4}".format(shortest_path_length))
        skel = Skeleton()
        skel.add_annotation(anno)
        skel.to_kzip("{{}/{0:.4}_vertpath.k.zip".format(write_paths, shortest_path_length))
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


def create_graph_from_coords(coords, max_dist=6000, force_single_cc=True,
                            mst=False):
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
    if force_single_cc:
        while not len(np.unique(pairs)) == len(coords):
            max_dist += max_dist / 3
            print("Generated skeleton is not a single connected component. "
                  "Increasing maximum node distance to {}".format(max_dist))
            pairs = kd_t.query_pairs(r=max_dist, output_type="ndarray")
    g = nx.Graph()
    g.add_nodes_from(np.arange(len(coords)))
    weights = np.linalg.norm(coords[pairs[:, 0]]-coords[pairs[:, 1]], axis=1)#np.array([np.linalg.norm(coords[p[0]]-coords[p[1]]) for p in pairs])
    # this is slow, but there seems no way to add weights from an array with the same ordering as edges, so one loop is needed..
    g.add_weighted_edges_from([[pairs[i][0], pairs[i][1], weights[i]] for i in range(len(pairs))])
    if mst:
        g = nx.minimum_spanning_tree(g)
    return g


def draw_glia_graph(G, dest_path, min_sv_size=0, ext_glia=None, iterations=150,
                    glia_key="glia_probas", node_size_cap=np.inf, mcmp=None, pos=None):
    """
    Draw graph with nodes colored in red (glia) and blue) depending on their
    class. Writes drawing to dest_path.

    Parameters
    ----------
    G : nx.Graph
    dest_path : str
    min_sv_size : int
    ext_glia : dict
        keys: node in G, values: number indicating class
    glia_key : str
    node_size_cap : int
    mcmp : color palette
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    if mcmp is None:
        mcmp = sns.diverging_palette(250, 15, s=99, l=60, center="dark",
                                     as_cmap=True)
    np.random.seed(0)
    seg_objs = list(G.nodes())
    glianess, size = get_glianess_dict(seg_objs, glia_thresh, glia_key, 5,
                                       use_sv_volume=True)
    if ext_glia is not None:
        for n in G.nodes():
            glianess[n] = ext_glia[n.id]
    plt.figure()
    n_size = np.array([size[n]**(1./3) for n in G.nodes()]).astype(np.float32)  # reduce cubic relation to a linear one
    # n_size = np.array([np.linalg.norm(size[n][1]-size[n][0]) for n in G.nodes()])
    if node_size_cap == "max":
        node_size_cap = np.max(n_size)
    n_size[n_size > node_size_cap] = node_size_cap
    col = np.array([glianess[n] for n in G.nodes()])
    col = col[n_size >= min_sv_size]
    nodelist = list(np.array(list(G.nodes()))[n_size > min_sv_size])
    n_size = n_size[n_size >= min_sv_size]
    n_size = n_size / np.max(n_size) * 25.
    if pos is None:
        pos = nx.spring_layout(G, weight="weight", iterations=iterations)
    nx.draw(G, nodelist=nodelist, node_color=col, node_size=n_size,
            cmap=mcmp, width=0.15, pos=pos, linewidths=0)
    plt.savefig(dest_path)
    plt.close()
    return pos


def nxGraph2kzip(g, coords, kzip_path):
    import tqdm
    scaling = get_dataset_scaling()
    coords = coords / scaling
    skel = Skeleton()
    anno = SkeletonAnnotation()
    anno.scaling = scaling
    node_mapping = {}
    pbar = tqdm.tqdm(total=len(coords) + len(g.edges()))
    for v in g.nodes():
        c = coords[v]
        n = SkeletonNode().from_scratch(anno, c[0], c[1], c[2])
        node_mapping[v] = n
        anno.addNode(n)
        pbar.update(1)
    for e in g.edges():
        anno.addEdge(node_mapping[e[0]], node_mapping[e[1]])
        pbar.update(1)
    skel.add_annotation(anno)
    skel.to_kzip(kzip_path)
    pbar.close()


# --------------------------------------------------------------- GLIA SPLITTING

def qsub_glia_splitting():
    """
    Start glia splitting -> generate final connected components of neuron vs.
    glia SVs
    """
    cc_dict = load_pkl2obj(wd + "/glia/cc_dict_rag_graphs.pkl")
    huge_ssvs = [it[0] for it in cc_dict.items() if len(it[1]) > 3e5]
    if len(huge_ssvs):
        print("%d huge SSVs detected (#SVs > 3e5)\n%s" %
              (len(huge_ssvs), huge_ssvs))
    script_folder = os.path.dirname(
        os.path.abspath(__file__)) + "/../../syconn/QSUB_scripts/"
    chs = chunkify(list(cc_dict.values()), 1000)
    qu.QSUB_script(chs, "split_glia", pe="openmp", queue=None,
                   script_folder=script_folder, n_max_co_processes=100)


def collect_glia_sv():
    """
    Collect glia super voxels (as returned by glia splitting) from all 'sv'
    SegmentationObjects contained in 'sv' SegmentationDataset (always uses
    default version as defined in config.ini).
    """
    cc_dict = load_pkl2obj(wd + "/glia/cc_dict_rag.pkl")
    # get single SV glia probas which were not included in the old RAG
    ids_in_rag = np.concatenate(list(cc_dict.values()))
    sds = SegmentationDataset("sv", working_dir=wd)
    # get all SV glia probas (faster than single access)
    multi_params = sds.so_dir_paths
    # glia predictions only used for SSVs which only have single SV and were
    # not contained in RAG
    glia_preds_list = start_multiprocess(collect_gliaSV_helper_chunked,
                                         multi_params, nb_cpus=20, debug=False)
    glia_preds = {}
    for dc in glia_preds_list:
        glia_preds.update(dc)
    print("Collected SV glianess.")
    # get SSV glia splits
    chs = chunkify(list(cc_dict.keys()), 1000)
    glia_svs = np.concatenate(start_multiprocess(collect_gliaSV_helper, chs,
                                                 nb_cpus=20))
    print("Collected SSV glia SVs.")
    # add missing SV glianess and store whole dataset classification
    missing_ids = np.setdiff1d(sds.ids, ids_in_rag)
    single_sv_glia = np.array([ix for ix in missing_ids if glia_preds[ix] == 1],
                              dtype=np.uint64)
    glia_svs = np.concatenate([single_sv_glia, glia_svs]).astype(np.uint64)
    print("Collected whole dataset glia predictions.")
    np.save(wd + "/glia/glia_svs.npy", glia_svs)
    neuron_svs = np.array(list(set(sds.ids).difference(set(glia_svs))),
                          dtype=np.uint64)
    np.save(wd + "/glia/neuron_svs.npy", neuron_svs)


def collect_gliaSV_helper(cc_ixs):
    glia_svids = []
    for cc_ix in cc_ixs:
        sso = SuperSegmentationObject(cc_ix, working_dir=wd,
                                      version="gliaremoval")
        sso.load_attr_dict()
        ad = sso.attr_dict
        glia_svids += list(flatten_list(ad["glia_svs"]))
    return np.array(glia_svids)


def collect_gliaSV_helper_chunked(path):
    """
    Fast, chunked way to collect glia predictions.

    Parameters
    ----------
    path : str

    Returns
    -------
    dict
    """
    ad = AttributeDict(path + "attr_dict.pkl")
    glia_preds = {}
    for k, v in ad.items():
        # see syconn.reps.segmentation_helper.glia_pred_so
        glia_pred = 0
        preds = np.array(v["glia_probas"][:, 1] > glia_thresh, dtype=np.int)
        pred = np.mean(v["glia_probas"][:, 1]) > glia_thresh
        if pred == 0:
            glia_pred = 0
        glia_votes = np.sum(preds)
        if glia_votes > int(len(preds) * 0.7):
            glia_pred = 1
        glia_preds[k] = glia_pred
    return glia_preds


def write_glia_rag(path2rag, suffix=""):
    assert os.path.isfile(path2rag), "Reconnect RAG has to be given."
    g = nx.read_edgelist(path2rag, nodetype=int,
                         delimiter=',')
    glia_svs = np.load(wd + "/glia/glia_svs.npy")
    neuron_g = g.copy()
    for ix in glia_svs:
        try:
            neuron_g.remove_node(ix)
        except:
            continue
    # create glia rag by removing neuron sv's
    glia_g = g.copy()
    for ix in neuron_g.nodes():
        glia_g.remove_node(ix)
    # add single CCs with single SV manually
    neuron_ids = neuron_g.nodes()
    all_neuron_ids = np.load(wd + "/glia/neuron_svs.npy")
    sds = SegmentationDataset("sv", working_dir=wd)
    all_size_dict = {}
    for i in range(len(sds.ids)):
        sv_ix, sv_size = sds.ids[i], sds.sizes[i]
        all_size_dict[sv_ix] = sv_size
    missing_neuron_svs = set(all_neuron_ids).difference(neuron_ids)
    before_cnt = len(neuron_g.nodes())
    for ix in missing_neuron_svs:
        if all_size_dict[ix] > min_single_sv_size:
            neuron_g.add_node(ix)
            neuron_g.add_edge(ix, ix)
    print("Added %d neuron CCs with one SV." % (
                len(neuron_g.nodes()) - before_cnt))
    ccs = sorted(list(nx.connected_components(neuron_g)), reverse=True, key=len)
    txt = knossos_ml_from_ccs([list(cc)[0] for cc in ccs], ccs)
    write_txt2kzip(wd + "/glia/neuron_rag_ml%s.k.zip" % suffix, txt,
                   "mergelist.txt")
    nx.write_edgelist(neuron_g, wd + "/glia/neuron_rag%s.bz2" % suffix)
    print("Nb neuron CC's:", len(ccs), len(ccs[0]))
    # add glia CCs with single SV
    missing_glia_svs = set(glia_svs).difference(glia_g.nodes())
    before_cnt = len(glia_g.nodes())
    for ix in missing_glia_svs:
        if all_size_dict[ix] > min_single_sv_size:
            glia_g.add_node(ix)
            glia_g.add_edge(ix, ix)
    print("Added %d glia CCs with one SV." % (len(glia_g.nodes()) - before_cnt))
    ccs = list(nx.connected_components(glia_g))
    print("Nb glia CC's:", len(ccs))
    nx.write_edgelist(glia_g, wd + "/glia/glia_rag%s.bz2" % suffix)
    txt = knossos_ml_from_ccs([list(cc)[0] for cc in ccs], ccs)
    write_txt2kzip(wd + "/glia/glia_rag_ml%s.k.zip" % suffix, txt,
                   "mergelist.txt")


def transform_rag_edgelist2pkl(rag):
    """
    Stores networkx graph as dictionary mapping (1) SSV IDs to lists of SV IDs
     and (2) SSV IDs to subgraphs (networkx)

    Parameters
    ----------
    rag : networkx.Graph

    Returns
    -------

    """
    ccs = nx.connected_component_subgraphs(rag)
    cc_dict_graph = {}
    cc_dict = {}
    for cc in ccs:
        curr_cc = list(cc.nodes())
        min_ix = np.min(curr_cc)
        if min_ix in cc_dict:
            raise ("laksnclkadsnfldskf")
        cc_dict_graph[min_ix] = cc
        cc_dict[min_ix] = curr_cc
    write_obj2pkl(wd + "/glia/cc_dict_rag_graphs.pkl", cc_dict_graph)
    write_obj2pkl(wd + "/glia/cc_dict_rag.pkl", cc_dict)