# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Philipp Schubert, Joergen Kornfeld
import itertools
from typing import List, Any, Optional, TYPE_CHECKING

import networkx as nx
import numpy as np
import tqdm
from knossos_utils.skeleton import Skeleton, SkeletonAnnotation, SkeletonNode
from scipy import spatial

if TYPE_CHECKING:
    from ..reps.super_segmentation import SuperSegmentationObject
from .. import global_params
from ..mp.mp_utils import start_multiprocess_imap as start_multiprocess


def bfs_smoothing(vertices, vertex_labels, max_edge_length=120, n_voting=40):
    """
    This function smooths vertex labels by applying a majority vote on a
    BFS subset of nodes for every node in the graph.
    
    Args:
        vertices (np.array): An array of shape N, 3 representing the vertices.
        vertex_labels (np.array): An array of shape N, 1 representing the vertex labels.
        max_edge_length (float): The maximum distance between vertices to consider them
            connected in the graph. Default is 120.
        n_voting (int): The number of collected nodes during BFS used for majority vote.
            Default is 40.
    
    Returns:
        np.array: An array representing the smoothed vertex labels.
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
    This function creates a subgraph for each node consisting of nodes until the 
    maximum number of nodes is reached.
    
    Args:
        g (Graph): The input graph.
        max_nb (int): The maximum number of nodes.
        verbose (bool): If True, the function will display a progress bar. 
                        Default is False.
        start_nodes (iterable): An iterable containing node IDs. Default is None.
    
    Returns:
        dict: A dictionary where the keys are the nodes and the values are the 
              subgraphs.
    """
    subnodes = {}
    if verbose:
        nb_nodes = g.number_of_nodes()
        pbar = tqdm.tqdm(total=nb_nodes, leave=False)
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


def chunkify_contiguous(l, n):
    """
    This function yields successive n-sized chunks from a list.
    
    Args:
        l (list): The input list.
        n (int): The size of the chunks.
    
    Yields:
        list: The next n-sized chunk from the list.
    
    Reference:
        https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
    """
    for i in range(0, len(l), n):
        yield l[i:i + n]


def split_subcc_join(g: nx.Graph, subgraph_size: int, lo_first_n: int = 1) -> List[List[Any]]:
    """
    This function creates a subgraph for each node consisting of nodes until the maximum number of
    nodes is reached.
    
    Args:
        g (nx.Graph): The supervoxel graph.
        subgraph_size (int): The size of subgraphs. The difference between `subgraph_size` and `lo_first_n` defines the
            supervoxel overlap.
        lo_first_n (int): Leave out first n nodes. Will collect `subgraph_size` nodes starting from the center node and then
            omit the first lo_first_n nodes, i.e. not use them as new starting nodes. Default is 1.
    
    Returns:
        list: A list of lists, where each inner list represents a subgraph with context.
    """
    start_node = list(g.nodes())[0]
    for n, d in dict(g.degree).items():
        if d == 1:
            start_node = n
            break
    dfs_nodes = list(nx.dfs_preorder_nodes(g, start_node))
    # get subgraphs via splicing of traversed node list into equally sized fragments. they might
    # be unconnected if branch sizes mod subgraph_size != 0, then a chunk will contain multiple connected components.
    chunks = list(chunkify_contiguous(dfs_nodes, lo_first_n))
    sub_graphs = []
    for ch in chunks:
        # collect all connected component subgraphs
        sg = g.subgraph(ch).copy()
        sub_graphs += list((sg.subgraph(c) for c in nx.connected_components(sg)))
    # add more context to subgraphs
    subgraphs_withcontext = []
    for sg in sub_graphs:
        # add context but omit artificial start node
        context_nodes = []
        for n in list(sg.nodes()):
            subgraph_nodes_with_context = []
            nb_edges = sg.number_of_nodes()
            for e in nx.bfs_edges(g, n):
                subgraph_nodes_with_context += list(e)
                nb_edges += 1
                if nb_edges == subgraph_size:
                    break
            context_nodes += subgraph_nodes_with_context
        # add original nodes
        context_nodes = list(set(context_nodes))
        for n in list(sg.nodes()):
            if n in context_nodes:
                context_nodes.remove(n)
        subgraph_nodes_with_context = list(sg.nodes()) + context_nodes
        subgraphs_withcontext.append(subgraph_nodes_with_context)
    return subgraphs_withcontext


def merge_nodes(G, nodes, new_node):
    """
    This function merges nodes in an unweighted, undirected graph.
    
    Args:
        G (Graph): The input graph.
        nodes (list): The nodes to be merged.
        new_node (any): The new node that will replace the merged nodes.
    """
    if G.is_directed():
        raise ValueError('Method "merge_nodes" is only valid for undirected graphs.')
    G.add_node(new_node)
    for n in nodes:
        for e in G.edges(n):
            # add edge between new node and original partner node
            edge = list(e)
            edge.remove(n)
            paired_node = edge[0]
            G.add_edge(new_node, paired_node)

    for n in nodes:  # remove the merged nodes
        G.remove_node(n)


def split_glia_graph(nx_g, thresh, clahe=False, nb_cpus=1, pred_key_appendix=""):
    """
    This function splits a graph into glia and non-glia connected components.
    
    Args:
        nx_g (nx.Graph): The input graph.
        thresh (float): The threshold for splitting.
        clahe (bool): If True, the function will use CLAHE (Contrast Limited 
            Adaptive Histogram Equalization). Default is False.
        nb_cpus (int): The number of CPUs to use. Default is 1.
        pred_key_appendix (str): The appendix for the prediction key. Default 
            is an empty string.
        verbose (bool): No description provided in the new docstring.
    
    Returns:
        list, list: Two lists representing the neuron and glia connected 
            components, respectively.
    """
    glia_key = "glia_probas"
    if clahe:
        glia_key += "_clahe"
    glia_key += pred_key_appendix
    glianess, size = get_glianess_dict(list(nx_g.nodes()), thresh, glia_key,
                                       nb_cpus=nb_cpus)
    return remove_glia_nodes(nx_g, size, glianess, return_removed_nodes=True)


def split_glia(sso, thresh, clahe=False, pred_key_appendix=""):
    """
    This function splits a SuperSegmentationObject into glia and non-glia SegmentationObjects.
    
    Args:
        sso (SuperSegmentationObject): The SuperSegmentationObject to be split.
        thresh (float): The threshold for splitting.
        clahe (bool): If True, the function will use CLAHE (Contrast Limited Adaptive 
        Histogram Equalization). Default is False.
        pred_key_appendix (str): The appendix for the prediction key. Default is an 
        empty string.
    
    Returns:
        list, list: Two lists representing the neuron and glia nodes, respectively.
    """
    nx_G = sso.rag
    nonglia_ccs, glia_ccs = split_glia_graph(nx_G, thresh=thresh, clahe=clahe,
                                             nb_cpus=sso.nb_cpus, pred_key_appendix=pred_key_appendix)
    return nonglia_ccs, glia_ccs


def create_ccsize_dict(g: nx.Graph, bbs: dict, is_connected_components: bool = False) -> dict:
    """
    This function calculates the bounding box size of connected components.
    
    Args:
        g (nx.Graph): The supervoxel graph.
        bbs (dict): A dictionary representing the bounding boxes in physical 
        units.
        is_connected_components (bool): If True, the graph `g` is already 
        connected components. If False, ``nx.connected_components`` is 
        applied. Default is False.
    
    Returns:
        dict: A look-up dictionary which stores the connected component 
        bounding box for every single node in the input Graph `g`.
    """
    if not is_connected_components:
        ccs = nx.connected_components(g)
    else:
        ccs = g
    node2cssize_dict = {}
    for cc in ccs:
        # if ID is not in bbs, it was skipped due to low voxel count
        curr_bbs = [bbs[n] for n in cc if n in bbs]
        if len(curr_bbs) == 0:
            raise ValueError(f'Could not find a single bounding box for connected component with IDs: {cc}.')
        else:
            curr_bbs = np.concatenate(curr_bbs)
            cc_size = np.linalg.norm(np.max(curr_bbs, axis=0) -
                                     np.min(curr_bbs, axis=0), ord=2)
        for n in cc:
            node2cssize_dict[n] = cc_size
    return node2cssize_dict


def get_glianess_dict(seg_objs, thresh, glia_key, nb_cpus=1,
                      use_sv_volume=False, verbose=False):
    """
    Generates a dictionary of glia predictions and sizes for a list of SegmentationObjects.
    
    Args:
        seg_objs (list): List of SegmentationObjects.
        thresh (float): Threshold for glia prediction.
        glia_key (str): Key to access glia predictions in the attribute dictionary of SegmentationObjects.
        nb_cpus (int, optional): Number of CPUs to use for multiprocessing. Defaults to 1.
        use_sv_volume (bool, optional): If True, use the volume of the supervoxel for size. 
            Otherwise, use the bounding box. Defaults to False.
        verbose (bool, optional): If True, print progress information. Defaults to False.
    
    Returns:
        tuple: Two dictionaries, the first mapping SegmentationObjects to their glia predictions, 
            and the second mapping SegmentationObjects to their sizes.
    """
    glianess = {}
    sizes = {}
    params = [[so, glia_key, thresh, use_sv_volume] for so in seg_objs]
    res = start_multiprocess(glia_loader_helper, params, nb_cpus=nb_cpus,
                             verbose=verbose, show_progress=verbose)
    for ii, el in enumerate(res):
        so = seg_objs[ii]
        glianess[so] = el[0]
        sizes[so] = el[1]
    return glianess, sizes


def glia_loader_helper(args):
    """
    Helper function for loading glia predictions and sizes for a single SegmentationObject.
    
    Args:
        args (tuple): A tuple containing a SegmentationObject, a glia key, a threshold, 
            and a boolean indicating whether to use supervoxel volume for size.
    
    Returns:
        tuple: A tuple containing the glia prediction and size for the SegmentationObject.
    """
    so, glia_key, thresh, use_sv_volume = args
    if glia_key not in so.attr_dict.keys():
        so.load_attr_dict()
    curr_glianess = so.glia_pred(thresh)
    if not use_sv_volume:
        curr_size = so.mesh_bb
    else:
        curr_size = so.size
    return curr_glianess, curr_size


def remove_glia_nodes(g, size_dict, glia_dict, return_removed_nodes=False):
    """
    Removes glia nodes from a graph based on glia and size vertex properties, and calculates 
    distance weights for shortest path analysis or similar. 
    
    Args:
        g (nx.Graph): Input graph.
        size_dict (dict): Dictionary mapping nodes to their sizes.
        glia_dict (dict): Dictionary mapping nodes to their glia predictions.
        return_removed_nodes (bool, optional): If True, return the removed nodes. Defaults to False.
    
    Returns:
        list: List of connected components of type neuron. If return_removed_nodes is True, 
            also returns a list of removed nodes.
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
    # nx.set_node_attributes(g, weights, 'weight')
    # nx.set_edge_attributes(g, e_weights, 'weights')

    # get neuron type connected component sizes
    g_neuron = g.copy()
    for n in g.nodes():
        if glia_dict[n] != 0:
            g_neuron.remove_node(n)
    neuron2ccsize_dict = create_ccsize_dict(g_neuron, size_dict)
    if np.all(np.array(list(neuron2ccsize_dict.values())) <=
              global_params.config['min_cc_size_ssv']):
        # no significant neuron SV
        if return_removed_nodes:
            return [], [list(g.nodes())]
        return []

    # get glia type connected component sizes
    g_glia = g.copy()
    for n in g.nodes():
        if glia_dict[n] == 0:
            g_glia.remove_node(n)
    glia2ccsize_dict = create_ccsize_dict(g_glia, size_dict)
    if np.all(np.array(list(glia2ccsize_dict.values())) <=
              global_params.config['min_cc_size_ssv']):
        # no significant glia SV
        if return_removed_nodes:
            return [list(g.nodes())], []
        return [list(g.nodes())]

    tiny_glia_fragments = []
    for n in g_glia.nodes():
        if glia2ccsize_dict[n] < global_params.config['min_cc_size_ssv']:
            tiny_glia_fragments += [n]

    # create new neuron graph without sufficiently big glia connected components
    g_neuron = g.copy()
    for n in g.nodes():
        if glia_dict[n] != 0 and n not in tiny_glia_fragments:
            g_neuron.remove_node(n)

    # find orphaned neuron SV's and add them to glia graph
    neuron2ccsize_dict = create_ccsize_dict(g_neuron, size_dict)
    g_tmp = g_neuron.copy()
    for n in g_tmp.nodes():
        if neuron2ccsize_dict[n] < global_params.config['min_cc_size_ssv']:
            g_neuron.remove_node(n)

    # create new glia graph with remaining nodes
    # (as the complementary set of sufficiently big neuron connected components)
    g_glia = g.copy()
    for n in g_neuron.nodes():
        g_glia.remove_node(n)

    neuron_ccs = list(nx.connected_components(g_neuron))
    if return_removed_nodes:
        glia_ccs = list(nx.connected_components(g_glia))
        assert len(g_glia) + len(g_neuron) == len(g)
        return neuron_ccs, glia_ccs
    return neuron_ccs


def glia_path_length(glia_path, glia_dict, write_paths=None):
    """
    Calculates the shortest path length through a glia path. This function assumes a single connected
    glia component within the path. It uses the mesh property of each SegmentationObject to build a
    graph from all vertices to find the shortest path through (or more precise: along the surface of)
    glia. Edges between non-glia vertices have negligible distance (0.0001) to ensure shortest path
    along non-glia surfaces.
    
    Args:
        glia_path (list): List of SegmentationObjects forming a path.
        glia_dict (dict): Dictionary mapping SegmentationObjects to their glia predictions.
        write_paths (bool, optional): If True, write the shortest path to a skeleton file. Defaults to None.
    
    Returns:
        float: Shortest path length in nanometers.
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
                g.add_edge(kk + curr_ind, curr_ix + curr_ind, weights=dist)
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
    """
    Calculates the Euclidean distance between two points.
    
    Args:
        a (np.array): First point.
        b (np.array): Second point.
    
    Returns:
        float: Euclidean distance between the two points.
    """
    return np.linalg.norm(a - b)


def get_glia_paths(g, glia_dict, node2ccsize_dict, min_cc_size_neuron,
                   node2ccsize_dict_glia, min_cc_size_glia):
    """
    Finds paths between neuron type nodes in a graph that contain glia nodes.
    
    Args:
        g (nx.Graph): Input graph.
        glia_dict (dict): Dictionary mapping nodes to their glia predictions.
        node2ccsize_dict (dict): Dictionary mapping neuron nodes to their sizes.
        min_cc_size_neuron (int): Minimum size for a neuron connected component.
        node2ccsize_dict_glia (dict): Dictionary mapping glia nodes to their sizes.
        min_cc_size_glia (int): Minimum size for a glia connected component.
    
    Returns:
        list: List of paths that contain glia nodes.
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
        if glia_nodes_already_exist:  # check if same glia path exists already
            continue
        glia_paths.append(paths[a][b])
        glia_svixs_in_paths.append(np.array([so.id for so in glia_nodes]))
    return glia_paths


def write_sopath2skeleton(so_path, dest_path, scaling=None, comment=None):
    """
    Writes a simple skeleton to a file where each node represents the center of mass of a
    SegmentationObject (SV), and edges are created in the order of the list.
    
    Args:
        so_path (list): List of SegmentationObjects.
        dest_path (str): Path to the destination file.
        scaling (np.ndarray or tuple, optional): Scaling factor for the skeleton. If not provided, 
            the default scaling from the global configuration is used.
        comment (str, optional): Comment to be added to the skeleton.
    
    Returns:
        None
    """
    if scaling is None:
        scaling = np.array(global_params.config['scaling'])
    skel = Skeleton()
    anno = SkeletonAnnotation()
    anno.scaling = scaling
    rep_nodes = []
    for so in so_path:
        vert = so.mesh[1].reshape((-1, 3))
        com = np.mean(vert, axis=0)
        kd_tree = spatial.cKDTree(vert)
        dist, nn_ix = kd_tree.query([com])
        nn = vert[nn_ix[0]] / scaling
        n = SkeletonNode().from_scratch(anno, nn[0], nn[1], nn[2])
        anno.addNode(n)
        rep_nodes.append(n)
    for i in range(1, len(rep_nodes)):
        anno.addEdge(rep_nodes[i - 1], rep_nodes[i])
    if comment is not None:
        anno.setComment(comment)
    skel.add_annotation(anno)
    skel.to_kzip(dest_path)


def coordpath2anno(coords: np.ndarray, scaling: Optional[np.ndarray] = None) -> SkeletonAnnotation:
    """
    Creates a skeleton from scaled coordinates. Assumes coordinates are in order for
    edge creation.
    
    Args:
        coords (np.array): Array of coordinates.
        scaling (np.ndarray, optional): Scaling factor for the skeleton. If not 
            provided, the default scaling from the global configuration is used.
    
    Returns:
        SkeletonAnnotation: Skeleton annotation created from the coordinates.
    """
    if scaling is None:
        scaling = global_params.config['scaling']
    anno = SkeletonAnnotation()
    anno.scaling = scaling
    rep_nodes = []
    for c in coords:
        n = SkeletonNode().from_scratch(anno, c[0] / scaling[0], c[1] / scaling[1],
                                        c[2] / scaling[2])
        anno.addNode(n)
        rep_nodes.append(n)
    for i in range(1, len(rep_nodes)):
        anno.addEdge(rep_nodes[i - 1], rep_nodes[i])
    return anno


def create_graph_from_coords(coords: np.ndarray, max_dist: float = 6000, force_single_cc: bool = True,
                             mst: bool = False) -> nx.Graph:
    """
    Generates a skeleton from sample locations by adding edges between points within a maximum distance 
    and then pruning the skeleton using a minimum spanning tree (MST). Nodes will have a 'position' 
    attribute.
    
    Args:
        coords (np.ndarray): Array of coordinates.
        max_dist (float, optional): Maximum distance between two nodes to consider them connected. 
            Defaults to 6000.
        force_single_cc (bool, optional): If True, forces the tree generated from coordinates to be a 
            single connected component. Defaults to True.
        mst (bool, optional): If True, computes the minimum spanning tree. Defaults to False.
    
    Returns:
        nx.Graph: Networkx graph with edges between nodes (coordinate indices) using the ordering of 
            coordinates. For example, the edge (1, 2) connects coordinate coord[1] and coord[2].
    """
    g = nx.Graph()
    if len(coords) == 1:
        g.add_node(0)
        g.add_weighted_edges_from([[0, 0, 0]])
        return g
    kd_t = spatial.cKDTree(coords)
    pairs = kd_t.query_pairs(r=max_dist, output_type="ndarray")
    g.add_nodes_from([(ix, dict(position=coord)) for ix, coord in enumerate(coords)])
    weights = np.linalg.norm(coords[pairs[:, 0]] - coords[pairs[:, 1]], axis=1)
    g.add_weighted_edges_from([[pairs[i][0], pairs[i][1], weights[i]] for i in range(len(pairs))])
    if force_single_cc:  # make sure its a connected component
        g = stitch_skel_nx(g)
    if mst:
        g = nx.minimum_spanning_tree(g)
    return g


def draw_glia_graph(G, dest_path, min_sv_size=0, ext_glia=None, iterations=150, seed=0,
                    glia_key="glia_probas", node_size_cap=np.inf, mcmp=None, pos=None):
    """
    Draws a graph with nodes colored in red (glia) and blue depending on their class. 
    Writes the drawing to the destination path.
    
    Args:
        G (nx.Graph): Graph to be drawn.
        dest_path (str): Path to the destination file.
        min_sv_size (int, optional): Minimum size of the supervoxel. Defaults to 0.
        ext_glia (dict, optional): Dictionary with node in G as keys and class number as 
            values.
        iterations (int, optional): Number of iterations for layout generation. Defaults to 
            150.
        seed (int, optional): Random seed for layout generation. Defaults to 0.
        glia_key (str, optional): Key to access glia probabilities. Defaults to "glia_probas".
        node_size_cap (int, optional): Maximum node size. Defaults to infinity.
        mcmp (color palette, optional): Color palette for the graph. If not provided, a 
            default palette is used.
        pos (dict, optional): Positions of nodes. If not provided, a spring layout is used.
    
    Returns:
        None
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
    n_size = np.array([size[n] ** (1. / 3) for n in G.nodes()]).astype(
        np.float32)  # reduce cubic relation to a linear one
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
        pos = nx.spring_layout(G, weight="weight", iterations=iterations, random_state=seed)
    nx.draw(G, nodelist=nodelist, node_color=col, node_size=n_size,
            cmap=mcmp, width=0.15, pos=pos, linewidths=0)
    plt.savefig(dest_path)
    plt.close()
    return pos


def nxGraph2kzip(g, coords, kzip_path):
    """
    Writes a networkx graph to a kzip file. The representative coordinate of a node is used as the 
    corresponding node location.
    
    Args:
        g (nx.Graph): Networkx graph to be written.
        coords (np.ndarray): Array of coordinates.
        kzip_path (str): Path to the destination kzip file.
    
    Returns:
        None
    """
    import tqdm
    scaling = global_params.config['scaling']
    coords = coords / scaling
    skel = Skeleton()
    anno = SkeletonAnnotation()
    anno.scaling = scaling
    node_mapping = {}
    pbar = tqdm.tqdm(total=len(coords) + len(g.edges()), leave=False)
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


def svgraph2kzip(ssv: 'SuperSegmentationObject', kzip_path: str):
    """
    Writes the supervoxel (SV) graph stored in the SuperSegmentationObject to a kzip file.
    The representative coordinate of a SV is used as the corresponding node location.
    
    Args:
        ssv (SuperSegmentationObject): Cell reconstruction object.
        kzip_path (str): Path to the output kzip file.
    
    Returns:
        None
    """
    sv_graph = nx.read_edgelist(ssv.edgelist_path, nodetype=int)
    coords = {ix: ssv.get_seg_obj('sv', ix).rep_coord for ix in sv_graph.nodes}
    import tqdm
    skel = Skeleton()
    anno = SkeletonAnnotation()
    anno.scaling = ssv.scaling
    node_mapping = {}
    pbar = tqdm.tqdm(total=len(coords) + len(sv_graph.edges()), leave=False)
    for v in sv_graph.nodes:
        c = coords[v]
        n = SkeletonNode().from_scratch(anno, c[0], c[1], c[2])
        n.setComment(f'{v}')
        node_mapping[v] = n
        anno.addNode(n)
        pbar.update(1)
    for e in sv_graph.edges():
        anno.addEdge(node_mapping[e[0]], node_mapping[e[1]])
        pbar.update(1)
    skel.add_annotation(anno)
    skel.to_kzip(kzip_path)
    pbar.close()


def stitch_skel_nx(skel_nx: nx.Graph, n_jobs: int = 1) -> nx.Graph:
    """
    Stitches connected components within a graph by recursively adding edges between the 
    closest components.
    
    Args:
        skel_nx (nx.Graph): Networkx graph. Nodes require a 'position' attribute.
        n_jobs (int, optional): Number of jobs used for query of cKDTree. Defaults to 1.
    
    Returns:
        nx.Graph: Single connected component graph.
    """
    if skel_nx.number_of_nodes() == 0:
        return skel_nx
    no_of_seg = nx.number_connected_components(skel_nx)
    if no_of_seg == 1:
        return skel_nx

    skel_nx_nodes = np.array([skel_nx.nodes[ix]['position'] for ix in skel_nx.nodes()], dtype=np.int64)

    while no_of_seg != 1:
        rest_nodes = []
        rest_nodes_ixs = []
        list_of_comp = np.array([c for c in sorted(nx.connected_components(skel_nx), key=len, reverse=True)])
        for single_rest_graph in list_of_comp[1:]:
            rest_nodes += [skel_nx_nodes[int(ix)] for ix in single_rest_graph]
            rest_nodes_ixs += list(single_rest_graph)
        current_set_of_nodes = [skel_nx_nodes[int(ix)] for ix in list_of_comp[0]]
        current_set_of_nodes_ixs = list(list_of_comp[0])
        tree = spatial.cKDTree(rest_nodes, 1)

        thread_lengths, indices = tree.query(current_set_of_nodes, n_jobs=n_jobs)

        start_thread_index = np.argmin(thread_lengths)
        stop_thread_index = indices[start_thread_index]
        e1 = current_set_of_nodes_ixs[start_thread_index]
        e2 = rest_nodes_ixs[stop_thread_index]
        skel_nx.add_edge(e1, e2)
        no_of_seg -= 1
    return skel_nx
