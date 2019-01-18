# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Sven Dorkenwald, Philipp Schubert, Joergen Kornfeld

import copy
from collections import Counter
from multiprocessing.pool import ThreadPool
import networkx as nx
import numpy as np
import os
import scipy
import scipy.ndimage
from collections import defaultdict
from scipy import spatial
from knossos_utils.skeleton_utils import annotation_to_nx_graph,\
    load_skeleton as load_skeleton_kzip

from .rep_helper import assign_rep_values, colorcode_vertices
from . import segmentation
from .segmentation import SegmentationObject
from .segmentation_helper import load_skeleton, find_missing_sv_views,\
    find_missing_sv_attributes, find_missing_sv_skeletons
from ..mp.mp_utils import start_multiprocess_obj, start_multiprocess_imap
import time
skeletopyze_available = False
from ..reps import log_reps
from ..config import global_params
from ..proc.meshes import in_bounding_box, write_mesh2kzip


def majority_vote(anno, prop, max_dist):
    """
    Smoothes (average using sliding window of 2 times max_dist and majority
    vote) property prediction in annotation, whereas for axoness somata are
    untouched.

    Parameters
    ----------
    anno : SkeletonAnnotation
    prop : str
        which property to average
    max_dist : int
        maximum distance (in nm) for sliding window used in majority voting
    """
    old_anno = copy.deepcopy(anno)
    nearest_nodes_list = nodes_in_pathlength(old_anno, max_dist)
    for nodes in nearest_nodes_list:
        curr_node_id = nodes[0].getID()
        new_node = anno.getNodeByID(curr_node_id)
        if prop == "axoness":
            if int(new_node.data["axoness_pred"]) == 2:
                new_node.data["axoness_pred"] = 2
                continue
        property_val = [int(n.data[prop+'_pred']) for n in nodes if int(n.data[prop+'_pred']) != 2]
        counter = Counter(property_val)
        new_ax = counter.most_common()[0][0]
        new_node.setDataElem(prop+'_pred', new_ax)


def nodes_in_pathlength(anno, max_path_len):
    """Find nodes reachable in max_path_len from source node, calculated for
    every node in anno.

    Parameters
    ----------
    anno : AnnotationObject
    max_path_len : float
        Maximum distance from source node

    Returns
    -------
    list of lists containing reachable nodes in max_path_len where
    outer list has length len(anno.getNodes())
    """
    skel_graph = annotation_to_nx_graph(anno)
    list_reachable_nodes = []
    for source_node in anno.getNodes():
        reachable_nodes = [source_node]
        curr_path_length = 0.0
        for edge in nx.bfs_edges(skel_graph, source_node):
            next_node = edge[1]
            next_node_coord = np.array(next_node.getCoordinate_scaled())
            curr_path_length += np.linalg.norm(next_node_coord - np.array(edge[0].getCoordinate_scaled()))
            if curr_path_length > max_path_len:
                break
            reachable_nodes.append(next_node)
        list_reachable_nodes.append(reachable_nodes)
    return list_reachable_nodes


def predict_sso_celltype(sso, model, nb_views=20, overwrite=False):
    sso.load_attr_dict()
    if not overwrite and "celltype_cnn" in sso.attr_dict:
        return
    out_d = sso_views_to_modelinput(sso, nb_views)
    res = model.predict_proba(out_d)
    clf = np.argmax(res, axis=1)
    ls, cnts = np.unique(clf, return_counts=True)
    pred = ls[np.argmax(cnts)]
    sso.save_attributes(["celltype_cnn"], [pred])
    sso.save_attributes(["celltype_cnn_probas"], [res])


def sso_views_to_modelinput(sso, nb_views, view_key=None):
    np.random.seed(0)
    assert len(sso.sv_ids) > 0
    views = sso.load_views(view_key=view_key)
    np.random.shuffle(views)
    # view shape: (#multi-views, 4 channels, 2 perspectives, 128, 256)
    views = views.swapaxes(1, 0).reshape((4, -1, 128, 256))
    assert views.shape[1] > 0
    if views.shape[1] < nb_views:
        rand_ixs = np.random.choice(np.arange(views.shape[1]),
                                    nb_views - views.shape[1])
        views = np.append(views, views[:, rand_ixs], axis=1)
    nb_samples = np.floor(views.shape[1] / nb_views)
    assert nb_samples > 0
    out_d = views[:, :int(nb_samples * nb_views)]
    out_d = out_d.reshape((4, -1, nb_views, 128, 256)).swapaxes(1, 0)
    return out_d


def radius_correction(sso):
    """
    radius correction : algorithm find nodes first and iterates over the finder bunch of meshes
    :param sso: super segmentation object
    :return: skeleton with the corrected diameters in the key 'diameters'
    """

    skel_radius = {}

    skel_node = sso.skeleton['nodes']
    diameters = sso.skeleton['diameters']
    # vert_sparse = vert[0::10]
    vert_sparse = sso.mesh[1].reshape((-1, 3))
    tree = spatial.cKDTree(skel_node * np.array([10, 10, 20]))
    centroid_arr = [[0, 0, 0]]


    dists, ixs = tree.query(vert_sparse, 1)
    all_found_node_ixs = np.unique(ixs)
    found_coords = skel_node[all_found_node_ixs]
    all_skel_node_ixs = np.arange(len(skel_node))
    missing_coords_ixs = list(set(all_skel_node_ixs) - set(all_found_node_ixs))

    for ii, el in enumerate(all_found_node_ixs):
        for i in np.where(ixs == el):
            vert = [vert_sparse[a] / np.array([10, 10, 20]) for a in i]

            x = [p[0] for p in vert]
            y = [p[1] for p in vert]
            z = [p[2] for p in vert]
            centroid = np.asarray((sum(x) / len(vert), sum(y) / len(vert), sum(z) / len(vert)))
            rad = []
            for vert_el in vert:
                rad.append(np.linalg.norm(centroid - vert_el))
            med_rad = np.median(rad)
            new_rad = [inx for inx in rad if abs(inx-med_rad) < 0.9*np.var(rad)]

            med_rad = np.median(new_rad)
            if new_rad == []:
                med_rad = diameters[el]
            else:
                log_reps.debug(rad, vert)
            diameters[el] = med_rad * 2
            if el < len(found_coords):
                skel_radius[str(found_coords[el])] = med_rad
            break
    found_coords = skel_node[all_found_node_ixs]
    found_tree = spatial.cKDTree(found_coords)
    for el in missing_coords_ixs:
        nearest_found_node = found_tree.query(skel_node[el], 1)
        diameters[el] = diameters[nearest_found_node[1]]
    sso.skeleton['diameters'] = diameters
    return sso.skeleton


def radius_correction_found_vertices(sso, plump_factor=1, num_found_vertices=10):
    """
    Algorithm finds two nearest vertices and takes the median of the distances for every node

    Parameters
    ----------
    sso : SuperSegmentationObject
    plump_factor : int
        multiplication factor for the radius
    num_found_vertices : int
        number of closest vertices queried for the node

    Returns
    -------
    skeleton with diameters estimated
    """

    skel_node = sso.skeleton['nodes']
    diameters = sso.skeleton['diameters']

    vert_sparse = sso.mesh[1].reshape((-1, 3))
    tree = spatial.cKDTree(vert_sparse)
    dists, all_found_vertices_ixs = tree.query(skel_node * sso.scaling, num_found_vertices)

    for ii, el in enumerate(skel_node):
        diameters[ii] = np.median(dists[ii]) *2/10

    sso.skeleton['diameters'] = diameters*plump_factor
    return sso.skeleton


def get_sso_axoness_from_coord(sso, coord, k=5):
    """
    Finds k nearest neighbor nodes within sso skeleton and returns majority 
    class of dendrite (0), axon (1) or soma (2).

    Parameters
    ----------
    sso : SuperSegmentationObject
    coord : np.array
        unscaled coordinate

    Returns
    -------
    int
    """
    coord = np.array(coord) * np.array(sso.scaling)
    sso.load_skeleton()
    kdt = spatial.cKDTree(sso.skeleton["nodes"] * np.array(sso.scaling))
    dists, ixs = kdt.query(coord, k=k)
    ixs = ixs[dists != np.inf]
    axs = sso.skeleton["axoness"][ixs]
    cnt = Counter(axs)
    return cnt.most_common(n=1)[0][0]


def calculate_skeleton(sso, size_threshold=1e20, kd=None,
                       coord_scaling=(8, 8, 4), plain=False, cleanup=True,
                       nb_threads=1):
    raise DeprecationWarning("Use 'create_sso_skeleton' instead.")
    if np.product(sso.shape) < size_threshold:
        # vx = self.load_voxels_downsampled(coord_scaling)
        # vx = self.voxels[::coord_scaling[0],
        #                  ::coord_scaling[1],
        #                  ::coord_scaling[2]]
        vx = sso.load_voxels_downsampled(downsampling=coord_scaling)
        vx = scipy.ndimage.morphology.binary_closing(
            np.pad(vx, 3, mode="constant", constant_values=0), iterations=3)
        vx = vx[3: -3, 3: -3, 3:-3]

        if plain:
            nodes, edges, diameters = \
                reskeletonize_plain(vx, coord_scaling=coord_scaling)
            nodes = np.array(nodes, dtype=np.int) + sso.bounding_box[0]
        else:
            nodes, edges, diameters = \
                reskeletonize_chunked(sso.id, sso.shape,
                                      sso.bounding_box[0],
                                      sso.scaling,
                                      voxels=vx,
                                      coord_scaling=coord_scaling,
                                      nb_threads=nb_threads)

    elif kd is not None:
        nodes, edges, diameters = \
            reskeletonize_chunked(sso.id, sso.shape,
                                  sso.bounding_box[0], sso.scaling,
                                  kd=kd, coord_scaling=coord_scaling,
                                  nb_threads=nb_threads)
    else:
        return

    nodes = np.array(nodes, dtype=np.int)
    edges = np.array(edges, dtype=np.int)
    diameters = np.array(diameters, dtype=np.float)

    sso.skeleton = {}
    sso.skeleton["edges"] = edges
    sso.skeleton["nodes"] = nodes
    sso.skeleton["diameters"] = diameters

    if cleanup:
        for i in range(2):
            if len(sso.skeleton["edges"]) > 2:
                sso.skeleton = cleanup_skeleton(sso.skeleton,
                                                    coord_scaling)


def load_voxels_downsampled(sso, downsampling=(2, 2, 1), nb_threads=10):
    def _load_sv_voxels_thread(args):
        sv_id = args[0]
        sv = segmentation.SegmentationObject(sv_id,
                                             obj_type="sv",
                                             version=sso.version_dict[
                                                 "sv"],
                                             working_dir=sso.working_dir,
                                             config=sso.config,
                                             voxel_caching=False)
        if sv.voxels_exist:
            box = [np.array(sv.bounding_box[0] - sso.bounding_box[0],
                            dtype=np.int)]

            box[0] /= downsampling
            size = np.array(sv.bounding_box[1] -
                            sv.bounding_box[0], dtype=np.float)
            size = np.ceil(size.astype(np.float) /
                           downsampling).astype(np.int)

            box.append(box[0] + size)

            sv_voxels = sv.voxels
            if not isinstance(sv_voxels, int):
                sv_voxels = sv_voxels[::downsampling[0],
                            ::downsampling[1],
                            ::downsampling[2]]

                voxels[box[0][0]: box[1][0],
                box[0][1]: box[1][1],
                box[0][2]: box[1][2]][sv_voxels] = True

    downsampling = np.array(downsampling, dtype=np.int)

    if len(sso.sv_ids) == 0:
        return None

    voxel_box_size = sso.bounding_box[1] - sso.bounding_box[0]
    voxel_box_size = voxel_box_size.astype(np.float)

    voxel_box_size = np.ceil(voxel_box_size / downsampling).astype(np.int)

    voxels = np.zeros(voxel_box_size, dtype=np.bool)

    multi_params = []
    for sv_id in sso.sv_ids:
        multi_params.append([sv_id])

    if nb_threads > 1:
        pool = ThreadPool(nb_threads)
        pool.map(_load_sv_voxels_thread, multi_params)
        pool.close()
        pool.join()
    else:
        map(_load_sv_voxels_thread, multi_params)

    return voxels


def create_new_skeleton(sv_id, sso):
    so = SegmentationObject(sv_id, obj_type="sv",
                            version=sso.version_dict[
                                "sv"],
                            working_dir=sso.working_dir,
                            config=sso.config)
    so.enable_locking = False
    so.load_attr_dict()
    nodes, diameters, edges = load_skeleton(so)

    return nodes, diameters, edges


def convert_coord(coord_list, scal):
    return np.array([coord_list[1] + 1, coord_list[0] + 1,
                     coord_list[2] + 1]) * np.array(scal)


def prune_stub_branches(sso=None, nx_g=None, scal=None, len_thres=1000,
                        preserve_annotations=True):
    """
    Removes short stub branches, that are often added by annotators but
    hardly represent true morphology.

    Parameters
    ----------
    sso : SuperSegmentationObject
    nx_g : network kx graph
    scal : array of size 3
        the scaled up factor
    len_thres : int
        threshold of the length below which it will be pruned
    preserve_annotations : bool

    Returns
    -------
    pruned network kx graph
    """
    if scal is None:
        scal = global_params.get_dataset_scaling()
    pruning_complete = False

    if preserve_annotations:
        new_nx_g = nx_g.copy()
    else:
        new_nx_g = nx_g

    # find all tip nodes in an anno, ie degree 1 nodes
    while not pruning_complete:
        nx_g = new_nx_g.copy()
        end_nodes = list({k for k, v in dict(nx_g.degree()).items() if v == 1})
        # DFS to first branch node
        for end_node in end_nodes:
            prune_nodes = []
            for curr_node in nx.traversal.dfs_preorder_nodes(nx_g, end_node):
                if nx_g.degree(curr_node) > 2:
                    loc_end = convert_coord(nx_g.node[end_node]['position'], scal)
                    loc_curr = convert_coord(nx_g.node[curr_node]['position'], scal)
                    b_len = np.linalg.norm(loc_end - loc_curr)

                    if b_len < len_thres:
                        # remove this stub, i.e. prune the nodes that were
                        # collected on our way to the branch point
                        for prune_node in prune_nodes:
                            new_nx_g.remove_node(prune_node)
                        break
                    else:
                        break
                prune_nodes.append(curr_node)
        if len(new_nx_g.nodes) == len(nx_g.nodes):
            pruning_complete = True
    # TODO: uncomment, or fix by using
    if nx.number_connected_components(new_nx_g) != 1:
        msg = 'Pruning of SV skeletons failed during "prune_stub_branches' \
              '" with {} connected components. Please check the underlying' \
              ' SSV {}. Performing stitching method to add missing edg' \
              'es recursively.'.format(nx.number_connected_components(new_nx_g),
                                       sso.id)
        new_nx_g = stitch_skel_nx(new_nx_g)
        log_reps.critical(msg)
    # # Important assert. Please don't remove
    # assert nx.number_connected_components(new_nx_g) == 1,\
    #     'Additional connected components created after pruning!'
    if sso is not None:
        sso = from_netkx_to_sso(sso, new_nx_g)
    return sso, new_nx_g


def sparsify_skeleton(sso, skel_nx, dot_prod_thresh=0.8, max_dist_thresh=500,
                      min_dist_thresh=50):
    """
    Reduces nodes in the skeleton. (from dense stacking to sparsed stacking)

    Parameters
    ----------
    sso : Super Segmentation Object
    dot_prod_thresh : float
        the 'straightness' of the edges
    skel_nx : networkx graph of the sso skel
    max_dist_thresh : int
        maximum distance desired between every node
    min_dist_thresh : int
        minimum distance desired between every node

    Returns
    -------
    sso containing the sparsed skeleton
    """

    sso.load_skeleton()
    scal = sso.scaling
    change = 1
    if sso.skeleton is None:
        sso.skeleton = dict()

    while change > 0:
        change = 0
        visiting_nodes = list({k for k, v in dict(skel_nx.degree()).items() if v == 2})
        for visiting_node in visiting_nodes:
            neighbours = [n for n in skel_nx.neighbors(visiting_node)]
            if skel_nx.degree(visiting_node) == 2:
                left_node = neighbours[0]
                right_node = neighbours[1]
                vector_left_node = np.array([int(skel_nx.node[left_node]['position'][ix]) - int(skel_nx.node[visiting_node]['position'][ix]) for ix in range(3)]) * scal
                vector_right_node = np.array([int(skel_nx.node[right_node]['position'][ix]) - int(skel_nx.node[visiting_node]['position'][ix]) for ix in range(3)]) * scal

                dot_prod = np.dot(vector_left_node/ np.linalg.norm(vector_left_node),vector_right_node/ np.linalg.norm(vector_right_node))
                dist = np.linalg.norm([int(skel_nx.node[right_node]['position'][ix]*scal[ix]) - int(skel_nx.node[left_node]['position'][ix]*scal[ix]) for ix in range(3)])

                if (abs(dot_prod) > dot_prod_thresh and dist < max_dist_thresh) or dist <= min_dist_thresh:

                    skel_nx.remove_node(visiting_node)
                    skel_nx.add_edge(left_node,right_node)
                    change += 1

    sso.skeleton['nodes'] = np.array([skel_nx.node[ix]['position'] for ix in skel_nx.nodes()], dtype=np.uint32)
    sso.skeleton['diameters'] = np.zeros(len(sso.skeleton['nodes']), dtype=np.float)

    temp_edges = np.array(skel_nx.edges()).reshape(-1)
    temp_edges_sorted = np.unique(np.sort(temp_edges))
    temp_edges_dict = {}

    for ii, ix in enumerate(temp_edges_sorted):
        temp_edges_dict[ix] = ii
    temp_edges = [temp_edges_dict[ix] for ix in temp_edges]

    sso.skeleton['edges'] = np.array(temp_edges).reshape([-1, 2])

    nx_g = nx.Graph()
    for inx, single_node in enumerate(sso.skeleton['nodes']):
        nx_g.add_node(inx, position=single_node)

    nx_g.add_edges_from(np.array(temp_edges).reshape([-1, 2]))

    return sso, nx_g


def smooth_skeleton(skel_nx, scal=None):
    if scal is None:
        scal = global_params.get_dataset_scaling()
    visiting_nodes = list({k for k, v in dict(skel_nx.degree()).items() if v == 2})

    for index, visiting_node in enumerate(visiting_nodes):

        neighbours = [n for n in skel_nx.neighbors(visiting_node)]

        # Why all these if == 2 statements?
        if skel_nx.degree(visiting_node) == 2:
            left_node = neighbours[0]
            right_node = neighbours[1]

        if skel_nx.degree(left_node) == 2 and skel_nx.degree(right_node) == 2:
                vector_left_node = np.array(
                    [int(skel_nx.node[left_node]['position'][ix]) - int(skel_nx.node[visiting_node]['position'][ix]) for ix in
                     range(3)]) * scal
                vector_right_node = np.array(
                    [int(skel_nx.node[right_node]['position'][ix]) - int(skel_nx.node[visiting_node]['position'][ix]) for ix
                     in range(3)]) * scal

                dot_prod = np.dot(vector_left_node / np.linalg.norm(vector_left_node),
                                  vector_right_node / np.linalg.norm(vector_right_node))
                dist = np.linalg.norm([int(skel_nx.node[right_node]['position'][ix] * scal[ix]) - int(
                    skel_nx.node[left_node]['position'][ix] * scal[ix]) for ix in range(3)])

                if abs(dot_prod) < 0.3:

                    x_dist = np.linalg.norm([int(skel_nx.node[visiting_node]['position'][ix] * scal[ix]) - int(
                        skel_nx.node[left_node]['position'][ix] * scal[ix]) for ix in range(3)])

                    y_dist = np.linalg.norm([int(skel_nx.node[visiting_node]['position'][ix] * scal[ix]) - int(
                        skel_nx.node[right_node]['position'][ix] * scal[ix]) for ix in range(3)])

                    p = [(int(skel_nx.node[right_node]['position'][ix] + int(skel_nx.node[left_node]['position'][ix]))*x_dist/(x_dist +y_dist)) for ix in range(3)]

                    final_node = (1*p + skel_nx.node[visiting_node]['position']*99)/100
                    print ('original ' , skel_nx.node[visiting_node]['position'], 'final ',final_node)

                    skel_nx.node[visiting_node]['position'] = np.array(final_node, dtype = np.int)

    return skel_nx


def from_netkx_to_sso(sso, skel_nx):
    sso.skeleton = {}
    sso.skeleton['nodes'] = np.array([skel_nx.node[ix]['position'] for ix in
                                      skel_nx.nodes()], dtype=np.uint32)
    sso.skeleton['diameters'] = np.zeros(len(sso.skeleton['nodes']),
                                         dtype=np.float)

    assert nx.number_connected_components(skel_nx) == 1

    # Important bit, please don't remove (needed after pruning)
    temp_edges = np.array(skel_nx.edges()).reshape(-1)
    temp_edges_sorted = np.unique(np.sort(temp_edges))
    temp_edges_dict = {}

    for ii, ix in enumerate(temp_edges_sorted):
        temp_edges_dict[ix] = ii

    temp_edges = [temp_edges_dict[ix] for ix in temp_edges]

    temp_edges = np.array(temp_edges, dtype=np.uint).reshape([-1, 2])
    sso.skeleton['edges'] = temp_edges

    return sso


def from_sso_to_netkx(sso):
    skel_nx = nx.Graph()
    sso.load_attr_dict()
    ssv_skel = {'nodes': [], 'edges': [], 'diameters': []}

    for sv_id in sso.sv_ids:
        nodes, diameters, edges = create_new_skeleton(sv_id, sso)

        ssv_skel['edges'] = np.concatenate(
            (ssv_skel['edges'], [(ix + (len(ssv_skel['nodes'])) / 3) for ix in edges]), axis=0)
        ssv_skel['nodes'] = np.concatenate((ssv_skel['nodes'], nodes), axis=0)

        ssv_skel['diameters'] = np.concatenate((ssv_skel['diameters'], diameters), axis=0)

    new_nodes = np.array(ssv_skel['nodes'], dtype=np.uint32).reshape((-1, 3))
    if len(new_nodes) == 0:
        sso.skeleton = ssv_skel
        return

    for inx, single_node in enumerate(new_nodes):
        skel_nx.add_node(inx, position=single_node)

    new_edges = np.array(ssv_skel['edges']).reshape((-1, 2))
    new_edges = [tuple(ix) for ix in new_edges]
    skel_nx.add_edges_from(new_edges)

    return skel_nx


def stitch_skel_nx(skel_nx):

    no_of_seg = nx.number_connected_components(skel_nx)

    skel_nx_nodes = [ii['position'] for ix, ii in skel_nx.node.items()]

    new_nodes = np.array([skel_nx.node[ix]['position'] for ix in skel_nx.nodes()], dtype=np.uint32)
    while no_of_seg != 1:

        rest_nodes = []
        current_set_of_nodes = []

        list_of_comp = np.array([c for c in sorted(nx.connected_components(skel_nx), key=len, reverse=True)])

        for single_rest_graph in list_of_comp[1:]:
            rest_nodes = rest_nodes + [skel_nx_nodes[int(ix)] for ix in single_rest_graph]

        for single_rest_graph in list_of_comp[:1]:
            current_set_of_nodes = current_set_of_nodes + [skel_nx_nodes[int(ix)] for ix in
                                                           single_rest_graph]

        tree = spatial.cKDTree(rest_nodes, 1)
        thread_lengths, indices = tree.query(current_set_of_nodes)

        start_thread_index = np.argmin(thread_lengths)
        stop_thread_index = indices[start_thread_index]

        start_thread_node = \
        np.where(np.sum(np.subtract(new_nodes, current_set_of_nodes[start_thread_index]), axis=1) == 0)[0][0]
        stop_thread_node = np.where(np.sum(np.subtract(new_nodes, rest_nodes[stop_thread_index]), axis=1) == 0)[0][0]

        skel_nx.add_edge(start_thread_node, stop_thread_node)
        no_of_seg -= 1

    return skel_nx


def create_sso_skeleton(sso, pruning_thresh=700, sparsify=True):
    """
    Creates the super super voxel skeleton

    Parameters
    ----------
    sso : Super Segmentation Object
    pruning_thresh : int
        threshold for pruning
    sparsify : bool
        will sparsify if True otherwise not

    Returns
    -------

    """
    # Creating network kx graph from sso skel
    skel_nx = from_sso_to_netkx(sso)

    if sparsify:
        sso, skel_nx = sparsify_skeleton(sso, skel_nx)

    # Stitching sso skeletons,
    skel_nx = stitch_skel_nx(skel_nx)

    # Sparse again after stitching. Inexpensive.
    if sparsify:
        sso, skel_nx = sparsify_skeleton(sso, skel_nx)

    # Pruning the stitched sso skeletons
    sso, skel_nx = prune_stub_branches(sso, skel_nx, len_thres=pruning_thresh)

    # Estimating the radii
    sso.skeleton = radius_correction_found_vertices(sso)

    return sso


# New Implementation of skeleton generation which makes use of ssv.rag

def from_netkx_to_arr(skel_nx):
    skeleton = {}
    skeleton['nodes'] = np.array(
        [skel_nx.node[ix]['position'] for ix in skel_nx.nodes()],
        dtype=np.uint32)
    skeleton['diameters'] = np.zeros(len(skeleton['nodes']), dtype=np.float32)

    # Important bit, please don't remove (needed after pruning)
    temp_edges = np.array(list(skel_nx.edges())).reshape(-1)
    temp_edges_sorted = np.unique(np.sort(temp_edges))
    temp_edges_dict = {}

    for ii, ix in enumerate(temp_edges_sorted):
        temp_edges_dict[ix] = ii

    temp_edges = [temp_edges_dict[ix] for ix in temp_edges]

    temp_edges = np.array(temp_edges, dtype=np.uint).reshape([-1, 2])
    skeleton['edges'] = temp_edges

    return skeleton['nodes'], skeleton['diameters'], skeleton['edges']


def sparsify_skeleton_fast(skel_nx, scal=None, dot_prod_thresh=0.8,
                           max_dist_thresh=500, min_dist_thresh=50):
    """
    Reduces nodes in the skeleton. (from dense stacking to sparsed stacking)

    Parameters
    ----------
    sso : Super Segmentation Object
    dot_prod_thresh : float
        the 'straightness' of the edges
    skel_nx : networkx graph of the sso skel
    max_dist_thresh : int
        maximum distance desired between every node
    min_dist_thresh : int
        minimum distance desired between every node
    scal : np.array

    Returns
    -------
    sso containing the sparsed skeleton
    """

    if scal is None:
        scal = global_params.get_dataset_scaling()
    change = 1

    while change > 0:
        change = 0
        visiting_nodes = list({k for k, v in dict(skel_nx.degree()).items() if v == 2})
        for visiting_node in visiting_nodes:
            neighbours = [n for n in skel_nx.neighbors(visiting_node)]
            if skel_nx.degree(visiting_node) == 2:
                left_node = neighbours[0]
                right_node = neighbours[1]
                vector_left_node = np.array([int(skel_nx.node[left_node]['position'][ix]) - int(skel_nx.node[visiting_node]['position'][ix]) for ix in range(3)]) * scal
                vector_right_node =np.array([int(skel_nx.node[right_node]['position'][ix]) - int(skel_nx.node[visiting_node]['position'][ix]) for ix in range(3)]) * scal

                dot_prod = np.dot(vector_left_node/ np.linalg.norm(vector_left_node),vector_right_node/ np.linalg.norm(vector_right_node))
                dist = np.linalg.norm([int(skel_nx.node[right_node]['position'][ix]*scal[ix]) - int(skel_nx.node[left_node]['position'][ix]*scal[ix]) for ix in range(3)])

                if (abs(dot_prod) > dot_prod_thresh and dist < max_dist_thresh) or dist <= min_dist_thresh:

                    skel_nx.remove_node(visiting_node)
                    skel_nx.add_edge(left_node, right_node)
                    change += 1
    return skel_nx


def create_new_skeleton_fast(args):
    so, sparsify = args
    so.enable_locking = False
    so.load_attr_dict()
    # ignore diameters, will be populated at the and of create_sso_skeleton_fast
    nodes, diameters, edges = load_skeleton(so)
    edges = np.array(edges).reshape((-1, 2))
    nodes = np.array(nodes).reshape((-1, 3)).astype(np.uint32)
    # create nx graph
    skel_nx = nx.Graph()
    skel_nx.add_nodes_from([(ix, dict(position=coord)) for ix, coord
                            in enumerate(nodes)])

    new_edges = [tuple(ix) for ix in edges]
    skel_nx.add_edges_from(new_edges)
    if sparsify:
        skel_nx = sparsify_skeleton_fast(skel_nx)
    n_cc = nx.number_connected_components(skel_nx)
    if n_cc > 1:
        log_reps.critical('SV {} contained {} connected components in its skel'
                          'eton representation. Stitching now.'
                          ''.format(so.id, n_cc))
        skel_nx = stitch_skel_nx(skel_nx)
    nodes, diameters, edges = from_netkx_to_arr(skel_nx)
    # just get nodes, diameters and edges
    return nodes, diameters, edges


def from_sso_to_netkx_fast(sso, sparsify=True):
    """
    Stitches the SV skeletons using sso.rag

    Parameters
    ----------
    sso : SuperSegmentationObject
    sparsify : bool
        Sparsify SV skeletons before stitching

    Returns
    -------
    nx.Graph
    """
    skel_nx = nx.Graph()
    sso.load_attr_dict()
    ssv_skel = {'nodes': [], 'edges': [], 'diameters': []}
    res = start_multiprocess_imap(create_new_skeleton_fast,
                                  [(sv, sparsify) for sv in sso.svs],
                                  nb_cpus=sso.nb_cpus, show_progress=False)
    nodes, diameters, edges, sv_id_arr = [], [], [], []
    # first offset is 0, last length is not needed
    n_nodes_per_sv = [0] + list(
        np.cumsum([len(el[0]) for el in res])[:-1])

    for ii in range(len(res)):
        if len(res[ii][0]) == 0:  # skip missing / empty skeletons, e.g. happens for small SVs
            continue
        nodes.append(res[ii][0])
        diameters.append(res[ii][1])
        edges.append(res[ii][2] + int(n_nodes_per_sv[ii]))
        # store mapping from node to SV ID
        sv_id_arr.append([sso.sv_ids[ii]] * len(res[ii][0]))

    ssv_skel['nodes'] = np.concatenate(nodes)
    ssv_skel['diameters'] = np.concatenate(diameters, axis=0)
    sv_id_arr = np.concatenate(sv_id_arr)
    node_ix_arr = np.arange(len(sv_id_arr))
    # stitching
    if len(sso.sv_ids) > 1:
        # iterates over SV object edges
        for e1, e2 in sso.load_edgelist():
            # get closest edge between SV nodes in question and new edge add to edges
            nodes1 = ssv_skel['nodes'][sv_id_arr == e1.id] * sso.scaling
            nodes2 = ssv_skel['nodes'][sv_id_arr == e2.id] * sso.scaling
            nodes1 = nodes1.astype(np.float32)
            nodes2 = nodes2.astype(np.float32)
            if len(nodes1) == 0 or len(nodes2) == 0:
                continue  # SV without skeleton
            nodes1_ix = node_ix_arr[sv_id_arr == e1.id]
            nodes2_ix = node_ix_arr[sv_id_arr == e2.id]
            tree = spatial.cKDTree(nodes1)
            dists, node_ixs1 = tree.query(nodes2)
            # # get global index of nodes
            ix2 = nodes2_ix[np.argmin(dists)]
            ix1 = nodes1_ix[node_ixs1[np.argmin(dists)]]
            # node_dist_check = np.linalg.norm(ssv_skel['nodes'][ix1].astype(np.float32) -
            #                                   ssv_skel['nodes'][ix2].astype(np.float32))
            # if np.min(dists) < node_dist_check or node_dist_check > 10e3:
            #     raise ValueError
            edges.append(np.array([[ix1, ix2]], dtype=np.uint32))
    ssv_skel['edges'] = np.concatenate(edges)

    if len(ssv_skel['nodes']) == 0:
        sso.skeleton = ssv_skel
        return
    skel_nx.add_nodes_from([(ix, dict(position=coord)) for ix, coord
                            in enumerate(ssv_skel['nodes'])])
    edges = [tuple(ix) for ix in ssv_skel['edges']]
    skel_nx.add_edges_from(edges)
    if nx.number_connected_components(skel_nx) != 1:
        msg = 'Stitching of SV skeletons failed during "from_sso_to_netkx_' \
              'fast" with {} connected components. Please check the underlying'\
              ' RAG of SSV {}. Performing stitching method to add missing edg' \
              'es recursively.'.format(nx.number_connected_components(skel_nx),
                                       sso.id)
        skel_nx = stitch_skel_nx(skel_nx)
        log_reps.critical(msg)
        assert nx.number_connected_components(skel_nx) == 1
    sso.skeleton = ssv_skel
    return skel_nx


def create_sso_skeleton_fast(sso, pruning_thresh=700, sparsify=True):
    """
    Creates the super super voxel skeleton. NOTE: If the underlying RAG does
    not connect close-by SVs do not use this method,
    but use 'create_sso_skeleton' instead. The latter will recursively add the
    shortest edge between two different SVs. It is ~10x faster on single CPU.
    To use multi-processing, set ssv.nb_cpus > 1

    Parameters
    ----------
    sso : Super Segmentation Object
    pruning_thresh : int
        threshold for pruning
    sparsify : bool
        will sparsify if True otherwise not

    Returns
    -------

    """
    # Creating network kx graph from sso skel
    log_reps.debug('Creating skeleton of SSO {}'.format(sso.id))
    skel_nx = from_sso_to_netkx_fast(sso)
    log_reps.debug('Number CC after stitching and sparsifying SSO {}:'.format(sso.id),
          nx.number_connected_components(skel_nx))
    # Sparse again after stitching. Inexpensive.
    if sparsify:
        sso, skel_nx = sparsify_skeleton(sso, skel_nx)
        log_reps.debug(
            'Number CC after 2nd sparsification SSO {}:'.format(sso.id),
            nx.number_connected_components(skel_nx))
    # Pruning the stitched sso skeletons
    sso, skel_nx = prune_stub_branches(sso, skel_nx, len_thres=pruning_thresh)
    log_reps.debug('Number CC after pruning SSO {}:'.format(sso.id),
          nx.number_connected_components(skel_nx))
    # Estimating the radii
    sso.skeleton = radius_correction_found_vertices(sso)

    return sso


def glia_pred_exists(so):
    so.load_attr_dict()
    return "glia_probas" in so.attr_dict


def views2tripletinput(views):
    views = views[:, :, :1] # use first view only
    out_d = np.concatenate([views,
                            np.ones_like(views),
                            np.ones_like(views)], axis=2)
    return out_d.astype(np.float32)


def get_pca_view_hists(sso, t_net, pca):
    views = sso.load_views()
    latent = t_net.predict_proba(views2tripletinput(views))
    latent = pca.transform(latent)
    hist0 = np.histogram(latent[:, 0], bins=50, range=[-2, 2], normed=True)
    hist1 = np.histogram(latent[:, 1], bins=50, range=[-3.2, 3], normed=True)
    hist2 = np.histogram(latent[:, 2], bins=50, range=[-3.5, 3.5], normed=True)
    return np.array([hist0, hist1, hist2])


def save_view_pca_proj(sso, t_net, pca, dest_dir, ls=20, s=6.0, special_points=(),
                       special_markers=(), special_kwargs=()):
    import matplotlib
    matplotlib.use("Agg", warn=False, force=True)
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    views = sso.load_views()
    latent = t_net.predict_proba(views2tripletinput(views))
    latent = pca.transform(latent)
    col = (np.array(latent) - latent.min(axis=0)) / (latent.max(axis=0)-latent.min(axis=0))
    col = np.concatenate([col, np.ones_like(col)[:, :1]], axis=1)
    for ii, (a, b) in enumerate([[0, 1], [0, 2], [1, 2]]):
        fig, ax = plt.subplots()
        plt.scatter(latent[:, a], latent[:, b], c=col, s=s, lw=0.5, marker="o",
                    edgecolors=col)
        if len(special_points) >= 0:
            for kk, sp in enumerate(special_points):
                if len(special_markers) == 0:
                    sm = "x"
                else:
                    sm = special_markers[kk]
                if len(special_kwargs) == 0:
                    plt.scatter(sp[None, a], sp[None, b], s=75.0, lw=2.3,
                                marker=sm, edgecolor="0.3", facecolor="none")
                else:
                    plt.scatter(sp[None, a], sp[None, b], **special_kwargs)
        fig.patch.set_facecolor('white')
        ax.tick_params(axis='x', which='major', labelsize=ls, direction='out',
                       length=4, width=3, right="off", top="off", pad=10)
        ax.tick_params(axis='y', which='major', labelsize=ls, direction='out',
                       length=4, width=3, right="off", top="off", pad=10)

        ax.tick_params(axis='x', which='minor', labelsize=ls, direction='out',
                       length=4, width=3, right="off", top="off", pad=10)
        ax.tick_params(axis='y', which='minor', labelsize=ls, direction='out',
                       length=4, width=3, right="off", top="off", pad=10)
        plt.xlabel(r"$Z_%d$" % (a+1), fontsize=ls)
        plt.ylabel(r"$Z_%d$" % (b+1), fontsize=ls)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(2))
        plt.tight_layout()
        plt.savefig(dest_dir+"/%d_pca_%d%d.png" % (sso.id, a+1, b+1), dpi=400)
        plt.close()


def extract_skel_features(ssv, feature_context_nm=8000, max_diameter=1000,
                          obj_types=("sj", "mi", "vc"), downsample_to=None):
    node_degrees = np.array(list(dict(ssv.weighted_graph().degree()).values()),
                            dtype=np.int)

    sizes = {}
    for obj_type in obj_types:
        objs = ssv.get_seg_objects(obj_type)
        sizes[obj_type] = np.array([obj.size for obj in objs],
                                   dtype=np.int)

    if downsample_to is not None:
        if downsample_to > len(ssv.skeleton["nodes"]):
            downsample_by = 1
        else:
            downsample_by = int(len(ssv.skeleton["nodes"]) /
                                float(downsample_to))
    else:
        downsample_by = 1

    features = []
    for i_node in range(len(ssv.skeleton["nodes"][::downsample_by])):
        this_i_node = i_node * downsample_by
        this_features = []

        paths = nx.single_source_dijkstra_path(ssv.weighted_graph(),
                                               this_i_node,
                                               feature_context_nm)
        neighs = np.array(list(paths.keys()), dtype=np.int)

        neigh_diameters = ssv.skeleton["diameters"][neighs]
        this_features.append(np.mean(neigh_diameters))
        this_features.append(np.std(neigh_diameters))
        hist_feat = np.histogram(neigh_diameters,bins=10,range=(0, max_diameter))[0]
        hist_feat = np.array(hist_feat) / hist_feat.sum()
        this_features += list(hist_feat)
        this_features.append(np.mean(node_degrees[neighs]))

        for obj_type in obj_types:
            neigh_objs = np.array(ssv.skeleton["assoc_%s" % obj_type])[
                neighs]
            neigh_objs = [item for sublist in neigh_objs for item in
                          sublist]
            neigh_objs = np.unique(np.array(neigh_objs))
            if len(neigh_objs) == 0:
                this_features += [0, 0, 0]
                continue

            this_features.append(len(neigh_objs))
            obj_sizes = sizes[obj_type][neigh_objs]
            this_features.append(np.mean(obj_sizes))
            this_features.append(np.std(obj_sizes))

        # box feature
        edge_len = feature_context_nm * 2
        bb = [ssv.skeleton["nodes"][this_i_node], np.array([edge_len,] * 3)]
        vol_tot = feature_context_nm ** 3
        node_density = np.sum(in_bounding_box(ssv.skeleton["nodes"], bb)) / vol_tot
        this_features.append(node_density)

        features.append(np.array(this_features))
    return np.array(features)


def associate_objs_with_skel_nodes(ssv, obj_types=("sj", "vc", "mi"),
                                   downsampling=(8, 8, 4)):
    if ssv.skeleton is None:
        ssv.load_skeleton()

    for obj_type in obj_types:
        voxels = []
        voxel_ids = [0]
        for obj in ssv.get_seg_objects(obj_type):
            vl = obj.load_voxel_list_downsampled_adapt(downsampling)

            if len(vl) == 0:
                continue

            if len(voxels) == 0:
                voxels = vl
            else:
                voxels = np.concatenate((voxels, vl))

            voxel_ids.append(voxel_ids[-1] + len(vl))

        if len(voxels) == 0:
            ssv.skeleton["assoc_%s" % obj_type] = [[]] * len(
                ssv.skeleton["nodes"])
            continue

        voxel_ids = np.array(voxel_ids)

        kdtree = scipy.spatial.cKDTree(voxels * ssv.scaling)
        balls = kdtree.query_ball_point(ssv.skeleton["nodes"] *
                                        ssv.scaling, 500)
        nodes_objs = []
        for i_node in range(len(ssv.skeleton["nodes"])):
            nodes_objs.append(list(np.unique(
                np.sum(voxel_ids[:, None] <= np.array(balls[i_node]),
                       axis=0) - 1)))

        ssv.skeleton["assoc_%s" % obj_type] = nodes_objs

    ssv.save_skeleton(to_kzip=False, to_object=True)


def skelnode_comment_dict(sso):
    comment_dict = {}
    skel = load_skeleton_kzip(sso.skeleton_kzip_path)["skeleton"]
    for n in skel.getNodes():
        c = frozenset(n.getCoordinate())
        comment_dict[c] = n.getComment()
    return comment_dict


def label_array_for_sso_skel(sso, comment_converter):
    """
    Converts skeleton node comments from annotation.xml in
    sso.skeleton_kzip_path (see SkeletonAnnotation class from knossos utils)
    to a label array of length (and same ordering) as sso.skeleton["nodes"].
    If comment was unspecified, it will get label -1

    Parameters
    ----------
    sso : SuperSegmentationObject
    comment_converter : dict
        Key: Comment, Value: integer label

    Returns
    -------
    np.array
        Label array of len(sso.skeleton["nodes"])
    """
    if sso.skeleton is None:
        sso.load_skeleton()
    cd = skelnode_comment_dict(sso)
    label_array = np.ones(len(sso.skeleton["nodes"]), dtype=np.int) * -1
    for ii, n in enumerate(sso.skeleton["nodes"]):
        comment = cd[frozenset(n.astype(np.int))].lower()
        try:
            label_array[ii] = comment_converter[comment]
        except KeyError:
            pass
    return label_array


def write_axpred_cnn(ssv, pred_key_appendix, dest_path=None, k=1):
    if dest_path is None:
        dest_path = ssv.skeleton_kzip_path_views
    pred_key = "axoness_preds%s" % pred_key_appendix
    if not ssv.attr_exists(pred_key):
        log_reps.info("Couldn't find specified axoness prediction. Falling back "
                      "to default.")
        preds = np.array(start_multiprocess_obj("axoness_preds",
                                                   [[sv, {
                                                       "pred_key_appendix": pred_key_appendix}]
                                                    for sv in ssv.svs],
                                                   nb_cpus=ssv.nb_cpus))
        preds = np.concatenate(preds)
    else:
        preds = ssv.lookup_in_attribute_dict(pred_key)
    log_reps.debug("Collected axoness:", Counter(preds).most_common())
    locs = ssv.sample_locations()
    log_reps.debug("Collected locations.")
    pred_coords = np.concatenate(locs)
    assert pred_coords.ndim == 2
    assert pred_coords.shape[1] == 3
    colors = np.array(np.array([[0.6, 0.6, 0.6, 1], [0.841, 0.138, 0.133, 1.],
                                [0.32, 0.32, 0.32, 1.]]) * 255, dtype=np.uint)
    ssv._pred2mesh(pred_coords, preds, "axoness.ply", dest_path=dest_path, k=k,
                   colors=colors)


def _cnn_axoness2skel(sso, pred_key_appendix="", k=1, force_reload=False,
                      save_skel=True, use_cache=False):
    """
    By default, will create 'axoness_preds_cnn' attribute in SSV attribute dict
     and save new skeleton attributes with keys "axoness" and "axoness_probas".

    Parameters
    ----------
    sso : SuperSegmentationObject
    pred_key_appendix : str
    k : int
    force_reload : bool
        Reload SV predictions.
    save_skel : bool
        Save SSV skeleton with prediction attirbutes
    use_cache : bool
        Write intermediate SV predictions in SSV attribute dict to disk

    Returns
    -------

    """
    if k != 1:
        log_reps.warn("Parameter 'k' is deprecated but was set to {}. "
                      "It is not longer used in this method.".format(k))
    if sso.skeleton is None:
        sso.load_skeleton()
    proba_key = "axoness_probas_cnn%s" % pred_key_appendix
    pred_key = "axoness_preds_cnn%s" % pred_key_appendix
    if not sso.attr_exists(pred_key) or not sso.attr_exists(proba_key) or\
            force_reload:
        preds = np.array(start_multiprocess_obj(
            "axoness_preds", [[sv, {"pred_key_appendix": pred_key_appendix}]
                              for sv in sso.svs],
                                                   nb_cpus=sso.nb_cpus))
        probas = np.array(start_multiprocess_obj(
            "axoness_probas", [[sv, {"pred_key_appendix": pred_key_appendix}]
                               for sv in sso.svs], nb_cpus=sso.nb_cpus))
        preds = np.concatenate(preds)
        probas = np.concatenate(probas)
        sso.attr_dict[proba_key] = probas
        sso.attr_dict[pred_key] = preds
        if use_cache:
            sso.save_attributes([proba_key, pred_key], [probas, preds])
    else:
        preds = sso.lookup_in_attribute_dict(pred_key)
        probas = sso.lookup_in_attribute_dict(proba_key)
    loc_coords = np.concatenate(sso.sample_locations())
    assert len(loc_coords) == len(preds), "Number of view coordinates is" \
                                          "different from number of view" \
                                          "predictions. SSO %d" % sso.id
    # find NN in loc_coords for every skeleton node and use their majority
    # prediction
    node_preds = colorcode_vertices(sso.skeleton["nodes"] * sso.scaling,
                                    loc_coords, preds, colors=[0, 1, 2], k=1)
    node_probas, ixs = assign_rep_values(sso.skeleton["nodes"] * sso.scaling,
                                         loc_coords, probas, return_ixs=True)
    assert np.max(ixs) <= len(loc_coords), "Maximum index for sample " \
                                           "coordinates is bigger than " \
                                           "length of sample coordinates."
    sso.skeleton["axoness%s" % pred_key_appendix] = node_preds
    sso.skeleton["axoness_probas%s" % pred_key_appendix] = node_probas
    sso.skeleton["view_ixs"] = ixs
    if save_skel:
        sso.save_skeleton()


def _average_node_axoness_views(sso, pred_key_appendix="", pred_key=None,
                                max_dist=10000, return_res=False,
                                use_cache=False):
    """
    Averages the axoness prediction along skeleton with maximum path length
    of 'max_dist'. Therefore, view indices were mapped to every skeleton
    node and collected while traversing the skeleton. The majority of the
    set of their predictions will be assigned to the source node.
    By default, will create 'axoness_preds_cnn' attribute in SSV attribute dict
     and save new skeleton attribute with key "%s_views_avg%d" % (pred_key, max_dist).

    Parameters
    ----------
    sso : SuperSegmentationObject
    pred_key : str
        Key for the stored SV predictions
    max_dist : int
    return_res : bool
    use_cache : bool
        Write intermediate SV predictions in SSV attribute dict to disk
    """
    if sso.skeleton is None:
        sso.load_skeleton()
    if len(sso.skeleton["edges"]) == 0:
        log_reps.error("Zero edges in skeleton of SSV %d. "
                       "Skipping averaging." % sso.id)
        return
    if pred_key is None:
        pred_key = "axoness_preds_cnn%s" % pred_key_appendix
    elif len(pred_key_appendix) > 0:
        raise ValueError("Only one of the two may be given: 'pred_key' or"
                         "'pred_key_appendix', but not both.")
    if type(pred_key) != str:
        raise ValueError("'pred_key' has to be of type str.")
    if not sso.attr_exists(pred_key) and ("axoness_preds_cnn" not in pred_key):
        if len(pred_key_appendix) > 0:
            log_reps.error("Couldn't find specified axoness prediction. Falling"
                           " back to default (-> per SV stored multi-view "
                           "prediction including SSV context")
        preds = np.array(start_multiprocess_obj(
            "axoness_preds", [[sv, {"pred_key_appendix": pred_key_appendix}]
                              for sv in sso.svs], nb_cpus=sso.nb_cpus))
        preds = np.concatenate(preds)
        sso.attr_dict[pred_key] = preds
        if use_cache:
            sso.save_attributes([pred_key], [preds])
    else:
        preds = sso.lookup_in_attribute_dict(pred_key)
    loc_coords = np.concatenate(sso.sample_locations())
    assert len(loc_coords) == len(preds), "Number of view coordinates is " \
                                          "different from number of view " \
                                          "predictions. SSO %d" % sso.id
    if "view_ixs" not in sso.skeleton.keys():
        log_reps.info("View indices were not yet assigned to skeleton nodes. "
                      "Running now '_cnn_axonness2skel(sso, "
                      "pred_key_appendix=pred_key_appendix, k=1)'")
        _cnn_axoness2skel(sso, pred_key_appendix=pred_key_appendix, k=1,
                          save_skel=not return_res, use_cache=use_cache)
    view_ixs = np.array(sso.skeleton["view_ixs"])
    avg_pred = []

    g = sso.weighted_graph()
    for n in range(g.number_of_nodes()):
        paths = nx.single_source_dijkstra_path(g, n, max_dist)
        neighs = np.array(list(paths.keys()), dtype=np.int)
        unique_view_ixs = np.unique(view_ixs[neighs], return_counts=False)
        cls, cnts = np.unique(preds[unique_view_ixs], return_counts=True)
        c = cls[np.argmax(cnts)]
        avg_pred.append(c)
    if return_res:
        return avg_pred
    sso.skeleton["axoness%s_avg%d" % (pred_key_appendix, max_dist)] = avg_pred
    sso.save_skeleton()


def majority_vote_compartments(sso, ax_pred_key='axoness'):
    """
    By default, will save new skeleton attribute with key
     ax_pred_key + "_comp_maj".

    Parameters
    ----------
    sso : SuperSegmentationObject
    ax_pred_key : str
        Key for the axoness predictions stored in sso.skeleton

    Returns
    -------

    """
    g = sso.weighted_graph(add_node_attr=(ax_pred_key, ))
    soma_free_g = g.copy()
    for n, d in g.nodes(data=True):
        if d[ax_pred_key] == 2:
            soma_free_g.remove_node(n)
    ccs = list(nx.connected_component_subgraphs(soma_free_g))
    new_axoness_dc = nx.get_node_attributes(g, ax_pred_key)
    for cc in ccs:
        preds = [d[ax_pred_key] for n, d in cc.nodes(data=True)]
        cls, cnts = np.unique(preds, return_counts=True)
        majority = cls[np.argmax(cnts)]
        probas = np.array(cnts, dtype=np.float32) / np.sum(cnts)
        # positively bias dendrite assignment
        if (majority == 1) and (probas[cls == 1] < 0.66):
            majority = 0
        for n in cc.nodes():
            new_axoness_dc[n] = majority
    nx.set_node_attributes(g, new_axoness_dc, ax_pred_key)
    new_axoness_arr = np.zeros((len(sso.skeleton["nodes"])))
    for n, d in g.nodes(data=True):
        new_axoness_arr[n] = d[ax_pred_key]
    sso.skeleton[ax_pred_key + "_comp_maj"] = new_axoness_arr
    sso.save_skeleton()


def find_incomplete_ssv_views(ssd, woglia, n_cores=global_params.NCORES_PER_NODE):
    sd = ssd.get_segmentationdataset("sv")
    incomplete_sv_ids = find_missing_sv_views(sd, woglia, n_cores)
    missing_ssv_ids = set()
    for sv_id in incomplete_sv_ids:
        try:
            ssv_id = ssd.mapping_dict_reversed[sv_id]
            missing_ssv_ids.add(ssv_id)
        except KeyError:
            pass  # sv does not exist in this SSD
    return list(missing_ssv_ids)


def find_incomplete_ssv_skeletons(ssd, n_cores=global_params.NCORES_PER_NODE):
    svs = np.concatenate([list(ssv.svs) for ssv in ssd.ssvs])
    incomplete_sv_ids = find_missing_sv_skeletons(svs, n_cores)
    missing_ssv_ids = set()
    for sv_id in incomplete_sv_ids:
        try:
            ssv_id = ssd.mapping_dict_reversed[sv_id]
            missing_ssv_ids.add(ssv_id)
        except KeyError:
            pass  # sv does not exist in this SSD
    return list(missing_ssv_ids)


def find_missing_sv_attributes_in_ssv(ssd, attr_key, n_cores=global_params.NCORES_PER_NODE):
    sd = ssd.get_segmentationdataset("sv")
    incomplete_sv_ids = find_missing_sv_attributes(sd, attr_key, n_cores)
    missing_ssv_ids = set()
    for sv_id in incomplete_sv_ids:
        try:
            ssv_id = ssd.mapping_dict_reversed[sv_id]
            missing_ssv_ids.add(ssv_id)
        except KeyError:
            pass  # sv does not exist in this SSD
    return list(missing_ssv_ids)


def predict_views_semseg(views, model, batch_size=20, verbose=False):
    """
    Predicts a view array of shape [N_LOCS, N_CH, N_VIEWS, X, Y] with
    N_LOCS locations each with N_VIEWS perspectives, N_CH different channels
    (e.g. shape of cell, mitochondria, synaptic junctions and vesicle clouds).

    Parameters
    ----------
    views : np.array
        shape of [N_LOCS, N_CH, N_VIEWS, X, Y] as uint8 scaled from 0 to 255
    model : pytorch model
    batch_size : int
    verbose : bool

    Returns
    -------

    """
    if verbose:
        log_reps.info('Reshaping view array with shape {}.'
                      ''.format(views.shape))
    views = views.astype(np.float32) / 255.
    views = views.swapaxes(1, 2)  # swap channel and view axis
    # N, 2, 4, 128, 256
    orig_shape = views.shape
    # reshape to predict single projections, N*2, 4, 128, 256
    views = views.reshape([-1] + list(orig_shape[2:]))

    if verbose:
        log_reps.info('Predicting view array with shape {}.'
                      ''.format(views.shape))
    # predict and reset to original shape: N, 2, 4, 128, 256
    labeled_views = model.predict_proba(views, bs=batch_size, verbose=verbose)
    labeled_views = np.argmax(labeled_views, axis=1)[:, None]
    if verbose:
        log_reps.info('Finished prediction of view array with shape {}.'
                      ''.format(views.shape))
    labeled_views = labeled_views.reshape(list(orig_shape[:2])
                                          + list(labeled_views.shape[1:]))
    # swap axes to get source shape
    labeled_views = labeled_views.swapaxes(2, 1)
    return labeled_views


def pred_svs_semseg(model, views, pred_key=None, svs=None, return_pred=False,
                    nb_cpus=1, verbose=False):
    """
    Predicts views of a list of SVs and saves them via SV.save_views.
    Efficient helper function for chunked predictions,
    therefore requires pre-loaded views.

    Parameters
    ----------
    model :
    views : List[np.array]
        N_SV each with np.array of shape [N_LOCS, N_CH, N_VIEWS, X, Y]
         as uint8 scaled from 0 to 255
    pred_key : str
    nb_cpus : int
        number CPUs for saving the SV views
        svs : list[SegmentationObject]
    svs : Optional[list[SegmentationObject]]
    return_pred : Optional[bool]
    verbose : bool

    Returns
    -------
    list[np.array]
        if 'return_pred=True' it returns the label views of input
    """
    if not return_pred and (svs is None or pred_key is None):
        raise ValueError('SV objects and "pred_key" have to be given if'
                         ' predictions should be saved at SV view storages.')
    part_views = np.cumsum([0] + [len(v) for v in views])
    assert len(part_views) == len(views) + 1
    views = np.concatenate(views)  # merge axis 0, i.e. N_SV and N_LOCS to N_SV*N_LOCS
    # views have shape: M, 4, 2, 128, 256
    label_views = predict_views_semseg(views, model, verbose=verbose)
    svs_labelviews = []
    for ii in range(len(part_views[:-1])):
        sv_label_views = label_views[part_views[ii]:part_views[ii + 1]]
        svs_labelviews.append(sv_label_views)
    assert len(part_views) == len(svs_labelviews) + 1
    if return_pred:
        return svs_labelviews
    params = [[sv, dict(views=views, index_views=False, woglia=True,
                        view_key=pred_key)]
              for sv, views in zip(svs, svs_labelviews)]
    start_multiprocess_obj('save_views', params, nb_cpus=nb_cpus)


def pred_sv_chunk_semseg(args):
    """

    Parameters
    ----------
    args :

    Returns
    -------

    """
    from syconn.proc.sd_proc import sos_dict_fact, init_sos
    from syconn.handler.prediction import InferenceModel
    from syconn.backend.storage import CompressedStorage
    so_chunk_paths = args[0]
    model_kwargs = args[1]
    so_kwargs = args[2]
    pred_kwargs = args[3]

    # By default use views after glia removal
    if 'woglia' in pred_kwargs:
        woglia = pred_kwargs["woglia"]
        del pred_kwargs["woglia"]
    else:
        woglia = True
    pred_key = pred_kwargs["pred_key"]
    if 'raw_only' in pred_kwargs:
        raw_only = pred_kwargs['raw_only']
        del pred_kwargs['raw_only']
    else:
        raw_only = False

    model = InferenceModel(**model_kwargs)
    for p in so_chunk_paths:
        # get raw views
        view_dc_p = p + "/views_woglia.pkl" if woglia else p + "/views.pkl"
        view_dc = CompressedStorage(view_dc_p, disable_locking=True)
        svixs = list(view_dc.keys())
        views = list(view_dc.values())
        if raw_only:
            views = views[:, :1]
        sd = sos_dict_fact(svixs, **so_kwargs)
        svs = init_sos(sd)
        label_views = pred_svs_semseg(model, views, svs, return_pred=True,
                                      verbose=True)
        # choose any SV to get a path constructor for the v
        # iew storage (is the same for all SVs of this chunk)
        lview_dc_p = svs[0].view_path(woglia, view_key=pred_key)
        label_vd = CompressedStorage(lview_dc_p, disable_locking=True)
        for ii in range(len(svs)):
            label_vd[svs[ii].id] = label_views[ii]
        label_vd.push()


def semseg2mesh(sso, semseg_key, nb_views=None, dest_path=None, k=1,
                colors=None, force_overwrite=False):
    """
    # TODO: optimize with cython
    Maps semantic segmentation to SSV mesh.

    Parameters
    ----------
    sso : SuperSegmentationObject
    semseg_key : str
    nb_views : int
    dest_path : Optional[str]
        Colored mesh will be written to k.zip and not returned.
    k : int
        Number of nearest vertices to average over. If k=0 unpredicted vertices
         will be treated as 'unpredicted' class.
    colors : Optional[Tuple[list]]
        np.array with as many entries as the maximum label of 'semseg_key'
        predictions with values between 0 and 255 (will be interpreted as uint8).
        If it is None, the majority label according to kNN will be returned
        instead. Note to add a color for unpredicted vertices if k==0; here
        illustrated with by the spine prediction example:
        if k=0: [neck, head, shaft, other, background, unpredicted]
        else: [neck, head, shaft, other, background]
    force_overwrite : bool

    Returns
    -------
    np.array, np.array, np.array, np.array
        indices, vertices, normals, color
    """
    ld = sso.label_dict('vertex')
    if force_overwrite or not semseg_key in ld:
        ts0 = time.time()  # view loading
        if nb_views is None:
            # load default
            i_views = sso.load_views(index_views=True).flatten()
        else:
            # load special views
            i_views = sso.load_views("index{}".format(nb_views)).flatten()
        spiness_views = sso.load_views(semseg_key).flatten()
        ts1 = time.time()
        log_reps.debug('Time to load index and shape views: '
                       '{:.2f}s.'.format(ts1 - ts0))
        ind = sso.mesh[0]
        dc = defaultdict(list)
        background_id = np.max(i_views)
        # color buffer holds traingle ID not vertex ID
        for ii in range(len(i_views)):
            triangle_ix = i_views[ii]
            if triangle_ix == background_id:
                continue
            l = spiness_views[ii]  # triangle label
            # get vertex ixs from triangle ixs via:
            vertex_ix = triangle_ix * 3
            dc[ind[vertex_ix]].append(l)
            dc[ind[vertex_ix + 1]].append(l)
            dc[ind[vertex_ix + 2]].append(l)
        ts2 = time.time()
        log_reps.debug('Time to generate look-up dict: '
                       '{:.2f}s.'.format(ts2 - ts1))
        # background label is highest label in prediction (see 'generate_palette' or
        # 'remap_rgb_labelviews' in multiviews.py)
        background_l = np.max(spiness_views)
        unpredicted_l = background_l + 1
        if unpredicted_l > 255:
            raise ValueError('Overflow in label view array.')
        vertex_labels = np.ones((len(sso.mesh[1]) // 3), dtype=np.uint8) * unpredicted_l
        for ix, v in dc.items():
            l, cnts = np.unique(v, return_counts=True)
            vertex_labels[ix] = l[np.argmax(cnts)]
        if k == 0:  # map actual prediction situation / coverage
            # keep unpredicted vertices and vertices with background labels
            predicted_vertices = sso.mesh[1].reshape(-1, 3)
            predictions = vertex_labels
        else:
            # remove unpredicted vertices
            predicted_vertices = sso.mesh[1].reshape(-1, 3)[vertex_labels != unpredicted_l]
            predictions = vertex_labels[vertex_labels != unpredicted_l]
            # remove background class
            predicted_vertices = predicted_vertices[predictions != background_id]
            predictions = predictions[predictions != background_id]
        ts3 = time.time()
        log_reps.debug('Time to map predictions on vertices: '
                       '{:.2f}s.'.format(ts3 - ts2))
        if k > 0:  # map predictions of predicted vertices to all vertices
            maj_vote = colorcode_vertices(
                sso.mesh[1].reshape((-1, 3)), predicted_vertices, predictions, k=k,
                return_color=False, nb_cpus=sso.nb_cpus)
        else:  # no vertex mask was applied in this case
            maj_vote = predictions
        ts4 = time.time()
        log_reps.debug('Time to map predictions on unpredicted vertices: '
                       '{:.2f}s.'.format(ts4 - ts3))
        # add prediction to mesh storage
        ld[semseg_key] = maj_vote
        ld.push()
    else:
        maj_vote = ld[semseg_key]
    if colors is not None:
        col = colors[maj_vote].astype(np.uint8)
        if np.sum(col) == 0:
            log_reps.warn('All colors-zero warning during "semseg2mesh"'
                          ' of SSO {}. Make sure color values have uint8 range '
                          '0...255'.format(sso.id))
    else:
        col = maj_vote
    if dest_path is not None:
        if colors is None:
            col = None  # set to None, because write_mesh2kzip only supports
            # RGBA colors and no labels
        write_mesh2kzip(dest_path, sso.mesh[0], sso.mesh[1], sso.mesh[2],
                        col, ply_fname=semseg_key + ".ply")
        return
    return sso.mesh[0], sso.mesh[1], sso.mesh[2], col

