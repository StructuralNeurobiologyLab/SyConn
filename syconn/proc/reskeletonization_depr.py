# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Philipp Schubert, Joergen Kornfeld

# TODO: outsource all skeletonization code, currently not used and methods are spread all over code base
try:
    import cPickle as pkl
except ImportError:
    import pickle as pkl
import numpy as np
import os
import itertools
import networkx as nx
from scipy import spatial
import scipy
import skeletopyze


def reskeletonize_plain(volume, coord_scaling=(2, 2, 1), node_offset=0):
    params = skeletopyze.Parameters()
    params.min_segment_length_ratio = .01
    params.min_segment_length = 10
    params.max_num_segments = 1000
    params.skip_explained_nodes = True
    params.explanation_weight = .1
    # params.boundary_weight = 1

    skeletopyze.setLogLevel(skeletopyze.LogLevel.All)
    skel = skeletopyze.get_skeleton_graph(volume, params)

    nodes = []
    edges = []
    diameters = []

    for n in skel.nodes():
        nodes.append(np.array([skel.locations(n).z() * coord_scaling[0],
                               skel.locations(n).y() * coord_scaling[1],
                               skel.locations(n).x() * coord_scaling[2]],
                              dtype=np.int))

        diameters.append(skel.diameters(n) * np.mean(coord_scaling))

    for e in skel.edges():
        edges.append(np.array([e.u + node_offset,
                               e.v + node_offset], dtype=np.int))

    return nodes, edges, diameters


def cleanup_skeleton(skeleton, scaling):
    scaling = np.array(scaling)

    edge_lengths = []
    for e in skeleton["edges"]:
        edge_lengths.append(np.linalg.norm(skeleton["nodes"][e[1]] - skeleton["nodes"][e[0]]))

    edge_lengths = np.array(edge_lengths)
    log_proc.info(np.mean(edge_lengths), np.std(edge_lengths))

    edges = skeleton["edges"][edge_lengths < np.median(edge_lengths) + np.std(edge_lengths)].tolist()

    graph = nx.from_edgelist(edges)
    cc = nx.connected_components(graph)

    i_cc = 0
    block_ids = np.ones(len(skeleton["nodes"]), dtype=np.int) * -1
    for this_cc in cc:
        block_ids[list(this_cc)] = i_cc
        i_cc += 1
    block_ids[block_ids < 0] = np.max(block_ids) + \
                               np.arange(np.sum([block_ids < 0]))

    print("N unique blocks:", len(np.unique(block_ids)))

    # node_ids, node_degrees = np.unique(edges, return_counts=True)
    # end_nodes = node_ids[node_degrees <= 1].tolist()

    node_ids, node_count = np.unique(edges, return_counts=True)

    node_degrees = np.zeros(len(skeleton["nodes"]), dtype=np.int)
    node_degrees[node_ids] = node_count

    end_nodes = np.argwhere(node_degrees <= 1).squeeze()

    distances = spatial.distance.cdist(skeleton["nodes"] * scaling,
                                             skeleton["nodes"] * scaling)

    max_dist = np.max(distances)
    distances[np.triu_indices_from(distances)] = max_dist + 1

    min_locs = np.dstack(np.unravel_index(np.argsort(distances.ravel()),
                                          distances.shape))[0]
    for min_loc in min_locs:
        if len(np.unique(block_ids)) == 1:
            break

        if block_ids[min_loc[0]] != block_ids[min_loc[1]]:
            edges.append(np.array([min_loc[0], min_loc[1]], dtype=np.int))
            block_ids[block_ids == block_ids[min_loc[0]]] = block_ids[min_loc[1]]



    # kdtree = scipy.spatial.cKDTree(skeleton["nodes"] * scaling)
    # max_dist = 0
    # while len(np.unique(block_ids)) > 1:
    #     max_dist += 5
    #     print "N unique blocks:", len(np.unique(block_ids))
    #
    #     node_ids, node_count = np.unique(edges, return_counts=True)
    #
    #     node_degrees = np.zeros(len(skeleton["nodes"]), dtype=np.int)
    #     node_degrees[node_ids] = node_count
    #
    #     end_nodes = np.argwhere(node_degrees <= 1).squeeze()
    #
    #     # kdtree = scipy.spatial.cKDTree(skeleton["nodes"][end_nodes] * scaling)
    #     for end_node in end_nodes:
    #         coord = skeleton["nodes"][end_node] * scaling
    #         ns = np.array(kdtree.query_ball_point(coord, max_dist),
    #                       dtype=np.int)
    #         ns = ns[block_ids[ns] != block_ids[end_node]]
    #         if len(ns) > 0:
    #             partner_id = ns[np.argmin(np.linalg.norm(
    #                 skeleton["nodes"][ns] - coord, axis=1))]
    #             edges.append(np.array([partner_id, end_node], dtype=np.int))
    #
    #             graph = nx.from_edgelist(edges)
    #             cc = nx.connected_components(graph)
    #
    #             i_cc = 0
    #             block_ids = np.ones(len(skeleton["nodes"]), dtype=np.int) * -1
    #             for this_cc in cc:
    #                 block_ids[list(this_cc)] = i_cc
    #                 i_cc += 1
    #             block_ids[block_ids < 0] = np.max(block_ids) +\
    #                                        np.arange(np.sum([block_ids < 0]))

    skeleton["edges"] = np.array(edges)
    return skeleton


def reskeletonize_block(block, coord_scaling=(2, 2, 1), node_offset=0,
                        block_offset=0):
    cc_block, n_cc = scipy.ndimage.label(block)

    nodes = []
    edges = []
    diameters = []
    block_ids = []

    for i_cc in range(1, n_cc+1):
        this_nodes, this_edges, this_diameters = \
            reskeletonize_plain(cc_block == i_cc, coord_scaling=coord_scaling,
                                node_offset=node_offset)

        nodes += this_nodes
        edges += this_edges
        diameters += this_diameters
        block_ids += [block_offset] * len(this_nodes)

        node_offset += len(this_nodes)
        block_offset += 1

    return nodes, edges, diameters, block_ids


def reskeltetonize_super_chunk(volume, chunk_size, overlap, coord_scaling):
    g_nodes = []
    g_edges = []
    g_diameters = []
    g_crossing_edge_nodes = []

    sc_chunk_size = chunk_size / coord_scaling
    sc_overlap = overlap / coord_scaling

    volume_shape = np.array(volume.shape)
    n_steps = np.ceil((volume_shape - 2 * sc_overlap) / sc_chunk_size.astype(np.float)).astype(np.int)
    steps = itertools.product(*[range(n_steps[i]) for i in range(3)])

    for step in steps:
        step = np.array(step, dtype=np.int)

        start = sc_chunk_size * step
        end = sc_overlap * 2 + sc_chunk_size * (step + 1)

        this_offset = start * coord_scaling - overlap

        this_volume = volume[start[0]: end[0],
                             start[1]: end[1],
                             start[2]: end[2]]

        v_sh = this_volume.shape
        if np.sum(this_volume[
                  sc_overlap[0]: v_sh[0]-sc_overlap[0],
                  sc_overlap[1]: v_sh[1]-sc_overlap[1],
                  sc_overlap[2]: v_sh[2]-sc_overlap[2]]) == 0:
            continue

        reskel = reskeletonize_block(this_volume,
                                     coord_scaling=coord_scaling)

        nodes = np.array(reskel[0])
        edges = np.array(reskel[1])
        diameters = np.array(reskel[2])

        if len(nodes) == 0 or len(edges) == 0:
            continue

        mask = np.all((nodes-overlap) >= 0, axis=1)

        new_ids = np.ones(len(nodes), dtype=np.int) * (-1)
        new_ids[mask] = np.array(range(int(np.sum(mask))), dtype=np.int)

        nodes = nodes[mask]

        if len(nodes) == 0:
            continue

        edges = new_ids[edges]
        diameters = diameters[mask]
        # block_ids = block_ids[mask]

        mask = np.all((nodes-overlap-chunk_size) < 0, axis=1)

        new_ids = np.ones(len(nodes), dtype=np.int) * (-1)
        new_ids[mask] = np.array(range(int(np.sum(mask))), dtype=np.int)
        new_ids = np.array(new_ids.tolist() + [-1])

        nodes = nodes[mask]

        if len(nodes) == 0:
            continue

        edges = new_ids[edges]

        crossing_edge_nodes = edges[np.any(edges == -1, axis=1)]
        crossing_edge_nodes = np.max(crossing_edge_nodes[~np.all(
            crossing_edge_nodes == -1, axis=1)], axis=1)

        edges = edges[np.all(edges >= 0, axis=1)]
        diameters = diameters[mask]

        g_crossing_edge_nodes += list(crossing_edge_nodes + len(g_nodes))

        nodes += this_offset
        g_edges += list(edges + len(g_nodes))
        g_nodes += nodes.tolist()
        g_diameters += list(diameters)

    return g_nodes, g_edges, g_diameters, g_crossing_edge_nodes


def reskeletonize_chunked(obj_id, volume_shape, volume_offset, scaling,
                          kd=None, voxels=None, coord_scaling=(2, 2, 1),
                          chunk_step_stride=4, max_con_dist_nm=100,
                          nb_threads=1):
    def _reskel_thread(args):
        chunk_id = args[0]

        this_offset = np.array([chunk_id[i_dim] * chunk_size[i_dim]
                                for i_dim in range(3)])

        if voxels is None and kd is not None:
            this_offset -= overlap
            chunk_volume = kd.from_overlaycubes_to_matrix(size=this_size,
                                                          offset=this_offset +
                                                                 volume_offset,
                                                          datatype=np.uint32,
                                                          mirror_oob=True,
                                                          nb_threads=4,
                                                          show_progress=False)

            chunk_volume = chunk_volume[::coord_scaling[0],
                           ::coord_scaling[1],
                           ::coord_scaling[2]]
            chunk_volume = chunk_volume == obj_id
        else:
            start = this_offset / coord_scaling
            end = start + chunk_step_stride * sc_chunk_size + 2 * sc_overlap

            chunk_volume = voxels[start[0]: end[0],
                           start[1]: end[1],
                           start[2]: end[2]]
        cv_sh = chunk_volume.shape
        sum_volume = np.sum(
            chunk_volume[sc_overlap[0]: cv_sh[0] - sc_overlap[0],
            sc_overlap[1]: cv_sh[1] - sc_overlap[1],
            sc_overlap[2]: cv_sh[2] - sc_overlap[2]])

        if sum_volume == 0:
            print("Discarded")
            return [[]], this_offset
        else:
            reskel = reskeltetonize_super_chunk(chunk_volume,
                                                chunk_size,
                                                overlap,
                                                coord_scaling)
            return reskel, this_offset

    scaling = np.array(scaling)
    coord_scaling = np.array(coord_scaling)
    volume_offset = np.array(volume_offset)

    chunk_size = np.array([160, 160, 80], dtype=np.int)
    n_chunks = np.ceil(volume_shape / chunk_size.astype(np.float)).astype(np.int)
    chunk_ids = itertools.product(*[range(0, n_chunks[i], chunk_step_stride) for i in range(3)])

    multi_params = []
    for chunk_id in chunk_ids:
        multi_params.append([chunk_id])

    # overlap = chunk_size / 2
    overlap = np.array([120, 120, 60], dtype=np.int)

    g_nodes = []
    g_edges = []
    g_diameters = []
    g_crossing_edge_nodes = []

    this_size = chunk_size * chunk_step_stride + overlap * 2

    sc_overlap = overlap / coord_scaling
    sc_chunk_size = chunk_size / coord_scaling
    if voxels is not None:
        voxels = np.pad(voxels, zip(sc_overlap, sc_overlap), mode="constant",
                        constant_values=False)

    if nb_threads > 1:
        pool = ThreadPool(nb_threads)
        results = pool.map(_reskel_thread, multi_params)
        pool.close()
        pool.join()
    else:
        results = map(_reskel_thread, multi_params)

    for result in results:
        reskel, this_offset = result
        if len(reskel[0]) > 0:
            edges = np.array(reskel[1]) + len(g_nodes)
            g_edges += edges.tolist()
            nodes = np.array(reskel[0]) + this_offset + volume_offset
            g_nodes += nodes.tolist()
            g_diameters += reskel[2]
            g_crossing_edge_nodes += reskel[3]



    # def reskeletonize_chunked(obj_id, volume_shape, volume_offset, scaling,
#                           kd=None, voxels=None, coord_scaling=(2, 2, 1),
#                           chunk_step_stride=4, max_con_dist_nm=100):
#     scaling = np.array(scaling)
#     coord_scaling = np.array(coord_scaling)
#     volume_offset = np.array(volume_offset)
#
#     chunk_size = np.array([160, 160, 80], dtype=np.int)
#     n_chunks = np.ceil(volume_shape / chunk_size.astype(np.float)).astype(np.int)
#     chunk_ids = itertools.product(*[range(0, n_chunks[i], chunk_step_stride) for i in range(3)])
#
#     # overlap = chunk_size / 2
#     overlap = np.array([120, 120, 60], dtype=np.int)
#
#     g_nodes = []
#     g_edges = []
#     g_diameters = []
#     g_crossing_edge_nodes = []
#
#     this_size = chunk_size * chunk_step_stride + overlap * 2
#
#     sc_overlap = overlap / coord_scaling
#     sc_chunk_size = chunk_size / coord_scaling
#     if voxels is not None:
#         voxels = np.pad(voxels, zip(sc_overlap, sc_overlap), mode="constant",
#                         constant_values=False)
#
#     for chunk_id in chunk_ids:
#         print chunk_id, "of", n_chunks - 1
#         this_offset = np.array([chunk_id[i_dim] * chunk_size[i_dim]
#                                 for i_dim in range(3)])
#
#         if voxels is None and kd is not None:
#             this_offset -= overlap
#             chunk_volume = kd.from_overlaycubes_to_matrix(size=this_size,
#                                                           offset=this_offset +
#                                                                  volume_offset,
#                                                           datatype=np.uint32,
#                                                           mirror_oob=True,
#                                                           nb_threads=4,
#                                                           show_progress=False)
#
#             chunk_volume = chunk_volume[::coord_scaling[0],
#                                         ::coord_scaling[1],
#                                         ::coord_scaling[2]]
#             chunk_volume = chunk_volume == obj_id
#         else:
#             start = this_offset / coord_scaling
#             end = start + chunk_step_stride * sc_chunk_size + 2 * sc_overlap
#
#             chunk_volume = voxels[start[0]: end[0],
#                                   start[1]: end[1],
#                                   start[2]: end[2]]
#         cv_sh = chunk_volume.shape
#         sum_volume = np.sum(chunk_volume[sc_overlap[0]: cv_sh[0]-sc_overlap[0],
#                                          sc_overlap[1]: cv_sh[1]-sc_overlap[1],
#                                          sc_overlap[2]: cv_sh[2]-sc_overlap[2]])
#
#         if sum_volume == 0:
#             print "Discarded"
#             continue
#         else:
#             print "Keep", sum_volume
#
#         reskel = reskeltetonize_super_chunk(chunk_volume,
#                                             chunk_size,
#                                             overlap,
#                                             coord_scaling)
#
#         if len(reskel[0]) > 0:
#             edges = np.array(reskel[1]) + len(g_nodes)
#             g_edges += edges.tolist()
#             nodes = np.array(reskel[0]) + this_offset + volume_offset
#             g_nodes += nodes.tolist()
#             g_diameters += reskel[2]
#             g_crossing_edge_nodes += reskel[3]

    graph = nx.from_edgelist(g_edges)
    cc = nx.connected_components(graph)

    i_cc = 0
    g_block_ids = np.zeros(len(g_nodes), dtype=np.int)
    for this_cc in cc:
        g_block_ids[list(this_cc)] = i_cc
        i_cc += 1

    g_nodes = np.array(g_nodes)
    g_diameters = np.array(g_diameters)

    print("N unique blocks:", len(np.unique(g_block_ids)))
    print("N candidate nodes:", len(g_crossing_edge_nodes))

    # PHASE 1 - candidate nodes:

    # if len(g_crossing_edge_nodes) > 0:
    #     g_crossing_edge_nodes = np.array(g_crossing_edge_nodes, dtype=np.int)
    #     g_block_ids = np.array(g_block_ids, dtype=np.int)
    #
    #     candidate_nodes = g_nodes[g_crossing_edge_nodes]
    #     candidate_block_ids = g_block_ids[g_crossing_edge_nodes]
    #
    #     distances = scipy.spatial.distance.cdist(candidate_nodes * scaling,
    #                                              candidate_nodes * scaling)
    #
    #     distances[np.triu_indices_from(distances)] = max_con_dist_nm + 1
    #     distances[distances > max_con_dist_nm] = max_con_dist_nm + 1
    #
    #     new_edges = set()
    #
    #     min_locs = np.dstack(np.unravel_index(np.argsort(distances.ravel()),
    #                                           distances.shape))[0]
    #     next_loc = 0
    #     max_loc = np.sum(range(len(candidate_nodes)))
    #     while len(np.unique(candidate_block_ids)) > 1 and next_loc < max_loc:
    #         n1, n2 = min_locs[next_loc]
    #         if distances[n1, n2] > max_con_dist_nm:
    #             break
    #
    #         if candidate_block_ids[n1] != candidate_block_ids[n2]:
    #             print "unique", np.unique(candidate_block_ids), "edges", \
    #                 len(new_edges)
    #
    #             new_edges.add(tuple([n1, n2]))
    #
    #             g_block_ids[g_block_ids == candidate_block_ids[n1]] = \
    #                 candidate_block_ids[n2]
    #
    #             candidate_block_ids[candidate_block_ids ==
    #                                 candidate_block_ids[n1]] =\
    #                 candidate_block_ids[n2]
    #
    #         next_loc += 1
    #
    #     if len(np.unique(candidate_block_ids)) > 1:
    #         print "Failed", len(np.unique(candidate_block_ids)),\
    #             len(np.unique(g_block_ids))
    #
    #     for edge in new_edges:
    #         g_edges.append(np.array([g_crossing_edge_nodes[edge[0]],
    #                                  g_crossing_edge_nodes[edge[1]]],
    #                                 dtype=np.int))

    # PHASE 2 - end nodes:

    if len(g_edges) == 0:
        return g_nodes, g_edges, g_diameters

    max_dist_step = np.linalg.norm(overlap * scaling) / 4

    node_ids, node_degrees_ex = np.unique(g_edges, return_counts=True)
    print("Remove lonely nodes")
    node_degrees = np.zeros(len(g_nodes), dtype=np.int)
    node_degrees[node_ids] = node_degrees_ex

    mask = node_degrees > 0
    new_ids = np.ones(len(g_nodes), dtype=np.int)
    new_ids[mask] = np.array(range(int(np.sum(mask))), dtype=np.int)
    new_ids = np.array(new_ids.tolist())

    g_nodes = g_nodes[mask]
    g_diameters = g_diameters[mask]
    g_block_ids = g_block_ids[mask]
    g_edges = new_ids[np.array(g_edges)].tolist()

    node_ids, node_degrees = np.unique(g_edges, return_counts=True)
    end_nodes = node_ids[node_degrees == 1]

    kdtree = spatial.cKDTree(g_nodes * scaling)
    distances = spatial.distance.cdist(g_nodes[end_nodes] * scaling,
                                             g_nodes[end_nodes] * scaling)

    distances[np.triu_indices_from(distances)] = max_dist_step + 1
    distances[distances > max_dist_step] = max_dist_step + 1

    min_locs = np.dstack(np.unravel_index(np.argsort(distances.ravel()),
                                          distances.shape))[0]
    for min_loc in min_locs:
        if distances[min_loc[0], min_loc[1]] > max_dist_step:
            break

        if g_block_ids[end_nodes[min_loc[0]]] != \
                g_block_ids[end_nodes[min_loc[1]]]:
            g_edges.append(np.array([end_nodes[min_loc[0]],
                                     end_nodes[min_loc[1]]], dtype=np.int))

            g_block_ids[g_block_ids == g_block_ids[end_nodes[min_loc[0]]]] = \
                g_block_ids[end_nodes[min_loc[1]]]

    # PHASE 3 - rest end nodes:

    max_dist = 0
    while len(np.unique(g_block_ids)) > 1:
        max_dist += max_dist_step
        print("N unique blocks:", len(np.unique(g_block_ids)))

        node_ids, node_degrees = np.unique(g_edges, return_counts=True)
        end_nodes = node_ids[node_degrees <= 1].tolist()

        for end_node in end_nodes:
            coord = g_nodes[end_node] * scaling
            ns = np.array(kdtree.query_ball_point(coord, max_dist),
                          dtype=np.int)
            ns = ns[g_block_ids[ns] != g_block_ids[end_node]]
            if len(ns) > 0:
                partner_id = ns[np.argmin(np.linalg.norm(g_nodes[ns] - coord,
                                                         axis=1))]
                g_edges.append(np.array([partner_id, end_node], dtype=np.int))

                graph = nx.from_edgelist(g_edges)
                cc = nx.connected_components(graph)

                i_cc = 0
                g_block_ids = np.zeros(len(g_nodes), dtype=np.int)
                for this_cc in cc:
                    g_block_ids[list(this_cc)] = i_cc
                    i_cc += 1

    print("N unique blocks:", len(np.unique(g_block_ids)))

    g_nodes = g_nodes.tolist()
    return g_nodes, g_edges, g_diameters