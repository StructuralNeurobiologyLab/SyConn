from collections import Counter
import itertools
from multiprocessing.pool import ThreadPool
import networkx as nx
import numpy as np
import scipy.ndimage
import scipy.spatial
from knossos_utils.skeleton_utils import annotation_to_nx_graph
import os
import copy
try:
    import skeletopyze
    skeletopyze_available = True
except:
    print "skeletopyze not found - you won't be able to compute skeletons. " \
              "Install skeletopyze from https://github.com/funkey/skeletopyze"

import super_segmentation as ss
import segmentation
from knossos_utils import knossosdataset
from scipy import spatial


def write_super_segmentation_dataset_thread(args):
    ssv_obj_ids = args[0]
    version = args[1]
    version_dict = args[2]
    working_dir = args[3]
    extract_only = args[4]
    attr_keys = args[5]

    ssd = ss.SuperSegmentationDataset(working_dir, version, version_dict)

    try:
        ssd.load_mapping_dict()
        mapping_dict_avail = True
    except:
        mapping_dict_avail = False

    attr_dict = dict(id=[])

    for ssv_obj_id in ssv_obj_ids:
        print ssv_obj_id
        ssv_obj = ssd.get_super_segmentation_object(ssv_obj_id,
                                                    new_mapping=True,
                                                    create=True)

        if ssv_obj.attr_dict_exists:
            ssv_obj.load_attr_dict()

        if not extract_only:

            if len(ssv_obj.attr_dict["sv"]) == 0:
                if mapping_dict_avail:
                    ssv_obj = ssd.get_super_segmentation_object(ssv_obj_id, True)

                    if ssv_obj.attr_dict_exists:
                        ssv_obj.load_attr_dict()
                else:
                    raise Exception("No mapping information found")
        if not extract_only:
            if "rep_coord" not in ssv_obj.attr_dict:
                ssv_obj.attr_dict["rep_coord"] = ssv_obj.rep_coord
            if "bounding_box" not in ssv_obj.attr_dict:
                ssv_obj.attr_dict["bounding_box"] = ssv_obj.bounding_box
            if "size" not in ssv_obj.attr_dict:
                ssv_obj.attr_dict["size"] = ssv_obj.size

        ssv_obj.attr_dict["sv"] = np.array(ssv_obj.attr_dict["sv"],
                                           dtype=np.int)

        if extract_only:
            ignore = False
            for attribute in attr_keys:
                if not attribute in ssv_obj.attr_dict:
                    ignore = True
                    break
            if ignore:
                continue

            attr_dict["id"].append(ssv_obj_id)

            for attribute in attr_keys:
                if attribute not in attr_dict:
                    attr_dict[attribute] = []

                if attribute in ssv_obj.attr_dict:
                    attr_dict[attribute].append(ssv_obj.attr_dict[attribute])
                else:
                    attr_dict[attribute].append(None)
        else:
            attr_dict["id"].append(ssv_obj_id)
            for attribute in ssv_obj.attr_dict.keys():
                if attribute not in attr_dict:
                    attr_dict[attribute] = []

                attr_dict[attribute].append(ssv_obj.attr_dict[attribute])

                ssv_obj.save_attr_dict()

    return attr_dict


def export_to_knossosdataset_thread(args):
    ssv_obj_ids = args[0]
    version = args[1]
    version_dict = args[2]
    working_dir = args[3]
    kd_path = args[4]
    nb_threads = args[5]

    kd = knossosdataset.KnossosDataset().initialize_from_knossos_path(kd_path)

    ssd = ss.SuperSegmentationDataset(working_dir, version, version_dict)
    ssd.load_mapping_dict()

    for ssv_obj_id in ssv_obj_ids:
        print ssv_obj_id

        ssv_obj = ssd.get_super_segmentation_object(ssv_obj_id, True)

        offset = ssv_obj.bounding_box[0]
        if not 0 in offset:
            kd.from_matrix_to_cubes(ssv_obj.bounding_booffset,
                                    data=ssv_obj.voxels.astype(np.uint32) *
                                         ssv_obj_id,
                                    overwrite=False,
                                    nb_threads=nb_threads)


def convert_knossosdataset_thread(args):
    version = args[0]
    version_dict = args[1]
    working_dir = args[2]
    nb_threads = args[3]
    sv_kd_path = args[4]
    ssv_kd_path = args[5]
    offsets = args[6]
    size = args[7]

    sv_kd = knossosdataset.KnossosDataset()
    sv_kd.initialize_from_knossos_path(sv_kd_path)
    ssv_kd = knossosdataset.KnossosDataset()
    ssv_kd.initialize_from_knossos_path(ssv_kd_path)

    ssd = ss.SuperSegmentationDataset(working_dir, version, version_dict)
    ssd.load_id_changer()

    for offset in offsets:
        block = sv_kd.from_overlaycubes_to_matrix(size, offset,
                                                  datatype=np.uint32,
                                                  nb_threads=nb_threads)

        block = ssd.id_changer[block]

        ssv_kd.from_matrix_to_cubes(offset,
                                    data=block.astype(np.uint32),
                                    datatype=np.uint32,
                                    overwrite=False,
                                    nb_threads=nb_threads)

        raw = sv_kd.from_raw_cubes_to_matrix(size, offset,
                                             nb_threads=nb_threads)

        ssv_kd.from_matrix_to_cubes(offset,
                                    data=raw,
                                    datatype=np.uint8,
                                    as_raw=True,
                                    overwrite=False,
                                    nb_threads=nb_threads)


def aggregate_segmentation_object_mappings_thread(args):
    ssv_obj_ids = args[0]
    version = args[1]
    version_dict = args[2]
    working_dir = args[3]
    obj_types = args[4]

    ssd = ss.SuperSegmentationDataset(working_dir, version, version_dict)
    ssd.load_mapping_dict()

    for ssv_id in ssv_obj_ids:
        ssv = ssd.get_super_segmentation_object(ssv_id, True)
        mappings = dict((obj_type, Counter()) for obj_type in obj_types)

        for sv in ssv.svs:
            sv.load_attr_dict()
            for obj_type in obj_types:
                if "mapping_%s_ids" % obj_type in sv.attr_dict:
                    keys = sv.attr_dict["mapping_%s_ids" % obj_type]
                    values = sv.attr_dict["mapping_%s_ratios" % obj_type]
                    mappings[obj_type] += Counter(dict(zip(keys, values)))

        ssv.load_attr_dict()
        for obj_type in obj_types:
            if obj_type in mappings:
                ssv.attr_dict["mapping_%s_ids" % obj_type] = \
                    mappings[obj_type].keys()
                ssv.attr_dict["mapping_%s_ratios" % obj_type] = \
                    mappings[obj_type].values()

        ssv.save_attr_dict()


def apply_mapping_decisions_thread(args):
    ssv_obj_ids = args[0]
    version = args[1]
    version_dict = args[2]
    working_dir = args[3]
    obj_types = args[4]

    ssd = ss.SuperSegmentationDataset(working_dir, version, version_dict)
    ssd.load_mapping_dict()

    for ssv_id in ssv_obj_ids:
        ssv = ssd.get_super_segmentation_object(ssv_id, True)
        for obj_type in obj_types:
            if obj_type == "sj":
                correct_for_background = True
            else:
                correct_for_background = False

            ssv.apply_mapping_decision(obj_type,
                                       correct_for_background=correct_for_background,
                                       save=True)


def reskeletonize_objects_small_ones_thread(args):
    ssv_obj_ids = args[0]
    version = args[1]
    version_dict = args[2]
    working_dir = args[3]

    ssd = ss.SuperSegmentationDataset(working_dir, version, version_dict)
    ssd.load_mapping_dict()

    for ssv_id in ssv_obj_ids:
        print "------------", ssv_id
        ssv = ssd.get_super_segmentation_object(ssv_id, True)
        if np.product(ssv.shape) > 1e10:
            continue
        # elif np.product(ssv.shape) > 10**3:
        #     ssv.calculate_skeleton(coord_scaling=(8, 8, 4))
        elif ssv.size > 0:
            ssv.calculate_skeleton(coord_scaling=(8, 8, 4), plain=True)
        else:
            ssv.skeleton = {"nodes": [], "edges": [], "diameters": []}
        ssv.save_skeleton()
        ssv.clear_cache()


def reskeletonize_objects_big_ones_thread(args):
    ssv_obj_ids = args[0]
    version = args[1]
    version_dict = args[2]
    working_dir = args[3]

    ssd = ss.SuperSegmentationDataset(working_dir, version, version_dict)
    ssd.load_mapping_dict()

    for ssv_id in ssv_obj_ids:
        ssv = ssd.get_super_segmentation_object(ssv_id, True)
        if np.product(ssv.shape) > 1e10:
            ssv.calculate_skeleton(coord_scaling=(10, 10, 5), plain=True)
        else:
            continue
        ssv.save_skeleton()
        ssv.clear_cache()


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
    print np.mean(edge_lengths), np.std(edge_lengths)

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

    print "N unique blocks:", len(np.unique(block_ids))

    # node_ids, node_degrees = np.unique(edges, return_counts=True)
    # end_nodes = node_ids[node_degrees <= 1].tolist()

    node_ids, node_count = np.unique(edges, return_counts=True)

    node_degrees = np.zeros(len(skeleton["nodes"]), dtype=np.int)
    node_degrees[node_ids] = node_count

    end_nodes = np.argwhere(node_degrees <= 1).squeeze()

    distances = scipy.spatial.distance.cdist(skeleton["nodes"] * scaling,
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
    assert skeletopyze_available

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
            print "Discarded"
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

    print "N unique blocks:", len(np.unique(g_block_ids))
    print "N candidate nodes:", len(g_crossing_edge_nodes)

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
    print "Remove lonely nodes"
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

    kdtree = scipy.spatial.cKDTree(g_nodes * scaling)
    distances = scipy.spatial.distance.cdist(g_nodes[end_nodes] * scaling,
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
        print "N unique blocks:", len(np.unique(g_block_ids))

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

    print "N unique blocks:", len(np.unique(g_block_ids))

    g_nodes = g_nodes.tolist()
    return g_nodes, g_edges, g_diameters


def export_skeletons(args):
    ssv_obj_ids = args[0]
    version = args[1]
    version_dict = args[2]
    working_dir = args[3]
    obj_types = args[4]
    apply_mapping = args[5]

    ssd = ss.SuperSegmentationDataset(working_dir, version, version_dict)
    ssd.load_mapping_dict()

    no_skel_cnt = 0
    for ssv_id in ssv_obj_ids:
        ssv = ssd.get_super_segmentation_object(ssv_id)

        try:
            ssv.load_skeleton()
            skeleton_avail = True
        except:
            skeleton_avail = False
            no_skel_cnt += 1

        if not skeleton_avail:
            continue

        if ssv.size == 0:
            continue

        if len(ssv.skeleton["nodes"]) == 0:
            continue

        try:
            ssv.save_skeleton_to_kzip()

            for obj_type in obj_types:
                if apply_mapping:
                    if obj_type == "sj":
                        correct_for_background = True
                    else:
                        correct_for_background = False
                    ssv.apply_mapping_decision(obj_type,
                                               correct_for_background=correct_for_background)

            ssv.save_objects_to_kzip_sparse(obj_types)

        except:
            pass

    return no_skel_cnt


def associate_objs_with_skel_nodes_thread(args):
    ssv_obj_ids = args[0]
    version = args[1]
    version_dict = args[2]
    working_dir = args[3]
    obj_types = args[4]

    ssd = ss.SuperSegmentationDataset(working_dir, version, version_dict)

    for ssv_id in ssv_obj_ids:
        ssv = ssd.get_super_segmentation_object(ssv_id)
        ssv.load_skeleton()
        if len(ssv.skeleton["nodes"]) > 0:
            ssv.associate_objs_with_skel_nodes(obj_types)


def predict_axoness_thread(args):
    ssv_obj_ids = args[0]
    version = args[1]
    version_dict = args[2]
    working_dir = args[3]

    ssd = ss.SuperSegmentationDataset(working_dir, version, version_dict)

    for ssv_id in ssv_obj_ids:
        ssv = ssd.get_super_segmentation_object(ssv_id)

        if not ssv.load_skeleton():
            continue

        ssv.load_attr_dict()
        if "assoc_sj" in ssv.attr_dict:
            ssv.predict_axoness(feature_context_nm=5000, clf_name="rfc")
        elif len(ssv.skeleton["nodes"]) > 0:
            try:
                ssv.associate_objs_with_skel_nodes(("sj", "mi", "vc"))
                ssv.predict_axoness(feature_context_nm=5000, clf_name="rfc")
            except:
                pass


def predict_cell_type_thread(args):
    ssv_obj_ids = args[0]
    version = args[1]
    version_dict = args[2]
    working_dir = args[3]

    ssd = ss.SuperSegmentationDataset(working_dir, version, version_dict)

    for ssv_id in ssv_obj_ids:
        ssv = ssd.get_super_segmentation_object(ssv_id)

        if not ssv.load_skeleton():
            continue

        ssv.load_attr_dict()
        if "assoc_sj" in ssv.attr_dict:
            ssv.predict_cell_type(feature_context_nm=25000, clf_name="rfc")
        elif len(ssv.skeleton["nodes"]) > 0:
            try:
                ssv.associate_objs_with_skel_nodes(("sj", "mi", "vc"))
                ssv.predict_cell_type(feature_context_nm=25000, clf_name="rfc")
            except:
                pass


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
        print "Prediciton already exists (SSV %d)." % sso.id
        return
    out_d = sso_views_to_modelinput(sso, nb_views)
    res = model.predict_proba(out_d)
    clf = np.argmax(res, axis=1)
    pred = np.argmax(np.array([np.sum(clf==0), np.sum(clf==1), np.sum(clf==2),
                     np.sum(clf == 3)]))
    sso.save_attributes(["celltype_cnn"], [pred])


def sso_views_to_modelinput(sso, nb_views):
    np.random.seed(0)
    assert len(sso.sv_ids) > 0
    views = np.concatenate(sso.load_views())
    np.random.shuffle(views)
    # view shape: (#multi-views, 4 channels, 2 perspectives, 128, 256)
    views = views.swapaxes(1, 0).reshape((4, -1, 128, 256))
    assert views.shape[1] > 0
    if views.shape[1] < nb_views:
        rand_ixs = np.random.choice(np.arange(views.shape[1]),
                                    nb_views - views.shape[1])
        views = np.append(views, views[:, rand_ixs], axis=1)
        print rand_ixs, views.shape
    nb_samples = np.floor(views.shape[1] / nb_views)
    assert nb_samples > 0
    out_d = views[:, :int(nb_samples * nb_views)]
    out_d = out_d.reshape((4, -1, nb_views, 128, 256)).swapaxes(1, 0)
    return out_d

# consumes an ssv_id
# returns a dict with the corrected diameters in the key 'diameters'
def radius_correction(args):

    ssv_id = args[0]
    skel_radius = {}


    sso = ssds.get_super_segmentation_object(ssv_id)
    sso.load_skeleton()
    sso_skel = sso.skeleton

    skel_node = sso_skel['nodes']
    diameters = sso_skel['diameters']

    sso_mesh = sso.mesh
    vert = sso_mesh[1].reshape((-1, 3))
    # vert_sparse = vert[0::10]
    vert_sparse = vert[:]
    tree = sp.cKDTree(skel_node * np.array([10, 10, 20]))
    centroid_arr = [[0, 0, 0]]

    # dists, ixs = tree.query(vert_sparse, 1)
    dists, ixs = tree.query(vert_sparse, 1)
    all_node_ixs = np.arange(len(skel_node))
    all_found_node_ixs = np.unique(ixs)
    found_coords = skel_node[all_found_node_ixs]
    all_skel_node_ixs = np.arange(len(skel_node))
    missing_coords_ixs = list(set(all_skel_node_ixs) - set(all_found_node_ixs))
    missing_coords = skel_node[missing_coords_ixs]

    for el in all_found_node_ixs:
        for i in np.where(ixs == el):
            vert = [[]]
            for a in i:
                vert.append(vert_sparse[a] / np.array([10, 10, 20]))
            x = [p[0] for p in vert[1:]]
            y = [p[1] for p in vert[1:]]
            z = [p[2] for p in vert[1:]]
            centroid = np.asarray((sum(x) / len(vert[1:]), sum(y) / len(vert[1:]), sum(z) / len(vert[1:])))
            rad = []
            for vert_el in vert[1:]:
                rad.append(np.linalg.norm(centroid - vert_el))
            # med_rad = np.median(rad)

            med_rad = np.median(rad)
            # new_rad = [inx for inx in rad if abs(inx-med_rad) < 0.988*np.var(rad)]
            new_rad = [inx for inx in rad if inx > 0.4 * med_rad]

            # med_rad = np.median(rad[len(rad)])
            med_rad = np.median(new_rad)


            diameters[el] = med_rad * 2
            # skel_node[el] = centroid
            if el < len(found_coords):
                skel_radius[str(found_coords[el])] = med_rad
    found_tree = sp.cKDTree(found_coords)
    for el in missing_coords_ixs:
        nearest_found_node = found_tree.query(skel_node[el], 1)
        diameters[el] = diameters[nearest_found_node[1]]
        # skel_node[el] = skel_node[nearest_found_node[1]]

    sso_skel['diameters'] = diameters

    return sso_skel


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


def map_cs_properties(cs):
    cs.load_attr_dict()
    if "neuron_partner_ct" in cs.attr_dict:
        return
    partners = cs.attr_dict["neuron_partners"]
    partner_ax = np.zeros(2, dtype=np.uint8)
    partner_ct = np.zeros(2, dtype=np.uint8)
    for kk, ix in enumerate(partners):
        sso = ss.SuperSegmentationObject(ix, working_dir="/wholebrain"
                                                         "/scratch/areaxfs/")
        sso.load_attr_dict()
        ax = get_sso_axoness_from_coord(sso, cs.rep_coord)
        partner_ax[kk] = np.uint(ax)
        partner_ct[kk] = np.uint(sso.attr_dict["celltype_cnn"])
    cs.attr_dict["neuron_partner_ax"] = partner_ax
    cs.attr_dict["neuron_partner_ct"] = partner_ct
    cs.save_attr_dict()