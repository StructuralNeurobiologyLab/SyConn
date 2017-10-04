# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Sven Dorkenwald, Philipp Schubert, Joergen Kornfeld

from collections import defaultdict
import networkx as nx
import numpy as np
import os
import scipy.spatial
import time

from ..representations import super_segmentation as ss, segmentation, \
    connectivity_helper as ch
from ..handler.compression import VoxelDict, AttributeDict


def combine_and_split_cs_agg_helper(args):
    wd = args[0]
    rel_cs_to_cs_agg_ids_items = args[1]
    voxel_rel_paths = args[2]
    cs_agg_version = args[3]
    cs_version = args[4]
    scaling = args[5]
    cs_gap_nm = args[6]

    cs = segmentation.SegmentationDataset("cs", working_dir=wd,
                                          version=cs_version)

    cs_agg = segmentation.SegmentationDataset("cs_agg", working_dir=wd,
                                              version=cs_agg_version)

    n_per_voxel_path = np.ceil(float(len(rel_cs_to_cs_agg_ids_items)) / len(voxel_rel_paths))

    n_items_for_path = 0
    cur_path_id = 0

    os.makedirs(cs.so_storage_path + voxel_rel_paths[cur_path_id])
    voxel_dc = VoxelDict(cs.so_storage_path + voxel_rel_paths[cur_path_id] +
                         "/voxel.pkl", read_only=False, timeout=3600)
    attr_dc = AttributeDict(cs.so_storage_path + voxel_rel_paths[cur_path_id] +
                            "/attr_dict.pkl", read_only=False, timeout=3600)

    p_parts = voxel_rel_paths[cur_path_id].strip("/").split("/")
    next_id = int("%.2d%.2d%d" % (int(p_parts[0]), int(p_parts[1]),
                                  int(p_parts[2])))

    for item in rel_cs_to_cs_agg_ids_items:
        n_items_for_path += 1

        ssv_ids = ch.sv_id_to_partner_ids_vec([item[0]])[0]

        voxel_list = cs_agg.get_segmentation_object(item[1][0]).voxel_list
        for cs_agg_id in item[1][1:]:
            cs_agg_object = cs_agg.get_segmentation_object(cs_agg_id)
            np.concatenate([voxel_list, cs_agg_object.voxel_list])

        distances = scipy.spatial.distance.cdist(voxel_list * scaling,
                                                 voxel_list * scaling)
        distances[np.triu_indices_from(distances)] = cs_gap_nm
        edges = np.array(np.where(distances < cs_gap_nm)).T

        graph = nx.from_edgelist(edges)
        cc = nx.connected_components(graph)

        i_cc = 0
        for this_cc in cc:
            print i_cc, next_id
            i_cc += 1
            this_vx = voxel_list[np.array(list(this_cc))]
            abs_offset = np.min(this_vx, axis=0)
            this_vx -= abs_offset

            id_mask = np.zeros(np.max(this_vx, axis=0) + 1, dtype=np.bool)
            id_mask[this_vx[:, 0], this_vx[:, 1], this_vx[:, 2]] = True
            voxel_dc[next_id] = [id_mask], [abs_offset]
            attr_dc[next_id] = dict(neuron_partners=ssv_ids)
            next_id += 100000

        if n_items_for_path > n_per_voxel_path:
            voxel_dc.save2pkl(cs.so_storage_path + voxel_rel_paths[cur_path_id] +
                              "/voxel.pkl")
            attr_dc.save2pkl(cs.so_storage_path + voxel_rel_paths[cur_path_id] +
                             "/attr_dict.pkl")

            cur_path_id += 1
            n_items_for_path = 0
            p_parts = voxel_rel_paths[cur_path_id].strip("/").split("/")

            next_id = int("%.2d%.2d%d" % (int(p_parts[0]), int(p_parts[1]),
                                          int(p_parts[2])))

            os.makedirs(cs.so_storage_path + voxel_rel_paths[cur_path_id])

            voxel_dc = VoxelDict(cs.so_storage_path + voxel_rel_paths[cur_path_id] +
                                 "voxel.pkl", read_only=False, timeout=3600)
            attr_dc = AttributeDict(cs.so_storage_path +
                                    voxel_rel_paths[cur_path_id] + "attr_dict.pkl",
                                    read_only=False, timeout=3600)

    if n_items_for_path > 0:
        voxel_dc.save2pkl(cs.so_storage_path + voxel_rel_paths[cur_path_id] +
                          "/voxel.pkl")
        attr_dc.save2pkl(cs.so_storage_path + voxel_rel_paths[cur_path_id] +
                         "/attr_dict.pkl")

    print "done"


def map_objects_to_cs_thread(args):
    paths = args[0]
    obj_types = args[1]
    cs_version = args[2]
    ssd_version = args[3]
    working_dir = args[4]
    max_map_dist_nm = args[5]

    cs_dataset = segmentation.SegmentationDataset("cs", version=cs_version,
                                                  working_dir=working_dir)

    ssd = ss.SuperSegmentationDataset(version=ssd_version,
                                      working_dir=working_dir)

    bbs_dict = {}
    ids_dict = {}
    for obj_type in obj_types:
        objd = segmentation.SegmentationDataset(obj_type=obj_type,
                                                working_dir=working_dir,
                                                version=ssd.version_dict[obj_type])
        bbs_dict[obj_type] = objd.load_cached_data("bounding_box")
        ids_dict[obj_type] = objd.ids

    for p in paths:
        this_attr_dc = AttributeDict(p + "/attr_dict.pkl",
                                     read_only=False, timeout=3600)
        this_vx_dc = VoxelDict(p + "/voxel.pkl", read_only=True,
                               timeout=3600)

        for cs_id in this_vx_dc.keys():
            print cs_id
            cs_obj = cs_dataset.get_segmentation_object(cs_id)
            cs_obj.attr_dict = this_attr_dc[cs_id]

            mapping_feats = map_objects_to_single_cs(cs_obj,
                                                     ssd_version=ssd_version,
                                                     max_map_dist_nm=max_map_dist_nm,
                                                     obj_types=obj_types,
                                                     attr_dict_loaded=True,
                                                     bbs_dict=bbs_dict,
                                                     ids_dict=ids_dict)

            cs_obj.attr_dict.update(mapping_feats)
            this_attr_dc[cs_id] = cs_obj.attr_dict
        this_attr_dc.save2pkl()


def map_objects_to_single_cs(cs_obj, ssd_version=None, max_map_dist_nm=2000,
                             obj_types=("sj", "mi", "vc"),
                             attr_dict_loaded=False, neuron_partners=None,
                             bbs_dict=None, ids_dict=None):
    if not attr_dict_loaded:
        cs_obj.load_attr_dict()

    candidate_ids = dict([(obj_types[i], defaultdict(list))
                          for i in range(len(obj_types))])
    mapping_feats = {}

    version_dict = None

    if neuron_partners is None:
        neuron_partners = cs_obj.attr_dict["neuron_partners"]

    for partner_id in neuron_partners:
        ssv = ss.SuperSegmentationObject(partner_id, version=ssd_version,
                                         working_dir=cs_obj.working_dir)
        ssv.load_attr_dict()
        version_dict = ssv.version_dict

        for obj_type in obj_types:
            candidate_ids[obj_type][partner_id] += \
                ssv.attr_dict["mapping_%s_ids" % obj_type]

    cs_obj_vxl_scaled = cs_obj.voxel_list * cs_obj.scaling

    try:
        cs_area = scipy.spatial.ConvexHull(cs_obj_vxl_scaled).area / 2.e6
        mapping_feats["cs_area"] = cs_area
    except:
        mapping_feats["cs_area"] = 0

    for obj_type in obj_types:
        if bbs_dict is None or ids_dict is None:
            objd = segmentation.SegmentationDataset(obj_type=obj_type,
                                                    working_dir=cs_obj.working_dir,
                                                    version=version_dict[obj_type])
            bbs = objd.load_cached_data("bounding_box")
            ids = objd.ids
        else:
            bbs = bbs_dict[obj_type]
            ids = ids_dict[obj_type]

        if obj_type == "sj":
            sj_ids = candidate_ids["sj"].values()
            sj_ids = np.unique(sj_ids[0] + sj_ids[1])
            sj_voxels = []
            n_vxs_per_sj = [0]
            considered_sj_ids = []

            for sj_id in sj_ids:
                sj_bb = bbs[ids == sj_id][0]
                if np.all(sj_bb[0] - cs_obj.bounding_box[1] < 0):
                    if np.all(sj_bb[1] - cs_obj.bounding_box[0] > 0):
                        sj = segmentation.SegmentationObject(sj_id,
                                                             obj_type="sj",
                                                             working_dir=cs_obj.working_dir,
                                                             version=
                                                             version_dict["sj"])

                        sj_voxels += list(sj.voxel_list)
                        n_vxs_per_sj.append(len(sj.voxel_list))
                        considered_sj_ids.append(sj_id)

            n_vxs_per_sj = np.array(n_vxs_per_sj)
            if len(sj_voxels) > 0:
                dists = scipy.spatial.distance.cdist(cs_obj.voxel_list,
                                                     sj_voxels)
                dist_mask = dists == 0

                vx_hits = np.any(dist_mask, axis=0)

                locs = np.cumsum(n_vxs_per_sj)
                overlapping_sj_mask = \
                    np.array([np.any(vx_hits[locs[i]: locs[i+1]])
                              for i in range(len(locs) - 1)], dtype=np.bool)

                n_sj_vx = np.sum(n_vxs_per_sj[1:][overlapping_sj_mask])
                # n_overlap_vxs = np.sum(dist_mask)

                cs_overlapping_vx = cs_obj.voxel_list[np.any(dist_mask, axis=1)]

                if len(cs_overlapping_vx) > 0:
                    try:
                        overlap_area = scipy.spatial.ConvexHull(
                            cs_overlapping_vx * cs_obj.scaling).area / 2.e6
                    except:
                        overlap_area = 0
                    sj_overlapped_part = float(len(cs_overlapping_vx)) / n_sj_vx
                else:
                    overlap_area = 0
                    sj_overlapped_part = 0

                cs_overlapped_part = float(len(cs_overlapping_vx)) / \
                                     len(cs_obj.voxel_list)

                mapping_feats["overlap_area"] = overlap_area
                mapping_feats["sj_overlapped_part"] = sj_overlapped_part
                mapping_feats["cs_overlapped_part"] = cs_overlapped_part
                mapping_feats["mapping_sj_ids"] = np.array(considered_sj_ids)[overlapping_sj_mask]
            else:
                mapping_feats["overlap_area"] = 0
                mapping_feats["sj_overlapped_part"] = 0
                mapping_feats["cs_overlapped_part"] = 0
                mapping_feats["mapping_sj_ids"] = []
        else:
            mapped_obj_ids = defaultdict(list)

            for partner_id in candidate_ids[obj_type].keys():
                obj_voxels = []
                n_vxs_per_obj = [0]
                considered_obj_ids = []

                for obj_id in candidate_ids[obj_type][partner_id]:
                    obj_bb = bbs[ids == obj_id][0]
                    if np.all(obj_bb[0] - cs_obj.bounding_box[1] - max_map_dist_nm / cs_obj.scaling < 0):
                        if np.all(obj_bb[1] - cs_obj.bounding_box[0] + max_map_dist_nm / cs_obj.scaling > 0):
                            obj = segmentation.SegmentationObject(obj_id=obj_id,
                                                                  obj_type=obj_type,
                                                                  version=version_dict[obj_type],
                                                                  working_dir=cs_obj.working_dir)

                            obj_voxels += list(obj.voxel_list[::10])
                            n_vxs_per_obj.append(len(obj.voxel_list[::10]))
                            considered_obj_ids.append(obj_id)

                obj_voxels = np.array(obj_voxels)
                n_vxs_per_obj = np.array(n_vxs_per_obj)
                if len(obj_voxels) > 0:
                    dists = scipy.spatial.distance.cdist(cs_obj_vxl_scaled,
                                                         obj_voxels * cs_obj.scaling)
                    dist_mask = dists < max_map_dist_nm

                    vx_hits = np.any(dist_mask, axis=0)

                    locs = np.cumsum(n_vxs_per_obj)
                    close_obj_mask = np.array([np.any(vx_hits[locs[i]: locs[i + 1]])
                                               for i in range(len(locs) - 1)],
                                              dtype=np.bool)
                    mapped_obj_ids[partner_id] = np.array(considered_obj_ids)[close_obj_mask]

                mapping_feats["mapping_%s_ids" % obj_type] = mapped_obj_ids

    return mapping_feats

















