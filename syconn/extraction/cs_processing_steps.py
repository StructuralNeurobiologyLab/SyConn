# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Philipp Schubert, Joergen Kornfeld
from collections import defaultdict
import networkx as nx
import numpy as np
import glob
import os
from scipy import spatial
from sklearn import ensemble, cross_validation, externals


from knossos_utils import knossosdataset, skeleton_utils, skeleton

from ..mp import qsub_utils as qu
from ..mp import mp_utils as sm
script_folder = os.path.abspath(os.path.dirname(__file__) + "/../QSUB_scripts/")

from ..reps import super_segmentation, segmentation, connectivity_helper as ch
from ..reps.rep_helper import subfold_from_ix, ix_from_subfold
from ..backend.storage import AttributeDict, VoxelStorage
from ..handler.basics import chunkify


def filter_relevant_cs_agg(cs_agg, ssd):
    sv_ids = ch.sv_id_to_partner_ids_vec(cs_agg.ids)

    cs_agg_ids = cs_agg.ids.copy()

    sv_ids[sv_ids >= len(ssd.id_changer)] = -1
    mapped_sv_ids = ssd.id_changer[sv_ids]

    mask = np.all(mapped_sv_ids > 0, axis=1)
    cs_agg_ids = cs_agg_ids[mask]
    filtered_mapped_sv_ids = mapped_sv_ids[mask]

    mask = filtered_mapped_sv_ids[:, 0] - filtered_mapped_sv_ids[:, 1] != 0
    cs_agg_ids = cs_agg_ids[mask]
    relevant_cs_agg = filtered_mapped_sv_ids[mask]

    relevant_cs_ids = np.left_shift(np.max(relevant_cs_agg, axis=1), 32) + np.min(relevant_cs_agg, axis=1)

    rel_cs_to_cs_agg_ids = defaultdict(list)
    for i_entry in range(len(relevant_cs_ids)):
        rel_cs_to_cs_agg_ids[relevant_cs_ids[i_entry]].\
            append(cs_agg_ids[i_entry])

    return rel_cs_to_cs_agg_ids


def combine_and_split_cs_agg(wd, cs_gap_nm=300, ssd_version=None,
                             cs_agg_version=None,
                             stride=1000, qsub_pe=None, qsub_queue=None,
                             nb_cpus=None, n_max_co_processes=None):

    ssd = super_segmentation.SuperSegmentationDataset(wd, version=ssd_version)
    cs_agg = segmentation.SegmentationDataset("cs_agg", working_dir=wd,
                                              version=cs_agg_version)

    rel_cs_to_cs_agg_ids = filter_relevant_cs_agg(cs_agg, ssd)

    voxel_rel_paths_2stage = np.unique([subfold_from_ix(ix, 100000)[:-2]
                                        for ix in range(100000)])

    voxel_rel_paths = [subfold_from_ix(ix, 100000) for ix in range(100000)]
    block_steps = np.linspace(0, len(voxel_rel_paths),
                              int(np.ceil(float(len(rel_cs_to_cs_agg_ids)) / stride)) + 1).astype(np.int)

    cs = segmentation.SegmentationDataset("cs_ssv", working_dir=wd, version="new",
                                          create=True, n_folders_fs=100000)

    for p in voxel_rel_paths_2stage:
        os.makedirs(cs.so_storage_path + p)

    rel_cs_to_cs_agg_ids_items = list(rel_cs_to_cs_agg_ids.items())
    i_block = 0
    multi_params = []
    for block in [rel_cs_to_cs_agg_ids_items[i:i + stride]
                  for i in range(0, len(rel_cs_to_cs_agg_ids_items), stride)]:
        multi_params.append([wd, block,
                             voxel_rel_paths[block_steps[i_block]: block_steps[i_block+1]],
                             cs_agg.version, cs.version, ssd.scaling, cs_gap_nm])
        i_block += 1

    if qsub_pe is None and qsub_queue is None:
        results = sm.start_multiprocess(_combine_and_split_cs_agg_thread,
                                        multi_params, nb_cpus=nb_cpus)

    elif qu.__QSUB__:
        path_to_out = qu.QSUB_script(multi_params,
                                     "combine_and_split_cs_agg",
                                     pe=qsub_pe, queue=qsub_queue,
                                     script_folder=script_folder,
                                     n_max_co_processes=n_max_co_processes)
    else:
        raise Exception("QSUB not available")

    return cs


def _combine_and_split_cs_agg_thread(args):
    wd = args[0]
    rel_cs_to_cs_agg_ids_items = args[1]
    voxel_rel_paths = args[2]
    cs_agg_version = args[3]
    cs_version = args[4]
    scaling = args[5]
    cs_gap_nm = args[6]

    cs = segmentation.SegmentationDataset("cs_ssv", working_dir=wd,
                                          version=cs_version)

    cs_agg = segmentation.SegmentationDataset("cs_agg", working_dir=wd,
                                              version=cs_agg_version)

    n_per_voxel_path = np.ceil(float(len(rel_cs_to_cs_agg_ids_items)) / len(voxel_rel_paths))

    n_items_for_path = 0
    cur_path_id = 0

    try:
        os.makedirs(cs.so_storage_path + voxel_rel_paths[cur_path_id])
    except:
        pass
    voxel_dc = VoxelStorage(cs.so_storage_path + voxel_rel_paths[cur_path_id] +
                         "/voxel.pkl", read_only=False)
    attr_dc = AttributeDict(cs.so_storage_path + voxel_rel_paths[cur_path_id] +
                            "/attr_dict.pkl", read_only=False)

    p_parts = voxel_rel_paths[cur_path_id].strip("/").split("/")
    next_id = int("%.2d%.2d%d" % (int(p_parts[0]), int(p_parts[1]),
                                  int(p_parts[2])))

    for item in rel_cs_to_cs_agg_ids_items:
        n_items_for_path += 1

        ssv_ids = ch.sv_id_to_partner_ids_vec([item[0]])[0]

        voxel_list = cs_agg.get_segmentation_object(item[1][0]).voxel_list
        for cs_agg_id in item[1][1:]:
            cs_agg_object = cs_agg.get_segmentation_object(cs_agg_id)
            voxel_list = np.concatenate([voxel_list, cs_agg_object.voxel_list])

        # if len(voxel_list) < 1e4:
        #     kdtree = spatial.cKDTree(voxel_list * scaling)
        #     pairs = kdtree.query_pairs(r=cs_gap_nm)
        #     graph = nx.from_edgelist(pairs)
        #     ccs = list(nx.connected_components(graph))
        # else:
        ccs = cc_large_voxel_lists(voxel_list * scaling, cs_gap_nm)

        i_cc = 0
        for this_cc in ccs:
            this_vx = voxel_list[np.array(list(this_cc))]
            abs_offset = np.min(this_vx, axis=0)
            this_vx -= abs_offset

            id_mask = np.zeros(np.max(this_vx, axis=0) + 1, dtype=np.bool)
            id_mask[this_vx[:, 0], this_vx[:, 1], this_vx[:, 2]] = True

            print(i_cc, next_id, len(this_vx), id_mask.shape)
            try:
                voxel_dc[next_id] = [id_mask], [abs_offset]
            except:
                print("failed", item)
                np.save(cs.so_storage_path + "/%d_%d_%d_%d.npy" %
                        (next_id, abs_offset[0], abs_offset[1], abs_offset[2]), this_vx)

            attr_dc[next_id] = dict(neuron_partners=ssv_ids)
            next_id += 100000

        if n_items_for_path > n_per_voxel_path:
            voxel_dc.push(cs.so_storage_path + voxel_rel_paths[cur_path_id] +
                              "/voxel.pkl")
            attr_dc.push(cs.so_storage_path + voxel_rel_paths[cur_path_id] +
                             "/attr_dict.pkl")

            cur_path_id += 1
            n_items_for_path = 0
            p_parts = voxel_rel_paths[cur_path_id].strip("/").split("/")

            next_id = int("%.2d%.2d%d" % (int(p_parts[0]), int(p_parts[1]),
                                          int(p_parts[2])))

            try:
                os.makedirs(cs.so_storage_path + voxel_rel_paths[cur_path_id])
            except:
                pass

            voxel_dc = VoxelStorage(cs.so_storage_path + voxel_rel_paths[cur_path_id] +
                                 "voxel.pkl", read_only=False)
            attr_dc = AttributeDict(cs.so_storage_path +
                                    voxel_rel_paths[cur_path_id] + "attr_dict.pkl",
                                    read_only=False)

    if n_items_for_path > 0:
        voxel_dc.push(cs.so_storage_path + voxel_rel_paths[cur_path_id] +
                          "/voxel.pkl")
        attr_dc.push(cs.so_storage_path + voxel_rel_paths[cur_path_id] +
                         "/attr_dict.pkl")

    print("done")


def cc_large_voxel_lists(voxel_list, cs_gap_nm, max_concurrent_nodes=5000):
    kdtree = spatial.cKDTree(voxel_list)

    checked_ids = np.array([], dtype=np.int)
    next_ids = np.array([0])
    ccs = [set(next_ids)]

    current_ccs = 0
    vx_ids = np.arange(len(voxel_list), dtype=np.int)

    while True:
        print("NEXT - %d - %d" % (len(next_ids), len(checked_ids)))
        for cc in ccs:
            print("N voxels in cc: %d" % (len(cc)))

        if len(next_ids) == 0:
            p_ids = vx_ids[~np.in1d(vx_ids, checked_ids)]
            if len(p_ids) == 0:
                break
            else:
                current_ccs += 1
                ccs.append(set([p_ids[0]]))
                next_ids = p_ids[:1]

        q_ids = kdtree.query_ball_point(voxel_list[next_ids], r=cs_gap_nm, )
        checked_ids = np.concatenate([checked_ids, next_ids])

        for q_id in q_ids:
            ccs[current_ccs].update(q_id)

        cc_ids = np.array(list(ccs[current_ccs]))
        next_ids = vx_ids[cc_ids[~np.in1d(cc_ids, checked_ids)][:max_concurrent_nodes]]

    return ccs


def map_objects_to_cs(wd, cs_version=None, ssd_version=None, max_map_dist_nm=2000,
                      obj_types=("sj", "mi", "vc"), stride=1000, qsub_pe=None,
                      qsub_queue=None, nb_cpus=1, n_max_co_processes=100):
    cs_dataset = segmentation.SegmentationDataset("cs_ssv", version=cs_version,
                                                  working_dir=wd)
    paths = glob.glob(cs_dataset.so_storage_path + "/*/*/*")

    multi_params = []
    for path_block in [paths[i:i + stride]
                       for i in range(0, len(paths), stride)]:
        multi_params.append([path_block, obj_types, cs_version, ssd_version,
                             wd, max_map_dist_nm])

    if qsub_pe is None and qsub_queue is None:
        results = sm.start_multiprocess(_map_objects_to_cs_thread,
                                        multi_params, nb_cpus=nb_cpus)

    elif qu.__QSUB__:
        path_to_out = qu.QSUB_script(multi_params,
                                     "map_objects_to_cs",
                                     pe=qsub_pe, queue=qsub_queue,
                                     script_folder=script_folder,
                                     n_max_co_processes=n_max_co_processes)
    else:
        raise Exception("QSUB not available")


def _map_objects_to_cs_thread(args):
    paths = args[0]
    obj_types = args[1]
    cs_version = args[2]
    ssd_version = args[3]
    working_dir = args[4]
    max_map_dist_nm = args[5]

    cs_dataset = segmentation.SegmentationDataset("cs_ssv", version=cs_version,
                                                  working_dir=working_dir)

    ssd = super_segmentation.SuperSegmentationDataset(version=ssd_version,
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
                                     read_only=False)
        this_vx_dc = VoxelStorage(p + "/voxel.pkl", read_only=True)

        for cs_id in this_vx_dc.keys():
            print(cs_id)
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
        this_attr_dc.push()


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
        ssv = super_segmentation.SuperSegmentationObject(partner_id,
                                                         version=ssd_version,
                                                         working_dir=cs_obj.working_dir)
        ssv.load_attr_dict()
        version_dict = ssv.version_dict

        for obj_type in obj_types:
            candidate_ids[obj_type][partner_id] += \
                ssv.attr_dict["mapping_%s_ids" % obj_type]

    cs_obj_vxl_scaled = cs_obj.voxel_list * cs_obj.scaling

    try:
        cs_area = spatial.ConvexHull(cs_obj_vxl_scaled).area / 2.e6
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
                dists = spatial.distance.cdist(cs_obj.voxel_list,
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
                        overlap_area = spatial.ConvexHull(
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
                    dists = spatial.distance.cdist(cs_obj_vxl_scaled,
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


def overlap_mapping_sj_to_cs(cs_sd, sj_sd, rep_coord_dist_nm=2000,
                             n_folders_fs=10000,
                             stride=20, qsub_pe=None, qsub_queue=None,
                             nb_cpus=None, n_max_co_processes=None):
    assert n_folders_fs % stride == 0

    wd = cs_sd.working_dir

    voxel_rel_paths = [subfold_from_ix(ix, n_folders_fs) for ix in range(n_folders_fs)]
    conn_sd = segmentation.SegmentationDataset("conn", working_dir=wd, version="new",
                                               create=True, n_folders_fs=n_folders_fs)

    for p in voxel_rel_paths:
        os.makedirs(conn_sd.so_storage_path + p)

    multi_params = []
    for block_bs in [[i, i+stride] for i in range(0, n_folders_fs, stride)]:
        multi_params.append([wd, block_bs[0], block_bs[1], conn_sd.version,
                             sj_sd.version, cs_sd.version,
                             rep_coord_dist_nm])

    if qsub_pe is None and qsub_queue is None:
        results = sm.start_multiprocess(_overlap_mapping_sj_to_cs_thread,
                                        multi_params, nb_cpus=nb_cpus)

    elif qu.__QSUB__:
        path_to_out = qu.QSUB_script(multi_params,
                                     "overlap_mapping_sj_to_cs",
                                     pe=qsub_pe, queue=qsub_queue,
                                     script_folder=script_folder,
                                     n_max_co_processes=n_max_co_processes)
    else:
        raise Exception("QSUB not available")


def _overlap_mapping_sj_to_cs_thread(args):
    wd, block_start, block_end, conn_sd_version, sj_sd_version, cs_sd_version, \
        rep_coord_dist_nm = args

    conn_sd = segmentation.SegmentationDataset("conn", working_dir=wd,
                                               version=conn_sd_version,
                                               create=False)
    sj_sd = segmentation.SegmentationDataset("sj", working_dir=wd,
                                             version=sj_sd_version,
                                             create=False)
    cs_sd = segmentation.SegmentationDataset("cs_ssv", working_dir=wd,
                                             version=cs_sd_version,
                                             create=False)

    cs_id_assignment = np.linspace(0, len(cs_sd.ids), conn_sd.n_folders_fs+1).astype(np.int)

    sj_kdtree = spatial.cKDTree(sj_sd.rep_coords[sj_sd.sizes > sj_sd.config.entries['Sizethresholds']['sj']] * sj_sd.scaling)

    for i_cs_start_id, cs_start_id in enumerate(cs_id_assignment[block_start: block_end]):

        rel_path = subfold_from_ix(i_cs_start_id + block_start, conn_sd.n_folders_fs)

        voxel_dc = VoxelStorage(conn_sd.so_storage_path + rel_path + "/voxel.pkl",
                                read_only=False)
        attr_dc = AttributeDict(conn_sd.so_storage_path + rel_path + "/attr_dict.pkl",
                                read_only=False)

        next_conn_id = i_cs_start_id + block_start
        n_items_for_path = 0
        for cs_list_id in range(cs_start_id, cs_id_assignment[block_start + i_cs_start_id + 1]):
            cs_id = cs_sd.ids[cs_list_id]

            print('CS ID: %d' % cs_id)

            cs = cs_sd.get_segmentation_object(cs_id)

            overlap_vx_l = overlap_mapping_sj_to_cs_single(cs, sj_sd,
                                                           sj_kdtree=sj_kdtree,
                                                           rep_coord_dist_nm=rep_coord_dist_nm)

            for l in overlap_vx_l:
                sj_id, overlap_vx = l

                bounding_box = [np.min(overlap_vx, axis=0),
                                np.max(overlap_vx, axis=0) + 1]

                vx = np.zeros(bounding_box[1] - bounding_box[0], dtype=np.bool)
                overlap_vx -= bounding_box[0]
                vx[overlap_vx[:, 0], overlap_vx[:, 1], overlap_vx[:, 2]] = True

                voxel_dc[next_conn_id] = [vx], [bounding_box[0]]

                attr_dc[next_conn_id] = {'sj_id': sj_id,
                                         'cs_id': cs_id,
                                         'ssv_partners': cs.lookup_in_attribute_dict('neuron_partners')}

                next_conn_id += conn_sd.n_folders_fs
                n_items_for_path += 1

        if n_items_for_path > 0:
            voxel_dc.push(conn_sd.so_storage_path + rel_path + "/voxel.pkl")
            attr_dc.push(conn_sd.so_storage_path + rel_path + "/attr_dict.pkl")


def overlap_mapping_sj_to_cs_single(cs, sj_sd, sj_kdtree=None, rep_coord_dist_nm=2000):
    cs_kdtree = spatial.cKDTree(cs.voxel_list * cs.scaling)

    if sj_kdtree is None:
        sj_kdtree = spatial.cKDTree(sj_sd.rep_coords[sj_sd.sizes > sj_sd.config.entries['Sizethresholds']['sj']] * sj_sd.scaling)

    cand_sj_ids_l = sj_kdtree.query_ball_point(cs.voxel_list * cs.scaling,
                                               r=rep_coord_dist_nm)
    u_cand_sj_ids = set()
    for l in cand_sj_ids_l:
        u_cand_sj_ids.update(l)

    if len(u_cand_sj_ids) == 0:
        return []

    u_cand_sj_ids = sj_sd.ids[sj_sd.sizes > sj_sd.config.entries['Sizethresholds']['sj']][np.array(list(u_cand_sj_ids))]

    print("%d candidate sjs" % len(u_cand_sj_ids))

    overlap_vx_l = []
    for sj_id in u_cand_sj_ids:
        sj = sj_sd.get_segmentation_object(sj_id, create=False)
        dists, _ = cs_kdtree.query(sj.voxel_list * sj.scaling,
                                   distance_upper_bound=1)

        overlap_vx = sj.voxel_list[dists == 0]
        if len(overlap_vx) > 0:
            overlap_vx_l.append([sj_id, overlap_vx])

    print("%d candidate sjs overlap" % len(overlap_vx_l))

    return overlap_vx_l


def overlap_mapping_sj_to_cs_via_kd(cs_sd, sj_sd, cs_kd,
                                    n_folders_fs=10000, n_job_chunks=1000,
                                    qsub_pe=None, qsub_queue=None,
                                    nb_cpus=None, n_max_co_processes=None):

    wd = cs_sd.working_dir

    rel_sj_ids = sj_sd.ids[sj_sd.sizes > sj_sd.config.entries['Sizethresholds']['sj']]

    voxel_rel_paths = [subfold_from_ix(ix, n_folders_fs) for ix in range(n_folders_fs)]
    conn_sd = segmentation.SegmentationDataset("conn", working_dir=wd, version="new",
                                               create=True, n_folders_fs=n_folders_fs)

    for p in voxel_rel_paths:
        os.makedirs(conn_sd.so_storage_path + p)

    sj_id_blocks = np.array_split(rel_sj_ids, n_job_chunks)
    voxel_rel_path_blocks = np.array_split(voxel_rel_paths, n_job_chunks)

    multi_params = []
    for i_block in range(n_job_chunks):
        multi_params.append([wd, sj_id_blocks[i_block],
                             voxel_rel_path_blocks[i_block], conn_sd.version,
                             sj_sd.version, cs_sd.version, cs_kd.knossos_path])

    if qsub_pe is None and qsub_queue is None:
        results = sm.start_multiprocess(_overlap_mapping_sj_to_cs_via_kd_thread,
                                        multi_params, nb_cpus=nb_cpus)

    elif qu.__QSUB__:
        path_to_out = qu.QSUB_script(multi_params,
                                     "overlap_mapping_sj_to_cs_via_kd",
                                     pe=qsub_pe, queue=qsub_queue,
                                     script_folder=script_folder,
                                     n_max_co_processes=n_max_co_processes)
    else:
        raise Exception("QSUB not available")
    return conn_sd


def _overlap_mapping_sj_to_cs_via_kd_thread(args):
    wd, sj_ids, voxel_rel_paths, conn_sd_version, sj_sd_version, \
        cs_sd_version, cs_kd_path = args

    conn_sd = segmentation.SegmentationDataset("conn", working_dir=wd,
                                               version=conn_sd_version,
                                               create=False)
    sj_sd = segmentation.SegmentationDataset("sj", working_dir=wd,
                                             version=sj_sd_version,
                                             create=False)
    cs_sd = segmentation.SegmentationDataset("cs_ssv", working_dir=wd,
                                             version=cs_sd_version,
                                             create=False)

    cs_kd = knossosdataset.KnossosDataset()
    cs_kd.initialize_from_knossos_path(cs_kd_path)

    sj_id_blocks = np.array_split(sj_ids, len(voxel_rel_paths))

    for i_sj_id_block, sj_id_block in enumerate(sj_id_blocks):
        rel_path = voxel_rel_paths[i_sj_id_block]

        voxel_dc = VoxelStorage(conn_sd.so_storage_path + rel_path + "/voxel.pkl",
                                read_only=False)
        attr_dc = AttributeDict(conn_sd.so_storage_path + rel_path + "/attr_dict.pkl",
                                read_only=False)

        next_conn_id = ix_from_subfold(rel_path,
                                       conn_sd.n_folders_fs)

        for sj_id in sj_id_block:
            sj = sj_sd.get_segmentation_object(sj_id)
            vxl = sj.voxel_list

            cs_ids = cs_kd.from_overlaycubes_to_list(vxl, datatype=np.uint64)
            u_cs_ids, c_cs_ids = np.unique(cs_ids, return_counts=True)

            zero_ratio = c_cs_ids[u_cs_ids == 0] / np.sum(c_cs_ids)

            for cs_id in u_cs_ids:
                if cs_id == 0:
                    continue

                cs = cs_sd.get_segmentation_object(cs_id)

                id_ratio = c_cs_ids[u_cs_ids == cs_id] / float(np.sum(c_cs_ids))
                overlap_vx = vxl[cs_ids == cs_id]
                cs_ratio = float(len(overlap_vx)) / cs.size

                bounding_box = [np.min(overlap_vx, axis=0),
                                np.max(overlap_vx, axis=0) + 1]

                vx_block = np.zeros(bounding_box[1] - bounding_box[0], dtype=np.bool)
                overlap_vx -= bounding_box[0]
                vx_block[overlap_vx[:, 0], overlap_vx[:, 1], overlap_vx[:, 2]] = True

                voxel_dc[next_conn_id] = [vx_block], [bounding_box[0]]

                attr_dc[next_conn_id] = {'sj_id': sj_id,
                                         'cs_id': cs_id,
                                         'id_sj_ratio': id_ratio,
                                         'id_cs_ratio': cs_ratio,
                                         'background_overlap_ratio': zero_ratio,
                                         'ssv_partners':
                                             cs.lookup_in_attribute_dict(
                                                 'neuron_partners')}

                next_conn_id += conn_sd.n_folders_fs

        voxel_dc.push(conn_sd.so_storage_path + rel_path + "/voxel.pkl")
        attr_dc.push(conn_sd.so_storage_path + rel_path + "/attr_dict.pkl")


def write_conn_gt_kzips(conn, n_objects, folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

    conn_ids = conn.ids[np.random.choice(len(conn.ids), n_objects, replace=False)]

    for conn_id in conn_ids:
        obj = conn.get_segmentation_object(conn_id)
        p = folder + "/obj_%d.k.zip" % conn_id

        obj.save_kzip(p)
        obj.mesh2kzip(p)

        a = skeleton.SkeletonAnnotation()
        a.scaling = obj.scaling
        a.comment = "rep coord - %d" % obj.size

        a.addNode(skeleton.SkeletonNode().from_scratch(a, obj.rep_coord[0],
                                                       obj.rep_coord[1],
                                                       obj.rep_coord[2],
                                                       radius=1))
        skeleton_utils.write_skeleton(folder + "/obj_%d.k.zip" % conn_id, [a])


def create_conn_syn_gt(conn, path_kzip):
    annos = skeleton_utils.loadj0126NML(path_kzip)

    label_coords = []
    labels = []
    for anno in annos:
        node = list(anno.getNodes())[0]
        labels.append(node.getComment())
        label_coords.append(np.array(node.getCoordinate()))

    labels = np.array(labels)
    label_coords = np.array(label_coords)

    conn_kdtree = spatial.cKDTree(conn.rep_coords * conn.scaling)
    ds, list_ids = conn_kdtree.query(label_coords * conn.scaling)

    conn_ids = conn.ids[list_ids]
    for label_id in np.where(ds > 0)[0]:
        _, close_ids = conn_kdtree.query(label_coords[label_id] * conn.scaling, k=100)

        # print("\n-------------")
        for close_id in close_ids:
            # print(close_id)
            conn_o = conn.get_segmentation_object(conn.ids[close_id])

            vx_ds = np.sum(np.abs(conn_o.voxel_list - label_coords[label_id]), axis=-1)

            if np.min(vx_ds) == 0:
                conn_ids[label_id] = conn.ids[close_id]
                break

        assert 0 in vx_ds

    features = []

    for conn_id in conn_ids:
        conn_o = conn.get_segmentation_object(conn_id)

        features.append(conn_o_features(conn_o))

    features = np.array(features)

    rfc = ensemble.RandomForestClassifier(n_estimators=200,
                                          max_features='sqrt',
                                          n_jobs=-1)

    v_features = features[labels != "ambiguous"]
    v_labels = labels[labels != "ambiguous"]
    v_labels = v_labels == "synaptic"
    v_labels = v_labels.astype(np.int)

    score = cross_validation.cross_val_score(rfc, v_features,
                                             v_labels, cv=10)
    print(np.mean(score), np.std(score))

    rfc.fit(v_features, v_labels)
    print(rfc.feature_importances_)

    if not os.path.exists(conn.path + "/conn_syn_rfc/"):
        os.makedirs(conn.path + "/conn_syn_rfc/")

    externals.joblib.dump(rfc, conn.path + "/conn_syn_rfc/rfc")

    return rfc, v_features, v_labels


def conn_o_features(conn_o):
    conn_o.load_attr_dict()

    features = [conn_o.size,
                conn_o.attr_dict["id_sj_ratio"][0],
                conn_o.attr_dict["id_cs_ratio"]]

    partner_ids = conn_o.lookup_in_attribute_dict("ssv_partners")
    for i_partner_id, partner_id in enumerate(partner_ids):
        features.append(conn_o.attr_dict["n_mi_objs_%d" % i_partner_id])
        features.append(conn_o.attr_dict["n_mi_vxs_%d" % i_partner_id])
        features.append(conn_o.attr_dict["n_vc_objs_%d" % i_partner_id])
        features.append(conn_o.attr_dict["n_vc_vxs_%d" % i_partner_id])

    return features


def map_objects_to_conn(wd, conn_version=None, ssd_version=None, mi_version=None,
                        vc_version=None, max_vx_dist_nm=2000,
                        max_rep_coord_dist_nm=4000, qsub_pe=None,
                        qsub_queue=None, nb_cpus=1, n_max_co_processes=100):

    conn_sd = segmentation.SegmentationDataset("conn", version=conn_version,
                                               working_dir=wd)

    multi_params = []
    for so_dir_path in conn_sd.so_dir_paths:
        multi_params.append([so_dir_path, wd, conn_version,
                             mi_version, vc_version, ssd_version,
                             max_vx_dist_nm, max_rep_coord_dist_nm])

    if qsub_pe is None and qsub_queue is None:
        results = sm.start_multiprocess(_map_objects_to_conn_thread,
                                        multi_params, nb_cpus=nb_cpus)

    elif qu.__QSUB__:
        path_to_out = qu.QSUB_script(multi_params,
                                     "map_objects_to_conn",
                                     pe=qsub_pe, queue=qsub_queue,
                                     script_folder=script_folder,
                                     n_max_co_processes=n_max_co_processes)
    else:
        raise Exception("QSUB not available")


def _map_objects_to_conn_thread(args):
    so_dir_path, wd, conn_version, mi_version, vc_version, ssd_version, \
        max_vx_dist_nm, max_rep_coord_dist_nm = args

    ssv = super_segmentation.SuperSegmentationDataset(working_dir=wd,
                                                      version=ssd_version)
    conn_sd = segmentation.SegmentationDataset(obj_type="conn",
                                               working_dir=wd,
                                               version=conn_version)
    mi_sd = segmentation.SegmentationDataset(obj_type="mi",
                                             working_dir=wd,
                                             version=mi_version)
    vc_sd = segmentation.SegmentationDataset(obj_type="vc",
                                             working_dir=wd,
                                             version=vc_version)

    this_attr_dc = AttributeDict(so_dir_path + "/attr_dict.pkl",
                                 read_only=False)

    for conn_id in this_attr_dc.keys():
        conn_o = conn_sd.get_segmentation_object(conn_id)
        conn_o.load_attr_dict()

        for k in list(conn_o.attr_dict.keys()):
            if k.startswith("n_mi_"):
                del(conn_o.attr_dict[k])
            if k.startswith("n_vc_"):
                del(conn_o.attr_dict[k])

        conn_feats = map_objects_to_single_conn(conn_o, ssv, mi_sd, vc_sd,
                                                max_vx_dist_nm=max_vx_dist_nm,
                                                max_rep_coord_dist_nm=max_rep_coord_dist_nm)

        conn_o.attr_dict.update(conn_feats)
        this_attr_dc[conn_id] = conn_o.attr_dict

    this_attr_dc.push()


def map_objects_to_single_conn(conn_o, ssv, mi_sd, vc_sd,
                               max_vx_dist_nm=2000,
                               max_rep_coord_dist_nm=4000):
    feats = {}

    partner_ids = conn_o.lookup_in_attribute_dict("ssv_partners")
    for i_partner_id, partner_id in enumerate(partner_ids):
        ssv_o = ssv.get_super_segmentation_object(partner_id)

        print(len(ssv_o.mi_ids))
        n_mi_objs, n_mi_vxs = map_objects_from_ssv(conn_o, mi_sd, ssv_o.mi_ids,
                                                   max_vx_dist_nm,
                                                   max_rep_coord_dist_nm)

        print(len(ssv_o.vc_ids))
        n_vc_objs, n_vc_vxs = map_objects_from_ssv(conn_o, vc_sd, ssv_o.vc_ids,
                                                   max_vx_dist_nm,
                                                   max_rep_coord_dist_nm)

        feats["n_mi_objs_%d" % i_partner_id] = n_mi_objs
        feats["n_mi_vxs_%d" % i_partner_id] = n_mi_vxs
        feats["n_vc_objs_%d" % i_partner_id] = n_vc_objs
        feats["n_vc_vxs_%d" % i_partner_id] = n_vc_vxs

    return feats


def map_objects_from_ssv(conn_o, obj_sd, obj_ids, max_vx_dist_nm,
                         max_rep_coord_dist_nm):
    obj_mask = np.in1d(obj_sd.ids, obj_ids)

    if np.sum(obj_mask) == 0:
        return 0, 0

    obj_rep_coords = obj_sd.load_cached_data("rep_coord")[obj_mask] * obj_sd.scaling

    obj_kdtree = spatial.cKDTree(obj_rep_coords)

    close_obj_ids = obj_sd.ids[obj_mask][obj_kdtree.query_ball_point(conn_o.rep_coord *
                                                                  conn_o.scaling,
                                                                  r=max_rep_coord_dist_nm)]

    conn_o_vx_kdtree = spatial.cKDTree(conn_o.voxel_list * conn_o.scaling)

    print(len(close_obj_ids))

    n_obj_vxs = []
    for close_obj_id in close_obj_ids:
        obj = obj_sd.get_segmentation_object(close_obj_id)
        obj_vxs = obj.voxel_list * obj.scaling

        ds, _ = conn_o_vx_kdtree.query(obj_vxs,
                                       distance_upper_bound=max_vx_dist_nm)

        n_obj_vxs.append(np.sum(ds < np.inf))

    n_obj_vxs = np.array(n_obj_vxs)

    print(n_obj_vxs)
    n_objects = np.sum(n_obj_vxs > 0)
    n_vxs = np.sum(n_obj_vxs)

    return n_objects, n_vxs


def classify_conn_objects(wd, conn_version=None, qsub_pe=None,
                          qsub_queue=None, nb_cpus=1, n_max_co_processes=100):

    conn_sd = segmentation.SegmentationDataset("conn", version=conn_version,
                                               working_dir=wd)

    multi_params = []
    for so_dir_path in conn_sd.so_dir_paths:
        multi_params.append([so_dir_path, wd, conn_version])

    if qsub_pe is None and qsub_queue is None:
        results = sm.start_multiprocess(_classify_conn_objects_thread,
                                        multi_params, nb_cpus=nb_cpus)

    elif qu.__QSUB__:
        path_to_out = qu.QSUB_script(multi_params,
                                     "classify_conn_objects",
                                     pe=qsub_pe, queue=qsub_queue,
                                     script_folder=script_folder,
                                     n_max_co_processes=n_max_co_processes)
    else:
        raise Exception("QSUB not available")


def _classify_conn_objects_thread(args):
    so_dir_path, wd, conn_version = args

    conn_sd = segmentation.SegmentationDataset(obj_type="conn",
                                               working_dir=wd,
                                               version=conn_version)
    rfc = externals.joblib.load(conn_sd.path + "/conn_syn_rfc/rfc")

    this_attr_dc = AttributeDict(so_dir_path + "/attr_dict.pkl",
                                 read_only=False)

    for conn_id in this_attr_dc.keys():
        conn_o = conn_sd.get_segmentation_object(conn_id)
        conn_o.load_attr_dict()

        feats = conn_o_features(conn_o)
        syn_prob = rfc.predict_proba([feats])[0][1]

        conn_o.attr_dict.update({"syn_prob": syn_prob})
        this_attr_dc[conn_id] = conn_o.attr_dict

    this_attr_dc.push()


def collect_axoness_from_ssv_partners(wd, conn_version=None,
                                      ssd_version=None, qsub_pe=None,
                                      qsub_queue=None, nb_cpus=1,
                                      n_max_co_processes=100):

    conn_sd = segmentation.SegmentationDataset("conn", version=conn_version,
                                               working_dir=wd)

    multi_params = []
    for so_dir_paths in chunkify(conn_sd.so_dir_paths, 4000):
        multi_params.append([so_dir_paths, wd, conn_version,
                             ssd_version])
    if qsub_pe is None and qsub_queue is None:
        results = sm.start_multiprocess_imap(_collect_axoness_from_ssv_partners_thread,
                                        multi_params, nb_cpus=nb_cpus)
    elif qu.__QSUB__:
        path_to_out = qu.QSUB_script(multi_params,
                                     "collect_axoness_from_ssv_partners",
                                     pe=qsub_pe, queue=qsub_queue,
                                     script_folder=script_folder,
                                     n_max_co_processes=n_max_co_processes)
    else:
        raise Exception("QSUB not available")


def _collect_axoness_from_ssv_partners_thread(args):
    so_dir_paths, wd, conn_version, ssd_version = args

    ssv = super_segmentation.SuperSegmentationDataset(working_dir=wd,
                                                      version=ssd_version)
    conn_sd = segmentation.SegmentationDataset(obj_type="conn",
                                               working_dir=wd,
                                               version=conn_version)
    for so_dir_path in so_dir_paths:
        this_attr_dc = AttributeDict(so_dir_path + "/attr_dict.pkl",
                                     read_only=False)

        for conn_id in this_attr_dc.keys():
            conn_o = conn_sd.get_segmentation_object(conn_id)
            conn_o.load_attr_dict()

            axoness = []
            for ssv_partner_id in conn_o.attr_dict["ssv_partners"]:
                ssv_o = ssv.get_super_segmentation_object(ssv_partner_id)
                axoness.append(ssv_o.axoness_for_coords([conn_o.rep_coord],
                                                        pred_type='axoness_preds_cnn_v2_views_avg10000')[0])

            conn_o.attr_dict.update({"partner_axoness": axoness})
            this_attr_dc[conn_id] = conn_o.attr_dict

        this_attr_dc.push()


def export_matrix(wd, conn_version=None, dest_name=None, syn_prob_t=.5):
    conn_sd = segmentation.SegmentationDataset("conn", version=conn_version,
                                               working_dir=wd)

    syn_prob = conn_sd.load_cached_data("syn_prob")

    m = syn_prob > syn_prob_t
    m_axs = conn_sd.load_cached_data("partner_axoness")[m]
    m_coords = conn_sd.rep_coords[m]
    # m_sizes = conn_sd.sizes[m]
    m_sizes = conn_sd.load_cached_data("mesh_area")[m] / 2
    m_ssv_partners = conn_sd.load_cached_data("ssv_partners")[m]
    m_syn_prob = syn_prob[m]
    m_syn_sign = conn_sd.load_cached_data("syn_sign")[m]

    m_sizes = np.multiply(m_sizes,m_syn_sign)

    table = np.concatenate([m_coords, m_ssv_partners, m_sizes[:, None], m_axs,
                            m_syn_prob[:, None]], axis=1)

    if dest_name is None:
        dest_name = conn_sd.path + "/conn_mat"

    np.savetxt(dest_name + ".csv", table, delimiter="\t", header="x\ty\tz\tssv1\tssv2\tsize\tcomp1\tcomp2\tsynprob")

    labels = np.array(["N/A", "D", "A", "S"])
    labels_ids = np.array([-1, 0, 1, 2])

    annotations = []
    m_sizes = np.abs(m_sizes)

    ms_axs = np.sort(m_axs, axis=1)
    u_axs = np.unique(ms_axs, axis=0)
    for u_ax in u_axs:
        anno = skeleton.SkeletonAnnotation()
        anno.scaling = conn_sd.scaling
        anno.comment = "%s - %s" % (labels[labels_ids == u_ax[0]][0], labels[labels_ids == u_ax[1]][0])

        for i_syn in np.where(np.sum(np.abs(ms_axs - u_ax), axis=1) == 0)[0]:
            c = m_coords[i_syn]
            # somewhat approximated from sphere volume:
            r = np.power(m_sizes[i_syn] / 3., 1 / 3.)
            #    r = m_sizes[i_syn]
            skel_node = skeleton.SkeletonNode(). \
            from_scratch(anno, c[0], c[1], c[2], radius=r)
            skel_node.data["ssv_partners"] = m_ssv_partners[i_syn]
            skel_node.data["size"] = m_sizes[i_syn]
            skel_node.data["syn_prob"] = m_syn_prob[i_syn]
            skel_node.data["sign"] = m_syn_sign[i_syn]
            anno.addNode(skel_node)
        annotations.append(anno)
    skeleton_utils.write_skeleton(dest_name + ".k.zip", annotations)

