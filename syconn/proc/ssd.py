from collections import defaultdict, Counter
import cPickle as pkl
import glob
from multiprocessing.pool import ThreadPool

import numpy as np
import os
import scipy
from knossos_utils import knossosdataset
from ..mp import shared_mem as sm, qsub_utils as qu
from ..reps import super_segmentation as ss, \
    super_segmentation_helper as ssh, segmentation

from ..reps import super_segmentation as ss

from ..mp import qsub_utils as qu
from ..mp import shared_mem as sm
script_folder = os.path.abspath(os.path.dirname(__file__) + "/QSUB_scripts/")


def aggregate_segmentation_object_mappings(ssd, obj_types,
                                           stride=1000, qsub_pe=None,
                                           qsub_queue=None, nb_cpus=1):
    for obj_type in obj_types:
        assert obj_type in ssd.version_dict
    assert "sv" in ssd.version_dict

    multi_params = []
    for ssv_id_block in [ssd.ssv_ids[i:i + stride]
                         for i in
                         range(0, len(ssd.ssv_ids), stride)]:
        multi_params.append([ssv_id_block, ssd.version, ssd.version_dict,
                             ssd.working_dir, obj_types])

    if qsub_pe is None and qsub_queue is None:
        results = sm.start_multiprocess(
            _aggregate_segmentation_object_mappings_thread,
            multi_params, nb_cpus=nb_cpus)

    elif qu.__QSUB__:
        path_to_out = qu.QSUB_script(multi_params,
                                     "aggregate_segmentation_object_mappings",
                                     pe=qsub_pe, queue=qsub_queue,
                                     script_folder=script_folder)

    else:
        raise Exception("QSUB not available")


def _aggregate_segmentation_object_mappings_thread(args):
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


def apply_mapping_decisions(ssd, obj_types, stride=1000, qsub_pe=None,
                            qsub_queue=None, nb_cpus=1):
    for obj_type in obj_types:
        assert obj_type in ssd.version_dict

    multi_params = []
    for ssv_id_block in [ssd.ssv_ids[i:i + stride]
                         for i in
                         range(0, len(ssd.ssv_ids), stride)]:
        multi_params.append([ssv_id_block, ssd.version, ssd.version_dict,
                             ssd.working_dir, obj_types])

    if qsub_pe is None and qsub_queue is None:
        results = sm.start_multiprocess(_apply_mapping_decisions_thread,
                                        multi_params, nb_cpus=nb_cpus)

    elif qu.__QSUB__:
        path_to_out = qu.QSUB_script(multi_params,
                                     "apply_mapping_decisions",
                                     pe=qsub_pe, queue=qsub_queue,
                                     script_folder=script_folder)

    else:
        raise Exception("QSUB not available")


def _apply_mapping_decisions_thread(args):
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


def calculate_skeleton(sso, size_threshold=1e20, kd=None,
                       coord_scaling=(8, 8, 4), plain=False, cleanup=True,
                       nb_threads=1):

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
                ssh.reskeletonize_plain(vx, coord_scaling=coord_scaling)
            nodes = np.array(nodes, dtype=np.int) + sso.bounding_box[0]
        else:
            nodes, edges, diameters = \
                ssh.reskeletonize_chunked(sso.id, sso.shape,
                                      sso.bounding_box[0],
                                      sso.scaling,
                                      voxels=vx,
                                      coord_scaling=coord_scaling,
                                      nb_threads=nb_threads)

    elif kd is not None:
        nodes, edges, diameters = \
            ssh.reskeletonize_chunked(sso.id, sso.shape,
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
                sso.skeleton = ssh.cleanup_skeleton(sso.skeleton,
                                                    coord_scaling)


def save_dataset_deep(ssd, extract_only=False, attr_keys=(), stride=1000,
                      qsub_pe=None, qsub_queue=None, nb_cpus=1,
                      n_max_co_processes=None):
    ssd.save_dataset_shallow()

    multi_params = []
    for ssv_id_block in [ssd.ssv_ids[i:i + stride]
                         for i in range(0, len(ssd.ssv_ids), stride)]:
        multi_params.append([ssv_id_block, ssd.version, ssd.version_dict,
                             ssd.working_dir, extract_only, attr_keys])

    if qsub_pe is None and qsub_queue is None:
        results = sm.start_multiprocess(
            _write_super_segmentation_dataset_thread,
            multi_params, nb_cpus=nb_cpus)

    elif qu.__QSUB__:
        path_to_out = qu.QSUB_script(multi_params,
                                     "write_super_segmentation_dataset",
                                     pe=qsub_pe, queue=qsub_queue,
                                     script_folder=script_folder,
                                     n_cores=nb_cpus,
                                     n_max_co_processes=n_max_co_processes)

        out_files = glob.glob(path_to_out + "/*")
        results = []
        for out_file in out_files:
            with open(out_file) as f:
                results.append(pkl.load(f))
    else:
        raise Exception("QSUB not available")

    attr_dict = {}
    for this_attr_dict in results:
        for attribute in this_attr_dict.keys():
            if not attribute in attr_dict:
                attr_dict[attribute] = []

            attr_dict[attribute] += this_attr_dict[attribute]

    if not ssd.mapping_dict_exists:
        ssd.mapping_dict = dict(zip(attr_dict["id"], attr_dict["sv"]))
        ssd.save_dataset_shallow()

    for attribute in attr_dict.keys():
        if extract_only:
            np.save(ssd.path + "/%ss_sel.npy" % attribute,
                    attr_dict[attribute])
        else:
            np.save(ssd.path + "/%ss.npy" % attribute,
                    attr_dict[attribute])


def _write_super_segmentation_dataset_thread(args):
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
        print(ssv_obj_id)
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


def export_to_knossosdataset(ssd, kd, stride=1000, qsub_pe=None,
                             qsub_queue=None, nb_cpus=10):
    multi_params = []
    for ssv_id_block in [ssd.ssv_ids[i:i + stride]
                         for i in range(0, len(ssd.ssv_ids), stride)]:
        multi_params.append([ssv_id_block, ssd.version, ssd.version_dict,
                             ssd.working_dir, kd.knossos_path, nb_cpus])

    if qsub_pe is None and qsub_queue is None:
        results = sm.start_multiprocess(_export_to_knossosdataset_thread,
                                        multi_params, nb_cpus=nb_cpus)

    elif qu.__QSUB__:
        path_to_out = qu.QSUB_script(multi_params,
                                     "export_to_knossosdataset",
                                     pe=qsub_pe, queue=qsub_queue,
                                     script_folder=script_folder)

    else:
        raise Exception("QSUB not available")


def _export_to_knossosdataset_thread(args):
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
        print(ssv_obj_id)

        ssv_obj = ssd.get_super_segmentation_object(ssv_obj_id, True)

        offset = ssv_obj.bounding_box[0]
        if not 0 in offset:
            kd.from_matrix_to_cubes(ssv_obj.bounding_booffset,
                                    data=ssv_obj.voxels.astype(np.uint32) *
                                         ssv_obj_id,
                                    overwrite=False,
                                    nb_threads=nb_threads)


def convert_knossosdataset(ssd, sv_kd_path, ssv_kd_path,
                           stride=256, qsub_pe=None, qsub_queue=None,
                           nb_cpus=None):
    ssd.save_dataset_shallow()
    sv_kd = knossosdataset.KnossosDataset()
    sv_kd.initialize_from_knossos_path(sv_kd_path)

    if not os.path.exists(ssv_kd_path):
        ssv_kd = knossosdataset.KnossosDataset()
        ssv_kd.initialize_without_conf(ssv_kd_path, sv_kd.boundary,
                                       sv_kd.scale,
                                       sv_kd.experiment_name,
                                       mags=[1])

    size = np.ones(3, dtype=np.int) * stride
    multi_params = []
    offsets = []
    for x in range(0, sv_kd.boundary[0], stride):
        for y in range(0, sv_kd.boundary[1], stride):
            for z in range(0, sv_kd.boundary[2], stride):
                offsets.append([x, y, z])
                if len(offsets) >= 20:
                    multi_params.append([ssd.version, ssd.version_dict,
                                         ssd.working_dir, nb_cpus,
                                         sv_kd_path, ssv_kd_path, offsets,
                                         size])
                    offsets = []

    if len(offsets) > 0:
        multi_params.append([ssd.version, ssd.version_dict,
                             ssd.working_dir, nb_cpus,
                             sv_kd_path, ssv_kd_path, offsets,
                             size])

    if qsub_pe is None and qsub_queue is None:
        results = sm.start_multiprocess(_convert_knossosdataset_thread,
                                        multi_params, nb_cpus=nb_cpus)

    elif qu.__QSUB__:
        path_to_out = qu.QSUB_script(multi_params,
                                     "convert_knossosdataset",
                                     pe=qsub_pe, queue=qsub_queue,
                                     script_folder=script_folder)

    else:
        raise Exception("QSUB not available")


def _convert_knossosdataset_thread(args):
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


def export_skeletons(ssd, obj_types, apply_mapping=True, stride=1000,
                     qsub_pe=None, qsub_queue=None, nb_cpus=1):
    multi_params = []
    for ssv_id_block in [ssd.ssv_ids[i:i + stride]
                         for i in
                         range(0, len(ssd.ssv_ids), stride)]:
        multi_params.append([ssv_id_block, ssd.version, ssd.version_dict,
                             ssd.working_dir, obj_types, apply_mapping])
    # TODO @Sven: which function is requiered here? I changed it from _export_skeletons to ssh.export_skeletons
    if qsub_pe is None and qsub_queue is None:
        results = sm.start_multiprocess(
            ssh.export_skeletons,
            multi_params, nb_cpus=nb_cpus)
        no_skel_cnt = np.sum(results)

    elif qu.__QSUB__:
        path_to_out = qu.QSUB_script(multi_params,
                                     "export_skeletons",
                                     n_cores=nb_cpus,
                                     pe=qsub_pe, queue=qsub_queue,
                                     script_folder=script_folder)
        out_files = glob.glob(path_to_out + "/*")
        no_skel_cnt = 0
        for out_file in out_files:
            with open(out_file) as f:
                no_skel_cnt += np.sum(pkl.load(f))

    else:
        raise Exception("QSUB not available")

    print("N no skeletons: %d" % no_skel_cnt)


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