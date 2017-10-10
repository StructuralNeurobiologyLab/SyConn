from collections import defaultdict, Counter
import cPickle as pkl
import glob
import numpy as np
import os
import scipy

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
                         xrange(0, len(ssd.ssv_ids), stride)]:
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
                         xrange(0, len(ssd.ssv_ids), stride)]:
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