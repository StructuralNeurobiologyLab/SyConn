# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max-Planck-Institute of Neurobiology, Munich, Germany
# Authors: Philipp Schubert, Joergen Kornfeld
try:
    import cPickle as pkl
except ImportError:
    import pickle as pkl
from typing import Iterable, List, Tuple
import glob
import numpy as np
import os
from collections import Counter

from ..mp import qsub_utils as qu
from ..mp import mp_utils as sm
from ..reps.super_segmentation import SuperSegmentationObject, \
    SuperSegmentationDataset
from ..reps import segmentation, super_segmentation
from ..proc.meshes import mesh_creator_sso
script_folder = os.path.abspath(os.path.dirname(__file__) + "/../QSUB_scripts/")


def save_dataset_deep(ssd, extract_only=False, attr_keys=(), stride=1000,
                      qsub_pe=None, qsub_queue=None, nb_cpus=1,
                      n_max_co_processes=None):
    ssd.save_dataset_shallow()

    multi_params = []
    for ssv_id_block in [ssd.ssv_ids[i:i + stride]
                         for i in range(0, len(ssd.ssv_ids), stride)]:
        multi_params.append([ssv_id_block, ssd.version, ssd.version_dict,
                             ssd.working_dir, extract_only, attr_keys,
                             ssd.type])

    if qsub_pe is None and qsub_queue is None:
        results = sm.start_multiprocess(
            _write_super_segmentation_dataset_thread,
            multi_params, nb_cpus=nb_cpus)

    elif qu.__BATCHJOB__:
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
    ssd_type = args[6]

    ssd = super_segmentation.SuperSegmentationDataset(working_dir, version,
                                                      ssd_type=ssd_type,
                                                      version_dict=version_dict)

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
                    ssv_obj = ssd.get_super_segmentation_object(ssv_obj_id,
                                                                True)

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


def aggregate_segmentation_object_mappings(ssd, obj_types,
                                           stride=1000, qsub_pe=None,
                                           qsub_queue=None, nb_cpus=1):
    """

    Parameters
    ----------
    ssd : SuperSegmentationDataset
    obj_types : List[str]
    stride : int
    qsub_pe : Optional[str]
    qsub_queue : Optional[str]
    nb_cpus : int
    """

    for obj_type in obj_types:
        assert obj_type in ssd.version_dict
    assert "sv" in ssd.version_dict

    multi_params = []
    for ssv_id_block in [ssd.ssv_ids[i:i + stride]
                         for i in
                         range(0, len(ssd.ssv_ids), stride)]:
        multi_params.append([ssv_id_block, ssd.version, ssd.version_dict,
                             ssd.working_dir, obj_types, ssd.type])

    if qsub_pe is None and qsub_queue is None:
        results = sm.start_multiprocess(
            _aggregate_segmentation_object_mappings_thread,
            multi_params, nb_cpus=nb_cpus)

    elif qu.__BATCHJOB__:
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
    ssd_type = args[5]

    ssd = super_segmentation.SuperSegmentationDataset(working_dir, version,
                                                      ssd_type=ssd_type,
                                                      version_dict=version_dict)
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
                    list(mappings[obj_type].keys())
                ssv.attr_dict["mapping_%s_ratios" % obj_type] = \
                    list(mappings[obj_type].values())

        ssv.save_attr_dict()


def apply_mapping_decisions(ssd, obj_types, stride=1000, qsub_pe=None,
                            qsub_queue=None, nb_cpus=1):
    """

    Parameters
    ----------
    ssd : SuperSegmentationDataset
    obj_types : List[str]
    stride : int
    qsub_pe : Optional[str]
    qsub_queue : Optional[str]
    nb_cpus : int
    """
    for obj_type in obj_types:
        assert obj_type in ssd.version_dict

    multi_params = []
    for ssv_id_block in [ssd.ssv_ids[i:i + stride]
                         for i in
                         range(0, len(ssd.ssv_ids), stride)]:
        multi_params.append([ssv_id_block, ssd.version, ssd.version_dict,
                             ssd.working_dir, obj_types, ssd.type])

    if qsub_pe is None and qsub_queue is None:
        results = sm.start_multiprocess(_apply_mapping_decisions_thread,
                                        multi_params, nb_cpus=nb_cpus)

    elif qu.__BATCHJOB__:
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
    ssd_type = args[5]

    ssd = super_segmentation.SuperSegmentationDataset(working_dir, version,
                                                      ssd_type=ssd_type,
                                                      version_dict=version_dict)
    ssd.load_mapping_dict()

    for ssv_id in ssv_obj_ids:
        ssv = ssd.get_super_segmentation_object(ssv_id, True)
        ssv.load_attr_dict()

        for obj_type in obj_types:

            lower_ratio = None
            upper_ratio = None
            sizethreshold = None

            if obj_type == "sj":
                correct_for_background = True
            else:
                correct_for_background = False

            assert obj_type in ssv.version_dict

            if not "mapping_%s_ratios" % obj_type in ssv.attr_dict:
                print("No mapping ratios found")
                continue

            if not "mapping_%s_ids" % obj_type in ssv.attr_dict:
                print("no mapping ids found")
                continue

            if lower_ratio is None:
                try:
                    lower_ratio = ssv.config.entries["LowerMappingRatios"][
                        obj_type]
                except:
                    raise ("Lower ratio undefined")

            if upper_ratio is None:
                try:
                    upper_ratio = ssv.config.entries["UpperMappingRatios"][
                        obj_type]
                except:
                    print("Upper ratio undefined - 1. assumed")
                    upper_ratio = 1.

            if sizethreshold is None:
                try:
                    sizethreshold = ssv.config.entries["Sizethresholds"][
                        obj_type]
                except:
                    raise ("Size threshold undefined")

            obj_ratios = np.array(ssv.attr_dict["mapping_%s_ratios" % obj_type])

            if correct_for_background:
                for i_so_id in range(
                        len(ssv.attr_dict["mapping_%s_ids" % obj_type])):
                    so_id = ssv.attr_dict["mapping_%s_ids" % obj_type][i_so_id]
                    obj_version = ssv.config.entries["Versions"][obj_type]
                    this_so = segmentation.SegmentationObject(
                        so_id, obj_type,
                        version=obj_version,
                        scaling=ssv.scaling,
                        working_dir=ssv.working_dir)
                    this_so.load_attr_dict()

                    if 0 in this_so.attr_dict["mapping_ids"]:
                        ratio_0 = this_so.attr_dict["mapping_ratios"][
                            this_so.attr_dict["mapping_ids"] == 0][0]

                        obj_ratios[i_so_id] /= (1 - ratio_0)

            id_mask = obj_ratios > lower_ratio
            if upper_ratio < 1.:
                id_mask[obj_ratios > upper_ratio] = False

            candidate_ids = \
            np.array(ssv.attr_dict["mapping_%s_ids" % obj_type])[id_mask]

            ssv.attr_dict[obj_type] = []
            for candidate_id in candidate_ids:
                obj = segmentation.SegmentationObject(candidate_id,
                                                      obj_type=obj_type,
                                                      version=
                                                      ssv.version_dict[
                                                          obj_type],
                                                      working_dir=ssv.working_dir,
                                                      config=ssv.config)
                if obj.size > sizethreshold:
                    ssv.attr_dict[obj_type].append(candidate_id)
            ssv.save_attr_dict()


def map_synaptic_conn_objects(ssd, conn_version=None, stride=1000,
                              qsub_pe=None, qsub_queue=None, nb_cpus=1,
                              n_max_co_processes=100):

    multi_params = []
    for ssv_id_block in [ssd.ssv_ids[i:i + stride]
                         for i in range(0, len(ssd.ssv_ids), stride)]:
        multi_params.append([ssv_id_block, ssd.version, ssd.version_dict,
                             ssd.working_dir, ssd.type, conn_version])

    if qsub_pe is None and qsub_queue is None:
        results = sm.start_multiprocess(
            _map_synaptic_conn_objects_thread,
            multi_params, nb_cpus=nb_cpus)

    elif qu.__BATCHJOB__:
        path_to_out = qu.QSUB_script(multi_params,
                                     "map_synaptic_conn_objects",
                                     pe=qsub_pe, queue=qsub_queue,
                                     script_folder=script_folder,
                                     n_max_co_processes=n_max_co_processes)

    else:
        raise Exception("QSUB not available")


def _map_synaptic_conn_objects_thread(args):
    ssv_obj_ids, version, version_dict, working_dir, \
        ssd_type, conn_version = args

    ssd = super_segmentation.SuperSegmentationDataset(working_dir, version,
                                                      ssd_type=ssd_type,
                                                      version_dict=version_dict)

    conn_sd = segmentation.SegmentationDataset(obj_type="conn",
                                               working_dir=working_dir,
                                               version=conn_version)

    ssv_partners = conn_sd.load_cached_data("ssv_partners")
    syn_prob = conn_sd.load_cached_data("syn_prob")
    conn_ids = conn_sd.load_cached_data("id")

    conn_ids = conn_ids[syn_prob > .5]
    ssv_partners = ssv_partners[syn_prob > .5]

    for ssv_id in ssv_obj_ids:
        ssv = ssd.get_super_segmentation_object(ssv_id, False)
        ssv.load_attr_dict()

        ssv_conn_ids = conn_ids[np.in1d(ssv_partners[:, 0], ssv.id)]
        ssv_conn_ids = np.concatenate([ssv_conn_ids,
                                       conn_ids[np.in1d(ssv_partners[:, 1], ssv.id)]])

        ssv.attr_dict["conn_ids"] = ssv_conn_ids
        ssv.save_attr_dict()


def mesh_proc_ssv(working_dir, version=None, ssd_type='ssv', nb_cpus=20):
    """
    Caches the SSV meshes locally with 20 cpus in parallel.

    Parameters
    ----------
    working_dir : str
        Path to working directory.
    version : str
        version identifier, like 'spgt' for spine ground truth SSD. Defaults
        to the SSD of the cellular SSVs.
    ssd_type : str
        Default is 'ssv'
    nb_cpus : int
        Default is 20.
    """
    ssds = super_segmentation.SuperSegmentationDataset(working_dir=working_dir,
                                                       version=version,
                                                       ssd_type=ssd_type)
    sm.start_multiprocess_imap(mesh_creator_sso, list(ssds.ssvs),
                               nb_cpus=nb_cpus, debug=False)


def split_ssv(ssv: SuperSegmentationObject, splitted_sv_ids: Iterable[int])\
        -> Tuple[SuperSegmentationObject, SuperSegmentationObject]:
    """Splits an SuperSegmentationObject into two."""

    if ssv._dataset is None:
        raise ValueError('SSV dataset has to be defined. Use "get_superseg'
                         'mentation_object" method to instantiate SSO objects,'
                         ' or assign "_dataset" yourself accordingly.')
    ssd = ssv._dataset
    orig_ids = set(ssv.sv_ids)
    # TODO: Support ssv.rag splitting
    splitted_sv_ids = set(splitted_sv_ids)
    if splitted_sv_ids.issubset(orig_ids):
        raise ValueError('All splitted SV IDs have to be part of the SSV.')
    set1 = orig_ids.difference(set(splitted_sv_ids))
    set2 = splitted_sv_ids
    # TODO: run SSD modification methods, e.g. cached numpy arrays holding SSV attributes
    # TODO: run contactsite modification methods, e.g. change all contactsites which SSV partners contain ssv.id etc.
    # TODO: run all classification models
    new_id1, new_id2 = list(get_available_ssv_ids(ssd, n=2))
    ssv1 = init_ssv(new_id1, list(set1), ssd=ssd)
    ssv2 = init_ssv(new_id2, list(set2), ssd=ssd)
    # TODO: add ignore flag or destroy original SSV in its SSD.
    return ssv1, ssv2


def init_ssv(ssv_id: int, sv_ids: List[int], ssd: SuperSegmentationDataset)\
        -> SuperSegmentationObject:
    """Initializes an SuperSegmentationObject and caches all relevant data.
    Cell organelles and supervoxel SegmentationDatasets must be initialized."""
    ssv = SuperSegmentationObject(ssv_id, sv_ids=sv_ids, version=ssd.version,
                                  create=True, working_dir=ssd.working_dir)
    ssv.preprocess()
    return ssv


def get_available_ssv_ids(ssd, n=2):
    cnt = 0
    for ii in range(np.max(ssd.ssv_ids) + n):
        if cnt == n:
            break
        if not ii in ssd.ssv_ids:
            cnt += 1
            yield ii
