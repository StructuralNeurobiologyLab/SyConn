# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max-Planck-Institute of Neurobiology, Munich, Germany
# Authors: Philipp Schubert, Joergen Kornfeld
from . import log_proc
from .. import global_params
from ..handler import basics
from ..mp import batchjob_utils as qu
from ..mp import mp_utils as sm
from ..proc.meshes import mesh_creator_sso
from ..reps import segmentation, super_segmentation
from ..reps.super_segmentation import SuperSegmentationObject, SuperSegmentationDataset

from typing import Iterable, Tuple
import numpy as np
from collections import Counter
from typing import Optional, List
from logging import Logger


def aggregate_segmentation_object_mappings(ssd: SuperSegmentationDataset,
                                           obj_types: List[str],
                                           n_jobs: Optional[int] = None,
                                           nb_cpus: Optional[int] = None):
    """

    Args:
        ssd: SuperSegmentationDataset
        obj_types: List[str]
        n_jobs: int
        nb_cpus: int
            cpus per job when using BatchJob or the total number
            of jobs used if single node multiprocessing.
    """
    for obj_type in obj_types:
        assert obj_type in ssd.version_dict
    assert "sv" in ssd.version_dict
    if n_jobs is None:
        n_jobs = global_params.config.ncore_total * 2

    multi_params = basics.chunkify(ssd.ssv_ids[np.argsort(ssd.load_cached_data('size'))[::-1]], n_jobs)
    multi_params = [(ssv_id_block, ssd.version, ssd.version_dict, ssd.working_dir,
                     obj_types, ssd.type) for ssv_id_block in multi_params]

    if not qu.batchjob_enabled():
        _ = sm.start_multiprocess_imap(
            _aggregate_segmentation_object_mappings_thread,
            multi_params, debug=False, nb_cpus=nb_cpus)

    else:
        _ = qu.batchjob_script(
            multi_params, "aggregate_segmentation_object_mappings",
            n_cores=nb_cpus, remove_jobfolder=True)


def _aggregate_segmentation_object_mappings_thread(args):
    ssv_obj_ids = args[0]
    version = args[1]
    version_dict = args[2]
    working_dir = args[3]
    obj_types = args[4]
    ssd_type = args[5]

    ssd = super_segmentation.SuperSegmentationDataset(working_dir, version, ssd_type=ssd_type,
                                                      version_dict=version_dict)
    ssd.load_mapping_dict()
    so_attr_of_interest = []
    # create cache for object attributes
    for obj_type in obj_types:
        so_attr_of_interest.extend([f"mapping_{obj_type}_ids", f"mapping_{obj_type}_ratios"])
    sd_cell = segmentation.SegmentationDataset('sv', config=ssd.config, cache_properties=so_attr_of_interest)

    for ssv_id in ssv_obj_ids:
        ssv = ssd.get_super_segmentation_object(ssv_id, True)
        mappings = dict((obj_type, Counter()) for obj_type in obj_types)
        for svid in ssv.sv_ids:
            sv = sd_cell.get_segmentation_object(svid)
            for obj_type in obj_types:
                if f"mapping_{obj_type}_ids" in sv.attr_dict:
                    keys = sv.attr_dict[f"mapping_{obj_type}_ids"]
                    values = sv.attr_dict[f"mapping_{obj_type}_ratios"]
                    mappings[obj_type] += Counter(dict(zip(keys, values)))
        ssv.load_attr_dict()
        for obj_type in obj_types:
            if obj_type in mappings:
                ssv.attr_dict[f"mapping_{obj_type}_ids"] = list(mappings[obj_type].keys())
                ssv.attr_dict[f"mapping_{obj_type}_ratios"] = list(mappings[obj_type].values())
        ssv.save_attr_dict()


def apply_mapping_decisions(ssd: SuperSegmentationDataset,
                            obj_types: List[str], n_jobs: Optional[int] = None,
                            nb_cpus: Optional[int] = None):
    """
    Requires prior execution of `aggregate_segmentation_object_mappings`.

    Args:
        ssd: SuperSegmentationDataset.
        obj_types:
        n_jobs:
        nb_cpus: cpus per job when using BatchJob or the total number
            of jobs used if single node multiprocessing.
    """
    for obj_type in obj_types:
        assert obj_type in ssd.version_dict
    if n_jobs is None:
        n_jobs = global_params.config.ncore_total * 2
    multi_params = basics.chunkify(ssd.ssv_ids[np.argsort(ssd.load_cached_data('size'))[::-1]], n_jobs)
    multi_params = [(ssv_id_block, ssd.version, ssd.version_dict, ssd.working_dir,
                     obj_types, ssd.type) for ssv_id_block in multi_params]

    if not qu.batchjob_enabled():
        _ = sm.start_multiprocess_imap(_apply_mapping_decisions_thread, multi_params)
    else:
        _ = qu.batchjob_script(
            multi_params, "apply_mapping_decisions", n_cores=nb_cpus, remove_jobfolder=True)


def _apply_mapping_decisions_thread(args):
    ssv_obj_ids = args[0]
    version = args[1]
    version_dict = args[2]
    working_dir = args[3]
    obj_types = args[4]
    ssd_type = args[5]

    ssd = super_segmentation.SuperSegmentationDataset(working_dir, version, ssd_type=ssd_type,
                                                      version_dict=version_dict)
    ssd.load_mapping_dict()

    cell_objects_dc = ssd.config['cell_objects']

    lower_ratio = None
    upper_ratio = None
    sizethreshold = None

    # cache size property of objects
    upper_ratios = {}
    sizethresholds = {}
    lower_ratios = {}
    sd_dc = {}
    for obj_t in obj_types:
        assert obj_t in ssd.version_dict
        sd_dc[obj_t] = segmentation.SegmentationDataset(obj_t, config=ssd.config, version=ssd.version_dict[obj_t],
                                                        cache_properties=['size'])
        if lower_ratio is None:
            try:
                lower_ratio = cell_objects_dc["lower_mapping_ratios"][obj_t]
            except KeyError:
                msg = "Lower ratio undefined."
                log_proc.critical(msg)
                raise ValueError(msg)

        if upper_ratio is None:
            try:
                upper_ratio = cell_objects_dc["upper_mapping_ratios"][obj_t]
            except KeyError:
                log_proc.error(f"Upper ratio undefined - 1. assumed.")
                upper_ratio = 1.

        if sizethreshold is None:
            try:
                sizethreshold = cell_objects_dc["sizethresholds"][obj_t]
            except KeyError:
                msg = "Size threshold undefined."
                log_proc.critical(msg)
                raise ValueError(msg)
        upper_ratios[obj_t] = upper_ratio
        lower_ratios[obj_t] = lower_ratio
        sizethresholds[obj_t] = sizethreshold

    for ssv_id in ssv_obj_ids:
        ssv = ssd.get_super_segmentation_object(ssv_id, True)
        ssv.load_attr_dict()

        for obj_type in obj_types:
            upper_ratio = upper_ratios[obj_type]
            lower_ratio = lower_ratios[obj_type]
            sizethreshold = sizethresholds[obj_type]

            if not "mapping_%s_ratios" % obj_type in ssv.attr_dict:
                msg = f"No mapping ratios found in SSV {ssv_id}."
                log_proc.error(msg)
                raise ValueError(msg)

            if not "mapping_%s_ids" % obj_type in ssv.attr_dict:
                msg = f"No mapping ids found in SSV {ssv_id}."
                log_proc.error(msg)
                raise ValueError(msg)

            obj_ratios = np.array(ssv.attr_dict[f"mapping_{obj_type}_ratios"])

            id_mask = obj_ratios > lower_ratio
            if upper_ratio < 1.:
                id_mask[obj_ratios > upper_ratio] = False

            candidate_ids = np.array(ssv.attr_dict[f"mapping_{obj_type}_ids"])[id_mask]

            ssv.attr_dict[obj_type] = []
            for candidate_id in candidate_ids:
                obj = sd_dc[obj_type].get_segmentation_object(candidate_id)
                if obj.size > sizethreshold:
                    ssv.attr_dict[obj_type].append(candidate_id)
        ssv.save_attr_dict()


def map_synssv_objects(synssv_version: Optional[str] = None,
                       log: Optional[Logger] = None,
                       nb_cpus=None, n_jobs=None,
                       syn_threshold=None):
    """
    Map syn_ssv objects and merge their meshes for all SSO objects contained in SSV SuperSegmentationDataset.

    Notes:
        * Stores meshes with keys: 'syn_ssv' and 'syn_ssv_sym', syn_ssv_asym (if synapse type is available). This may
          take a while.

    Args:
        synssv_version: String identifier.
        n_jobs: Number of jobs.
        log: Logger.
        nb_cpus: Number of cpus for local multi-processing.
        syn_threshold: Probability threshold applied during the mapping of syn_ssv objects.
    """
    if n_jobs is None:
        n_jobs = 4 * global_params.config.ncore_total
    if syn_threshold is None:
        syn_threshold = global_params.config['cell_objects']['thresh_synssv_proba']
    ssd = SuperSegmentationDataset(global_params.config.working_dir)
    multi_params = []
    for ssv_id_block in basics.chunkify(ssd.ssv_ids, n_jobs):
        multi_params.append([ssv_id_block, ssd.version, ssd.version_dict,
                             ssd.working_dir, ssd.type, synssv_version,
                             syn_threshold])

    if not qu.batchjob_enabled():
        _ = sm.start_multiprocess_imap(
            map_synssv_objects_thread,
            multi_params, nb_cpus=nb_cpus)

    else:
        _ = qu.batchjob_script(multi_params, "map_synssv_objects",
                               remove_jobfolder=True, log=log)


def map_synssv_objects_thread(args):
    ssv_obj_ids, version, version_dict, working_dir, \
    ssd_type, synssv_version, syn_threshold = args

    ssd = super_segmentation.SuperSegmentationDataset(working_dir, version,
                                                      ssd_type=ssd_type,
                                                      version_dict=version_dict)

    syn_ssv_sd = segmentation.SegmentationDataset(obj_type="syn_ssv",
                                                  working_dir=working_dir,
                                                  version=synssv_version)

    ssv_partners = syn_ssv_sd.load_cached_data("neuron_partners")
    syn_prob = syn_ssv_sd.load_cached_data("syn_prob")
    synssv_ids = syn_ssv_sd.load_cached_data("id")

    synssv_ids = synssv_ids[syn_prob > syn_threshold]
    ssv_partners = ssv_partners[syn_prob > syn_threshold]

    for ssv_id in ssv_obj_ids:
        # enable of SegmentationObjects, including their meshes -> reuse in typedsyns2mesh call
        ssv = ssd.get_super_segmentation_object(ssv_id, caching=True)
        ssv.load_attr_dict()

        curr_synssv_ids = synssv_ids[np.in1d(ssv_partners[:, 0], ssv.id)]
        curr_synssv_ids = np.concatenate([curr_synssv_ids,
                                          synssv_ids[np.in1d(ssv_partners[:, 1], ssv.id)]])
        ssv.attr_dict["syn_ssv"] = curr_synssv_ids
        ssv.save_attr_dict()
        # cache syn_ssv mesh and typed meshes if available
        ssv.load_mesh('syn_ssv')
        if global_params.config.syntype_available:
            ssv.typedsyns2mesh()


def mesh_proc_ssv(working_dir: str, version: Optional[str] = None,
                  ssd_type: str = 'ssv', nb_cpus: Optional[int] = None):
    """
    Caches the SSV meshes locally with 20 cpus in parallel.

    Args:
        working_dir: str
            Path to working directory.
        version: str
            version identifier, like 'spgt' for spine ground truth SSD. Defaults
            to the SSD of the cellular SSVs.
        ssd_type: str
            Default is 'ssv'.
        nb_cpus: int
            Default is ``cpu_count()``.

    Returns:

    """
    ssds = super_segmentation.SuperSegmentationDataset(working_dir=working_dir,
                                                       version=version,
                                                       ssd_type=ssd_type)
    sm.start_multiprocess_imap(mesh_creator_sso, list(ssds.ssvs),
                               nb_cpus=nb_cpus, debug=False)


def split_ssv(ssv: SuperSegmentationObject, splitted_sv_ids: Iterable[int]) \
        -> Tuple[SuperSegmentationObject, SuperSegmentationObject]:
    """Splits an SuperSegmentationObject into two."""

    if ssv._ssd is None:
        raise ValueError('SSV dataset has to be defined. Use "get_superseg'
                         'mentation_object" method to instantiate SSO objects,'
                         ' or assign "_dataset".')
    ssd = ssv._ssd
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


def init_ssv(ssv_id: int, sv_ids: List[int], ssd: SuperSegmentationDataset) \
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
