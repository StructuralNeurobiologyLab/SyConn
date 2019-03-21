# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Philipp Schubert, Joergen Kornfeld, Maria Kawula

import numpy as np
import os
from knossos_utils import knossosdataset
knossosdataset._set_noprint(True)

from ..mp import batchjob_utils as qu
from ..mp import mp_utils as sm
from ..reps import super_segmentation, segmentation
from ..backend.storage import AttributeDict, CompressedStorage
from ..handler.basics import chunkify
from .. import global_params


def collect_properties_from_ssv_partners(wd, obj_version=None, ssd_version=None,
                                         qsub_pe=None, qsub_queue=None,
                                         n_max_co_processes=None):
    """
    Collect axoness, cell types and spiness from synaptic partners and stores
    them in syn_ssv objects. Also maps syn_type_sym_ratio to the synaptic sign
    (-1 for asym., 1 for sym. synapses).
    Parameters
    ----------
    wd : str
    obj_version : str
    ssd_version : int
    qsub_pe : str
    qsub_queue : str
    n_max_co_processes : int
        Number of parallel jobs
    """

    ssd = super_segmentation.SuperSegmentationDataset(working_dir=wd,
                                                      version=ssd_version)

    multi_params = []

    for ids_small_chunk in chunkify(ssd.ssv_ids, 200):
        multi_params.append([wd, obj_version, ssd_version, ids_small_chunk])

    if (qsub_pe is None and qsub_queue is None) or not qu.batchjob_enabled():
        _ = sm.start_multiprocess_imap(
            _collect_properties_from_ssv_partners_thread, multi_params,
            nb_cpus=n_max_co_processes)
    elif qu.batchjob_enabled():
        _ = qu.QSUB_script(
            multi_params, "collect_properties_from_ssv_partners", pe=qsub_pe,
            queue=qsub_queue, script_folder=None,
            n_max_co_processes=n_max_co_processes)
    else:
        raise Exception("QSUB not available")

    # iterate over paths with syn
    sd_syn_ssv = segmentation.SegmentationDataset("syn_ssv", working_dir=wd,
                                                  version=obj_version)

    multi_params = []
    for so_dir_paths in chunkify(sd_syn_ssv.so_dir_paths, 2000):
        multi_params.append([so_dir_paths, wd, obj_version,
                             ssd_version])
    if (qsub_pe is None and qsub_queue is None) or not qu.batchjob_enabled():
        _ = sm.start_multiprocess_imap(
            _from_cell_to_syn_dict, multi_params,
            nb_cpus=n_max_co_processes)
    elif qu.batchjob_enabled():
        _ = qu.QSUB_script(
            multi_params, "from_cell_to_syn_dict", pe=qsub_pe,
            queue=qsub_queue, script_folder=None,
            n_max_co_processes=n_max_co_processes)
    else:
        raise Exception("QSUB not available")

    # delete cache_dc
    _delete_all_cache_dc(ssd)


def _collect_properties_from_ssv_partners_thread(args):
    """
    Helper function of 'collect_properties_from_ssv_partners'.

    Parameters
    ----------
    args : Tuple
        see 'collect_properties_from_ssv_partners'
    """
    wd, obj_version, ssd_version, ssv_ids = args

    sd_syn_ssv = segmentation.SegmentationDataset(obj_type="syn_ssv",
                                                  working_dir=wd,
                                                  version=obj_version)
    ssd = super_segmentation.SuperSegmentationDataset(working_dir=wd,
                                                      version=ssd_version)

    syn_neuronpartners = sd_syn_ssv.load_cached_data("neuron_partners")
    for ssv_id in ssv_ids:  # Iterate over cells
        ssv_o = ssd.get_super_segmentation_object(ssv_id)
        ssv_o.load_attr_dict()
        cache_dc = CompressedStorage(ssv_o.ssv_dir + "/cache_syn.pkl",
                                     read_only=False, disable_locking=True)

        curr_ssv_mask = (syn_neuronpartners[:, 0] == ssv_id) | \
                        (syn_neuronpartners[:, 1] == ssv_id)
        ssv_synids = sd_syn_ssv.ids[curr_ssv_mask]
        ssv_syncoords = sd_syn_ssv.rep_coords[curr_ssv_mask]

        try:
            ct = ssv_o.attr_dict['celltype_cnn']
        except KeyError:
            ct = -1
        celltypes = [ct] * len(ssv_synids)

        curr_ax, latent_morph = ssv_o.attr_for_coords(
            ssv_syncoords, attr_keys=['axoness_avg10000', 'latent_morph'])

        curr_sp = ssv_o.semseg_for_coords(ssv_syncoords, 'spiness')

        cache_dc['partner_axoness'] = np.array(curr_ax)
        cache_dc['synssv_ids'] = np.array(ssv_synids)
        cache_dc['partner_spiness'] = np.array(curr_sp)
        cache_dc['partner_celltypes'] = np.array(celltypes)
        cache_dc['latent_morph'] = np.array(latent_morph)

        cache_dc.push()


def _from_cell_to_syn_dict(args):
    """
    args : Tuple
        see 'collect_properties_from_ssv_partners'
    """
    so_dir_paths, wd, obj_version, ssd_version = args

    ssd = super_segmentation.SuperSegmentationDataset(working_dir=wd,
                                                      version=ssd_version)
    sd_syn_ssv = segmentation.SegmentationDataset(obj_type="syn_ssv",
                                                  working_dir=wd,
                                                  version=obj_version)

    for so_dir_path in so_dir_paths:
        this_attr_dc = AttributeDict(so_dir_path + "/attr_dict.pkl",
                                     read_only=False, disable_locking=True)
        for synssv_id in this_attr_dc.keys():
            synssv_o = sd_syn_ssv.get_segmentation_object(synssv_id)
            synssv_o.load_attr_dict()

            sym_asym_ratio = synssv_o.attr_dict['syn_type_sym_ratio']
            syn_sign = -1 if sym_asym_ratio > global_params.sym_thresh else 1

            axoness = []
            latent_morph = []
            spiness = []
            celltypes = []

            for ssv_partner_id in synssv_o.attr_dict["neuron_partners"]:
                ssv_o = ssd.get_super_segmentation_object(ssv_partner_id)
                ssv_o.load_attr_dict()
                cache_dc = CompressedStorage(ssv_o.ssv_dir + "/cache_syn.pkl")

                index = np.transpose(np.nonzero(cache_dc['synssv_ids'] ==
                                                synssv_id))
                if len(index) != 1:
                    msg = "useful error message"
                    raise ValueError(msg)
                index = index[0][0]
                axoness.append(cache_dc['partner_axoness'][index])
                spiness.append(cache_dc['partner_spiness'][index])
                celltypes.append(cache_dc['partner_celltypes'][index])
                latent_morph.append(cache_dc['latent_morph'][index])

            synssv_o.attr_dict.update({'partner_axoness': axoness,
                                       'partner_spiness': spiness,
                                       'partner_celltypes': celltypes,
                                       'syn_sign': syn_sign,
                                       'latent_morph': latent_morph})
            this_attr_dc[synssv_id] = synssv_o.attr_dict
        this_attr_dc.push()


def _delete_all_cache_dc(ssd):
    for ssv_o in ssd.ssvs:  # Iterate over cells
        ssv_o.load_attr_dict()
        if os.path.exists(ssv_o.ssv_dir + "/cache_syn.pkl"):
            os.remove(ssv_o.ssv_dir + "/cache_syn.pkl")
