# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Philipp Schubert, Joergen Kornfeld

from collections import defaultdict
import numpy as np
from logging import Logger
import os
from scipy import spatial
from sklearn import ensemble
from sklearn.model_selection import cross_val_score
from knossos_utils import knossosdataset, skeleton_utils, skeleton
from typing import Union, Optional, Dict, List, Callable, TYPE_CHECKING, Tuple
knossosdataset._set_noprint(True)
import time
import joblib
import datetime
import tqdm
import shutil
import pickle as pkl

from ..handler.basics import kd_factory, chunkify
from ..mp import batchjob_utils as qu
from ..mp import mp_utils as sm
from ..reps import super_segmentation, segmentation, connectivity_helper as ch
from ..reps import segmentation_helper as seghelp
from ..reps.rep_helper import subfold_from_ix, ix_from_subfold, get_unique_subfold_ixs
from ..backend.storage import AttributeDict, VoxelStorage, CompressedStorage, MeshStorage
from ..handler.config import initialize_logging
from . import log_extraction
from .. import global_params


def collect_properties_from_ssv_partners(wd, obj_version=None, ssd_version=None, debug=False):
    """
    Collect axoness, cell types and spiness from synaptic partners and stores
    them in syn_ssv objects. Also maps syn_type_sym_ratio to the synaptic sign
    (-1 for asym., 1 for sym. synapses).

    The following keys will be available in the ``attr_dict`` of ``syn_ssv``
    typed :class:`~syconn.reps.segmentation.SegmentationObject`:
        * 'partner_axoness': Cell compartment type (axon: 1, dendrite: 0, soma: 2,
          en-passant bouton: 3, terminal bouton: 4) of the partner neurons.
        * 'partner_spiness': Spine compartment predictions (0: dendritic shaft,
          1: spine head, 2: spine neck, 3: other) of both neurons.
        * 'partner_celltypes': Celltype of the both neurons.
        * 'latent_morph': Local morphology embeddings of the pre- and post-
          synaptic partners.

    Parameters
    ----------
    wd : str
    obj_version : str
    ssd_version : str
        Number of parallel jobs
    debug : bool
    """

    ssd = super_segmentation.SuperSegmentationDataset(working_dir=wd,
                                                      version=ssd_version)

    multi_params = []

    for ids_small_chunk in chunkify(ssd.ssv_ids, global_params.config.ncore_total):
        multi_params.append([wd, obj_version, ssd_version, ids_small_chunk])

    if not qu.batchjob_enabled():
        _ = sm.start_multiprocess_imap(
            _collect_properties_from_ssv_partners_thread, multi_params,
            debug=debug)
    else:
        _ = qu.batchjob_script(
            multi_params, "collect_properties_from_ssv_partners",
            remove_jobfolder=True)

    # iterate over paths with syn
    sd_syn_ssv = segmentation.SegmentationDataset("syn_ssv", working_dir=wd,
                                                  version=obj_version)

    multi_params = []
    for so_dir_paths in chunkify(sd_syn_ssv.so_dir_paths, global_params.config.ncore_total):
        multi_params.append([so_dir_paths, wd, obj_version,
                             ssd_version])
    if not qu.batchjob_enabled():
        _ = sm.start_multiprocess_imap(
            _from_cell_to_syn_dict, multi_params,
            debug=debug)
    else:
        _ = qu.batchjob_script(
            multi_params, "from_cell_to_syn_dict", remove_jobfolder=True)
    log_extraction.debug('Deleting cache dictionaries now.')
    # delete cache_dc
    sm.start_multiprocess_imap(_delete_all_cache_dc, (ssd.ssv_ids, ssd.working_dir),
                               nb_cpus=global_params.config['ncores_per_node'])
    log_extraction.debug('Deleted all cache dictionaries.')


def _collect_properties_from_ssv_partners_thread(args):
    """
    Helper function of 'collect_properties_from_ssv_partners'.

    Parameters
    ----------
    args : Tuple
        see 'collect_properties_from_ssv_partners'
    """
    wd, obj_version, ssd_version, ssv_ids = args

    semseg2coords_kwargs = global_params.config['spines']['semseg2coords_spines']

    sd_syn_ssv = segmentation.SegmentationDataset(obj_type="syn_ssv",
                                                  working_dir=wd,
                                                  version=obj_version)
    ssd = super_segmentation.SuperSegmentationDataset(working_dir=wd,
                                                      version=ssd_version)

    syn_neuronpartners = sd_syn_ssv.load_cached_data("neuron_partners")
    pred_key_ax = "{}_avg{}".format(global_params.config['compartments'][
                                        'view_properties_semsegax']['semseg_key'],
                                    global_params.config['compartments'][
                                        'dist_axoness_averaging'])
    for ssv_id in ssv_ids:  # Iterate over cells
        ssv_o = ssd.get_super_segmentation_object(ssv_id)
        ssv_o.load_attr_dict()
        cache_dc = CompressedStorage(ssv_o.ssv_dir + "/cache_syn.pkl",
                                     read_only=False, disable_locking=True)

        curr_ssv_mask = (syn_neuronpartners[:, 0] == ssv_id) | \
                        (syn_neuronpartners[:, 1] == ssv_id)
        ssv_synids = sd_syn_ssv.ids[curr_ssv_mask]
        if len(ssv_synids) == 0:
            cache_dc['partner_spineheadvol'] = np.zeros((0, ), dtype=np.float)
            cache_dc['partner_axoness'] = np.zeros((0, ), dtype=np.int)
            cache_dc['synssv_ids'] = ssv_synids
            cache_dc['partner_spiness'] = np.zeros((0, ), dtype=np.int)
            cache_dc['partner_celltypes'] = np.zeros((0, ), dtype=np.int)
            cache_dc['latent_morph'] = np.zeros((0, ), dtype=np.float)
            cache_dc.push()
            continue
        ssv_syncoords = sd_syn_ssv.rep_coords[curr_ssv_mask]

        try:
            ct = ssv_o.attr_dict['celltype_cnn_e3']  # TODO: add keyword to global_params.py
        except KeyError:
            ct = -1
        celltypes = [ct] * len(ssv_synids)

        curr_ax, latent_morph = ssv_o.attr_for_coords(
            ssv_syncoords, attr_keys=[pred_key_ax, 'latent_morph'])

        curr_sp = ssv_o.semseg_for_coords(ssv_syncoords, 'spiness', **semseg2coords_kwargs)
        sh_vol = ssv_o.attr_for_coords(ssv_syncoords, attr_keys=['spinehead_vol'], k=2)[0]
        if len(ssv_o.skeleton['nodes']) > 1:
            # if only one skeleton node, sh_vol only contains one element per location
            sh_vol = np.max(sh_vol, axis=1)
        # # This should be reported during spine head volume calculation.
        # sh_vol_zero = (sh_vol == 0) & (curr_sp == 1) & (curr_ax == 0)
        # if np.any(sh_vol_zero):
        #     log_extraction.warn(f'Empty spinehead volume at {ssv_syncoords[sh_vol_zero]}'
        #                         f' in SSO {ssv_id}.')
        if np.any(sh_vol == -1):
            log_extraction.warn(f'No spinehead volume at {ssv_syncoords[sh_vol == -1]}'
                                f' in SSO {ssv_id}.')

        cache_dc['partner_spineheadvol'] = np.array(sh_vol)
        cache_dc['partner_axoness'] = curr_ax
        cache_dc['synssv_ids'] = ssv_synids
        cache_dc['partner_spiness'] = curr_sp
        cache_dc['partner_celltypes'] = np.array(celltypes)
        cache_dc['latent_morph'] = latent_morph
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
    cell_obj_conf = global_params.config['cell_objects']
    for so_dir_path in so_dir_paths:
        this_attr_dc = AttributeDict(so_dir_path + "/attr_dict.pkl",
                                     read_only=False, disable_locking=True)
        for synssv_id in this_attr_dc.keys():
            synssv_o = sd_syn_ssv.get_segmentation_object(synssv_id)
            synssv_o.load_attr_dict()

            sym_asym_ratio = synssv_o.attr_dict['syn_type_sym_ratio']
            syn_sign = -1 if sym_asym_ratio > cell_obj_conf['sym_thresh'] else 1

            axoness = []
            latent_morph = []
            spinehead_vol = []
            spiness = []
            celltypes = []

            for ssv_partner_id in synssv_o.attr_dict["neuron_partners"]:
                ssv_o = ssd.get_super_segmentation_object(ssv_partner_id)
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
                spinehead_vol.append(cache_dc['partner_spineheadvol'][index])

            synssv_o.attr_dict.update({'partner_axoness': axoness,
                                       'partner_spiness': spiness,
                                       'partner_celltypes': celltypes,
                                       'partner_spineheadvol': spinehead_vol,
                                       'syn_sign': syn_sign,
                                       'latent_morph': latent_morph})
            this_attr_dc[synssv_id] = synssv_o.attr_dict
        this_attr_dc.push()


def _delete_all_cache_dc(args):
    ssv_id, working_dir = args
    ssv_o = super_segmentation.SuperSegmentationObject(ssv_id, working_dir=working_dir)
    if os.path.exists(ssv_o.ssv_dir + "/cache_syn.pkl"):
        os.remove(ssv_o.ssv_dir + "/cache_syn.pkl")


# code for splitting 'syn' objects, which are generated as overlap between CS and SJ, see below.
def filter_relevant_syn(sd_syn, ssd):
    """
    This function filters (likely ;) ) the intra-ssv contact
    sites (inside of an ssv, not between ssvs) that do not need to be agglomerated.

    Parameters
    ----------
    sd_syn : SegmentationDataset
    ssd : SuperSegmentationDataset

    Returns
    -------
    Dict[list]
        lookup from SSV-wide synapses to SV syn. objects, keys: SSV syn ID;
        values: List of SV syn IDs.
    """
    # get all cs IDs belonging to syn objects and then retrieve corresponding
    # SVs IDs via bit shift

    syn_cs_ids = sd_syn.load_cached_data('cs_id')
    sv_ids = ch.sv_id_to_partner_ids_vec(syn_cs_ids)

    syn_ids = sd_syn.ids.copy()
    # this might mean that all syn between svs with IDs>max(np.uint32) are discarded
    sv_ids[sv_ids >= len(ssd.id_changer)] = -1
    mapped_sv_ids = ssd.id_changer[sv_ids]

    mask = np.all(mapped_sv_ids > 0, axis=1)
    syn_ids = syn_ids[mask]
    filtered_mapped_sv_ids = mapped_sv_ids[mask]

    # this identifies all inter-ssv contact sites
    mask = filtered_mapped_sv_ids[:, 0] - filtered_mapped_sv_ids[:, 1] != 0
    syn_ids = syn_ids[mask]
    relevant_syns = filtered_mapped_sv_ids[mask]

    relevant_synssv_ids = np.left_shift(np.max(relevant_syns, axis=1), 32) + \
                          np.min(relevant_syns, axis=1)

    # create lookup from SSV-wide synapses to SV syn. objects
    rel_synssv_to_syn_ids = defaultdict(list)
    for i_entry in range(len(relevant_synssv_ids)):
        rel_synssv_to_syn_ids[relevant_synssv_ids[i_entry]].\
            append(syn_ids[i_entry])

    return rel_synssv_to_syn_ids


def combine_and_split_syn(wd, cs_gap_nm=300, ssd_version=None, syn_version=None,
                          nb_cpus=None, n_folders_fs=10000, log=None, overwrite=False):
    """
    Creates 'syn_ssv' objects from 'syn' objects. Therefore, computes connected
    syn-objects on SSV level and aggregates the respective 'syn' attributes
    ['cs_id', 'id_cs_ratio', 'cs_size']. This method requires the execution of
    'syn_gen_via_cset' (or equivalent) beforehand.

    All objects of the resulting 'syn_ssv' SegmentationDataset contain the
    following attributes:
    ['syn_sign', 'syn_type_sym_ratio', 'asym_prop', 'sym_prop', 'cs_ids',
    'cs_size', 'id_cs_ratio', 'neuron_partners']

    Parameters
    ----------
    wd :
    cs_gap_nm :
    ssd_version :
    syn_version :
    nb_cpus :
    log:
    n_folders_fs:
    overwrite:

    """
    ssd = super_segmentation.SuperSegmentationDataset(wd, version=ssd_version)
    syn_sd = segmentation.SegmentationDataset("syn", working_dir=wd,
                                              version=syn_version)
    # TODO: this procedure creates folders with single and double digits, e.g. '0' and '00'. Single digit folders are not
    #  used during write-outs, they are probably generated within this method's makedirs
    rel_synssv_to_syn_ids = filter_relevant_syn(syn_sd, ssd)
    storage_location_ids = get_unique_subfold_ixs(n_folders_fs)

    n_used_paths = min(global_params.config.ncore_total * 10, len(storage_location_ids),
                       len(rel_synssv_to_syn_ids))
    voxel_rel_paths = chunkify([subfold_from_ix(ix, n_folders_fs) for ix in storage_location_ids],
                               n_used_paths)
    # target SD for SSV syn objects
    sd_syn_ssv = segmentation.SegmentationDataset("syn_ssv", working_dir=wd,
                                                  version="0", create=False,
                                                  n_folders_fs=n_folders_fs)
    if os.path.exists(sd_syn_ssv.so_storage_path):
        if not overwrite:
            raise FileExistsError(f'"{sd_syn_ssv.so_storage_path}" already exists, but '
                                  f'overwrite was set to False.')
        shutil.rmtree(sd_syn_ssv.so_storage_path)

    # prepare folder structure
    voxel_rel_paths_2stage = np.unique([subfold_from_ix(ix, n_folders_fs)[:-2]
                                        for ix in storage_location_ids])
    for p in voxel_rel_paths_2stage:
        os.makedirs(sd_syn_ssv.so_storage_path + p)

    # TODO: apply weighting-scheme to balance worker load
    rel_synssv_to_syn_ids_items = list(rel_synssv_to_syn_ids.items())

    rel_synssv_to_syn_ids_items_chunked = chunkify(rel_synssv_to_syn_ids_items, n_used_paths)
    multi_params = [(wd, rel_synssv_to_syn_ids_items_chunked[ii], voxel_rel_paths[ii],
                    syn_sd.version, sd_syn_ssv.version, ssd.scaling, cs_gap_nm) for
                    ii in range(n_used_paths)]
    if not qu.batchjob_enabled():
        _ = sm.start_multiprocess_imap(_combine_and_split_syn_thread,
                                       multi_params, nb_cpus=nb_cpus, debug=False)
    else:
        _ = qu.batchjob_script(
            multi_params, "combine_and_split_syn", remove_jobfolder=True, log=log)


def _combine_and_split_syn_thread(args):
    wd = args[0]
    rel_cs_to_cs_agg_ids_items = args[1]
    voxel_rel_paths = args[2]
    syn_version = args[3]
    syn_ssv_version = args[4]
    scaling = args[5]  # TODO: use syn scaling..
    cs_gap_nm = args[6]

    sd_syn_ssv = segmentation.SegmentationDataset("syn_ssv", working_dir=wd,
                                                  version=syn_ssv_version)
    sd_syn = segmentation.SegmentationDataset("syn", working_dir=wd,
                                              version=syn_version)

    cell_obj_cnf = global_params.config['cell_objects']
    use_new_subfold = global_params.config.use_new_subfold
    # TODO: add to config, also used in 'ix_from_subfold' if 'global_params.config.use_new_subfold=True'
    div_base = 1e3
    id_chunk_cnt = 0
    n_per_voxel_path = np.ceil(float(len(rel_cs_to_cs_agg_ids_items)) / len(voxel_rel_paths))
    n_items_for_path = 0
    cur_path_id = 0
    base_dir = sd_syn_ssv.so_storage_path + voxel_rel_paths[cur_path_id]
    os.makedirs(base_dir, exist_ok=True)
    # get ID/path to storage to save intermediate results
    base_id = ix_from_subfold(voxel_rel_paths[cur_path_id], sd_syn.n_folders_fs)
    next_id = base_id

    voxel_dc = VoxelStorage(base_dir + "/voxel.pkl", read_only=False)
    attr_dc = AttributeDict(base_dir + "/attr_dict.pkl", read_only=False)
    mesh_dc = MeshStorage(base_dir + "/mesh.pkl", read_only=False)

    for item in rel_cs_to_cs_agg_ids_items:
        n_items_for_path += 1
        ssv_ids = ch.sv_id_to_partner_ids_vec([item[0]])[0]
        syn = sd_syn.get_segmentation_object(item[1][0])

        syn.load_attr_dict()
        syn_attr_list = [syn.attr_dict]  # used to collect syn properties
        voxel_list = [syn.voxel_list]
        # store index of syn. objects for attribute dict retrieval
        synix_list = [0] * len(syn.voxel_list)
        for syn_ix, syn_id in enumerate(item[1][1:]):
            syn_object = sd_syn.get_segmentation_object(syn_id)
            syn_object.load_attr_dict()
            syn_attr_list.append(syn_object.attr_dict)
            voxel_list.append(syn_object.voxel_list)
            synix_list += [syn_ix] * len(syn_object.voxel_list)
        syn_attr_list = np.array(syn_attr_list)
        synix_list = np.array(synix_list)

        if len(synix_list) == 0:
            msg = 'Voxels not available for syn-object {}.'.format(str(syn))
            log_extraction.error(msg)
            raise ValueError(msg)

        ccs = connected_cluster_kdtree(voxel_list, dist_intra_object=cs_gap_nm,
                                       dist_inter_object=20000, scale=scaling)

        voxel_list = np.concatenate(voxel_list)

        for this_cc in ccs:
            this_cc_mask = np.array(list(this_cc))
            # retrieve the index of the syn objects selected for this CC
            this_syn_ixs, this_syn_ids_cnt = np.unique(synix_list[this_cc_mask],
                                                       return_counts=True)
            this_agg_syn_weights = this_syn_ids_cnt / np.sum(this_syn_ids_cnt)
            if np.sum(this_syn_ids_cnt) < cell_obj_cnf['min_obj_vx']['syn_ssv']:
                continue
            this_attr = syn_attr_list[this_syn_ixs]
            this_vx = voxel_list[this_cc_mask]
            abs_offset = np.min(this_vx, axis=0)
            this_vx -= abs_offset
            id_mask = np.zeros(np.max(this_vx, axis=0) + 1, dtype=np.bool)
            id_mask[this_vx[:, 0], this_vx[:, 1], this_vx[:, 2]] = True
            syn_ssv = sd_syn_ssv.get_segmentation_object(next_id)
            if (os.path.abspath(syn_ssv.attr_dict_path)
                    != os.path.abspath(base_dir + "/attr_dict.pkl")):
                raise ValueError(f'Path mis-match!')
            this_attr_dc = dict(neuron_partners=ssv_ids)
            try:
                voxel_dc[next_id] = [id_mask], [abs_offset]
                syn_ssv._voxels = syn_ssv.load_voxels(voxel_dc=voxel_dc)
                syn_ssv.calculate_rep_coord(voxel_dc=voxel_dc)
                syn_ssv.calculate_bounding_box(voxel_dc=voxel_dc)
                this_attr_dc["rep_coord"] = syn_ssv.rep_coord
                this_attr_dc["bounding_box"] = syn_ssv.bounding_box
                this_attr_dc["size"] = syn_ssv.size
                ind, vert, normals = syn_ssv._mesh_from_scratch()
                mesh_dc[syn_ssv.id] = [ind, vert, normals]
                this_attr_dc["mesh_bb"] = syn_ssv.mesh_bb
                this_attr_dc["mesh_area"] = syn_ssv.mesh_area
            except Exception as e:
                debug_out_fname = "{}/{}_{}_{}_{}.npy".format(
                    sd_syn_ssv.so_storage_path, next_id, abs_offset[0],
                    abs_offset[1], abs_offset[2])
                msg = f"Saving {syn_ssv} failed with {e}. Debug file at " \
                      f"{debug_out_fname}."
                log_extraction.error(msg)
                np.save(debug_out_fname, this_vx)
                raise ValueError(msg)
            # aggregate syn properties
            syn_props_agg = {}
            for dc in this_attr:
                for k in ['id_cs_ratio', 'cs_id', 'cs_size', 'sym_prop', 'asym_prop']:
                    syn_props_agg.setdefault(k, []).append(dc[k])
            # store cs and sj IDs
            syn_props_agg['cs_ids'] = syn_props_agg['cs_id']
            del syn_props_agg['cs_id']

            # weight cs and syn size by number of voxels present for this connected component,
            # i.e. use 'this_agg_syn_weights' as weight
            syn_props_agg['cs_size'] = np.sum(this_agg_syn_weights * np.array(syn_props_agg['cs_size']))

            # agglomerate the syn-to-cs ratio
            syn_props_agg['id_cs_ratio'] = np.sum(this_agg_syn_weights * np.array(syn_props_agg['id_cs_ratio']))

            # type weights as weighted sum of syn fragments
            sym_prop = np.sum(this_agg_syn_weights * np.array(syn_props_agg['sym_prop']))
            asym_prop = np.sum(this_agg_syn_weights * np.array(syn_props_agg['asym_prop']))
            syn_props_agg['sym_prop'] = sym_prop
            syn_props_agg['asym_prop'] = asym_prop

            if sym_prop + asym_prop == 0:
                sym_ratio = -1
            else:
                sym_ratio = sym_prop / float(asym_prop + sym_prop)
            syn_props_agg["syn_type_sym_ratio"] = sym_ratio
            syn_sign = -1 if sym_ratio > cell_obj_cnf['sym_thresh'] else 1
            syn_props_agg["syn_sign"] = syn_sign

            # add syn_ssv dict to AttributeStorage
            this_attr_dc.update(syn_props_agg)
            attr_dc[next_id] = this_attr_dc
            if use_new_subfold:
                next_id += np.uint(1)
                if next_id - base_id >= div_base:
                    # next ID chunk mapped to this storage
                    id_chunk_cnt += 1
                    old_base_id = base_id
                    base_id += np.uint(sd_syn_ssv.n_folders_fs*div_base) * id_chunk_cnt
                    assert subfold_from_ix(base_id, sd_syn_ssv.n_folders_fs, old_version=False) == \
                           subfold_from_ix(old_base_id, sd_syn_ssv.n_folders_fs, old_version=False)
                    next_id = base_id
            else:
                next_id += np.uint(sd_syn.n_folders_fs)

        if n_items_for_path > n_per_voxel_path:
            voxel_dc.push()
            attr_dc.push()
            mesh_dc.push()
            cur_path_id += 1
            n_items_for_path = 0
            id_chunk_cnt = 0
            base_id = ix_from_subfold(voxel_rel_paths[cur_path_id], sd_syn.n_folders_fs)
            next_id = base_id
            base_dir = sd_syn_ssv.so_storage_path + voxel_rel_paths[cur_path_id]
            os.makedirs(base_dir, exist_ok=True)
            voxel_dc = VoxelStorage(base_dir + "/voxel.pkl", read_only=False)
            attr_dc = AttributeDict(base_dir + "/attr_dict.pkl", read_only=False)
            mesh_dc = MeshStorage(base_dir + "/mesh.pkl", read_only=False)

    if n_items_for_path > 0:
        voxel_dc.push()
        attr_dc.push()
        mesh_dc.push()


def connected_cluster_kdtree(voxel_coords: List[np.ndarray], dist_intra_object: float,
                             dist_inter_object: float, scale: np.ndarray) -> List[set]:
    """
    Identify connected components within N objects. Two stage process: 1st stage adds edges between every
    object voxel which are at most 2 voxels apart. The edges are added to a global graph which is used to
    calculate connected components. In the 2nd stage, connected components are considered close if they are within a
    maximum distance of `dist_inter_object` between a random voxel used as their representative coordinate.
    Close connected components will then be connected if the minimum distance between any of their voxels is
    smaller than `dist_intra_object`.

    Args:
        voxel_coords: List of numpy arrays in voxel coordinates.
        dist_intra_object: Maximum distance between two voxels of different synapses to
            consider them the same object. In nm.
        dist_inter_object: Maximum distance between two objects to check for close voxels
            between them. In nm.
        scale: Voxel sizes in nm (XYZ).

    Returns:
        Connected components across all N input objects with at most `dist_intra_cluster` distance.
    """
    import networkx as nx
    graph = nx.Graph()
    ixs_offset = np.cumsum([0] + [len(syn_vxs) for syn_vxs in voxel_coords[:-1]])
    # add intra object edges
    for ii in range(len(voxel_coords)):
        off = ixs_offset[ii]
        graph.add_nodes_from(np.arange(len(voxel_coords[ii])) + off)
        kdtree = spatial.cKDTree(voxel_coords[ii])
        pairs = np.array(list(kdtree.query_pairs(r=2)), dtype=np.int)
        graph.add_edges_from(pairs + off)
    del kdtree, pairs
    voxel_coords_flat = np.concatenate(voxel_coords) * scale
    ccs = [np.array(list(cc)) for cc in nx.connected_components(graph)]
    rep_coords = np.array([voxel_coords_flat[cc[0]] for cc in ccs])
    kdtree = spatial.cKDTree(rep_coords)
    pairs = kdtree.query_pairs(r=dist_inter_object)
    del kdtree
    # add minimal inter-object edges
    for c1, c2 in pairs:
        c1_ixs = ccs[c1]
        c2_ixs = ccs[c2]
        kd1 = spatial.cKDTree(voxel_coords_flat[c1_ixs])
        dists, nn_ixs = kd1.query(voxel_coords_flat[c2_ixs], distance_upper_bound=dist_intra_object)
        if min(dists) > dist_intra_object:
            continue
        argmin = np.argmin(dists)
        ix_c1 = c1_ixs[nn_ixs[argmin]]
        ix_c2 = c2_ixs[argmin]
        graph.add_edge(ix_c1, ix_c2)
    return list(nx.connected_components(graph))


def combine_and_split_syn_old(
        wd, cs_gap_nm=300, ssd_version=None, syn_version=None, nb_cpus=None,
        n_folders_fs=10000, log=None):
    """
    Creates 'syn_ssv' objects from 'syn' objects. Therefore, computes connected
    syn-objects on SSV level and aggregates the respective 'syn' attributes
    ['sj_id', 'cs_id', 'id_sj_ratio', 'id_cs_ratio', 'background_overlap_ratio',
    'cs_size', 'sj_size_pseudo']. This method requires the execution of
    'syn_gen_via_cset' (or equivalent) beforehand.

    All objects of the resulting 'syn_ssv' SegmentationDataset contain the
    following attributes:
    ['syn_sign', 'syn_type_sym_ratio', 'sj_ids', 'cs_ids', 'id_sj_ratio',
    'id_cs_ratio', 'background_overlap_ratio', 'neuron_partners']

    Parameters
    ----------
    wd :
    cs_gap_nm :
    ssd_version :
    syn_version :
    nb_cpus :
    log:
    n_folders_fs:

    """
    ssd = super_segmentation.SuperSegmentationDataset(wd, version=ssd_version)
    syn_sd = segmentation.SegmentationDataset("syn", working_dir=wd,
                                              version=syn_version)

    rel_synssv_to_syn_ids = filter_relevant_syn(syn_sd, ssd)
    storage_location_ids = get_unique_subfold_ixs(n_folders_fs)
    voxel_rel_paths_2stage = np.unique([subfold_from_ix(ix, n_folders_fs)[:-2]
                                        for ix in storage_location_ids])

    n_used_paths = min(global_params.config.ncore_total * 10, len(storage_location_ids),
                       len(rel_synssv_to_syn_ids))
    voxel_rel_paths = chunkify([subfold_from_ix(ix, n_folders_fs) for ix in storage_location_ids],
                               n_used_paths)

    # target SD for SSV syn objects
    sd_syn_ssv = segmentation.SegmentationDataset("syn_ssv", working_dir=wd,
                                                  version="0", create=True,
                                                  n_folders_fs=n_folders_fs)
    dataset_path = sd_syn_ssv.so_storage_path
    if os.path.exists(dataset_path):
        shutil.rmtree(dataset_path)

    for p in voxel_rel_paths_2stage:
        os.makedirs(sd_syn_ssv.so_storage_path + p)

    rel_synssv_to_syn_ids_items = list(rel_synssv_to_syn_ids.items())

    rel_synssv_to_syn_ids_items_chunked = chunkify(rel_synssv_to_syn_ids_items, n_used_paths)
    multi_params = [(wd, rel_synssv_to_syn_ids_items_chunked[ii], voxel_rel_paths[ii],
                    syn_sd.version, sd_syn_ssv.version, ssd.scaling, cs_gap_nm) for
                    ii in range(n_used_paths)]
    if not qu.batchjob_enabled():
        _ = sm.start_multiprocess_imap(_combine_and_split_syn_thread_old,
                                       multi_params, nb_cpus=nb_cpus, debug=False)
    else:
        _ = qu.batchjob_script(
            multi_params, "combine_and_split_syn_old", remove_jobfolder=True, log=log)

    return sd_syn_ssv


def _combine_and_split_syn_thread_old(args):
    wd = args[0]
    rel_cs_to_cs_agg_ids_items = args[1]
    voxel_rel_paths = args[2]
    syn_version = args[3]
    syn_ssv_version = args[4]
    scaling = args[5]
    cs_gap_nm = args[6]

    sd_syn_ssv = segmentation.SegmentationDataset("syn_ssv", working_dir=wd,
                                                  version=syn_ssv_version)

    sd_syn = segmentation.SegmentationDataset("syn", working_dir=wd,
                                              version=syn_version)

    n_per_voxel_path = np.ceil(float(len(rel_cs_to_cs_agg_ids_items)) / len(voxel_rel_paths))
    if n_per_voxel_path > sd_syn.n_folders_fs:
        log_extraction.warning('Number of items per storage dict for "syn" objects'
                               ' is bigger than `segmentation.SegmentationDataset'
                               '("syn", working_dir=wd).n_folders_fs`.')

    n_items_for_path = 0
    cur_path_id = 0
    base_dir = sd_syn_ssv.so_storage_path + voxel_rel_paths[cur_path_id]
    os.makedirs(base_dir, exist_ok=True)
    voxel_dc = VoxelStorage(base_dir + "/voxel.pkl", read_only=False)
    attr_dc = AttributeDict(base_dir + "/attr_dict.pkl", read_only=False)
    # get ID/path to storage to save intermediate results
    next_id = ix_from_subfold(voxel_rel_paths[cur_path_id], sd_syn.n_folders_fs)
    for item in rel_cs_to_cs_agg_ids_items:
        n_items_for_path += 1
        ssv_ids = ch.sv_id_to_partner_ids_vec([item[0]])[0]
        syn = sd_syn.get_segmentation_object(item[1][0])
        syn.load_attr_dict()
        syn_attr_list = [syn.attr_dict]  # used to collect syn properties
        voxel_list = syn.voxel_list
        # store index of syn. objects for attribute dict retrieval
        synix_list = [0] * len(voxel_list)
        for syn_ix, syn_id in enumerate(item[1][1:]):
            syn_object = sd_syn.get_segmentation_object(syn_id)
            syn_object.load_attr_dict()
            syn_attr_list.append(syn_object.attr_dict)
            voxel_list = np.concatenate([voxel_list, syn_object.voxel_list])
            synix_list += [syn_ix] * len(syn_object.voxel_list)
        syn_attr_list = np.array(syn_attr_list)
        synix_list = np.array(synix_list)
        if len(voxel_list) == 0:
            msg = 'Voxels not available for syn-object {}.'.format(str(syn))
            log_extraction.error(msg)
            raise ValueError(msg)
        ccs = cc_large_voxel_lists(voxel_list * scaling, cs_gap_nm)
        for this_cc in ccs:
            this_cc_mask = np.array(list(this_cc))
            # retrieve the index of the syn objects selected for this CC
            this_syn_ixs, this_syn_ids_cnt = np.unique(synix_list[this_cc_mask],
                                                       return_counts=True)
            this_agg_syn_weights = this_syn_ids_cnt / np.sum(this_syn_ids_cnt)
            if np.sum(this_syn_ids_cnt) < global_params.config['cell_objects']['min_obj_vx']['syn_ssv']:
                continue
            this_attr = syn_attr_list[this_syn_ixs]
            this_vx = voxel_list[this_cc_mask]
            abs_offset = np.min(this_vx, axis=0)
            this_vx -= abs_offset
            id_mask = np.zeros(np.max(this_vx, axis=0) + 1, dtype=np.bool)
            id_mask[this_vx[:, 0], this_vx[:, 1], this_vx[:, 2]] = True

            try:
                voxel_dc[next_id] = [id_mask], [abs_offset]
            except Exception:
                debug_out_fname = "{}/{}_{}_{}_{}.npy".format(
                    sd_syn_ssv.so_storage_path, next_id, abs_offset[0],
                    abs_offset[1], abs_offset[2])
                msg = "Saving syn_ssv {} failed. Debug file at {}." \
                      "".format(item, debug_out_fname)
                log_extraction.error(msg)
                np.save(debug_out_fname, this_vx)
                raise ValueError(msg)
            # aggregate syn properties
            syn_props_agg = {}
            for dc in this_attr:
                for k in ['background_overlap_ratio', 'id_cs_ratio', 'id_sj_ratio', 'cs_id',
                          'sj_id', 'sj_size_pseudo', 'cs_size']:
                    syn_props_agg.setdefault(k, []).append(dc[k])
            # store cs and sj IDs
            syn_props_agg['sj_ids'] = syn_props_agg['sj_id']
            del syn_props_agg['sj_id']
            syn_props_agg['cs_ids'] = syn_props_agg['cs_id']
            del syn_props_agg['cs_id']
            # calculate weighted mean of sj, cs and background ratios
            syn_props_agg['sj_size_pseudo'] = this_agg_syn_weights * \
                                              np.array(syn_props_agg['sj_size_pseudo'])
            syn_props_agg['cs_size'] = this_agg_syn_weights * \
                                       np.array(syn_props_agg['cs_size'])
            syn_props_agg['id_sj_ratio'] = this_agg_syn_weights * \
                                           np.array(syn_props_agg['id_sj_ratio'])
            syn_props_agg['id_cs_ratio'] = this_agg_syn_weights * \
                                           np.array(syn_props_agg['id_cs_ratio'])
            syn_props_agg['background_overlap_ratio'] = this_agg_syn_weights * np.array(
                syn_props_agg['background_overlap_ratio'])
            sj_size_pseudo_norm = np.sum(syn_props_agg['sj_size_pseudo'])
            cs_size_norm = np.sum(syn_props_agg['cs_size'])
            sj_s_w = syn_props_agg['id_sj_ratio'] * syn_props_agg['sj_size_pseudo']
            syn_props_agg['id_sj_ratio'] = np.sum(sj_s_w) / sj_size_pseudo_norm
            back_s_w = syn_props_agg['background_overlap_ratio'] * \
                       syn_props_agg['sj_size_pseudo']
            syn_props_agg['background_overlap_ratio'] = np.sum(back_s_w) / sj_size_pseudo_norm
            cs_s_w = syn_props_agg['id_cs_ratio'] * syn_props_agg['cs_size']
            syn_props_agg['id_cs_ratio'] = np.sum(cs_s_w) / cs_size_norm

            del syn_props_agg['cs_size']
            del syn_props_agg['sj_size_pseudo']
            # add syn_ssv dict to AttributeStorage
            this_attr_dc = dict(neuron_partners=ssv_ids)
            this_attr_dc.update(syn_props_agg)
            attr_dc[next_id] = this_attr_dc
            if global_params.config.use_new_subfold:
                next_id += 1
            else:
                next_id += sd_syn.n_folders_fs

        if n_items_for_path > n_per_voxel_path:
            # TODO: passing explicit dest_path might not be required here
            voxel_dc.push(sd_syn_ssv.so_storage_path + voxel_rel_paths[cur_path_id] +
                              "/voxel.pkl")
            attr_dc.push(sd_syn_ssv.so_storage_path + voxel_rel_paths[cur_path_id] +
                         "/attr_dict.pkl")

            cur_path_id += 1
            n_items_for_path = 0

            next_id = ix_from_subfold(voxel_rel_paths[cur_path_id], sd_syn.n_folders_fs)

            base_dir = sd_syn_ssv.so_storage_path + voxel_rel_paths[cur_path_id]
            os.makedirs(base_dir, exist_ok=True)
            voxel_dc = VoxelStorage(base_dir + "voxel.pkl", read_only=False)
            attr_dc = AttributeDict(base_dir + "attr_dict.pkl", read_only=False)

    if n_items_for_path > 0:
        # TODO: passing explicit dest_path might not be required here
        voxel_dc.push(sd_syn_ssv.so_storage_path + voxel_rel_paths[cur_path_id] +
                          "/voxel.pkl")
        attr_dc.push(sd_syn_ssv.so_storage_path + voxel_rel_paths[cur_path_id] +
                         "/attr_dict.pkl")


def filter_relevant_cs_agg(cs_agg, ssd):
    """
    This function filters (likely ;) ) the intra-ssv contact
    sites (inside of a ssv, not between ssvs) that do not need to be agglomerated.

    :param cs_agg:
    :param ssd:
    :return:
    """
    sv_ids = ch.sv_id_to_partner_ids_vec(cs_agg.ids)

    cs_agg_ids = cs_agg.ids.copy()

    # this might mean that all cs between svs with IDs>max(np.uint32) are discarded
    sv_ids[sv_ids >= len(ssd.id_changer)] = -1
    mapped_sv_ids = ssd.id_changer[sv_ids]

    mask = np.all(mapped_sv_ids > 0, axis=1)
    cs_agg_ids = cs_agg_ids[mask]
    filtered_mapped_sv_ids = mapped_sv_ids[mask]

    # this identifies all inter-ssv contact sites
    mask = filtered_mapped_sv_ids[:, 0] - filtered_mapped_sv_ids[:, 1] != 0
    cs_agg_ids = cs_agg_ids[mask]
    relevant_cs_agg = filtered_mapped_sv_ids[mask]

    relevant_cs_ids = np.left_shift(np.max(relevant_cs_agg, axis=1), 32) + np.min(relevant_cs_agg, axis=1)

    rel_cs_to_cs_agg_ids = defaultdict(list)
    for i_entry in range(len(relevant_cs_ids)):
        rel_cs_to_cs_agg_ids[relevant_cs_ids[i_entry]].\
            append(cs_agg_ids[i_entry])

    return rel_cs_to_cs_agg_ids


# TODO: Use this in case contact objects are required
def combine_and_split_cs_agg(wd, cs_gap_nm=300, ssd_version=None,
                             cs_agg_version=None, n_folders_fs=10000,
                             stride=1000, nb_cpus=None):

    ssd = super_segmentation.SuperSegmentationDataset(wd, version=ssd_version)
    cs_agg = segmentation.SegmentationDataset("cs_agg", working_dir=wd,
                                              version=cs_agg_version)

    rel_cs_to_cs_agg_ids = filter_relevant_cs_agg(cs_agg, ssd)
    storage_location_ids = get_unique_subfold_ixs(n_folders_fs)
    voxel_rel_paths_2stage = np.unique([subfold_from_ix(ix, n_folders_fs)[:-2]
                                        for ix in storage_location_ids])

    voxel_rel_paths = [subfold_from_ix(ix, n_folders_fs) for ix in storage_location_ids]
    block_steps = np.linspace(0, len(voxel_rel_paths),
                              int(np.ceil(float(len(rel_cs_to_cs_agg_ids)) / stride)) + 1).astype(np.int)

    cs = segmentation.SegmentationDataset("cs_ssv", working_dir=wd, version="new",
                                          create=True, n_folders_fs=n_folders_fs)

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

    if not qu.batchjob_enabled():
        sm.start_multiprocess(_combine_and_split_cs_agg_thread,
                              multi_params, nb_cpus=nb_cpus)

    else:
        qu.batchjob_script(multi_params, "combine_and_split_cs_agg",
                           remove_jobfolder=True)

    return cs


# TODO: Use this in case contact objects are required
def _combine_and_split_cs_agg_thread(args):
    wd = args[0]
    rel_cs_to_cs_agg_ids_items = args[1]
    voxel_rel_paths = args[2]
    cs_version = args[3]
    cs_ssv_version = args[4]
    scaling = args[5]
    cs_gap_nm = args[6]

    # TODO: changed cs type to 'cs_ssv', check if that is adapted everywhere
    cs = segmentation.SegmentationDataset("cs_ssv", working_dir=wd,
                                          version=cs_ssv_version)

    cs_agg = segmentation.SegmentationDataset("cs", working_dir=wd,
                                              version=cs_version)

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

        ccs = cc_large_voxel_lists(voxel_list * scaling, cs_gap_nm)

        for this_cc in ccs:
            this_vx = voxel_list[np.array(list(this_cc))]
            abs_offset = np.min(this_vx, axis=0)
            this_vx -= abs_offset

            id_mask = np.zeros(np.max(this_vx, axis=0) + 1, dtype=np.bool)
            id_mask[this_vx[:, 0], this_vx[:, 1], this_vx[:, 2]] = True

            try:
                voxel_dc[next_id] = [id_mask], [abs_offset]
            except:
                log_extraction.error("failed {}".format(item))
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


def cc_large_voxel_lists(voxel_list, cs_gap_nm, max_concurrent_nodes=5000,
                         verbose=False):
    kdtree = spatial.cKDTree(voxel_list)

    checked_ids = np.array([], dtype=np.int)
    next_ids = np.array([0])
    ccs = [set(next_ids)]

    current_ccs = 0
    vx_ids = np.arange(len(voxel_list), dtype=np.int)

    while True:
        if verbose:
            log_extraction.debug("NEXT - %d - %d" % (len(next_ids),
                                                     len(checked_ids)))
            for cc in ccs:
                log_extraction.debug("N voxels in cc: %d" % (len(cc)))

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


def extract_synapse_type(sj_sd, kd_asym_path, kd_sym_path,
                         trafo_dict_path=None, stride=100,
                         nb_cpus=None, sym_label=1, asym_label=1):
    """
    Extract synapse type from KnossosDatasets. Stores sym.-asym. ratio in
    syn_ssv object attribute dict.

    Notes:
        If the synapse type prediction was carried out via
        :func:`~syconn.exec.exec_dense_prediction.predict_synapsetype` labels
        are as follows:
            * Label 1: asymmetric.
            * Label 2: symmetric.

    Parameters
    ----------
    sj_sd : SegmentationDataset
    kd_asym_path : str
    kd_sym_path : str
    trafo_dict_path : dict
    stride : int
    nb_cpus : int
    sym_label: int
        Label of symmetric class within `kd_sym_path`.
    asym_label: int
        Label of asymmetric class within `kd_asym_path`.
    """
    log_extraction.warning('DeprecationWarning: Synapse type extraction is now included '
                           'in the object extraction. ')
    assert "syn_ssv" in sj_sd.version_dict
    if (sym_label == asym_label) and (kd_asym_path == kd_sym_path):
        raise ValueError('Both KnossosDatasets and labels for symmetric and '
                         'asymmetric synapses are identical. Either one '
                         'must differ.')
    paths = sj_sd.so_dir_paths

    # Partitioning the work
    multi_params = []
    for path_block in [paths[i:i + stride] for i in range(0, len(paths), stride)]:
        multi_params.append([path_block, sj_sd.version, sj_sd.working_dir,
                             kd_asym_path, kd_sym_path, trafo_dict_path,
                             sym_label, asym_label])

    # Running workers - Extracting mapping
    if not qu.batchjob_enabled():
        _ = sm.start_multiprocess_imap(_extract_synapse_type_thread,
                                       multi_params, nb_cpus=nb_cpus)

    else:
        _ = qu.batchjob_script(
            multi_params, "extract_synapse_type", n_cores=nb_cpus)


def _extract_synapse_type_thread(args):
    paths = args[0]
    obj_version = args[1]
    working_dir = args[2]
    kd_asym_path = args[3]
    kd_sym_path = args[4]
    trafo_dict_path = args[5]
    sym_label = args[6]
    asym_label = args[7]

    if trafo_dict_path is not None:
        with open(trafo_dict_path, "rb") as f:
            trafo_dict = pkl.load(f)
    else:
        trafo_dict = None

    kd_asym = kd_factory(kd_asym_path)
    kd_sym = kd_factory(kd_sym_path)

    seg_dataset = segmentation.SegmentationDataset("syn_ssv",
                                                   version=obj_version,
                                                   working_dir=working_dir)
    for p in paths:
        this_attr_dc = AttributeDict(p + "/attr_dict.pkl",
                                     read_only=False, disable_locking=True)
        for so_id in this_attr_dc.keys():
            so = seg_dataset.get_segmentation_object(so_id)
            so.attr_dict = this_attr_dc[so_id]
            so.load_voxel_list()

            vxl = so.voxel_list

            if trafo_dict is not None:
                vxl -= trafo_dict[so_id]
                vxl = vxl[:, [1, 0, 2]]
            # TODO: remove try-except
            if global_params.config.syntype_available:
                try:
                    asym_prop = np.mean(kd_asym.from_raw_cubes_to_list(vxl) == asym_label)
                    sym_prop = np.mean(kd_sym.from_raw_cubes_to_list(vxl) == sym_label)
                except:
                    log_extraction.error("Failed to read raw cubes during synapse type "
                                         "extraction.")
                    sym_prop = 0
                    asym_prop = 0
            else:
                sym_prop = 0
                asym_prop = 0

            if sym_prop + asym_prop == 0:
                sym_ratio = -1
            else:
                sym_ratio = sym_prop / float(asym_prop + sym_prop)
            so.attr_dict["syn_type_sym_ratio"] = sym_ratio
            syn_sign = -1 if sym_ratio > global_params.config['cell_objects']['sym_thresh'] else 1
            so.attr_dict["syn_sign"] = syn_sign
            this_attr_dc[so_id] = so.attr_dict
        this_attr_dc.push()


def map_objects_from_synssv_partners(wd: str, obj_version: Optional[str] = None,
                                     ssd_version: Optional[str] = None,
                                     n_jobs: Optional[int] = None,
                                     debug: bool = False, log: Logger = None,
                                     max_rep_coord_dist_nm: Optional[float] = None,
                                     max_vert_dist_nm: Optional[float] = None):
    """
    Map sub-cellular objects of the synaptic partners of 'syn_ssv' objects and stores
    them in their attribute dict.

    The following keys will be available in the ``attr_dict`` of ``syn_ssv``-typed
    :class:`~syconn.reps.segmentation.SegmentationObject`:
        * 'n_mi_objs_%d':
        * 'n_mi_vxs_%d':
        * 'n_mi_objs_%d':
        * 'n_mi_vxs_%d':

    Args:
        wd:
        obj_version:
        ssd_version:
        n_jobs:
        debug:
        log:
        max_rep_coord_dist_nm:
        max_vert_dist_nm:

    Returns:

    """
    if n_jobs is None:
        n_jobs = global_params.config.ncore_total * 4
    if max_rep_coord_dist_nm is None:
        max_rep_coord_dist_nm = global_params.config['cell_objects']['max_rep_coord_dist_nm']
    if max_vert_dist_nm is None:
        max_vert_dist_nm = global_params.config['cell_objects']['max_vx_dist_nm']  # TODO: rename in config
    ssd = super_segmentation.SuperSegmentationDataset(working_dir=wd,
                                                      version=ssd_version)

    multi_params = []

    for ids_small_chunk in chunkify(ssd.ssv_ids, n_jobs):
        multi_params.append([wd, obj_version, ssd_version, ids_small_chunk,
                             max_rep_coord_dist_nm, max_vert_dist_nm])

    if not qu.batchjob_enabled():
        _ = sm.start_multiprocess_imap(
            _map_objects_from_synssv_partners_thread, multi_params,
            debug=debug)
    else:
        _ = qu.batchjob_script(
            multi_params, "map_objects_from_synssv_partners", log=log,
            remove_jobfolder=True)

    # iterate over paths with syn
    sd_syn_ssv = segmentation.SegmentationDataset("syn_ssv", working_dir=wd,
                                                  version=obj_version)

    multi_params = []
    for so_dir_paths in chunkify(sd_syn_ssv.so_dir_paths, n_jobs):
        multi_params.append([so_dir_paths, wd, obj_version,
                             ssd_version])
    if not qu.batchjob_enabled():
        _ = sm.start_multiprocess_imap(
            _objects_from_cell_to_syn_dict, multi_params,
            debug=False)
    else:
        _ = qu.batchjob_script(
            multi_params, "objects_from_cell_to_syn_dict", log=log,
            remove_jobfolder=True)
    if log is None:
        log = log_extraction
    log.debug('Deleting cache dictionaries now.')
    # delete cache_dc
    sm.start_multiprocess_imap(_delete_all_cache_dc, ssd.ssv_ids,
                               nb_cpus=global_params.config['ncores_per_node'])
    log.debug('Deleted all cache dictionaries.')


def _map_objects_from_synssv_partners_thread(args: tuple):
    """
    Helper function of 'map_objects_from_synssv_partners'.

    Args:
        args: see 'map_objects_from_synssv_partners'

    Returns:

    """
    # TODO: add global overwrite kwarg
    overwrite = True
    wd, obj_version, ssd_version, ssv_ids, max_rep_coord_dist_nm, max_vert_dist_nm = args

    sd_syn_ssv = segmentation.SegmentationDataset(obj_type="syn_ssv",
                                                  working_dir=wd,
                                                  version=obj_version)
    ssd = super_segmentation.SuperSegmentationDataset(working_dir=wd,
                                                      version=ssd_version)
    sd_vc = segmentation.SegmentationDataset(obj_type="vc", working_dir=wd)
    sd_mi = segmentation.SegmentationDataset(obj_type="mi", working_dir=wd)

    syn_neuronpartners = sd_syn_ssv.load_cached_data("neuron_partners")
    # dts = dict(id_mask=0, kds=0, map_verts=0, directio=0, meshcache=0)
    for ssv_id in ssv_ids:  # Iterate over cells
        ssv_o = ssd.get_super_segmentation_object(ssv_id)

        # start = time.time()
        ssv_o.load_attr_dict()
        if overwrite and os.path.isfile(ssv_o.ssv_dir + "/cache_syn.pkl"):
            os.remove(ssv_o.ssv_dir + "/cache_syn.pkl")
        cache_dc = CompressedStorage(ssv_o.ssv_dir + "/cache_syn.pkl",
                                     read_only=False, disable_locking=True)
        if not overwrite and ('n_vc_vxs' in cache_dc):
            continue
        # dts['directio'] += time.time() - start

        curr_ssv_mask = (syn_neuronpartners[:, 0] == ssv_id) | \
                        (syn_neuronpartners[:, 1] == ssv_id)
        synssv_ids = sd_syn_ssv.ids[curr_ssv_mask]
        n_synssv = len(synssv_ids)
        n_mi_objs = np.zeros((n_synssv,), dtype=np.int)
        n_mi_vxs = np.zeros((n_synssv,), dtype=np.int)
        n_vc_objs = np.zeros((n_synssv,), dtype=np.int)
        n_vc_vxs = np.zeros((n_synssv,), dtype=np.int)
        cache_dc['synssv_ids'] = synssv_ids
        if n_synssv == 0:
            cache_dc['n_mi_objs'] = n_mi_objs
            cache_dc['n_mi_vxs'] = n_mi_vxs
            cache_dc['n_vc_objs'] = n_vc_objs
            cache_dc['n_vc_vxs'] = n_vc_vxs
            cache_dc.push()
            continue
        # start = time.time()
        vc_mask = np.in1d(sd_vc.ids, ssv_o.vc_ids)
        mi_mask = np.in1d(sd_mi.ids, ssv_o.mi_ids)
        # dts['id_mask'] += time.time() - start
        vc_ids = sd_vc.ids[vc_mask]
        mi_ids = sd_mi.ids[mi_mask]
        if len(vc_ids) < 2 and len(mi_ids) < 2:
            continue
        vc_sizes = sd_vc.sizes[vc_mask]
        mi_sizes = sd_mi.sizes[mi_mask]

        # start = time.time()
        kdtree_synssv = spatial.cKDTree(sd_syn_ssv.rep_coords[curr_ssv_mask] * sd_syn_ssv.scaling)

        # vesicle clouds
        kdtree_vc = spatial.cKDTree(sd_vc.rep_coords[vc_mask] * sd_vc.scaling)

        # mitos
        kdtree_mi = spatial.cKDTree(sd_mi.rep_coords[mi_mask] * sd_mi.scaling)

        # returns a list of neighboring objects for every synssv (note: ix is now the index within ssv_o.mi_ids
        close_mi_ixs = kdtree_synssv.query_ball_tree(kdtree_mi, r=max_rep_coord_dist_nm)
        close_vc_ixs = kdtree_synssv.query_ball_tree(kdtree_vc, r=max_rep_coord_dist_nm)
        # dts['kds'] += time.time() - start

        # start = time.time()
        close_mi_ids = mi_ids[np.unique(np.concatenate(close_mi_ixs)).astype(np.int)]
        close_vc_ids = vc_ids[np.unique(np.concatenate(close_vc_ixs).astype(np.int))]

        md_mi = seghelp.load_so_meshes_bulk(sd_mi.get_segmentation_object(close_mi_ids))
        md_vc = seghelp.load_so_meshes_bulk(sd_vc.get_segmentation_object(close_vc_ids))
        # md_synssv = seghelp.load_so_meshes_bulk(sd_syn_ssv.get_segmentation_object(synssv_ids))
        # dts['meshcache'] += time.time() - start

        # start = time.time()
        for ii, synssv_id in enumerate(synssv_ids):
            synssv_obj = sd_syn_ssv.get_segmentation_object(synssv_id)
            # synssv_obj._mesh = md_synssv[synssv_id]
            mis = sd_mi.get_segmentation_object(mi_ids[close_mi_ixs[ii]])
            # load cached meshes
            for jj, ix in enumerate(close_mi_ixs[ii]):
                mi = mis[jj]
                mi._size = mi_sizes[ix]
                mi._mesh = md_mi[mi.id]
            vcs = sd_vc.get_segmentation_object(vc_ids[close_vc_ixs[ii]])
            for jj, ix in enumerate(close_vc_ixs[ii]):
                vc = vcs[jj]
                vc._size = vc_sizes[ix]
                vc._mesh = md_vc[vc.id]
            n_mi_objs[ii], n_mi_vxs[ii] = _map_objects_from_synssv(synssv_obj, mis, max_vert_dist_nm)
            n_vc_objs[ii], n_vc_vxs[ii] = _map_objects_from_synssv(synssv_obj, vcs, max_vert_dist_nm)
        # dts['map_verts'] += time.time() - start

        # start = time.time()
        cache_dc['n_mi_objs'] = n_mi_objs
        cache_dc['n_mi_vxs'] = n_mi_vxs
        cache_dc['n_vc_objs'] = n_vc_objs
        cache_dc['n_vc_vxs'] = n_vc_vxs
        cache_dc.push()
        # dts['directio'] += time.time() - start


def _map_objects_from_synssv(synssv_o, seg_objs, max_vert_dist_nm, sample_fact=2):
    """
    TODO: Loading meshes for approximating close-by object volume is slow - exchange with summed object size?

    Maps cellular organelles to syn_ssv objects. Needed for the RFC model which
    is executed in 'classify_synssv_objects'.
    Helper function of `objects_to_single_synssv`.

    Args:
        synssv_o: 'syn_ssv' synapse object.
        seg_objs: SegmentationObject of type 'vc' or 'mi'
        max_vert_dist_nm: Query radius for SegmentationObject vertices. Used to estimate
            number of nearby object voxels.
        sample_fact: only use every Xth vertex.

    Returns:
        Number of SegmentationObjects with >0 vertices, approximated number of
        object voxels within `max_vert_dist_nm`.
    """
    # synssv_kdtree = spatial.cKDTree(synssv_o.mesh[1].reshape(-1, 3)[::sample_fact])
    synssv_kdtree = spatial.cKDTree(synssv_o.voxel_list[::sample_fact] * synssv_o.scaling)

    n_obj_vxs = []
    for obj in seg_objs:
        # use mesh vertices instead of voxels
        obj_vxs = obj.mesh[1].reshape(-1, 3)[::sample_fact]

        ds, _ = synssv_kdtree.query(obj_vxs, distance_upper_bound=max_vert_dist_nm)
        # surface fraction of subcellular object which is close to synapse
        close_frac = np.sum(ds < np.inf) / len(obj_vxs)

        # estimate number of voxels by close-by surface area fraction times total number of voxels
        n_obj_vxs.append(close_frac * obj.size)

    n_obj_vxs = np.array(n_obj_vxs)

    n_objects = np.sum(n_obj_vxs > 0)
    n_vxs = np.sum(n_obj_vxs)

    return n_objects, n_vxs


def _objects_from_cell_to_syn_dict(args):
    """
    args : Tuple
        see 'map_objects_from_synssv_partners'
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
            map_dc = dict()
            for ii, ssv_partner_id in enumerate(synssv_o.attr_dict["neuron_partners"]):
                ssv_o = ssd.get_super_segmentation_object(ssv_partner_id)
                cache_dc = CompressedStorage(ssv_o.ssv_dir + "/cache_syn.pkl")

                index = np.transpose(np.nonzero(cache_dc['synssv_ids'] == synssv_id))
                if len(index) != 1:
                    msg = "Partner cell ID mismatch."
                    log_extraction.error(msg)
                    raise ValueError(msg)
                index = index[0][0]
                map_dc[f'n_mi_objs_{ii}'] = cache_dc['n_mi_objs'][index]
                map_dc[f'n_mi_vxs_{ii}'] = cache_dc['n_mi_vxs'][index]
                map_dc[f'n_vc_objs_{ii}'] = cache_dc['n_vc_objs'][index]
                map_dc[f'n_vc_vxs_{ii}'] = cache_dc['n_vc_vxs'][index]
            synssv_o.attr_dict.update(map_dc)
            this_attr_dc[synssv_id] = synssv_o.attr_dict
        this_attr_dc.push()


def classify_synssv_objects(wd, obj_version=None, log=None, nb_cpus=None):
    """
    TODO: Replace by new synapse detection.
    Classify SSV contact sites into synaptic or non-synaptic using an RFC model
    and store the result in the attribute dict of the syn_ssv objects.
    For requirements see `synssv_o_features`.

    Args:
        wd:
        obj_version:
        log:
        nb_cpus:

    Returns:

    """
    sd_syn_ssv = segmentation.SegmentationDataset("syn_ssv", working_dir=wd,
                                                  version=obj_version)

    multi_params = chunkify(sd_syn_ssv.so_dir_paths, global_params.config.ncore_total)
    multi_params = [(so_dir_paths, wd, obj_version) for so_dir_paths in
                    multi_params]

    if not qu.batchjob_enabled():
        _ = sm.start_multiprocess_imap(_classify_synssv_objects_thread,
                                        multi_params, nb_cpus=nb_cpus)

    else:
        _ = qu.batchjob_script(
            multi_params,  "classify_synssv_objects", log=log,
            remove_jobfolder=True)


def _classify_synssv_objects_thread(args):
    """
    Helper function of 'classify_synssv_objects'.

    Parameters
    ----------
    args : Tuple
        see 'classify_synssv_objects'
    """
    so_dir_paths, wd, obj_version = args

    sd_syn_ssv = segmentation.SegmentationDataset(obj_type="syn_ssv",
                                                  working_dir=wd,
                                                  version=obj_version)
    rfc = joblib.load(global_params.config.mpath_syn_rfc)

    for so_dir_path in so_dir_paths:
        this_attr_dc = AttributeDict(so_dir_path + "/attr_dict.pkl",
                                     read_only=False)

        for synssv_id in this_attr_dc.keys():
            synssv_o = sd_syn_ssv.get_segmentation_object(synssv_id)
            synssv_o.attr_dict = this_attr_dc[synssv_id]

            feats = synssv_o_features(synssv_o)
            syn_prob = rfc.predict_proba([feats])[0][1]

            synssv_o.attr_dict.update({"syn_prob": syn_prob})
            this_attr_dc[synssv_id] = synssv_o.attr_dict

        this_attr_dc.push()


# Code for property extraction of contact sites (syn_ssv)

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


def create_syn_rfc(conn: 'segmentation.SegmentationDataset', path_kzip: str, overwrite: bool = False,
                   rfc_path_out: str = None, max_dist_vx: int = 20) -> \
        Tuple[ensemble.RandomForestClassifier, np.ndarray, np.ndarray]:
    """
    Trains a random forest classifier (RFC) to distinguish between synaptic and non-synaptic
    objects. Features are generated from the objects in `conn` associated with the annotated
    coordinates stored in `path_kzip`.
    Will write the trained classifier to ``global_params.config.mpath_syn_rfc``.

    Args:
        conn: :class:`~syconn.reps.segmentation.SegmentationDataset` object of
            type ``syn_ssv``. Used to identify synaptic object candidates annotated
            in the kzip file at `path_kzip`.
        path_kzip: Path to kzip file with synapse labels as node comments
            ("non-synaptic", "synaptic"; labels used for classifier are 0 and 1
            respectively).
        overwrite: Replace existing files.
        rfc_path_out: Filename for dumped RFC.
        max_dist_vx: Maximum voxel distance between sample and target.

    Returns:
        The trained random forest classifier and the feature and label data.
    """
    log = log_extraction
    if global_params.config.working_dir is not None or rfc_path_out is not None:
        if rfc_path_out is None:
            model_base_dir = os.path.split(global_params.config.mpath_syn_rfc)[0]
            log(f'Working directory is set to {global_params.config.working_dir} - '
                f'trained RFC will be dumped at {model_base_dir}.')
            os.makedirs(model_base_dir, exist_ok=True)
            rfc_path_out = global_params.config.mpath_syn_rfc
        else:
            log = initialize_logging('create_syn_rfc', os.path.dirname(rfc_path_out))
        if os.path.isfile(rfc_path_out) and not overwrite:
            raise FileExistsError()
    assert conn.type == 'syn_ssv'
    anno = skeleton_utils.load_skeleton(path_kzip)['Synapse annotation']

    log.info(f'Initiated RFC fitting procedure with GT file "{path_kzip}" and {conn}.')

    base_dir = os.path.split(path_kzip)[0]
    mapped_synssv_objects_kzip = f'{base_dir}/mapped_synssv.k.zip'
    if os.path.isfile(mapped_synssv_objects_kzip):
        if not overwrite:
            raise FileExistsError(f'File with mapped synssv objects already exists '
                                  f'at "{mapped_synssv_objects_kzip}"')
        os.remove(mapped_synssv_objects_kzip)
    label_coords = []
    labels = []
    for node in anno.getNodes():
        c = node.getComment()
        if not ((c == 'synaptic') | (c == 'non-synaptic')):
            continue
        labels.append(c)
        label_coords.append(np.array(node.getCoordinate()))

    labels = np.array(labels)
    label_coords = np.array(label_coords)

    # get deterministic order by sorting by coordinate first and then seeded shuffling
    ixs = [i[0] for i in sorted(enumerate(label_coords),
                                key=lambda x: [x[1][0], x[1][1], x[1][2]])]
    ixs_random = np.arange(len(ixs))
    np.random.seed(0)
    np.random.shuffle(ixs_random)
    ixs = np.array(ixs)
    label_coords = label_coords[ixs][ixs_random]
    labels = labels[ixs][ixs_random]
    conn_kdtree = spatial.cKDTree(conn.rep_coords * conn.scaling)
    ds, list_ids = conn_kdtree.query(label_coords * conn.scaling)
    synssv_ids = conn.ids[list_ids]
    mask = np.ones(synssv_ids.shape, dtype=np.bool)
    log.info(f'Mapped {len(labels)} GT coordinates to {conn.type}-objects.')
    for label_id in np.where(ds > 0)[0]:
        dists, close_ids = conn_kdtree.query(label_coords[label_id] * conn.scaling,
                                             k=20)
        for close_id in close_ids[np.argsort(dists)]:
            conn_o = conn.get_segmentation_object(conn.ids[close_id])
            vx_ds = np.sum(np.abs(conn_o.voxel_list - label_coords[label_id]),
                           axis=-1)
            if np.min(vx_ds) < max_dist_vx:
                synssv_ids[label_id] = conn.ids[close_id]
                break
        if np.min(vx_ds) > max_dist_vx:
            mask[label_id] = 0

    if np.sum(mask) == 0:
        raise ValueError
    synssv_ids = synssv_ids[mask]
    labels = labels[mask]
    log.info(f'Found {np.sum(mask)} samples with a distance < {max_dist_vx} vx to the target.')

    log.info(f'Synapse features will now be generated and written to {mapped_synssv_objects_kzip}.')
    features = []
    skel = skeleton.Skeleton()
    anno = skeleton.SkeletonAnnotation()
    anno.scaling = conn.scaling
    pbar = tqdm.tqdm(total=len(synssv_ids))
    for kk, synssv_id in enumerate(synssv_ids):
        synssv_o = conn.get_segmentation_object(synssv_id)
        rep_coord = synssv_o.rep_coord * conn.scaling
        # synssv_o.mesh2kzip(mapped_synssv_objects_kzip, ext_color=None,
        # ply_name='{}.ply'.format(synssv_id))
        n = skeleton.SkeletonNode().from_scratch(anno, rep_coord[0], rep_coord[1], rep_coord[2])
        n.setComment('{}'.format(labels[kk]))
        anno.addNode(n)
        rep_coord = label_coords[kk] * conn.scaling
        n_l = skeleton.SkeletonNode().from_scratch(anno, rep_coord[0], rep_coord[1], rep_coord[2])
        n_l.setComment('gt node; {}'.format(labels[kk]))
        anno.addNode(n_l)
        anno.addEdge(n, n_l)
        features.append(synssv_o_features(synssv_o))
        pbar.update(1)
    pbar.close()
    skel.add_annotation(anno)
    skel.to_kzip(mapped_synssv_objects_kzip)
    features = np.array(features)
    rfc = ensemble.RandomForestClassifier(n_estimators=400, max_features='sqrt',
                                          n_jobs=-1, random_state=0,
                                          oob_score=True)
    mask_annotated = (labels == "synaptic") | (labels == 'non-synaptic')
    v_features = features[mask_annotated]
    v_labels = labels[mask_annotated]
    v_labels = (v_labels == "synaptic").astype(np.int)
    score = cross_val_score(rfc, v_features, v_labels, cv=10)
    log.info('RFC oob score: {:.4f}'.format(rfc.oob_score))
    log.info('RFC CV score +- std: {:.4f} +- {:.4f}'.format(
        np.mean(score), np.std(score)))

    rfc.fit(v_features, v_labels)
    acc = rfc.score(v_features, v_labels)
    log.info(f'Training accuracy: {acc:.4f}')
    feature_names = synssv_o_featurenames()
    feature_imp = rfc.feature_importances_
    assert len(feature_imp) == len(feature_names)
    log.info('RFC importances:\n' + "\n".join(
        [f"{feature_names[ii]}: {feature_imp[ii]}" for ii in range(len(feature_imp))]))
    if rfc_path_out is not None:
        joblib.dump(rfc, rfc_path_out)
        log.info(f'Wrote parameters of trained RFC to "{rfc_path_out}".')
    else:
        log.info('Working directory and rfc_path_out not set - trained RFC was not dumped to file.')

    return rfc, v_features, v_labels


def synssv_o_features(synssv_o: segmentation.SegmentationObject):
    """
    Collects syn_ssv feature for synapse prediction using an RFC.

    Parameters
    ----------
    synssv_o : SegmentationObject

    Returns
    -------
    List
    """
    features = [synssv_o.size, synssv_o.mesh_area,
                synssv_o.attr_dict["id_cs_ratio"]]

    partner_ids = synssv_o.attr_dict("neuron_partners")
    for i_partner_id, partner_id in enumerate(partner_ids):
        features.append(synssv_o.attr_dict["n_mi_objs_%d" % i_partner_id])
        features.append(synssv_o.attr_dict["n_mi_vxs_%d" % i_partner_id])
        features.append(synssv_o.attr_dict["n_vc_objs_%d" % i_partner_id])
        features.append(synssv_o.attr_dict["n_vc_vxs_%d" % i_partner_id])
    return features


def synssv_o_featurenames():
    return ['size_vx', 'mesh_area_um2', 'id_cs_ratio', 'n_mi_objs_neuron1',
            'n_mi_vxs_neuron1', 'n_vc_objs_neuron1', 'n_vc_vxs_neuron1',
            'n_mi_objs_neuron2', 'n_mi_vxs_neuron2', 'n_vc_objs_neuron2',
            'n_vc_vxs_neuron2']


def export_matrix(obj_version=None, dest_folder=None, threshold_syn=None):
    """
    Writes .csv and .kzip summary file of connectivity matrix.

    Parameters
    ----------
    wd : str
    obj_version : str
    dest_folder : str
        Path to csv file
    threshold_syn : float
    """
    if threshold_syn is None:
        threshold_syn = global_params.config['cell_objects']['thresh_synssv_proba']
    if dest_folder is None:
        dest_folder = global_params.config.working_dir + '/connectivity_matrix/'
    os.makedirs(os.path.split(dest_folder)[0], exist_ok=True)
    dest_name = dest_folder + '/conn_mat'
    sd_syn_ssv = segmentation.SegmentationDataset("syn_ssv", working_dir=global_params.config.working_dir,
                                                  version=obj_version)

    syn_prob = sd_syn_ssv.load_cached_data("syn_prob")

    m = syn_prob > threshold_syn
    m_axs = sd_syn_ssv.load_cached_data("partner_axoness")[m]
    m_cts = sd_syn_ssv.load_cached_data("partner_celltypes")[m]
    m_sp = sd_syn_ssv.load_cached_data("partner_spiness")[m]
    m_coords = sd_syn_ssv.rep_coords[m]
    # m_sizes = sd_syn_ssv.sizes[m]
    m_sizes = sd_syn_ssv.load_cached_data("mesh_area")[m] / 2
    m_ssv_partners = sd_syn_ssv.load_cached_data("neuron_partners")[m]
    m_syn_prob = syn_prob[m]
    m_syn_sign = sd_syn_ssv.load_cached_data("syn_sign")[m]
    m_syn_asym_ratio = sd_syn_ssv.load_cached_data("syn_type_sym_ratio")[m]
    m_spineheadvol = sd_syn_ssv.load_cached_data("partner_spineheadvol")[m]
    m_latent_morph = sd_syn_ssv.load_cached_data("latent_morph")[m]  # N, 2, m
    m_latent_morph = m_latent_morph.reshape(len(m_latent_morph), -1)  # N, 2*m

    # (loop of skeleton node generation)
    # make sure cache-arrays have ndim == 2, TODO: check when writing cached arrays
    m_sizes = np.multiply(m_sizes, m_syn_sign).squeeze()[:, None]  # N, 1
    m_axs = m_axs.squeeze()  # N, 2
    m_sp = m_sp.squeeze()  # N, 2
    m_syn_prob = m_syn_prob.squeeze()[:, None]  # N, 1
    table = np.concatenate([m_coords, m_ssv_partners, m_sizes, m_axs, m_cts,
                            m_sp, m_syn_prob, m_spineheadvol, m_latent_morph], axis=1)

    # do not overwrite previous files
    if os.path.isfile(dest_name + '.csv'):
        st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
        os.rename(dest_name + '.csv', '{}_{}.csv'.format(dest_name, st))

    np.savetxt(dest_name + ".csv", table, delimiter="\t",
               header="x\ty\tz\tssv1\tssv2\tsize\tcomp1\tcomp2\tcelltype1\t"
                      "celltype2\tspiness1\tspiness2\tsynprob\tspinehead_vol1"
                      "\tspinehead_vol2" +
                      "".join(["\tlatentmorph1_{}".format(ix) for ix in range(
                          global_params.config['tcmn']['ndim_embedding'])]) +
                      "".join(["\tlatentmorph2_{}".format(ix) for ix in range(
                          global_params.config['tcmn']['ndim_embedding'])])
               )

    ax_labels = np.array(["N/A", "D", "A", "S"])   # TODO: this is already defined in handler.multiviews!
    ax_label_ids = np.array([-1, 0, 1, 2])
    # Documentation of prediction labels, maybe add somewhere to .k.zip or .csv
    ct_labels = ['N/A', 'EA', 'MSN', 'GP', 'INT']   # TODO: this is already defined in handler.multiviews!
    ct_label_ids = np.array([-1, 0, 1, 2, 3])
    sp_labels = ['N/A', 'neck', 'head', 'shaft', 'other']  # TODO: this is already defined in handler.multiviews!
    sp_label_ids = np.array([-1, 0, 1, 2, 3])

    annotations = []
    m_sizes = np.abs(m_sizes)

    ms_axs = np.sort(m_axs, axis=1)
    # transform labels 3 and 4 to 1 (bouton and terminal to axon to apply correct filter)
    ms_axs[ms_axs == 3] = 1
    ms_axs[ms_axs == 4] = 1
    # vigra currently requires numpy==1.11.1
    try:
        u_axs = np.unique(ms_axs, axis=0)
    except TypeError:  # in case numpy < 1.13
        u_axs = np.vstack({tuple(row) for row in ms_axs})
    for u_ax in u_axs:
        anno = skeleton.SkeletonAnnotation()
        anno.scaling = sd_syn_ssv.scaling
        cmt = "{} - {}".format(ax_labels[ax_label_ids == u_ax[0]][0],
                               ax_labels[ax_label_ids == u_ax[1]][0])
        anno.comment = cmt
        for i_syn in np.where(np.sum(np.abs(ms_axs - u_ax), axis=1) == 0)[0]:
            c = m_coords[i_syn]
            # somewhat approximated from sphere volume:
            r = np.power(m_sizes[i_syn] / 3., 1 / 3.)
            #    r = m_sizes[i_syn]
            skel_node = skeleton.SkeletonNode(). \
            from_scratch(anno, c[0], c[1], c[2], radius=r)
            skel_node.data["ids"] = m_ssv_partners[i_syn]
            skel_node.data["size"] = m_sizes[i_syn]
            skel_node.data["syn_prob"] = m_syn_prob[i_syn]
            skel_node.data["sign"] = m_syn_sign[i_syn]
            skel_node.data["in_ex_frac"] = m_syn_asym_ratio[i_syn]
            skel_node.data['sp'] = m_sp[i_syn]
            skel_node.data['ct'] = m_cts[i_syn]
            skel_node.data['ax'] = m_axs[i_syn]
            skel_node.data['latent_morph'] = m_latent_morph[i_syn]
            anno.addNode(skel_node)
        annotations.append(anno)

    # do not overwrite previous files
    if os.path.isfile(dest_name + '.k.zip'):
        st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
        os.rename(dest_name + '.k.zip', '{}_{}.k.zip'.format(dest_name, st))
    skeleton_utils.write_skeleton(dest_name + ".k.zip", annotations)

