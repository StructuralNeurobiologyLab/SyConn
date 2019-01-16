# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max-Planck-Institute of Neurobiology, Munich, Germany
# Authors: Philipp Schubert, Joergen Kornfeld
import time
import numpy as np
import networkx as nx
import pandas as pd

from ..reps import super_segmentation as ss
from ..reps import segmentation
from ..config import global_params

# TODO: unclear what and when this was used for, refactor and use in current project

def extract_connectivity_thread(args):
    sj_obj_ids = args[0]
    sj_version = args[1]
    ssd_version = args[2]
    working_dir = args[3]

    ssd = ss.SuperSegmentationDataset(working_dir,
                                      version=ssd_version)

    sd = segmentation.SegmentationDataset("sj", version=sj_version,
                                          working_dir=working_dir)

    cons = []
    for sj_obj_id in sj_obj_ids:
        sj = sd.get_segmentation_object(sj_obj_id)
        con = extract_connectivity_information(sj, ssd)
        if con is not None:
            if len(cons) == 0:
                cons = con
            else:
                cons = np.concatenate((cons, con))

    return cons


def sv_id_to_partner_ids_vec(cs_ids):
    sv_ids = np.right_shift(cs_ids, 32)
    sv_ids = np.concatenate((sv_ids[:, None],
                             (cs_ids - np.left_shift(sv_ids, 32))[:, None]),
                            axis=1)
    return sv_ids


def extract_connectivity_information(sj, ssd):
    sj.load_attr_dict()

    if not "connectivity" in sj.attr_dict:
        return

    ss_con_ids = ssd.id_changer[np.array(list(sj.attr_dict["connectivity"].keys()),
                                         dtype=np.int)]
    if len(ss_con_ids) == 0:
        return

    con_cnts = np.array(sj.attr_dict["connectivity"].values(), dtype=np.int)

    # Removing intracellular sjs
    ss_con_cnts = con_cnts[ss_con_ids[:, 0] != ss_con_ids[:, 1]]
    if len(ss_con_cnts) == 0:
        return

    ss_con_ids = ss_con_ids[ss_con_ids[:, 0] != ss_con_ids[:, 1]]

    # Adding the counts up
    cs_ids = np.left_shift(np.max(ss_con_ids, axis=1), 32) + \
             np.min(ss_con_ids, axis=1)
    unique_cs_ids, idx = np.unique(cs_ids, return_inverse=True)
    cs_con_cnts = np.bincount(idx, ss_con_cnts)
    cs_con_cnts = cs_con_cnts / np.sum(cs_con_cnts)

    # Going back to ssd domain
    sso_ids = np.right_shift(unique_cs_ids, 32)
    sso_ids = np.concatenate((sso_ids[:, None],
                              (unique_cs_ids -
                               np.left_shift(sso_ids, 32))[:, None]), axis=1)

    # Threshold overlap
    sso_ids = sso_ids[cs_con_cnts > .3]

    if len(sso_ids) == 0:
        return

    cs_con_cnts = cs_con_cnts[cs_con_cnts > .3]
    cs_con_cnts /= np.sum(cs_con_cnts)

    sizes = sj.size * cs_con_cnts * np.product(sj.scaling) / 1e9

    sj_ids = np.array([sj.id] * len(sizes))
    sj_types = np.array([sj.attr_dict["type_ratio"]] * len(sizes))
    sj_coords = np.array([sj.rep_coord] * len(sizes))

    return np.concatenate([sso_ids, sj_ids[:, None], sizes[:, None], sj_types[:, None], sj_coords], axis=1)


def connectivity_to_nx_graph():
    """
    Creates a directed networkx graph with attributes from the
    stored raw connectivity data.

    Returns
    -------

    """

    cd_dict = load_cached_data_dict()

    idx_filter = cd_dict['synaptivity_proba'] > 0.5
    #  & (df_dict['syn_size'] < 5.)

    for k, v in cd_dict.items():
        cd_dict[k] = v[idx_filter]

    idx_filter = (cd_dict['neuron_partner_ax_0']\
                 + cd_dict['neuron_partner_ax_1']) == 1

    for k, v in cd_dict.items():
        cd_dict[k] = v[idx_filter]

    nxg = nx.DiGraph()
    start = time.time()
    print('Starting graph construction')
    for idx in range(0, len(cd_dict['ids'])):

        # find out which one is pre and which one is post
        # 1 indicates pre, i.e. identified as axon by the classifier
        if cd_dict['neuron_partner_ax_0'][idx] == 1:
            u = cd_dict['ssv_partner_0'][idx]
            v = cd_dict['ssv_partner_1'][idx]
        else:
            v = cd_dict['ssv_partner_0'][idx]
            u = cd_dict['ssv_partner_1'][idx]

        nxg.add_edge(u, v)
        # for each synapse create edge with attributes
    print('Done with graph construction, took {0}'.format(time.time()-start))

    return nxg


def load_cached_data_dict(wd=None, syn_version=None):
    """
    Loads all cached data from a contact site segmentation dataset into a
    dictionary for further processing.

    Parameters
    ----------
    wd : str
    syn_version : str

    Returns
    -------

    """
    if wd is None:
        wd = global_params.wd
    start = time.time()
    csd = segmentation.SegmentationDataset(obj_type='syn_ssv', working_dir=wd,
                                           version=syn_version)
    cd_dict = dict()
    cd_dict['ids'] = csd.load_cached_data('id')
    # in um2, overlap of cs and sj
    cd_dict['syn_size'] =\
        csd.load_cached_data('mesh_area') / 2  # as used in syn_analysis.py -> export_matrix
    cd_dict['synaptivity_proba'] = \
        csd.load_cached_data('syn_prob')
    cd_dict['coord_x'] = \
        csd.load_cached_data('rep_coord')[:, 0].astype(np.int)
    cd_dict['coord_y'] = \
        csd.load_cached_data('rep_coord')[:, 1].astype(np.int)
    cd_dict['coord_z'] = \
        csd.load_cached_data('rep_coord')[:, 2].astype(np.int)
    cd_dict['ssv_partner_0'] = \
        csd.load_cached_data('neuron_partners')[:, 0].astype(np.int)
    cd_dict['ssv_partner_1'] = \
        csd.load_cached_data('neuron_partners')[:, 1].astype(np.int)
    cd_dict['neuron_partner_ax_0'] = \
        csd.load_cached_data('partner_axoness')[:, 0].astype(np.int)
    cd_dict['neuron_partner_ax_1'] = \
        csd.load_cached_data('partner_axoness')[:, 1].astype(np.int)
    cd_dict['neuron_partner_ct_0'] = \
        csd.load_cached_data('partner_celltypes')[:, 0].astype(np.int)
    cd_dict['neuron_partner_ct_1'] = \
        csd.load_cached_data('partner_celltypes')[:, 1].astype(np.int)
    cd_dict['neuron_partner_sp_0'] = \
        csd.load_cached_data('partner_spiness')[:, 0].astype(np.int)
    cd_dict['neuron_partner_sp_1'] = \
        csd.load_cached_data('partner_spiness')[:, 1].astype(np.int)
    print('Getting all objects took: {0}'.format(time.time() - start))
    return cd_dict


def connectivity_exporter(human_cell_type_labels=True,
                          cell_type_map={0: 'EA', 1: 'MSN', 2: 'GP', 3: 'INT'},
                          human_pre_post_labels=True,
                          pre_post_map={1: 'pre', 0: 'post'},
                          only_axo_dendritric=True,
                          out_path = None, no_ids=True, only_synapses=True):
    """
    Exports connectivity information to a csv file.

    -------

    """
    if out_path is None:
        out_path = global_params + '/connectivity_matrix/j0126_matrix_v1.csv'
    # parse contact site segmentation dataset
    df_dict = load_cached_data_dict()

    if only_synapses == False:
        start = time.time()
        df = pd.DataFrame(df_dict)
        df.to_csv(out_path, index=False)
        print('Export to csv took: {0}'.format(time.time() - start))
    else:

        idx_filter = df_dict['synaptivity_proba'] > 0.5
        #  & (df_dict['syn_size'] < 5.)

        for k, v in df_dict.items():
            df_dict[k] = v[idx_filter]

        print('{0} synapses of'
              '{1} contact sites'.format(sum(idx_filter),
                                         len(idx_filter)))
        if no_ids:
            del df_dict['ids']

        if human_cell_type_labels:
            df_dict['neuron_partner_ct_0'] = np.array([cell_type_map[int(el)] for el in
                                              df_dict['neuron_partner_ct_0']])
            df_dict['neuron_partner_ct_1'] = np.array([cell_type_map[int(el)] for el in
                                              df_dict['neuron_partner_ct_1']])

        if only_axo_dendritric:
            idx_filter = (df_dict['neuron_partner_ax_0']
                         + df_dict['neuron_partner_ax_1']) == 1

            for k, v in df_dict.items():
                df_dict[k] = v[idx_filter]

            print('{0} axo-dendritic synapses'.format(sum(idx_filter)))

        if human_pre_post_labels:
            df_dict['neuron_partner_ax_0'] = np.array([pre_post_map[int(el)] for el in
                                              df_dict['neuron_partner_ax_0']])
            df_dict['neuron_partner_ax_1'] = np.array([pre_post_map[int(el)] for el in
                                              df_dict['neuron_partner_ax_1']])

        start = time.time()
        df = pd.DataFrame(df_dict)
        df.to_csv(out_path, index=False)
        print('Export to csv took: {0}'.format(time.time() - start))
    return

