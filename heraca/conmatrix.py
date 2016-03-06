# -*- coding: utf-8 -*-
__author__ = 'pschuber'
import matplotlib
matplotlib.use('Agg')
try:
    from NewSkeleton import annotationUtils as au
except:
    import annotationUtils as au
try:
    from NewSkeleton.NewSkeletonUtils import annotation_from_nodes
except:
    from NewSkeletonUtils import annotation_from_nodes
import os
import re
import numpy as np
from numpy import array as arr
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import pearsonr
from sklearn.metrics import precision_recall_curve
import scipy.spatial
import community
import networkx as nx
import matplotlib.pyplot as plt
from NewSkeleton import NewSkeleton
from processing.mapper import feature_valid_syns, calc_syn_dict
from processing.learning_rfc import plot_pr
from contactsite import convert_to_standard_cs_name
from processing.cell_types import load_celltype_feats,\
    load_celltype_probas, get_id_dict_from_skel_ids, \
    load_cell_gt, load_celltype_gt
from utils.datahandler import get_skelID_from_path, get_filepaths_from_dir,\
 write_obj2pkl, load_pkl2obj
from processing.learning_rfc import cell_classification
import seaborn.apionly as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sb
from matplotlib import colors
try:
    from NewSkeleton import annotationUtils as au
except:
    import annotationUtils as au
import matplotlib.gridspec as gridspec
import pandas as pd
from matplotlib import pyplot as pp
from matplotlib import gridspec


def type_sorted_wiring(gt_path='/lustre/pschuber/gt_cell_types/',
                       confidence_lvl=0.8, binary=False, max_syn_size=0.4,
                       load_gt=False, syn_only=True, big_entries=False):
    """
    Calculate wiring of consensus skeletons sorted by type classification
    :return:
    """
    # assert os.path.isfile(gt_path + 'wiring/skel_ids.npy'),\
    #     "Couldn't find mandatory files."
    supp = ""
    skeleton_ids, skeleton_feats = load_celltype_feats(gt_path)
    skeleton_ids2, skel_type_probas = load_celltype_probas(gt_path)
    assert np.all(np.equal(skeleton_ids, skeleton_ids2)), "Skeleton ordering wrong for"\
                                                  "probabilities and features."
    bool_arr = np.zeros(len(skeleton_ids))
    cell_type_pred_dict = {}
    if load_gt:
        supp = "_gt"
        skel_types = load_cell_gt(skeleton_ids)
        bool_arr = skel_types != -1
        bool_arr = bool_arr.astype(np.bool)
        skeleton_ids = skel_ids[bool_arr]
        for k, skel_id in enumerate(skeleton_ids):
            cell_type_pred_dict[skel_id] = skel_types[k]
            # print skel_types[k]
    else:
        # load loo results of evaluation
        cell_type_pred_dict = load_pkl2obj(gt_path+'loo_cell_pred_dict_novel.pkl')
        # remove all skeletons under confidence level
        for k, probas in enumerate(skel_type_probas):
            if np.max(probas) > confidence_lvl:
                bool_arr[k] = 1
        bool_arr = bool_arr.astype(np.bool)
        skeleton_ids = skeleton_ids[bool_arr]
    print "%d/%d are under confidence level %0.2f and being removed." % \
          (np.sum(~bool_arr), len(skeleton_ids2), confidence_lvl)
    # remove identical skeletons
    ident_cnt = 0
    skeleton_ids = skeleton_ids.tolist()
    for skel_id in skeleton_ids:
        if skel_id in [497, 474, 307, 366, 71, 385, 434, 503, 521, 285, 546,
                       158, 604]:
            skeleton_ids.remove(skel_id)
            ident_cnt += 1
    skeleton_ids = arr(skeleton_ids)
    print "Removed %d skeletons because of similarity." % ident_cnt

    # create matrix
    if syn_only:
        syn_props = load_pkl2obj('/lustre/sdorkenw/synapse_matrices/'
                                 'phil_dict.pkl')
        area_key = 'sizes_area'
        total_area_key = 'total_size_area'
        syn_pred_key = 'syn_types_pred_maj'
    else:
        syn_props = load_pkl2obj('/lustre/sdorkenw/synapse_matrices/'
                                 'phil_dict_all.pkl')
        area_key = 'cs_area'
        total_area_key = 'total_cs_area'
        syn_pred_key = 'syn_types_pred'
    dendrite_ids = set()
    pure_dendrite_ids = set()
    axon_ids = set()
    pure_axon_ids = set()
    dendrite_multiple_syns_ids = set()
    axon_multiple_syns_ids = set()
    axon_axon_ids = set()
    axon_axon_pairs = []
    for pair_name, pair in syn_props.iteritems():
        # if pair[total_area_key] != 0:
        skel_id1, skel_id2 = re.findall('(\d+)_(\d+)', pair_name)[0]
        skel_id1 = int(skel_id1)
        skel_id2 = int(skel_id2)
        if skel_id1 not in skeleton_ids or skel_id2 not in skeleton_ids:
            continue
        axon_ids.add(skel_id1)
        dendrite_ids.add(skel_id2)
        pure_axon_ids.add(skel_id1)
        pure_dendrite_ids.add(skel_id2)
        if len(pair[area_key]) > 1:
            dendrite_multiple_syns_ids.add(skel_id2)
            axon_multiple_syns_ids.add(skel_id1)
        if np.any(np.array(pair['partner_axoness']) == 1):
            axon_axon_ids.add(skel_id1)
            axon_axon_ids.add(skel_id2)
            axon_axon_pairs.append((skel_id1, skel_id2))
    all_used_ids = set()
    all_used_ids.update(axon_axon_ids)
    all_used_ids.update(axon_ids)
    all_used_ids.update(dendrite_ids)
    print "%d/%d cells have no connection between each other." %\
          (len(skeleton_ids) - len(all_used_ids), len(skeleton_ids))
    print "Using %d unique cells in wiring." % len(all_used_ids)
    axon_axon_ids = np.array(list(axon_axon_ids))
    axon_ids = np.array(list(axon_ids))
    pure_axon_ids = np.array(list(axon_ids))
    dendrite_ids = np.array(list(dendrite_ids))
    pure_dendrite_ids = np.array(list(dendrite_ids))
    axon_multiple_syns_ids = np.array(list(axon_multiple_syns_ids))
    dendrite_multiple_syns_ids = np.array(list(dendrite_multiple_syns_ids))

    # sort dendrites, axons using its type prediction. order is determined by
    # dictionaries get_id_dict_from_skel_ids
    dendrite_pred = np.array([cell_type_pred_dict[den_id] for den_id in
                              dendrite_ids])
    type_sorted_ixs = np.argsort(dendrite_pred, kind='mergesort')
    dendrite_pred = dendrite_pred[type_sorted_ixs]
    dendrite_ids = dendrite_ids[type_sorted_ixs]
    print "GP axons:", dendrite_ids[dendrite_pred==2]
    print "Ranges for dendrites[%d]: %s" % (len(dendrite_ids),
                                            class_ranges(dendrite_pred))

    pure_dendrite_pred = np.array([cell_type_pred_dict[den_id] for den_id in
                              pure_dendrite_ids])
    type_sorted_ixs = np.argsort(pure_dendrite_pred, kind='mergesort')
    pure_dendrite_pred = pure_dendrite_pred[type_sorted_ixs]
    pure_dendrite_ids = pure_dendrite_ids[type_sorted_ixs]
    print "Ranges for dendrites[%d]: %s" % (len(pure_dendrite_ids),
                                            class_ranges(pure_dendrite_pred))

    axon_pred = np.array([cell_type_pred_dict[den_id] for den_id in
                          axon_ids])
    type_sorted_ixs = np.argsort(axon_pred, kind='mergesort')
    axon_pred = axon_pred[type_sorted_ixs]
    axon_ids = axon_ids[type_sorted_ixs]
    print "Ranges for axons[%d]: %s" % (len(axon_pred), class_ranges(axon_pred))

    pure_axon_pred = np.array([cell_type_pred_dict[den_id] for den_id in
                          pure_axon_ids])
    type_sorted_ixs = np.argsort(pure_axon_pred, kind='mergesort')
    pure_axon_pred = pure_axon_pred[type_sorted_ixs]
    pure_axon_ids = pure_axon_ids[type_sorted_ixs]
    print "Ranges for axons[%d]: %s" % (len(pure_axon_ids),
                                        class_ranges(pure_axon_pred))

    ax_ax_pred = np.array([cell_type_pred_dict[ax_id] for ax_id in
                           axon_axon_ids])
    type_sorted_ixs = np.argsort(ax_ax_pred, kind='mergesort')
    ax_ax_pred = ax_ax_pred[type_sorted_ixs]
    axon_axon_ids = axon_axon_ids[type_sorted_ixs]
    print "Ranges for axons (ax-ax)[%d]: %s" % (len(ax_ax_pred),
                                                class_ranges(ax_ax_pred))

    ax_multi_syn_pred = np.array([cell_type_pred_dict[mult_syn_skel_id] for
                           mult_syn_skel_id in axon_multiple_syns_ids])
    type_sorted_ixs = np.argsort(ax_multi_syn_pred, kind='mergesort')
    ax_multi_syn_pred = ax_multi_syn_pred[type_sorted_ixs]
    axon_multiple_syns_ids = axon_multiple_syns_ids[type_sorted_ixs]
    print "Ranges for axons (multi-syn)[%d]: %s" % (len(ax_multi_syn_pred),
                                                class_ranges(ax_multi_syn_pred))

    den_multi_syn_pred = np.array([cell_type_pred_dict[mult_syn_skel_id] for
                           mult_syn_skel_id in dendrite_multiple_syns_ids])
    type_sorted_ixs = np.argsort(den_multi_syn_pred, kind='mergesort')
    den_multi_syn_pred = den_multi_syn_pred[type_sorted_ixs]
    dendrite_multiple_syns_ids = dendrite_multiple_syns_ids[type_sorted_ixs]
    print "Ranges for dendrites (multi-syn)[%d]: %s" % (len(den_multi_syn_pred),
                                                class_ranges(den_multi_syn_pred))

    den_id_dict, rev_den_id_dict = get_id_dict_from_skel_ids(dendrite_ids)
    ax_id_dict, rev_ax_id_dict = get_id_dict_from_skel_ids(axon_ids)

    # build reduced matrix
    wiring = np.zeros((len(dendrite_ids), len(axon_ids), 3), dtype=np.float)
    wiring_multiple_syns = np.zeros((len(dendrite_ids), len(axon_ids), 3),
                                    dtype=np.float)
    cum_wiring = np.zeros((4, 4, 3))
    cum_wiring_axon = np.zeros((4, 4, 3))
    wiring_axoness = np.zeros((len(dendrite_ids), len(axon_ids), 3),
                              dtype=np.float)
    for pair_name, pair in syn_props.iteritems():
        if pair[total_area_key] != 0:
            synapse_type = cell_classification(pair[syn_pred_key])
            skel_id1, skel_id2 = re.findall('(\d+)_(\d+)', pair_name)[0]
            skel_id1 = int(skel_id1)
            skel_id2 = int(skel_id2)
            if skel_id1 not in skeleton_ids or skel_id2 not in skeleton_ids:
                continue
            dendrite_pos = den_id_dict[skel_id2]
            axon_pos = ax_id_dict[skel_id1]
            cum_den_pos = cell_type_pred_dict[skel_id2]
            cum_ax_pos = cell_type_pred_dict[skel_id1]
            if np.any(np.array(pair['partner_axoness']) == 1):
                indiv_syn_sizes = np.array(pair[area_key])
                indiv_syn_axoness = np.array(pair['partner_axoness']) == 1
                axon_axon_syn_size = indiv_syn_sizes[indiv_syn_axoness]
                pair[area_key] = indiv_syn_sizes[~indiv_syn_axoness]
                pair[total_area_key] = np.sum(pair[area_key])
                y_axon_axon = np.sum(axon_axon_syn_size)
                y_axon_axon_display = np.min((y_axon_axon, max_syn_size))
                if binary:
                    y_axon_axon = 1.
                    y_axon_axon_display = 1.
                if synapse_type == 0:
                    y_entry = np.array([0, y_axon_axon, 0])
                    cum_wiring_axon[cum_den_pos, cum_ax_pos] += y_entry
                    y_entry = np.array([0, y_axon_axon_display, 0])
                else:
                    y_entry = np.array([0, 0, y_axon_axon])
                    cum_wiring_axon[cum_den_pos, cum_ax_pos] += y_entry
                    y_entry = np.array([0, 0, y_axon_axon_display])
                wiring_axoness[dendrite_pos, axon_pos] = y_entry
                if pair[total_area_key] == 0:
                    continue
            y = pair[total_area_key]
            y_display = np.min((y, max_syn_size))
            if len(pair[area_key]) > 1:
                if synapse_type == 0:
                    y_entry = np.array([0, y_display, 0])
                else:
                    y_entry = np.array([0, 0, y_display])
                wiring_multiple_syns[dendrite_pos, axon_pos] = y_entry
            if binary:
                y = 1.
                y_display = 1.
            if synapse_type == 0:
                y_entry = np.array([0, y, 0])
                cum_wiring[cum_den_pos, cum_ax_pos] += y_entry
                y_entry = np.array([0, y_display, 0])
            else:
                y_entry = np.array([0, 0, y])
                cum_wiring[cum_den_pos, cum_ax_pos] += y_entry
                y_entry = np.array([0, 0, y_display])
            wiring[dendrite_pos, axon_pos] = y_entry
    nb_axon_axon_syn = np.sum(wiring_axoness != 0)
    nb_syn = np.sum(wiring != 0)
    max_val = [np.max(wiring[..., 1]), np.max(wiring[..., 2])]
    max_val_axon_axon = [np.max(wiring_axoness[..., 1]),
                         np.max(wiring_axoness[..., 2])]
    ax_borders = class_ranges(axon_pred)[1:-1]
    den_borders = class_ranges(dendrite_pred)[1:-1]
    maj_vote = get_cell_majority_synsign(cum_wiring)
    maj_vote_axoness = get_cell_majority_synsign(cum_wiring_axon)
    print "Proportion axon-axonic:", nb_axon_axon_syn / float(nb_axon_axon_syn+nb_syn)

    # # find maximum inhibit->msn (use max_syn_size = 3.0)
    # sector_3_1 = wiring[ax_borders[0]:ax_borders[1], ax_borders[-1]:, 2]
    # max_entry = np.where(sector_3_1 == sector_3_1.max())
    # print "Maximal int->MSN:", max_entry
    # print "Syn. size:", wiring[ax_borders[0]+max_entry[0], ax_borders[-1]+max_entry[1]]
    # for i in range(len(max_entry[0])):
    #     print rev_ax_id_dict[ax_borders[0]+max_entry[0][i]], rev_ax_id_dict[ax_borders[-1]+max_entry[1][i]]
    #
    # # find maximum msn->gp (use max_syn_size = 3.0)
    # sector_1_2 = wiring[ax_borders[-2]:, ax_borders[0]:ax_borders[1], 2]
    # max_entry = np.where(sector_1_2 == sector_1_2.max())
    # print "Maximal MSN->GP:", max_entry
    # print "Syn. size:", wiring[ax_borders[-2]+max_entry[0], ax_borders[0]+max_entry[1]]
    # for i in range(len(max_entry[0])):
    #     print rev_ax_id_dict[ax_borders[-2]+max_entry[0][i]], rev_ax_id_dict[ax_borders[0]+max_entry[1][i]]

    # # find maximum axon->msn (use max_syn_size = 3.0)
    # sector_0_1 = wiring[ax_borders[0]:ax_borders[1], 0:ax_borders[0], 2]
    # max_entry = np.where(sector_0_1 == sector_0_1.max())
    # print "Maximal exax->MSN:", max_entry
    # print "Syn. size:", wiring[ax_borders[0]+max_entry[0], max_entry[1]]
    # for i in range(len(max_entry[0])):
    #     print rev_ax_id_dict[ax_borders[0]+max_entry[0][i]], rev_ax_id_dict[max_entry[1][i]]
    #
    # # # find maximum msn->msn (use max_syn_size = 3.0)
    # sector_0_1 = wiring[ax_borders[0]:ax_borders[1], ax_borders[0]:ax_borders[1], 2]
    # max_entry = np.where(sector_0_1 == sector_0_1.max())
    # print "Maximal MSN->MSN:", max_entry
    # print "Syn. size:", wiring[ax_borders[0]+max_entry[0], ax_borders[0]+max_entry[1]]
    # for i in range(len(max_entry[0])):
    #     print rev_ax_id_dict[ax_borders[0]+max_entry[0][i]], rev_ax_id_dict[ax_borders[0]+max_entry[1][i]]
    #     print cell_type_pred_dict[rev_ax_id_dict[ax_borders[0]+max_entry[0][i]]], cell_type_pred_dict[rev_ax_id_dict[ax_borders[0]+max_entry[1][i]]]

    # normalize each channel
    if not binary:
        wiring[:, :, 1] /= max_val[0]
        wiring[:, :, 2] /= max_val[1]
        wiring_axoness[:, :, 1] /= max_val_axon_axon[0]
        wiring_axoness[:, :, 2] /= max_val_axon_axon[1]
        wiring_multiple_syns[:, :, 1] /= max_val[0]
        wiring_multiple_syns[:, :, 2] /= max_val[1]
    max_val_sym = 0.2
    # # get max MSN-> MSN:
    print "MSN->MSN"
    entry_1 = ax_id_dict[382] # ax_borders[0] + 75 #(ax_borders[0]+max_entry[0])[0]
    entry_2 = ax_id_dict[164] #ax_borders[2] + 6 #(ax_borders[2]+max_entry[1])[0]
    print rev_ax_id_dict[entry_2], cell_type_pred_dict[rev_ax_id_dict[entry_2]]
    print rev_ax_id_dict[entry_1], cell_type_pred_dict[rev_ax_id_dict[entry_1]]
    print "Synapse size:", wiring[entry_1, entry_2]
    msn_msn_row = entry_2
    msn_msn_col = entry_1
    # # get max int-> MSN:
    print "Int->MSN"
    entry_1 = ax_id_dict[371] # ax_borders[0] + 75 #(ax_borders[0]+max_entry[0])[0]
    entry_2 = ax_id_dict[472] #ax_borders[2] + 6 #(ax_borders[2]+max_entry[1])[0]
    print rev_ax_id_dict[entry_2], cell_type_pred_dict[rev_ax_id_dict[entry_2]]
    print rev_ax_id_dict[entry_1], cell_type_pred_dict[rev_ax_id_dict[entry_1]]
    print "Synapse size:", wiring[entry_1, entry_2]
    int_row = entry_2
    int_col = entry_1
    # # get max MSN->gp:
    print "MSN->GP"
    entry_1 = ax_id_dict[578] #[1]ax_borders[1] + 3#190 #(max_entry[0])[0]
    entry_2 = ax_id_dict[1] #ax_borders[0] + 93#371 #(max_entry[1])[0]
    print rev_ax_id_dict[entry_2], cell_type_pred_dict[rev_ax_id_dict[entry_2]]
    print rev_ax_id_dict[entry_1], cell_type_pred_dict[rev_ax_id_dict[entry_1]]
    print "Synapse size:", wiring[entry_1, entry_2]
    msn_gp_row = entry_2
    msn_gp_col = entry_1
    # # Get rows of GP and MSN cell, close up:
    gp_row = ax_id_dict[241]
    msn_row = ax_id_dict[31]
    gp_col = ax_id_dict[190]
    msn_col = ax_id_dict[496]
    get_close_up(wiring[:, (msn_row, gp_row, int_row, msn_gp_row, msn_msn_row)],
                 den_borders, [gp_col, msn_col, int_col, msn_gp_col, msn_msn_col])
    # print "Wrote clouse up of gp in row %d and msn in row %d." % (gp_row, msn_row)
    print "Found %d synapses." % np.sum(wiring != 0)
    if not syn_only:
        supp += '_CS'
        plot_wiring_cs(wiring, den_borders, ax_borders, max_val, confidence_lvl,
                    binary, add_fname=supp)
        plot_wiring_cs(wiring_axoness, den_borders, ax_borders, max_val_axon_axon,
                    confidence_lvl, binary, add_fname=supp+'_axon_axon')

        plot_wiring_cum_cs(cum_wiring, class_ranges(pure_dendrite_pred),
                        class_ranges(pure_axon_pred), confidence_lvl, binary,
                        add_fname=supp)

        plot_wiring_cum_cs(cum_wiring_axon, class_ranges(ax_ax_pred),
                        class_ranges(ax_ax_pred), confidence_lvl, binary,
                        add_fname=supp+'_axon_axon')

        plot_wiring_cs(wiring_multiple_syns, den_borders, ax_borders, max_val,
                    confidence_lvl, binary, add_fname=supp+'_multiple_syns')
    else:
        supp += ''
        plot_wiring(wiring, den_borders, ax_borders, max_val, confidence_lvl,
                    binary, add_fname=supp, big_entries=big_entries,
                    maj_vote=maj_vote)

        plot_wiring(wiring_axoness, den_borders, ax_borders, max_val_axon_axon,
                    confidence_lvl, binary, add_fname=supp+'_axon_axon',
                    big_entries=big_entries, maj_vote=maj_vote_axoness)

        plot_wiring_cum(cum_wiring, class_ranges(dendrite_pred),
                        class_ranges(axon_pred), confidence_lvl, binary,
                        add_fname=supp, max_val_sym=max_val_sym,
                        maj_vote=maj_vote)

        plot_wiring_cum(cum_wiring_axon, class_ranges(ax_ax_pred),
                        class_ranges(ax_ax_pred), confidence_lvl, binary,
                        add_fname=supp+'_axon_axon', maj_vote=maj_vote_axoness)

        plot_wiring(wiring_multiple_syns, den_borders, ax_borders, max_val,
                    confidence_lvl, binary, add_fname=supp+'_multiple_syns',
                    big_entries=big_entries, maj_vote=maj_vote)


def get_cell_majority_synsign(cum_wiring):
    cum_rows = np.sum(cum_wiring, axis=0)
    maj_vote = np.zeros((4))
    for i in range(4):
        maj_vote[i] = cum_rows[i, 2] > cum_rows[i, 1]
    return maj_vote


def get_close_up(wiring, den_borders, col_entries):
    for k, b in enumerate(den_borders):
        b += k * 1
        wiring = np.concatenate((wiring[:b, :], np.zeros((1, wiring.shape[1], 3)),
                                 wiring[b:, :]), axis=0)
    closeup = np.zeros((wiring.shape[0], len(col_entries)))
    for i in range(wiring.shape[0]):
        for j in range(wiring.shape[1]):
            if wiring[i, j, 1] != 0:
                closeup[i, j] = -wiring[i, j, 1]
            elif wiring[i, j, 2] != 0:
                closeup[i, j] = wiring[i, j, 2]
    # closeup = closeup[::-1]
    matplotlib.rcParams.update({'font.size': 14})
    fig = pp.figure()
    # Create scatter plot
    gs = gridspec.GridSpec(1, 2, width_ratios=[20, 1])
    gs.update(wspace=0.05, hspace=0.08)
    ax = pp.subplot(gs[0, 0], frameon=False)
    #dark_blue = sns.diverging_palette(133, 255, center="light", as_cmap=True)
    dark_blue = sns.diverging_palette(282., 120, s=99., l=50.,
                                      center="light", as_cmap=True)
    cax = ax.matshow(closeup.transpose(1, 0), cmap=dark_blue,
                     extent=[0, wiring.shape[0], wiring.shape[1], 0],
                     interpolation="none")
    ax.set_xlim(0, wiring.shape[0])
    ax.set_ylim(0, wiring.shape[1])
    plt.grid(False)
    plt.axis('off')
    for k, b in enumerate(den_borders):
        b += k * 1
        plt.axvline(b+0.5, color='k', lw=0.5, snap=True, antialiased=True)
    cbar_ax = pp.subplot(gs[0, 1])
    cbar_ax.yaxis.set_ticks_position('none')
    cb = fig.colorbar(cax, cax=cbar_ax, ticks=[])
    fig.savefig('/lustre/pschuber/figures/wiring/type_wiring_closeup.png',
                dpi=600)
    plt.close()
    matplotlib.rcParams.update({'font.size': 14})
    fig = pp.figure()
    # Create scatter plot
    gs = gridspec.GridSpec(1, 2, width_ratios=[20, 1])
    gs.update(wspace=0.05, hspace=0.08)
    ax = pp.subplot(gs[0, 0], frameon=False)
    #dark_blue = sns.diverging_palette(133, 255, center="light", as_cmap=True)
    dark_blue = sns.diverging_palette(282., 120, s=99., l=50.,
                                      center="light", as_cmap=True)
    cax = ax.matshow(closeup.transpose(1, 0), cmap=dark_blue,
                     extent=[0, wiring.shape[0], wiring.shape[1], 0],
                     interpolation="none")
    ax.set_xlim(0, wiring.shape[0])
    ax.set_ylim(0, wiring.shape[1])
    plt.grid(False)
    plt.axis('off')
    for k, b in enumerate(den_borders):
        b += k * 1
        plt.axvline(b+0.5, color='k', lw=0.5, snap=True, antialiased=True)
    for col_entry in col_entries:
        plt.axvline(col_entry+0.5, color="0.4", lw=0.5, snap=True, antialiased=True)
    cbar_ax = pp.subplot(gs[0, 1])
    cbar_ax.yaxis.set_ticks_position('none')
    cb = fig.colorbar(cax, cax=cbar_ax, ticks=[])
    fig.savefig('/lustre/pschuber/figures/wiring/type_wiring_closeup_marker.png',
                dpi=600)
    plt.close()


def get_cum_pos(den_ranges, ax_ranges, den_pos, ax_pos):
    """
    Calculates the position of synapse in cumulated matrix
    """
    den_cum_pos = 0
    ax_cum_pos = 0
    for i in range(1, len(den_ranges)):
        if (den_pos >= den_ranges[i-1]) and (den_pos < den_ranges[i]):
            den_cum_pos = i-1
    for i in range(1,  len(ax_ranges)):
        if (ax_pos >= ax_ranges[i-1]) and (ax_pos < ax_ranges[i]):
            ax_cum_pos = i-1
    return den_cum_pos, ax_cum_pos


def plot_wiring(wiring, den_borders, ax_borders, max_val, confidence_lvl,
                binary, big_entries=False, add_fname='', maj_vote=[]):
    """
    :param wiring:
    :param den_borders:
    :param ax_borders:
    :param max_val:
    :param confidence_lvl:
    :param binary:
    :param add_fname:
    :param big_entries: changes entries to 3x3 squares
    :return:
    """
    for k, b in enumerate(den_borders):
        b += k * 1
        wiring = np.concatenate((wiring[:b, :], np.zeros((1, wiring.shape[1], 3)),
                                 wiring[b:, :]), axis=0)
    for k, b in enumerate(ax_borders):
        b += k * 1
        wiring = np.concatenate((wiring[:, :b], np.zeros((wiring.shape[0], 1, 3)),
                                 wiring[:, b:]), axis=1)
    intensity_plot = np.zeros((wiring.shape[0], wiring.shape[1]))
    print "Found majority vote for cell types:", maj_vote
    ax_borders_h = arr([0, ax_borders[0], ax_borders[1], ax_borders[2], wiring.shape[1]])+arr([0, 1, 2, 3, 4])
    den_borders_h = arr([0, ax_borders[0], ax_borders[1], ax_borders[2], wiring.shape[0]])+arr([0, 1, 2, 3, 4])
    for i in range(wiring.shape[0]):
        for j in range(wiring.shape[1]):
            den_pos, ax_pos = get_cum_pos(ax_borders_h, ax_borders_h, i, j)
            syn_sign = maj_vote[ax_pos]
            if wiring[i, j, 1] != 0:
                intensity_plot[i, j] = (-1)**syn_sign * wiring[i, j, 1]
            elif wiring[i, j, 2] != 0:
                intensity_plot[i, j] = (-1)**syn_sign * wiring[i, j, 2]
            if big_entries:
                for add_i in [-1, 0, 1]:
                    for add_j in [-1, 0, 1]:
                        den_pos_i, ax_pos_j = get_cum_pos(
                            ax_borders_h, ax_borders_h, i+add_i, j+add_j)
                        if (i+add_i >= wiring.shape[0]) or (i+add_i < 0) or\
                            (j+add_j >= wiring.shape[1]) or (j+add_j < 0) or\
                            (den_pos_i != den_pos) or (ax_pos_j != ax_pos):
                            continue
                        if wiring[i, j, 1] != 0:
                            #if intensity_plot[i+add_i, j+add_j] >= -wiring[i, j, 1]:
                                intensity_plot[i+add_i, j+add_j] = (-1)**(syn_sign+1) * wiring[i, j, 1]
                        elif wiring[i, j, 2] != 0:
                            #if intensity_plot[i+add_i, j+add_j] <= wiring[i, j, 2]:
                                intensity_plot[i+add_i, j+add_j] = (-1)**(syn_sign+1) * wiring[i, j, 2]
    if not big_entries:
        np.save('/lustre/pschuber/figures/wiring/connectivity_matrix.npy',
                intensity_plot)
    print "Plotting wiring diagram with maxval", max_val, "and supplement", add_fname
    print "Max/Min in plot:", np.min(intensity_plot), np.max(intensity_plot)
    tmp_max_val = np.zeros((2))
    tmp_max_val[1] = np.min(intensity_plot)
    tmp_max_val[0] = np.max(intensity_plot)
    matplotlib.rcParams.update({'font.size': 14})
    fig = pp.figure()
    # Create scatter plot
    gs = gridspec.GridSpec(1, 2, width_ratios=[20, 1])
    gs.update(wspace=0.05, hspace=0.08)
    ax = pp.subplot(gs[0, 0], frameon=False)
    #dark_blue = sns.diverging_palette(133, 255, center="light", as_cmap=True)
    dark_blue = sns.diverging_palette(282., 120, s=99., l=50.,
                                      center="light", as_cmap=True)
    cax = ax.matshow(intensity_plot.transpose(1, 0), cmap=dark_blue,
                     extent=[0, wiring.shape[0], wiring.shape[1], 0],
                     interpolation="none")
    ax.set_xlabel('Post', fontsize=18)
    ax.set_ylabel('Pre', fontsize=18)
    ax.set_xlim(0, wiring.shape[0])
    ax.set_ylim(0, wiring.shape[1])
    plt.grid(False)
    plt.axis('off')

    for k, b in enumerate(den_borders):
        b += k * 1
        plt.axvline(b+0.5, color='k', lw=0.5, snap=True, antialiased=True)
    for k, b in enumerate(ax_borders):
        b += k * 1
        plt.axhline(b+0.5, color='k', lw=0.5, snap=True, antialiased=True)

    cbar_ax = pp.subplot(gs[0, 1])
    cbar_ax.yaxis.set_ticks_position('none')
    cb = fig.colorbar(cax, cax=cbar_ax, ticks=[])#[tmp_max_val[1], 0, tmp_max_val[0]])
    # if not binary:
    #     cb.ax.set_yticklabels(['         Asym[%0.3f]' % max_val[1], '0',
    #                            'Sym[%0.3f]           ' % max_val[0]], rotation=90)
    # else:
    #     pass
    #     cb.ax.set_yticklabels(['0', "Symmetric", '1'], rotation=90)
    # if not binary:
    #     cb.set_label(u'Area of Synaptic Junctions [µm$^2$]')
    # else:
    #     cb.set_label(u'Synaptic Junctions')
    #plt.show(block=False)
    plt.close()

    if not binary:
        fig.savefig('/lustre/pschuber/figures/wiring/type_wiring%s_conf'
                    'lvl%d_be%s.png' % (add_fname, int(confidence_lvl*10),
                                   str(big_entries)), dpi=600)
    else:
        fig.savefig('/lustre/pschuber/figures/wiring/type_wiring%s_conf'
            'lvl%d_be%s_binary.png' % (add_fname, int(confidence_lvl*10),
                                       str(big_entries)), dpi=600)


def plot_wiring_cum(wiring, den_borders, ax_borders, confidence_lvl,
                    binary, add_fname='', max_val_sym=None, maj_vote=[]):
    # plot intensities, averaged per sector
    nb_cells_per_sector = np.zeros((4, 4))
    intensity_plot = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            diff_den = den_borders[i+1] - den_borders[i]
            diff_ax = ax_borders[j+1] - ax_borders[j]
            nb_cells_per_sector[i, j] = diff_den * diff_ax
            if nb_cells_per_sector[i, j] != 0:
                sector_intensity = np.sum(wiring[i, j]) / nb_cells_per_sector[i, j]
            else:
                sector_intensity = 0
            syn_sign = maj_vote[j]
            if wiring[i, j, 1] > wiring[i, j, 2]:
                intensity_plot[i, j] = (-1)**(syn_sign+1) * sector_intensity
            else:
                intensity_plot[i, j] = (-1)**(syn_sign+1) * np.min((sector_intensity, 0.1))
    np.save('/lustre/pschuber/figures/wiring/cumulated_connectivity_matrix.npy',
            intensity_plot)
    ind = np.arange(4)
    intensity_plot = intensity_plot.transpose(1, 0)[::-1]
    max_val = np.array([np.max(intensity_plot),
                        np.abs(np.min(intensity_plot))])
    row_sum = np.sum(np.sum(wiring.transpose(1, 0, 2)[::-1], axis=2), axis=1)
    col_sum = np.sum(np.sum(wiring.transpose(1, 0, 2)[::-1], axis=2), axis=0)
    # for i in range(4):
    #     if row_sum[i] != 0:
    #         intensity_plot[i] /= row_sum[i]
    # intensity_plot[:, :, 1] = intensity_plot[:, :, 1] / max_val[0]
    # intensity_plot[:, :, 2] = intensity_plot[:, :, 2] / max_val[1]
    max_val_tmp = np.array([np.max(intensity_plot),
                        np.abs(np.min(intensity_plot))])
    intensity_plot[intensity_plot < 0] /= max_val_tmp[1]
    intensity_plot[intensity_plot > 0] /= max_val_tmp[0]
    print "Plotting cumulative matrix with supplement", add_fname
    print "Max/Min in plot:", np.min(intensity_plot), np.max(intensity_plot)
    print max_val
    matplotlib.rcParams.update({'font.size': 14})
    fig = pp.figure()
    # Create scatter plot
    gs = gridspec.GridSpec(2, 3, width_ratios=[10, 1, 0.5], height_ratios=[1, 10])
    gs.update(wspace=0.05, hspace=0.08)
    ax = pp.subplot(gs[1, 0], frameon=False)
    #dark_blue = sns.diverging_palette(133, 255, center="light", as_cmap=True)
    dark_blue = sns.diverging_palette(282., 120., s=99., l=50.,
                                      center="light", as_cmap=True)
    cax = ax.matshow(intensity_plot, cmap=dark_blue, extent=[0, 4, 0, 4])
    ax.grid(color='k', linestyle='-')
    cbar_ax = pp.subplot(gs[1, 2])
    cbar_ax.yaxis.set_ticks_position('none')
    cb = fig.colorbar(cax, cax=cbar_ax, ticks=[])#[-1, 0, 1])
    # cb.ax.set_yticklabels(['         Asym[%0.4f]' % max_val[1], '0',
    #                        'Sym[%0.4f]         ' % max_val[0]], rotation=90)
    # if not binary:
    #     cb.set_label(u'Average Area of Synaptic Junctions [µm$^2$]')
    # else:
    #     cb.set_label(u'Average Number of Synaptic Junctions')
    axr = pp.subplot(gs[1, 1], sharey=ax, yticks=[],
                     xticks=[],#[0, max(row_sum)],
                     frameon=False,
                     xlim=(np.min(row_sum), np.max(row_sum)), ylim=(0, 4))
    axr.tick_params(axis='x', which='major', right="off", top="off", left="off",
                    pad=10, bottom="off", labelsize=12, direction='out',
                    length=4, width=1)
    axr.spines['top'].set_visible(False)
    axr.spines['right'].set_visible(False)
    axr.spines['left'].set_visible(False)
    axr.spines['bottom'].set_visible(False)
    axr.get_xaxis().tick_bottom()
    axr.get_yaxis().tick_left()
    axr.barh(ind, row_sum[::-1], 1, color='0.6', linewidth=0)
    axt = pp.subplot(gs[0, 0], sharex=ax, xticks=[],
                     yticks=[],#[0, max(col_sum)],
                     frameon=False, xlim=(0, 4), ylim=(np.min(col_sum),
                                                       np.max(col_sum)))
    axt.tick_params(axis='y', which='major', right="off", bottom="off", top="off",
                    left="off", pad=10, labelsize=12, direction='out', length=4,
                    width=1)
    axr.spines['top'].set_visible(False)
    axr.spines['right'].set_visible(False)
    axr.spines['left'].set_visible(False)
    axr.spines['bottom'].set_visible(False)
    axt.get_xaxis().tick_bottom()
    axt.get_yaxis().tick_left()
    axt.bar(ind, col_sum, 1, color='0.6', linewidth=0)
    # plt.show(block=False)
    plt.close()
    if not binary:
        fig.savefig('/lustre/pschuber/figures/wiring/type_wiring_cum%s_conf'
                    'lvl%d.png' % (add_fname, int(confidence_lvl*10)), dpi=600)
    else:
        fig.savefig('/lustre/pschuber/figures/wiring/type_wiring_cum%s_conf'
            'lvl%d_binary.png' % (add_fname, int(confidence_lvl*10)), dpi=600)


def type_sorted_wiring_cs(gt_path='/lustre/pschuber/gt_cell_types/',
                       confidence_lvl=0.8, binary=False, max_syn_size=0.2,
                       load_gt=False):
    """
    Calculate wiring of consensus skeletons sorted by type classification
    :return:
    """
    # assert os.path.isfile(gt_path + 'wiring/skel_ids.npy'),\
    #     "Couldn't find mandatory files."
    skel_ids, skeleton_feats = load_celltype_feats(gt_path)
    skel_ids2, skel_type_probas = load_celltype_probas(gt_path)
    assert np.all(np.equal(skel_ids, skel_ids2)), "Skeleton ordering wrong for"\
                                                  "probabilities and features."
    bool_arr = np.zeros(len(skel_ids))
    cell_type_pred_dict = {}
    if load_gt:
        skel_types = load_cell_gt(skel_ids)
        bool_arr = skel_types != -1
        bool_arr = bool_arr.astype(np.bool)
        skeleton_ids = skel_ids[bool_arr]
        for k, skel_id in enumerate(skel_ids):
            cell_type_pred_dict[skel_id] = skel_types[k]
            # print skel_types[k]
    else:
        #skel_type_probas = np.load(gt_path + 'wiring/skel_type_probas.npy')
        #skeleton_feats = np.load(gt_path + 'wiring/skeleton_feats.npy')
        for k, skel_id in enumerate(skel_ids):
            cell_type_pred_dict[skel_id] = np.argmax(skel_type_probas[k])
        # load loo results of evaluation
        proba_fname = gt_path+'loo_proba_rf_2labels_False_pca_False_evenlabels_False.npy'
        probas = np.load(proba_fname)
        # get corresponding skeleton ids
        _, _, help_skel_ids = load_celltype_gt()
        # rewrite "prediction" of samples which are in trainings set with loo-proba
        for k, skel_id in enumerate(help_skel_ids):
            cell_type_pred_dict[skel_id] = np.argmax(probas[k])
        write_obj2pkl(cell_type_pred_dict, gt_path + 'cell_pred_dict.pkl')

        # remove all skeletons under confidence level
        for k, probas in enumerate(skel_type_probas):
            if np.max(probas) > confidence_lvl:
                bool_arr[k] = 1
        bool_arr = bool_arr.astype(np.bool)
        skeleton_ids = skel_ids[bool_arr]
    print "%d/%d are under confidence level %0.2f and being removed." % \
          (np.sum(~bool_arr), len(skel_ids), confidence_lvl)

    # create matrix
    syn_props = load_pkl2obj('/lustre/sdorkenw/synapse_matrices/'
                             'phil_dict_no_exclusion_all.pkl')
    area_key = 'cs_area'
    total_area_key = 'total_cs_area'
    dendrite_ids = set()
    axon_ids = set()
    for pair_name, pair in syn_props.iteritems():
        # if pair[total_area_key] != 0:
        skel_id1, skel_id2 = re.findall('(\d+)_(\d+)', pair_name)[0]
        skel_id1 = int(skel_id1)
        skel_id2 = int(skel_id2)
        if skel_id1 not in skeleton_ids or skel_id2 not in skeleton_ids:
            continue
        axon_ids.add(skel_id1)
        dendrite_ids.add(skel_id2)

    all_used_ids = set()
    all_used_ids.update(axon_ids)
    all_used_ids.update(dendrite_ids)
    print "%d/%d cells have no connection between each other." %\
          (len(skeleton_ids) - len(all_used_ids), len(skeleton_ids))
    print "Using %d unique cells in wiring." % len(all_used_ids)
    axon_ids = np.array(list(axon_ids))
    dendrite_ids = np.array(list(dendrite_ids))

    # sort dendrites, axons using its type prediction. order is determined by
    # dictionaries get_id_dict_from_skel_ids
    dendrite_pred = np.array([cell_type_pred_dict[den_id] for den_id in
                              dendrite_ids])
    type_sorted_ixs = np.argsort(dendrite_pred, kind='mergesort')
    dendrite_pred = dendrite_pred[type_sorted_ixs]
    dendrite_ids = dendrite_ids[type_sorted_ixs]
    print "GP axons:", dendrite_ids[dendrite_pred==2]
    print "Ranges for dendrites[%d]: %s" % (len(dendrite_ids),
                                            class_ranges(dendrite_pred))

    axon_pred = np.array([cell_type_pred_dict[den_id] for den_id in axon_ids])
    type_sorted_ixs = np.argsort(axon_pred, kind='mergesort')
    axon_pred = axon_pred[type_sorted_ixs]
    axon_ids = axon_ids[type_sorted_ixs]
    print "GP axons:", axon_ids[axon_pred==2]
    print "Ranges for axons[%d]: %s" % (len(axon_pred), class_ranges(axon_pred))

    den_id_dict, rev_den_id_dict = get_id_dict_from_skel_ids(dendrite_ids)
    ax_id_dict, rev_ax_id_dict = get_id_dict_from_skel_ids(axon_ids)

    wiring = np.zeros((len(dendrite_ids), len(axon_ids), 1), dtype=np.float)
    cum_wiring = np.zeros((4, 4))
    for pair_name, pair in syn_props.iteritems():
        if pair[total_area_key] != 0:
            skel_id1, skel_id2 = re.findall('(\d+)_(\d+)', pair_name)[0]
            skel_id1 = int(skel_id1)
            skel_id2 = int(skel_id2)
            if skel_id1 not in skeleton_ids or skel_id2 not in skeleton_ids:
                continue
            dendrite_pos = den_id_dict[skel_id2]
            axon_pos = ax_id_dict[skel_id1]
            cum_den_pos = cell_type_pred_dict[skel_id2]
            cum_ax_pos = cell_type_pred_dict[skel_id1]
            y = pair[total_area_key]
            if binary:
                y = 1.
            wiring[dendrite_pos, axon_pos] = np.min((y, max_syn_size))
            cum_wiring[cum_den_pos, cum_ax_pos] += y
    ax_borders = class_ranges(axon_pred)[1:-1]
    den_borders = class_ranges(dendrite_pred)[1:-1]
    supp = '_CS'
    plot_wiring_cs(wiring, den_borders, ax_borders, confidence_lvl,
                binary, add_fname=supp)

    plot_wiring_cum_cs(cum_wiring, class_ranges(dendrite_pred),
                    class_ranges(axon_pred), confidence_lvl, binary,
                    add_fname=supp)


def plot_wiring_cs(wiring, den_borders, ax_borders, confidence_lvl,
                binary, add_fname='_CS'):
    fig = plt.figure()
    ax = plt.gca()
    max_val = np.max(wiring)
    for k, b in enumerate(den_borders):
        b += k * 1
        wiring = np.concatenate((wiring[:b, :], np.ones((1, wiring.shape[1], 1)),
                                 wiring[b:, :]), axis=0)
    for k, b in enumerate(ax_borders):
        b += k * 1
        wiring = np.concatenate((wiring[:, :b], np.ones((wiring.shape[0], 1, 1)),
                                 wiring[:, b:]), axis=1)
    im = ax.matshow(np.max(wiring.transpose(1, 0, 2), axis=2), interpolation="none",
                   extent=[0, wiring.shape[0], wiring.shape[1], 0], cmap='gray')
    ax.set_xlabel('Post', fontsize=18)
    ax.set_ylabel('Pre', fontsize=18)
    ax.set_xlim(0, wiring.shape[0])
    ax.set_ylim(0, wiring.shape[1])
    plt.grid(False)
    plt.axis('off')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    a = np.array([[0, 1]])
    plt.figure()
    img = plt.imshow(a, cmap='gray')
    plt.gca().set_visible(False)
    cb = plt.colorbar(cax=cax, ticks=[0, 1])
    if not binary:
        cb.ax.set_yticklabels(['0', "%0.3g+" % max_val], rotation=90)
        cb.set_label(u'Area of Synaptic Junctions [µm$^2$]')
    else:
        cb.ax.set_yticklabels(['0', '1'], rotation=90)
        cb.set_label(u'Synaptic Junction')
    plt.close()
    if not binary:
        fig.savefig('/lustre/pschuber/figures/wiring/type_wiring%s_conf'
                    'lvl%d.png' % (add_fname, int(confidence_lvl*10)), dpi=600)
    else:
        fig.savefig('/lustre/pschuber/figures/wiring/type_wiring%s_conf'
            'lvl%d_binary.png' % (add_fname, int(confidence_lvl*10)), dpi=600)


def plot_wiring_cum_cs(wiring, den_borders, ax_borders, confidence_lvl,
                    binary, add_fname=''):
    #print "Shape of wiring", wiring.shape
    # plot cumulated wiring

    # plot intensities, averaged per sector
    nb_cells_per_sector = np.zeros((4, 4))
    intensity_plot = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            diff_den = den_borders[i+1] - den_borders[i]
            diff_ax = ax_borders[j+1] - ax_borders[j]
            nb_cells_per_sector[i, j] = diff_den * diff_ax
            if nb_cells_per_sector[i, j] != 0:
                sector_intensity = np.sum(wiring[i, j]) / nb_cells_per_sector[i, j]
            else:
                sector_intensity = 0
            intensity_plot[i, j] = sector_intensity
    ind = np.arange(4)
    intensity_plot = intensity_plot.transpose(1, 0)[::-1]
    wiring = wiring.transpose(1, 0)[::-1]
    intensity_plot[intensity_plot > 1.0] = 1.0
    max_val = np.max(intensity_plot)
    row_sum = np.sum(wiring, axis=1)
    col_sum = np.sum(wiring, axis=0)
    intensity_plot = (intensity_plot - intensity_plot.min())/\
                     np.max((intensity_plot.max() - intensity_plot.min()))

    print row_sum
    print col_sum
    from matplotlib import pyplot as pp
    from matplotlib import gridspec
    matplotlib.rcParams.update({'font.size': 14})
    fig = pp.figure()
    # Create scatter plot
    gs = gridspec.GridSpec(2, 3, width_ratios=[10, 1, 0.5], height_ratios=[1, 10])
    gs.update(wspace=0.05, hspace=0.08)
    ax = pp.subplot(gs[1, 0], frameon=False)
    cax = ax.matshow(intensity_plot, cmap='gray_r', extent=[0, 4, 0, 4])
    ax.grid(color='k', linestyle='-')
    cbar_ax = pp.subplot(gs[1, 2])
    cbar_ax.yaxis.set_ticks_position('left')
    cb = fig.colorbar(cax, cax=cbar_ax, ticks=[0, 1])
    cb.ax.set_yticklabels(['0', '%0.4f' % max_val], rotation=90)
    if not binary:
        cb.set_label(u'Average Area of Contact Sites [µm$^2$]')
    else:
        cb.set_label(u'Average Number of Contact Sites')

    axr = pp.subplot(gs[1, 1], sharey=ax, yticks=[],
                     xticks=[0, max(row_sum)], frameon=True,
                     xlim=(np.min(row_sum), np.max(row_sum)), ylim=(0, 4))
    axr.tick_params(axis='x', which='major', right="off", top="off", pad=10,
                    labelsize=12, direction='out', length=4, width=1)
    axr.spines['top'].set_visible(False)
    axr.spines['right'].set_visible(False)
    axr.get_xaxis().tick_bottom()
    axr.get_yaxis().tick_left()
    axr.barh(ind, row_sum[::-1], 1, color='0.6', linewidth=0)
    axt = pp.subplot(gs[0, 0], sharex=ax, xticks=[],
                     yticks=[0, max(col_sum)],
                     frameon=True, xlim=(0, 4), ylim=(np.min(col_sum),
                                                       np.max(col_sum)))
    axt.tick_params(axis='y', which='major', right="off", bottom="off", pad=10,
                    labelsize=12, direction='out', length=4, width=1)
    axt.spines['top'].set_visible(False)
    axt.spines['right'].set_visible(False)
    axt.get_xaxis().tick_bottom()
    axt.get_yaxis().tick_left()
    axt.bar(ind, col_sum, 1, color='0.6', linewidth=0)
    plt.show(block=False)
    if not binary:
        fig.savefig('/lustre/pschuber/figures/wiring/type_wiring_cum%s_conf'
                    'lvl%d.png' % (add_fname, int(confidence_lvl*10)), dpi=600)
    else:
        fig.savefig('/lustre/pschuber/figures/wiring/type_wiring_cum%s_conf'
            'lvl%d_binary.png' % (add_fname, int(confidence_lvl*10)), dpi=600)


def class_ranges(pred_arr):
    if len(pred_arr) == 0:
        return np.array([0, 0, 0, 0, 0])
    class1 = np.argmax(pred_arr == 1)
    class2 = np.max((class1, np.argmax(pred_arr == 2)))
    class3 = np.max((class2, np.argmax(pred_arr == 3)))
    return np.array([0, class1, class2, class3, len(pred_arr)])


def sort_matrix(matrix):
    pairwise_dist = scipy.spatial.distance.pdist(matrix)
    pairwise_dist = scipy.spatial.distance.squareform(pairwise_dist)
    all_sorts = []
    values = []
    for i in range(1):#matrix.shape[0]):
        value = 0
        sorting = []
        mask = np.zeros((len(matrix),), dtype=np.int)
        el = i
        while True:
            mask[el] = 1
            sorting.append(el)
            curr_dec = np.array(pairwise_dist[el, :])
            curr_dec[mask.astype(np.bool)] = np.inf
            el = np.argmin(curr_dec)
            if len(sorting) == len(matrix):
                break
            value += pairwise_dist[sorting[-1], el]
        values.append(value)
        all_sorts.append(sorting)
    best_sorting = all_sorts[np.argmin(values)]
    return best_sorting[::-1]


def draw_nxgraph(G):

    #first compute the best partition
    partition = community.best_partition(G)

    #drawing
    size = float(len(set(partition.values())))
    pos = nx.spring_layout(G)
    count = 0.
    for com in set(partition.values()) :
        count = count + 1.
        list_nodes = [nodes for nodes in partition.keys()
                                    if partition[nodes] == com]
        nx.draw_networkx_nodes(G, pos, list_nodes, node_size = 20,
                                    node_color = str(count / size))

    nx.draw_networkx_edges(G, pos, alpha=0.5)
    plt.show()


def calc_wiring_matrix(pre_dict, all_post_ids):
    G = nx.Graph()
    all_post_ids = list(set(all_post_ids))
    # x axis is post synapse, y axis pre
    nb_pre_ids = len(pre_dict.keys())
    pre_dict_keys = pre_dict.keys()
    nb_post_ids = len(all_post_ids)
    area_matrix = np.zeros((nb_pre_ids, nb_post_ids))
    size_matrix = np.zeros((nb_pre_ids, nb_post_ids))
    rel_size_matrix = np.zeros((nb_pre_ids, nb_post_ids))
    wiring_matrix = np.zeros((nb_pre_ids, nb_post_ids))
    post_id_dict = {}
    pre_id_dict = {}
    cnt = 0
    for id in all_post_ids:
        post_id_dict[id] = cnt
        cnt += 1
    cnt = 0
    for id in pre_dict_keys:
        pre_id_dict[id] = cnt
        cnt += 1
    for pre_id in pre_dict_keys:
        syns = pre_dict[pre_id]
        for key in syns.keys():
            syn = syns[key]
            post_id = syn['post_id']
            x = pre_id_dict[pre_id]
            y = post_id_dict[post_id]
            area_matrix[x, y] = (syn['cs_area'])
            size_matrix[x, y] = (syn['az_size_abs'])
            rel_size_matrix[x, y] = syn['az_size_rel']
            wiring_matrix[x, y] += 1
            G.add_edge(pre_id, post_id, weight=float(syn['az_size_rel']))
    size_std = np.std(size_matrix)
    size_mean = np.mean(size_matrix)
    max_size = size_mean + 3*size_std
    area_std = np.std(area_matrix)
    area_mean = np.mean(area_matrix)
    max_area = area_mean + 3*area_std
    area_matrix[area_matrix > max_area] = max_area
    size_matrix[size_matrix > max_size] = max_size
    # corr
    t_area = np.linspace(area_matrix.min(), area_matrix.max(), 500)[:100]
    t_size = np.linspace(size_matrix.min(), size_matrix.max(), 500)[:100]
    #bin_width = (area_matrix.max()-area_matrix.min())/500
    #print (size_matrix.max()-size_matrix.min())/500*9*9*20./1e9, size_matrix.min()*9*9*20./1e9
    corr_matrix = np.zeros((len(t_area), len(t_size)))
    pval_matrix = np.zeros((len(t_area), len(t_size)))
    for i, t1 in enumerate(t_area):
        for j, t2 in enumerate(t_size):
            rho, p_val = pearsonr(arr(area_matrix.flatten()>t1, dtype=np.int),
                                   arr(size_matrix.flatten()>t2, dtype=np.int))
            corr_matrix[i,j] = rho #np.nan_to_num(rho)
            pval_matrix[i,j] = p_val #np.nan_to_num(p_val)
    return area_matrix, size_matrix, corr_matrix, G


def prec_rec_syns_with_csarea(pre_dict_all, pre_dict_syns):
    X = []
    y = []
    for syns in pre_dict_syns.values():
        for syn in syns.values():
            X.append(syn['cs_area'])
            y.append(1)
    for syns in pre_dict_all.values():
        for syn in syns.values():
            X.append(syn['cs_area'])
            y.append(0)
    pr, re, t = precision_recall_curve(y, X)
    plot_pr(pr, re, r=[0.0, 1.01])
    plt.savefig('/lustre/pschuber/figures/syns/syn_csarea_predictor.png', dpi=300)


def wiring_diagram_old(cs_dir='/lustre/pschuber/m_consensi_relu/'
                            'nml_obj/contact_sites_new/', recompute=False):
    """
    Create a wiring diagram out of the contact site nml. Parses all
    neccessary information. Only uses contact sites containing a p4 object in order
    to assign pre and post synaptic side.
    :param csite_nml_path: str path to contact_site.nml
    :return: array of size n x n, with n skeletons containing a p4
    """
    if recompute:
        #features, axoness_info = feature_valid_syns(cs_dir)
        features, axoness_info, syn_pred = feature_valid_syns(cs_dir, only_az=True)
        features_cs, axoness_info_cs, syn_pred_cs = feature_valid_syns(cs_dir, only_az=False)
        features_cs = np.concatenate((features_cs, features[~syn_pred]), axis=0)
        axoness_info_cs = np.concatenate((axoness_info_cs, axoness_info[~syn_pred]),
                                         axis=0)
        features_cs, axoness_info_cs, pre_dict_cs, all_post_ids_cs,\
        valid_syn_array_cs, ax_dict = calc_syn_dict(features_cs, axoness_info_cs,
                                                    get_all=True)
        features, axoness_info, pre_dict, all_post_ids, valid_syn_array,\
            ax_dict = calc_syn_dict(features[syn_pred], axoness_info[syn_pred])
        prec_rec_syns_with_csarea(pre_dict_cs, pre_dict)
        pre_dict_cs_keys = pre_dict_cs.keys()
        all_post_ids_cs += all_post_ids
        for pre_id, val in pre_dict.iteritems():
            old_syns = pre_dict[pre_id]
            if pre_id in pre_dict_cs_keys:
                syns = pre_dict_cs[pre_id]
                for post_id, old_syn in old_syns.iteritems():
                    if post_id in syns.keys():
                        syns[post_id]['cs_area'] += old_syn['cs_area']
                        syns[post_id]['az_size_abs'] += old_syn['az_size_abs']
                    else:
                        syns[post_id] = old_syn
            else:
                pre_dict_cs[pre_id] = old_syns
        area_matrix, size_matrix, corr_matrix, G = \
            calc_wiring_matrix(pre_dict_cs, all_post_ids_cs)
        np.save(cs_dir+'area_matrix.npy', area_matrix)
        np.save(cs_dir+'size_matrix.npy', size_matrix)
        np.save(cs_dir+'corr_matrix.npy', corr_matrix)
        nx.write_graphml(G, cs_dir+"graph_vis.graphml")
    else:
        area_matrix = np.load(cs_dir+'area_matrix.npy')
        size_matrix = np.load(cs_dir+'size_matrix.npy')
        corr_matrix = np.load(cs_dir+'corr_matrix.npy')
        G = nx.read_graphml(cs_dir+"graph_vis.graphml")
    # ordering
    # sorting = sort_matrix(size_matrix.T)
    # size_matrix = size_matrix[:, sorting]
    # area_matrix = area_matrix[:, sorting]
    # ordering_x = np.argsort(np.sum(wiring_matrix, axis=0))[::-1]
    # wiring_matrix = wiring_matrix[:, ordering_x]
    # ordering_y = np.argsort(np.sum(wiring_matrix, axis=1))
    #
    # wiring_matrix = wiring_matrix[ordering_y, :]
    directory_path = os.path.dirname(cs_dir)
    # cs area
    #axon_ordering = np.argsort(np.sum(size_matrix, axis=1))[::-1]
    plt.figure()
    ax = plt.gca()
    im = ax.matshow(area_matrix, cmap='gnuplot2_r')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.15)
    #ax.set_title('Wiring Matrix -- Contact Site Area')
    ax.set_xlabel('post', fontsize=18)
    ax.set_ylabel('pre', fontsize=18)
    cb = plt.colorbar(im, cax=cax)
    cb.set_label(u'CS area / µm$^2$')
    plt.savefig(directory_path + '/../../../figures/wiring_matrix'
                                 '/wiring_matrix_cs_area.png', dpi=300)
    plt.close()
    plt.figure()
    ax = plt.gca()
    im = ax.matshow(size_matrix * 9*9*20./1e9, cmap='gnuplot2_r', interpolation="none")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.15)
    # ax.set_title('Wiring Matrix -- Active Zone Overlap (abs.)')
    ax.set_xlabel('post', fontsize=18)
    ax.set_ylabel('pre', fontsize=18)
    cb = plt.colorbar(im, cax=cax)
    cb.set_label(u'size of synaptic junction [µm$^3$]')
    plt.savefig(directory_path + '/../../../figures/wiring_matrix'
                                 '/wiring_matrix_az_overlap_abs.png', dpi=300)
    plt.close()
    draw_nxgraph(G)
    # axon_ordering = np.argsort(np.sum(size_matrix, axis=0))[::-1]
    # size_matrix = size_matrix[:, axon_ordering]
    # area_matrix = area_matrix[:, axon_ordering]
    # plt.figure()
    # ax = plt.gca()
    # im = ax.matshow(area_matrix[axon_ordering, :], cmap='gnuplot')
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="5%", pad=0.15)
    # #ax.set_title('Wiring Matrix -- Contact Site Area')
    # ax.set_xlabel('post', fontsize=18)
    # ax.set_ylabel('pre', fontsize=18)
    # cb  = plt.colorbar(im, cax=cax)
    # cb.set_label(u'CS area / µm$^2$')
    # plt.savefig(directory_path + '/../../../figures/wiring_matrix'
    #                              '/0_wiring_matrix_cs_area.png', dpi=300)
    # plt.close()
    # plt.figure()
    # ax = plt.gca()
    # im = ax.matshow(size_matrix * 9*9*20./1e9, cmap='gnuplot', interpolation="none")
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="5%", pad=0.15)
    # # ax.set_title('Wiring Matrix -- Active Zone Overlap (abs.)')
    # ax.set_xlabel('post', fontsize=18)
    # ax.set_ylabel('pre', fontsize=18)
    # cb  = plt.colorbar(im, cax=cax)
    # cb.set_label(u'size of synaptic junction [µm$^3$]')
    # plt.savefig(directory_path + '/../../../figures/wiring_matrix'
    #                              '/0_wiring_matrix_az_overlap_abs.png', dpi=300)
    # plt.close()
    # Synapses
    # plt.matshow(wiring_matrix, cmap='gray')
    # plt.title('Wiring Matrix -- Synapse')
    # plt.xlabel('post')
    # plt.ylabel('pre')
    # plt.savefig(directory_path + '/../../../figures/wiring_matrix.png')
    # plt.close()
    # Az size

    # correlation data
    # plt.figure()
    # ax = plt.gca()
    # im = ax.matshow(corr_matrix, cmap='gnuplot')
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="5%", pad=0.15)
    # #ax.set_title('Spearman Correlation Matrix')
    # ax.set_xlabel('area threshold', fontsize=18)
    # ax.set_ylabel('size threshold', fontsize=18)
    # cb  = plt.colorbar(im, cax=cax)
    # cb.set_label('Spearman Correlation')
    # plt.savefig(directory_path + '/../../../figures/wiring_matrix/'
    #                              'corr_matrix.png', dpi=300)
    # plt.close()
    # plt.figure()
    # ax = plt.gca()
    # im = ax.matshow(rel_size_matrix, cmap='gnuplot')
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="5%", pad=0.15)
    # ax.set_title('Wiring Matrix -- Active Zone Overlap (rel.)')
    # ax.set_xlabel('post')
    # ax.set_ylabel('pre')
    # cb  = plt.colorbar(im, cax=cax)
    # cb.set_label('overlap / 1')
    # plt.savefig(directory_path + '/../../../figures/wiring_matrix_az_overlap_rel.png')
    # plt.close()
    return


def get_cs_of_mapped_skel(skel_path):
    """
    Gather all contact site of mapped skeleton at skel_path and writes .nml to
    */nml_obj/cs_of_skel*.nml
    :param skel_path: str Path to k.zip
    """
    dir, filename = os.path.split(skel_path)
    skel_id = re.findall('iter_\d+_(\d+)-', filename)[0]
    contact_sites_of_skel = NewSkeleton()
    contact_sites_of_skel.scaling = [9, 9, 20]
    paths = get_filepaths_from_dir(dir+'/contact_sites/', ending='skel_'+skel_id)
    paths += get_filepaths_from_dir(dir+'/contact_sites/', ending=skel_id+'.nml')
    for path in paths:
        anno = au.loadj0126NML(path)[0]
        contact_sites_of_skel.add_annotation(anno)
    print "Writing file" + dir + '/cs_of_skel%s.nml' % skel_id
    contact_sites_of_skel.toNml(dir+'/cs_of_skel%s.nml' % skel_id)
    return
