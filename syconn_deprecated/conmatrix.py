# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max-Planck-Institute for Medical Research, Heidelberg, Germany
# Authors: Sven Dorkenwald, Philipp Schubert, Joergen Kornfeld

import matplotlib
matplotlib.use('Agg')
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import re
import os
from matplotlib import gridspec
from matplotlib import pyplot as pp
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy import array as arr

from syconn.contactsites import conn_dict_wrapper
from syconn.processing.cell_types import load_celltype_probas, \
    get_id_dict_from_skel_ids
from syconn.processing.learning_rfc import cell_classification
from syconn.utils.datahandler import load_pkl2obj


def type_sorted_wiring(wd, confidence_lvl=0.3, binary=False, max_syn_size=0.4,
                       syn_only=True, big_entries=True):
    """Calculate wiring of consensus skeletons sorted by predicted
    cell type classification and axoness prediction

    Parameters
    ----------
    wd : str
    confidence_lvl : float
        minimum probability of cell type prediction to keep cell
    binary : bool
        if True existence of synapse is weighted by 1, else 0
    max_syn_size : float
        maximum cumulated synapse size shown in plot
    syn_only : bool
        take only contact sites with synapse classification result of 1 into
        account
    big_entries : bool
        artificially increase pixel size from 1 to 3 for better visualization
    """

    if not os.path.exists(wd + "/figures/"):
        os.makedirs(wd + "/figures/")

    supp = ""
    skeleton_ids, cell_type_probas = load_celltype_probas(wd)
    cell_type_pred_dict = load_pkl2obj(wd + '/neurons/celltype_pred_dict.pkl')
    bool_arr = np.zeros(len(skeleton_ids))
    # remove all skeletons under confidence level
    for k, probas in enumerate(cell_type_probas):
        if np.max(probas) > confidence_lvl:
            bool_arr[k] = 1
    bool_arr = bool_arr.astype(np.bool)
    skeleton_ids = skeleton_ids[bool_arr].tolist()
    # print "%d/%d are under confidence level %0.2f and being removed." % \
    #       (np.sum(~bool_arr), len(skeleton_ids), confidence_lvl)
    if not os.path.isfile(wd + '/contactsites/connectivity_dict.pkl'):
        conn_dict_wrapper(wd, all=False)
        conn_dict_wrapper(wd, all=True)
    # create matrix
    if syn_only:
        syn_props = load_pkl2obj(wd + '/contactsites/connectivity_dict.pkl')
        area_key = 'sizes_area'
        total_area_key = 'total_size_area'
        syn_pred_key = 'syn_types_pred_maj'
    else:
        syn_props = load_pkl2obj(wd + '/contactsites/connectivity_dict_all.pkl')
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
    # print "%d/%d cells have no connection between each other." %\
    #       (len(skeleton_ids) - len(all_used_ids), len(skeleton_ids))
    # print "Using %d unique cells in wiring." % len(all_used_ids)
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

    pure_dendrite_pred = np.array([cell_type_pred_dict[den_id] for den_id in
                              pure_dendrite_ids])
    type_sorted_ixs = np.argsort(pure_dendrite_pred, kind='mergesort')
    pure_dendrite_pred = pure_dendrite_pred[type_sorted_ixs]
    pure_dendrite_ids = pure_dendrite_ids[type_sorted_ixs]

    axon_pred = np.array([cell_type_pred_dict[den_id] for den_id in
                          axon_ids])
    type_sorted_ixs = np.argsort(axon_pred, kind='mergesort')
    axon_pred = axon_pred[type_sorted_ixs]
    axon_ids = axon_ids[type_sorted_ixs]

    pure_axon_pred = np.array([cell_type_pred_dict[den_id] for den_id in
                              pure_axon_ids])
    type_sorted_ixs = np.argsort(pure_axon_pred, kind='mergesort')
    pure_axon_pred = pure_axon_pred[type_sorted_ixs]
    pure_axon_ids = pure_axon_ids[type_sorted_ixs]

    ax_ax_pred = np.array([cell_type_pred_dict[ax_id] for ax_id in
                           axon_axon_ids])
    type_sorted_ixs = np.argsort(ax_ax_pred, kind='mergesort')
    ax_ax_pred = ax_ax_pred[type_sorted_ixs]

    ax_multi_syn_pred = np.array([cell_type_pred_dict[mult_syn_skel_id] for
                           mult_syn_skel_id in axon_multiple_syns_ids])
    type_sorted_ixs = np.argsort(ax_multi_syn_pred, kind='mergesort')
    ax_multi_syn_pred = ax_multi_syn_pred[type_sorted_ixs]

    den_multi_syn_pred = np.array([cell_type_pred_dict[mult_syn_skel_id] for
                           mult_syn_skel_id in dendrite_multiple_syns_ids])
    type_sorted_ixs = np.argsort(den_multi_syn_pred, kind='mergesort')
    den_multi_syn_pred = den_multi_syn_pred[type_sorted_ixs]

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
    max_val = [np.max(wiring[..., 1]), np.max(wiring[..., 2])]
    max_val_axon_axon = [np.max(wiring_axoness[..., 1]),
                         np.max(wiring_axoness[..., 2])]
    ax_borders = class_ranges(axon_pred)[1:-1]
    den_borders = class_ranges(dendrite_pred)[1:-1]
    maj_vote = get_cell_majority_synsign(cum_wiring)
    maj_vote_axoness = get_cell_majority_synsign(cum_wiring_axon)
    # normalize each channel
    if not binary:
        wiring[:, :, 1] /= max_val[0]
        wiring[:, :, 2] /= max_val[1]
        wiring_axoness[:, :, 1] /= max_val_axon_axon[0]
        wiring_axoness[:, :, 2] /= max_val_axon_axon[1]
        wiring_multiple_syns[:, :, 1] /= max_val[0]
        wiring_multiple_syns[:, :, 2] /= max_val[1]

    if not syn_only:
        supp += '_CS'
        plot_wiring_cs(wiring, den_borders, ax_borders, confidence_lvl,
                    binary, wd, add_fname=supp)

        plot_wiring_cs(wiring_axoness, den_borders, ax_borders,
                    confidence_lvl, binary, wd, add_fname=supp+'_axon_axon')

        plot_wiring_cum_cs(cum_wiring, class_ranges(pure_dendrite_pred),
                           class_ranges(pure_axon_pred), confidence_lvl, binary,
                           wd, add_fname=supp)

        plot_wiring_cum_cs(cum_wiring_axon, class_ranges(ax_ax_pred),
                        class_ranges(ax_ax_pred), confidence_lvl, binary, wd,
                        add_fname=supp+'_axon_axon')

        plot_wiring_cs(wiring_multiple_syns, den_borders, ax_borders,
                       confidence_lvl, binary, wd,
                       add_fname=supp+'_multiple_syns')
    else:
        supp += ''
        plot_wiring(wiring, den_borders, ax_borders, max_val, confidence_lvl,
                    binary, wd, add_fname=supp, big_entries=big_entries,
                    maj_vote=maj_vote)

        # plot_wiring(wiring_axoness, den_borders, ax_borders, max_val_axon_axon,
        #             confidence_lvl, binary, wd, add_fname=supp+'_axon_axon',
        #             big_entries=big_entries, maj_vote=maj_vote_axoness)

        plot_wiring_cum(cum_wiring, class_ranges(dendrite_pred),
                        class_ranges(axon_pred), confidence_lvl, max_val,
                        binary, wd, add_fname=supp, maj_vote=maj_vote)

        plot_wiring(wiring_multiple_syns, den_borders, ax_borders, max_val,
                    confidence_lvl, binary, wd, add_fname=supp+'_multiple_syns',
                    big_entries=big_entries, maj_vote=maj_vote)

        return cum_wiring


def get_cell_majority_synsign(avg_wiring):
    """Calculates majority synaptic sign of rows in average wiring

    Parameters
    ----------
    avg_wiring : np.array
        averaged wiring

    Returns
    -------
    np.array of int
        majority vote of synapse sign (row wise)
    """
    cum_rows = np.sum(avg_wiring, axis=0)
    maj_vote = np.zeros((4))
    for i in range(4):
        maj_vote[i] = cum_rows[i, 2] > cum_rows[i, 1]
    return maj_vote


def get_cum_pos(den_ranges, ax_ranges, den_pos, ax_pos):
    """Calculates the position of synapse in average matrix, i.e. which sector
    it belongs to.
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
                binary, wd, big_entries=True, add_fname='', maj_vote=()):
    """Plot type sorted connectivity matrix and save to figures folder in
    working directory

    Parameters
    ----------
    wiring : np.array
        symmetric 2D array of size #cells x #cells
    den_borders:
    cell type boarders on post synaptic site
    ax_borders:
        cell type boarders on pre synaptic site
    max_val : float
        maximum cumulated contact area shown in plot
    confidence_lvl : float
        minimum probability of cell type prediction to keep cell
    binary : bool
        if True existence of synapse is weighted by 1, else 0
    add_fname : str
        supplement of image file
    maj_vote : tuple

    big_entries : bool
        artificially increase pixel size from 1 to 3 for better visualization
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
    ax_borders_h = arr([0, ax_borders[0], ax_borders[1], ax_borders[2],
                        wiring.shape[1]])+arr([0, 1, 2, 3, 4])
    wiring *= -1
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
                            intensity_plot[i+add_i, j+add_j] = \
                                (-1)**(syn_sign+1) * wiring[i, j, 1]
                        elif wiring[i, j, 2] != 0:
                            intensity_plot[i+add_i, j+add_j] = \
                                (-1)**(syn_sign+1) * wiring[i, j, 2]
    if not big_entries:
        np.save(wd + '/figures/connectivity_matrix.npy',
                intensity_plot)
    tmp_max_val = np.max(np.abs(intensity_plot))
    matplotlib.rcParams.update({'font.size': 14})
    fig = pp.figure()
    # Create scatter plot
    gs = gridspec.GridSpec(1, 2, width_ratios=[20, 1])
    gs.update(wspace=0.05, hspace=0.08)
    ax = pp.subplot(gs[0, 0], frameon=False)

    cax = ax.matshow(-intensity_plot.transpose(1, 0), cmap=diverge_map(),
                     extent=[0, wiring.shape[0], wiring.shape[1], 0],
                     interpolation="none", vmin=-tmp_max_val,
                     vmax=tmp_max_val)
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
    cb = fig.colorbar(cax, cax=cbar_ax, ticks=[])
    plt.close()

    if not binary:
        fig.savefig(wd + '/figures/type_wiring%s_conf'
                    'lvl%d_be%s.png' % (add_fname, int(confidence_lvl*10),
                                   str(big_entries)), dpi=600)
    else:
        fig.savefig(wd + '/figures/type_wiring%s_conf'
            'lvl%d_be%s_binary.png' % (add_fname, int(confidence_lvl*10),
                                       str(big_entries)), dpi=600)


def plot_wiring_cum(wiring, den_borders, ax_borders, confidence_lvl, max_val,
                    binary, wd, add_fname='', maj_vote=()):
    """Plot wiring diagram on celltype-to-celltype level, e.g. connectivity
    between EA and MSN
    """
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
    np.save(wd + '/figures/cumulated_connectivity_matrix.npy',
            intensity_plot)
    ind = np.arange(4)
    intensity_plot = intensity_plot.transpose(1, 0)[::-1]
    row_sum = np.sum(np.sum(wiring.transpose(1, 0, 2)[::-1], axis=2), axis=1)
    col_sum = np.sum(np.sum(wiring.transpose(1, 0, 2)[::-1], axis=2), axis=0)
    max_val_tmp = np.array([np.max(intensity_plot),
                            np.abs(np.min(intensity_plot))])
    intensity_plot[intensity_plot < 0] /= max_val_tmp[1]
    intensity_plot[intensity_plot > 0] /= max_val_tmp[0]
    matplotlib.rcParams.update({'font.size': 14})
    fig = pp.figure()
    # Create scatter plot
    gs = gridspec.GridSpec(2, 3, width_ratios=[10, 1, 0.5], height_ratios=[1, 10])
    gs.update(wspace=0.05, hspace=0.08)
    ax = pp.subplot(gs[1, 0], frameon=False)
    tmp_max_val = np.max(np.abs(intensity_plot))
    cax = ax.matshow(intensity_plot, cmap=diverge_map(), extent=[0, 4, 0, 4],
                     vmin=-tmp_max_val, vmax=tmp_max_val)
    ax.grid(color='k', linestyle='-')
    cbar_ax = pp.subplot(gs[1, 2])
    cbar_ax.yaxis.set_ticks_position('none')
    axr = pp.subplot(gs[1, 1], sharey=ax, yticks=[],
                     xticks=[],
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
                     yticks=[],
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
    plt.close()
    if not binary:
        fig.savefig(wd + '/figures/type_wiring_cum%s_conf'
                    'lvl%d.png' % (add_fname, int(confidence_lvl*10)), dpi=600)
    else:
        fig.savefig(wd + '/figures/type_wiring_cum%s_conf'
                    'lvl%d_binary.png' % (add_fname, int(confidence_lvl*10)),
                    dpi=600)


def type_sorted_wiring_cs(wd, confidence_lvl=0.3, binary=False,
                          max_syn_size=0.4):
    """Same as type_sorted_wiring but for all contact sites
    (synapse classification 0 and 1)
    """
    skeleton_ids, cell_type_probas = load_celltype_probas(wd)
    cell_type_pred_dict = load_pkl2obj(wd + '/neurons/celltype_pred_dict.pkl')
    bool_arr = np.zeros(len(skeleton_ids))
    # remove all skeletons under confidence level
    for k, probas in enumerate(cell_type_probas):
        if np.max(probas) > confidence_lvl:
            bool_arr[k] = 1
    bool_arr = bool_arr.astype(np.bool)
    skeleton_ids = skeleton_ids[bool_arr]

    # create matrix
    syn_props = load_pkl2obj(wd + '/contactsites/connectivity_dict_all.pkl')
    total_area_key = 'total_cs_area'
    dendrite_ids = set()
    axon_ids = set()
    for pair_name, pair in syn_props.iteritems():
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
    # print "%d/%d cells have no connection between each other." %\
    #       (len(skeleton_ids) - len(all_used_ids), len(skeleton_ids))
    # print "Using %d unique cells in wiring." % len(all_used_ids)
    axon_ids = np.array(list(axon_ids))
    dendrite_ids = np.array(list(dendrite_ids))

    # sort dendrites, axons using its type prediction. order is determined by
    # dictionaries get_id_dict_from_skel_ids
    dendrite_pred = np.array([cell_type_pred_dict[den_id] for den_id in
                              dendrite_ids])
    type_sorted_ixs = np.argsort(dendrite_pred, kind='mergesort')
    dendrite_pred = dendrite_pred[type_sorted_ixs]
    dendrite_ids = dendrite_ids[type_sorted_ixs]

    axon_pred = np.array([cell_type_pred_dict[den_id] for den_id in axon_ids])
    type_sorted_ixs = np.argsort(axon_pred, kind='mergesort')
    axon_pred = axon_pred[type_sorted_ixs]
    axon_ids = axon_ids[type_sorted_ixs]

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
                binary, wd, add_fname='_CS'):
    """Same as plot_wiring, but using all contact sites
    """
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
    ax.matshow(np.max(wiring.transpose(1, 0, 2), axis=2), interpolation="none",
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
    plt.imshow(a, cmap='gray')
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
        fig.savefig(wd + '/figures/type_wiring%s_conf'
                    'lvl%d.png' % (add_fname, int(confidence_lvl*10)), dpi=600)
    else:
        fig.savefig(wd + '/figures/type_wiring%s_conf'
            'lvl%d_binary.png' % (add_fname, int(confidence_lvl*10)), dpi=600)


def plot_wiring_cum_cs(wiring, den_borders, ax_borders, confidence_lvl,
                       binary, wd, add_fname=''):
    """Same as plot wiring, but using all contact sites"""
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
        fig.savefig(wd + '/figures/type_wiring_cum%s_conf'
                    'lvl%d.png' % (add_fname, int(confidence_lvl*10)), dpi=600)
    else:
        fig.savefig(wd + '/figures/type_wiring_cum%s_conf'
            'lvl%d_binary.png' % (add_fname, int(confidence_lvl*10)), dpi=600)


def make_colormap(seq):
    """Return a LinearSegmentedColormap
    seq: a sequence of floats and RGB-tuples. The floats should be increasing
    and in the interval (0,1).
    """
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])
    return mcolors.LinearSegmentedColormap('CustomMap', cdict)


def diverge_map(low=(239/255., 65/255., 50/255.),
                high=(39/255., 184/255., 148/255.)):
    """Low and high are colors that will be used for the two
    ends of the spectrum. they can be either color strings
    or rgb color tuples
    """
    c = mcolors.ColorConverter().to_rgb
    if isinstance(low, basestring): low = c(low)
    if isinstance(high, basestring): high = c(high)
    return make_colormap([low, c('white'), 0.5, c('white'), high])


def class_ranges(pred_arr):
    """Helper function to get extent of cell types in sorted prediction

    Parameters
    ----------
    pred_arr : np.array
        sorted array of cell type predictions

    Returns
    -------
    np.array
        indices of changing cell type labels in pred_arr
    """
    if len(pred_arr) == 0:
        return np.array([0, 0, 0, 0, 0])
    class1 = np.argmax(pred_arr == 1)
    class2 = np.max((class1, np.argmax(pred_arr == 2)))
    class3 = np.max((class2, np.argmax(pred_arr == 3)))
    return np.array([0, class1, class2, class3, len(pred_arr)])
