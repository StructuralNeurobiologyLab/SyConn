# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max-Planck-Institute of Neurobiology, Munich, Germany
# Authors: Philipp Schubert, Joergen Kornfeld
import time
from logging import Logger
from typing import Optional, Union

import matplotlib
import networkx as nx
import numpy as np

from . import log_reps
from .. import global_params
from ..handler.prediction import int2str_converter
from ..reps import segmentation

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib import gridspec
from collections import defaultdict
from scipy import ndimage


def cs_id_to_partner_ids_vec(cs_ids):
    sv_ids = np.right_shift(cs_ids, 32)
    sv_ids = np.concatenate((sv_ids[:, None], (cs_ids - np.left_shift(sv_ids, 32))[:, None]),
                            axis=1)
    return sv_ids


def cs_id_to_partner_inverse(partner_ids: Union[np.ndarray, list]) -> int:
    """
    Input permutation invariant transformation to bit-shift-based ID, which is used for `syn` and `cs`
    :class:`~syconn.reps.segmentation.SegmentationObject`.

    Args:
        partner_ids: :class:`~syconn.reps.super_segmentation_object.SuperSegmentationObject` IDs.

    Returns:
        Contact site or synapse fragment ID.
    """
    partner_ids = np.sort(partner_ids).astype(np.uint32)
    return (partner_ids[0] << 32) + partner_ids[1]


def connectivity_to_nx_graph(cd_dict):
    """
    Creates a directed networkx graph with attributes from the
    stored raw connectivity data.

        Parameters
    ----------
    cd_dict : dict

    Returns
    -------

    """
    nxg = nx.DiGraph()
    start = time.time()
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
    log_reps.debug('Done with graph ({1} nodes) construction, took {0}'.format(
        time.time() - start, nxg.number_of_nodes()))

    return nxg


def load_cached_data_dict(thresh_syn_prob=None, axodend_only=True, wd=None,
                          syn_version=None):
    """
    Loads all cached data from a contact site segmentation dataset into a
    dictionary for further processing.

    Parameters
    ----------
    wd : str
    syn_version : str
    thresh_syn_prob : float
        All synapses below `thresh_syn_prob` will be filtered.
    axodend_only: If True, returns only axo-dendritic synapse, all
        synapses otherwise.

    Returns
    -------

    """
    if wd is None:
        wd = global_params.config.working_dir
    if thresh_syn_prob is None:
        thresh_syn_prob = global_params.config['cell_objects']['thresh_synssv_proba']
    start = time.time()
    csd = segmentation.SegmentationDataset(obj_type='syn_ssv', working_dir=wd,
                                           version=syn_version)
    cd_dict = dict()
    cd_dict['ids'] = csd.load_numpy_data('id')
    # in um2, overlap of cs and sj
    cd_dict['syn_size'] = \
        csd.load_numpy_data('mesh_area') / 2  # as used in export_matrix
    cd_dict['synaptivity_proba'] = \
        csd.load_numpy_data('syn_prob')
    # -1 for inhibitory, +1 for excitatory
    cd_dict['syn_sign'] = \
        csd.load_numpy_data('syn_sign').astype(np.int)
    cd_dict['coord_x'] = \
        csd.load_numpy_data('rep_coord')[:, 0].astype(np.int)
    cd_dict['coord_y'] = \
        csd.load_numpy_data('rep_coord')[:, 1].astype(np.int)
    cd_dict['coord_z'] = \
        csd.load_numpy_data('rep_coord')[:, 2].astype(np.int)
    cd_dict['ssv_partner_0'] = \
        csd.load_numpy_data('neuron_partners')[:, 0].astype(np.int)
    cd_dict['ssv_partner_1'] = \
        csd.load_numpy_data('neuron_partners')[:, 1].astype(np.int)
    cd_dict['neuron_partner_ax_0'] = \
        csd.load_numpy_data('partner_axoness')[:, 0].astype(np.int)
    cd_dict['neuron_partner_ax_1'] = \
        csd.load_numpy_data('partner_axoness')[:, 1].astype(np.int)
    cd_dict['neuron_partner_ct_0'] = \
        csd.load_numpy_data('partner_celltypes')[:, 0].astype(np.int)
    cd_dict['neuron_partner_ct_1'] = \
        csd.load_numpy_data('partner_celltypes')[:, 1].astype(np.int)
    cd_dict['neuron_partner_sp_0'] = \
        csd.load_numpy_data('partner_spiness')[:, 0].astype(np.int)
    cd_dict['neuron_partner_sp_1'] = \
        csd.load_numpy_data('partner_spiness')[:, 1].astype(np.int)

    log_reps.debug('Getting {1} objects took: {0}'.format(time.time() - start,
                                                          len(csd.ids)))

    idx_filter = cd_dict['synaptivity_proba'] >= thresh_syn_prob
    cd_dict['neuron_partner_ax_0'][cd_dict['neuron_partner_ax_0'] == 3] = 1
    cd_dict['neuron_partner_ax_0'][cd_dict['neuron_partner_ax_0'] == 4] = 1
    cd_dict['neuron_partner_ax_1'][cd_dict['neuron_partner_ax_1'] == 3] = 1
    cd_dict['neuron_partner_ax_1'][cd_dict['neuron_partner_ax_1'] == 4] = 1
    n_syns = np.sum(idx_filter)
    if axodend_only:
        idx_filter = idx_filter & ((cd_dict['neuron_partner_ax_0'] +
                                    cd_dict['neuron_partner_ax_1']) == 1)
    n_syns_after_axdend = np.sum(idx_filter)
    for k, v in cd_dict.items():
        cd_dict[k] = v[idx_filter]

    log_reps.debug('Finished conn. dictionary with {} synaptic '
                   'objects. {} above prob. threshold {}. '
                   '{} syns after axo-dend. filter '
                   '(changes only if "axodend_only=True").'
                   ''.format(len(idx_filter), n_syns, thresh_syn_prob,
                             n_syns, n_syns_after_axdend))
    return cd_dict


def generate_wiring_array(log: Optional[Logger] = None, **load_cached_data_dict_kwargs):
    """
    Creates a 2D wiring array with quadratic shape (#cells x #cells) sorted by
    cell type. X-axis: post-synaptic partners, y: pre-synaptic partners.
    Assumes label 1 in 'partner_axoness' represents axon compartments. Does not
    support ``axodend_only=False``!
    Required for :func:`~plot_wiring` and :func:`plot_cumul_wiring`.

    Notes:
        * Work-in-progress.

    Args:
        log: Logger.
        **load_cached_data_dict_kwargs: See :func:`~load_cached_data_dict`

    Returns:
        The wiring diagram as a 2D array.

    """
    if 'axodend_only=True' in load_cached_data_dict_kwargs:
        raise ValueError("'axodend_only=False' is not supported!")
    cd_dict = load_cached_data_dict(**load_cached_data_dict_kwargs)
    if log is None:
        log = log_reps
    # analyze scope of underlying data
    all_ssv_ids = set(cd_dict['ssv_partner_0'].tolist()).union(set(cd_dict['ssv_partner_1']))
    n_cells = len(all_ssv_ids)
    wiring = np.zeros((n_cells, n_cells))
    celltypes = np.unique([cd_dict['neuron_partner_ct_0'], cd_dict['neuron_partner_ct_1']])
    ssvs_flattened = []
    borders = []
    np.random.seed(0)
    # create list of cells used for celltype-sorted x and y axis
    for ct in celltypes:
        l0 = cd_dict['ssv_partner_0'][cd_dict['neuron_partner_ct_0'] == ct]
        l1 = cd_dict['ssv_partner_1'][cd_dict['neuron_partner_ct_1'] == ct]
        curr_ct_ssvs = np.unique(np.concatenate([l0, l1]))
        # destroy sorting of `np.unique`
        np.random.shuffle(curr_ct_ssvs)
        curr_ct_ssvs = curr_ct_ssvs.tolist()
        ssvs_flattened += curr_ct_ssvs
        borders.append(len(curr_ct_ssvs))
    borders = np.cumsum(borders)
    assert borders[-1] == len(wiring)
    # sum per cell-pair synaptic connections multiplied by synaptic sign (-1 or 1)
    cumul_syn_dc = defaultdict(list)
    # synapse size: in um2, mesh area of the overlap between cs and sj divided by 2
    for ii, syn_size in enumerate(cd_dict['syn_size']):
        cell_pair = (cd_dict['ssv_partner_0'][ii], cd_dict['ssv_partner_1'][ii])
        if cd_dict['neuron_partner_ax_1'][ii] == 1:
            cell_pair = cell_pair[::-1]
        elif cd_dict['neuron_partner_ax_0'][ii] == 1:
            pass
        else:
            raise ValueError('No axon prediction found within synapse.')
        cumul_syn_dc[cell_pair].append(syn_size * cd_dict['syn_sign'][ii])
    cumul_syn_dc = dict(cumul_syn_dc)
    rev_map = {ssvs_flattened[ii]: ii for ii in range(n_cells)}
    for pre_id, post_id in cumul_syn_dc:
        pre_ix = rev_map[pre_id]
        post_ix = rev_map[post_id]
        syns = cumul_syn_dc[(pre_id, post_id)]
        syns_pos = np.sum([syn for syn in syns if syn > 0])
        syns_neg = np.abs(np.sum([syn for syn in syns if syn < 0]))
        sign = -1 if syns_neg > syns_pos else 1
        wiring[post_ix, pre_ix] = sign * (syns_pos + syns_neg)
    ct_borders = [(int2str_converter(celltypes[ii], gt_type='ctgt_v2'), borders[ii]) for ii in range(len(celltypes))]
    log.info(f'Found the following cell types (label, starting index in wiring diagram: {ct_borders}')
    return wiring, borders[:-1]


def plot_wiring(path, wiring, den_borders, ax_borders, cumul=False, log: Optional[Logger] = None):
    """Plot type sorted connectivity matrix. Saved in folder given by `path`.

    Notes:
        * Work-in-progress.
        * `wiring` is generated by :func:`~generate_wiring_array`.

    Parameters
    ----------
    path: Path to directory.
    wiring : Quadratic 2D array of size #cells x #cells, x-axis: dendrite partners, y: axon partners.
    den_borders: Cell type borders on post synaptic site. Used to split the connectivity matrix into quadrants.
    ax_borders: Cell type borders on pre synaptic site. Used to split the connectivity matrix into quadrants.
    cumul: Accumulate quadrant values.
    log: Logger.
    """
    if log is None:
        log = log_reps
    if cumul:
        entry_width = 1
    else:
        entry_width = int(np.max([20 / 29297 * wiring.shape[0], 1]))
    intensity_plot = np.array(wiring)
    intensity_plot_neg = intensity_plot < 0
    intensity_plot_pos = intensity_plot > 0
    borders = [0] + list(ax_borders) + [intensity_plot.shape[1]]
    for i_border in range(1, len(borders)):
        start = borders[i_border - 1]
        end = borders[i_border]
        sign = np.sum(intensity_plot_pos[:, start: end]) - \
               np.sum(intensity_plot_neg[:, start: end]) > 0
        # adapt either negative or positive elements according to the majority vote
        if sign:
            intensity_plot[:, start: end][intensity_plot[:, start: end] < 0] *= -1
        else:
            intensity_plot[:, start: end][intensity_plot[:, start: end] > 0] *= -1

    intensity_plot_neg = intensity_plot[intensity_plot < 0]
    intensity_plot_pos = intensity_plot[intensity_plot > 0]

    int_cut_pos = np.mean(intensity_plot_pos) + np.std(intensity_plot_pos)
    if np.isnan(int_cut_pos):
        int_cut_pos = 1
    int_cut_neg = np.abs(np.mean(intensity_plot_neg)) + np.std(intensity_plot_neg)
    if np.isnan(int_cut_neg):
        int_cut_neg = 1

    # balance max values
    intensity_plot[intensity_plot > 0] = intensity_plot[intensity_plot > 0] / int_cut_pos
    intensity_plot[intensity_plot < 0] = intensity_plot[intensity_plot < 0] / int_cut_neg
    log.info(f'1-sigma cut-off for excitatory cells: {int_cut_pos}')
    log.info(f'1-sigma cut-off for inhibitory cells: {int_cut_neg}')
    log.debug(f'Initial wiring diagram shape: {intensity_plot.shape}')

    if not cumul:
        # TODO: refactor, this becomes slow for shapes > (10k, 10k)
        log_reps.debug(f'Increasing the matrix entries from pixels of edge length 1 to {entry_width} .')
        for k, b in enumerate(den_borders):
            b += k * entry_width
            intensity_plot = np.concatenate(
                (intensity_plot[:b, :], np.zeros((entry_width, intensity_plot.shape[1])),
                 intensity_plot[b:, :]), axis=0)

        for k, b in enumerate(ax_borders):
            b += k * entry_width
            intensity_plot = np.concatenate(
                (intensity_plot[:, :b], np.zeros((intensity_plot.shape[0], entry_width)),
                 intensity_plot[:, b:]), axis=1)

        log.debug(f'Wiring diagram shape after adding hline columns and rows: {intensity_plot.shape}')
    else:
        log.debug(f'Wiring diagram shape after adding hline columns and rows: {intensity_plot.shape}')

    # TODO: becomes slow for large entry_width
    bin_intensity_plot = intensity_plot != 0
    bin_intensity_plot = bin_intensity_plot.astype(np.float)
    intensity_plot = ndimage.convolve(intensity_plot, np.ones((entry_width, entry_width)))
    bin_intensity_plot = ndimage.convolve(bin_intensity_plot, np.ones((entry_width, entry_width)))
    intensity_plot /= bin_intensity_plot

    matplotlib.rcParams.update({'font.size': 14})
    fig = plt.figure()
    # Create scatter plot, why?
    gs = gridspec.GridSpec(1, 2, width_ratios=[20, 1])
    gs.update(wspace=0.05, hspace=0.08)
    ax = plt.subplot(gs[0, 0], frameon=False)

    cax = ax.matshow(intensity_plot.transpose((1, 0)),
                     cmap=diverge_map(),
                     extent=[0, intensity_plot.shape[0], intensity_plot.shape[1], 0],
                     interpolation="none", vmin=-1,
                     vmax=1)
    ax.set_xlabel('Post', fontsize=18)
    ax.set_ylabel('Pre', fontsize=18)
    ax.set_xlim(0, intensity_plot.shape[0])
    ax.set_ylim(0, intensity_plot.shape[1])
    plt.grid(False)
    plt.axis('off')

    if cumul:
        for k, b in enumerate(den_borders):
            plt.axvline(b, color='k', lw=0.5, snap=False,
                        antialiased=True)
        for k, b in enumerate(ax_borders):
            plt.axhline(b, color='k', lw=0.5, snap=False,
                        antialiased=True)
    else:
        for k, b in enumerate(den_borders):
            b += k * entry_width
            plt.axvline(b + 0.5, color='k', lw=0.5, snap=False,
                        antialiased=True)
        for k, b in enumerate(ax_borders):
            b += k * entry_width
            plt.axhline(b + 0.5, color='k', lw=0.5, snap=False,
                        antialiased=True)

    cbar_ax = plt.subplot(gs[0, 1])
    cbar_ax.yaxis.set_ticks_position('none')
    cb = fig.colorbar(cax, cax=cbar_ax, ticks=[])
    plt.close()

    if cumul:
        mat_name = "/matrix_cum_%d_%d" % (int(int_cut_neg * 100000), int(int_cut_pos * 100000))
        fig.savefig(path + mat_name + '.png', dpi=600)
    else:
        mat_name = "/matrix_%d_%d_%d" % (
            intensity_plot.shape[0], int(int_cut_neg * 100000), int(int_cut_pos * 100000))
        fig.savefig(path + mat_name + '.png', dpi=600)
    # TODO: refine summary log
    sum_str = f"Cut value negative: {int_cut_neg}\n"
    sum_str += f"Cut value positive: {int_cut_pos}\n"
    sum_str += f"Post-synaptic borders: {den_borders}\n"
    sum_str += f"Pre-synaptic borders: {ax_borders}\n"
    with open(path + mat_name + '.txt', 'w') as f:
        f.write(sum_str)


def plot_cumul_wiring(path, wiring, borders, min_cumul_synarea=0, log: Optional[Logger] = None):
    """
    Synaptic area between cell type pairs. Synaptic areas are summed and then
    divided by the number of cell pairs to compute the average cumulated synaptic area
    between each

    Notes:
        * Work-in-progress.
        * `wiring` is generated by :func:`~generate_wiring_array`.

    Args:
        path:
        wiring:
        borders:
        min_cumul_synarea:
        log: Logger.
    """
    cumul_matrix = np.zeros([len(borders) + 1, len(borders) + 1])
    borders = [0] + list(borders) + [wiring.shape[1]]
    for i_ax_border in range(1, len(borders)):
        for i_de_border in range(1, len(borders)):
            ax_start = borders[i_ax_border - 1]
            ax_end = borders[i_ax_border]
            de_start = borders[i_de_border - 1]
            de_end = borders[i_de_border]
            cumul = wiring[de_start: de_end, ax_start: ax_end].flatten()
            pos = np.sum(cumul[cumul > 0])
            neg = np.abs(np.sum(cumul[cumul < 0]))
            sign = -1 if neg > pos else 1
            cumul = sign * (pos + neg)
            if np.abs(cumul) < min_cumul_synarea:
                cumul = 0
            else:
                # convert to density (average cumul. synaptic area between cell pairs)
                cumul /= (ax_end - ax_start) * (de_end - de_start)
            cumul_matrix[i_de_border - 1, i_ax_border - 1] = cumul
    plot_wiring(path, cumul_matrix, list(range(1, len(borders) + 1)), list(range(1, len(borders) + 1)), cumul=True,
                log=log)


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


def diverge_map(high=(239 / 255., 65 / 255., 50 / 255.),
                low=(39 / 255., 184 / 255., 148 / 255.)):
    """Low and high are colors that will be used for the two
    ends of the spectrum. they can be either color strings
    or rgb color tuples
    """
    c = mcolors.ColorConverter().to_rgb
    if isinstance(low, str): low = c(low)
    if isinstance(high, str): high = c(high)
    return make_colormap([low, c('white'), 0.5, c('white'), high])


def connectivity_hists_j0251(proba_thresh_syn: float = 0.8, proba_thresh_celltype: float = None,
                             r=(0.05, 2)):
    """
    Args:
        proba_thresh_syn: Synapse probability. Filters synapses below threshold.
        proba_thresh_celltype: Cell type probability. Filters cells below threshold.
        r: Range of synapse mesh area (um^2).

    Returns:

    """
    from syconn.handler.prediction import int2str_converter, certainty_estimate
    from syconn.reps.segmentation import SegmentationDataset
    from syconn.reps.super_segmentation import SuperSegmentationDataset
    from scipy.special import softmax
    import pandas as pd
    import tqdm
    import os
    import seaborn as sns
    def ctclass_converter(x): return int2str_converter(x, gt_type='ctgt_j0251_v2')
    target_dir = f'/wholebrain/scratch/pschuber/tmp/thresh{int(proba_thresh_syn * 100)}/'
    os.makedirs(target_dir, exist_ok=True)
    nclass = 11
    log_scale = True
    plot_n_celltypes = 5
    palette = sns.color_palette('dark', n_colors=nclass)
    palette = {ctclass_converter(kk): palette[kk] for kk in range(nclass)}
    sd_syn_ssv = SegmentationDataset('syn_ssv')
    if proba_thresh_celltype is not None:
        ssd = SuperSegmentationDataset()
        ct_probas = [certainty_estimate(proba) for proba in tqdm.tqdm(ssd.load_cached_data('celltype_cnn_e3_probas'),
                                                                      desc='Cells')]
        ct_proba_lookup = {cellid: ct_probas[k] for k, cellid in enumerate(ssd.ssv_ids)}
        del ct_probas
    ax = sd_syn_ssv.load_cached_data('partner_axoness')
    ct = sd_syn_ssv.load_cached_data('partner_celltypes')
    area = sd_syn_ssv.load_cached_data('mesh_area')
    # size = sd_syn_ssv.load_cached_data('size')
    # syn_sign = sd_syn_ssv.load_cached_data('syn_sign')
    # area *= syn_sign
    partners = sd_syn_ssv.load_cached_data('neuron_partners')

    proba = sd_syn_ssv.load_cached_data('syn_prob')
    m = (proba >= proba_thresh_syn) & (area >= r[0]) & (area <= r[1])
    print(f'Found {np.sum(m)} synapses after filtering with probaility threshold {proba_thresh_syn} and '
          f'size filter (min/max [um^2]: {r}).')
    ax[(ax == 3) | (ax == 4)] = 1  # set boutons to axon class
    ax[(ax == 5) | (ax == 6)] = 0  # set spine head and neck to dendrite class
    m = m & (np.sum(ax, axis=1) == 1)  # get all axo-dendritic synapses
    print(f'Found {np.sum(m)} synapses after filtering non axo-dendritic ones.')
    ct = ct[m]
    ax = ax[m]
    area = area[m]
    # size = size[m]
    partners = partners[m]
    if log_scale:
        area = np.log10(area)
        r = np.log(r)
    ct_receiving = {ctclass_converter(k): {ctclass_converter(kk): [] for kk in range(nclass)} for k in range(nclass)}
    ct_targets = {ctclass_converter(k): {ctclass_converter(kk): [] for kk in range(nclass)} for k in range(nclass)}
    for ix in tqdm.tqdm(range(area.shape[0]), total=area.shape[0], desc='Synapses'):
        post_ix, pre_ix = np.argsort(ax[ix])
        if proba_thresh_celltype is not None:
            post_cell_id, pre_cell_id = partners[ix][post_ix], partners[ix][pre_ix]
            celltype_probas = np.array([ct_proba_lookup[post_cell_id], ct_proba_lookup[pre_cell_id]])
            if np.any(celltype_probas < proba_thresh_celltype):
                continue
        syn_ct = ct[ix]
        pre_ct = ctclass_converter(syn_ct[pre_ix])
        post_ct = ctclass_converter(syn_ct[post_ix])
        ct_receiving[post_ct][pre_ct].append(area[ix])
        ct_targets[pre_ct][post_ct].append(area[ix])
    for ct_label in tqdm.tqdm(map(ctclass_converter, range(nclass)), total=nclass):
        data_rec = ct_receiving[ct_label]
        sizes = np.argsort([len(v) for v in data_rec.values()])[::-1]
        highest_cts = np.array(list(data_rec.keys()))[sizes][:plot_n_celltypes]
        df = pd.DataFrame(data={'mesh_area': np.concatenate([data_rec[k] for k in highest_cts]),
                                'cell_type': np.concatenate([[k]*len(data_rec[k]) for k in highest_cts])})
        create_kde(f'{target_dir}/incoming{ct_label}.png', df, palette=palette, r=r)
        df = pd.DataFrame(data={'mesh_area[um^2]': [np.sum(10**np.array(data_rec[k])) for k in data_rec],
                                'n_synapses': [len(data_rec[k]) for k in data_rec],
                                'cell_type': [k for k in data_rec]})
        df.to_csv(f'{target_dir}/incoming{ct_label}_sum.csv')
        data_out = ct_targets[ct_label]
        sizes = np.argsort([len(v) for v in data_out.values()])[::-1]
        highest_cts = np.array(list(data_out.keys()))[sizes][:plot_n_celltypes]
        df = pd.DataFrame(data={'mesh_area': np.concatenate([data_out[k] for k in highest_cts]),
                                'cell_type': np.concatenate([[k]*len(data_out[k]) for k in highest_cts])})
        create_kde(f'{target_dir}/outgoing{ct_label}.png', df, palette=palette, r=r)
        df = pd.DataFrame(data={'mesh_area[um^2]': [np.sum(10**np.array(data_out[k])) for k in data_out],
                                'n_synapses': [len(data_out[k]) for k in data_out],
                                'cell_type': [k for k in data_out]})
        df.to_csv(f'{target_dir}/outgoing{ct_label}_sum.csv')


def create_kde(dest_p, qs, ls=20, legend=False, r=None, **kwargs):
    """


    Parameters
    ----------
    dest_p :
    qs :
    r :
    legend :
    ls :

    Returns
    -------

    """
    r = np.array(r)
    import seaborn as sns
    # fig, ax = plt.subplots()
    plt.figure()
    # ax = sns.swarmplot(data=qs, clip_on=False, **kwargs)
    # fill=True, kde=True, kde_kws=dict(bw_adjust=0.5), common_bins=False,
    sns.displot(data=qs, x="mesh_area", hue="cell_type",
                # kind="kde", bw_adjust=0.5,
                fill=True, kde=True, kde_kws=dict(bw_adjust=0.5), common_bins=True,
                edgecolor='none', multiple="layer", common_norm=False, stat='density',
                **kwargs)  # , common_norm=False
    if r is not None:
        xmin, xmax = plt.xlim()
        if xmin > r[0]:
            r[0] = xmin
        if xmax < r[1]:
            r[1] = xmax
        plt.xlim(r)
    # if not legend:
    #     plt.gca().legend().set_visible(False)
    # # sns.set_style("ticks", {"xtick.major.size": 20, "ytick.major.size": 20})
    # ax.tick_params(axis='x', which='major', labelsize=ls, direction='out',
    #                length=4, width=3, right="off", top="off", pad=10)
    # ax.tick_params(axis='y', which='major', labelsize=ls, direction='out',
    #                length=4, width=3, right="off", top="off", pad=10)
    # ax.tick_params(axis='x', which='minor', labelsize=ls, direction='out',
    #                length=4, width=3, right="off", top="off", pad=10)
    # ax.tick_params(axis='y', which='minor', labelsize=ls, direction='out',
    #                length=4, width=3, right="off", top="off", pad=10)
    # ax.spines['left'].set_linewidth(3)
    # ax.spines['bottom'].set_linewidth(3)
    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)
    plt.savefig(dest_p, dpi=300)
    qs.to_csv(dest_p[:-4] + ".csv")
    plt.close('all')
