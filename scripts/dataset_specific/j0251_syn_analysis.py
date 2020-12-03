# SyConn
# Copyright (c) 2016 Philipp J. Schubert
# All rights reserved
import time
from logging import Logger
from typing import Optional

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


def connectivity_hists_j0251(proba_thresh_syn: float = 0.8, proba_thresh_celltype: float = None,
                             r: Optional[tuple] = None, use_spinehead_vol: bool = False,
                             size_quantity: str = 'vx'):
    """
    Experimental.

    Args:
        proba_thresh_syn: Synapse probability. Filters synapses below threshold.
        proba_thresh_celltype: Cell type probability. Filters cells below threshold.
        r: Range of synapse mesh area (um^2).
        use_spinehead_vol: Use spinehead volume instead of ``mesh_area / 2``.
        size_quantity: One of 'vx', 'spinehead_vol', 'mesh_area'

    """
    from syconn.handler.prediction import int2str_converter, certainty_estimate
    from syconn.reps.segmentation import SegmentationDataset
    from syconn.reps.super_segmentation import SuperSegmentationDataset
    from scipy.special import softmax
    import pandas as pd
    import tqdm
    import os
    import seaborn as sns
    if size_quantity not in ['vx', 'spinehead_vol', 'mesh_area']:
        raise ValueError(f'Unknown size quantitiy.')
    if r is None:
        r = dict(vx=(1e3, 20e3), mesh_area=(0.5, 4), spinehead_vol=(0.01, 3))[size_quantity]
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
        ct_probas = [certainty_estimate(proba) for proba in tqdm.tqdm(ssd.load_numpy_data('celltype_cnn_e3_probas'),
                                                                      desc='Cells')]
        ct_proba_lookup = {cellid: ct_probas[k] for k, cellid in enumerate(ssd.ssv_ids)}
        del ct_probas
    ax = sd_syn_ssv.load_numpy_data('partner_axoness')
    ct = sd_syn_ssv.load_numpy_data('partner_celltypes')
    area = sd_syn_ssv.load_numpy_data('mesh_area')
    sh_vol = sd_syn_ssv.load_numpy_data('partner_spineheadvol')

    # size = sd_syn_ssv.load_numpy_data('size')
    # syn_sign = sd_syn_ssv.load_numpy_data('syn_sign')
    # area *= syn_sign
    partners = sd_syn_ssv.load_numpy_data('neuron_partners')

    proba = sd_syn_ssv.load_numpy_data('syn_prob')
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
    size = size[m]
    sh_vol = sh_vol[m]
    partners = partners[m]
    if log_scale:
        area = np.log10(area)
        sh_vol = np.log10(sh_vol)
        size = np.log10(size)
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
        if size_quantity == 'vx':
            size_quantity = size[ix]
        elif size_quantity == 'spinehead_vol':
            size_quantity = sh_vol[ix][post_ix]
        elif size_quantity == 'mesh_area':
            size_quantity = area[ix]
        ct_receiving[post_ct][pre_ct].append(size_quantity)
        ct_targets[pre_ct][post_ct].append(size_quantity)
    size_quantity_label = 'spinehead_vol' if use_spinehead_vol else 'mesh_area'
    print('Area/volume is in µm^2 or µm^3 respectively.')
    for ct_label in tqdm.tqdm(map(ctclass_converter, range(nclass)), total=nclass, desc='Cell types'):
        data_rec = ct_receiving[ct_label]
        sizes = np.argsort([len(v) for v in data_rec.values()])[::-1]
        highest_cts = np.array(list(data_rec.keys()))[sizes][:plot_n_celltypes]
        df = pd.DataFrame(data={size_quantity_label: np.concatenate([data_rec[k] for k in highest_cts]),
                                'cell_type': np.concatenate([[k]*len(data_rec[k]) for k in highest_cts])})
        create_kde(f'{target_dir}/incoming{ct_label}_{size_quantity_label}.png', df, palette=palette, r=r)
        df = pd.DataFrame(data={size_quantity_label: [np.sum(10**np.array(data_rec[k])) for k in data_rec],
                                'n_synapses': [len(data_rec[k]) for k in data_rec],
                                'cell_type': [k for k in data_rec]})
        df.to_csv(f'{target_dir}/incoming{ct_label}_{size_quantity_label}_sum.csv')
        data_out = ct_targets[ct_label]
        sizes = np.argsort([len(v) for v in data_out.values()])[::-1]
        highest_cts = np.array(list(data_out.keys()))[sizes][:plot_n_celltypes]
        df = pd.DataFrame(data={size_quantity_label: np.concatenate([data_out[k] for k in highest_cts]),
                                'cell_type': np.concatenate([[k]*len(data_out[k]) for k in highest_cts])})
        create_kde(f'{target_dir}/outgoing{ct_label}_{size_quantity_label}.png', df, palette=palette, r=r)
        df = pd.DataFrame(data={size_quantity_label: [np.sum(10**np.array(data_out[k])) for k in data_out],
                                'n_synapses': [len(data_out[k]) for k in data_out],
                                'cell_type': [k for k in data_out]})
        df.to_csv(f'{target_dir}/outgoing{ct_label}_{size_quantity_label}_sum.csv')


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
