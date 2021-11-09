# SyConn
# Copyright (c) 2016 Philipp J. Schubert
# All rights reserved
from typing import Optional
import pandas as pd
import tqdm
import os
from collections import defaultdict

import seaborn as sns
from scipy.special import softmax
import numpy as np
import matplotlib.pyplot as plt

from syconn.handler.prediction import int2str_converter, certainty_estimate, str2int_converter
from syconn.reps.segmentation import SegmentationDataset
from syconn.handler import basics
from syconn.handler.config import initialize_logging
from syconn.reps.super_segmentation import SuperSegmentationDataset


def connectivity_hists_j0251(proba_thresh_syn: float = 0.8, proba_thresh_celltype: float = None,
                             r: Optional[tuple] = None, r_filter: Optional[tuple] = None,
                             size_quantity: str = 'vx', filter_quant: Optional[str] = None):
    """
    Experimental.
    Mesh are in µm^2, spine head vol in µm^3 and vx in [1].
    Args:
        proba_thresh_syn: Synapse probability. Filters synapses below threshold.
        proba_thresh_celltype: Cell type probability. Filters cells below threshold.
        r: Range of synapse size quantitiy for plotting.
            ```range_dict = dict(vx=(500, 200e3), mesh_area=(0.5, 4), spinehead_vol=(0.005, 3))```.
        r_filter: Range of synapse size quantity for filtering.
            ```range_dict_filter = dict(vx=(500, np.inf), mesh_area=(0.01, np.inf), spinehead_vol=(0.005, np.inf))```.
        size_quantity: One of 'vx', 'spinehead_vol', 'mesh_area'
        filter_quant: Quantity to filter synapses, either 'vx', 'spinehead_vol' or 'mesh_area'.

    """
    range_dict = dict(vx=(500, 200e3), mesh_area=(0.5, 4), spinehead_vol=(0.005, 3))
    range_dict_filter = dict(vx=(500, np.inf), mesh_area=(0.01, np.inf), spinehead_vol=(0.005, np.inf))
    if size_quantity not in ['vx', 'spinehead_vol', 'mesh_area']:
        raise ValueError(f'Unknown size quantitiy.')
    if filter_quant is None:
        filter_quant = str(size_quantity)
    if filter_quant not in ['vx', 'mesh_area', 'spinehead_vol']:
        raise ValueError(f'Unknown filter quantitiy.')
    if r_filter is None:
        r_filter = range_dict_filter[filter_quant]
    if r is None:
        r = range_dict[size_quantity]
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
        cached_dict_path = f'/wholebrain/scratch/pschuber/tmp/._ssv_ct_probas_{hash(str(ssd))}.pkl'
        if not os.path.isfile(cached_dict_path):
            ct_probas = [certainty_estimate(proba, is_logit=True) for proba in
                         tqdm.tqdm(ssd.load_numpy_data('celltype_cnn_e3_probas'), desc='Cells')]
            ct_proba_lookup = {cellid: ct_probas[k] for k, cellid in enumerate(ssd.ssv_ids)}
            basics.write_obj2pkl(cached_dict_path, ct_proba_lookup)
            del ct_probas
        else:
            ct_proba_lookup = basics.load_pkl2obj(cached_dict_path)
    ax = sd_syn_ssv.load_numpy_data('partner_axoness')
    ct = sd_syn_ssv.load_numpy_data('partner_celltypes')
    area = sd_syn_ssv.load_numpy_data('mesh_area')
    sh_vol = sd_syn_ssv.load_numpy_data('partner_spineheadvol')

    size = sd_syn_ssv.load_numpy_data('size')
    # syn_sign = sd_syn_ssv.load_numpy_data('syn_sign')
    # area *= syn_sign
    partners = sd_syn_ssv.load_numpy_data('neuron_partners')

    proba = sd_syn_ssv.load_numpy_data('syn_prob')
    if filter_quant == 'vx':
        filter_quant_val = size
    elif filter_quant == 'mesh_area':
        filter_quant_val = area
    elif filter_quant == 'spinehead_vol':
        filter_quant_val = np.max(sh_vol, axis=1)
        assert len(filter_quant_val) == len(sh_vol)
    else:
        raise ValueError()
    m = (proba >= proba_thresh_syn) & (filter_quant_val >= r_filter[0]) & (filter_quant_val <= r_filter[1])
    if size_quantity == 'spinehead_vol' and filter_quant != 'spinehead_vol':
        m = m & (np.max(sh_vol, axis=1) > 0)
    del filter_quant_val
    print(f'Found {np.sum(m)} synapses after filtering with probability threshold {proba_thresh_syn} and '
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
        r = np.log10(r)
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
            val = size[ix]
        elif size_quantity == 'spinehead_vol':
            val = sh_vol[ix][post_ix]
        elif size_quantity == 'mesh_area':
            val = area[ix]
        else:
            raise ValueError
        ct_receiving[post_ct][pre_ct].append(val)
        ct_targets[pre_ct][post_ct].append(val)
    x_label = f'{size_quantity}'
    if log_scale:
        x_label += ' (log)'
    if size_quantity == 'spinehead_vol':
        x_label += '[µm^3]'
    elif size_quantity == 'mesh_area':
        x_label += '[µm^2]'
    for ct_label in tqdm.tqdm(map(ctclass_converter, range(nclass)), total=nclass, desc='Cell types'):
        if ct_label != 'MSN':
            continue
        data_rec = ct_receiving[ct_label]
        sizes = np.argsort([len(v) for v in data_rec.values()])[::-1]
        highest_cts = np.array(list(data_rec.keys()))[sizes][:plot_n_celltypes]
        df = pd.DataFrame(data={size_quantity: np.concatenate([data_rec[k] for k in highest_cts]),
                                'cell_type': np.concatenate([[k]*len(data_rec[k]) for k in highest_cts])})
        create_kde(f'{target_dir}/incoming{ct_label}_{size_quantity}.png', df, size_quantity, palette=palette,
                   r=r, x_label=x_label)
        df = pd.DataFrame(data={size_quantity: [np.sum(10**np.array(data_rec[k])) for k in data_rec],
                                'n_synapses': [len(data_rec[k]) for k in data_rec],
                                'cell_type': [k for k in data_rec]})
        df.to_csv(f'{target_dir}/incoming{ct_label}_{size_quantity}_sum.csv')
        data_out = ct_targets[ct_label]
        sizes = np.argsort([len(v) for v in data_out.values()])[::-1]
        highest_cts = np.array(list(data_out.keys()))[sizes][:plot_n_celltypes]
        df = pd.DataFrame(data={size_quantity: np.concatenate([data_out[k] for k in highest_cts]),
                                'cell_type': np.concatenate([[k]*len(data_out[k]) for k in highest_cts])})
        create_kde(f'{target_dir}/outgoing{ct_label}_{size_quantity}.png', df, size_quantity, palette=palette,
                   r=r, x_label=x_label)
        df = pd.DataFrame(data={size_quantity: [np.sum(10**np.array(data_out[k])) for k in data_out],
                                'n_synapses': [len(data_out[k]) for k in data_out],
                                'cell_type': [k for k in data_out]})
        df.to_csv(f'{target_dir}/outgoing{ct_label}_{size_quantity}_sum.csv')


def syn_fracs(proba_thresh_syn: float = 0.8, proba_thresh_celltype: float = None,
              filter_quant: Optional[str] = 'mesh_area', ct_label_pre: str = 'HVC',
              ct_label_post: str = 'MSN', nsyns_min: int = 10, max_rel_weight: float = 0.3):
    """
    Experimental.
    Mesh area in µm^2, spine head vol in µm^3 and vx in [1].
    Args:
        proba_thresh_syn: Synapse probability. Filters synapses below threshold.
        proba_thresh_celltype: Cell type probability. Filters cells below threshold.
        filter_quant: Quantity to filter synapses, either 'vx', 'spinehead_vol' or 'mesh_area'.
            Filter range is determined by: ``dict(vx=(500, np.inf), mesh_area=(0.01, np.inf), spinehead_vol=(0.005, np.inf))``.
        ct_label_pre: Cell type of pre-synaptic neuron.
        ct_label_post: Cell type of post-synaptic neuron.
        nsyns_min: Minimum number of synapses onto post-synaptic cell. If below it is excluded.
        max_rel_weight: Maximum

    """
    range_dict = dict(vx=(500, np.inf), mesh_area=(0.01, np.inf), spinehead_vol=(0.005, np.inf))
    target_dir = f'/wholebrain/scratch/pschuber/tmp/syn_fracs/thresh{int(max_rel_weight * 100)}/'
    os.makedirs(target_dir, exist_ok=True)
    log = initialize_logging(f'syn_fracs_{int(max_rel_weight * 100)}_{ct_label_pre}_{ct_label_post}', target_dir)
    sd_syn_ssv = SegmentationDataset('syn_ssv')
    ssd = SuperSegmentationDataset()
    log.info(f'{sd_syn_ssv}\nnsyns_min={nsyns_min}, max_rel_weight={max_rel_weight}')
    log.info(f'{ssd}\nct_label_pre={ct_label_pre}, ct_label_post={ct_label_post}')
    if proba_thresh_celltype is not None:
        cached_dict_path = f'/wholebrain/scratch/pschuber/tmp/._ssv_ct_probas_{hash(str(ssd))}.pkl'
        if not os.path.isfile(cached_dict_path):
            ct_proba_lookup = dict()
            probas = ssd.load_numpy_data('celltype_cnn_e3_probas')
            for ix in tqdm.tqdm(range(len(probas)), desc='Cells'):
                ct_proba_lookup[ssd.ssv_ids[ix]] = certainty_estimate(probas[ix], is_logit=True)
            basics.write_obj2pkl(cached_dict_path, ct_proba_lookup)
            del probas
        else:
            ct_proba_lookup = basics.load_pkl2obj(cached_dict_path)
            assert not np.isinf(ct_proba_lookup[ssd.ssv_ids[0]])
    ax = sd_syn_ssv.load_numpy_data('partner_axoness')
    ct = sd_syn_ssv.load_numpy_data('partner_celltypes')
    area = sd_syn_ssv.load_numpy_data('mesh_area')
    sh_vol = sd_syn_ssv.load_numpy_data('partner_spineheadvol')

    size = sd_syn_ssv.load_numpy_data('size')
    # syn_sign = sd_syn_ssv.load_numpy_data('syn_sign')
    # area *= syn_sign
    partners = sd_syn_ssv.load_numpy_data('neuron_partners')

    proba = sd_syn_ssv.load_numpy_data('syn_prob')
    if filter_quant == 'vx':
        filter_quant_val = size
    elif filter_quant == 'mesh_area':
        filter_quant_val = area
    elif filter_quant == 'spinehead_vol':
        filter_quant_val = np.max(sh_vol, axis=1)
        assert len(filter_quant_val) == len(sh_vol)
    else:
        raise ValueError()
    m = (proba >= proba_thresh_syn) & (filter_quant_val >= range_dict[filter_quant][0]) & \
        (filter_quant_val <= range_dict[filter_quant][1])
    del filter_quant_val
    log.info(f'Found {np.sum(m)} synapses after filtering with probaility threshold {proba_thresh_syn} and '
          f'size filter (min/max [um^2]: {range_dict[filter_quant]}).')
    ax[(ax == 3) | (ax == 4)] = 1  # set boutons to axon class
    ax[(ax == 5) | (ax == 6)] = 0  # set spine head and neck to dendrite class
    m = m & (np.sum(ax, axis=1) == 1)  # get all axo-dendritic synapses
    m = m & np.any(ct == str2int_converter(ct_label_pre, gt_type='ctgt_j0251_v2'), axis=1)
    m = m & np.any(ct == str2int_converter(ct_label_post, gt_type='ctgt_j0251_v2'), axis=1)
    log.info(f'Found {np.sum(m)} synapses after filtering non axo-dendritic and non {ct_label_pre}->{ct_label_post} ones.')
    area = area[m]
    size = size[m]
    partners = partners[m]
    ct = ct[m]
    posts = {'area': defaultdict(list), 'vx': defaultdict(list)}
    for ix in tqdm.tqdm(range(partners.shape[0]), total=len(partners), desc='Syns'):
        if proba_thresh_celltype is not None:
            p1, p2 = partners[ix]
            if (ct_proba_lookup[p1] < proba_thresh_celltype) or (ct_proba_lookup[p2] < proba_thresh_celltype):
                continue
        post_ix = ct[ix].tolist().index(str2int_converter(ct_label_post, gt_type='ctgt_j0251_v2'))
        if ax[ix, post_ix] != 0:  # post cell type is not post-synaptic here
            continue
        cell_id = partners[ix, post_ix]
        posts['area'][cell_id].append(area[ix])
        posts['vx'][cell_id].append(size[ix])

    # normalize per cell
    n_msn_receiving_hvc = len(posts['area'])
    n_msn_above_rel_thresh_vx = 0
    n_msn_above_rel_thresh_area = 0
    total_area, total_vx = [], []
    total_max_rel_weights_area = []
    total_max_rel_weights_vx = []
    for k, v in posts['area'].items():
        if len(v) < nsyns_min:
            continue
        v = np.array(v)
        v = v / v.sum()
        total_max_rel_weights_area.append(v.max())
        if v.max() < max_rel_weight:
            continue
        n_msn_above_rel_thresh_area += 1
        total_area.extend(v.tolist())
    for k, v in posts['vx'].items():
        if len(v) < nsyns_min:
            continue
        v = np.array(v)
        v = v / v.sum()
        total_max_rel_weights_vx.append(v.max())
        if v.max() < max_rel_weight:
            continue
        n_msn_above_rel_thresh_vx += 1
        total_vx.extend(v.tolist())
    log.info(f'Found {len(total_area)} synapses in cells which contain at least one synapse above rel. thresh (area).')
    log.info(f'Found {len(total_vx)} synapses in cells which contain at least one synapse above above rel. thresh (vx).')
    log.info(f'Cell type prediction threshold: {proba_thresh_celltype}')
    log.info(f'Synapse rel. weight threshold: {max_rel_weight}')
    log.info(f'#{ct_label_post} receiving {ct_label_pre}: {n_msn_receiving_hvc}\n'
             f'N > rel. thresh. (vx): {n_msn_above_rel_thresh_vx}, '
             f'{n_msn_above_rel_thresh_vx / n_msn_receiving_hvc:.4f}\n'
             f', N > rel. thresh. (area): {n_msn_above_rel_thresh_area}, '
             f'{n_msn_above_rel_thresh_area / n_msn_receiving_hvc:.4f}')
    plt.figure()
    ax = sns.displot(data=total_area,
                    fill=True, kde=True, kde_kws=dict(bw_adjust=0.5), common_bins=True,
                    edgecolor='none', multiple="layer", common_norm=False, stat='density',)
    ax.set(xlabel='rel. syn. mesh area')
    plt.xlim(0, 1)
    plt.savefig(f"{target_dir}/mesh_area_{ct_label_pre}on{ct_label_post}.png", dpi=300)
    plt.close('all')

    plt.figure()
    ax = sns.displot(data=total_vx,
                    fill=True, kde=True, kde_kws=dict(bw_adjust=0.5), common_bins=True,
                    edgecolor='none', multiple="layer", common_norm=False, stat='density',)
    ax.set(xlabel='rel. syn. voxel count')
    plt.xlim(0, 1)
    plt.savefig(f"{target_dir}/voxel_{ct_label_pre}on{ct_label_post}.png", dpi=300)
    plt.close('all')

    plt.figure()
    ax = sns.displot(data=total_max_rel_weights_vx, kind='ecdf', stat='proportion',)
    ax.set(xlabel='max. rel. syn. mesh area')
    plt.savefig(f"{target_dir}/max_rel_weights_vx_{ct_label_pre}on{ct_label_post}.png", dpi=300)
    plt.close('all')

    plt.figure()
    ax = sns.displot(data=total_max_rel_weights_area, kind='ecdf', stat='proportion',)
    ax.set(xlabel='max. rel. syn. voxel cnt')
    plt.savefig(f"{target_dir}/max_rel_weights_area_{ct_label_pre}on{ct_label_post}.png", dpi=300)
    plt.close('all')


def create_kde(dest_p, qs, size_quant_key: str, ls=20, legend=False, r=None, x_label: str = "", **kwargs):
    """


    Parameters
    ----------
    dest_p :
    qs :
    size_quant_key :
    r :
    x_label :
    legend :
    ls :

    Returns
    -------

    """
    if x_label == "":
        x_label = size_quant_key
    r = np.array(r)
    import seaborn as sns
    # fig, ax = plt.subplots()
    plt.figure()
    # ax = sns.swarmplot(data=qs, clip_on=False, **kwargs)
    # fill=True, kde=True, kde_kws=dict(bw_adjust=0.5), common_bins=False,
    ax = sns.displot(data=qs, x=size_quant_key, hue="cell_type",
                # kind="kde", bw_adjust=0.5,
                fill=True, kde=True, kde_kws=dict(bw_adjust=0.5), common_bins=True,
                edgecolor='none', multiple="layer", common_norm=False, stat='density',
                **kwargs)  # , common_norm=False
    ax.set(xlabel=x_label)
    if r is not None:
        xmin, xmax = plt.xlim()
        if xmin > r[0]:
            r[0] = xmin
        if xmax < r[1]:
            r[1] = xmax
        plt.xlim(r)
    # if not legend:
    #     plt.gca().legend().set_visible(False)
    # sns.set_style("ticks", {"xtick.major.size": 20, "ytick.major.size": 20})
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
