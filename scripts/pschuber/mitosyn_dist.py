import logging
import os.path

import open3d as o3d
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import ks_2samp, ranksums
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm
import pandas
import pandas as pd
from collections import defaultdict
import pickle as pkl
import scipy

from typing import Optional

from syconn.mp.mp_utils import start_multiprocess_imap
from syconn.handler import basics, config
from syconn import global_params
from syconn.handler.prediction import int2str_converter, certainty_estimate, str2int_converter
from syconn.reps.super_segmentation import SuperSegmentationDataset, SuperSegmentationObject
from syconn.reps.segmentation import SegmentationDataset


def mito_syn_dist(ssv: SuperSegmentationObject, sd_syn_ssv: SegmentationDataset):
    """

    Args:
        ssv: SuperSegmentationObject.
        sd_syn_ssv:

    Returns:
        Distance (in µm) to nearest mitochondrium for all syn_ssv, synapse count density along skeleton,
        mito count density along skeleton, syn mesh areas in µm^2. Count densities are in µm^-1.
    """
    # # TODO: add adaptive compartment filter
    # if comp_of_interest is not None and len(np.setdiff1d(comp_of_interest, (1, 3, 4))) > 0:
    #     raise NotImplementedError
    # else:
    #     comp_of_interest = (1, 3, 4)
    ssv.load_attr_dict()
    pcd = o3d.geometry.PointCloud()
    mito_vertices = ssv.mi_mesh[1].reshape(-1, 3)
    pcd.points = o3d.utility.Vector3dVector(mito_vertices)
    pcd = pcd.voxel_down_sample(voxel_size=200)
    kdt = cKDTree(np.asarray(pcd.points))
    del pcd
    syn_ssv = [sd_syn_ssv.get_segmentation_object(syn_id) for syn_id in ssv.attr_dict['syn_ssv']]
    syn_reps = np.array([syn.rep_coord * ssv.scaling for syn in syn_ssv], dtype=np.float32)
    syn_ids = np.array([syn.id for syn in syn_ssv], dtype=np.uint)
    syn_ax = np.array([syn.attr_dict['partner_axoness'] for syn in syn_ssv], dtype=np.int32)
    syn_partners = np.array([syn.attr_dict['neuron_partners'] for syn in syn_ssv], dtype=np.int32)
    syn_prob = np.array([syn.attr_dict['syn_prob'] for syn in syn_ssv], dtype=np.float32)
    syn_area = np.array([syn.attr_dict['mesh_area'] for syn in syn_ssv], dtype=np.float32)
    syn_mask = (((syn_ax[syn_partners == ssv.id] == 1) | (syn_ax[syn_partners == ssv.id] == 3) |
                 (syn_ax[syn_partners == ssv.id] == 4)) & (syn_prob >= prob_thresh) &
                (syn_ax[syn_partners != ssv.id] == 0))
    filtered_syn_area = syn_area[syn_mask]
    filtered_syn_reps = syn_reps[syn_mask]
    filtered_syn_ids = syn_ids[syn_mask]
    if len(filtered_syn_reps) == 0:
        logging.warning(f'Did not find any synapse after filtering in {ssv}')
        return np.zeros((0, 3)), [], [], [], []
    dists, _ = kdt.query(filtered_syn_reps)
    ax_length = ssv.total_edge_length(compartments_of_interest=comp_of_interest) / 1e3  # nm to um
    syn_density = len(dists) / ax_length

    # # calculate mito counts within axon
    # mito_vertices = []
    # mito_ids = []
    # # collect individual mito vertices and their IDs
    # for m in ssv.mis:
    #     mito_vertices.append(m.mesh[1].reshape(-1, 3))
    #     mito_ids.append([m.id] * len(mito_vertices[-1]))
    # mito_vertices = np.concatenate(mito_vertices)
    # mito_ids = np.concatenate(mito_ids)
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(mito_vertices)
    # pcd, idcs = pcd.voxel_down_sample_and_trace(200, pcd.get_min_bound(), pcd.get_max_bound())
    # # slice out only IDs belonging to vertices that "survived" downsampling.
    # idcs = np.max(idcs, axis=1)
    # mito_ids = mito_ids[idcs]
    # total_mito_vert_cnt = {k: cnt for k, cnt in zip(*np.unique(mito_ids, return_counts=True))}
    # # get all axon skeleton nodes and find their nearest mito vertex / mito ID
    # skeleton_node_labels = np.array(ssv.skeleton['axoness_avg10000'])
    # skeleton_node_labels[skeleton_node_labels == 3] = 1
    # skeleton_node_labels[skeleton_node_labels == 4] = 1
    # kdt = cKDTree(ssv.skeleton['nodes'] * ssv.scaling)
    # dists_skelnodes, nn_ids = kdt.query(np.asarray(pcd.points))
    # mito_ids = mito_ids[(dists_skelnodes < 1e3) & (skeleton_node_labels[nn_ids] == 1)]
    # mito_ids, cnts = np.unique(mito_ids, return_counts=True)
    # filtered_ids = []
    # for ix, cnt in zip(mito_ids, cnts):
    #     if cnt / total_mito_vert_cnt[ix] > 0.5:
    #         filtered_ids.append(ix)
    # mito_density = len(filtered_ids) / ax_length
    mito_density = np.zeros_like(syn_density)
    return dists / 1e3, syn_density, mito_density, filtered_syn_area, filtered_syn_ids


def filter_ssvs_(ssv_ids):
    ssv_ids_of_interest = []
    ssd = SuperSegmentationDataset()
    for ssv_id in ssv_ids:
        ssv = ssd.get_super_segmentation_object(ssv_id)
        cert = certainty_estimate(ssv.lookup_in_attribute_dict('celltype_cnn_e3_probas'), is_logit=True)
        if cert < celltype_certainty_thresh:
            continue
        ax_length = ssv.total_edge_length(compartments_of_interest=comp_of_interest)
        if ax_length < min_edge_length:
            continue
        if celltype_label in ['MSN', 'GPe', 'GPi', 'TAN', 'FS', 'LTS', 'NGF']:
            den_length = ssv.total_edge_length(compartments_of_interest=[0])
            if den_length < min_edge_length_dendrite:
                continue
            soma_length = ssv.total_edge_length(compartments_of_interest=[2])
            if soma_length < min_edge_length_soma:
                continue
        ssv_ids_of_interest.append(ssv_id)
    return ssv_ids_of_interest


def get_properties_of_interest(ssv_ids_of_interest):
    distances = []
    syn_densities = []
    mito_densities = []
    syn_areas = []
    syn_ids = []
    ssd = SuperSegmentationDataset()
    for ssv_id in ssv_ids_of_interest:
        ssv = ssd.get_super_segmentation_object(ssv_id)
        # filter full cells
        if celltype_label in ['MSN', 'GPe', 'GPi', 'TAN', 'FS', 'LTS', 'NGF']:
            ssv.load_skeleton()
            comp_present = np.unique(ssv.skeleton['axoness_avg10000'])
            comp_present[comp_present == 3] = 1
            comp_present[comp_present == 4] = 1
            if (0 not in comp_present) or (1 not in comp_present) or (2 not in comp_present):
                raise ValueError(f'Cell without all three compartments found.')
        dsts, syn_density, mito_density, syn_area, syn_id = mito_syn_dist(ssv, sd_syn)
        if len(dsts) == 0 or np.min(dsts) == np.inf:
            continue
        syn_densities.append(syn_density)
        mito_densities.append(mito_density)
        distances.append(dsts)
        syn_areas.append(syn_area)
        syn_ids.append(syn_id)
    return distances, syn_densities, mito_densities, syn_areas, syn_ids


def probe_mito_dist_random_locs(ssv_id):
    ssd = SuperSegmentationDataset()
    ssv = ssd.get_super_segmentation_object(ssv_id)
    n_locations = 1000
    np.random.seed(np.int64(ssv.id))
    ssv.load_skeleton()
    ssv.load_attr_dict()

    nodes = ssv.skeleton['nodes']
    cmpt_type = ssv.skeleton['axoness_avg10000']
    nodes = nodes[(cmpt_type == 1) | (cmpt_type == 3) | (cmpt_type == 4)]
    np.random.shuffle(nodes)
    sample = nodes[:n_locations] * ssv.scaling
    cell_vertices = ssv.mesh[1].reshape((-1, 3))
    node_kdt = cKDTree(sample)
    # get closest vertex to every skeleton node and use this as sample
    _, nnixs = node_kdt.query(cell_vertices)  # get closest node for every vertex
    # get all vertices associated with one node
    node2vertices = defaultdict(list)
    for ix, node_ix in enumerate(nnixs):
        node2vertices[node_ix].append(ix)
    # for each node, draw one associated vertex as new sample location
    nnixs = np.array([np.random.choice(bucket, 1)[0] for bucket in node2vertices.values()], dtype=np.int64)
    sample = cell_vertices[nnixs]
    # get the closest mito distances
    pcd = o3d.geometry.PointCloud()
    mito_vertices = ssv.mi_mesh[1].reshape(-1, 3)
    pcd.points = o3d.utility.Vector3dVector(mito_vertices)
    pcd = pcd.voxel_down_sample(voxel_size=200)
    kdt = cKDTree(np.asarray(pcd.points))
    del pcd
    dists, _ = kdt.query(sample)
    return dists / 1e3  # in µm


def ecdf(data, array: bool=True):
    """Credits to Trenton McKinney: https://stackoverflow.com/questions/69300483/how-to-use-markers-with-ecdf-plot
    Compute ECDF for a one-dimensional array of measurements."""
    # Number of data points: n
    n = len(data)
    # x-data for the ECDF: x
    x = np.sort(data)
    # y-data for the ECDF: y
    y = np.arange(1, n+1) / n
    if not array:
        return pandas.DataFrame({'x': x, 'y': y})
    else:
        return x, y


def write_syn_assessment_files(base_dir):
    os.makedirs(f'{base_dir}/syn_assessment/', exist_ok=True)
    gp_upper = np.load(f'{base_dir}/GP_syns_upper.npy')[:25]
    gp_lower = np.load(f'{base_dir}/GP_syns_lower.npy')[:25]
    msn_upper = np.load(f'{base_dir}/MSN_syns_upper.npy')[:25]
    msn_lower = np.load(f'{base_dir}/MSN_syns_lower.npy')[:25]
    syns = sd_syn.get_segmentation_object(gp_upper) + sd_syn.get_segmentation_object(gp_lower) + \
           sd_syn.get_segmentation_object(msn_upper) + sd_syn.get_segmentation_object(msn_lower)
    # cache attributes
    for s in syns:
        s.load_attr_dict()
        s.mesh2kzip(f'{base_dir}/syn_assessment/syn_meshes.k.zip')
    mapping = {k: v for k, v in zip([s.id for s in syns], ['GP_upper'] * 25 + ['GP_lower'] * 25 +
                                    ['MSN_upper'] * 25 + ['MSN_lower'] * 25)}
    np.random.shuffle(syns)
    df = pd.DataFrame(
        data=dict(coord=[s.rep_coord for s in syns], type=[mapping[s.id] for s in syns], id=[s.id for s in syns]))
    df.to_csv(f'{base_dir}/syn_assessment/syns_orig.csv')
    df = pd.DataFrame(data=dict(coord=[s.rep_coord for s in syns], id=[s.id for s in syns]))
    df.to_csv(f'{base_dir}/syn_assessment/syns.csv')
    with open(f'{base_dir}/syn_assessment/mapping.pkl', 'wb') as f:
        pkl.dump(mapping, f)


if __name__ == '__main__':
    np.random.seed(0)
    # Only plot GPi/GPe and MSN
    # range between 20 nm and 20 µm
    # print median shift of lower and upper half
    # log-scale vs non log-scale
    # syn requirements
    prob_thresh = 0.8
    # cell requirements
    comp_of_interest = [1, 3, 4]
    max_dist = np.inf  # in µm. Distances will be capped by this value
    min_edge_length = 100e3  # in nm
    min_edge_length_dendrite = 50e3  # in nm
    min_edge_length_soma = 5e3  # in nm
    celltype_certainty_thresh = 0.75
    print(f'Using syn. proba {prob_thresh}, compartments of interest: {comp_of_interest}, '
          f'max. mito-syn dist. {max_dist} µm, min. path length of cell {min_edge_length // 1e3} µm,'
          f'cell type certainty thresh. {celltype_certainty_thresh}.')

    dest_dir = f'/wholebrain/scratch/pschuber/syconn_v2_paper/figures/syn_mito_analysis/synproba_{prob_thresh}/'
    os.makedirs(dest_dir, exist_ok=True)
    global_params.wd = '/ssdscratch/songbird/j0251/rag_flat_Jan2019_v3/'
    sd_syn = SegmentationDataset(
        'syn_ssv', cache_properties=['rep_coord', 'partner_axoness', 'neuron_partners', 'syn_prob',
                                     'mesh_area'])
    if os.path.isfile(f'{dest_dir}/syn_dens_labels.npy'):
        dists_total = np.load(f'{dest_dir}/dists_total.npy')
        syn_ids_total = np.load(f'{dest_dir}/syn_ids_total.npy')
        syn_areas_total = np.load(f'{dest_dir}/syn_areas_total.npy')
        celltype_label_total = np.load(f'{dest_dir}/celltype_label_total.npy')
        syn_densities_total = np.load(f'{dest_dir}/syn_densities_total.npy')
        mito_densities_total = np.load(f'{dest_dir}/mito_densities_total.npy')
        syn_dens_labels = np.load(f'{dest_dir}/syn_dens_labels.npy')
    else:
        ssd = SuperSegmentationDataset()
        cts = ssd.load_numpy_data('celltype_cnn_e3')
        dists_total = []
        syn_areas_total = []
        syn_ids_total = []
        syn_densities_total = []
        mito_densities_total = []
        celltype_label_total = []
        syn_dens_labels = []
        for ct in [str2int_converter('MSN', gt_type='ctgt_j0251_v2'),
                   str2int_converter('GPe', gt_type='ctgt_j0251_v2'),
                   str2int_converter('GPi', gt_type='ctgt_j0251_v2')]:  # np.unique(cts)
            celltype_label = int2str_converter(ct, gt_type='ctgt_j0251_v2')
            # sub-sample cells of cell type ct deterministically.
            ssv_ids = ssd.ssv_ids[cts == ct]
            distances = []
            syn_areas = []
            syn_ids = []
            syn_densities = []
            mito_densities = []
            if os.path.isfile(f'{dest_dir}/ssv_ids_of_interest_ct{ct}.npy'):
                ssv_ids_of_interest = np.load(f'{dest_dir}/ssv_ids_of_interest_ct{ct}.npy')
            else:
                ssv_ids_of_interest = start_multiprocess_imap(filter_ssvs_, basics.chunkify(ssv_ids, 500),
                                                              nb_cpus=None)
                ssv_ids_of_interest = np.concatenate(ssv_ids_of_interest)
                np.save(f'{dest_dir}/ssv_ids_of_interest_ct{ct}.npy', ssv_ids_of_interest)
            if len(ssv_ids_of_interest) == 0:
                print(f'WARNING: Did not find any {celltype_label} cell after filtering.')
                continue
            print(f'Found {len(ssv_ids_of_interest)} valid {celltype_label}s.')
            res = start_multiprocess_imap(get_properties_of_interest,
                                          basics.chunkify(ssv_ids_of_interest, 500),
                                          nb_cpus=None)
            # res is a list of tuples of lists [(distances, syn dens, mito dens), (...), ...]
            for dsts, syn_density, mito_density, syn_area, syn_id in res:
                syn_densities.extend(syn_density)
                mito_densities.extend(mito_density)
                distances.extend(dsts)
                syn_areas.extend(syn_area)
                syn_ids.extend(syn_id)
            if len(distances) == 0:
                print(f'WARNING: Did not find any {celltype_label} synapse after filtering.')
                continue
            distances = np.concatenate(distances)
            syn_areas = np.concatenate(syn_areas)
            syn_ids = np.concatenate(syn_ids)
            dists_total.append(distances)
            syn_areas_total.append(syn_areas)
            syn_ids_total.append(syn_ids)
            celltype_label_total.append([celltype_label] * len(distances))
            print(f'Found {len(distances)} valid {celltype_label} synapses.')

            syn_densities = np.array(syn_densities)
            # label array can be used for syns and mitos
            syn_dens_labels.append([celltype_label] * len(syn_densities))
            syn_densities_total.append(syn_densities)
            mito_densities_total.append(mito_densities)
            assert len(mito_densities) == len(syn_densities)

            plt.figure()
            sns.histplot(x=distances)
            plt.xlabel('Mito-syn distance [µm]')
            plt.savefig(f'{dest_dir}/mito_dens_ct{celltype_label}.png')
        dists_total = np.concatenate(dists_total)
        syn_areas_total = np.concatenate(syn_areas_total)
        syn_ids_total = np.concatenate(syn_ids_total)
        celltype_label_total = np.concatenate(celltype_label_total)
        syn_dens_labels = np.concatenate(syn_dens_labels)
        syn_densities_total = np.concatenate(syn_densities_total)
        mito_densities_total = np.concatenate(mito_densities_total)
        np.save(f'{dest_dir}/dists_total.npy', dists_total)
        np.save(f'{dest_dir}/syn_areas_total.npy', syn_areas_total)
        np.save(f'{dest_dir}/syn_ids_total.npy', syn_ids_total)
        np.save(f'{dest_dir}/celltype_label_total.npy', celltype_label_total)
        np.save(f'{dest_dir}/syn_densities_total.npy', syn_densities_total)
        np.save(f'{dest_dir}/mito_densities_total.npy', mito_densities_total)
        np.save(f'{dest_dir}/syn_dens_labels.npy', syn_dens_labels)

    if not os.path.isfile(f'{dest_dir}/dists_control_total.npy'):
        dsts_control_total = []
        celltype_labels_control_total = []
        for ct in [str2int_converter('MSN', gt_type='ctgt_j0251_v2'),
                   str2int_converter('GPe', gt_type='ctgt_j0251_v2'),
                   str2int_converter('GPi', gt_type='ctgt_j0251_v2')]:
            celltype_label = int2str_converter(ct, gt_type='ctgt_j0251_v2')
            ssv_ids_of_interest = np.load(f'{dest_dir}/ssv_ids_of_interest_ct{ct}.npy')
            dsts = start_multiprocess_imap(probe_mito_dist_random_locs, ssv_ids_of_interest, nb_cpus=None, debug=False)
            dsts = np.concatenate(dsts)
            celltype_labels = [celltype_label] * len(dsts)
            dsts_control_total.append(dsts)
            celltype_labels_control_total.append(celltype_labels)
        celltype_labels_control_total = np.concatenate(celltype_labels_control_total)
        dsts_control_total = np.concatenate(dsts_control_total)
        np.save(f'{dest_dir}/dists_control_total.npy', dsts_control_total)
        np.save(f'{dest_dir}/celltype_label_control_total.npy', celltype_labels_control_total)
    else:
        dsts_control_total = np.load(f'{dest_dir}/dists_control_total.npy')
        celltype_labels_control_total = np.load(f'{dest_dir}/celltype_label_control_total.npy')

    celltype_label_total = celltype_label_total[dists_total <= max_dist]
    syn_areas_total = syn_areas_total[dists_total <= max_dist]
    syn_ids_total = syn_ids_total[dists_total <= max_dist]
    dists_total = dists_total[dists_total <= max_dist]

    cts_to_plot = ['MSN', 'GP']
    cmap = sns.color_palette("colorblind", len(cts_to_plot))

    dists_total_upper = dists_total
    syn_areas_total_upper = syn_areas_total
    syn_ids_total_upper = syn_ids_total
    celltype_label_total_upper = celltype_label_total
    dists_total_lower = np.array(dists_total)
    syn_areas_total_lower = np.array(syn_areas_total)
    syn_ids_total_lower = np.array(syn_ids_total)
    celltype_label_total_lower = np.array(celltype_label_total)

    celltype_label_total_upper[(celltype_label_total_upper == 'GPe') | (celltype_label_total_upper == 'GPi')] = 'GP'
    celltype_label_total_lower[(celltype_label_total_lower == 'GPe') | (celltype_label_total_lower == 'GPi')] = 'GP'
    syn_dens_labels[(syn_dens_labels == 'GPe') | (syn_dens_labels == 'GPi')] = 'GP'

    celltype_labels_control_total[(celltype_labels_control_total == 'GPe') | (celltype_labels_control_total == 'GPi')] = 'GP'

    for celltype_label in np.unique(celltype_label_total):
        distances = dists_total[celltype_label_total == celltype_label]
        print(f'Using {len(distances)} {celltype_label} synapses.')
        print(f'median+-std dist: {(np.median(distances)):.4f}+-{(np.std(distances)):.4f}')
        print(f'Median area: {np.median(syn_areas_total[celltype_label_total == celltype_label])}')
        print(f'Using {np.sum(celltype_labels_control_total == celltype_label)} {celltype_label} control locations.')
        print(f'median+-std dist: {(np.mean(dsts_control_total[celltype_labels_control_total == celltype_label])):.4f}+-'
              f'{(np.std(dsts_control_total[celltype_labels_control_total == celltype_label])):.4f}')
        print()

    for ct in np.unique(syn_dens_labels):
        if ct not in cts_to_plot:
            # per cell quantities
            syn_densities_total = syn_densities_total[syn_dens_labels != ct]
            mito_densities_total = mito_densities_total[syn_dens_labels != ct]
            syn_dens_labels = syn_dens_labels[syn_dens_labels != ct]

            # per synapse quantity
            dists_total_upper = dists_total_upper[celltype_label_total_upper != ct]
            syn_areas_total_upper = syn_areas_total_upper[celltype_label_total_upper != ct]
            syn_ids_total_upper = syn_ids_total_upper[celltype_label_total_upper != ct]
            celltype_label_total_upper = celltype_label_total_upper[celltype_label_total_upper != ct]

            dists_total_lower = dists_total_lower[celltype_label_total_lower != ct]
            syn_areas_total_lower = syn_areas_total_lower[celltype_label_total_lower != ct]
            syn_ids_total_lower = syn_ids_total_lower[celltype_label_total_lower != ct]
            celltype_label_total_lower = celltype_label_total_lower[celltype_label_total_lower != ct]
        else:
            # at this point syn_areas_total_upper == syn_areas_total_lower
            med = np.median(syn_areas_total_upper[celltype_label_total_upper == ct])
            # upper half
            celltype_spec_area_mask = ((syn_areas_total_upper > med) & (celltype_label_total_upper == ct)) | (celltype_label_total_upper != ct)
            dists_total_upper = dists_total_upper[celltype_spec_area_mask]
            celltype_label_total_upper = celltype_label_total_upper[celltype_spec_area_mask]
            syn_areas_total_upper = syn_areas_total_upper[celltype_spec_area_mask]
            syn_ids_total_upper = syn_ids_total_upper[celltype_spec_area_mask]
            print(f'{ct}: Distance median upper half {np.median(dists_total_upper[celltype_label_total_upper == ct])}')

            # lower half
            celltype_spec_area_mask = ((syn_areas_total_lower <= med) & (celltype_label_total_lower == ct)) | (
                    celltype_label_total_lower != ct)
            dists_total_lower = dists_total_lower[celltype_spec_area_mask]
            celltype_label_total_lower = celltype_label_total_lower[celltype_spec_area_mask]
            syn_areas_total_lower = syn_areas_total_lower[celltype_spec_area_mask]
            syn_ids_total_lower = syn_ids_total_lower[celltype_spec_area_mask]
            print(f'{ct}: Distance median upper half {np.median(dists_total_lower[celltype_label_total_lower == ct])}')

            # print sample IDs
            ct_syns_upper = syn_ids_total_upper[celltype_label_total_upper == ct]
            np.random.shuffle(ct_syns_upper)
            np.save(f'{dest_dir}/{ct}_syns_upper.npy', ct_syns_upper[:50])
            print(f'{len(ct_syns_upper[:50])} random {ct} synapses upper: {ct_syns_upper[:50]}')

            ct_syns_lower = syn_ids_total_lower[celltype_label_total_lower == ct]
            np.random.shuffle(ct_syns_lower)
            np.save(f'{dest_dir}/{ct}_syns_lower.npy', ct_syns_lower[:50])
            print(f'{len(ct_syns_lower[:50])} random {ct} synapses lower: {ct_syns_lower[:50]}')

            print(f'Found {len(syn_dens_labels[syn_dens_labels == ct])} {ct} cells.')
            print(f'Mean+-std syn. density: {np.mean(syn_densities_total[syn_dens_labels == ct])} +- '
                  f'{np.std(syn_densities_total[syn_dens_labels == ct])}')
            print(f'Median, Q1, Q3 of syn. density: {np.median(syn_densities_total[syn_dens_labels == ct])},'
                  f'{np.quantile(syn_densities_total[syn_dens_labels == ct], 0.25)},'
                  f'{np.quantile(syn_densities_total[syn_dens_labels == ct], 0.75)}')
            print(f'Mean+-std mito. density: {np.mean(mito_densities_total[syn_dens_labels == ct])} +- '
                  f'{np.std(mito_densities_total[syn_dens_labels == ct])}')

    # compare inter-celltype (upper)
    log = config.initialize_logging('kstest_results', f'{dest_dir}/')
    log.info(f'scipy version: {scipy.__version__}; statistics generated with scipy.stats.ks_2samp(..., '
             f'alternative="two-sided", mode="asymp")')
    test_ks_upper_gp_msn = ks_2samp(dists_total_upper[celltype_label_total_upper == 'MSN'],
                                    dists_total_upper[celltype_label_total_upper == 'GP'],
                                    alternative='two-sided', mode='asymp')
    log.info(f'KS test results for GP (upper) vs MSN (upper):\n{test_ks_upper_gp_msn}')

    # compare intra-celltype upper vs. lower
    test_ks_upper_gp_lower_gp = ks_2samp(dists_total_upper[celltype_label_total_upper == 'GP'],
                                         dists_total_lower[celltype_label_total_lower == 'GP'],
                                         alternative='two-sided', mode='asymp')
    log.info(f'KS test results for GP (upper) vs GP (lower):\n{test_ks_upper_gp_lower_gp}')

    test_ks_upper_msn_lower_msn = ks_2samp(dists_total_upper[celltype_label_total_upper == 'MSN'],
                                           dists_total_lower[celltype_label_total_lower == 'MSN'],
                                           alternative='two-sided', mode='asymp')
    log.info(f'KS test results for MSN (upper) vs MSN (lower):\n{test_ks_upper_msn_lower_msn}')

    # compare upper and lower to control (intra-celltype)
    test_ks_upper_gp_control = ks_2samp(dists_total_upper[celltype_label_total_upper == 'GP'],
                                        dsts_control_total[celltype_labels_control_total == 'GP'],
                                        alternative='two-sided', mode='asymp')
    log.info(f'KS test results for GP (upper) vs GP (control):\n{test_ks_upper_gp_control}')

    test_ks_lower_gp_control = ks_2samp(dists_total_lower[celltype_label_total_lower == 'GP'],
                                        dsts_control_total[celltype_labels_control_total == 'GP'],
                                        alternative='two-sided', mode='asymp')
    log.info(f'KS test results for GP (lower) vs GP (control):\n{test_ks_lower_gp_control}')

    # use mode='asymp' because it won't finish otherwise
    test_ks_upper_msn_control = ks_2samp(dists_total_upper[celltype_label_total_upper == 'MSN'],
                                         dsts_control_total[celltype_labels_control_total == 'MSN'],
                                         alternative='two-sided', mode='asymp')
    log.info(f'KS test results for MSN (upper) vs MSN (control):\n{test_ks_upper_msn_control}')

    test_ks_lower_msn_control = ks_2samp(dists_total_lower[celltype_label_total_lower == 'MSN'],
                                         dsts_control_total[celltype_labels_control_total == 'MSN'],
                                         alternative='two-sided', mode='asymp')
    log.info(f'KS test results for MSN (lower) vs MSN (control):\n{test_ks_lower_msn_control}')

    df = pandas.DataFrame(data={'densities': syn_densities_total, 'celltype': syn_dens_labels})
    log = config.initialize_logging('ranksumtest_results', f'{dest_dir}/')
    log.info('Using Wilcoxon_rank-sum_test "scipy.stats.ranksums" for GP vs MSN synapse counts')
    test_rank = ranksums(syn_densities_total[syn_dens_labels == 'MSN'],
                         syn_densities_total[syn_dens_labels == 'GP'])
    log.info(f'{test_rank}')

    plt.figure(figsize=(2.5, 3.5))
    df = pandas.DataFrame.from_dict(dict(distances=dists_total_upper,
                                         celltype=celltype_label_total_upper))
    df.to_csv(f'{dest_dir}/mito_syn_dist_cum_medianSplit_totalUpper.csv')
    for ct, ls in zip(df['celltype'].unique(), ['-', '-', '-', '-', '-']):
        x, y = ecdf(df[df['celltype'] == ct].distances)
        print(f'{ct}: Distance median upper half {np.median(df[df["celltype"] == ct].distances)}')
        plt.plot(x, y, linestyle=ls, label=ct, c=cmap[cts_to_plot.index(ct)])
    df = pandas.DataFrame.from_dict(dict(distances=dists_total_lower,
                                         celltype=celltype_label_total_lower))
    df.to_csv(f'{dest_dir}/mito_syn_dist_cum_medianSplit_totalLower.csv')
    for ct, ls in zip(df['celltype'].unique(), ['--', '--', '--', '--', '--']):
        x, y = ecdf(df[df['celltype'] == ct].distances)
        print(f'{ct}: Distance median lower half {np.median(df[df["celltype"] == ct].distances)}')
        plt.plot(x, y, linestyle=ls, label=ct, c=cmap[cts_to_plot.index(ct)])
    df = pandas.DataFrame.from_dict(dict(distances=dsts_control_total,
                                         celltype=celltype_labels_control_total))
    df.to_csv(f'{dest_dir}/mito_syn_dist_cum_medianSplit_control.csv')
    for ct, ls in zip(df['celltype'].unique(), [':', ':', ':', ':', ':']):
        x, y = ecdf(df[df['celltype'] == ct].distances)
        plt.plot(x, y, linestyle=ls, label=ct, c=cmap[cts_to_plot.index(ct)])
    plt.xlabel('Mito-syn distance [µm]')
    plt.xlim(0.04, 10)
    sns.despine()
    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig(f'{dest_dir}/mito_syn_dist_cum_medianSplit.png', dpi=400)
    plt.xlim(0.04, 20)
    plt.xscale('log')
    plt.tight_layout()
    plt.savefig(f'{dest_dir}/mito_syn_dist_cum_log_medianSplit.png', dpi=400)

    plt.figure(figsize=(2.5, 3.5))
    df = pandas.DataFrame.from_dict(dict(distances=np.concatenate([dists_total_upper, dists_total_lower]),
                          celltype=np.concatenate([celltype_label_total_upper, celltype_label_total_lower])))
    df.to_csv(f'{dest_dir}/mito_syn_dist_cum_alldata.csv')
    for ct, ls in zip(df['celltype'].unique(), ['-', '-', '-', '-', '-']):
        x, y = ecdf(df[df['celltype'] == ct].distances)
        plt.plot(x, y, linestyle=ls, label=ct, c=cmap[cts_to_plot.index(ct)])
    df = pandas.DataFrame.from_dict(dict(distances=dsts_control_total,
                                         celltype=celltype_labels_control_total))
    df.to_csv(f'{dest_dir}/mito_syn_dist_cum_control.csv')
    for ct, ls in zip(df['celltype'].unique(), [':', ':', ':', ':', ':']):
        x, y = ecdf(df[df['celltype'] == ct].distances)
        plt.plot(x, y, linestyle=ls, label=ct, c=cmap[cts_to_plot.index(ct)])
    plt.xlabel('Mito-syn distance [µm]')
    plt.xlim(0.04, 10)
    sns.despine()
    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig(f'{dest_dir}/mito_syn_dist_cum.png', dpi=400)
    plt.xlim(0.04, 20)
    plt.xscale('log')
    plt.tight_layout()
    plt.savefig(f'{dest_dir}/mito_syn_dist_cum_log.png', dpi=400)

    plt.figure()
    sns.displot(data=dict(distances=dists_total_lower,
                          celltype=celltype_label_total_lower),
                hue='celltype', x='distances',
                element="step", fill=False, cumulative=True, stat="density", common_norm=False,
                kind='hist', rug=False)
    plt.xlabel('Mito-syn distance [µm]')
    plt.xscale('log')
    sns.despine()
    plt.xlim(0.02, 20)
    plt.tight_layout()
    plt.savefig(f'{dest_dir}/mito_syn_dist_cum_lower_half.png', dpi=400)

    plt.figure()
    sns.displot(data=dict(distances=dists_total_upper,
                          celltype=celltype_label_total_upper),
                hue='celltype', x='distances',
                element="step", fill=False, cumulative=True, stat="density", common_norm=False,
                kind='hist', rug=False)
    plt.xlabel('Mito-syn distance [µm]')
    plt.xscale('log')
    sns.despine()
    plt.xlim(0.02, 20)
    plt.tight_layout()
    plt.savefig(f'{dest_dir}/mito_syn_dist_cum_upper_half.png', dpi=400)

    plt.figure()
    sns.displot(data=dict(distances=dists_total_upper, celltype=celltype_label_total_upper), hue='celltype', x='distances',
                kind='kde', rug=False, common_norm=False,)
    plt.xlabel('Mito-syn distance [µm]')
    sns.despine()
    plt.xlim(0.0, 20)
    plt.tight_layout()
    plt.savefig(f'{dest_dir}/mito_syn_dist_kde_upper.png', dpi=400)

    plt.figure()
    sns.displot(data=dict(distances=dists_total_lower, celltype=celltype_label_total_lower), hue='celltype', x='distances',
                kind='kde', rug=False, common_norm=False,)
    plt.xlabel('Mito-syn distance [µm]')
    sns.despine()
    plt.xlim(0.0, 20)
    plt.tight_layout()
    plt.savefig(f'{dest_dir}/mito_syn_dist_kde_lower.png', dpi=400)

    df = pandas.DataFrame(data={'densities': syn_densities_total, 'celltype': syn_dens_labels})
    df.to_csv(f'{dest_dir}/syn_count.csv')
    plt.figure()
    # sns.catplot(data=df, y='densities', x='celltype', kind="violin", cut=0)
    bp = sns.boxplot(data=df, y='densities', x='celltype')
    sns.despine()
    plt.xlabel('cell type')
    plt.ylabel('synapse count [µm^-1]')
    plt.tight_layout()

    plt.savefig(f'{dest_dir}/syn_count.png', dpi=400)

    df = pandas.DataFrame(data={'densities': mito_densities_total, 'celltype': syn_dens_labels})
    df.to_csv(f'{dest_dir}/mito_count.csv')
    plt.figure()
    # sns.catplot(data=df, y='densities', x='celltype', kind="violin", cut=0)
    sns.boxplot(data=df, y='densities', x='celltype')
    sns.despine()
    plt.xlabel('cell type')
    plt.ylabel('mito. count [µm^-1]')
    plt.tight_layout()
    plt.savefig(f'{dest_dir}/mito_count.png', dpi=400)

    plt.close('all')

    write_syn_assessment_files(dest_dir)


