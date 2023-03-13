# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Alexandra Rother

import numpy as np
from syconn import global_params
from syconn.reps.super_segmentation import SuperSegmentationDataset

def find_full_cells_sso(cellid, celltype, soma_centre = True):
    """
    function finds full cells of a specific celltype if the cells have a dendrite, soma and axon in axoness_avg10000.
    
    Args:
        ssd: segmentation dataset 
        celltype: number of the celltype that is searched for; celltypes: j0126: STN=0, modulatory=1, MSN=2, LMAN=3, HVC=4, GP=5, INT=6 
            # j0256: STN=0, DA=1, MSN=2, LMAN=3, HVC=4, TAN=5, GPe=6, GPi=7, FS=8, LTS=9, NGF=10
        soma_centre: if True calculates average of soma skeleton notes as approximation to the soma centre
    
    Returns:
        array: an array with cell_ids of the full_cells and if soma centre was calculated also a dictionary for each cell with its soma_centre
    """
    ssd = SuperSegmentationDataset(working_dir=global_params.config.working_dir)
    cell = ssd.get_super_segmentation_object(cellid)
    cell.load_skeleton()
    axoness = cell.skeleton["axoness_avg10000"]
    axoness[axoness == 3] = 1
    axoness[axoness == 4] = 1
    unique_preds = np.unique(axoness)
    if not (0 in unique_preds and 1 in unique_preds and 2 in unique_preds):
        return 0, 0
    if soma_centre:
        soma_inds = np.nonzero(cell.skeleton["axoness_avg10000"] == 2)[0]
        if len(soma_inds) == 0:
            raise ValueError
        positions = cell.skeleton["nodes"][soma_inds] * ssd.scaling #transform to nm
        soma_centre_coord = np.sum(positions, axis=0) / len(positions)

    if soma_centre:
        return cell.id, soma_centre_coord
    else:
        return cell.id, 0


def synapse_amount_percell(celltype, sd_synssv,cellids, syn_proba):
    '''
    gives amount of synapses for each cell with defined synapse probability and writes it in a dictionary
    
    Args:
        celltype: celltype analysis is wanted for
        sd_synssv: synapse daatset
        syn_proba: synapse probability
        cellids: cellids of cells wanted amount of synapses for
    
    Returns:
        dict: dictionary with cell_ids as keys and amount of synapses
    '''
    syn_prob = sd_synssv.load_cached_data("syn_prob")
    m = syn_prob > syn_proba
    m_cts = sd_synssv.load_cached_data("partner_celltypes")[m]
    m_ssv_partners = sd_synssv.load_cached_data("neuron_partners")[m]
    inds = np.any(m_cts == celltype, axis=1)
    m_ssv_cts = m_ssv_partners[inds]

    syn_amount_dict = {i: len(m_ssv_cts[np.any(m_ssv_cts == i, axis=1)]) for i in cellids}

    return syn_amount_dict