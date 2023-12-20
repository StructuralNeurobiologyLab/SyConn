# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Philipp Schubert, Joergen Kornfeld
import os
from logging import Logger
from typing import Optional, Union

import networkx as nx
import numpy as np

from . import log_proc
from .graphs import create_ccsize_dict
from .. import global_params
from ..handler.basics import load_pkl2obj, chunkify, flatten_list, \
    write_txt2kzip, write_obj2pkl
from ..mp import batchjob_utils as qu
from ..mp.mp_utils import start_multiprocess_imap as start_multiprocess
from ..reps.rep_helper import knossos_ml_from_ccs
from ..reps.segmentation import SegmentationDataset
from ..reps.super_segmentation_object import SuperSegmentationObject


def run_glia_splitting():
    """
    This function initiates the astrocyte splitting process, generating the final
    connected components of neuron vs. glia SuperVoxels (SVs). It loads the 
    connected components (CC) dictionary from the glia directory, sorts the values 
    in descending order, and then starts the batch job script for glia splitting.
    """
    cc_dict = load_pkl2obj(global_params.config.working_dir + "/glia/cc_dict_rag_graphs.pkl")
    chs = chunkify(sorted(list(cc_dict.values()), key=len, reverse=True),
                   global_params.config.ncore_total * 2)
    qu.batchjob_script(chs, "split_glia", n_cores=1, remove_jobfolder=True)


def collect_glia_sv():
    """
    This function collects astrocyte SuperVoxels (SVs) from all 'sv' SegmentationObjects 
    contained in 'sv' SegmentationDataset. It loads the connected components (CC) dictionary 
    from the glia directory, gets single SV glia probabilities which were not included in the 
    old RAG, and then collects the astrocyte SVs. It also saves the collected SVs and neuron 
    predictions in the glia directory. This function always uses the default version as 
    defined in the config.yml.
    """
    cc_dict = load_pkl2obj(global_params.config.working_dir + "/glia/cc_dict_rag.pkl")
    # get single SV glia probas which were not included in the old RAG
    ids_in_rag = np.concatenate(list(cc_dict.values()))
    sds = SegmentationDataset("sv", working_dir=global_params.config.working_dir)

    # get SSV glia splits
    chs = chunkify(list(cc_dict.keys()), global_params.config['ncores_per_node'] * 10)
    astrocyte_svs = np.concatenate(start_multiprocess(collect_gliaSV_helper, chs, debug=False,
                                                      nb_cpus=global_params.config['ncores_per_node']))
    log_proc.info("Collected SSV glia SVs.")
    # Missing SVs were sorted out by the size filter
    # TODO: Decide of those should be added to the glia RAG or not
    missing_ids = np.setdiff1d(sds.ids, ids_in_rag)
    np.save(global_params.config.working_dir + "/glia/astrocyte_svs.npy", astrocyte_svs)
    neuron_svs = np.array(list(set(sds.ids).difference(set(astrocyte_svs).union(set(missing_ids)))),
                          dtype=np.uint64)
    assert len((set(neuron_svs).union(set(astrocyte_svs)).union(set(missing_ids))).difference(set(
        sds.ids))) == 0
    np.save(global_params.config.working_dir + "/glia/neuron_svs.npy", neuron_svs)
    np.save(global_params.config.working_dir + "/glia/pruned_svs.npy", missing_ids)
    log_proc.info("Collected whole dataset glia and neuron predictions.")


def collect_gliaSV_helper(cc_ixs):
    """
    This helper function collects astrocyte SuperVoxels (SVs) for a given list of connected 
    components indices. It initializes a SuperSegmentationObject for each index, loads its 
    attribute dictionary, and appends the astrocyte SVs to the list. It returns the list as a 
    numpy array.
    
    Args:
        cc_ixs (list): List of connected components indices.
    
    Returns:
        np.array: Array of astrocyte SuperVoxels (SVs).
    """
    astrocyte_svs = []
    for cc_ix in cc_ixs:
        sso = SuperSegmentationObject(cc_ix, working_dir=global_params.config.working_dir,
                                      version="gliaremoval")
        sso.load_attr_dict()
        ad = sso.attr_dict
        astrocyte_svs += list(flatten_list(ad["astrocyte_svs"]))
    return np.array(astrocyte_svs, dtype=np.uint64)


def write_astrocyte_svgraph(rag: Union[nx.Graph, str], min_ssv_size: float,
                            log: Optional[Logger] = None):
    """
    This function stores astrocyte and neuron Region Adjacency Graphs (RAGs) as networkx edge 
    lists and as knossos merge lists in "wd + /glia/" or "wd + /neuron/". It creates neuron and 
    astrocyte RAGs, prepares a SuperVoxel (SV) size dictionary, removes small neuron and 
    astrocyte connected components (CCs), and writes the CCs to files.
    
    Args:
        rag (Union[nx.Graph, str]): SV agglomeration.
        min_ssv_size (float): Minimum bounding box diagonal in nm.
        log (Optional[Logger]): Logger for logging information. If None, the default logger is used.
    """
    if log is None:
        log = log_proc
    if type(rag) is str:
        assert os.path.isfile(rag), "RAG has to be given."
        g = nx.read_edgelist(rag, nodetype=np.uint, delimiter=',')
    else:
        g = rag
    # create neuron RAG by glia removal
    neuron_g = g.copy()
    astrocyte_svs = np.load(global_params.config.working_dir + "/glia/astrocyte_svs.npy")
    for ix in astrocyte_svs:
        neuron_g.remove_node(ix)
    # create astrocyte rag by removing neuron sv's
    astrocyte_g = g.copy()
    for ix in neuron_g.nodes():
        astrocyte_g.remove_node(ix)

    # create dictionary with CC sizes (BBD)
    log.info("Finished neuron and glia RAG, now preparing CC size dict.")
    sds = SegmentationDataset("sv", working_dir=global_params.config.working_dir, cache_properties=['size'])
    sv_size_dict = {}
    bbs = sds.load_numpy_data('bounding_box') * sds.scaling
    for ii in range(len(sds.ids)):
        sv_size_dict[sds.ids[ii]] = bbs[ii]
    ccsize_dict = create_ccsize_dict(g, sv_size_dict)
    log.info("Finished preparation of SSV size dictionary based on bounding box diagonal of corresponding SVs.")

    # add CCs with single neuron SV manually
    neuron_ids = list(neuron_g.nodes())
    all_neuron_ids = np.load(global_params.config.working_dir + "/glia/neuron_svs.npy")
    # remove small Neuron CCs
    missing_neuron_svs = set(all_neuron_ids).difference(set(neuron_ids))
    if len(missing_neuron_svs) > 0:
        msg = "Missing %d astrocyte CCs with one SV." % len(missing_neuron_svs)
        log.error(msg)
        raise ValueError(msg)
    before_cnt = len(neuron_g.nodes())
    for ix in neuron_ids:
        if ccsize_dict[ix] < min_ssv_size:
            neuron_g.remove_node(ix)
    log.info("Removed %d neuron CCs because of size." % (before_cnt - len(neuron_g.nodes())))
    ccs = list(nx.connected_components(neuron_g))
    cnt_neuron_sv = 0
    with open(global_params.config.neuron_svagg_list_path, 'w') as f:
        for cc in ccs:
            f.write(','.join([str(el) for el in cc]) + '\n')
            cnt_neuron_sv += len(cc)
    nx.write_edgelist(neuron_g, global_params.config.neuron_svgraph_path)
    log.info(f"Nb neuron CCs: {len(ccs)}")
    log.info(f"Nb neuron SVs: {cnt_neuron_sv}")

    # add glia CCs with single SV
    astrocyte_ids = list(astrocyte_g.nodes())
    missing_astrocyte_svs = set(astrocyte_svs).difference(set(astrocyte_ids))
    if len(missing_astrocyte_svs) > 0:
        msg = "Missing %d astrocyte CCs with one SV." % len(missing_astrocyte_svs)
        log.error(msg)
        raise ValueError(msg)
    before_cnt = len(astrocyte_g.nodes())
    for ix in astrocyte_ids:
        if ccsize_dict[ix] < min_ssv_size:
            astrocyte_g.remove_node(ix)
    log.info("Removed %d astrocyte CCs because of size." % (before_cnt - len(astrocyte_g.nodes())))
    ccs = list(nx.connected_components(astrocyte_g))
    total_size = 0
    for n in astrocyte_g.nodes():
        total_size += sds.get_segmentation_object(n).size
    total_size_cmm = np.prod(sds.scaling) * total_size / 1e18
    log.info("Glia RAG contains {} SVs in {} CCs ({} mm^3; {} Gvx).".format(
        astrocyte_g.number_of_nodes(), len(ccs), total_size_cmm, total_size / 1e9))
    with open(global_params.config.astrocyte_svagg_list_path, 'w') as f:
        for cc in ccs:
            f.write(','.join([str(el) for el in cc]) + '\n')
    nx.write_edgelist(astrocyte_g, global_params.config.astrocyte_svgraph_path())


def transform_rag_edgelist2pkl(rag):
    """
    This function stores a networkx graph as a dictionary mapping SuperSegmentationObject (SSO) 
    IDs to lists of SuperVoxel (SV) IDs and SSO IDs to subgraphs. It iterates over the connected 
    components of the graph, creates dictionaries mapping the minimum index of each component to 
    the list of nodes and the subgraph, and writes these dictionaries to pickle files.

    Args:
        rag (nx.Graph): The Region Adjacency Graph (RAG) to be transformed.

    Raises:
        ValueError: If multiple SuperSegmentationObject (SSO) IDs are encountered in the same connected component.
    """
    ccs = (rag.subgraph(c) for c in nx.connected_components(rag))
    cc_dict_graph = {}
    cc_dict = {}
    for cc in ccs:
        curr_cc = list(cc.nodes())
        min_ix = np.min(curr_cc)
        if min_ix in cc_dict:
            raise ValueError('Multiple SSV IDs')
        cc_dict_graph[min_ix] = cc
        cc_dict[min_ix] = curr_cc
    write_obj2pkl(global_params.config.working_dir + "/glia/cc_dict_rag_graphs.pkl",
                  cc_dict_graph)
    write_obj2pkl(global_params.config.working_dir + "/glia/cc_dict_rag.pkl", cc_dict)
