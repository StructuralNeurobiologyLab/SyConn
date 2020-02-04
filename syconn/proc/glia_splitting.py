# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Philipp Schubert, Joergen Kornfeld
import os
import networkx as nx
import numpy as np

from ..backend.storage import AttributeDict
from .. import global_params
from ..handler.basics import load_pkl2obj, chunkify, flatten_list, \
    write_txt2kzip, write_obj2pkl
from ..mp import batchjob_utils as qu
from ..mp.mp_utils import start_multiprocess_imap as start_multiprocess
from ..reps.rep_helper import knossos_ml_from_ccs
from ..reps.segmentation import SegmentationDataset
from ..reps.super_segmentation_object import SuperSegmentationObject
from .graphs import create_ccsize_dict
from . import log_proc


def qsub_glia_splitting():
    """
    Start glia splitting -> generate final connected components of neuron vs.
    glia SVs.
    """
    cc_dict = load_pkl2obj(global_params.config.working_dir + "/glia/cc_dict_rag_graphs.pkl")
    huge_ssvs = [it[0] for it in cc_dict.items() if len(it[1]) >
                 global_params.config['glia']['rendering_max_nb_sv']]
    if len(huge_ssvs):
        log_proc.info("{} huge SSVs detected (#SVs > {})".format(
            len(huge_ssvs), global_params.config['glia']['rendering_max_nb_sv']))
    chs = chunkify(sorted(list(cc_dict.values()), key=len, reverse=True),
                   global_params.config.ncore_total * 2)
    qu.batchjob_script(chs, "split_glia", n_cores=1,
                       n_max_co_processes=global_params.config.ncore_total * 2,
                       remove_jobfolder=True)


def collect_glia_sv():
    """
    Collect glia super voxels (as returned by glia splitting) from all 'sv'
    SegmentationObjects contained in 'sv' SegmentationDataset (always uses
    default version as defined in config.yml).
    """
    cc_dict = load_pkl2obj(global_params.config.working_dir + "/glia/cc_dict_rag.pkl")
    # get single SV glia probas which were not included in the old RAG
    ids_in_rag = np.concatenate(list(cc_dict.values()))
    sds = SegmentationDataset("sv", working_dir=global_params.config.working_dir)

    # get SSV glia splits
    chs = chunkify(list(cc_dict.keys()), global_params.config['ncores_per_node'])
    glia_svs = np.concatenate(start_multiprocess(collect_gliaSV_helper, chs, debug=False,
                                                 nb_cpus=global_params.config['ncores_per_node']))
    log_proc.info("Collected SSV glia SVs.")
    # Missing SVs were sorted out by the size filter
    # TODO: Decide of those should be added to the glia RAG or not
    missing_ids = np.setdiff1d(sds.ids, ids_in_rag)
    np.save(global_params.config.working_dir + "/glia/glia_svs.npy", glia_svs)
    neuron_svs = np.array(list(set(sds.ids).difference(set(glia_svs).union(set(missing_ids)))),
                          dtype=np.uint64)
    assert len((set(neuron_svs).union(set(glia_svs)).union(set(missing_ids))).difference(set(
        sds.ids))) == 0
    np.save(global_params.config.working_dir + "/glia/neuron_svs.npy", neuron_svs)
    np.save(global_params.config.working_dir + "/glia/pruned_svs.npy", missing_ids)
    log_proc.info("Collected whole dataset glia and neuron predictions.")


def collect_gliaSV_helper(cc_ixs):
    glia_svids = []
    for cc_ix in cc_ixs:
        sso = SuperSegmentationObject(cc_ix, working_dir=global_params.config.working_dir,
                                      version="gliaremoval")
        sso.load_attr_dict()
        ad = sso.attr_dict
        glia_svids += list(flatten_list(ad["glia_svs"]))
    return np.array(glia_svids, dtype=np.uint)


def write_glia_rag(rag, min_ssv_size, suffix=""):
    """
    Stores glia and neuron RAGs in "wd + /glia/" or "wd + /neuron/" as networkx
    edge list and as knossos merge list.

    Parameters
    ----------
    rag : str or nx.Graph
    min_ssv_size : float
        Bounding box diagonal in NM
    suffix : str
        Suffix for saved RAGs
    """
    if type(rag) is str:
        assert os.path.isfile(rag), "RAG has to be given."
        g = nx.read_edgelist(rag, nodetype=np.uint, delimiter=',')
    else:
        g = rag
    # create neuron RAG by glia removal
    neuron_g = g.copy()
    glia_svs = np.load(global_params.config.working_dir + "/glia/glia_svs.npy")
    for ix in glia_svs:
        neuron_g.remove_node(ix)
    # create glia rag by removing neuron sv's
    glia_g = g.copy()
    for ix in neuron_g.nodes():
        glia_g.remove_node(ix)

    # create dictionary with CC sizes (BBD)
    log_proc.info("Finished neuron and glia RAG, now preparing CC size dict.")
    sds = SegmentationDataset("sv", working_dir=global_params.config.working_dir)
    sv_size_dict = {}
    bbs = sds.load_cached_data('bounding_box') * sds.scaling
    for ii in range(len(sds.ids)):
        sv_size_dict[sds.ids[ii]] = bbs[ii]
    ccsize_dict = create_ccsize_dict(g, sv_size_dict)
    log_proc.info("Finished preparation of SSV size dictionary based "
                  "on bounding box diagonal of corresponding SVs.")

    # add CCs with single neuron SV manually
    neuron_ids = list(neuron_g.nodes())
    all_neuron_ids = np.load(global_params.config.working_dir + "/glia/neuron_svs.npy")
    # remove small Neuron CCs
    missing_neuron_svs = set(all_neuron_ids).difference(set(neuron_ids))
    if len(missing_neuron_svs) > 0:
        msg = "Missing %d glia CCs with one SV." % len(missing_neuron_svs)
        log_proc.error(msg)
        raise ValueError(msg)
    before_cnt = len(neuron_g.nodes())
    for ix in neuron_ids:
        if ccsize_dict[ix] < min_ssv_size:
            neuron_g.remove_node(ix)
    log_proc.info("Removed %d neuron CCs because of size." %
          (before_cnt - len(neuron_g.nodes())))
    ccs = list(nx.connected_components(neuron_g))
    # Added np.min(list(cc)) to have deterministic SSV ID
    txt = knossos_ml_from_ccs([np.min(list(cc)) for cc in ccs], ccs)
    write_txt2kzip(global_params.config.working_dir + "/glia/neuron_rag_ml%s.k.zip" % suffix, txt,
                   "mergelist.txt")
    nx.write_edgelist(neuron_g, global_params.config.working_dir + "/glia/neuron_rag%s.bz2" % suffix)
    log_proc.info("Nb neuron CCs: {}".format(len(ccs)))
    log_proc.info("Nb neuron SVs: {}".format(len([n for cc in ccs for n in cc])))

    # add glia CCs with single SV
    glia_ids = list(glia_g.nodes())
    missing_glia_svs = set(glia_svs).difference(set(glia_ids))
    if len(missing_glia_svs) > 0:
        msg = "Missing %d glia CCs with one SV." % len(missing_glia_svs)
        log_proc.error(msg)
        raise ValueError(msg)
    before_cnt = len(glia_g.nodes())
    for ix in glia_ids:
        if ccsize_dict[ix] < min_ssv_size:
            glia_g.remove_node(ix)
    log_proc.info("Removed %d glia CCs because of size." %
                  (before_cnt - len(glia_g.nodes())))
    ccs = list(nx.connected_components(glia_g))
    log_proc.info("Nb glia CCs: {}".format(len(ccs)))
    log_proc.info("Nb glia SVs: {}".format(len([n for cc in ccs for n in cc])))
    nx.write_edgelist(glia_g, global_params.config.working_dir + "/glia/glia_rag%s.bz2" % suffix)
    # Added np.min(list(cc)) to have deterministic SSV ID
    txt = knossos_ml_from_ccs([np.min(list(cc)) for cc in ccs], ccs)
    write_txt2kzip(global_params.config.working_dir + "/glia/glia_rag_ml%s.k.zip" % suffix, txt,
                   "mergelist.txt")


def transform_rag_edgelist2pkl(rag):
    """
    Stores networkx graph as dictionary mapping (1) SSV IDs to lists of SV IDs
     and (2) SSV IDs to subgraphs (networkx)

    Parameters
    ----------
    rag : networkx.Graph
    """
    ccs = nx.connected_component_subgraphs(rag)
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
