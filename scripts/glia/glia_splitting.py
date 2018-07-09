import os
from syconn.config.global_params import wd, glia_thresh, min_single_sv_size
from syconn.mp import qsub_utils as qu
from syconn.mp.shared_mem import start_multiprocess
from syconn.reps.segmentation import SegmentationDataset
from syconn.reps.super_segmentation import SuperSegmentationObject
from syconn.handler.basics import *
from syconn.handler.compression import AttributeDict
from syconn.reps.rep_helper import parse_cc_dict_from_kml, knossos_ml_from_ccs
import numpy as np
import networkx as nx
import itertools


def qsub_glia_splitting():
    """
    Start glia splitting -> generate final connected components of neuron vs.
    glia SVs
    """
    cc_dict = load_pkl2obj(wd + "/glia/cc_dict_rag_graphs.pkl")
    huge_ssvs = [it[0] for it in cc_dict.items() if len(it[1]) > 3e5]
    if len(huge_ssvs):
        print("%d huge SSVs detected (#SVs > 3e5)\n%s" %
              (len(huge_ssvs), huge_ssvs))
    script_folder = os.path.dirname(
        os.path.abspath(__file__)) + "/../../syconn/QSUB_scripts/"
    chs = chunkify(cc_dict.values(), 1000)
    qu.QSUB_script(chs, "split_glia", pe="openmp", queue=None,
                   script_folder=script_folder, n_max_co_processes=100)


def collect_glia_sv():
    """
    Collect glia super voxels (as returned by glia splitting) from all 'sv'
    SegmentationObjects contained in 'sv' SegmentationDataset (always uses
    default version as defined in config.ini).
    """
    cc_dict = load_pkl2obj(wd + "/glia/cc_dict_rag.pkl")
    # get single SV glia probas which were not included in the old RAG
    ids_in_rag = np.concatenate(cc_dict.values())
    sds = SegmentationDataset("sv", working_dir=wd)
    # get all SV glia probas (faster than single access)
    # TODO: check if there is a way to auto-generate folder structure
    f1 = np.arange(0, 100)
    f2 = np.arange(0, 100)
    f3 = np.arange(0, 10)
    all_poss_attr_dicts = list(itertools.product(f1, f2, f3))
    assert len(all_poss_attr_dicts) == 100*100*10
    fold = sds.so_storage_path
    multi_params = ["%s/%d/%d/%d/" % (fold, par[0], par[1], par[2])
                    for par in all_poss_attr_dicts]
    # glia predictions only used for SSVs which only have single SV and were
    # not contained in RAG
    glia_preds_list = start_multiprocess(collect_gliaSV_helper_chunked,
                                         multi_params, nb_cpus=20, debug=False)
    glia_preds = {}
    for dc in glia_preds_list:
        glia_preds.update(dc)
    print("Collected SV glianess.")
    # get SSV glia splits
    chs = chunkify(cc_dict.keys(), 1000)
    glia_svs = np.concatenate(start_multiprocess(collect_gliaSV_helper, chs,
                                                 nb_cpus=20))
    print("Collected SSV glia SVs.")
    # add missing SV glianess and store whole dataset classification
    missing_ids = np.setdiff1d(sds.ids, ids_in_rag)
    single_sv_glia = np.array([ix for ix in missing_ids if glia_preds[ix] == 1],
                              dtype=np.uint64)
    glia_svs = np.concatenate([single_sv_glia, glia_svs]).astype(np.uint64)
    print("Collected whole dataset glia predictions.")
    np.save(wd + "/glia/glia_svs.npy", glia_svs)
    neuron_svs = np.array(list(set(sds.ids).difference(set(glia_svs))),
                          dtype=np.uint64)
    np.save(wd + "/glia/neuron_svs.npy", neuron_svs)


def collect_gliaSV_helper(cc_ixs):
    glia_svids = []
    for cc_ix in cc_ixs:
        sso = SuperSegmentationObject(cc_ix, working_dir=wd,
                                      version="gliaremoval")
        sso.load_attr_dict()
        ad = sso.attr_dict
        glia_svids += list(flatten_list(ad["glia_svs"]))
    return np.array(glia_svids)


def collect_gliaSV_helper_chunked(path):
    """
    Fast, chunked way to collect glia predictions.

    Parameters
    ----------
    path : str

    Returns
    -------
    dict
    """
    ad = AttributeDict(path + "attr_dict.pkl")
    glia_preds = {}
    for k, v in ad.iteritems():
        # see syconn.reps.segmentation_helper.glia_pred_so
        glia_pred = 0
        preds = np.array(v["glia_probas"][:, 1] > glia_thresh, dtype=np.int)
        pred = np.mean(v["glia_probas"][:, 1]) > glia_thresh
        if pred == 0:
            glia_pred = 0
        glia_votes = np.sum(preds)
        if glia_votes > int(len(preds) * 0.7):
            glia_pred = 1
        glia_preds[k] = glia_pred
    return glia_preds


def write_glia_rag(path2rag, suffix=""):
    assert os.path.isfile(path2rag), "Reconnect RAG has to be given."
    g = nx.read_edgelist(path2rag, nodetype=int,
                         delimiter=',')
    glia_svs = np.load(wd + "/glia/glia_svs.npy")
    neuron_g = g.copy()
    for ix in glia_svs:
        try:
            neuron_g.remove_node(ix)
        except:
            continue
    # create glia rag by removing neuron sv's
    glia_g = g.copy()
    for ix in neuron_g.nodes():
        glia_g.remove_node(ix)
    # add single CCs with single SV manually
    neuron_ids = neuron_g.nodes()
    all_neuron_ids = np.load(wd + "/glia/neuron_svs.npy")
    sds = SegmentationDataset("sv", working_dir=wd)
    all_size_dict = {}
    for i in range(len(sds.ids)):
        sv_ix, sv_size = sds.ids[i], sds.sizes[i]
        all_size_dict[sv_ix] = sv_size
    missing_neuron_svs = set(all_neuron_ids).difference(neuron_ids)
    before_cnt = len(neuron_g.nodes())
    for ix in missing_neuron_svs:
        if all_size_dict[ix] > min_single_sv_size:
            neuron_g.add_node(ix)
            neuron_g.add_edge(ix, ix)
    print "Added %d neuron CCs with one SV." % (len(neuron_g.nodes()) - before_cnt)
    ccs = sorted(list(nx.connected_components(neuron_g)), reverse=True, key=len)
    txt = knossos_ml_from_ccs([list(cc)[0] for cc in ccs], ccs)
    write_txt2kzip(wd + "/glia/neuron_rag_ml%s.k.zip" % suffix, txt, "mergelist.txt")
    nx.write_edgelist(neuron_g, wd + "/glia/neuron_rag%s.bz2" % suffix)
    print "Nb neuron CC's:", len(ccs), len(ccs[0])
    # add glia CCs with single SV
    missing_glia_svs = set(glia_svs).difference(glia_g.nodes())
    before_cnt = len(glia_g.nodes())
    for ix in missing_glia_svs:
        if all_size_dict[ix] > min_single_sv_size:
            glia_g.add_node(ix)
            glia_g.add_edge(ix, ix)
    print "Added %d glia CCs with one SV." % (len(glia_g.nodes()) - before_cnt)
    ccs = list(nx.connected_components(glia_g))
    print "Nb glia CC's:", len(ccs)
    nx.write_edgelist(glia_g, wd + "/glia/glia_rag%s.bz2" % suffix)
    txt = knossos_ml_from_ccs([list(cc)[0] for cc in ccs], ccs)
    write_txt2kzip(wd + "/glia/glia_rag_ml%s.k.zip" % suffix, txt, "mergelist.txt")


def write_recon_EL2pkl(rag):
    ccs = nx.connected_component_subgraphs(rag)
    cc_dict_graph = {}
    cc_dict = {}
    for cc in ccs:
        curr_cc = list(cc.nodes())
        min_ix = np.min(curr_cc)
        if min_ix in cc_dict:
            raise("laksnclkadsnfldskf")
        cc_dict_graph[min_ix] = cc
        cc_dict[min_ix] = curr_cc
    write_obj2pkl(wd + "/glia/cc_dict_rag_graphs.pkl", cc_dict_graph)
    write_obj2pkl(wd + "/glia/cc_dict_rag.pkl", cc_dict)


if __name__ == "__main__":
    # path to networkx file containing the initial rag, create alternative formats
    rag_fname = wd + "/rag.nx"
    G = nx.read_edgelist(rag_fname, nodetype=int, delimiter=',')
    if not os.path.isdir(wd + "/glia/"):
        os.makedirs(wd + "/glia/")
    write_recon_EL2pkl(G)

    # first perform glia splitting based on multi-view predictions, results are
    # stored at SuperSegmentationDataset ssv_gliaremoval
    qsub_glia_splitting()

    # collect all neuron and glia SVs and store them in numpy array
    collect_glia_sv()

    # here use reconnected RAG or initial rag
    recon_nx = wd + "/reconnect_rag.nx"
    # create glia / neuron RAGs
    write_glia_rag(recon_nx)