# -*- coding: utf-8 -*-
import copy
import gc
import numpy as np
import re
import time
import os
from multiprocessing import Pool, Manager, cpu_count
from numpy import array as arr
from scipy import spatial
from sys import stdout
import cPickle as pickle

from scipy.sparse.csgraph._tools import csgraph_from_dense
from sklearn.externals import joblib

from knossos_utils.knossosdataset import KnossosDataset
from syconn.multi_proc.multi_proc_main import start_multiprocess
from syconn.utils import skeleton_utils as su
from syconn.processing.axoness import predict_axoness_from_nodes
from syconn.processing.mapper import feature_valid_syns, calc_syn_dict
from syconn.processing.synapticity import parse_synfeature_from_node
from syconn.utils.datahandler import get_filepaths_from_dir, \
    load_ordered_mapped_skeleton, get_paths_of_skelID
from syconn.utils.datahandler import write_obj2pkl, load_pkl2obj
from syconn.utils.skeleton import Skeleton, SkeletonAnnotation

__author__ = 'pschuber'


def collect_contact_sites(cs_dir, only_sj=False):
    """Collect contact site nodes and corresponding features of all contact sites.

    Parameters
    ----------
    cs_dir : str
        Path to contact site directory
    only_sj : bool
        If only synapse candidates are to be saved
    cs_dir : str
        path to contact site directory
    only_sj : bool

    Returns
    -------
    list of SkeletonNodes, np.array
        CS nodes (n x 3), synapse feature (n x #feat)
    """
    if not only_sj:
        search_folder = ['cs_sj/', 'cs_vc_sj/', 'cs/', 'cs_vc/']
    else:
        search_folder = ['cs_sj/', 'cs_vc_sj/']
    sample_list_len = []
    cs_fpaths = []
    for k, ending in enumerate(search_folder):
        curr_dir = cs_dir+ending
        curr_fpaths = get_filepaths_from_dir(curr_dir, ending='nml')
        cs_fpaths += curr_fpaths
        sample_list_len.append(len(curr_fpaths))
    print "Collecting contact sites (%d CS). Only sj is %s." % (len(cs_fpaths),
                                                                str(only_sj))
    if len(cs_fpaths) == 0:
        print "No data available. Returning."
        return [], np.zeros((0, ))
    nb_cpus = cpu_count() - 2
    pool = Pool(processes=nb_cpus)
    m = Manager()
    q = m.Queue()
    params = [(sample, q) for sample in cs_fpaths]
    result = pool.map_async(calc_cs_node, params)
    while True:
        if result.ready():
            break
        else:
            size = float(q.qsize())
            stdout.write("\r%0.2f" % (size / len(params)))
            stdout.flush()
            time.sleep(1)
    res = result.get()
    pool.close()
    pool.join()
    feats = []
    for syn_nodes in res:
        for node in syn_nodes:
            if 'center' in node.getComment():
                feat = node.data['syn_feat']
                feats.append(feat)
    res = arr(res)
    feats = arr(feats)
    assert len(res) == len(feats), 'feats and nodes have different length!'
    return res, feats


def write_summaries(wd):
    """Write information about contact sites and synapses, axoness and
    connectivity.

    Parameters
    ----------
    wd : str
        Path to working directory of SyConn
    """
    cs_dir = wd + '/contactsites/'
    cs_nodes, cs_feats = collect_contact_sites(cs_dir, only_sj=True)
    write_cs_summary(cs_nodes, cs_feats, cs_dir)
    cs_nodes, cs_feats = collect_contact_sites(cs_dir, only_sj=False)
    features, axoness_info, syn_pred = feature_valid_syns(cs_dir, only_sj=True,
                                                          only_syn=False)
    features, axoness_info, pre_dict, all_post_ids, valid_syn_array, ax_dict =\
        calc_syn_dict(features[syn_pred], axoness_info[syn_pred], get_all=True)
    write_obj2pkl(pre_dict, cs_dir + 'pre_dict.pkl')
    write_obj2pkl(ax_dict, cs_dir + 'axoness_dict.pkl')
    write_cs_summary(cs_nodes, cs_feats, cs_dir, supp='_all', syn_only=False)
    features, axoness_info, syn_pred = feature_valid_syns(cs_dir, only_sj=False,
                                                          only_syn=False,
                                                          all_contacts=True)
    features, axoness_info, pre_dict, all_post_ids, valid_syn_array, ax_dict =\
        calc_syn_dict(features, axoness_info, get_all=True)
    write_obj2pkl(pre_dict, cs_dir + 'pre_dict_all.pkl')
    write_obj2pkl(ax_dict, cs_dir + 'axoness_dict_all.pkl')
    gc.collect()
    write_property_dict(cs_dir)
    conn_dict_wrapper(wd, all=False)
    conn_dict_wrapper(wd, all=True)


def write_cs_summary(cs_nodes, cs_feats, cs_dir, supp='', syn_only=True):
    """Writes contact site summary of all contact sites without sampling

    Parameters
    ----------
    cs_nodes: lsit of SkeletonNodes
        contact site nodes
    cs_feats: np.array
        synapse features of contact sites
    cs_dir: str
        path to contact site directory
    supp : str
        supplement for resulting file
    syn_only: bool
        if only synapses are to be saved.
    """
    dummy_skel = Skeleton()
    dummy_anno = SkeletonAnnotation()
    dummy_anno.setComment('CS Summary')
    if len(cs_nodes) == 0:
        write_obj2pkl({}, cs_dir + 'cs_dict%s.pkl' % supp)
        dummy_skel.add_annotation(dummy_anno)
        fname = cs_dir + 'cs_summary%s.k.zip' % supp
        dummy_skel.to_kzip(fname)
        return
    clf_path = cs_dir + '/../models/rf_synapses/rfc_syn.pkl'
    print "\nUsing %s for synapse prediction." % clf_path
    rfc_syn = joblib.load(clf_path)
    probas = rfc_syn.predict_proba(cs_feats)
    preds = rfc_syn.predict(cs_feats)
    cnt = 0
    cs_dict = {}
    for syn_nodes in cs_nodes:
        for node in syn_nodes:
            if 'center' in node.getComment():
                proba = probas[cnt]
                pred = preds[cnt]
                node.data['syn_proba'] = proba[1]
                node.data['syn_pred'] = pred
                feat = node.data['syn_feat']
                assert np.array_equal(feat, cs_feats[cnt]), "Features unequal!"
                cs_name = node.data['cs_name']
                for dummy_node in syn_nodes:
                    dummy_anno.addNode(dummy_node)
                cnt += 1
                if syn_only and 'sj' not in node.getComment():
                    continue
                if 'sj' in node.getComment():
                    overlap_vx = np.load(cs_dir + '/overlap_vx/' +
                                         cs_name + 'ol_vx.npy')
                else:
                    overlap_vx = np.zeros((0, ))
                cs = {}
                cs['overlap_vx'] = overlap_vx
                cs['syn_pred'] = pred
                cs['syn_proba'] = proba[1]
                cs['center_coord'] = node.getCoordinate()
                cs['syn_feats'] = feat
                cs['cs_dist'] = float(node.data['cs_dist'])
                cs['mean_cs_area'] = float(node.data['mean_cs_area'])
                cs['overlap_area'] = float(node.data['overlap_area'])
                cs['overlap'] = float(node.data['overlap'])
                cs['abs_ol'] = float(node.data['abs_ol'])
                cs['overlap_cs'] = float(node.data['overlap_cs'])
                cs['cs_name'] = node.data['cs_name']
                cs_dict[cs_name] = cs
    dummy_skel.add_annotation(dummy_anno)
    fname = cs_dir + 'cs_summary%s.k.zip' % supp
    dummy_skel.to_kzip(fname)
    write_obj2pkl(cs_dict, cs_dir + 'cs_dict%s.pkl' % supp)
    print "Saved CS summary at %s." % fname


def calc_cs_node(args):
    """Helper function. Calculates three nodes for given contantct site annotation
    containing center of cs and the nearest skeleton nodes.
    The skeleton nodes containg the property axoness, radius and skel_ID.
    The center contains all above information.

    Parameters
    ----------
    args: list
        Filepath to AnnotationObject containing one CS list to store number
        of processed cs and queue

    Returns
    -------
    list of SkeletonNodes
    Three nodes: nodes of adjacent skeletons and center node of CS
    """
    cs_path, q = args
    anno = su.loadj0126NML(cs_path)[0]
    ids, axoness = predict_axoness_from_nodes(anno)
    syn_nodes = []
    center_node = None
    for node in anno.getNodes():
        n_comment = node.getComment()
        if 'center' in n_comment:
            try:
                feat = parse_synfeature_from_node(node)
            except KeyError:
                feat = re.findall('[\d.e+e-]+', node.data['syn_feat'])
                feat = arr([float(f) for f in feat])
            center_node = copy.copy(node)
            center_node.data['syn_feat'] = feat
            cs_ix = re.findall('cs(\d+)', anno.getComment())[0]
            add_comment = ''
            if 'vc' in anno.getComment():
                add_comment += '_vc'
            if 'sj' in anno.getComment():
                add_comment += '_sj'
            center_node.setComment('cs%s%s_skel_%s_%s' % (cs_ix, add_comment,
                                   str(ids[0]), str(ids[1])))
    center_node_comment = center_node.getComment()
    center_node.data['vc_size'] = None
    center_node.data['sj_size'] = None
    for node in anno.getNodes():
        n_comment = node.getComment()
        if 'skelnode_area' in n_comment:
            skel_node = copy.copy(node)
            try:
                skel_id = int(re.findall('(\d+)_skelnode',
                                         skel_node.getComment())[0])
            except IndexError:
                skel_id = int(re.findall('syn(\d+)', skel_node.getComment())[0])
            assert skel_id in ids, 'Wrong skeleton ID during axoness parsing.'
            skel_node.data['skelID'] = skel_id
            skel_node.setComment(center_node_comment+'_skelnode'+str(skel_id))
            syn_nodes.append(skel_node)
        if '_vc-' in n_comment:
            center_node.data['vc_size'] = node.data['radius']
        # if '_sj-' in n_comment:
        #     center_node.data['sj_size'] = node.data['radius']
    center_node.setComment(center_node_comment+'_center')
    syn_nodes.append(center_node)
    assert len(syn_nodes) == 3, 'Number of synapse nodes different than three'
    q.put(1)
    return syn_nodes


def get_spine_summary(syn_nodes, cs_dir):
    """Write out dictionary containing spines (dictionaries) with contact site name
    and spine number as key. The spine dictionary contains three values
    'head_diameter', 'branch_dist' and head_proba

    Parameters
    ----------
    syn_nodes : list of SkeletonNodes
    cs_dir: str
        path to contact site data
    """
    spine_summary = {}
    for nodes in syn_nodes:
        skel1_node, skel2_node, center_node = nodes
        spiness1 = int(skel1_node.data['spiness_pred'])
        spiness2 = int(skel2_node.data['spiness_pred'])
        cs_name = center_node.getComment()
        if spiness1 == 1:
            spine_node = skel1_node
            spine = get_spine_dict(spine_node)
            spine_summary[cs_name] = spine
        if spiness2 == 1:
            spine_node = skel2_node
            spine = get_spine_dict(spine_node)
            spine_summary[cs_name] = spine
    write_obj2pkl(spine_summary, cs_dir+'/spine_summary.pkl')


def get_spine_dict(spine_node, max_dist=100):
    """Gather spine information from spine endnote

    Parameters
    ----------
    spine_node: SkeletonNode
        Node containing spine information
    max_dist : float
        Maximum distance from spine head to endpoint in nm

    Returns
    -------
    dict
        Spine properties 'head_diameter' and 'branch_dist' and 'head_proba'
    """
    spine = {}
    endpoint_dist = float(spine_node.data['end_dist'])
    if endpoint_dist > max_dist:
        return None
    spine['head_diameter'] = float(spine_node.data['head_diameter'])
    spine['branch_dist'] = float(spine_node.data['branch_dist'])
    spine['head_proba'] = float(spine_node.data['spiness_proba1'])
    return spine


def update_axoness_dict(cs_dir, syn_only=True):
    """Update axoness dictionary with current data pulled from 'neurons/'
    folder

    Parameters
    ----------
    cs_dir: str
        path to contact site data
    syn_only : bool
        if only synapses are to be saved.

    Returns
    -------
    dict
        updated axoness dictionary
    """
    print "Writing axoness dictionary with syn_only=%s." % (str(syn_only))
    if syn_only:
        dict_path = cs_dir + 'cs_dict.pkl'
    else:
        dict_path = cs_dir + 'cs_dict_all.pkl'
    cs_dict = load_pkl2obj(dict_path)
    cs_keys = cs_dict.keys()
    params = []
    for k in cs_keys:
        if syn_only and not cs_dict[k]['syn_pred']:
            continue
        # new_names = convert_to_standard_cs_name(k)
        skels = re.findall('[\d]+', k)
        ax_key = skels[2] + '_' + skels[0] + '_' + skels[1]
        new_names = [k]
        param = None
        for var in new_names:
            try:
                center_coord = cs_dict[var]['center_coord']
                param = (center_coord, ax_key)
                break
            except KeyError:
                continue
        if param is None:
            print "Didnt find CS of key %s" % k
        else:
            params.append(param)
    new_vals = start_multiprocess(update_single_cs_axoness, params, debug=False)
    new_ax_dict = {}
    for k, val in new_vals:
        new_ax_dict[k] = val
    return new_ax_dict


def update_single_cs_axoness(params):
    """Get axoness of adjacent tracings of contact site

    Find nearest tracing node to representative coordinate of contact site of
    touching tracings.

    Parameters
    ----------
    params : tuple (numpy.array, str)
        Representative coordinate and contact site key

    Returns
    -------
    str, dict
        contact site key, dictionary of skeleton axoness
    """
    center_coord = params[0]
    key = params[1]
    cs_nb, skel1, skel2 = re.findall('(\d+)_(\d+)_(\d+)', key)[0]
    skel1_path, skel2_path = get_paths_of_skelID([skel1, skel2],
            traced_skel_dir='/lustre/pschuber/mapped_soma_tracings/nml_obj/')
    skel1_anno = load_ordered_mapped_skeleton(skel1_path)[0]
    skel2_anno = load_ordered_mapped_skeleton(skel2_path)[0]
    a_nodes = [node for node in skel1_anno.getNodes()]
    b_nodes = [node for node in skel2_anno.getNodes()]
    a_coords = arr([node.getCoordinate() for node in a_nodes])\
               * skel1_anno.scaling
    b_coords = arr([node.getCoordinate() for node in b_nodes])\
               * skel2_anno.scaling
    a_tree = spatial.cKDTree(a_coords)
    b_tree = spatial.cKDTree(b_coords)
    center_coord *= np.array(skel1_anno.scaling)
    _, a_near = a_tree.query(center_coord, 1)
    _, b_near = b_tree.query(center_coord, 1)
    axoness_entry = {skel1: a_nodes[a_near].data["axoness_pred"],
                     skel2: b_nodes[b_near].data["axoness_pred"]}
    return key, axoness_entry


def convert_to_standard_cs_name(name):
    """Convert to all possible contact site key combinations

    Parameters
    ----------
    name : str
        Contact site key

    Returns
    -------
    list of str
        possible key variations
    """
    cs_nb, skel1, skel2 = re.findall('(\d+)_(\d+)_(\d+)', name)[0]
    new_name1 = 'skel_%s_%s_cs%s_sj' % (skel1, skel2, cs_nb)
    new_name2 = 'skel_%s_%s_cs%s_vc_sj' % (skel1, skel2, cs_nb)
    new_name3 = 'skel_%s_%s_cs%s_sj' % (skel2, skel1, cs_nb)
    new_name4 = 'skel_%s_%s_cs%s_vc_sj' % (skel2, skel1, cs_nb)
    new_name5 = 'skel_%s_%s_cs%s' % (skel2, skel1, cs_nb)
    new_name6 = 'skel_%s_%s_cs%s' % (skel1, skel2, cs_nb)
    return [new_name1, new_name2, new_name3, new_name4, new_name5, new_name6]


def update_property_dict(cs_dir):
    """Update property dictionary

    Parameters
    ----------
    cs_dir : str

    Returns
    -------
    dict
        update property dictionary
    """
    ax_dict = load_pkl2obj(cs_dir + 'axoness_dict.pkl')
    cs_dict = load_pkl2obj(cs_dir + 'cs_dict.pkl')
    ax_keys = ax_dict.keys()
    params = []
    for k in ax_keys:
        new_names = convert_to_standard_cs_name(k)
        param = None
        for var in new_names:
            try:
                center_coord = cs_dict[var]['center_coord']
                param = (center_coord, k, cs_dir)
                break
            except KeyError:
                continue
        if param is None:
            print "Didnt find CS of key %s" % k
        else:
            params.append(param)
    new_vals = start_multiprocess(update_single_cs_properties, params,
                                  debug=False)
    new_property_dict = {}
    for k, val in new_vals:
        new_property_dict[k] = val
    return new_property_dict


def update_single_cs_properties(params):
    """Helper function of update_property_dict
    """
    center_coord = params[0]
    key = params[1]
    cs_dir = params[2]
    cs_nb, skel1, skel2 = re.findall('(\d+)_(\d+)_(\d+)', key)[0]
    skel1_path, skel2_path = get_paths_of_skelID([skel1, skel2],
            traced_skel_dir=cs_dir + '/../neurons/')
    skel1_anno = load_ordered_mapped_skeleton(skel1_path)[0]
    skel2_anno = load_ordered_mapped_skeleton(skel2_path)[0]
    a_nodes = [node for node in skel1_anno.getNodes()]
    b_nodes = [node for node in skel2_anno.getNodes()]
    a_coords = arr([node.getCoordinate() for node in a_nodes])\
               * skel1_anno.scaling
    b_coords = arr([node.getCoordinate() for node in b_nodes])\
               * skel2_anno.scaling
    a_tree = spatial.cKDTree(a_coords)
    b_tree = spatial.cKDTree(b_coords)
    _, a_near = a_tree.query(center_coord, 1)
    _, b_near = b_tree.query(center_coord, 1)
    property_entry = {skel1: {'spiness': int(a_nodes[a_near].data["spiness_pred"]),
                              'axoness': int(a_nodes[a_near].data["axoness_pred"])},
                      skel2: {'spiness': int(b_nodes[b_near].data["spiness_pred"]),
                              'axoness': int(b_nodes[b_near].data["axoness_pred"])}}
    return key, property_entry


def write_property_dict(cs_dir):
    """Write property dictionary to working directory

    Parameters
    ----------
    cs_dir : str
    """
    new_property_dict = update_property_dict(cs_dir)
    write_obj2pkl(new_property_dict, cs_dir + 'property_dict.pkl')


def write_axoness_dicts(cs_dir):
    """Write axoness dictionaries to working directory

    Parameters
    ----------
    cs_dir : str
    """
    new_ax_all = update_axoness_dict(cs_dir, syn_only=False)
    write_obj2pkl(new_ax_all, cs_dir + '/axoness_dict_all.pkl')
    new_ax = update_axoness_dict(cs_dir, syn_only=True)
    write_obj2pkl(new_ax, cs_dir + '/axoness_dict.pkl')


def get_number_cs_details(cs_path):
    """Prints number of synapse candidates and number of contact sites

    Parameters
    ----------
    cs_path : str
    """
    sj_samples = [path for path in
                       get_filepaths_from_dir(cs_path+'cs_sj/', ending='nml')]
    sj_vc_samples = [path for path in
                        get_filepaths_from_dir(cs_path+'cs_vc_sj/', ending='nml')]
    vc_samples = [path for path in
                       get_filepaths_from_dir(cs_path+'cs_vc/', ending='nml')]
    cs_samples = [path for path in
                       get_filepaths_from_dir(cs_path+'cs/', ending='nml')]
    cs_only = len(vc_samples) + len(cs_samples)
    syn_only= len(sj_samples)+len(sj_vc_samples)
    print "Found %d syn-candidates and %d contact sites." % (syn_only,
                                                             cs_only+syn_only)


def conn_dict_wrapper(wd, all=False):
    """Wrapper function to write connectivity dictionary to working directory
    """
    if all:
        suffix = "_all"
    else:
        suffix = ""
    synapse_matrix(wd, suffix=suffix)
    syn_type_majority_vote(wd, suffix=suffix)


def synapse_matrix(wd, type_threshold=0.225, suffix="",
                   exclude_dendrodendro=True):
    """

    Parameters
    ----------
    wd : str
    type_threshold : float
    suffix : str
    exclude_dendrodendro : bool
    """
    save_folder = wd + '/contactsites/'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    with open(save_folder + "/cs_dict%s.pkl" % suffix, "r") as f:
        cs_dict = pickle.load(f)
    with open(save_folder + "/axoness_dict%s.pkl" % suffix, "r") as f:
        ax_dict = pickle.load(f)
    with open(wd + '/neurons/celltype_pred_dict.pkl', "r") as f:
        cell_type_dict = pickle.load(f)

    kd_asym = KnossosDataset()
    kd_asym.initialize_from_knossos_path(wd + '/knossosdatasets/asymmetric/')
    kd_sym = KnossosDataset()
    kd_sym.initialize_from_knossos_path(wd + '/knossosdatasets/symmetric/')

    ids = []
    cs_keys = cs_dict.keys()
    for cs_key in cs_keys:
        skels = np.array(re.findall('[\d]+', cs_key)[:2], dtype=np.int)
        ids.append(skels)

    ids = np.unique(ids)
    id_mapper = {}
    for ii in range(len(ids)):
        id_mapper[ids[ii]] = ii

    axo_axo_dict = {}

    size_dict = {}
    coord_dict = {}
    syn_type_dict = {}
    partner_cell_type_dict = {}
    partner_ax_dict = {}
    cs_key_dict = {}
    phil_dict = {}
    for first_key in ids:
        for second_key in ids:
            size_dict[str(first_key) + "_" + str(second_key)] = []
            coord_dict[str(first_key) + "_" + str(second_key)] = []
            syn_type_dict[str(first_key) + "_" + str(second_key)] = []
            partner_cell_type_dict[str(first_key) + "_" + str(second_key)] = []
            partner_ax_dict[str(first_key) + "_" + str(second_key)] = []
            cs_key_dict[str(first_key) + "_" + str(second_key)] = []
            phil_dict[str(first_key) + "_" + str(second_key)] = {}
            phil_dict[str(first_key) + "_" + str(second_key)]["total_size_area"] = 0
            phil_dict[str(first_key) + "_" + str(second_key)]["sizes_area"] = []
            phil_dict[str(first_key) + "_" + str(second_key)]["syn_types_prob"] = []
            phil_dict[str(first_key) + "_" + str(second_key)]["syn_types_pred"] = []
            phil_dict[str(first_key) + "_" + str(second_key)]["coords"] = []
            phil_dict[str(first_key) + "_" + str(second_key)]['partner_axoness'] = []
            phil_dict[str(first_key) + "_" + str(second_key)]['partner_cell_type'] = []
            phil_dict[str(first_key) + "_" + str(second_key)]['cs_area'] = []
            phil_dict[str(first_key) + "_" + str(second_key)]['total_cs_area'] = 0
            phil_dict[str(first_key) + "_" + str(second_key)]['syn_pred'] = []

    fails = []
    syn_matrix = np.zeros([len(ids), len(ids)], dtype=np.int)
    for ii, cs_key in enumerate(cs_keys):
        print "%d of %d" % (ii+1, len(cs_keys))
        skels = re.findall('[\d]+', cs_key)
        # if cs_dict[cs_key]['syn_pred']:
        ax_key = skels[2] + '_' + skels[0] + '_' + skels[1]
        ax = ax_dict[ax_key]
        this_keys = ax.keys()
        if cs_dict[cs_key]['syn_pred']:
            overlap_vx = cs_dict[cs_key]['overlap_vx'] / np.array([9,9,20])
            sym_values = kd_sym.from_raw_cubes_to_list(overlap_vx)
            asym_values = kd_asym.from_raw_cubes_to_list(overlap_vx)
            this_type = float(np.sum(sym_values))/(np.sum(sym_values)+np.sum(asym_values))
            overlap = cs_dict[cs_key]['overlap_area']/2
        else:
            this_type = 0
            overlap = 0
        # print ax[this_keys[0]], ax[this_keys[1]]
        for ii in range(2):
            if int(ax[this_keys[ii]]) == 1 or not exclude_dendrodendro:
                if int(ax[this_keys[(ii+1)%2]]) == 1:
                    if this_keys[ii] in axo_axo_dict:
                        axo_axo_dict[this_keys[ii]].append(this_keys[(ii+1)%2])
                    else:
                        axo_axo_dict[this_keys[ii]] = [this_keys[(ii+1)%2]]

                syn_matrix[id_mapper[int(this_keys[ii])], id_mapper[int(this_keys[(ii+1)%2])]] += 1

                dict_key = this_keys[ii] + "_" + this_keys[(ii+1)%2]
                cs_key_dict[dict_key].append(cs_key)
                coord_dict[dict_key].append(cs_dict[cs_key]['center_coord'])
                syn_type_dict[dict_key].append(this_type)
                size_dict[dict_key].append(overlap)
                partner_ax_dict[dict_key].append(int(ax[this_keys[(ii+1)%2]]))
                partner_cell_type_dict[dict_key].append(cell_type_dict[int(this_keys[(ii+1)%2])])

                phil_dict[dict_key]["total_size_area"] += overlap
                phil_dict[dict_key]["sizes_area"].append(overlap)
                phil_dict[dict_key]["cs_area"].append(cs_dict[cs_key]['mean_cs_area']/2.e6)
                phil_dict[dict_key]["total_cs_area"] += cs_dict[cs_key]['mean_cs_area']/2.e6
                phil_dict[dict_key]["syn_types_prob"].append(this_type)
                phil_dict[dict_key]["syn_types_pred"].append(int(this_type > type_threshold))
                phil_dict[dict_key]["coords"].append(cs_dict[cs_key]['center_coord'])
                phil_dict[dict_key]['partner_axoness'].append(int(ax[this_keys[(ii+1)%2]]))
                phil_dict[dict_key]['partner_cell_type'].append(cell_type_dict[int(this_keys[(ii+1)%2])])
                phil_dict[dict_key]['syn_pred'].append(cs_dict[cs_key]['syn_pred'])

    if not exclude_dendrodendro:
        suffix = "_no_exclusion" + suffix

    np.save(save_folder+"/syn_matrix%s.npy" % suffix, syn_matrix)
    with open(save_folder + "/id_mapper%s.pkl" % suffix, "w") as f:
        pickle.dump(id_mapper, f)
    with open(save_folder + "/size_dict%s.pkl" % suffix, "w") as f:
        pickle.dump(size_dict, f)
    with open(save_folder + "/syn_type_dict%s.pkl" % suffix, "w") as f:
        pickle.dump(syn_type_dict, f)
    with open(save_folder + "/coord_dict%s.pkl" % suffix, "w") as f:
        pickle.dump(coord_dict, f)
    with open(save_folder + "/cs_key_dict%s.pkl" % suffix, "w") as f:
        pickle.dump(cs_key_dict, f)
    with open(save_folder + "/partner_ax_dict%s.pkl" % suffix, "w") as f:
        pickle.dump(partner_ax_dict, f)
    with open(save_folder + "/partner_cell_type_dict%s.pkl" % suffix, "w") as f:
        pickle.dump(partner_cell_type_dict, f)
    with open(save_folder + "/connectivity_dict%s.pkl" % suffix, "w") as f:
        pickle.dump(phil_dict, f)


def syn_type_majority_vote(wd, suffix=""):
    """Majority vote of synapse type

    Parameters
    ----------

    Returns
    -------
    int, int
    """
    save_folder = wd + '/contactsites/'
    with open(save_folder + "/connectivity_dict%s.pkl" % suffix, "r") as f:
        phil_dict = pickle.load(f)
    with open(save_folder + "/cs_dict%s.pkl" % suffix, "r") as f:
        cs_dict = pickle.load(f)

    maj_type_dict = {}

    ids = []
    cs_keys = cs_dict.keys()
    for cs_key in cs_keys:
        skels = np.array(re.findall('[\d]+', cs_key)[:2], dtype=np.int)
        ids.append(skels)

    ids = np.unique(ids)
    id_mapper = {}
    for ii in range(len(ids)):
        id_mapper[ids[ii]] = ii

    types = []
    sizes = []
    for first_key in ids:
        types.append([])
        sizes.append([])
        for second_key in ids:
            key = str(first_key) + "_" + str(second_key)
            types[-1].append(phil_dict[key]["syn_types_pred"])
            sizes[-1].append(phil_dict[key]["sizes_area"])
            phil_dict[key]["syn_types_pred_maj"] = None
            maj_type_dict[key] = []

    maj_types = []
    for ii in range(len(ids)):
        this_types = []
        this_sizes = []
        for jj in range(len(ids)):
            this_types += types[ii][jj]
            this_sizes += sizes[ii][jj]
        if len(this_types) > 0:
            # maj_types.append(np.argmax(np.bincount(this_types)))
            sum0 = np.sum(np.array(this_sizes)[np.array(this_types)==0])
            sum1 = np.sum(np.array(this_sizes)[np.array(this_types)==1])
            print sum0, sum1
            maj_types.append(int(sum0 < sum1))
        else:
            maj_types.append(-1)

    change_count = 0
    syn_count = 0
    for ii, first_key in enumerate(ids):
        for second_key in ids:
            key = str(first_key) + "_" + str(second_key)
            if maj_types[ii] != -1:
                phil_dict[key]["syn_types_pred_maj"] = [maj_types[ii]]*len(phil_dict[key]["syn_types_pred"])
                maj_type_dict[key] = [maj_types[ii]]*len(phil_dict[key]["syn_types_pred"])
                change_count += np.abs(np.sum(phil_dict[key]["syn_types_pred"])-np.sum(maj_type_dict[key]))
                syn_count += len(maj_type_dict[key])

    with open(save_folder + "/maj_syn_types%s.pkl" % suffix, "w") as f:
        pickle.dump(maj_type_dict, f)
    with open(save_folder + "/connectivity_dict%s.pkl" % suffix, "w") as f:
        pickle.dump(phil_dict, f)

    print change_count, syn_count