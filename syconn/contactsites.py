# -*- coding: utf-8 -*-
import copy
import gc
import os
import re
import time
from multiprocessing import Pool, Manager, cpu_count
from sys import stdout

import numpy as np
from numpy import array as arr
from scipy import spatial
from sklearn.externals import joblib

from syconn.processing.axoness import predict_axoness_from_nodes
from syconn.processing.learning_rfc import start_multiprocess
from syconn.processing.mapper import feature_valid_syns, calc_syn_dict
from syconn.processing.synapticity import parse_synfeature_from_node
from syconn.utils import annotationUtils as au
from syconn.utils.datahandler import get_filepaths_from_dir, \
    load_ordered_mapped_skeleton, get_paths_of_skelID
from syconn.utils.datahandler import write_obj2pkl, load_pkl2obj
from syconn.utils.newskeleton import NewSkeleton, SkeletonAnnotation

__author__ = 'pschuber'


def collect_contact_sites(cs_dir, only_az=False):
    """
    Collects all information about n contact sites.
    :param cs_dir:
    :param only_az:
    :return: list of CS nodes (3 per cs), arr of syn feature (n x #feat)
    """
    if not only_az:
        search_folder = ['cs_az/', 'cs_p4_az/', 'cs/', 'cs_p4/']
    else:
        search_folder = ['cs_az/', 'cs_p4_az/']
    sample_list_len = []
    cs_fpaths = []
    for k, ending in enumerate(search_folder):
        curr_dir = cs_dir+ending
        curr_fpaths = get_filepaths_from_dir(curr_dir, ending='nml')
        cs_fpaths += curr_fpaths
        sample_list_len.append(len(curr_fpaths))
    print "Collecting contact sites (%d CS). Only az is %s." % (len(cs_fpaths),
                                                                str(only_az))
    nb_cpus = cpu_count() - 2
    pool = Pool(processes=nb_cpus)
    m = Manager()
    q = m.Queue()
    params = [(sample, q) for sample in cs_fpaths]
    result = pool.map_async(calc_cs_node, params)
    #result = map(calc_cs_node, params)
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
    svens_res = []
    svens_res2 = []
    for syn_nodes in res:
        for node in syn_nodes:
            if 'center' in node.getComment():
                feat = node.data['syn_feat']
                feats.append(feat)
                svens_res.append(feat[2])
                svens_res2.append(node.getCoordinate())
    #np.save('/lustre/pschuber/svens_az/absol.npy', arr(svens_res, dtype=np.float))
    #np.save('/lustre/pschuber/svens_az/coord.npy', arr(svens_res2, dtype=np.int))
    res = arr(res)
    feats = arr(feats)
    assert len(res) == len(feats), 'feats and nodes have different length!'
    return res, feats


def write_summaries(cs_dir, eval_sampling=False, write_syn_sum=False):
    """
    Write information about contact sites and synapses.
    :param cs_dir: String Path to contact_sites_new folder of mapped skeletons
    :return:
    """
    cs_nodes, cs_feats = collect_contact_sites(cs_dir, only_az=True)
    if write_syn_sum:
        write_syn_summary(cs_nodes, cs_dir, eval_sampling)
        return
    write_cs_summary(cs_nodes, cs_feats, cs_dir)
    cs_nodes, cs_feats = collect_contact_sites(cs_dir, only_az=False)
    features, axoness_info, syn_pred = feature_valid_syns(cs_dir, only_az=True,
                                                          only_syn=False)
    features, axoness_info, pre_dict, all_post_ids, valid_syn_array, ax_dict =\
        calc_syn_dict(features[syn_pred], axoness_info[syn_pred], get_all=True)
    write_obj2pkl(pre_dict, cs_dir + 'pre_dict.pkl')
    write_obj2pkl(ax_dict, cs_dir + 'axoness_dict.pkl')
    write_cs_summary(cs_nodes, cs_feats, cs_dir, supp='_all', syn_only=False)
    features, axoness_info, syn_pred = feature_valid_syns(cs_dir, only_az=False,
                                                          only_syn=False, all_contacts=True)
    features, axoness_info, pre_dict, all_post_ids, valid_syn_array, ax_dict =\
        calc_syn_dict(features, axoness_info, get_all=True)
    write_obj2pkl(pre_dict, cs_dir + 'pre_dict_all.pkl')
    write_obj2pkl(ax_dict, cs_dir + 'axoness_dict_all.pkl')
    gc.collect()
    get_spine_summary(cs_nodes, cs_feats, cs_dir)


def write_cs_summary(cs_nodes, cs_feats, cs_dir, clf_path='/lustre/pschuber/'
                    'gt_syn_mapping/updated_gt3/rf/rf_syn.pkl', supp='',
                     syn_only=True):
    """
    Writs contact site summary of all contact sites without sampling.
    :param cs_dir:
    :return:
    """
    print "\nUsing %s for synapse prediction." % clf_path
    rfc_syn = joblib.load(clf_path)
    dummy_skel = NewSkeleton()
    dummy_anno = SkeletonAnnotation()
    dummy_anno.setComment('CS Summary')
    probas = rfc_syn.predict_proba(cs_feats)
    preds = rfc_syn.predict(cs_feats)
    cnt = 0
    cs_dict = {}
    # only write certain number of syns to cs_summary, s.t. isotropic distribution
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
                # add to cs_dict
                if syn_only and not 'az' in node.getComment():
                    continue
                try:
                    try:
                        overlap_vx = np.load(cs_dir+'/overlap_vx/'+cs_name+'ol_vx.npy')
                    except:
                         overlap_vx = np.load(cs_dir+'/overlap_vx/'+cs_name+'ol_vx.pkl.npy')
                except:
                    overlap_vx = []
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


def write_syn_summary(cs_nodes, cs_dir, eval_sampling, pred_syn=True,
        clf_path='/lustre/pschuber/gt_syn_mapping/updated_gt3/rf/rf_syn.pkl'):
    """
    Writes all synapses (i.e. using synapse prediction) to summary file if
     pred_syn=True. Otherwise writes all CS with synapse prediction!
     If eval_sampling synapses/cs are sampled isotropically.
    :param cs_dir:
    :return:
    """
    axoness_dict = load_pkl2obj(cs_dir + 'axoness_dict.pkl')
    print "Using %s for synapse prediction." % clf_path
    rfc_syn = joblib.load(clf_path)
    dummy_skel = NewSkeleton()
    feats = []
    # calculate volume binning
    box_extent, tree1, tree2 = get_boxes_extent()
    box_cnt = np.zeros((len(box_extent), ))
    #count number of syns per binning
    nb_cc_cs = 0
    for syn_nodes in cs_nodes:
        for node in syn_nodes:
            if 'center' in node.getComment():
                bool_arr = coord_in_boxes(node.getCoordinate(), box_extent,
                                          tree1, tree2)
                if np.sum(bool_arr) == 0:
                    nb_cc_cs += 1
                box_cnt[bool_arr] += 1
                feat = node.data['syn_feat']
                feats.append(feat)
    print "Box counts: %d \t %0.4f \t %0.4f" % (len(box_cnt), np.mean(box_cnt),
                                                np.std(box_cnt))
    probas = rfc_syn.predict_proba(feats)
    preds = rfc_syn.predict(feats)
    print "Found %d CS in total (out: %d, in: %d) and %d synapses." % \
          (len(cs_nodes), np.sum(box_cnt), nb_cc_cs, np.sum(preds))

    # only write certain number of syns to cs_summary, s.t. isotropic distribution
    if eval_sampling:
        max_cnt = 2
        print np.min((np.min(box_cnt), 10))
    else:
        max_cnt = np.inf
    box_cnt = np.zeros((len(box_extent), ))
    cnt = 0
    for syn_nodes in cs_nodes:
        dummy_anno = SkeletonAnnotation()
        for node in syn_nodes:
            if 'center' in node.getComment():
                cs_comment = node.getComment()
                skel_ids = re.findall('skel_(\d+_\d+)', cs_comment)[0]
                cs_nb = re.findall('cs(\d+_)', cs_comment)[0]
                anno_comment = cs_nb + skel_ids
                dummy_anno.setComment(anno_comment)
                proba = probas[cnt]
                pred = preds[cnt]
                cnt += 1
                if pred_syn:
                    if pred == 0:
                        continue
                bool_arr = coord_in_boxes(node.getCoordinate(), box_extent, tree1,
                                          tree2)
                if box_cnt[bool_arr] >= max_cnt:
                    continue
                if np.sum(bool_arr) == 0:
                    continue
                box_cnt[bool_arr] += 1
                node.data['syn_proba'] = proba
                node.data['syn_pred'] = pred
                for dummy_node in syn_nodes:
                    comment = dummy_node.getComment()
                    if 'skelnode' in comment:
                        id = re.findall('skelnode(\d+)', comment)[0]
                        axoness = axoness_dict[anno_comment][id]
                        function_comment = 'post'
                        if int(axoness) == 1:
                            function_comment = 'pre'
                        dummy_node.setComment(function_comment + '_' + id)
                    else:
                        dummy_node.setComment('cleft')
                    dummy_anno.addNode(dummy_node)
                dummy_skel.add_annotation(dummy_anno)
    skel_fp = get_filepaths_from_dir('/lustre/pschuber/m_consensi_rr/nml_obj/')
    print "Adding %d cell tracings." % len(skel_fp)
    for fp in skel_fp:
        annotation = load_ordered_mapped_skeleton(fp)[0]
        id = re.findall('iter_0_(\d+)-', fp)[0]
        annotation.setComment(id)
        dummy_skel.add_annotation(annotation)
    print "Looked at %d CS." % cnt
    fname = cs_dir + 'syn_summary.k.zip'
    if eval_sampling:
        fname = fname[:-6] + '_eval_test.k.zip'
    dummy_skel.to_kzip(fname)
    print "Box counts: %d \t %0.4f \t %0.4f" % (len(box_cnt), np.mean(box_cnt),
                                                np.std(box_cnt))
    print "Saved sampled syn summary at %s." % fname


def coord_in_boxes(coord, boxes, tree1, tree2):
    """
    Check if coordinate is in one of the boxes using tree1
    :return: bool Inside any box or not
    """
    bool_arr = np.zeros((len(boxes), ), dtype=np.bool)
    res1 = tree1.query(coord, k=40)
    # res2 = tree2.query(coord, k=30)
    # inter = list(set(res1[1]).intersection(res2[1]))
    for k in res1[1]:
        box_extent = boxes[k]
        inside = np.all(coord >= box_extent[0]) and np.all(coord <= box_extent[1])
        if inside:
            bool_arr[k] = True
            break
    return bool_arr


def get_boxes_extent():
    """
    Calculate volume binning of j0126.
    :return: arr Box extent of every 3D bin except center cube,
    tree1 cKDTree containing all offset coordinates, tree2 containing coordinates
    of offset+extent.
    """
    cc_coord = np.array([4885, 4757, 2630])
    cc_size = np.array([1110, 1110, 480])
    box_size = cc_size
    ds_extent = np.array([10880, 10624,  5760])
    box_origin = np.array(cc_coord)
    while np.any(box_origin>0):
        box_origin -= box_size
    print box_origin
    nb_boxes = np.ceil(1.0 * (ds_extent - box_origin) / box_size ).astype(np.int)
    print nb_boxes
    boxes_extent = []
    cnt = 0
    for i in range(nb_boxes[0]):
        for j in range( nb_boxes[1]):
            for k in range(nb_boxes[2]):
                cnt_arr = np.array([i,j,k])
                offset = cnt_arr*box_size + box_origin
                box_extent = [offset, offset + box_size]
                inside_cc = np.all(box_extent[0] > cc_coord) and \
                np.all(box_extent[0] <= cc_size+cc_coord)
                if inside_cc:
                    cnt += 1
                    continue
                boxes_extent.append(box_extent)
    print "#inside cc:", cnt
    print "#boxes", len(boxes_extent)
    boxes_extent = np.array(boxes_extent)
    tree1 = spatial.KDTree(boxes_extent[:, 1])
    tree2 = spatial.KDTree(boxes_extent[:, 0])
    return boxes_extent, tree1, tree2


def calc_cs_node(args):
    """
    Helper function. Calculates three nodes for given contantct site annotation
    containing center of cs and the nearest skeleton nodes.
    The skeleton nodes containg the property axoness, radius and skel_ID.
    The center contains all above information.
    :param args: str Filepath to AnnotationObject containing one CS
                 list to store number of processed cs.
    :return: Three nodes: nodes of adjacent skeletons and center node of CS
    """
    cs_path, q = args
    anno = au.loadj0126NML(cs_path)[0]
    ids, axoness = predict_axoness_from_nodes(anno)
    syn_nodes = []
    for node in anno.getNodes():
        n_comment = node.getComment()
        if 'center' in n_comment:
            try:
                feat = parse_synfeature_from_node(node)
            except KeyError:
                #feat = parse_synfeature_from_txt(n_comment)
                feat = re.findall('[\d.e+e-]+', node.data['syn_feat'])
                feat = arr([float(f) for f in feat])
            center_node = copy.copy(node)
            center_node.data['syn_feat'] = feat
            cs_ix = re.findall('cs(\d+)', anno.getComment())[0]
            add_comment = ''
            if 'p4' in anno.getComment():
                add_comment += '_p4'
            if 'az' in anno.getComment():
                add_comment += '_az'
            center_node.setComment('cs%s%s_skel_%s_%s' % (cs_ix, add_comment,
                                                    str(ids[0]), str(ids[1])))
    center_node_comment = center_node.getComment()
    center_node.data['p4_size'] = None
    center_node.data['az_size'] = None
    for node in anno.getNodes():
        n_comment = node.getComment()
        if 'skelnode_area' in n_comment:
            skel_node = copy.copy(node)
            try:
                skel_id = int(re.findall('(\d+)_skelnode', skel_node.getComment())[0])
            except IndexError:
                skel_id = int(re.findall('syn(\d+)', skel_node.getComment())[0])
            assert skel_id in ids, 'Wrong skeleton ID during axoness parsing.'
            skel_node.data['skelID'] = skel_id
            skel_node.setComment(center_node_comment+'_skelnode'+str(skel_id))
            syn_nodes.append(skel_node)
        if '_p4-' in n_comment:
            center_node.data['p4_size'] = node.data['radius']
        # if '_az-' in n_comment:
        #     center_node.data['az_size'] = node.data['radius']
    center_node.setComment(center_node_comment+'_center')
    syn_nodes.append(center_node)
    assert len(syn_nodes) == 3, 'Number of synapse nodes different than three'
    q.put(1)
    return syn_nodes


def get_spine_summary(syn_nodes, cs_feats, cs_dir):
    """
    Write out dictionary containing spines (dictionaries) with contact site name
    and spine number as key. The spine dictionary contains three values
    'head_diameter', 'branch_dist' and head_proba
    :param cs_dir: path to contact site data
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
    """
    Gather spine information from spine endnote.
    :param spine_node: SkeletonNode containing spine information
    :param max_dist: float Maximum distance from spine head to endpoint in nm
    :return: dict with values 'head_diameter' and 'branch_dist' and 'head_proba'
    """
    spine = {}
    endpoint_dist = float(spine_node.data['end_dist'])
    if endpoint_dist > max_dist:
        return None
    spine['head_diameter'] = float(spine_node.data['head_diameter'])
    spine['branch_dist'] = float(spine_node.data['branch_dist'])
    spine['head_proba'] = float(spine_node.data['spiness_proba1'])
    return spine


def get_cs_coords(path='/lustre/pschuber/m_consensi_rr/nml_obj/contact_sites_new2/',
                  dest_path='/lustre/pschuber/m_consensi_rr/cs_coords1.npy'):
    cs_nodes, cs_feats = collect_contact_sites(path)
    coords = np.zeros((len(cs_nodes), 3))
    for ii, nodes in enumerate(cs_nodes):
        coords[ii] = nodes[-1].getCoordinate()
    np.save(dest_path, coords)
    print "Saved coords at %s." % dest_path


def check_cs_determination(recompute=False):
    """
    Determine determination of contact site calculation by comparing coordinates.
    :return:
    """
    if recompute:
        get_cs_coords('/lustre/pschuber/m_consensi_rr/nml_obj/contact_sites_new2/',
                  dest_path='/lustre/pschuber/m_consensi_rr/cs_coords1.npy')
        get_cs_coords('/lustre/pschuber/m_consensi_rr/nml_obj/contact_sites_new3/',
                  dest_path='/lustre/pschuber/m_consensi_rr/cs_coords2.npy')
    coords1 = np.load('/lustre/pschuber/m_consensi_rr/cs_coords1.npy')
    coords2 = np.load('/lustre/pschuber/m_consensi_rr/cs_coords2.npy')
    coord1_tree = spatial.cKDTree(coords1)
    dists, ixs = coord1_tree.query(coords2)
    max_dist = 1
    print "Proportion of same coords", np.sum(dists <= max_dist)/float(len(dists))


def write_cs_eval(cs_path='/lustre/pschuber/consensi_fuer_joergen/'
                          'nml_obj/contact_sites_new/',
                  sample_portion=[0.1, 0.2, 0.2, 0.2]):
    """
    Writes out files for evaluation. Contact sites are split in four categories
    "cs", "cs_p4", "cs_az" and "cs_p4_az".
    :param cs_path: str Path to pairwise contact sites.
    :param list of int Portion of files to be sampled from each category
    """
    sample_portion = [.5, 0.5, 0.5, 0.5]
    if not os.path.exists(cs_path+'eval/'):
        os.makedirs(cs_path+'eval/')
    for k, ending in enumerate(['cs/', 'cs_p4/', 'cs_az/', 'cs_p4_az/']):
        curr_dir = cs_path+ending
        file_paths = get_filepaths_from_dir(curr_dir, ending='nml')
        portion = sample_portion[k]
        dummy_skel = NewSkeleton()
        nb_elements = int(len(file_paths)*portion)
        rand_ixs = np.random.choice(len(file_paths), size=nb_elements,
                                    replace=False)
        for fpath in arr(file_paths)[rand_ixs]:
            curr_anno = au.loadj0126NML(fpath)[0]
            dummy_skel.add_annotation(curr_anno)
        print "Writing", cs_path+'eval/'+ending[:-1]+'_sampling_helmstaedter.nml'
        dummy_skel.toNml(cs_path+'eval/'+ending[:-1]+'_sampling_helmstaedter.nml')


def update_axoness_dict(cs_dir, syn_only=True):
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
    cs_nb, skel1, skel2 = re.findall('(\d+)_(\d+)_(\d+)', name)[0]
    new_name1 = 'skel_%s_%s_cs%s_az' % (skel1, skel2, cs_nb)
    new_name2 = 'skel_%s_%s_cs%s_p4_az' % (skel1, skel2, cs_nb)
    new_name3 = 'skel_%s_%s_cs%s_az' % (skel2, skel1, cs_nb)
    new_name4 = 'skel_%s_%s_cs%s_p4_az' % (skel2, skel1, cs_nb)
    new_name5 = 'skel_%s_%s_cs%s' % (skel2, skel1, cs_nb)
    new_name6 = 'skel_%s_%s_cs%s' % (skel1, skel2, cs_nb)
    return [new_name1, new_name2, new_name3, new_name4, new_name5, new_name6]


def update_property_dict(cs_dir):
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
                param = (center_coord, k)
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
    """
    Helper function of update_property_dict
    :param params: center coord and dictionary key (cs/synapse) of axoness dict
    :return: property dict entry with spiness and axoness of CS/synapse
    """
    center_coord = params[0]
    key = params[1]
    cs_nb, skel1, skel2 = re.findall('(\d+)_(\d+)_(\d+)', key)[0]
    skel1_path, skel2_path = get_paths_of_skelID([skel1, skel2],
            traced_skel_dir='/lustre/pschuber/st250_pt3_minvotes18/nml_obj/')
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
    new_property_dict = update_property_dict(cs_dir)
    write_obj2pkl(new_property_dict, cs_dir + 'property_dict.pkl')


def write_joergen_plot(cs_dir, gt_path='/lustre/pschuber/gt_cell_types/',
                       recompute=False):
    # find exitatory axon - medium spiny synapses
    if not os.path.isfile(gt_path + '/wiring/joergen_plot.npy') or recompute:
        prop_dict = load_pkl2obj(cs_dir + 'property_dict.pkl')
        cell_type_dict = load_pkl2obj(gt_path + 'cell_pred_dict.pkl')
        cs_dict = load_pkl2obj(cs_dir + 'cs_dict.pkl')
        #cell_type_dict = load_pkl2obj('/lustre/pschuber/gt_cell_types/'
        #                                       'consensi_celltype_labels2.pkl')
        syn_props = load_pkl2obj('/lustre/sdorkenw/synapse_matrices/phil_dict_all.pkl')
        ex_axons = []
        med_spiny_neurons = []
        for pair_name, pair in syn_props.iteritems():
            if pair['total_size_area'] != 0:
                skel_id1, skel_id2 = re.findall('(\d+)_(\d+)', pair_name)[0]
                skel_id1 = int(skel_id1)
                skel_id2 = int(skel_id2)
                try:
                    if (cell_type_dict[skel_id1] == 0) and (cell_type_dict[skel_id2] == 1):
                        ex_axons.append(skel_id1)
                        med_spiny_neurons.append(skel_id2)
                except KeyError:
                    print "Skipping %d and %d" % (skel_id1, skel_id2)
        # get cell properties
        coords = {}
        shaft_syn_frac = {}
        for med_spiny in med_spiny_neurons:
            syn_cnt = 0
            shaft_cnt = 0
            for k, val in prop_dict.iteritems():
                if str(med_spiny) in val.keys():
                    syn_cnt += 1
                    if int(val[str(med_spiny)]['spiness']) == 0:
                        shaft_cnt += 1
                        new_names = convert_to_standard_cs_name(k)
                        for var in new_names:
                            try:
                                center_coord = cs_dict[var]['center_coord']
                                coords[k] = center_coord
                                break
                            except KeyError:
                                continue
            shaft_syn_frac[med_spiny] = shaft_cnt / float(syn_cnt)
        branch_density = {}
        celltype_feats = load_pkl2obj(gt_path + '/wiring/skel_feat_dict.pkl')
        for ex_ax in ex_axons:
            branch_density[ex_ax] = celltype_feats[ex_ax][0]
        # draw plot
        x = []
        y = []
        for i in range(len(ex_axons)):
            ex_ax = ex_axons[i]
            med_sp = med_spiny_neurons[i]
            y.append(shaft_syn_frac[med_sp])
            x.append(branch_density[ex_ax])
        plot_arr = np.array([x, y])
        pair_arr = np.array([ex_axons, med_spiny_neurons])
        np.save(gt_path + '/wiring/joergen_plot.npy', plot_arr)
        np.save(gt_path + '/wiring/joergen_plot_pair_ids.npy', pair_arr)
        write_obj2pkl(coords, gt_path + '/wiring/syn_coords.npy')
    else:
        plot_arr = np.load(gt_path + '/wiring/joergen_plot.npy')
        pair_arr = np.load(gt_path + '/wiring/joergen_plot_pair_ids.npy')
        coords = load_pkl2obj(gt_path + '/wiring/syn_coords.npy')
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(plot_arr[0], plot_arr[1], 'o')
    print "Found %d cell pairs of medium spiny neurons and excitatory axons." %\
        len(plot_arr[0])
    plt.xlabel(u'Branch Point Density [µm$^3$]')
    plt.ylabel('Fraction of Shaft Synapses')
    plt.title('Synapses: Medium Spiny Neuron -- Excitatory Axons')
    plt.ylim(-0.1, 1.1)
    plt.show(block=False)
    raise()


def write_axoness_dicts(cs_dir):
    new_ax_all = update_axoness_dict(cs_dir, syn_only=False)
    write_obj2pkl(new_ax_all, cs_dir + '/axoness_dict_all.pkl')
    # new_ax = update_axoness_dict(cs_dir, syn_only=True)
    # write_obj2pkl(new_ax_all, cs_dir + '/axoness_dict.pkl')


def get_number_cs_details(cs_path):
    az_samples = [path for path in
                       get_filepaths_from_dir(cs_path+'cs_az/', ending='nml')]
    az_p4_samples = [path for path in
                        get_filepaths_from_dir(cs_path+'cs_p4_az/', ending='nml')]
    p4_samples = [path for path in
                       get_filepaths_from_dir(cs_path+'cs_p4/', ending='nml')]
    cs_samples = [path for path in
                       get_filepaths_from_dir(cs_path+'cs/', ending='nml')]
    cs_only = len(p4_samples) + len(cs_samples)
    syn_only= len(az_samples)+len(az_p4_samples)
    print "Found %d syn-candidates and %d contact sites." % (syn_only, cs_only+syn_only)
