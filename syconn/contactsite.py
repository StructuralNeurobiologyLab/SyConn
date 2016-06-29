# -*- coding: utf-8 -*-
try:
    from NewSkeleton import annotationUtils as au
except:
    import annotationUtils as au
try:
    from NewSkeleton.NewSkeletonUtils import annotation_from_nodes
except:
    from NewSkeletonUtils import annotation_from_nodes
import re
from numpy import array as arr
import numpy as np
from utils.datahandler import load_anno_list, get_filepaths_from_dir, \
    load_ordered_mapped_skeleton, get_paths_of_skelID, write_obj2pkl,\
     load_pkl2obj
from scipy import spatial
from NewSkeleton import NewSkeleton, SkeletonAnnotation
from multiprocessing import Pool, Manager, cpu_count
from sys import stdout
import time
from sklearn.externals import joblib
import copy
from processing.axoness import predict_axoness_from_nodes
from processing.synapticity import parse_synfeature_from_node
import os
from processing.learning_rfc import start_multiprocess
import gc
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
    # cs_nodes, cs_feats = collect_contact_sites(cs_dir, only_az=True)
    # if write_syn_sum:
    #     write_syn_summary(cs_nodes, cs_dir, eval_sampling)
    #     return
    # write_cs_summary(cs_nodes, cs_feats, cs_dir)
    cs_nodes, cs_feats = collect_contact_sites(cs_dir, only_az=False)
    # features, axoness_info, syn_pred = feature_valid_syns(cs_dir, only_az=True,
    #                                                       only_syn=False)
    # features, axoness_info, pre_dict, all_post_ids, valid_syn_array, ax_dict =\
    #     calc_syn_dict(features[syn_pred], axoness_info[syn_pred], get_all=True)
    # write_obj2pkl(pre_dict, cs_dir + 'pre_dict.pkl')
    # write_obj2pkl(ax_dict, cs_dir + 'axoness_dict.pkl')
    # write_cs_summary(cs_nodes, cs_feats, cs_dir, supp='_all', syn_only=False)
    # features, axoness_info, syn_pred = feature_valid_syns(cs_dir, only_az=False,
    #                                                       only_syn=False, all_contacts=True)
    # features, axoness_info, pre_dict, all_post_ids, valid_syn_array, ax_dict =\
    #     calc_syn_dict(features, axoness_info, get_all=True)
    # write_obj2pkl(pre_dict, cs_dir + 'pre_dict_all.pkl')
    # write_obj2pkl(ax_dict, cs_dir + 'axoness_dict_all.pkl')
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


def convert_to_standard_cs_name(name):
    cs_nb, skel1, skel2 = re.findall('(\d+)_(\d+)_(\d+)', name)[0]
    new_name1 = 'skel_%s_%s_cs%s_az' % (skel1, skel2, cs_nb)
    new_name2 = 'skel_%s_%s_cs%s_p4_az' % (skel1, skel2, cs_nb)
    new_name3 = 'skel_%s_%s_cs%s_az' % (skel2, skel1, cs_nb)
    new_name4 = 'skel_%s_%s_cs%s_p4_az' % (skel2, skel1, cs_nb)
    new_name5 = 'skel_%s_%s_cs%s' % (skel2, skel1, cs_nb)
    new_name6 = 'skel_%s_%s_cs%s' % (skel1, skel2, cs_nb)
    return [new_name1, new_name2, new_name3, new_name4, new_name5, new_name6]


def feature_valid_syns(cs_dir, only_az=True, only_syn=True, all_contacts=False):
    """
    Returns the features of valid synapses predicted by synapse rfc.
    :param cs_dir: Path to computed contact sites.
    :param only_az: Return feature of all contact sites with mapped az.
    :param only_syn: Returns feature only if synapse was predicted
    :param all_contacts: Use all contact sites for feature extraction
    :return: array of features, array of contact site IDS, boolean array of syn-
    apse prediction
    """
    cs_fpaths = []
    if only_az:
        search_folder = ['cs_az/', 'cs_p4_az/']
    elif all_contacts:
        search_folder = ['cs_az/', 'cs_p4_az/', 'cs/', 'cs_p4/']
    else:
        search_folder = ['cs/', 'cs_p4/']
    sample_list_len = []
    for k, ending in enumerate(search_folder):
        curr_dir = cs_dir+ending
        curr_fpaths = get_filepaths_from_dir(curr_dir, ending='nml')
        cs_fpaths += curr_fpaths
        sample_list_len.append(len(curr_fpaths))
    print "Collecting results of synapse mapping. (%d CS)" % len(cs_fpaths)
    nb_cpus = cpu_count()
    pool = Pool(processes=nb_cpus)
    m = Manager()
    q = m.Queue()
    params = [(sample, q) for sample in cs_fpaths]
    result = pool.map_async(readout_cs_info, params)
    #result = map(readout_cs_info, params)
    # monitor loop
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
    res = arr(res)
    non_instances = arr([isinstance(el, np.ndarray) for el in res[:,0]])
    cs_infos = res[non_instances]
    features = arr([el.astype(np.float) for el in cs_infos[:,0]], dtype=np.float)
    if not only_az or not only_syn or all_contacts:
        syn_pred = np.ones((len(features), ))
    else:
        rfc_syn = joblib.load('/lustre/pschuber/gt_syn_mapping/rfc/rfc_syn.pkl')
        syn_pred = rfc_syn.predict(features)
    axoness_info = cs_infos[:, 1]#[syn_pred.astype(np.bool)]
    error_cnt = np.sum(~non_instances)
    #features = features[syn_pred.astype(np.bool)]
    print "Found %d synapses with axoness information. Gathering all" \
          " contact sites with valid pre/pos information." % len(axoness_info)
    syn_fpaths = arr(cs_fpaths)[non_instances][syn_pred.astype(np.bool)]
    false_cnt = np.sum(~syn_pred.astype(np.bool))
    true_cnt = np.sum(syn_pred)
    print "\nTrue synapses:", true_cnt / float(true_cnt+false_cnt)
    print "False synapses:", false_cnt / float(true_cnt+false_cnt)
    print "error count:", error_cnt
    return features, axoness_info, syn_pred.astype(np.bool)


def readout_cs_info(args):
    """
    Helper function of feature_valid_syns
    :param args: tuple of path to file and queue
    :return: array of synapse features, str contact site ID
    """
    cspath, q = args
    if q is not None:
        q.put(1)
    cs = read_pair_cs(cspath)
    for node in cs.getNodes():
        if 'center' in node.getComment():
            feat = parse_synfeature_from_node(node)
            break
    return feat, cs.getComment()


def calc_syn_dict(features, axoness_info, get_all=False):
    """
    Creates dictionary of synapses. Keys are ids of pre cells and values are
    dictionaries of corresponding synapses with post cell ids.
    :param features: synapse feature
    :param axoness_info: string containing axoness information of cells
    :return: filtered features and axoness info, syn_dict and list of all post
    cell ids
    """
    """
    """
    total_size = float(len(axoness_info))
    ax_ax_cnt = 0
    den_den_cnt = 0
    all_post_ids = []
    pre_dict = {}
    val_syn_ixs = []
    valid_syn_array = np.ones_like(features)
    axoness_dict = {}
    for k, ax_info in enumerate(axoness_info):
        stdout.write("\r%0.2f" % (k / total_size))
        stdout.flush()
        cell1, cell2 = re.findall('(\d+)axoness(\d+)', ax_info)
        cs_nb = re.findall('cs(\d+)', ax_info)[0]
        cell_ids = arr([cell1[0], cell2[0]], dtype=np.int)
        cell_axoness = arr([cell1[1], cell2[1]], dtype=np.int)
        axoness_entry = {str(cell1[0]): cell1[1], str(cell2[0]): cell2[1]}
        axoness_dict[cs_nb + '_' + cell1[0] + '_' + cell2[0]] = axoness_entry
        if cell_axoness[0] == cell_axoness[1]:
            if cell_axoness[0] == 1:
                ax_ax_cnt += 1
                # if ax_ax_cnt < 20:
                #     print "AX:", syn_fpaths[k]
            else:
                den_den_cnt += 1
                # if den_den_cnt < 20:
                #     print "den:", syn_fpaths[k]
                valid_syn_array[k] = 0
                if not get_all:
                    continue
        val_syn_ixs.append(k)
        pre_ix = np.argmax(cell_axoness)
        pre_id = cell_ids[pre_ix]
        if pre_ix == 0:
            post_ix = 1
        else:
            post_ix = 0
        post_id = cell_ids[post_ix]
        all_post_ids += [post_id]
        syn_dict = {}
        syn_dict['post_id'] = post_id
        syn_dict['post_axoness'] = cell_axoness[post_ix]
        syn_dict['cs_area'] = features[k, 1]
        syn_dict['az_size_abs'] = features[k, 2]
        syn_dict['az_size_rel'] = features[k, 3]
        if pre_id in pre_dict.keys():
            syns = pre_dict[pre_id]
            if post_id in syns.keys():
                syns[post_id]['cs_area'] += features[k, 1]
                syns[post_id]['az_size_abs'] += features[k, 2]
            else:
                syns[post_id] = syn_dict
        else:
            syns = {}
            syns[post_id] = syn_dict
            pre_dict[pre_id] = syns
    print "Axon-Axon synapse:", ax_ax_cnt
    print "Dendrite-Dendrite synapse", den_den_cnt
    return features[val_syn_ixs], axoness_info[val_syn_ixs], pre_dict,\
           all_post_ids, valid_syn_array, axoness_dict
