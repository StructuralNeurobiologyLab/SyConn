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
from processing.mapper import feature_valid_syns, calc_syn_dict
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
