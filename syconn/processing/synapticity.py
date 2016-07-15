import numpy as np
import os
import re
import time
from multiprocessing import Pool, Manager, cpu_count
from numpy import array as arr
from sys import stdout

from knossos_utils.knossosdataset import KnossosDataset as kd

import syconn.new_skeleton.annotationUtils as au
from learning_rfc import save_train_clf
from syconn.utils.datahandler import get_filepaths_from_dir

__author__ = 'philipp'


def save_synapse_clf(gt_path, clf_used='rf'):
    """
    Save synapse clf specified by clf_used to gt_directory.
    :param gt_path: str to directory of synapse ground truth
    :param clf_used: 'rf' or 'svm'
    """
    all_gt_samples = [path for path in get_filepaths_from_dir(gt_path, ending='nml')]
    gt_az_samples = [path for path in all_gt_samples if not 'p4_az.nml' in path]
    gt_az_p4_samples = [path for path in all_gt_samples if 'p4_az.nml' in path]
    X_az, Y_az = calc_syn_feature(gt_az_samples)
    X_p4_az, Y_p4_az = calc_syn_feature(gt_az_p4_samples)
    X_total = np.concatenate((X_az, X_p4_az), axis=0)
    y_total = np.concatenate((Y_az, Y_p4_az), axis=0)
    save_train_clf(X_total, y_total, clf_used, gt_path)


def helper_load_az_feat(args):
    """

    :param args:
    :return:
    """
    path, cs_path, q = args
    anno = au.loadj0126NML(path)[0]
    anno_nodes = list(anno.getNodes())
    for node in anno_nodes:
        if 'center' in node.getComment():
            center_coords = arr(node.getCoordinate_scaled(), dtype=np.int)
            break
    q.put(1)
    return center_coords, path


def calc_syn_feature(gt_samples, ignore_keys=['Barrier', 'Skel'],
                     new_data=False, test_data=False, detailed_cs_dir='/lustre/'
                        'pschuber/m_consensi_rr/nml_obj/contact_sites_new3/'):
    """
    collect synpase feature of all contact sites. Additionally, ground truth
    values if test_data is True.
    :param gt_samples: List of paths to contact sites
    :param ignore_keys: Which keys to ignore in string if collecting GT value
    :param new_data: outdated
    :param test_data: whether to collect GT value
    :param detailed_cs_dir: path to folder containing the contact sites
    :return:
    """
    for ending in ['', 'cs', 'cs_p4', 'cs_az', 'cs_p4_az', 'pairwise',
                   'feature']:
        if not os.path.exists(detailed_cs_dir+ending):
            os.makedirs(detailed_cs_dir+ending)
    np.random.shuffle(gt_samples)
    nb_cpus = cpu_count()
    pool = Pool(processes=nb_cpus)
    m = Manager()
    q = m.Queue()
    params = [(sample, ignore_keys, detailed_cs_dir,
               q, new_data, test_data) for sample in gt_samples]
    result = pool.map_async(pairwise_syn_feature_calc, params)
    while True:
        if result.ready():
            break
        else:
            size = float(q.qsize())
            stdout.write("\r%0.2f" % (size / len(params)))
            stdout.flush()
            time.sleep(4)
    res = result.get()
    pool.close()
    pool.join()
    print "\n", len(res), len(filter(None, res))
    res = np.array(filter(None, res))
    non_instances = arr([isinstance(el, np.ndarray) for el in res[:,0]])
    print len(res[non_instances])
    X = np.array(res[:, 0][non_instances].tolist(), dtype=np.float)
    Y = np.array(res[:, 1][non_instances], dtype=np.int)
    return X, Y


def pairwise_syn_feature_calc(args):
    """
    Helper function for calc_syn_feature. Collects feature of contact site.
    :param args: path to contact sites, list of ingore keys, path to
    contact_sites folder, q of multiprocess manager, bool new data(old),
    bool test_data (whether to collect gt_value)
    :return: synapse feature, ground truth value
    """
    syn_candidate, ignore_keys, detailed_cs_dir, q,\
    new_data, test_data = args
    if isinstance(syn_candidate, str):
        syn_candidate = au.loadj0126NML(syn_candidate)[0]
    q.put(1)
    if not test_data:
        gt_value = None
    else:
        gt_value = np.zeros((0,))
    for node in syn_candidate.getNodes():
        node_comment = node.getComment()
        if 'center' in node_comment:
            if np.sum([key in node_comment for key in ignore_keys]):
                return
            if (not 'False' in node_comment) and (not 'True' in node_comment):
                if not test_data:
                    return
            gt_value = 'True' in node_comment
            break
    feat = parse_synfeature_from_node(node)
    return feat, np.array(gt_value)


def parse_synfeature_from_txt(txt):
    """
    Parases values of features from string.
    :param txt: String with values of feature_names, like 'area1.5_dist2.3'
    :return: array of float values for each feature
    """
    feature_names = ['dist', 'area', 'areaol', 'relol', 'absol', 'csrelol']
    feat_arr = np.zeros((len(feature_names, )))
    for k, name in enumerate(feature_names):
        matches = re.findall('%s(\d+.\d+)' % name, txt)
        if len(matches) == 0:
            continue
        feat_arr[k] = float(matches)
    return feat_arr


def parse_synfeature_from_node(node):
    """
    Parases values of features from string.
    :param node: node with values of feature_names
    :return: array of float values for each feature
    """
    feature_names = ['cs_dist', 'mean_cs_area', 'overlap_area', 'overlap',
                     'abs_ol', 'overlap_cs']
    feat_arr = np.zeros((len(feature_names, )))
    for k, feat_name in enumerate(feature_names):
        feat = float(node.data[feat_name])
        feat_arr[k] = feat
    return feat_arr


def syn_sign_prediction(voxels, threshold=.25,
                        kd_path_sym='/lustre/sdorkenw/j0126_3d_symmetric/',
                        kd_path_asym='/lustre/sdorkenw/j0126_3d_asymmetric/'):

    kd_asym = kd()
    kd_asym.initialize_from_knossos_path(kd_path_asym)
    kd_sym = kd()
    kd_sym.initialize_from_knossos_path(kd_path_sym)
    sym_values = kd_sym.from_raw_cubes_to_list(voxels)
    asym_values = kd_asym.from_raw_cubes_to_list(voxels)
    ratio = float(np.sum(sym_values))/(np.sum(sym_values)+np.sum(asym_values))

    if threshold is None:
        return ratio
    else:
        return int(ratio > threshold)
