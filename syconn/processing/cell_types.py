# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve,\
    precision_recall_fscore_support, accuracy_score
from learning_rfc import *
from syconn.processing.features import *
from syconn.utils.datahandler import get_filepaths_from_dir,\
    load_ordered_mapped_skeleton, load_mapped_skeleton, get_paths_of_skelID,\
    write_obj2pkl, load_pkl2obj, get_skelID_from_path
from syconn.utils.neuron import Neuron
__author__ = 'pschuber'

rf_params = {'n_estimators': 4000, 'oob_score': True, 'n_jobs': -1,
             'class_weight': 'balanced', 'max_features': 0.66}

rf_params_nodewise = {'n_estimators': 2000, 'oob_score': True, 'n_jobs': -1,
                      'class_weight': 'balanced', 'max_features': 'auto'}


def get_cell_type_labels():
    """

    :return:
    """
    labels = {}
    labels["HVC"] = 0
    labels["LMAN"] = 0
    labels["STN"] = 0
    labels["MSN"] = 1
    labels["GP"] = 2
    labels["FS"] = 3
    return labels


def get_cell_type_classes_dict():
    """

    :return:
    """
    label_strings_dict = {}
    label_strings_dict[0] = "Excitatory Axons"
    label_strings_dict[1] = "Medium spiny Neurons"
    label_strings_dict[2] = "Pallidal-like Neurons"
    label_strings_dict[3] = "Inhibitory Interneurons"
    cell_type_label_dict = get_cell_type_labels()
    cell_type_classes_dict = {}
    for key, value in cell_type_label_dict.iteritems():
        try:
            cell_type_classes_dict[label_strings_dict[value]].append(key)
        except KeyError:
            cell_type_classes_dict[label_strings_dict[value]] = [key]
    return cell_type_classes_dict


def save_cell_type_clf(gt_path='/lustre/pschuber/gt_cell_types/',
                       clf_used='rf', load_data=True):
    """
    Save axoness clf specified by clf_used to gt_directory.
    :param gt_path: str to directory of axoness ground truth
    :param clf_used: 'rf' or 'svm'
    """
    X_cells_types, Y_cells_types = load_celltype_gt(load_data=load_data)
    save_train_clf(X_cells_types, Y_cells_types, clf_used, gt_path, params=rf_params)


def load_celltype_gt(load_data=True, return_ids=False):
    """
    (HVC, LMAN, STN) => excitatory axons (0)
    (MSN) => medium spiny neuron (1)
    (GP) => pallidal-like neurons (2)
    (FS) => inhibitory interneuron (3)
    :param gt_path:
    :return:
    """
    if not load_data:
        save_cell_type_feats()
    skel_ids, skel_feat = load_celltype_feats()
    skel_label = load_cell_gt(skel_ids)
    bool_arr = skel_label != -1
    x = skel_feat[bool_arr]
    y = skel_label[bool_arr]
    if return_ids:
        return x, y.astype(np.uint), skel_ids[bool_arr]
    return x, y.astype(np.uint)


def find_cell_types_from_dict(cell_type, skel_dir):
    """
    returns list containing all paths of cells of type cell_type.
    """
    # TODO: skel_label = load_cell_gt(skel_ids)
    # TODO: bool_arr = skel_label != -1
    # TODO: y = skel_label[bool_arr]
    # TODO: find skel path using id
    cell_type_dict = load_pkl2obj('/lustre/pschuber/gt_cell_types/'
                                        'consensi_celltype_labels_reviewed3.pkl')
    fpaths = []
    label_dict = get_cell_type_labels()
    if cell_type in label_dict.keys():
        cell_type = label_dict.values()[label_dict.keys().index(cell_type)]
    for key, value in cell_type_dict.iteritems():
        if value == cell_type:
            path = get_paths_of_skelID([str(key)], traced_skel_dir=skel_dir)[0]
            fpaths.append(path)
    return fpaths


def write_cell_type_examples(dest='/home/pschuber/cell_type_examples/'):
    """
    Write example skeletons of each class to axon, in order to inspect them.
    """
    if not os.path.isdir(dest):
        os.makedirs(dest)
    ctc_dict = get_cell_type_classes_dict()
    for key, type_names in ctc_dict.iteritems():
        fpaths = []
        for name in type_names:
            fpaths += find_cell_types_from_dict(name)
        print "Found %d %s." % (len(fpaths), type_names)
        np.random.seed()
        np.random.shuffle(fpaths)
        for k, fpath in enumerate(fpaths[:5]):
            shutil.copyfile(fpath, dest + '%s_%d.k.zip' % (key, k))


def get_id_dict_from_skel_ids(skel_ids):
    """
    Calc dictionary to get new label (from 0 to len(skel_paths) from
    skeleton ID.
    :param skel_ids: list of skeleton ids
    :return: id_dict, ewv_id_dict
    """
    id_dict = {}
    rev_id_dict = {}
    for skel_id in skel_ids:
        id_dict[skel_id] = len(id_dict.keys())
        rev_id_dict[len(rev_id_dict.keys())] = skel_id
    return id_dict, rev_id_dict


def load_cell_gt(skel_ids):
    """
    Load cell types of skel ids
    :return: array of cell type classes
    """
    skel_labels = -1 * np.ones(len(skel_ids))
    consensi_celltype_label = load_pkl2obj('/lustre/pschuber/gt_cell_types/'
                                        'consensi_celltype_labels_reviewed3.pkl')
    for i in range(len(skel_ids)):
        try:
            class_nb = consensi_celltype_label[skel_ids[i]]
            skel_labels[i] = class_nb
        except KeyError:
            pass
    print "Using %d/%d labeled skeletons as GT for cell type RFC training." % \
          (len(skel_labels[skel_labels != -1]), len(skel_labels))
    return skel_labels


def load_celltype_feats(gt_path='/lustre/pschuber/gt_cell_types/'):
    """
    Loads cell type feature and corresponding ids from dictionaries.
    :param gt_path:
    :return: skeleton_ids, skeleton_feats
    """
    feat_dict = load_pkl2obj(gt_path + '/wiring/skel_feat_dict.pkl')
    skeleton_feats = []
    skeleton_ids = []
    for key, val in feat_dict.iteritems():
        skeleton_feats.append(val)
        skeleton_ids.append(key)
    ixs = np.argsort(skeleton_ids)
    return arr(skeleton_ids)[ixs], arr(skeleton_feats)[ixs]


def load_celltype_probas(gt_path='/lustre/pschuber/gt_cell_types/'):
    """
    Loads cell type probabilities and corresponding ids from dictionaries.
    :param gt_path:
    :return: skeleton_ids, skel_probas
    """
    proba_dict = load_pkl2obj(gt_path + '/wiring/celltype_proba_dict.pkl')
    skel_probas = []
    skeleton_ids = []
    for key, val in proba_dict.iteritems():
        skel_probas.append(val)
        skeleton_ids.append(key)
    ixs = np.argsort(skeleton_ids)
    return arr(skeleton_ids)[ixs], arr(skel_probas)[ixs]


def save_cell_type_feats(skeleton_dir='/lustre/pschuber/m_consensi_rr/nml_obj/',
                           gt_path='/lustre/pschuber/gt_cell_types/'):
    skel_paths = get_filepaths_from_dir(skeleton_dir)
    # predict skeleton cell type probability
    skel_ids = []
    feat_dict = {}
    result_tuple = start_multiprocess(calc_neuron_feat, skel_paths, debug=False,
                                      nb_cpus=16)
    for feat, skel_id in result_tuple:
        skel_ids.append(skel_id)
        feat_dict[skel_id] = feat[0]
    write_obj2pkl(feat_dict, gt_path + '/wiring/skel_feat_dict.pkl')
    # get example neuron and write neuron feature names
    cell = load_mapped_skeleton(skel_paths[0], True, True)[0]
    cell.filename = skel_paths[0]
    neuron = Neuron(cell)
    _ = neuron.neuron_features
    feat_names = neuron.neuron_feature_names
    np.save(gt_path + 'feat_names.npy', feat_names)
    #np.save(gt_path + '/wiring/skeleton_feats.npy',
    #        skeleton_feats)
    #np.save(gt_path +'/wiring/skel_ids.npy', skel_ids)


def calc_neuron_feat(path):
    """
    Calculate neuron features using neuron class.
    :param path: path to mapped annotation kzip
    :return: feature array, skel ID
    """
    orig_skel_id = get_skelID_from_path(path)
    cell = load_mapped_skeleton(path, True, True)[0]
    cell.filename = path
    neuron = Neuron(cell)
    feats = neuron.neuron_features
    if np.any(np.isnan(feats)):
        print "Found nans in feautres of skel %s" % path, \
            neuron.neuron_feature_names[np.isnan(neuron.neuron_features[0])]
        print neuron.neuron_features[0][np.isnan(neuron.neuron_features[0])]
    return feats, orig_skel_id


def write_feats_importance(gt_path='/lustre/pschuber/gt_cell_types/',
                           load_data=True, clf_used='rf'):
    """
    Writes out feature importances and feature names to gt_dir
    """
    save_cell_type_clf(gt_path, load_data=load_data, clf_used=clf_used)
    feat_names = np.load(gt_path + 'feat_names.npy')

    rf = joblib.load(gt_path + '/%s/%s.pkl' % (clf_used, clf_used))
    skeleton_ids, skeleton_feats = load_celltype_feats(gt_path)
    skel_type_probas = rf.predict_proba(skeleton_feats)
    proba_dict = {}
    print "# of feats and feat-names:", skeleton_feats.shape[1], len(feat_names)
    for ii, proba in enumerate(skel_type_probas):
        proba_dict[skeleton_ids[ii]] = proba

    importances = rf.feature_importances_
    feature_importance(rf, save_path='/home/pschuber/figures/cell_types/'
                                     'rf_feat_importance.png')
    tree_imp = [tree.feature_importances_ for tree in rf.estimators_]
    print "Print feature importance of rf with %d trees." % len(tree_imp)
    std = np.std(tree_imp, axis=0) / np.sqrt(len(tree_imp))
    assert len(importances) == len(feat_names), "Number of names and features" \
                                                "differs."
    np.save(gt_path + 'feat_importances.npy', importances)
    np.save(gt_path + 'feat_std.npy', std)


def draw_feat_hist(gt_path='/lustre/pschuber/gt_cell_types/', k=15,
                   classes=[0, 1, 2, 3], nb_bars=20):
    """
    Draws the histgoram(s) of the most k important feature(s)
    :param gt_path: str Path to gt folder
    :param k: int Number of features to be plotted
    :return:
    """
    label_strings_dict = {}
    label_strings_dict[0] = "Excitatory Axons"
    label_strings_dict[1] = "Medium Spiny Neurons"
    label_strings_dict[2] = "Pallidal-like Neurons"
    label_strings_dict[3] = "Inhibitory Interneurons"
    importances = np.load(gt_path + 'feat_importances.npy')
    feat_names = np.load(gt_path + 'feat_names.npy')
    skel_ids, feats = load_celltype_feats(gt_path)
    skel_ids2, skel_type_probas = load_celltype_probas(gt_path)
    assert np.all(np.equal(skel_ids, skel_ids2)), "Skeleton ordering wrong for"\
                                                  "probabilities and features."
    skel_preds = np.argmax(skel_type_probas, axis=1)
    indices = np.argsort(importances)[::-1]
    print feat_names[indices]
    for n in range(k):
        ix = indices[n]
        fig, ax = plt.subplots()
        fig.patch.set_facecolor('white')

        ax.tick_params(axis='x', which='major', labelsize=16, direction='out',
                        length=8, width=2,  right="off", top="off", pad=10)
        ax.tick_params(axis='y', which='major', labelsize=16, direction='out',
                        length=8, width=2,  right="off", top="off", pad=10)

        ax.tick_params(axis='x', which='minor', labelsize=12, direction='out',
                        length=4, width=1, right="off", top="off", pad=10)
        ax.tick_params(axis='y', which='minor', labelsize=12, direction='out',
                        length=4, width=1, right="off", top="off", pad=10)

        ax.spines['left'].set_linewidth(2)
        ax.spines['bottom'].set_linewidth(2)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        curr_feat_name = feat_names[ix]

        ax.set_ylabel('Relative Counts', fontsize=18)
        ax.set_xlabel('%s' % curr_feat_name, fontsize=18)

        plt.gcf().subplots_adjust(bottom=0.15)
        colors = ['r', 'b', arr([135, 206, 250])/255.,
                  arr([255, 255, 224])/255.]
        print "Plotting %d. feature %s" % (n, curr_feat_name)
        mask_arr = np.zeros(len(feats), dtype=np.bool)
        for c in classes:
            mask_arr += skel_preds == c
        x_max = np.max(feats[mask_arr, ix])
        x_min = np.min(feats[mask_arr, ix])
        ranges = (x_min, x_max)
        print "Using range", ranges
        for ii, c in enumerate(classes):
            mask_arr = skel_preds == c
            print "Found %d cells of class %d" % (np.sum(mask_arr), c)
            ax.hist(feats[mask_arr][:, ix], normed=True, range=ranges, alpha=0.8,
                    bins=nb_bars, label=label_strings_dict[c], color=colors[ii])
        ymin, ymax = ax.get_ylim()
        xmin, xmax = ax.get_xlim()
        ax.set_xlim(xmin, xmax*1.1)
        ax.set_ylim(ymin, ymax*1.1)
        legend = ax.legend(loc="upper right", frameon=False, prop={'size': 16})
        fig_path = '/home/pschuber/figures/cell_types/'
        plt.savefig(fig_path + "/feat%dhist%d.png" % (n, len(classes)), dpi=600)


def feat_calc_helper(params):
    """

    :param params:
    :return:
    """
    skel_path, cell_label = params
    skel = load_ordered_mapped_skeleton(skel_path)[0]
    morph_feat, spine_feat, node_feat_ids = morphology_feature_celltypes(skel_path)
    node_feats = np.concatenate((morph_feat, spine_feat), axis=1)
    skel_nodes = skel.getNodes()
    x = np.zeros((len(skel_nodes), 32))
    y = np.zeros(len(skel_nodes)) + cell_label
    for ii, node in enumerate(skel_nodes):
        node_id = node.getID()
        x[ii] = node_feats[node_feat_ids.index(node_id)]
    print "Finished", skel_path
    return x, y


def morphology_feature_celltypes(source, max_nn_dist=6000):
    """
    Calculates features for discrimination tasks of neurite identities, such as
    axon vs. dendrite or cell types classification. Estimated on interpolated
    skeleton nodes. Features are calculated with a sliding window approach for
    each node. Window is 2*max_nn_dist (nm).
    :param source: str Path to anno or MappedSkeletonObject
    :param max_nn_dist: float Radius in which neighboring nodes are found and
    used for calculating features in nm.
    :return: array of features for each node. number of nodes x 28 (22 radius
    feature and 6 object features)
    """
    if isinstance(source, basestring):
        anno = load_ordered_mapped_skeleton(source)[0]
        # build mito sample tree
        mitos, p4, az = load_objpkl_from_kzip(source)
    else:
        anno = source.old_anno
        if source.mitos is None:
            mitos, p4, az = load_objpkl_from_kzip(anno.filename)
        else:
            mitos = source.mitos
            p4 = source.p4
            az = source.az
    m_dict, p4_dict, az_dict = (mitos.object_dict, p4.object_dict,
                                az.object_dict)
    nearby_node_list = nodes_in_pathlength(anno, max_nn_dist)
    node_coords = []
    node_radii = []
    node_ids = []
    rad_feats = np.zeros((len(nearby_node_list), 14))
    spiness_feats = np.zeros((len(nearby_node_list), 12))
    for jj, nodes in enumerate(nearby_node_list):
        node_coords.append(nodes[0].getCoordinate_scaled())
        node_radii.append(nodes[0].getDataElem("radius"))
        node_ids.append(nodes[0].getID())
        ax_preds = np.zeros((len(nodes)), dtype=np.uint16)
        type_rad_ranges = [1000, 500, 1500]
        for ii, node in enumerate(nodes):
            ax_preds[ii] = int(node.data["axoness_pred"])
        for i in range(3):
            ix_begin = i * 6
            ix_end = (i+1) * 6
            ax_nodes = arr(nodes)[ax_preds == i]
            if len(ax_nodes) == 0:
                continue
            if i == 2:
                rad_feats[jj, ix_begin:] = radius_feats_from_nodes(ax_nodes,
                                    nb_bins=10, max_rad=type_rad_ranges[i])[:2]
            else:
                rad_feats[jj, ix_begin:ix_end] = radius_feats_from_nodes(ax_nodes,
                                    nb_bins=10, max_rad=type_rad_ranges[i])[:6]
            if i != 2:
                spiness_feats[jj, ix_begin:ix_end] = spiness_feats_from_nodes(ax_nodes)
    m_feat = objfeat2skelnode_cellnodes(node_coords, node_radii, node_ids,
                              nearby_node_list, m_dict, anno.scaling)
    p4_feat = objfeat2skelnode_cellnodes(node_coords, node_radii, node_ids,
                               nearby_node_list, p4_dict, anno.scaling)
    az_feat = objfeat2skelnode_cellnodes(node_coords, node_radii, node_ids,
                               nearby_node_list, az_dict, anno.scaling, True)
    morph_feat = np.concatenate((rad_feats, m_feat, p4_feat, az_feat), axis=1)
    if np.any(np.isnan(morph_feat)):
        print "Found nans in morphological features of %s: %s" % \
              (source, np.where(np.isnan(morph_feat)))
        morph_feat = np.nan_to_num(morph_feat.astype(np.float32))
    if np.any(np.isnan(spiness_feats)):
        print "Found nans in spinhead features of %s: %s" % \
              (source, np.where(np.isnan(spiness_feats)))
        spiness_feats = np.nan_to_num(spiness_feats.astype(np.float32))
    return morph_feat, spiness_feats, node_ids


def objfeat2skelnode_cellnodes(node_coords, node_radii, node_ids,
                               nearby_node_list, obj_dict, scaling):
    """
    Calculate features of SegmentationDatasetObjects along Skeleton.
    :param node_coords:
    :param node_radii:
    :param node_ids:
    :param nearby_node_list:
    :param obj_dict:
    :param scaling:
    :return: array of dimension nb_skelnodes x 2. The two features are:
    absolute number of assigned objects and mean voxel size of the objects.
    """
    skeleton_tree = spatial.cKDTree(node_coords)
    nb_skelnodes = len(node_coords)
    obj_assignment = [[] for i in range(nb_skelnodes)]
    nb_objs = len(obj_dict.keys())
    hull_samples = np.zeros((nb_objs, 100, 3))
    key_list = []
    obj_features = np.zeros((nb_skelnodes, 2))
    for i, obj_key in enumerate(obj_dict.keys()):
        obj_object = obj_dict[obj_key]
        m_hull = obj_object.hull_voxels * scaling
        random_ixs = np.random.choice(np.arange(len(m_hull)), size=100)
        hull_samples[i] = m_hull[random_ixs]
        key_list.append(obj_key)
    for i in range(nb_objs):
        dists, nearest_skel_ixs = skeleton_tree.query(hull_samples[i], 1)
        for ix in list(set(nearest_skel_ixs)):
            if np.min(dists[nearest_skel_ixs == ix]) > node_radii[ix]*10:
                continue
            obj_assignment[ix] += [key_list[i]]
    for k in range(nb_skelnodes):
        nn_nodes = nearby_node_list[k]
        nn_ids = [nn.getID() for nn in nn_nodes]
        assigned_objs = []
        for nn_id in nn_ids:
            assigned_objs += obj_assignment[node_ids.index(nn_id)]
        obj_features[k, 0] = len(assigned_objs)
        if len(assigned_objs) == 0:
            continue
        obj_sizes = [obj_dict[key].size for key in assigned_objs]
        obj_features[k, 1] = np.mean(obj_sizes)
    return obj_features
