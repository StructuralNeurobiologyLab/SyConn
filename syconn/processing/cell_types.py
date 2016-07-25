# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt

from learning_rfc import *
from syconn.multi_proc.multi_proc_main import start_multiprocess
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
    """Cell type labels for HVC(0), LMAN(0), STN(0), MSN(1), GP(2), FS(3)

    Returns
    -------
    dict
        convetion dictionary for cell type labels (str), returns integer
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
    Returns
    -------
    dict
        dictionary from integer label to full cell name as string
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


def save_cell_type_clf(gt_path, clf_used='rf', load_data=True):
    """Save axoness clf specified by clf_used to gt_directory.

    Parameters
    ----------
    gt_path : str
        path to cell type gt
    celf_used : str
    load_data : bool
    """
    X_cells_types, Y_cells_types = load_celltype_gt(load_data=load_data)
    save_train_clf(X_cells_types, Y_cells_types, clf_used, gt_path, params=rf_params)


def load_celltype_gt(wd, load_data=True, return_ids=False):
    """Load ground truth of cell types

    (HVC, LMAN, STN) => excitatory axons (0)
    (MSN) => medium spiny neuron (1)
    (GP) => pallidal-like neurons (2)
    (FS) => inhibitory interneuron (3)

    Parameters
    ----------
    wd : str
        Path to working directory

    Returns
    -------
    numpy.array, numpy.array
        cell type features, cell type labels
    """
    if not load_data:
        save_cell_type_feats(wd)
    skel_ids, skel_feat = load_celltype_feats(wd)
    skel_label = load_cell_gt(skel_ids)
    bool_arr = skel_label != -1
    x = skel_feat[bool_arr]
    y = skel_label[bool_arr]
    if return_ids:
        return x, y.astype(np.uint), skel_ids[bool_arr]
    return x, y.astype(np.uint)


def find_cell_types_from_dict(wd, cell_type):
    """
    Parameters
    ----------
    wd : str
        Path to working directory
    cell_type : int
        label (0 = EA, 1 = MSN, 2 = GP, 3 = INT)
    Returns
    -------
    list of str
        paths of cells of type cell_type.
    """
    skel_dir = wd + '/neurons/'
    cell_type_dict = load_pkl2obj(wd + '/neurons/celltype_labels.pkl')
    fpaths = []
    label_dict = get_cell_type_labels()
    if cell_type in label_dict.keys():
        cell_type = label_dict.values()[label_dict.keys().index(cell_type)]
    for key, value in cell_type_dict.iteritems():
        if value == cell_type:
            path = get_paths_of_skelID([str(key)], traced_skel_dir=skel_dir)[0]
            fpaths.append(path)
    return fpaths


def get_id_dict_from_skel_ids(skel_ids):
    """
    Calc dictionary to get new label (from 0 to len(skel_paths) from
    skeleton ID.

    Parameters
    ----------
    skel_ids : list of int

    Returns
    -------
    dict, dict
        Dictionary to get new ID with old ID, dictionary to get old ID
        with new ID
    """
    id_dict = {}
    rev_id_dict = {}
    for skel_id in skel_ids:
        id_dict[skel_id] = len(id_dict.keys())
        rev_id_dict[len(rev_id_dict.keys())] = skel_id
    return id_dict, rev_id_dict


def load_cell_gt(skel_ids, wd):
    """Load cell types of skel ids
    Parameters
    ----------
    skel_ids : list of int
    wd : str

    Returns
    -------
    np.array
        cell type labels
    """
    skel_labels = -1 * np.ones(len(skel_ids))
    consensi_celltype_label = load_pkl2obj(wd + '/neurons/celltype_labels.pkl')
    for i in range(len(skel_ids)):
        try:
            class_nb = consensi_celltype_label[skel_ids[i]]
            skel_labels[i] = class_nb
        except KeyError:
            pass
    print "Using %d/%d labeled skeletons as GT for cell type RFC training." % \
          (len(skel_labels[skel_labels != -1]), len(skel_labels))
    return skel_labels


def load_celltype_feats(wd):
    """Loads cell type feature and corresponding ids from dictionaries

    Parameters
    ----------
    wd : str
        Path to working directory

    Returns
    -------
    np.array
        cell type features
    """
    if not os.path.isfile(wd + '/neurons/celltype_feat_dict.pkl'):
        save_cell_type_feats(wd)
    feat_dict = load_pkl2obj(wd + '/neurons/celltype_feat_dict.pkl')
    skeleton_feats = np.zeros((len(feat_dict.keys()),
                               len(feat_dict.values()[0])))
    skeleton_ids = []
    ii = 0
    for key, val in feat_dict.iteritems():
        skeleton_feats[ii] = arr(val)
        skeleton_ids.append(key)
        ii += 1
    ixs = np.argsort(skeleton_ids)
    return arr(skeleton_ids)[ixs], skeleton_feats[ixs]


def load_celltype_probas(wd):
    """Loads cell type probabilities and corresponding ids from dictionaries

    Parameters
    ----------
    wd : str
        Path to working directory

    Returns
    -------
    np.array, np.array
        cell ids, cell label probabilities
    """
    if not os.path.isfile(wd + '/neurons/celltype_proba_dict.pkl'):
        predict_celltype_label(wd)
    proba_dict = load_pkl2obj(wd + '/neurons/celltype_proba_dict.pkl')
    skel_probas = []
    skeleton_ids = []
    for key, val in proba_dict.iteritems():
        skel_probas.append(val)
        skeleton_ids.append(key)
    ixs = np.argsort(skeleton_ids)
    return arr(skeleton_ids)[ixs], arr(skel_probas)[ixs]


def load_celltype_preds(wd):
    """Loads cell type predictions and corresponding ids from dictionaries

    Parameters
    ----------
    wd : str
        Path to working directory

    Returns
    -------
    np.array, np.array
        cell ids, cell labels
    """
    if not os.path.isfile(wd + '/neurons/celltype_pred_dict.pkl'):
        predict_celltype_label(wd)
    pred_dict = load_pkl2obj(wd + '/neurons/celltype_pred_dict.pkl')
    skel_probas = []
    skeleton_ids = []
    for key, val in pred_dict.iteritems():
        skel_probas.append(val)
        skeleton_ids.append(key)
    ixs = np.argsort(skeleton_ids)
    return arr(skeleton_ids)[ixs], arr(skel_probas)[ixs]


def predict_celltype_label(wd):
    """Predict celltyoe labels in working directory with pre-trained classifier
    in subfolder models/rf_celltypes/rf.pkl

    Parameters
    ----------
    wd : str
        path to working directory
    """
    rf = joblib.load(wd + '/models/rf_celltypes/rf.pkl')
    skeleton_ids, skeleton_feats = load_celltype_feats(wd)
    skel_type_probas = rf.predict_proba(skeleton_feats)
    proba_dict = {}
    cell_type_pred_dict = {}
    for ii, proba in enumerate(skel_type_probas):
        proba_dict[skeleton_ids[ii]] = proba
        cell_type_pred_dict[skeleton_ids[ii]] = np.argmax(proba)
    write_obj2pkl(cell_type_pred_dict, wd + '/neurons/'
                                            'celltype_pred_dict.pkl')
    write_obj2pkl(proba_dict, wd + '/neurons/celltype_proba_dict.pkl')


def save_cell_type_feats(wd):
    """Saves cell type feature for type prediction

    Parameters
    ----------
    wd : str
        Path to working directory
    """
    skel_dir = wd + '/neurons/'
    skel_paths = get_filepaths_from_dir(skel_dir)
    print "Calculating cell type feats of %d tracings." % len(skel_paths)
    # predict skeleton cell type probability
    skel_ids = []
    feat_dict = {}
    result_tuple = start_multiprocess(calc_neuron_feat, skel_paths, debug=True)
    for feat, skel_id in result_tuple:
        skel_ids.append(skel_id)
        feat_dict[skel_id] = feat[0]
    write_obj2pkl(feat_dict, wd + '/neurons/celltype_feat_dict.pkl')
    # get example neuron and write neuron feature names
    cell = load_mapped_skeleton(skel_paths[0], True, True)[0]
    cell.filename = skel_paths[0]
    neuron = Neuron(cell)
    _ = neuron.neuron_features
    feat_names = neuron.neuron_feature_names
    np.save(wd + '/neurons/celltype_feat_names.npy', feat_names)


def calc_neuron_feat(path):
    """Calculate neuron features using neuron class

    Parameters
    ----------
    path : str
        path to mapped annotation kzip

    Returns
    -------
    numpy.array, numpy.array
        cell type features, skeleton ID
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
    print "Finished feature extraction of %s" % path
    return feats, orig_skel_id


def write_feats_importance(wd, load_data=True, clf_used='rf'):
    """Writes out feature importances and feature names to gt_dir

    Parameters
    ----------
    wd : str
    load_data : bool
    clf_used : str
    """
    save_cell_type_clf(wd, load_data=load_data, clf_used=clf_used)
    feat_names = np.load(wd + '/neurons/feat_names.npy')

    rf = joblib.load(wd + '/models/celltypes/%s.pkl' % (clf_used))
    importances = rf.feature_importances_
    feature_importance(rf, save_path='/home/pschuber/figures/cell_types/'
                                     'rf_feat_importance.png')
    tree_imp = [tree.feature_importances_ for tree in rf.estimators_]
    print "Print feature importance of rf with %d trees." % len(tree_imp)
    std = np.std(tree_imp, axis=0) / np.sqrt(len(tree_imp))
    assert len(importances) == len(feat_names), "Number of names and features" \
                                                "differs."
    np.save(wd + '/neurons/feat_importances.npy', importances)
    np.save(wd + '/neurons/feat_std.npy', std)


def draw_feat_hist(wd, k=15, classes=(0, 1, 2, 3), nb_bars=20):
    """Draws the histgoram(s) of the most k important feature(s)

    Parameters
    ----------
    wd : str
        Path to working directory
    k : int
        Number of features to be plotted
    classes : tuple
        Class labels to evaluate
    nb_bars : int
        Number of bars in histogram
    """
    label_strings_dict = {}
    label_strings_dict[0] = "Excitatory Axons"
    label_strings_dict[1] = "Medium Spiny Neurons"
    label_strings_dict[2] = "Pallidal-like Neurons"
    label_strings_dict[3] = "Inhibitory Interneurons"
    importances = np.load(wd + '/neurons/feat_importances.npy')
    feat_names = np.load(wd + '/neurons/feat_names.npy')
    skel_ids, feats = load_celltype_feats(wd)
    skel_ids2, skel_type_probas = load_celltype_probas(wd)
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
        fig_path = wd + '/figures/'
        plt.savefig(fig_path + "/feat%dhist%d.png" % (n, len(classes)), dpi=600)
