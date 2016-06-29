# -*- coding: utf-8 -*-
__author__ = 'pschuber'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import shutil
from numpy import array as arr
import numpy as np
from scipy import spatial
try:
    from NewSkeleton import annotationUtils as au
except:
    import annotationUtils as au
try:
    from NewSkeleton.NewSkeletonUtils import annotation_from_nodes
except:
    from NewSkeletonUtils import annotation_from_nodes
from heraca.utils.datahandler import get_filepaths_from_dir,\
    load_ordered_mapped_skeleton, load_mapped_skeleton, get_paths_of_skelID,\
    write_obj2pkl, load_pkl2obj, get_skelID_from_path
from learning_rfc import save_train_clf,\
    feature_importance, start_multiprocess, init_clf
import heraca.neuron as neuron
from sklearn.externals import joblib
try:
    from NewSkeleton import annotationUtils as au
except:
    import annotationUtils as au


rf_params = {'n_estimators': 4000, 'oob_score': True, 'n_jobs': -1,
             'class_weight': 'balanced', 'max_features': 0.66}

def get_cell_type_labels():
    labels = {}
    labels["HVC"] = 0
    labels["LMAN"] = 0
    labels["STN"] = 0
    labels["MSN"] = 1
    labels["GP"] = 2
    labels["FS"] = 3
    return labels


def get_cell_type_classes_dict():
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
    X_cells_types, Y_cells_types = load_celltype_gt(gt_path, load_data=load_data)
    save_train_clf(X_cells_types, Y_cells_types, clf_used, gt_path, params=rf_params)


def load_celltype_gt(gt_path="/lustre/pschuber/gt_cell_types/",
                     load_data=True, return_ids=False):
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


def find_cell_types_from_dict(cell_type='MSN', skel_dir='/lustre/pschuber/mapped_soma_tracings/nml_obj/'):
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
    # dummyskel = NewSkeleton()
    for key, value in cell_type_dict.iteritems():
        if value == cell_type:
            path = get_paths_of_skelID([str(key)], traced_skel_dir=skel_dir)[0]
            fpaths.append(path)
            # skel = load_ordered_mapped_skeleton(key)[0]
            # dummyskel.add_annotation(skel)
    # dummyskel.toNml('/lustre/pschuber/%s_list.nml' % cell_type)
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


def write_consensi_labels():
    """
    Match gt skels with consensi skel ids
    writes dictionary of skel_ids with corresponding label
    :return:
    """
    skel_gt_path = '/lustre/pschuber/gt_cell_types/current_celltype_gt.k.zip'
    consensi_skel = get_filepaths_from_dir('/lustre/pschuber/m_consensi_rr/'
                                           'nml_obj/')
    gt_skels = au.loadj0126NML(skel_gt_path)
    consensi_celltype_label = {}
    cellname2label_dict = get_cell_type_labels()
    coords = []
    for ii, skel in enumerate(gt_skels):
        for node in skel.getNodes():
            coords.append(node.getCoordinate_scaled())
            break
    nml_tree = spatial.cKDTree(coords)
    for kk, skel_inner_path in enumerate(consensi_skel):
        skel_inner = load_ordered_mapped_skeleton(skel_inner_path)[0]
        skel_coords = []
        for node in skel_inner.getNodes():
            skel_coords.append(node.getCoordinate_scaled())
        near_gt = nml_tree.query_ball_point(skel_coords, 1)
        nb_equal = list(set([id for sublist in near_gt for id in sublist]))[0]
        print set([id for sublist in near_gt for id in sublist])
        skel_ID = get_skelID_from_path(skel_inner.filename)
        gt_skel = gt_skels[nb_equal]
        try:
            class_nb = cellname2label_dict[gt_skel.getComment()]
            consensi_celltype_label[skel_ID] = class_nb
            print class_nb
        except KeyError:
            print "Didnt find", gt_skel.getComment()
    # for ii, skel in enumerate(gt_skels):
    #     print "%d/%d" % (ii, len(gt_skels))
    #     for kk, skel_inner_path in enumerate(consensi_skel):
    #         if skel_inner_path in did_skels:
    #             continue
    #         skel_inner = load_ordered_mapped_skeleton(skel_inner_path)[0]
    #         if similarity_check(skel, skel_inner):
    #             did_skels.append(skel_inner_path)
    #             skel_ID = get_skelID_from_path(skel_inner.filename)
    #             consensi_celltype_label[skel_ID] = skel.getComment()
    #             break
    #         if kk == len(consensi_skel):
    #             print "Didnt find gt value for %s", skel_inner.filename
    write_obj2pkl(consensi_celltype_label, '/lustre/pschuber/gt_cell_types/'
                                           'consensi_celltype_labels2.pkl')


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
            #print "Didnt find", skel_ids[i]
            #class_nb = -1
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
    return arr(skeleton_ids)[ixs], arr(skeleton_feats)[ixs]#[:, (11, 3, 22, 55,
        # 21, 29, 10, 6, 14, 9, 17, 0, 47, 12, 8, 18, 7,
        # 13, 52, 1, 19, 44, 39, 5, 4, 51, 33, 46)]


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
    feat_names = np.load(gt_path + 'feat_names.npy')#[(11, 3, 22, 55,
        # 21, 29, 10, 6, 14, 9, 17, 0, 47, 12, 8, 18, 7,
        # 13, 52, 1, 19, 44, 39, 5, 4, 51, 33, 46),]

    rf = joblib.load(gt_path + '/%s/%s.pkl' % (clf_used, clf_used))
    skeleton_ids, skeleton_feats = load_celltype_feats(gt_path)
    skel_type_probas = rf.predict_proba(skeleton_feats)
    proba_dict = {}
    print "# of feats and feat-names:", skeleton_feats.shape[1], len(feat_names)
    for ii, proba in enumerate(skel_type_probas):
        proba_dict[skeleton_ids[ii]] = proba

    proba_dict = load_pkl2obj(gt_path + '/wiring/celltype_proba_dict.pkl')
    # np.save(gt_path + '/wiring/skel_ids.npy', skel_ids)
    # np.save(gt_path + '/wiring/skel_type_probas.npy', skel_type_probas)

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
    feat_names = np.load(gt_path + 'feat_names.npy')#[(11, 3, 22, 55,
        # 21, 29, 10, 6, 14, 9, 17, 0, 47, 12, 8, 18, 7,
        # 13, 52, 1, 19, 44, 39, 5, 4, 51, 33, 46),]

    #feats = np.load(gt_path + 'wiring/skeleton_feats.npy')
    #skel_ids = np.load(gt_path + 'wiring/skel_ids.npy')
    #skel_type_probas = np.load('/lustre/pschuber/gt_cell_types/wiring/'
    #                           'skel_type_probas.npy')
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


def get_soma_tasks_names():
    soma_task_p = '/lustre/pschuber/soma_tracing_skels/'
    find_ids = ['241', '190', '31', '496']
    paths = get_filepaths_from_dir(soma_task_p)
    for path in paths:
        skels = au.loadj0126NML(path)
        if skels[0].comment in find_ids:
            print path


def write_wrong_predicted_celltypes():
    gt_path='/lustre/pschuber/gt_cell_types/'
    gt_dict = load_pkl2obj('/lustre/pschuber/gt_cell_types/'
                                           'consensi_celltype_labels2.pkl')
    proba_dict = proba_dict = load_pkl2obj(gt_path + '/wiring/celltype_proba_dict.pkl')
    for k, v in gt_dict.iteritems():
        if (v==3) and (v != np.argmax(proba_dict[k])):
            print k, v, np.argmax(proba_dict[k])


def write_classes2_3():
    for i in [2,3]:
        paths=find_cell_types_from_dict(cell_type=i)
        for p in paths:
            dir, fname = os.path.split(p)
            shutil.copyfile(p, '/lustre/pschuber/cell_types_analysis/rare_classes/'
                            '%s' % (fname))
            print "Writing", fname


def grid_search_rfc_params():
    gt_path='/lustre/pschuber/gt_cell_types/'
    X, Y = load_celltype_gt(gt_path, load_data=True)
    mean_l = []
    std_l = []
    params_l = []
    for weights in [None, 'balanced']:
        for max_feats in [0.2, 0.33, 0.4, 0.5, 0.66, int(np.sqrt(X.shape[1]))]:
            for nb_trees in np.arange(2000, 6000, 1000):
                vals = []
                clf = init_clf('rf', params={'n_estimators': nb_trees,
                                             'max_features': max_feats,
                                             'oob_score': True,
                                             'class_weight': weights})
                for i in range(5):
                    clf.fit(X, Y)
                    vals.append(clf.oob_score_)
                mean_l.append(np.mean(vals))
                std_l.append(np.std(vals))
                print "oob-score of %0.4f +- %0.4f" % (np.mean(vals), np.std(vals))
                print "Params:", [max_feats, nb_trees, weights]
                params_l.append([max_feats, nb_trees, weights])
    opt_oob = np.max(mean_l)
    opt_std = std_l[np.argmax(mean_l)]
    print "Found optimal params with oob-score of %0.4f +- %0.4f" % (opt_oob, opt_std)
    print params_l[np.argmax(mean_l)]


rf_params_nodewise = {'n_estimators': 2000, 'oob_score': True, 'n_jobs': -1,
             'class_weight': 'balanced', 'max_features': 'auto'}


def feature_context_eval_nodewise(fig_path='/lustre/pschuber/figures/'
                                  'cell_types_nodewise/', load_data=True):
    """
    Calculates performance of nodewise celltype classification in given certain
    window ranges (250, 130000, 52)
    :param fig_path: path where to store results.
    :param load_data: if false recomputes the leave-one-out procedure
    :return: None
    """
    prec_den = []
    prec_ax = []
    prec_soma = []
    rec_den = []
    rec_ax = []
    rec_soma = []
    cell_acc_total = []
    fs_den = []
    fs_ax = []
    fs_soma = []
    dists = np.linspace(250, 13000, 52)
    for dist in dists:
        print "Processing Context Range:", dist
        try:
            prec, rec, fs, cell_acc = first_celltypes_nodewise_eval(
            load_data=load_data, fig_path='/lustre/pschuber/'
            'figures/cell_types_nodewise/%d/' % dist, dist=dist)
        except:
            print "Skipped", dist
            continue
        prec_den.append(prec[0])
        prec_ax.append(prec[1])
        prec_soma.append(prec[2])
        rec_den.append(rec[0])
        rec_ax.append(rec[1])
        rec_soma.append(rec[2])
        fs_den.append(fs[0])
        fs_ax.append(fs[1])
        fs_soma.append(fs[2])
        cell_acc_total += [cell_acc]
    dists = np.array(dists, dtype=np.float) / 1000
    plt.figure()
    plt.plot(dists, cell_acc_total)
    plt.xlabel(u'Context Range [µm]')
    plt.ylabel('Accuracy of Cell Prediction')
    plt.savefig(fig_path+'/cell_pred_accuracy_context_eval.png')

    plot_pr([prec_den, prec_ax, prec_soma], [rec_den, rec_ax, rec_soma],
            legend_labels=['Dendrite', 'Axon', 'Soma'], title='')
    plt.savefig(fig_path+'/pr_nodewise_pred_context_eval.png')

    plt.figure()
    plot_pr([fs_den, fs_ax, fs_soma], [np.array(dists)*2]*3,
            legend_labels=['Dendrite', 'Axon', 'Soma'],
            colorVals=['0.6', [0.841, 0.138, 0.133, 1.], '0.32'],
            xlabel=u'Context Range [µm]', r_x=[0, np.max(dists) * 2],
            save_path=fig_path+'/node_pred_fscore_context_eval.png',
            ylabel='F-Score', l_pos="lower right")


def first_celltypes_nodewise_eval(load_data=True, dist=6000,
                                  fig_path='/lustre/pschuber/figures/'
                                           'cell_types_nodewise/',
                                  recompute=False, plot=True):
    """
    evaluates performance of nodewise celltype classification for a single
    context range. The latter defines the window in which to calculate context
    features.
    :param load_data: load data from stored files
    :param dist: context range in nm (total range is 2*dist)
    :param fig_path: destination folder results
    :return:
    """
    X_cells, Y_cells = load_gt_celltype_nodewise(max_nn_dist=dist, recompute=recompute)
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
    if not load_data:
        print "Predicting %d cells." % len(X_cells)
        print "Feautre dimensionality:", X_cells[0].shape[1]
        res = []
        loo = cross_validation.LeaveOneOut(len(X_cells))
        node_proba_res = []
        node_gt_res = []
        node_pred_res = []
        # llo with whole cells
        for train_ixs, test_ixs in loo:
            X_train = arr([item for sublist in arr(X_cells)[train_ixs] for
                           item in sublist.tolist()])
            y_train = arr([item for sublist in arr(Y_cells)[train_ixs] for
                           item in sublist.tolist()])
            rf = RandomForestClassifier(**rf_params_nodewise)
            rf.fit(X_train, y_train)
            X_single_cell = X_cells[test_ixs][0]
            node_pred = rf.predict(X_single_cell)
            node_pred_res += node_pred.tolist()
            node_proba = rf.predict_proba(X_single_cell)
            node_proba_res += node_proba.tolist()
            node_gt_res += Y_cells[test_ixs][0].tolist()
            cell_pred = cell_classification(node_pred)
            res.append(Y_cells[test_ixs][0][0] == cell_pred)
            print test_ixs, "\t",\
                np.sum(node_pred == Y_cells[test_ixs][0])/float(len(node_pred)),\
                cell_pred, Y_cells[test_ixs][0][0]
        node_proba_res = arr(node_proba_res)
        node_gt_res = arr(node_gt_res)
        node_pred_res = arr(node_pred_res)
        np.save(fig_path+'cell_pred_gt_node_proba_res.npy', node_proba_res)
        np.save(fig_path+'cell_pred_gt_node_gt_res.npy', node_gt_res)
        np.save(fig_path+'cell_pred_gt_node_pred_res.npy', node_pred_res)
        np.save(fig_path+'cell_pred_gt_res.npy', res)
    else:
        node_proba_res = np.load(fig_path+'cell_pred_gt_node_proba_res.npy')
        node_gt_res = np.load(fig_path+'cell_pred_gt_node_gt_res.npy')
        node_pred_res = np.load(fig_path+'cell_pred_gt_node_pred_res.npy')
        res = np.load(fig_path+'cell_pred_gt_res.npy')
    print "Support:", np.sum(node_gt_res == 0), np.sum(node_gt_res == 1),\
            np.sum(node_gt_res == 2)
    y_test_0 = np.zeros_like(node_gt_res)
    y_test_0[node_gt_res == 0] = 1
    y_test_1 = np.zeros_like(node_gt_res)
    y_test_1[node_gt_res == 1] = 1
    y_test_2 = np.zeros_like(node_gt_res)
    y_test_2[node_gt_res == 2] = 1
    y_test_3 = np.zeros_like(node_gt_res)
    y_test_3[node_gt_res == 3] = 1
    prec, rec, tr = precision_recall_curve(y_test_0, node_proba_res[:, 0])
    fs = fscore(rec, prec)

    if plot:
        label_strings_dict = {}
        label_strings_dict[0] = "Excitatory Axons"
        label_strings_dict[1] = "Medium spiny Neurons"
        label_strings_dict[2] = "Pallidal-like Neurons"
        label_strings_dict[3] = "Inhibitory Interneurons"
        # find threshold for precision value
        tresh4prec = .9
        prec_l = []
        rec_l = []
        t_l = []
        fopt_l = []
        topt_l = []
        prec, rec, t = precision_recall_curve(y_test_0, node_proba_res[:, 0])
        prec_l.append(prec)
        rec_l.append(rec)
        f_scores = fscore(rec, prec)
        fopt_l.append(np.max(f_scores))
        topt_l.append(t[np.argmax(f_scores)])
        t_l.append(t)
        print "[0] Threshold for precision bigger than %0.2f:" % tresh4prec,\
            t[np.argmax(tresh4prec <= prec)]
        plt.close()
        prec, rec, t = precision_recall_curve(y_test_1, node_proba_res[:, 1])
        prec_l.append(prec)
        rec_l.append(rec)
        f_scores = fscore(rec, prec)
        fopt_l.append(np.max(f_scores))
        topt_l.append(t[np.argmax(f_scores)])
        t_l.append(t)
        print "[1]Threshold for precision bigger than %0.2f:" % tresh4prec,\
            t[np.argmax(tresh4prec <= prec)]

        prec, rec, t = precision_recall_curve(y_test_2, node_proba_res[:, 2])
        prec_l.append(prec)
        rec_l.append(rec)
        f_scores = fscore(rec, prec)
        fopt_l.append(np.max(f_scores))
        topt_l.append(t[np.argmax(f_scores)])
        t_l.append(t)
        print "[2]Threshold for precision bigger than %0.2f:" % tresh4prec,\
            t[np.argmax(tresh4prec <= prec)]

        prec, rec, t = precision_recall_curve(y_test_3, node_proba_res[:, 3])
        prec_l.append(prec)
        rec_l.append(rec)
        f_scores = fscore(rec, prec)
        fopt_l.append(np.max(f_scores))
        topt_l.append(t[np.argmax(f_scores)])
        t_l.append(t)
        print "[3]Threshold for precision bigger than %0.2f:" % tresh4prec,\
            t[np.min((np.argmax(tresh4prec <= prec), len(t)-1))]

        new_colors = ["0.1", "0.3", "0.5", "0.82"]
        plot_pr(prec_l, rec_l, title='', legend_labels=label_strings_dict.values(),
                r=[0., 1.01], r_x=[0., 1.01], colorVals=new_colors, ls=20)
        plt.savefig('/lustre/pschuber/figures/cell_types_nodewise/'
                    'clf_all_big.png', dpi=300)
        plt.close()
        plot_pr(prec_l, rec_l, title='', legend_labels=label_strings_dict.values(),
                r=[0.5, 1.01], r_x=[0.5, 1.01], colorVals=new_colors, ls=20)
        plt.savefig('/lustre/pschuber/figures/cell_types_nodewise/'
                    'clf_all.png', dpi=300)
        plt.close()


    print "optimal threshold for 0 at %0.3f with fs %0.3f" % \
          (tr[np.argmax(fs)], np.max(fs))
    # plot_pr(prec, rec, title='ExAx Classification')
    # plt.savefig(fig_path+'node_clf_pr_0.png', dpi=300)
    # plt.close()
    prec, rec, tr = precision_recall_curve(y_test_1, node_proba_res[:, 1])
    fs = fscore(rec, prec)
    print "optimal threshold for 1 at %0.3f with fs %0.3f" % \
          (tr[np.argmax(fs)], np.max(fs))
    print "fs at threshold=0.5", fs[np.argmax(tr>0.5)]
    # plot_pr(prec, rec, title='MSN Classification')
    # plt.savefig(fig_path+'node_clf_pr_msn.png', dpi=300)
    # plt.close()
    prec, rec, tr = precision_recall_curve(y_test_2, node_proba_res[:, 2])
    fs = fscore(rec, prec)
    print "optimal threshold for 2 at %0.3f with fs %0.3f" % \
          (tr[np.argmax(fs)], np.max(fs))
    # plot_pr(prec, rec, title='GP Classification')
    # plt.savefig(fig_path + 'node_clf_pr_gp.png', dpi=300)
    # plt.close()
    prec, rec, tr = precision_recall_curve(y_test_3, node_proba_res[:, 3])
    fs = fscore(rec, prec)
    print "optimal threshold for 3 at %0.3f with fs %0.3f" % \
          (tr[np.argmax(fs)], np.max(fs))
    # plot_pr(prec, rec, title='FS Classification')
    # plt.savefig(fig_path + 'node_clf_pr_fs.png', dpi=300)
    # plt.close()
    prec, rec, fs, supp = precision_recall_fscore_support(node_gt_res, node_pred_res)
    print 'Multiclass Classification', prec, rec, fs, supp
    print "Weighted F-score mean:", np.sum(fs * supp) / np.sum(supp)
    print 'Multiclass Accuracy:', accuracy_score(node_gt_res, node_pred_res)
    cell_acc = np.sum(res) / float(len(res))
    print "Cell classification acc. (%d cells):\t%0.4f" % (len(res), cell_acc)
    return prec, rec, fs, cell_acc


def load_gt_celltype_nodewise(recompute=False, max_nn_dist=6000):
    gt_path = '/lustre/pschuber/gt_cell_types_nodewise/'
    if not os.path.isfile(gt_path+'nodeswise_gt_x_%d.npy' % max_nn_dist)\
            or recompute:
        print "Recomputing features for nodewise celltype prediction." \
              "Saving it to %s." % gt_path
        _, cell_labels, skel_ids = load_celltype_gt(return_ids=True)
        skel_paths = get_paths_of_skelID(skel_ids, "/lustre/pschuber/"
                                        "mapped_soma_tracings/nml_obj/")
        res = start_multiprocess(feat_calc_helper, zip(skel_paths, cell_labels),
                                 nb_cpus=16, debug=False)
        x_total = []
        y_total = []
        for el in res:
            x_total.append(el[0])
            y_total.append(el[1])
        np.save(gt_path+'nodeswise_gt_x_%d.npy' % max_nn_dist, arr(x_total))
        np.save(gt_path+'nodeswise_gt_y_%d.npy' % max_nn_dist, arr(y_total))
    else:
        x_total = np.load(gt_path+'nodeswise_gt_x_%d.npy' % max_nn_dist)
        y_total = np.load(gt_path+'nodeswise_gt_y_%d.npy' % max_nn_dist)
    return x_total, y_total


def feat_calc_helper(params):
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
                               nearby_node_list, obj_dict, scaling, is_az=False):
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
    # TODO: add az sign.
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
