import numpy as np
from multiprocessing import Pool, Manager, cpu_count, Process
import multi_proc.pool
import time
from sklearn.externals import joblib
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.svm import SVC
import pylab as pl
import pandas as pd
import zipfile
import tempfile
from collections import Counter
import shutil
import os
import matplotlib.patches as patches
from sklearn.decomposition import PCA

__author__ = 'pschuber'

rf_params = {'n_estimators': 2000, 'oob_score': True, 'n_jobs': -1,
             'class_weight': 'balanced', 'max_features': 'auto'}

svm_params = {'kernel': 'rbf', 'class_weight': 'balanced', 'probability': True,
              'C': 2.0, 'gamma': 0.3, 'cache_size': 2000}

ada_params = {'n_estimators': 100, 'learning_rate': 1.}

lr_params = {'class_weight': 'balanced', 'solver': 'lbfgs', 'penalty': 'l2',
             'multi_class': 'multinomial'}


def init_clf(clf_used, params=None):
    if params is not None:
        params_used = params
    elif clf_used == 'svm':
        params_used = svm_params
    elif clf_used == 'ada_boost':
        params_used = rf_params
    elif clf_used == 'lr':
        params_used = lr_params
    else:
        params_used = rf_params
    #print "Initializing %s-classifier with parameters:" % clf_used
    #print params_used
    if clf_used == 'svm':
        clf = SVC(**params_used)
    elif clf_used == 'ada_boost':
        rf = RandomForestClassifier(**rf_params)
        clf = AdaBoostClassifier(base_estimator=rf, **params_used)
    elif clf_used == 'lr':
        clf = LogisticRegressionCV(**params_used)
    else:
        clf = RandomForestClassifier(**params_used)
    return clf


def load_rfcs(rf_axoness_p, rf_spiness_p):
    """
    Loads pickeled Random Forest Classifier for axoness and spiness. If path is
    not valid returns None
    :param rf_axoness_p: str Path to pickeled axonnes rf directory
    :param rf_spiness_p: str Path to pickeled spiness rf directory
    :return: RFC axoness, spiness
    """
    if os.path.isfile(rf_axoness_p):
        rfc_axoness = joblib.load(rf_axoness_p)
        print "Found RFC for axoness. SkeletonNodes will contain axoness " \
              "probability."
    else:
        rfc_axoness = None
        print "WARNING: Could not predict axoness of SkeletonNodes. " \
              "Pretrained RFC file not found."
    if os.path.isfile(rf_spiness_p):
        rfc_spiness = joblib.load(rf_spiness_p)
        print "Found RFC for spiness. SkeletonNodes will contain spiness " \
              "probability."
    else:
        rfc_spiness = None
        print "WARNING: Could not predict spiness of SkeletonNodes. " \
              "Pretrained RFC file not found."
    return rfc_axoness, rfc_spiness


def save_train_clf(X, y, clf_used, dir_path, use_pca=False, params=None):
    """
    Train classifier specified by clf_used to dir_path. Train with features X
    and labels y
    :param X: arr features
    :param y: arr labels
    :param clf_used: 'rf' or 'svm' for RandomForest or SupportVectorMachine,
     respectively
    :param dir_path: directory where to save pkl files of clf
    """
    print "Start saving procedure for clf %s with features of shape %s." %\
          (clf_used, X.shape)
    if use_pca:
        old_dim = X.shape[1]
        pca = PCA(n_components=0.99)
        X = pca.fit_transform(X)
        print pca.explained_variance_ratio_
        print "Reduced feature space dimension %d, instead of %d" % (X.shape[1],
                                                                     old_dim)
    clf = init_clf(clf_used, params=params)
    try:
        clf.fit(X, y)
    except ValueError:
        print "Found nans in features, converting to number."
        X = np.nan_to_num(X)
        clf.fit(X, y)
    clf_dir = dir_path + '%s/' % clf_used
    if os.path.exists(clf_dir):
        shutil.rmtree(clf_dir)
    if clf_used == 'rf':
        print "Random Forest oob score:", clf.oob_score_
    os.makedirs(clf_dir)
    joblib.dump(clf, clf_dir + '/%s.pkl' % clf_used)
    print "%s-Classifier written to %s" % (clf_used, clf_dir)


def three_liner(r_az, p_az, r_p4_az, p_p4_az, pos_az, pos_p4_az,
                true_az, true_p4_az, true_p4=0.018):
    tot_p4 = 0#3989
    tot_az = 3456
    tot_p4_az = 4027


    r = (r_az*tot_az*true_az + r_p4_az*tot_p4_az*true_p4_az) / \
        (tot_p4*true_p4 + tot_az*true_az + tot_p4_az*true_p4_az)
    p = (p_az*tot_az*pos_az + p_p4_az*tot_p4_az*pos_p4_az) / \
        (tot_az*pos_az + tot_p4_az*pos_p4_az)

    # print "Merge Approach"
    # print "precision:", p
    # print "recall:", r
    # print "fscore:", fscore(r, p)
    return r, p


def novel_multiclass_prediction(f_scores, thresholds, probs):
    pred = -1 * np.ones((len(probs), ))
    tprobs = np.array(probs > thresholds, dtype=np.int)
    for i in range(len(probs)):
        if np.sum(tprobs[i]) != 0:
            pred[i] = np.argmax(f_scores * tprobs[i])
    return pred


def fscore(rec, prec, beta=1.):
    prec = np.array(prec)
    rec = np.array(rec)
    f_score = (1. + beta**2) * (prec * rec) / (beta**2 * prec + rec)
    return np.nan_to_num(f_score)


def plot_corr(x, y, title='', xr=[-1, -1], yr=[-1, -1], save_path=None, nbins=5,
              xlabel='Size x', ylabel='Size y'):
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots()
    fig.patch.set_facecolor('white')
    ax.tick_params(axis='x', which='major', labelsize=16, direction='out',
                    length=8, width=2,  right="off", top="off", pad=10)
    ax.tick_params(axis='y', which='major', labelsize=16, direction='out',
                    length=8, width=2,  right="off", top="off", pad=10)

    ax.tick_params(axis='x', which='minor', labelsize=12, direction='out',
                    length=8, width=2, right="off", top="off", pad=10)
    ax.tick_params(axis='y', which='minor', labelsize=12, direction='out',
                    length=8, width=2, right="off", top="off", pad=10)

    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # plt.locator_params(axis='x', nbins=nbins)
    # plt.locator_params(axis='y', nbins=nbins)
    # plt.title(title)
    if not -1 in xr:
        plt.xlim(xr)
    if not -1 in yr:
        plt.ylim(yr)
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.gcf().subplots_adjust(left=0.15)
    plt.xlabel(xlabel, fontsize=18)
    plt.ylabel(ylabel, fontsize=18)
    ax.set_xscale("log")
    ax.set_yscale("log")

    # plt.plot(recall, precision, lw=3)
    # pl.title('Precision Recall Curve')
    # plt.show(block=False)

    plt.scatter(x, y, s=1, c="0.35")
    if save_path is None:
        plt.show(block=False)
    else:
        plt.savefig(save_path, dpi=600)


def plot_pr(precision, recall, title='', r=[0.67, 1.01], legend_labels=None,
            save_path=None, nbins=5, colorVals=None,
            xlabel='Recall', ylabel='Precision', l_pos="lower left",
            legend=True, r_x=[0.67, 1.01], ls=22):
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots()
    fig.patch.set_facecolor('white')
    ax.tick_params(axis='x', which='major', labelsize=ls, direction='out',
                    length=4, width=3,  right="off", top="off", pad=10)
    ax.tick_params(axis='y', which='major', labelsize=ls, direction='out',
                    length=4, width=3,  right="off", top="off", pad=10)

    ax.tick_params(axis='x', which='minor', labelsize=ls, direction='out',
                    length=4, width=3, right="off", top="off", pad=10)
    ax.tick_params(axis='y', which='minor', labelsize=ls, direction='out',
                    length=4, width=3, right="off", top="off", pad=10)

    ax.spines['left'].set_linewidth(3)
    ax.spines['bottom'].set_linewidth(3)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.locator_params(axis='x', nbins=nbins)
    plt.locator_params(axis='y', nbins=nbins)
    plt.title(title)
    if not -1 in r:
        plt.xlim(r_x)
        plt.ylim(r)

    plt.xlabel(xlabel, fontsize=ls)
    plt.ylabel(ylabel, fontsize=ls)

    # plt.plot(recall, precision, lw=3)
    # pl.title('Precision Recall Curve')
    # plt.show(block=False)
    plt.tight_layout()
    if isinstance(recall, list):
        if colorVals is None:
            colorVals = [[0.171, 0.485, 0.731, 1.],
                         [0.175, 0.585, 0.301, 1.],
                         [0.841, 0.138, 0.133, 1.]]
        if len(colorVals) < len(recall):
            colorVals += ["0.35"] * (len(recall) - len(colorVals))
        if len(colorVals) > len(recall):
            colorVals = ["0.35", "0.7"]
        if legend_labels is None:
            legend_labels = ["Mitochondria", "Vesicle Clouds", "Synaptic Junctions"]
        handles = []
        for ii in range(len(recall)):
            handles.append(patches.Patch(color=colorVals[ii], label=legend_labels[ii]))
            plt.plot(recall[ii], precision[ii], lw=4, c=colorVals[ii])
        if legend:
            plt.legend(handles=handles, loc=l_pos, frameon=False, prop={'size': ls})
    else:
        plt.plot(recall, precision, lw=4, c="0.35")
    if save_path is None:
        plt.show(block=False)
    else:
        plt.savefig(save_path, dpi=300)


def feature_importance(rf, save_path=None):
    """
    Plots feature importance of sklearn RandomForest.
    :param rf: RandomForestClassifier of sklearn
    """
    importances = rf.feature_importances_
    nb = len(importances)
    tree_imp = [tree.feature_importances_ for tree in rf.estimators_]
    print "Print feature importance of rf with %d trees." % len(tree_imp)
    std = np.std(tree_imp, axis=0) / np.sqrt(len(tree_imp))
    indices = np.argsort(importances)[::-1]
    # Print the feature ranking
    print("Feature ranking:")
    for f in range(nb):
        print("%d. feature %d (%f)" % (f + 1, indices[f],
                                       importances[indices[f]]))

    # Plot the feature importances of the forest
    pl.figure()
    pl.title("Feature importances")
    pl.bar(range(nb), importances[indices],
           color="r", yerr=std[indices], align="center")
    pl.xticks(range(nb), indices)
    pl.xlim([-1, nb])
    if save_path is not None:
        pl.savefig(save_path)
    pl.close()


def write_feat2csv(fpath, feat_arr, feat_names=[]):
    """
    Writes array with column names to csv file at fpath
    :param fpath: str Path to file
    :param feat_arr: NumpyArray 2D shape
    :param feat_names: list of strings, same length as  length of last array
    dimension
    :return:
    """
    if feat_names is [] or (len(feat_names) != feat_arr.shape[1]):
        df = pd.DataFrame(feat_arr)
        print feat_arr.shape, len(feat_names)
    else:
        df = pd.DataFrame(feat_arr, columns=feat_names)
    df.to_csv(fpath)
    return


def load_csv2feat(fpath, property='axoness'):
    """
    Load csvfile from kzip and return numpy array and list of header names.
    List line is supposed to be the probability prediction.
    :param fpath: str Source file path
    :return: NumpyArray, List of strings
    """
    zf = zipfile.ZipFile(fpath, 'r')
    df = pd.DataFrame()
    for filename in ['%s_feat.csv' % property]:
        temp = tempfile.TemporaryFile()
        temp.write(zf.read(filename))
        temp.seek(0)
        df = pd.DataFrame.from_csv(temp)
    return df.as_matrix(), df.columns.values.tolist()


def loo_proba(x, y, clf_used='rf', use_pca=False, params=None):
    print "Performing LOO with %s and %d features. Using PCA: %s" % \
          (clf_used, x.shape[1], str(use_pca))
    if use_pca:
        old_dim = x.shape[1]
        pca = PCA(n_components=0.999)
        x = pca.fit_transform(x)
        print pca.explained_variance_ratio_
        print "Reduced feature space dimension %d, instead of %d" % (x.shape[1],
                                                                     old_dim)
    nans_in_X = np.sum(np.isnan(x))
    if nans_in_X > 0:
        print np.where(np.isnan(x))
        print "Found %d nans in features, converting to number." % nans_in_X
        x = np.nan_to_num(x)
    loo = cross_validation.LeaveOneOut(len(x))
    shape = (len(x), len(list(set(y))))
    prob = np.zeros(shape, dtype=np.float)
    pred = np.zeros((len(x), 1), dtype=np.int)
    cnt = 0
    print "rf params:", rf_params
    for train_ixs, test_ixs in loo:
        x_train = x[train_ixs]
        x_test = x[test_ixs]
        y_train = y[train_ixs]
        clf = init_clf(clf_used, params)
        clf.fit(x_train, y_train)
        prob[cnt] = clf.predict_proba(x_test)
        pred[cnt] = clf.predict(x_test)
        np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
        if pred[cnt] == y[test_ixs]:
            print test_ixs, "\t", prob[cnt], pred[cnt], y[test_ixs]
        else:
            print test_ixs, "\t", prob[cnt], pred[cnt], y[test_ixs], "\t WRONG"
        cnt += 1
    # feature_importance(rf)
    return prob, pred


def start_multiprocess(func, params, debug=False, nb_cpus=None):
    """

    :param func:
    :param params:
    :return:
    """
    # found NoDaemonProcess on stackexchange by Chris Arndt - enables
    # multprocessed grid search with gpu's
    class NoDaemonProcess(Process):
        # make 'daemon' attribute always return False
        def _get_daemon(self):
            return False

        def _set_daemon(self, value):
            pass
        daemon = property(_get_daemon, _set_daemon)

    # We sub-class multi_proc.pool.Pool instead of multi_proc.Pool
    # because the latter is only a wrapper function, not a proper class.
    class MyPool(multi_proc.pool.Pool):
        Process = NoDaemonProcess
    if nb_cpus is None:
        nb_cpus = max(cpu_count() - 2, 1)
    if debug:
        nb_cpus = 1
    print "Computing %d parameters with %d cpus." % (len(params), nb_cpus)
    start = time.time()
    if not debug:
        pool = MyPool(nb_cpus)
        result = pool.map(func, params)
        pool.close()
        pool.join()
    else:
        result = map(func, params)

    print "\nTime to compute grid:", time.time() - start
    return result


def cell_classification(node_pred):
    """
    :param node_pred: array of integers
    :return: maximum occuring integer in array
    """
    if len(node_pred) == 0:
        return np.zeros(0)
    counter = Counter(node_pred)
    return counter.most_common()[0][0]
