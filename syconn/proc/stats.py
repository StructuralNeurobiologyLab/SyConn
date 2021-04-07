# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max-Planck-Institute of Neurobiology, Munich, Germany
# Authors: Philipp Schubert, Joergen Kornfeld
import os
import time

import numpy as np

from . import log_proc

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.ticker as ticker
from sklearn.metrics import precision_recall_curve, classification_report, precision_recall_fscore_support, accuracy_score, \
    average_precision_score
from sklearn.manifold import TSNE as TSNE_sc
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import label_binarize
import joblib
import matplotlib.patches as mpatches

# required for readthedocs build
from ..handler.basics import kd_factory, load_pkl2obj, write_obj2pkl
from ..handler.config import Config, DynConfig
from ..reps.segmentation import SegmentationDataset

try:
    import seaborn as sns
except ImportError:
    pass


def model_performance(proba, labels, model_dir=None, prefix="", n_labels=3,
                      fscore_beta=1, target_names=None, add_text=''):
    header = "-------------------------------\n\t\t%s\n" % prefix
    if target_names is None:
        target_names = ["Dendrite", "Axon", "Soma"]
    all_prec, all_rec = [], []
    header += classification_report(labels, np.argmax(proba, axis=1), labels=np.arange(len(target_names)),
                                    digits=4, target_names=target_names)
    header += "acc.: %0.4f" % accuracy_score(labels, np.argmax(proba, axis=1))
    header += "\n-------------------------------\n"
    log_proc.info(header)
    plot_pr(all_prec, all_rec, r=[0.6, 1.01], legend_labels=target_names)
    if model_dir is not None:
        os.makedirs(model_dir, exist_ok=True)
        text_file = open(model_dir + '/prec_rec_%s.txt' % prefix, "w")
        text_file.write(header)
        text_file.write(add_text)
        text_file.close()
        prec, rec, fs, supp = precision_recall_fscore_support(labels, np.argmax(proba, axis=1))
        np.save(model_dir + '/prec_rec_%s.npy' % prefix, [prec, rec, fs])
        plt.savefig(model_dir + '/prec_rec_%s.png' % prefix)
    plt.close()


def model_performance_predonly(y_pred, y_true, model_dir=None, prefix="",
                               target_names=None, labels=None):
    y_pred = np.array(y_pred, dtype=np.int32)
    y_true = np.array(y_true, dtype=np.int32)
    header = "----------------------------------------------------\n\t\t" \
             "%s\n" % prefix
    if target_names is None:
        target_names = ["Dendrite", "Axon", "Soma"]
    header += classification_report(y_true, y_pred, digits=4,
                                    target_names=target_names, labels=labels)
    header += "acc.: {:.4f} -- {} wrongly predicted samples." \
              "".format(accuracy_score(y_true, y_pred), np.sum(y_true != y_pred))
    header += "\n-------------------------------------------------\n"
    log_proc.info(header)
    if model_dir is not None:
        text_file = open(model_dir + '/prec_rec_%s.txt' % prefix, "w")
        text_file.write(header)
        text_file.close()
        # prec, rec, fs, supp = precision_recall_fscore_support(labels, pred)
        # np.save(model_dir + '/prec_rec_%s.npy' % prefix, [prec, rec, fs])
    plt.close()


def hist(vals, labels=None, dest_path=None, axis_labels=None, x_lim=None,
         y_lim=None, y_log_scale=False, ls=22, color=None, **kwargs):
    sns.set_style("white")
    fig, ax = plt.subplots()
    fig.patch.set_facecolor('white')
    ax.tick_params(axis='x', which='major', labelsize=ls - 4, direction='out',
                   length=4, width=3, right="off", top="off", pad=10)
    ax.tick_params(axis='y', which='major', labelsize=ls - 4, direction='out',
                   length=4, width=3, right="off", top="off", pad=10)
    #
    # ax.tick_params(axis='x', which='minor', labelsize=ls, direction='out',
    #                 length=4, width=3, right="off", top="off", pad=10)
    # ax.tick_params(axis='y', which='minor', labelsize=ls, direction='out',
    #                 length=4, width=3, right="off", top="off", pad=10)

    ax.spines['left'].set_linewidth(3)
    ax.spines['bottom'].set_linewidth(3)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    if not "norm_hist" in kwargs:
        norm_hist = False
    else:
        norm_hist = kwargs["norm_hist"]
        del kwargs["norm_hist"]
    r = (np.concatenate(vals).min(), np.concatenate(vals).max())
    if x_lim is not None:
        r = x_lim
    if labels is None:
        sns.distplot(vals, hist=True, rug=False, norm_hist=False,
                     hist_kws={"range": r}, color=color, **kwargs)
    else:
        if color is None:
            color = [None] * len(vals)
        for i in range(len(vals)):
            sns.distplot(vals[i], hist=True, rug=False, label=labels[i],
                         kde=False, norm_hist=norm_hist, color=color[i],
                         hist_kws={"range": r}, **kwargs)
        plt.legend(prop={'size': ls})
    if axis_labels is not None:
        plt.xlabel(axis_labels[0], fontsize=ls)
        plt.ylabel(axis_labels[1], fontsize=ls)
    # plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    # plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(30))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(20))
    if x_lim is not None:
        plt.xlim(x_lim)
    if y_lim is not None:
        plt.ylim(y_lim)
    if y_log_scale:
        plt.yscale("log")
    sns.despine()
    plt.tight_layout()
    if dest_path is not None:
        plt.savefig(dest_path, dpi=400)
        plt.close()
    else:
        plt.show()


def fscore(rec, prec, beta=1.):
    """Calculates f-score with beta value

    Parameters
    ----------
    rec : np.array
        recall
    prec : np.array
        precision
    beta : float
        weighting of precision

    Returns
    -------
    np.array
        f-score
    """
    prec = np.array(prec)
    rec = np.array(rec)
    f_score = (1. + beta ** 2) * (prec * rec) / (beta ** 2 * prec + rec)
    return np.nan_to_num(f_score)


def array2xls(dest_p, arr):
    import xlsxwriter
    workbook = xlsxwriter.Workbook(dest_p)
    worksheet = workbook.add_worksheet()
    col = 0
    for row, data in enumerate(arr):
        worksheet.write_row(row, col, data)
    workbook.close()


def plot_pr(precision, recall, title='', r=[0.67, 1.01], legend_labels=None,
            save_path=None, nbins=5, colorVals=None,
            xlabel='Recall', ylabel='Precision', l_pos="lower left",
            legend=True, r_x=[0.67, 1.01], ls=22, xtick_labels=()):
    fig, ax = plt.subplots()
    fig.patch.set_facecolor('white')
    ax.tick_params(axis='x', which='major', labelsize=ls, direction='out',
                   length=4, width=3, right="off", top="off", pad=10)
    ax.tick_params(axis='y', which='major', labelsize=ls, direction='out',
                   length=4, width=3, right="off", top="off", pad=10)

    ax.tick_params(axis='x', which='minor', labelsize=ls, direction='out',
                   length=4, width=3, right="off", top="off", pad=10)
    ax.tick_params(axis='y', which='minor', labelsize=ls, direction='out',
                   length=4, width=3, right="off", top="off", pad=10)

    ax.spines['left'].set_linewidth(3)
    ax.spines['bottom'].set_linewidth(3)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # plt.locator_params(axis='x', nbins=nbins)
    # plt.locator_params(axis='y', nbins=nbins)
    plt.title(title)
    if not -1 in r:
        plt.xlim(r_x)
        plt.ylim(r)

    plt.xlabel(xlabel, fontsize=ls)
    plt.ylabel(ylabel, fontsize=ls)

    if save_path is not None:
        dest_dir, fname = os.path.split(save_path)
        if legend_labels is not None:
            ll = [["legend labels"] + list(legend_labels)]
        else:
            ll = [[]]
        array2xls(dest_dir + "/" + os.path.splitext(fname)[0] + ".xlsx",
                  ll + [["labels", xlabel, ylabel]] + [xtick_labels] + [precision] + [recall])

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
            plt.plot(recall[ii], precision[ii], "--o", lw=3, c=colorVals[ii], ms=8)
        if legend:
            plt.legend(handles=handles, loc=l_pos, frameon=False, prop={'size': ls})
    else:
        plt.plot(recall, precision, "--o", lw=3, ms=8, c="0.35")
    # ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    if len(xtick_labels) > 0:
        plt.xticks(recall, xtick_labels, rotation="vertical")
    plt.tight_layout()
    if save_path is None:
        plt.show(block=False)
    else:
        plt.savefig(save_path, dpi=600)


def cluster_summary(train_d, train_l, valid_d, valid_l, fold, prefix="", pca=None,
                    return_valid_pred=False):
    """
    Create clustering summary and save results to folder.

    Parameters
    ----------
    train_d :
    train_l :
    valid_d :
    valid_l :
    fold : str
        destination folder
    """
    if prefix == "celltype":
        target_names = ["EA", "MSN", "GP", "INT"]
        bin_labels = label_binarize(valid_l,
                                    classes=np.arange(len(target_names)))
    elif prefix == "axoness":
        target_names = ["dendrite", "axon", "soma"]
        bin_labels = label_binarize(valid_l,
                                    classes=np.arange(len(target_names)))
        bin_labels = np.hstack((bin_labels, 1 - bin_labels))
    elif prefix == "ctgt_v2":
        unique_c = np.unique(train_l)
        assert 6 not in unique_c and 9 not in unique_c  # hack because we have missing classes in
        # the GT
        train_l[train_l == 7] = 6  # convert TAN label to be 6
        train_l[train_l == 8] = 7  # convert INT labels to be 7
        str2int_label = dict(STN=0, DA=1, MSN=2, LMAN=3, HVC=4, GP=5, TAN=6, INT=7)
        int2str_label = {v: k for k, v in str2int_label.items()}
        target_names = [int2str_label[ii] for ii in range(8)]
        bin_labels = label_binarize(valid_l,
                                    classes=np.arange(len(target_names)))
    else:
        raise ValueError("Unknown data. Please add a valid prefix.")
    if pca is None:
        pca = PCA(n_components=3, whiten=True, random_state=0)
        pca.fit(train_d)

    summary_txt = ""
    # kNN classification with 3D latent space
    nbrs = KNeighborsClassifier(n_neighbors=5, algorithm='kd_tree', n_jobs=16,
                                weights="uniform")
    nbrs.fit(pca.transform(train_d), train_l.ravel())
    joblib.dump(nbrs, fold + "/knn_embedding_%s.sav" % prefix)
    pred = nbrs.predict_proba(pca.transform(valid_d))
    summary_txt += "3D latent space results for %s:" % prefix
    summary_txt += "Captured variance: {}".format(pca.explained_variance_ratio_)
    summary_txt += classification_report(valid_l, np.argmax(pred, axis=1),
                                         target_names=target_names, digits=4)
    plt.figure()
    colors = []
    for i in range(len(target_names)):
        precision, recall, thresh = precision_recall_curve(bin_labels[:, i], pred[:, i])
        auc = average_precision_score(bin_labels[:, i], pred[:, i])

        # Plot Precision-Recall curve
        lines, = plt.plot(recall, precision, lw=3, label='%s: %0.4f' % (target_names[i], auc))
        colors.append(lines.get_color())
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.05])
    plt.title('Precision-Recall')
    plt.legend(loc="lower left")
    plt.show(block=False)
    plt.savefig(fold + "/%s_valid_prec_rec_3d.png" % prefix)
    plt.close()

    # RFC performance on pca latent space
    rfc = RandomForestClassifier(n_estimators=1000, oob_score=True, class_weight="balanced")
    rfc.fit(train_d, train_l.ravel())
    pred = rfc.predict_proba(valid_d)
    summary_txt += "Complete latent space results for %s using RFC:" % prefix
    summary_txt += str(classification_report(valid_l, np.argmax(pred, axis=1),
                                             target_names=target_names, digits=4))

    # kNN classification for whole latent space
    nbrs = KNeighborsClassifier(n_neighbors=5, algorithm='kd_tree', n_jobs=16,
                                weights="uniform")
    nbrs.fit(train_d, train_l.ravel())
    pred = nbrs.predict_proba(valid_d)
    summary_txt += "Complete latent space results for %s using kNN:" % prefix
    summary_txt += str(classification_report(valid_l, np.argmax(pred, axis=1),
                                             target_names=target_names, digits=4))

    text_file = open(fold + '/%s_performance_summary.txt' % prefix, "w")
    text_file.write(summary_txt)
    text_file.close()

    colors = []
    plt.figure()
    for i in range(len(target_names)):
        precision, recall, thresh = precision_recall_curve(bin_labels[:, i],
                                                           pred[:, i])
        auc = average_precision_score(bin_labels[:, i], pred[:, i])

        # Plot Precision-Recall curve
        lines, = plt.plot(recall, precision, lw=3,
                          label='%s: %0.4f' % (target_names[i], auc))
        colors.append(lines.get_color())
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.05])
    plt.title('Precision-Recall')
    plt.legend(loc="lower left")
    plt.show(block=False)
    plt.savefig(fold + "/%s_valid_prec_rec.png" % prefix)
    plt.close()
    # plot densities in pca or tSNE latent space
    # if not os.path.isfile(fold + "/%s_train_kde_pca.png" % prefix):
    _ = projection_pca(valid_d, valid_l, fold + "/%s_valid_kde_pca.png" %
                       prefix, pca=pca, colors=colors, target_names=target_names)
    _ = projection_pca(train_d, train_l, fold + "/%s_train_kde_pca.png" %
                       prefix, pca=pca, colors=colors, target_names=target_names)
    tsne_kwargs = {"n_components": 2, "random_state": 0,
                   "perplexity": 20, "n_iter": 10000}
    # projection_tSNE(train_d, train_l, fold + "/%s_train_kde_tsne.png" % prefix,
    #                 colors=colors, target_names=target_names, **tsne_kwargs)
    if return_valid_pred:
        return pred


def projection_pca(ds_d, ds_l, dest_path, pca=None, colors=None, do_3d=True,
                   target_names=None):
    """

    Parameters
    ----------
    ds_d : np.array
        data in feature space, e.g. (#data, #feature)
    ds_l :
        sparse labels, i.e. (#data, 1)
    dest_path: str
        file name of plot
    pca: PCA
        prefitted PCA object to use to prject data of ds_d
    """
    log_proc.info("Starting pca visualisation.")
    # pca vis
    paper_rc = {'lines.linewidth': 1, 'lines.markersize': 1}
    sns.set_context(rc=paper_rc)
    if ds_l.ndim == 2:
        ds_l = ds_l[:, 0]
    nb_labels = np.unique(ds_l)
    if pca is None:
        pca = PCA(3, whiten=True, random_state=0)
        pca.fit(ds_d)
    res = pca.transform(ds_d)
    # density plot 1st and 2nd PC
    plt.figure()
    plt.ylabel('$Z_2$', fontsize=15)
    plt.xlabel('$Z_1$', fontsize=15)
    if colors is None:
        # colors = ["r", "g", "b", "y", "k"]
        if len(target_names) == 5:
            colors = ["r", "g", "b", "y", "k"]
        else:
            cmap = plt.cm.get_cmap("Accent", len(target_names))
            colors = [cmap(i) for i in range(len(target_names))]
    if target_names is None:
        target_names = ["%d" % i for i in nb_labels]
    for i in nb_labels:
        cur_pal = sns.light_palette(colors[i], as_cmap=True)
        d0, d1 = res[ds_l == i][:, 0], res[ds_l == i][:, 1]
        ax = sns.kdeplot(d0, d1, shade=False, cmap=cur_pal,
                         alpha=0.6, shade_lowest=False, gridsize=100)
        ax.patch.set_facecolor('white')
        ax.collections[0].set_alpha(0)
        plt.scatter(res[ds_l == i][:, 0], res[ds_l == i][:, 1],
                    s=1.2, lw=0, alpha=0.5, color=colors[i], label=target_names[i])
    handles = []
    for ii in range(len(target_names)):
        handles.append(mpatches.Patch(color=colors[ii], label=target_names[ii]))
    plt.legend(handles=handles, loc="best")
    plt.savefig(dest_path, dpi=300)
    plt.close()
    if do_3d:
        # density plot 1st and 3rd PC
        plt.figure()
        plt.ylabel('$Z_3$', fontsize=15)
        plt.xlabel('$Z_1$', fontsize=15)
        if colors is None:
            colors = ["r", "g", "b", "y", "k"]
        if target_names is None:
            target_names = ["%d" % i for i in nb_labels]
        for i in nb_labels:
            cur_pal = sns.light_palette(colors[i], as_cmap=True)
            d0, d2 = res[ds_l == i][:, 0], res[ds_l == i][:, 2]
            ax = sns.kdeplot(d0, d2, shade=False, cmap=cur_pal,
                             alpha=0.6, shade_lowest=False, gridsize=100)
            ax.patch.set_facecolor('white')
            ax.collections[0].set_alpha(0)
            plt.scatter(res[ds_l == i][:, 0], res[ds_l == i][:, 2],
                        s=1.2, lw=0, alpha=0.5, color=colors[i], label=target_names[i])
        handles = []
        for ii in range(len(target_names)):
            handles.append(mpatches.Patch(color=colors[ii], label=target_names[ii]))
        plt.legend(handles=handles, loc="best")
        plt.savefig(os.path.splitext(dest_path)[0] + "_2.png", dpi=300)
        plt.close()

        # density plot 2nd and 3rd PC
        plt.figure()
        plt.ylabel('$Z_3$', fontsize=15)
        plt.xlabel('$Z_2$', fontsize=15)
        if colors is None:
            colors = ["r", "g", "b", "y", "k"]
        if target_names is None:
            target_names = ["%d" % i for i in nb_labels]
        for i in nb_labels:
            cur_pal = sns.light_palette(colors[i], as_cmap=True)
            d1, d2 = res[ds_l == i][:, 1], res[ds_l == i][:, 2]
            ax = sns.kdeplot(d1, d2, shade=False, cmap=cur_pal,
                             alpha=0.6, shade_lowest=False, gridsize=100)
            ax.patch.set_facecolor('white')
            ax.collections[0].set_alpha(0)
            plt.scatter(res[ds_l == i][:, 1], res[ds_l == i][:, 2],
                        s=1.2, lw=0, alpha=0.5, color=colors[i], label=target_names[i])
        handles = []
        for ii in range(len(target_names)):
            handles.append(mpatches.Patch(color=colors[ii], label=target_names[ii]))
        plt.legend(handles=handles, loc="best")
        plt.savefig(os.path.splitext(dest_path)[0] + "_3.png", dpi=300)
        plt.close()
    return pca


def projection_tSNE(ds_d, ds_l, dest_path, colors=None, target_names=None,
                    do_3d=False, cmap_ident="prism", **tsne_kwargs):
    """

    Parameters
    ----------
    ds_d : np.array
        data in feature space, e.g. (#data, #feature)
    ds_l :
        sparse labels, i.e. (#data, 1)
    dest_path: str
        file name of plot
    pca: PCA
        prefitted PCA object to use to prject data of ds_d
    """
    # tsne vis
    log_proc.info("Starting tSNE visualisation.")
    paper_rc = {'lines.linewidth': 1, 'lines.markersize': 1}
    sns.set_context(rc=paper_rc)
    if ds_l.ndim == 2:
        ds_l = ds_l[:, 0]
    assert ds_l.ndim == 1
    nb_labels = np.unique(ds_l)
    tsne = TSNE_sc(**tsne_kwargs)
    tsne.fit(ds_d)
    while True:
        try:
            res = tsne.fit_transform(ds_d)
            break
        except MemoryError:
            log_proc.info("Downsampling data for tSNE visualization")
            ds_d = ds_d[::2]
            ds_l = ds_l[::2]

    # density plot
    plt.figure()
    plt.ylabel('$Z_2$', fontsize=15)
    plt.xlabel('$Z_1$', fontsize=15)
    if colors is None:
        # colors = ["r", "g", "b", "y", "k"]
        if len(target_names) == 5:
            colors = ["r", "g", "b", "y", "k"]
        else:
            cmap = plt.cm.get_cmap(cmap_ident, len(target_names))
            colors = [cmap(i) for i in range(len(target_names))]
    if target_names is None:
        target_names = ["%d" % i for i in nb_labels]
    for i in nb_labels:
        # cur_pal = sns.light_palette(colors[i], as_cmap=True)
        # d0, d1 = res[ds_l == i][:, 0], res[ds_l == i][:, 1]
        # ax = sns.kdeplot(d0, d1, shade=False, cmap=cur_pal,
        #                  alpha=0.75, shade_lowest=False, label="%d" % i,
        #                  n_levels=1, gridsize=100, ls=1, lw=1)
        # ax.patch.set_facecolor('white')
        # ax.collections[0].set_alpha(0)
        plt.scatter(res[ds_l == i][:, 0], res[ds_l == i][:, 1],
                    s=1.2, lw=0, alpha=1, color=colors[i], label=target_names[i])
    handles = []
    for ii in range(len(target_names)):
        handles.append(mpatches.Patch(color=colors[ii], label=target_names[ii]))
    plt.legend(handles=handles, loc="best")
    plt.savefig(dest_path, dpi=300)
    plt.close()

    if do_3d:
        # density plot 1st and 3rd PC
        plt.figure()
        plt.ylabel('$Z_3$', fontsize=15)
        plt.xlabel('$Z_1$', fontsize=15)
        for i in nb_labels:
            # cur_pal = sns.light_palette(colors[i], as_cmap=True)
            # d0, d2 = res[ds_l == i][:, 0], res[ds_l == i][:, 2]
            # ax = sns.kdeplot(d0, d2, shade=False, cmap=cur_pal,
            #                  alpha=0.6, shade_lowest=False, label="%d" % i
            #                  , gridsize=100, ls=0.6, lw=0.6)
            # ax.patch.set_facecolor('white')
            # ax.collections[0].set_alpha(0)
            plt.scatter(res[ds_l == i][:, 0], res[ds_l == i][:, 2],
                        s=1.2, lw=0, alpha=0.5, color=colors[i], label=target_names[i])
        handles = []
        for ii in range(len(target_names)):
            handles.append(mpatches.Patch(color=colors[ii], label=target_names[ii]))
        plt.legend(handles=handles, loc="best")
        plt.savefig(os.path.splitext(dest_path)[0] + "_2.png", dpi=300)
        plt.close()

        # density plot 2nd and 3rd PC
        plt.figure()
        plt.ylabel('$Z_3$', fontsize=15)
        plt.xlabel('$Z_2$', fontsize=15)
        for i in nb_labels:
            # cur_pal = sns.light_palette(colors[i], as_cmap=True)
            # d1, d2 = res[ds_l == i][:, 1], res[ds_l == i][:, 2]
            # ax = sns.kdeplot(d1, d2, shade=False, cmap=cur_pal,
            #                  alpha=0.6, shade_lowest=False, label="%d" % i
            #                  , gridsize=100, ls=0.6, lw=0.6)
            # ax.patch.set_facecolor('white')
            # ax.collections[0].set_alpha(0)
            plt.scatter(res[ds_l == i][:, 1], res[ds_l == i][:, 2],
                        s=1.2, lw=0, alpha=0.5, color=colors[i], label=target_names[i])
        handles = []
        for ii in range(len(target_names)):
            handles.append(mpatches.Patch(color=colors[ii], label=target_names[ii]))
        plt.legend(handles=handles, loc="best")
        plt.savefig(os.path.splitext(dest_path)[0] + "_3.png", dpi=300)
        plt.close()
    return tsne


class FileTimer:
    """
    ContextDecorator for timing. Stores the results as dict in a pkl file.

    Examples:
        The script SyConn/examples/start.py uses `FileTimer` to track the execution time of several
        major steps of the analysis. The results are written as ``dict`` to the file '.timing.pkl'
        in the working directory. The timing data can be accessed after the run to by initializing
        `FileTimer` with the output file:

            ft = FileTimer(path_to_timings_pkl)
            # this is a dict with the step names as keys and the timings in seconds as values
            print(ft.timings)

    """
    def __init__(self, working_dir: str, overwrite: bool = False, add_detail_vols: bool = False):
        if working_dir.endswith('.pkl'):
            fname = working_dir
            working_dir = os.path.abspath(os.path.split(working_dir)[0])
        else:
            fname = working_dir + '/.timing.pkl'
        self.fname = fname
        self.working_dir = working_dir
        self.step_name = None
        self.overwrite = overwrite
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        self.timings = {}
        self.t0, self.t1, self.interval = None, None, None
        self._load_prev()

        self.add_detail_vols = add_detail_vols
        self.kd = None
        self._dataset_shape = None
        self._dataset_nvoxels = None
        self._dataset_mm3 = None

    @property
    def dataset_shape(self) -> float:
        """

        Returns:
            Data set size in giga voxels.
        """
        if self._dataset_shape is None:
            self.prepare_vol_info()
        return self._dataset_shape

    @property
    def dataset_nvoxels(self) -> float:
        """

        Returns:
            Data set size in giga voxels.
        """
        if self._dataset_nvoxels is None:
            self.prepare_vol_info()
        if not self.add_detail_vols:
            return self._dataset_nvoxels['cube']  # whole data cube size
        return self._dataset_nvoxels

    @property
    def dataset_mm3(self) -> float:
        """

        Returns:
            Data set size in cubic mm.
        """
        if self._dataset_mm3 is None:
            self.prepare_vol_info()
        if not self.add_detail_vols:
            return self._dataset_mm3['cube']  # whole data cube size
        return self._dataset_mm3

    def _load_prev(self):
        if os.path.isfile(self.fname):
            if self.overwrite:
                os.remove(self.fname)
            else:
                prev = load_pkl2obj(self.fname)
                if not type(prev) is dict:
                    raise TypeError(f'Incompatible FileTimer type "{type(prev)}".')
                self.timings = prev

    def start(self, step_name: str):
        if self.step_name is not None:
            raise ValueError(f'Previous timing was not stopped.')
        self.t0 = time.perf_counter()
        self.step_name = step_name

    def stop(self):
        self.t1 = time.perf_counter()
        self.interval = self.t1 - self.t0
        if self.step_name is None:
            raise ValueError(f'No step name set. Please call the FileTimer instance and pass the '
                             f'step name as string.')
        self._load_prev()
        self.timings[self.step_name] = self.interval
        write_obj2pkl(self.fname, self.timings)
        self.step_name = None

    def __enter__(self):
        # do not start counting here to enable manual (with start and stop methods) interface and
        # context decorators. Timing difference between __enter__ and __call__ is not relevant for
        # our applications
        return self

    def __call__(self, step_name: str):
        self.start(step_name)

    def __exit__(self, *args):
        self.stop()

    def prepare_vol_info(self):
        # get data set properties
        if self._dataset_mm3 is not None:
            return
        conf = DynConfig(wd=self.working_dir, fix_config=True)
        try:
            bb = conf.entries['cube_of_interest_bb']
        except KeyError:
            bb = None
        self.kd = kd_factory(conf.entries['paths']['kd_seg'])
        if bb is None:
            bb = np.array([np.zeros(3, dtype=np.int32), self.kd.boundary])
        else:
            bb = np.array(bb)
        self._dataset_shape = bb[1] - bb[0]
        self._dataset_nvoxels = {'cube': np.prod(self.dataset_shape) / 1e9}
        self._dataset_mm3 = {'cube': np.prod(self.dataset_shape * self.kd.scale) / 1e18}
        if self.add_detail_vols:
            sd = SegmentationDataset('sv', config=conf)
            for k in ['total', 'glia', 'neuron']:
                vol_mm3 = sd.get_volume(k)
                self._dataset_mm3[k] = vol_mm3
                self._dataset_nvoxels[k] = vol_mm3 * 1e9 / np.prod(self.kd.scale)  # in GVx -> 1e18 / 1e9 = 1e9

    def prepare_report(self) -> str:
        self.prepare_vol_info()
        experiment_str = f'{self.kd.experiment_name} ({self.dataset_mm3}' \
                         f' mm^3; {self.dataset_nvoxels} GVx)'
        # python dicts are insertion order sensitive
        dt_tot = np.sum(np.array(list(self.timings.values())))
        dt_tot_str = time.strftime("{}d:{}h:{}min:{}s".format(*self._s2str(dt_tot)))
        time_summary_str = f"\nEM data analysis of experiment '{experiment_str}' finished " \
                           f"after {dt_tot_str}.\n"
        n_steps = len(self.timings)
        for i, (step_name, step_dt) in enumerate(self.timings.items()):
            step_dt_per = f"{(step_dt / dt_tot * 100):.1f}"
            step_dt = time.strftime("{}d:{}h:{}min:{}s".format(*self._s2str(step_dt)))
            step_str = '{:<10}{:<40}{:<20}{:<4s}\n'.format(f'[{i+1}/{n_steps}]', step_name,
                                                           step_dt, f'{step_dt_per}%')
            time_summary_str += step_str
        return time_summary_str

    @staticmethod
    def _s2str(seconds: float) -> tuple:
        d, h = divmod(seconds, 3600 * 24)
        h, min = divmod(h, 3600)
        min, s = divmod(min, 60)
        return int(d), int(h), int(min), int(s)
