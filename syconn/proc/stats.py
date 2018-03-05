# NeuroPatch
# Copyright (c) 2016 Philipp J. Schubert
# All rights reserved
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.ticker as ticker
from sklearn.metrics import precision_recall_curve, roc_auc_score, \
    classification_report, precision_recall_fscore_support, accuracy_score


def model_performance(proba, labels, model_dir=None, prefix="", n_labels=3,
                      fscore_beta=1, target_names=None):
    header = "-------------------------------\n\t\t%s\n" % prefix
    if target_names is None:
        target_names = ["Dendrite", "Axon", "Soma"]
    all_prec, all_rec = [], []
    for i in range(n_labels):
        curr_labels = np.array(labels, dtype=np.int)
        curr_labels[labels != i] = 0
        curr_labels[labels == i] = 1
        prec, rec, t = precision_recall_curve(curr_labels, np.array(proba)[:, i])
        all_prec.append(prec)
        all_rec.append(rec)
        f = fscore(rec, prec, beta=fscore_beta)
        t_opt = t[np.argmax(f)]
        header += "\nthresh, f%d-score, recall, precision, roc-auc, supp " \
                  "\n%0.6f %0.6f %0.6f %0.6f %0.6f %d\n\n" \
                 % (fscore_beta, t_opt, f[np.argmax(f)], rec[np.argmax(f)],
                    prec[np.argmax(f)],
                    roc_auc_score(np.array(curr_labels), np.array(proba[:, i])),
                    len(proba))
    header += classification_report(labels, np.argmax(proba, axis=1), digits=4,
                                target_names=target_names)
    header += "acc.: %0.4f" % accuracy_score(labels, np.argmax(proba, axis=1))
    header += "\n-------------------------------\n"
    print header
    plot_pr(all_prec, all_rec, r=[0.6, 1.01], legend_labels=target_names)
    if model_dir is not None:
        text_file = open(model_dir + '/prec_rec_%s.txt' % prefix, "w")
        text_file.write(header)
        text_file.close()
        prec, rec, fs, supp = precision_recall_fscore_support(labels, np.argmax(proba, axis=1))
        np.save(model_dir + '/prec_rec_%s.npy' % prefix, [prec, rec, fs])
        # plt.savefig(model_dir + '/prec_rec_%s.png' % prefix)
    plt.close()


def model_performance_predonly(pred, labels, model_dir=None, prefix="", target_names=None):
    header = "-------------------------------\n\t\t%s\n" % prefix
    if target_names is None:
        target_names = ["Dendrite", "Axon", "Soma"]
    header += classification_report(labels, pred, digits=4,
                                target_names=target_names)
    header += "acc.: %0.4f" % accuracy_score(labels, pred)
    header += "\n-------------------------------\n"
    print header
    if model_dir is not None:
        text_file = open(model_dir + '/prec_rec_%s.txt' % prefix, "w")
        text_file.write(header)
        text_file.close()
        prec, rec, fs, supp = precision_recall_fscore_support(labels, pred)
        np.save(model_dir + '/prec_rec_%s.npy' % prefix, [prec, rec, fs])
    plt.close()


def hist(vals, labels=None, dest_path=None, axis_labels=None, x_lim=None,
         y_lim=None, y_log_scale=False, ls=22, color=None, **kwargs):
    sns.set_style("white")
    fig, ax = plt.subplots()
    fig.patch.set_facecolor('white')
    ax.tick_params(axis='x', which='major', labelsize=ls-4, direction='out',
                    length=4, width=3,  right="off", top="off", pad=10)
    ax.tick_params(axis='y', which='major', labelsize=ls-4, direction='out',
                    length=4, width=3,  right="off", top="off", pad=10)
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
    f_score = (1. + beta**2) * (prec * rec) / (beta**2 * prec + rec)
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
            ll = [["legend labels"]+list(legend_labels)]
        else:
            ll = [[]]
        array2xls(dest_dir + "/" + os.path.splitext(fname)[0] + ".xlsx", ll + [["labels", xlabel, ylabel]] + [xtick_labels] + [precision] + [recall])

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
