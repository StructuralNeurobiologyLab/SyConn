# SyConn
# Copyright (c) 2016 Philipp J. Schubert
# All rights reserved
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.ticker as ticker
import seaborn as sns
sns.reset_orig()
import sys
import time
import numpy as np
from syconn.handler.basics import chunkify, load_pkl2obj, write_obj2pkl, write_txt2kzip, get_filepaths_from_dir
from syconn.proc.stats import model_performance, model_performance_predonly
from syconn.reps.super_segmentation_object import predict_sos_views, render_sso_coords
from syconn.reps.super_segmentation_helper import write_axpred
from syconn.reps.rep_helper import knossos_ml_from_ccs
from syconn.reps.super_segmentation_dataset import SuperSegmentationDataset
from syconn.reps.segmentation import SegmentationDataset
if 1:
    from elektronn2.neuromancer.model import modelload
    from elektronn2.config import config
else:
    from numa.neuromancer.model import modelload
    from numa.config import config
import os
import tqdm
from syconn.proc.skel_based_classifier import SkelClassifier
import shutil
from knossos_utils.skeleton_utils import load_skeleton
import re


class NeuralNetworkInterface(object):
    def __init__(self, model_path, arch='marvin', imposed_batch_size=1,
                 channels_to_load=(0, 1, 2, 3), normal=False, nb_labels=2):
        self.imposed_batch_size = imposed_batch_size
        self.channels_to_load = channels_to_load
        self.arch = arch
        self._path = model_path
        self._fname = os.path.split(model_path)[1]
        self.nb_labels = nb_labels
        self.normal = normal
        if config.device is None:
            from elektronn2.utils.gpu import initgpu
            initgpu(0)
        self.model = modelload(model_path, replace_bn='const',
                               imposed_batch_size=imposed_batch_size)
        self.original_do_rates = self.model.dropout_rates
        self.model.dropout_rates = ([0.0, ] * len(self.original_do_rates))

    def predict_proba(self, x, verbose=False):
        x = x.astype(np.float32)
        bs = self.model.batch_size
        if self.arch == "rec_view":
            batches = [np.arange(i * bs, (i + 1) * bs) for i in
                       range(x.shape[1] / bs)]
            proba = np.ones((x.shape[1], 4, self.nb_labels))
        else:
            batches = [np.arange(i * bs, (i + 1) * bs) for i in
                       range(len(x) / bs)]
            proba = np.ones((len(x), self.nb_labels))
        if verbose:
            cnt = 0
            start = time.time()
        pbar = tqdm.tqdm(total=len(batches), ncols=80, leave=False,
                         unit='it', unit_scale=True, dynamic_ncols=False)
        for b in batches:
            if verbose:
                sys.stdout.write("\r%0.2f" % (float(cnt) / len(batches)))
                sys.stdout.flush()
                cnt += 1
            x_b = x[b]
            proba[b] = self.model.predict(x_b)[None, ]
            pbar.update()
        overhead = len(x) % bs
        # TODO: add proper axis handling, maybe introduce axistags
        if overhead != 0:
            new_x_b = x[-overhead:]
            if len(new_x_b) < bs:
                add_shape = list(new_x_b.shape)
                add_shape[0] = bs - len(new_x_b)
                new_x_b = np.concatenate((np.zeros((add_shape), dtype=np.float32), new_x_b))
            proba[-overhead:] = self.model.predict(new_x_b)[-overhead:]
        if verbose:
            end = time.time()
            sys.stdout.write("\r%0.2f\n" % 1.0)
            sys.stdout.flush()
            print "Prediction of %d samples took %0.2fs; %0.4fs/sample." %\
                  (len(x), end-start, (end-start)/len(x))
        return proba


def get_test_candidates():
    ssd = SuperSegmentationDataset("/wholebrain/scratch/areaxfs3/",
                                   version="axgt")
    sv_ixs_gt = set(np.concatenate([ssv.sv_ids for ssv in ssd.ssvs]))
    old_sd = SegmentationDataset("sv", working_dir="/wholebrain/scratch/areaxfs/")
    ssd_all = SuperSegmentationDataset("/wholebrain/scratch/areaxfs3/")
    ssv_bb = ssd_all.load_cached_data("bounding_box")
    ssv_bb_size = np.linalg.norm((ssv_bb[:, 1] - ssv_bb[:,0])*ssd.scaling, axis=1)
    p60, p90 = np.percentile(ssv_bb_size, [85, 98])
    ssv_ids = np.array(ssd_all.ssv_ids)[(ssv_bb_size > p60) & (ssv_bb_size < p90)]
    np.random.seed(0)
    np.random.shuffle(ssv_ids)
    gt_candidates = []
    dest_folder = "/wholebrain/scratch/pschuber/cmn_paper/data/axoness_comparison/test_ssv/"
    if not os.path.isdir(dest_folder):
        os.makedirs(dest_folder)
    for ssv_ix in ssv_ids:
        ssv = ssd_all.get_super_segmentation_object(ssv_ix)
        inter_set = set(ssv.sv_ids).intersection(sv_ixs_gt)
        if len(inter_set) == 0:
            gt_candidates.append(ssv_ix)
            kzip_p = dest_folder + "%d.k.zip" % ssv_ix
            ssv.load_skeleton()
            if os.path.isfile(kzip_p) or ssv.skeleton is None:
                continue
            for sv in ssv.svs:
                new_view_path = sv.view_path(woglia=True)
                old_view_path = old_sd.get_segmentation_object(sv.id).view_path(woglia=True)
                if not os.path.isfile(new_view_path):
                    print old_view_path, new_view_path
                    # copy views to new SSD in areaxfs3
                    shutil.copy(old_view_path, new_view_path)
            # save kzip
            ssv.load_attr_dict()
            ssv.predict_nodes(sbc, feature_context_nm=4000)
            ssv.predict_nodes(sbc, feature_context_nm=8000)
            views = ssv.load_views()
            assert len(views) == len(ssv.sample_locations())
            probas = m.predict_proba(np.concatenate(views))
            ssv.attr_dict["axoness_probas_cnn_gt"] = probas
            ssv.attr_dict["axoness_preds_cnn_gt"] = np.argmax(probas, axis=1)
            ssv.save_attr_dict()
            ssv.cnn_axoness_2_skel(pred_key_appendix="_gt", k=1)
            ssv.save_skeleton_to_kzip(kzip_p, additional_keys=["axoness_cnn_k1_gt",
                                                               "axoness_fc4000_avgwind0",
                                                               "axoness_fc8000_avgwind0"])
            write_axpred(ssv, pred_key_appendix="_cnn_gt", k=1,
                         dest_path=kzip_p)
        if len(gt_candidates) == 100:
            break
    write_obj2pkl(dest_folder + "test_set.pkl", gt_candidates)
    kml = knossos_ml_from_ccs(gt_candidates, [ssd_all.get_super_segmentation_object(ssv_ix).sv_ids for ssv_ix in gt_candidates])
    write_txt2kzip(dest_folder + "test_set.k.zip", kml, "mergelist.txt")


def eval_test_candidates():
    dest_folder = "/wholebrain/scratch/pschuber/cmn_paper/data/axoness_comparison/test_ssv/"
    ssd_all = SuperSegmentationDataset("/wholebrain/scratch/areaxfs3/")
    sc = SkelClassifier(target_type="axoness", working_dir="/wholebrain/scratch/areaxfs3/")
    rfc_4000_wo2 = sc.load_classifier("rfc", feature_context_nm=4000,
                                      prefix="(2,)")
    rfc_8000_wo2 = sc.load_classifier("rfc", feature_context_nm=8000,
                                      prefix="(2,)")
    kzip_ps = get_filepaths_from_dir(dest_folder)
    cnv_dict = {"gt_axon": 1, "gt_soma": 2, "gt_dendrite": 0}
    all_labels = []
    all_preds_skel_4000 = []
    all_preds_skel_8000 = []
    all_preds_skel_4000_wo2 = []
    all_preds_skel_8000_wo2 = []
    all_preds_cnn = []
    currently_annotated = ["34299393.001.k", "8733319.001.k", "30030341.001.k", "28985344.001.k", "12474080.001.k"]
    for fname in kzip_ps:
        if not np.any([ca in fname for ca in currently_annotated]):
            continue
        sso_id = int(re.findall("(\d+).\d+.k.zip", fname)[0])
        sso = ssd_all.get_super_segmentation_object(sso_id)
        sso.load_skeleton()
        skel_nodes = sso.skeleton["nodes"]
        feats_4000 = sso.skel_features(feature_context_nm=4000)
        preds_4000_wo2 = np.argmax(rfc_4000_wo2.predict_proba(feats_4000), axis=1)
        feats_8000 = sso.skel_features(feature_context_nm=8000)
        preds_8000_wo2 = np.argmax(rfc_8000_wo2.predict_proba(feats_8000), axis=1)
        preds_4000wo2_dc = {}
        preds_8000wo2_dc = {}
        for i in range(len(skel_nodes)):
            preds_4000wo2_dc[frozenset(skel_nodes[i])] = preds_4000_wo2[i]
            preds_8000wo2_dc[frozenset(skel_nodes[i])] = preds_8000_wo2[i]
        try:
            skel = load_skeleton(fname)["skeleton"]
        except KeyError:
            continue
        for n in skel.getNodes():
            c = n.getComment()
            try:
                label = cnv_dict[c]
            except KeyError:
                continue
            all_labels.append(label)
            all_preds_skel_4000.append(int(n.data["axoness_fc4000_avgwind0"]))
            all_preds_skel_8000.append(int(n.data["axoness_fc8000_avgwind0"]))
            all_preds_skel_4000_wo2.append(preds_4000wo2_dc[frozenset(n.getCoordinate())])
            all_preds_skel_8000_wo2.append(preds_8000wo2_dc[frozenset(n.getCoordinate())])
            all_preds_cnn.append(int(n.data["axoness_cnn_k1_gt"]))
    print "Collected %d labeled nodes." % len(all_labels)
    model_performance_predonly(all_preds_skel_4000, all_labels, model_dir=dest_folder, prefix="skel_4000")
    model_performance_predonly(all_preds_skel_8000, all_labels, model_dir=dest_folder, prefix="skel_8000")
    model_performance_predonly(all_preds_cnn, all_labels, model_dir=dest_folder, prefix="cnn_k1")

    all_labels = np.array(all_labels)
    all_preds_skel_4000_wo2 = np.array(all_preds_skel_4000_wo2)
    all_preds_skel_8000_wo2 = np.array(all_preds_skel_8000_wo2)
    all_preds_skel_4000_wo2 = all_preds_skel_4000_wo2[all_labels != 2]
    all_preds_skel_8000_wo2 = all_preds_skel_8000_wo2[all_labels != 2]
    all_labels = all_labels[all_labels != 2]
    # and without soma
    model_performance_predonly(all_preds_skel_4000_wo2, all_labels, model_dir=dest_folder, prefix="skel_4000_wo2")
    model_performance_predonly(all_preds_skel_8000_wo2, all_labels, model_dir=dest_folder, prefix="skel_8000_wo2")


# model which is NOT trained on all samples, but on training samples
# as defined in axgt_splitting_v3.pkl
model_p = "/wholebrain/u/pschuber/CNN_Training/nupa_cnn/axoness_old/" \
          "g4_axoness_v0_run3/g4_axoness_v0_run3-FINAL.mdl"
def get_axoness_model():
    m = NeuralNetworkInterface(model_p, imposed_batch_size=200, nb_labels=3)
    _ = m.predict_proba(np.zeros((1, 4, 2, 128, 256)))
    return m


def plot_bars(ind, values, title='', legend_labels=None,
            save_path=None, colorVals=None, r_x=None, xtick_rotation=0,
            xlabel='', ylabel='', l_pos="upper right",
            legend=True, ls=22, xtick_labels=(), width=None):

    if width == None:
        width = 0.35

    def array2_xls(dest_p, arr):
        import xlsxwriter

        workbook = xlsxwriter.Workbook(dest_p)
        worksheet = workbook.add_worksheet()
        col = 0

        for row, data in enumerate(arr):
            worksheet.write_row(row, col, data)

        workbook.close()

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

    plt.title(title)

    plt.xlabel(xlabel, fontsize=ls)
    plt.ylabel(ylabel, fontsize=ls)

    if save_path is not None:
        dest_dir, fname = os.path.split(save_path)
        if legend_labels is not None:
            ll = [["legend labels"] + list(legend_labels)]
        else:
            ll = [[]]
        array2_xls(dest_dir + "/" + os.path.splitext(fname)[0] + ".xlsx", ll + [["labels", xlabel, ylabel]] + [xtick_labels] + [ind] + [list(arr) for arr in values])

    handles = []
    for ii in range(len(values)):
        rects = ax.bar(ind+width/len(values)*(ii-len(values)//2), values[ii], width/len(values), color=colorVals[ii])
    for ii in range(len(values)):
        handles.append(patches.Patch(color=colorVals[ii],
                                     label=legend_labels[ii]))

    if legend:
        ax.legend(handles=handles, frameon=False,
                   prop={'size': ls}, loc='upper right')
    if plt.xlim(r_x) is not None:
        plt.xlim(r_x)
    ax.set_xticks(ind)
    if len(xtick_labels) > 0:
        # ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.set_xticklabels(xtick_labels, rotation=xtick_rotation)
    plt.tight_layout()
    if save_path is None:
        plt.show(block=False)
    else:
        plt.savefig(save_path, dpi=400)


def plot_axoness_comparison():
    name = ["skel_4000", "skel_8000", "cnn_k1", "skel_4000_wo2", "skel_8000_wo2"]
    xtick_labels = ["RFC-4", "RFC-8", "CNN", "RFC*-4", "RFC*-8"]
    # the f-score value is the weighted f-score average of the three classes
    fscore_den = np.array([0.8429, 0.8860, 0.9460, 0.8809, 0.9341])
    fscore_ax = np.array([0.3814, 0.4141, 0.8999, 0.7968, 0.8716])
    fscore_so = np.array([0.2728, 0.3219, 0.9960, 0, 0])
    fscore_overall_unweighted = np.mean(np.array([fscore_den[:], fscore_ax, fscore_so]), axis=0)
    fscore_overall_unweighted[-2] = np.mean([fscore_den[-2], fscore_ax[-2]])
    fscore_overall_unweighted[-1] = np.mean([fscore_den[-1], fscore_ax[-1]])
    fscore_overall_weighted = [0.4851, 0.5295, 0.9635, 0.8542, 0.9142]


    plot_bars(np.arange(len(xtick_labels))+0.8, [fscore_so, fscore_den, fscore_ax, fscore_overall_unweighted], legend=False, xtick_labels=xtick_labels, width=0.8, xtick_rotation=90,
            xlabel="model", ylabel="F-Score", legend_labels=["soma", "dendrite", "axon", "avg."], r_x=[0, 6], colorVals=[[0.32, 0.32, 0.32, 1.], [0.6, 0.6, 0.6, 1], [0.841, 0.138, 0.133, 1.],
                           np.array([11, 129, 220, 255]) / 255.],
            save_path="/wholebrain/scratch/pschuber/cmn_paper/figures/axoness_comparison/FINAL_PLOT.png",)# colorVals=['0.32', '0.66'])
    # plot_pr(fscore, np.arange(1, len(fscore)+1), xtick_labels=xtick_labels, legend=False,
    #         xlabel="", ylabel="F-Score", r=[0.8, 1.01], r_x=[0, len(fscore) + 1],
    #             save_path="/wholebrain/scratch/pschuber/cmn_paper/figures//glia_performances.png")
    # plot_pr_stacked([fscore_neg, fscore], np.arange(1, len(fscore) + 1), xtick_labels=xtick_labels,
    #         legend=True, legend_labels=["non-glia", "glia", "avg"], colorVals=["0.3", "0.6"],
    #         xlabel="", ylabel="F-Score", r=[0.8, 1.01], r_x=[0, len(fscore) + 1],
    #         save_path="/wholebrain/scratch/pschuber/cmn_paper/figures//glia_performances_stacked.png")


if __name__ == "__main__":
    m = get_axoness_model()
    # ssd_old = SuperSegmentationDataset("/wholebrain/scratch/areaxfs/",
    #                                version="axgt_phil")
    eval_valid = False
    ssd = SuperSegmentationDataset("/wholebrain/scratch/areaxfs3/",
                                   version="axgt")
    sbc = SkelClassifier("axoness", working_dir="/wholebrain/scratch/areaxfs3/", create=False)

    # create test set candidates
    if 0:
        get_test_candidates()

    # eval test set
    if 0:
        eval_test_candidates()
    if 1:
        plot_axoness_comparison()

    # evaluate cnn on "pseude" train/valid set (views are different form actual views used during training)
    # evaluate rfc on train and valid dataset
    if 0:
        working_dir = "/wholebrain/scratch/pschuber/cmn_paper/data/axoness_comparison/"
        axgt_labels = load_pkl2obj("%s/ssv_axgt.pkl" % working_dir)
        splitting = load_pkl2obj("%s/axgt_splitting_v3.pkl" % working_dir)
        total_cnt = 0
        total_proba_view = []
        total_proba_node = []
        total_proba_rfc = []
        total_proba_rfc_large = []
        total_labels_view = []
        total_labels_node = []
        total_labels_rfc = []
        total_labels_rfc_large = []
        for ssv in ssd.ssvs:
            if not ssv.id in axgt_labels:
                print "Skipping", ssv.id
                continue
            if eval_valid and not ssv.id in splitting["valid"]:
                continue
            if not eval_valid and not ssv.id in splitting["train"]:
                continue
            print "\n-------------------------"
            print "[%d] Label: %d" % (ssv.id, axgt_labels[ssv.id])
            if eval_valid:
                assert ssv.id in splitting["valid"]
            else:
                assert ssv.id in splitting["train"]
            ssv.load_attr_dict()
            ssv.load_skeleton()
            # ssv.predict_nodes(sbc, feature_context_nm=4000)
            # ssv.predict_nodes(sbc, feature_context_nm=8000)
            views = ssv.load_views()
            assert len(views) == len(ssv.sample_locations())
            # probas = m.predict_proba(np.concatenate(views))
            # ssv.attr_dict["axoness_probas_cnn_gt"] = probas
            # ssv.attr_dict["axoness_preds_cnn_gt"] = np.argmax(probas, axis=1)
            # ssv.save_attr_dict()
            # view performance
            total_proba_view.append(ssv.attr_dict["axoness_probas_cnn_gt"])
            total_labels_view.append([axgt_labels[ssv.id]]*len(ssv.attr_dict["axoness_preds_cnn_gt"]))
            # transform to nodes
            # ssv.cnn_axoness_2_skel(pred_key_appendix="_gt", k=1)
            # node performance
            total_proba_node.append(ssv.skeleton["axoness_cnn_k1_gt_probas"])
            total_labels_node.append([axgt_labels[ssv.id]]*len(ssv.skeleton["axoness_cnn_k1_gt"]))
            total_proba_rfc.append(ssv.skeleton["axoness_fc4000_avgwind0_proba"])
            total_proba_rfc_large.append(ssv.skeleton["axoness_fc8000_avgwind0_proba"])
            total_labels_rfc.append([axgt_labels[ssv.id]]*len(ssv.skeleton["axoness_fc4000_avgwind0"]))
            total_labels_rfc_large.append([axgt_labels[ssv.id]]*len(ssv.skeleton["axoness_fc8000_avgwind0"]))
            kzip_p = "%s/%d_%d_%s.k.zip" % (working_dir, axgt_labels[ssv.id], ssv.id, "train" if ssv.id in splitting["train"] else "valid")
            ssv.save_skeleton_to_kzip(kzip_p, additional_keys=["axoness_cnn_k1", "axoness_fc4000_avgwind0", "axoness_fc8000_avgwind0"])
            write_axpred(ssv, pred_key_appendix="_cnn_gt", k=1,
                         dest_path=kzip_p)
            print "-------------------------\n"
        model_performance(np.concatenate(total_proba_view), np.concatenate(total_labels_view),
                          model_dir=working_dir, prefix="cnn_views_%s" % ("vaild" if eval_valid else "train"))
        model_performance(np.concatenate(total_proba_node), np.concatenate(total_labels_node),
                          model_dir=working_dir, prefix="cnn_node-wise%s" % ("vaild" if eval_valid else "train"))
        model_performance(np.concatenate(total_proba_rfc), np.concatenate(total_labels_rfc),
                          model_dir=working_dir, prefix="rfc%s" % ("vaild" if eval_valid else "train"))
        model_performance(np.concatenate(total_proba_rfc_large), np.concatenate(total_labels_rfc_large),
                          model_dir=working_dir, prefix="rfc_large%s" % ("vaild" if eval_valid else "train"))