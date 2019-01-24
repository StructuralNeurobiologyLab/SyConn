# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Sven Dorkenwald, Philipp Schubert, Joergen Kornfeld

try:
    import cPickle as pkl
except ImportError:
    import pickle as pkl
import glob
import numpy as np
import os
import re
from collections import Counter
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.externals import joblib
import matplotlib
matplotlib.use("Agg", warn=False, force=True)
from matplotlib import pyplot as plt

from . import skel_based_classifier_helper as sbch
from ..handler.basics import load_pkl2obj
from ..handler.logger import initialize_logging
from ..reps import super_segmentation as ss
from ..proc.stats import model_performance
from ..mp import batchjob_utils as qu
from ..mp import mp_utils as sm
logger_skel = initialize_logging('skeleton')
feature_set = ["Mean diameter", "STD diameter", "Hist1", "Hist2", "Hist3",
               "Hist4", "Hist5", "Hist6", "Hist7", "Hist8", "Hist9", "Hist10",
               "Mean node degree", "N sj", "Mean size sj", "STD size sj",
               "N mi", "Mean size mi", "STD size mi", "N vc", "Mean size vc",
               "STD size vc", "node density"]

colorVals = [[0.841, 0.138, 0.133, 1.],
             [0.35, 0.725, 0.106, 1.],
             [0.100, 0.200, 0.600, 1.],
             [0.05, 0.05, 0.05, 1.],
             [0.25, 0.25, 0.25, 1.]] + [[0.45, 0.45, 0.45, 1.]]*20

colors = {"axgt": [colorVals[0], ".7", ".3"],
          "spgt": [".7", "r", "k"],
          "ctgt": np.array([[127, 98, 170], [239, 102, 142], [177, 181, 53],
                            [92, 181, 170]]) / 255.}

legend_labels = {"axgt": ("Axon", "Dendrite", "Soma"),
                 "ctgt": ("EA", "MSN", "GP", "INT"),
                 "spgt": ("Shaft", "Head")}#, "Neck")}

comment_converter = {"axgt": {"soma": 2, "axon": 1, "dendrite": 0},
                     "spgt": {"shaft": 0, "head": 1, "neck": 2},
                     "ctgt": {}}


class SkelClassifier(object):
    def __init__(self, target_type, working_dir=None, create=False):
        assert target_type in ["axoness", "spiness"]
        if target_type == "axoness":
            ssd_version = "axgt"
        elif target_type == "spiness":
            ssd_version = "spgt"
        else:
            raise ValueError("'target_type' has to be one of "
                             "['axoness', 'spiness']")
        self._target_type = target_type
        self._ssd_version = ssd_version
        self._working_dir = working_dir
        self._clf = None
        self._ssd = None
        self.label_dict = None
        self.splitting_dict = None

        if create and not os.path.exists(self.path):
            os.makedirs(self.path)
        if create and not os.path.exists(self.labels_path):
            os.makedirs(self.labels_path)
        if create and not os.path.exists(self.feat_path):
            os.makedirs(self.feat_path)
        if create and not os.path.exists(self.clf_path):
            os.makedirs(self.clf_path)
        if create and not os.path.exists(self.plots_path):
            os.makedirs(self.plots_path)

    @property
    def target_type(self):
        return self._target_type

    @property
    def working_dir(self):
        return self._working_dir

    @property
    def ssd_version(self):
        return str(self._ssd_version)

    @property
    def splitting_fname(self):
        return self.working_dir + "/ssv_{}/{}_splitting.pkl".format(self.ssd_version, self.ssd_version)

    @property
    def label_dict_fname(self):
        return self.working_dir + "/ssv_%s/%s_labels.pkl" \
               % (self.ssd_version, self.ssd_version)

    @property
    def path(self):
        return self.working_dir + "/skel_clf_{}_{}".format(
            self.target_type, self.ssd_version)

    @property
    def labels_path(self):
        return self.path + "/labels/"

    @property
    def plots_path(self):
        return self.path + "/plots/"

    @property
    def feat_path(self):
        return self.path + "/features/"

    @property
    def clf_path(self):
        return self.path + "/clfs/"

    @property
    def ss_dataset(self):
        if self._ssd is None:
            self._ssd = ss.SuperSegmentationDataset(ssd_type='ssv',
                                                    working_dir=self.working_dir,
                                                    version=self.ssd_version)
        return self._ssd

    def avail_feature_contexts(self, clf_name):
        paths = glob.glob(self.clf_path + "/*{}*.pkl".format(clf_name))
        feature_contexts = set()
        for path in paths:
            fc = int(re.findall("[\d]+", os.path.basename(path))[-1])
            feature_contexts.add(fc)

        return np.array(list(feature_contexts))

    def load_label_dict(self):
        if self.label_dict is None:
            if self.ssd_version in ["axgt", "spgt", "ctgt"]:
                with open(self.label_dict_fname, "rb") as f:
                    self.label_dict = pkl.load(f)
            else:
                msg = "SSD version wrong."
                logger_skel.critical(msg)
                raise ValueError(msg)

    def load_splitting_dict(self):
        if self.splitting_dict is None:
            assert os.path.isfile(self.splitting_fname)
            self.splitting_dict = load_pkl2obj(self.splitting_fname)

    def generate_data(self, feature_contexts_nm=(500, 1000, 2000, 4000, 8000), stride=10,
                      qsub_pe=None, qsub_queue=None, nb_cpus=1, overwrite=True):
        self.load_label_dict()
        self.load_splitting_dict()
        multi_params = []
        sso_ids = np.concatenate(list(self.splitting_dict.values())).astype(
            np.int)
        for fc_block in [feature_contexts_nm[i:i + stride]
                         for i in range(0, len(feature_contexts_nm), stride)]:

            for this_id in sso_ids:
                multi_params.append([this_id, self.ssd_version,
                                     self.working_dir,
                                     fc_block,
                                     self.feat_path + "/features_%d_%d.npy",
                                    comment_converter[self.ssd_version], overwrite])

        if (qsub_pe is None and qsub_queue is None) or not qu.batchjob_enabled():
            results = sm.start_multiprocess(sbch.generate_clf_data_thread,
                multi_params, nb_cpus=nb_cpus)

        elif qu.batchjob_enabled():
            path_to_out = qu.QSUB_script(multi_params,
                                         "generate_clf_data",
                                         pe=qsub_pe, queue=qsub_queue,
                                         script_folder=None)
        else:
            msg = "QSUB not available"
            logger_skel.critical(msg)
            raise Exception(msg)

    def classifier_production(self, clf_name="rfc", n_estimators=2000,
                              feature_contexts_nm=(500, 1000, 2000, 4000, 8000), qsub_pe=None,
                              qsub_queue=None, nb_cpus=1, production=False):
        self.load_label_dict()
        multi_params = []
        for feature_context_nm in feature_contexts_nm:
            multi_params.append([self.working_dir, self.target_type,
                                 clf_name, n_estimators,
                                 feature_context_nm, production])
        if (qsub_pe is None and qsub_queue is None) or not qu.batchjob_enabled():
            results = sm.start_multiprocess(classifier_production_thread,
                multi_params, nb_cpus=nb_cpus)

        elif qu.batchjob_enabled():
            path_to_out = qu.QSUB_script(multi_params,
                                         "classifier_production",
                                         n_cores=nb_cpus,
                                         pe=qsub_pe, queue=qsub_queue,
                                         script_folder=None)

        else:
            msg = "QSUB not available"
            logger_skel.critical(msg)
            raise Exception(msg)

    def create_splitting(self, ratios=(.6, .2, .2)):
        assert not os.path.isfile(self.splitting_fname), "Splitting file exists."
        print("Creating dataset splits.")
        self.load_label_dict()
        classes = np.array(self.label_dict.values(), dtype=np.int)
        unique_classes = np.unique(classes)

        id_bin_dict = {"train": [], "valid": [], "test": []}
        for this_class in unique_classes:
            sso_ids = np.array(list(self.label_dict.keys()), dtype=np.uint)[classes == this_class]

            weights = []
            for sso_id in sso_ids:
                sso = self.ss_dataset.get_super_segmentation_object(sso_id)
                sso.load_skeleton()
                weights.append(len(sso.skeleton["nodes"]))

            print("Class weight:", this_class, np.sum(weights))

            cum_weights = np.cumsum(np.array(weights) / float(np.sum(weights)))
            train_mask = cum_weights < ratios[0]
            valid_mask = np.logical_and(cum_weights >= ratios[0],
                                        cum_weights < ratios[0] + ratios[1])
            test_mask = cum_weights >= ratios[0] + ratios[1]

            id_bin_dict["train"] += list(sso_ids[train_mask])
            id_bin_dict["valid"] += list(sso_ids[valid_mask])
            id_bin_dict["test"] += list(sso_ids[test_mask])

        with open(self.splitting_fname, "wb") as f:
            pkl.dump(id_bin_dict, f)

    def id_bins(self):
        if not os.path.exists(self.splitting_fname) and self.splitting_dict is None:
            self.create_splitting()

        if self.splitting_dict is None:
            part_dict = load_pkl2obj(self.splitting_fname)
        else:
            part_dict = self.splitting_dict

        id_bin_dict = {}
        for key in part_dict.keys():
            for sso_id in part_dict[key]:
                id_bin_dict[sso_id] = key
        return id_bin_dict

    def load_data(self, feature_context_nm, ratio=(0.7, .15, .15)):
        self.load_splitting_dict()
        self.load_label_dict()

        id_bin_dict = self.id_bins()

        feature_dict = {}
        labels_dict = {}
        sso_ids = np.concatenate(list(self.splitting_dict.values())).astype(np.int)
        for sso_id in sso_ids:
            if not sso_id in id_bin_dict:
                continue
            this_feats = np.load(self.feat_path +
                                 "/features_{}_{}.npy".format(feature_context_nm, sso_id))
            labels_fname = self.feat_path +\
                            "/labels_{}_{}.npy".format(feature_context_nm, sso_id)
            if not os.path.isfile(labels_fname):
                this_labels = [self.label_dict[sso_id]] * len(this_feats)
            else:
                # if file exists, then current SSO contains multiple classes
                # only use features where we have specified labels
                this_labels = np.load(labels_fname)
                this_feats = this_feats[this_labels != -1]
                this_labels = this_labels[this_labels != -1]
                if self.target_type == "spiness":
                    # set neck labels to shaft labels, and then as a postporcessing assign skeleton ndoes between branch point and spine head as neck
                    this_labels[this_labels == 2] = 0
                this_labels = this_labels.tolist()
                cnt = Counter(this_labels)
                print("Found node specific labels in SSV {}. {}".format(sso_id, cnt))
            if id_bin_dict[sso_id] in feature_dict:
                feature_dict[id_bin_dict[sso_id]] = \
                    np.concatenate([feature_dict[id_bin_dict[sso_id]],
                                    this_feats])
            else:
                feature_dict[id_bin_dict[sso_id]] = this_feats

            if id_bin_dict[sso_id] in labels_dict:
                labels_dict[id_bin_dict[sso_id]] += this_labels
            else:
                labels_dict[id_bin_dict[sso_id]] = this_labels
        labels_dict["train"] = np.array(labels_dict["train"], dtype=np.int)
        labels_dict["valid"] = np.array(labels_dict["valid"], dtype=np.int)
        if "test" not in labels_dict:
            labels_dict["test"] = []
        if "test" not in feature_dict:
            feature_dict["test"] = np.zeros((0, feature_dict["train"].shape[1]), np.float)
        labels_dict["test"] = np.array(labels_dict["test"], dtype=np.int)
        print("--------DATASET SUMMARY--------\n\n"
              "train\n%s\n\nvalid\n%s\n\ntest\n%s\n" %
              (Counter(labels_dict["train"]), Counter(labels_dict["valid"]),
               Counter(labels_dict["test"])))
        return feature_dict["train"], labels_dict["train"], \
               feature_dict["valid"], labels_dict["valid"], \
               feature_dict["test"], labels_dict["test"]

    def score(self, probs, labels, balanced=True):
        pred = np.argmax(probs, axis=1)
        labels = np.array(labels, dtype=np.int)
        classes = np.unique(labels)
        if balanced:
            weights = 1.e8/np.unique(labels, return_counts=True)[1]
            labels = labels.copy()
            label_weights = np.zeros_like(labels, dtype=np.float)
            for i_class in range(len(classes)):
                this_class = classes[i_class]
                label_weights[labels == this_class] = weights[this_class]
        else:
            label_weights = np.ones_like(labels, dtype=np.float)

        overall_score = np.sum((pred == labels) * label_weights) / \
                        float(np.sum(label_weights))
        print("Overall acc: {0:.5}\n".format(overall_score))

        score_dict = {}
        for i_class in range(len(classes)):
            this_class = classes[i_class]
            false_pos = float(np.sum((pred[labels != this_class] == this_class) * label_weights[labels != this_class]))
            false_neg = float(np.sum((pred[labels == this_class] != this_class) * label_weights[labels == this_class]))
            true_pos = float(np.sum((pred[labels == this_class] == this_class) * label_weights[labels == this_class]))

            precision = true_pos / (false_pos + true_pos)
            recall = true_pos / (false_neg + true_pos)
            if precision + recall == 0:
                f_score = 0
                ValueError("F-Score is 0.")
            else:
                f_score = 2 * precision * recall / (recall + precision)
            score_dict[this_class] = [precision, recall, f_score]

            print("class: {}: p: {:.4}, r: {:.4}, f: "
                  "{:.4}".format(this_class, precision, recall, f_score))
        return score_dict, label_weights

    def train_clf(self, name, n_estimators=2000, feature_context_nm=4000,
                  balanced=True, production=False, performance=False,
                  save=False, fast=False, leave_out_classes=()):
        if name == "rfc":
            clf = self.create_rfc(n_estimators=n_estimators)
        elif name == "ext":
            clf = self.create_ext(n_estimators=n_estimators)
        else:
            msg = "Unsupported classifier selected. Please chosse either" \
                  " 'ext' or 'rfc'."
            logger_skel.critical(msg)
            raise ValueError(msg)

        print("\n --- {} ---\n".format(name))
        tr_feats, tr_labels, v_feats, v_labels, te_feats, te_labels = \
            self.load_data(feature_context_nm=feature_context_nm)

        for c in leave_out_classes:
            tr_feats = tr_feats[tr_labels != c]
            tr_labels = tr_labels[tr_labels != c]
            if len(v_feats) > 0:
                v_feats = v_feats[v_labels != c]
                v_labels = v_labels[v_labels != c]
            if len(te_feats) > 0:
                te_feats = te_feats[te_labels != c]
                te_labels = te_labels[te_labels != c]

        if fast:
            classes = np.unique(tr_labels)
            for i_class in classes:
                tr_labels[i_class] = i_class
                te_feats[i_class] = i_class
                te_labels[i_class] = i_class
            tr_feats = tr_feats[:40]
            v_feats = v_feats[:40]
            te_feats = te_feats[:40]
            tr_labels = tr_labels[:40]
            v_labels = v_labels[:40]
            te_labels = te_labels[:40]


        if production:
            tr_feats = np.concatenate([tr_feats, v_feats, te_feats])
            # v_feats = tr_feats
            # te_feats = tr_feats
            tr_labels = np.concatenate([tr_labels, v_labels, te_labels])
            # v_labels = tr_labels
            # te_labels = tr_labels
        elif performance:
            tr_feats = np.concatenate([tr_feats, v_feats])
            v_feats = te_feats
            tr_labels = np.concatenate([tr_labels, v_labels])
            v_labels = te_labels

        # tr_labels = np.array(tr_labels)
        # if balanced:
        #     weights = 1.e8/np.unique(tr_labels, return_counts=True)[1]
        #     label_weights = np.zeros_like(tr_labels, dtype=np.float)
        #     for i_class in range(len(classes)):
        #         this_class = classes[i_class]
        #         label_weights[tr_labels == this_class] = weights[this_class]
        # else:
        #     label_weights = np.ones_like(tr_labels, dtype=np.float)

        unique_labels, labels_cnt = np.unique(tr_labels, return_counts=True)
        sample_weights = np.zeros_like(tr_labels, dtype=np.float32)
        for i_label in range(len(unique_labels)):
            sample_weights[tr_labels == unique_labels[i_label]] = 1. / labels_cnt[i_label]

        clf.fit(tr_feats, tr_labels, sample_weight=sample_weights)
        summary_str = ""
        if name in ["rfc", "ext"]:
            summary_str += "OOB score (train set): %.10f" % clf.oob_score_
        #
        # print "Label occ.:", np.unique(v_labels, return_counts=True)
        #
        # probs = clf.predict_proba(v_feats)
        #
        # print "Not balanced: "
        #
        # self.score(probs, v_labels, balanced=False)
        #
        # print "Label occ.:", np.unique(v_labels, return_counts=True)
        #
        # print "Balanced: "
        #
        # score, label_weights = self.score(probs, v_labels, balanced=True)

        feat_imps = np.array(clf.feature_importances_)

        sorting = np.argsort(feat_imps)[::-1]
        feat_set = np.array(feature_set)[sorting]
        feat_imps = feat_imps[sorting]

        summary_str += "\nFEATURE IMPORTANCES--------------------\n"
        for i_feat in range(np.min([5, len(feat_imps)])):#len(feat_imps)):
            summary_str += "%s: %.5f" % (feat_set[i_feat], feat_imps[i_feat])
        print("{}".format(summary_str))
        if save:
            prefix = "%s" % repr(leave_out_classes) if \
                len(leave_out_classes) > 0 else ""
            self.save_classifier(clf, name, feature_context_nm,
                                 production=production, prefix=prefix)
            with open(self.plots_path + prefix + "_summary.txt", 'w') as f:
                      f.write(summary_str)
        if not production:
            v_proba = clf.predict_proba(v_feats)
            if len(te_feats) > 0:
                te_proba = clf.predict_proba(te_feats)
            else:
                te_proba = np.zeros((0, v_proba.shape[-1]))
            self.eval_performance(v_proba, v_labels, te_proba, te_labels, leave_out_classes,
                                  [name, str(n_estimators), str(feature_context_nm), str(leave_out_classes)])

    def create_rfc(self, n_estimators=2000):
        rfc = RandomForestClassifier(warm_start=False, oob_score=True,
                                     max_features="auto",
                                     # max_depth=4,
                                     n_estimators=n_estimators,
                                     class_weight="balanced",
                                     n_jobs=-1)
        return rfc

    def create_ext(self, n_estimators=2000):
        ext = ExtraTreesClassifier(warm_start=False, oob_score=True,
                                   max_features="sqrt",
                                   n_estimators=n_estimators,
                                   bootstrap=True,
                                   # class_weight="balanced",
                                   n_jobs=-1)
        return ext

    def train_mlp(self, feature_context_nm=5000):
        raise(NotImplementedError)
        print("\n --- MLP ---\n")
        tr_feats, tr_labels, v_feats, v_labels, te_feats, te_labels = \
            self.load_data(feature_context_nm=feature_context_nm)

        norm = np.max(np.concatenate([tr_feats, v_feats, te_feats]), axis=0)
        tr_feats /= norm
        v_feats /= norm
        te_feats /= norm

        print(tr_feats.shape[0], 'train samples')
        print(v_feats.shape[0], 'test samples')

        num_classes = len(np.unique(tr_labels))
        print("N classes", num_classes)

        tr_labels = keras.utils.to_categorical(tr_labels, num_classes)
        v_labels = keras.utils.to_categorical(v_labels, num_classes)
        te_labels = keras.utils.to_categorical(te_labels, num_classes)

        model = Sequential()
        model.add(Dense(256, activation='relu', input_shape=(22,)))
        model.add(Dropout(0.3))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.1))
        model.add(Dense(num_classes, activation='softmax'))

        model.summary()

        model.compile(loss='categorical_crossentropy',
                      optimizer=RMSprop(),
                      metrics=['accuracy'])

        history = model.fit(tr_feats, tr_labels,
                            batch_size=512,
                            epochs=10,
                            verbose=1,
                            validation_data=(v_feats, v_labels))

        score = model.evaluate(te_feats, te_labels, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

    def save_classifier(self, clf, name, feature_context_nm, production=False,
                        prefix=""):
        save_p = self.clf_path + '/clf_%s_%d%s%s.pkl' % (name, feature_context_nm, "_prod" if production else "", prefix)
        joblib.dump(clf, save_p)

    def load_classifier(self, name, feature_context_nm, production=False,
                        prefix=""):
        save_p = self.clf_path + '/clf_%s_%d%s%s.pkl' % (
        name, feature_context_nm, "_prod" if production else "", prefix)
        clf = joblib.load(save_p)
        return clf

    def plot_lines(self, data, x_label, y_label, path, legend_labels=None):
        plt.clf()
        fig, ax = plt.subplots()

        ax.spines['left'].set_linewidth(3)
        ax.spines['bottom'].set_linewidth(3)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        fig.patch.set_facecolor('white')

        plt.xlabel(x_label, fontsize=24)
        plt.ylabel(y_label, fontsize=24)

        plt.tick_params(axis='both', which='major', labelsize=24,
                        direction='out',
                        length=8, width=3., right="off", top="off", pad=10)
        plt.tick_params(axis='both', which='minor', labelsize=12,
                        direction='out',
                        length=8, width=3., right="off", top="off", pad=10)

        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1.02)

        ax.set_xticks(np.arange(0., 1.1, 0.2))
        ax.set_xticks(np.arange(0., 1.1, 0.1), minor=True)
        ax.set_yticks(np.arange(0., 1.1, 0.2))
        ax.set_yticks(np.arange(0., 1.1, 0.1), minor=True)

        for i_line in range(len(data)):
            if legend_labels is None:
                plt.plot(data[i_line][0], data[i_line][1],
                         c=colors[self.ssd_version][i_line], lw=3, alpha=0.8)
            else:
                plt.plot(data[i_line][0], data[i_line][1], c=colors[self.ssd_version][i_line],
                         label=legend_labels[i_line], lw=3, alpha=0.8)

        legend = plt.legend(loc="best", frameon=False, prop={'size': 23})
        plt.tight_layout()
        plt.savefig(path)
        plt.close()

    def get_curves(self, probs, labels):
        classes = np.unique(labels)
        labels = np.array(labels, dtype=np.int)
        weights = 1.e8/np.unique(labels, return_counts=True)[1]
        labels = labels.copy()
        label_weights = np.zeros_like(labels, dtype=np.float)
        for i_class in range(len(classes)):
            this_class = classes[i_class]
            label_weights[labels == this_class] = weights[this_class]

        labels = np.array(labels, dtype=np.int)
        curves = []
        for i_class in range(probs.shape[1]):
            # curves.append(precision_recall_curve(labels==i_class, probs[:, i_class], sample_weight=label_weights))
            class_curve = []
            precision = []
            recall = []
            f_score = []
            for t in np.arange(0.0, 1.0, 0.01):
                pred = probs[:, i_class] >= t
                false_pos = float(np.sum(pred[labels != i_class].astype(np.float) * label_weights[labels != i_class]))
                false_neg = float(np.sum(np.invert(pred[labels == i_class]).astype(np.float) * label_weights[labels == i_class]))
                true_pos = float(np.sum(pred[labels == i_class].astype(np.float) * label_weights[labels == i_class]))

                if true_pos + false_pos == 0:
                    continue

                precision.append(true_pos / (false_pos + true_pos))
                recall.append(true_pos / (false_neg + true_pos))
                if precision[-1] + recall[-1] == 0:
                    f_score.append(0)
                else:
                    f_score.append(2 * precision[-1] * recall[-1] / (recall[-1] + precision[-1]))

                # print precision[-1], recall[-1]
            curves.append([precision, recall, f_score, np.arange(0.0, 1.01, 0.01)])
        return curves

    def eval_performance(self, probs_valid, labels_valid, probs_test,
                         labels_test, leave_out_classes=(), identifiers=()):
        prefix = ""
        for ident in identifiers:
            prefix += str(ident) + "_"
        prefix += "valid"
        tgt_names = list(legend_labels[self.ssd_version])
        if len(leave_out_classes) > 0:
            for c in leave_out_classes:
                tgt_names[c] = None
            tgt_names = [el for el in tgt_names if el is not None]
        model_performance(probs_valid, labels_valid, model_dir=self.plots_path,
                          n_labels=len(legend_labels[self.ssd_version]) - len(leave_out_classes),
                          target_names=tgt_names,
                          prefix=prefix)
        if len(probs_test) > 0:
            prefix = ""
            for ident in identifiers:
                prefix += str(ident) + "_"
            prefix += "test"
            model_performance(probs_test, labels_test, model_dir=self.plots_path,
                              n_labels=len(legend_labels[self.ssd_version]),
                              target_names=legend_labels[self.ssd_version],
                              prefix=prefix)


def classifier_production_thread(args):
    working_dir = args[0]
    target_type = args[1]
    clf_name = args[2]
    n_estimators = args[3]
    feature_context_nm = args[4]
    production = args[5]

    sc = SkelClassifier(target_type, working_dir=working_dir,
                        create=True)

    sc.train_clf(name=clf_name, n_estimators=n_estimators,
                 feature_context_nm=feature_context_nm, production=production,
                 save=True)
