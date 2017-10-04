import cPickle as pkl
import glob
import numpy as np
import os
import re
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.externals import joblib
from sklearn.metrics import precision_recall_fscore_support, precision_recall_curve

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

try:
    import keras
    from keras.models import Sequential
    from keras.layers import Dense, Dropout
    from keras.optimizers import RMSprop
    keras_avail = True
except ImportError:
    print('Keras and tensorflow not available.')
    keras_avail = False

import skel_based_classifier_helper as sbch
import super_segmentation as ss

from ..mp import qsub_utils as qu
from ..mp import shared_mem as sm
script_folder = os.path.abspath(os.path.dirname(__file__) + "/../qsub_scripts/")

feature_set = ["Mean diameter", "STD diameter", "Hist1", "Hist2", "Hist3",
               "Hist4", "Hist5", "Hist6", "Hist7", "Hist8", "Hist9", "Hist10",
               "Mean node degree", "N sj", "Mean size sj", "STD size sj",
               "N mi", "Mean size mi", "STD size mi", "N vc", "Mean size vc",
               "STD size vc"]

colorVals = [[0.841, 0.138, 0.133, 1.],
             [0.35, 0.725, 0.106, 1.],
             [0.100, 0.200, 0.600, 1.],
             [0.05, 0.05, 0.05, 1.],
             [0.25, 0.25, 0.25, 1.]] + [[0.45, 0.45, 0.45, 1.]]*20

colors = {"axgt": [colorVals[0], ".7", ".3"],
          "ctgt": np.array([[127, 98, 170], [239, 102, 142], [177, 181, 53],
                            [92, 181, 170]]) / 255.}

legend_labels = {"axgt": ("Axon", "Dendrite", "Soma"),
                 "ctgt": ("EA", "MSN", "GP", "INT")}


class SkelClassifier(object):
    def __init__(self, working_dir=None, ssd_version=None, create=False):
        self._ssd_version = ssd_version
        self._working_dir = working_dir
        self._clf = None
        self._ssd = None
        self.label_dict = None

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
    def working_dir(self):
        return self._working_dir

    @property
    def ssd_version(self):
        return str(self._ssd_version)

    @property
    def path(self):
        return self.working_dir + "/skel_clf_%s" % self.ssd_version

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
            self._ssd = ss.SuperSegmentationDataset(
                working_dir=self.working_dir, version=self.ssd_version)
        return self._ssd

    def avail_feature_contexts(self, clf_name):
        paths = glob.glob(self.clf_path + "/*%s*.pkl" % clf_name)
        feature_contexts = set()
        for path in paths:
            fc = int(re.findall("[\d]+", os.path.basename(path))[-1])
            feature_contexts.add(fc)

        return np.array(list(feature_contexts))

    def load_label_dict(self):
        if self.label_dict is None:
            if self.ssd_version == "axgt":
                with open(self.working_dir + "/axgt_labels.pkl", "r") as f:
                    self.label_dict = pkl.load(f)
            elif self.ssd_version == "ctgt":
                with open(self.working_dir + "/ctgt_labels.pkl", "r") as f:
                    self.label_dict = pkl.load(f)
            else:
                raise()

    def generate_data(self, feature_contexts_nm=[8000], stride=10,
                      qsub_pe=None, qsub_queue=None, nb_cpus=1):
        self.load_label_dict()

        multi_params = []
        for fc_block in [feature_contexts_nm[i:i + stride]
                         for i in xrange(0, len(feature_contexts_nm), stride)]:
            for this_id in self.label_dict.keys():
                multi_params.append([this_id, self.ssd_version,
                                     self.working_dir,
                                     fc_block,
                                     self.feat_path + "/features_%d_%d.npy"])

        if qsub_pe is None and qsub_queue is None:
            results = sm.start_multiprocess(
                sbch.generate_clf_data_thread,
                multi_params, nb_cpus=nb_cpus)

        elif qu.__QSUB__:
            path_to_out = qu.QSUB_script(multi_params,
                                         "generate_clf_data",
                                         pe=qsub_pe, queue=qsub_queue,
                                         script_folder=script_folder)

        else:
            raise Exception("QSUB not available")

    def classifier_production(self, clf_name="ext", n_estimators=200,
                              feature_contexts_nm=[4000], qsub_pe=None,
                              qsub_queue=None, nb_cpus=1):
        self.load_label_dict()
        multi_params = []
        for feature_context_nm in feature_contexts_nm:
            multi_params.append([self.working_dir, self.ssd_version,
                                 clf_name, n_estimators,
                                 feature_context_nm])
        if qsub_pe is None and qsub_queue is None:
            results = sm.classifier_production_thread(
                sbch.generate_clf_data_thread,
                multi_params, nb_cpus=nb_cpus)

        elif qu.__QSUB__:
            path_to_out = qu.QSUB_script(multi_params,
                                         "classifier_production",
                                         n_cores=nb_cpus,
                                         pe=qsub_pe, queue=qsub_queue,
                                         script_folder=script_folder)

        else:
            raise Exception("QSUB not available")

    def create_splitting(self, ratios=(.6, .2, .2)):
        self.load_label_dict()
        classes = np.array(self.label_dict.values(), dtype=np.int)
        unique_classes = np.unique(classes)

        id_bin_dict = {"train": [], "valid": [], "test": []}
        for this_class in unique_classes:
            sso_ids = np.array(self.label_dict.keys())[classes == this_class]

            weights = []
            for sso_id in sso_ids:
                sso = self.ss_dataset.get_super_segmentation_object(sso_id)
                sso.load_skeleton()
                weights.append(len(sso.skeleton["nodes"]))

            print this_class, np.sum(weights)

            cum_weights = np.cumsum(np.array(weights) / float(np.sum(weights)))
            train_mask = cum_weights < ratios[0]
            valid_mask = np.logical_and(cum_weights >= ratios[0],
                                        cum_weights < ratios[0] + ratios[1])
            test_mask = cum_weights >= ratios[0] + ratios[1]

            id_bin_dict["train"] += list(sso_ids[train_mask])
            id_bin_dict["valid"] += list(sso_ids[valid_mask])
            id_bin_dict["test"] += list(sso_ids[test_mask])

        with open(self.path + "/%s_splitting.pkl" % self.ssd_version, "w") as f:
            pkl.dump(id_bin_dict, f)

    def id_bins(self):
        if not os.path.exists(self.path + "/%s_splitting.pkl" % self.ssd_version):
            self.create_splitting()

        with open(self.path + "/%s_splitting.pkl" % self.ssd_version, "r") as f:
            part_dict = pkl.load(f)

        id_bin_dict = {}
        for key in part_dict.keys():
            for sso_id in part_dict[key]:
                id_bin_dict[sso_id] = key
        return id_bin_dict

    def load_data(self, feature_context_nm=8000, ratio=(0.7, .15, .15)):
        self.load_label_dict()

        id_bin_dict = self.id_bins()

        feature_dict = {}
        labels_dict = {}
        for sso_id in self.label_dict.keys():
            if not sso_id in id_bin_dict:
                continue

            this_feats = np.load(self.feat_path +
                                 "/features_%d_%d.npy" % (feature_context_nm, sso_id))

            if id_bin_dict[sso_id] in feature_dict:
                feature_dict[id_bin_dict[sso_id]] = \
                    np.concatenate([feature_dict[id_bin_dict[sso_id]],
                                    this_feats])
            else:
                feature_dict[id_bin_dict[sso_id]] = this_feats

            if id_bin_dict[sso_id] in labels_dict:
                labels_dict[id_bin_dict[sso_id]] += \
                    [self.label_dict[sso_id]] * len(this_feats)
            else:
                labels_dict[id_bin_dict[sso_id]] = \
                    [self.label_dict[sso_id]] * len(this_feats)

        feature_dict["test"] = feature_dict["valid"]
        labels_dict["test"] = labels_dict["valid"]

        return feature_dict["train"], labels_dict["train"], \
               feature_dict["valid"], labels_dict["valid"], \
               feature_dict["test"], labels_dict["test"]

    def score(self, probs, labels, balanced=True):
        pred = np.argmax(probs, axis=1)
        labels = np.array(labels)
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
        print "Overall acc: %.5f\n" % overall_score

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
                raise()
            else:
                f_score = 2 * precision * recall / (recall + precision)
            score_dict[this_class] = [precision, recall, f_score]

            print "class: %d: p: %.4f, r: %.4f, f: %.4f" % \
                  (this_class, precision, recall, f_score)
        return score_dict, label_weights

    def train_clf(self, name, n_estimators=2000, feature_context_nm=8000,
                  balanced=True, production=False, performance=False,
                  save=False, fast=False):
        if name == "rfc":
            clf = self.create_rfc(n_estimators=n_estimators)
        elif name == "ext":
            clf = self.create_ext(n_estimators=n_estimators)
        else:
            raise()

        print "\n --- %s ---\n" % name
        tr_feats, tr_labels, v_feats, v_labels, te_feats, te_labels = \
            self.load_data(feature_context_nm=feature_context_nm)

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

        if name in ["rfc", "ext"]:
            print "OOB score (train set): %.10f" % clf.oob_score_
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

        print "\nFEATURE IMPORTANCES"
        print "--------------------\n"
        for i_feat in range(len(feat_imps)):
            print "%s: %.5f" % (feat_set[i_feat], feat_imps[i_feat])

        if save:
            self.save_classifier(clf, name, feature_context_nm)

        self.eval_performance(clf.predict_proba(v_feats), v_labels,
                              clf.predict_proba(te_feats), te_labels,
                              [name, str(n_estimators), str(feature_context_nm)])

    def create_rfc(self, n_estimators=20):
        rfc = RandomForestClassifier(warm_start=False, oob_score=True,
                                     max_features="auto",
                                     # max_depth=4,
                                     n_estimators=n_estimators,
                                     class_weight="balanced",
                                     n_jobs=-1)
        return rfc

    def create_ext(self, n_estimators=20):
        ext = ExtraTreesClassifier(warm_start=False, oob_score=True,
                                   max_features="sqrt",
                                   n_estimators=n_estimators,
                                   bootstrap=True,
                                   # class_weight="balanced",
                                   n_jobs=-1)
        return ext

    def train_mlp(self, feature_context_nm=8000):
        print "\n --- MLP ---\n"
        tr_feats, tr_labels, v_feats, v_labels, te_feats, te_labels = \
            self.load_data(feature_context_nm=feature_context_nm)

        norm = np.max(np.concatenate([tr_feats, v_feats, te_feats]), axis=0)
        tr_feats /= norm
        v_feats /= norm
        te_feats /= norm

        print(tr_feats.shape[0], 'train samples')
        print(v_feats.shape[0], 'test samples')

        num_classes = len(np.unique(tr_labels))
        print "N classes", num_classes

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

    def save_classifier(self, clf, name, feature_context_nm):
        joblib.dump(clf, self.clf_path + '/clf_%s_%d.pkl' %
                    (name, feature_context_nm))

    def load_classifier(self, name, feature_context_nm):
        clf = joblib.load(self.clf_path + '/clf_%s_%d.pkl' %
                          (name, feature_context_nm))

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
        labels = np.array(labels)
        weights = 1.e8/np.unique(labels, return_counts=True)[1]
        labels = labels.copy()
        label_weights = np.zeros_like(labels, dtype=np.float)
        for i_class in range(len(classes)):
            this_class = classes[i_class]
            label_weights[labels == this_class] = weights[this_class]

        labels = np.array(labels)
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

                print precision[-1], recall[-1]
            curves.append([precision, recall, f_score, np.arange(0.0, 1.01, 0.01)])
        return curves

    def eval_performance(self, probs_valid, labels_valid, probs_test,
                         labels_test, identifiers=()):
        curves_valid = self.get_curves(probs_valid, labels_valid)
        curves_test = self.get_curves(probs_test, labels_test)

        # ov_prec_valid, ov_rec_valid, ov_fs_valid, _ = \
        #     precision_recall_fscore_support(labels_valid,
        #                                     np.argmax(probs_valid, axis=1),
        #                                     average="weighted")

        ov_prec_test, ov_rec_test, ov_fs_test, _ = \
            precision_recall_fscore_support(labels_test,
                                            np.argmax(probs_test, axis=1),
                                            average="weighted")
        path = self.plots_path + "prc"
        for ident in identifiers:
            path += "_%s" % ident

        self.plot_lines(curves_valid, "Recall", "Precision", path + "_valid.pdf",
                        legend_labels=legend_labels[self.ssd_version])
        self.plot_lines(curves_test, "Recall", "Precision", path + "_test.pdf",
                        legend_labels=legend_labels[self.ssd_version])

        print "Best F-Score:"
        print "Class-wise"
        # for i_class in range(probs_valid.shape[1]):
            # print "valid:", legend_labels[self.ssd_version][i_class], np.max(curves_valid[i_class][2])
            # print "same params in test:", legend_labels[self.ssd_version][i_class], curves_test[i_class][2][np.argmax(curves_valid[i_class][2])]
            # print "same params in test:", legend_labels[self.ssd_version][i_class], np.max(curves_test[i_class][2])

        print "\nOverall"
        for i_class in range(probs_valid.shape[1]):
            # print "valid:", np.max(ov_fs_valid)
            print "test:", np.max(ov_fs_test)

