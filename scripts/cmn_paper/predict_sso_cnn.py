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
from sklearn.metrics import accuracy_score
import time
import numpy as np
from syconn.handler.basics import chunkify, load_pkl2obj, write_obj2pkl, write_txt2kzip, get_filepaths_from_dir
from syconn.proc.stats import model_performance, model_performance_predonly
from syconn.reps.super_segmentation_object import predict_sos_views, render_sso_coords
from syconn.reps.super_segmentation_helper import write_axpred, extract_skel_features, associate_objs_with_skel_nodes
from syconn.reps.rep_helper import knossos_ml_from_ccs, colorcode_vertices
from syconn.reps.super_segmentation_dataset import SuperSegmentationDataset
from syconn.reps.segmentation import SegmentationDataset
from syconn.handler.prediction import NeuralNetworkInterface, get_tripletnet_model, get_knn_tnet_embedding, views2tripletinput, force_correct_norm
import os
from syconn.mp.shared_mem import start_multiprocess_imap
from syconn.proc.skel_based_classifier import SkelClassifier
import shutil
from knossos_utils.skeleton_utils import load_skeleton
import re


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
            assert len(views) == len(np.concatenate(ssv.sample_locations()))
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
    old_sd = SegmentationDataset("sv", working_dir="/wholebrain/scratch/areaxfs/")
    sc = SkelClassifier(target_type="axoness", working_dir="/wholebrain/scratch/areaxfs3/")
    rfc_4000 = sc.load_classifier("rfc", feature_context_nm=4000,)
    rfc_8000 = sc.load_classifier("rfc", feature_context_nm=8000,)
    rfc_4000_wo2 = sc.load_classifier("rfc", feature_context_nm=4000,
                                      prefix="(2,)")
    rfc_8000_wo2 = sc.load_classifier("rfc", feature_context_nm=8000,
                                      prefix="(2,)")
    cnn_model = get_axoness_model()
    t_net = get_tripletnet_model()
    knn_clf = get_knn_tnet_embedding()
    kzip_ps = get_filepaths_from_dir(dest_folder)
    cnv_dict = {"gt_axon": 1, "gt_soma": 2, "gt_dendrite": 0}
    all_labels = []
    all_preds_skel_4000 = []
    all_preds_skel_8000 = []
    all_preds_skel_4000_maj = []
    all_preds_skel_8000_maj = []
    all_preds_skel_4000_wo2 = []
    all_preds_skel_4000_wo2_maj = []
    all_preds_skel_8000_wo2 = []
    all_preds_skel_8000_wo2_maj = []
    all_preds_latent = []
    all_preds_latent_maj = []
    all_preds_cnn = []
    all_preds_cnn_previous = []
    all_preds_cnn_previous_maj = []
    all_preds_cnn_maj = []
    all_coords = []  # debug: Check if
    # V1
    # currently_annotated = ["34299393.001.k", "8733319.001.k", "30030341.001.k", "28985344.001.k", "12474080.001.k"]
    # # V2 -- 28 SSV
    currently_annotated = ["34299393.001.k", "8733319.001.k", "30030341.001.k", "28985344.002.k", "12474080.001.k", "15065120.001.k", "31798150.001.k",
                           "28531969.001.k", "31936512.001.k", "141995.004.k", "1931265.001.k", "32946948.002.k", "18916097.001.k", "24300036.002.k",
                           "782342.001.k", "15372930.001.k.zip", "28418691.001.k", "8138368.001.k", "34053762.001.k", "13463680.001.k", "14539904.001.k",
                           "23629521.001.k", "22081952.001.k", "5491713.001.k", "10408322.001.k", "22892545.001.k", "13082627.001.k", "32292865.001.k"]
    # start_multiprocess_imap(preproc_gt_skels, kzip_ps, nb_cpus=20)
    sso_sizes = []
    sso_lenghts = []
    for fname in kzip_ps:
        if not np.any([ca in fname for ca in currently_annotated]):
            continue
        try:
            skel = load_skeleton(fname)["skeleton"]
        except KeyError:
            print("Skipped file %s." % fname)
            continue
        print(fname)
        sso_id = int(re.findall("(\d+).\d+.k.zip", fname)[0])
        sso = ssd_all.get_super_segmentation_object(sso_id)

        # cache skeleton to make sure its nodes are aligned with their features
        skel_fname = dest_folder + os.path.split(fname)[1][:-6] + "skel.pkl"
        if os.path.isfile(skel_fname):
            sso.skeleton = load_pkl2obj(skel_fname)
        else:
            # Build SSO skeleton from k.zip data to retrieve prediction of RFC
            # trained only on dendrite and axon... Was necessary because SSV
            # skeletons were updated in the meantime..
            nodes = []
            edges = []
            radii = []
            nb_nodes = len(skel.getNodes())
            axoness_fc4000_avgwind0 = np.zeros(nb_nodes)
            axoness_fc8000_avgwind0 = np.zeros(nb_nodes)
            axoness_cnn_k1_gt = np.zeros(nb_nodes)
            node_lookup = {}
            for ii, n in enumerate(skel.getNodes()):
                nodes.append(n.getCoordinate())
                radii.append(n.data["radius"])
                axoness_cnn_k1_gt[ii] = int(n.data["axoness_cnn_k1_gt"])
                # compute those directly
                # axoness_fc4000_avgwind0[ii] = int(n.data["axoness_fc4000_avgwind0"])
                # axoness_fc8000_avgwind0[ii] = int(n.data["axoness_fc8000_avgwind0"])
                node_lookup[frozenset(nodes[-1])] = len(nodes) - 1
            for n1, n2 in skel.iter_edges():
                ix_n1 = node_lookup[frozenset(n1.getCoordinate())]
                ix_n2 = node_lookup[frozenset(n2.getCoordinate())]
                e = [ix_n1, ix_n2]
                edges.append(e)
            nodes = np.array(nodes)
            edges = np.array(edges, dtype=np.uint)
            radii = np.array(radii)
            sso.skeleton = dict(edges=edges, nodes=nodes, diameters=radii*2,
                                axoness_cnn_k1_gt=axoness_cnn_k1_gt)
            write_obj2pkl(skel_fname, sso.skeleton)
        # DONE....
        sso_sizes.append(sso.size)
        sso_lenghts.append(sso.total_edge_length())
        associate_objs_with_skel_nodes(sso)
        feats_4000_cache_fname = dest_folder + os.path.split(fname)[1][:-6] + "_4000.npy"
        if not os.path.isfile(feats_4000_cache_fname):
            feats_4000 = extract_skel_features(sso, feature_context_nm=4000)  # sso.skel_features(feature_context_nm=4000), avoid this call because it would trigger caching mechanism
            np.save(feats_4000_cache_fname, feats_4000)
        else:
            feats_4000 = np.load(feats_4000_cache_fname)
        feats_8000_cache_fname = dest_folder + os.path.split(fname)[1][:-6] + "_8000.npy"
        if not os.path.isfile(feats_8000_cache_fname):
            feats_8000 = extract_skel_features(sso, feature_context_nm=8000)  # sso.skel_features(feature_context_nm=8000), avoid this call because it would trigger caching mechanism
            np.save(feats_8000_cache_fname, feats_8000)
        else:
            feats_8000 = np.load(feats_8000_cache_fname)
        preds_4000_wo2 = np.argmax(rfc_4000_wo2.predict_proba(feats_4000), axis=1)
        preds_8000_wo2 = np.argmax(rfc_8000_wo2.predict_proba(feats_8000), axis=1)
        preds_4000 = np.argmax(rfc_4000.predict_proba(feats_4000), axis=1)
        preds_8000 = np.argmax(rfc_8000.predict_proba(feats_8000), axis=1)
        sso.skeleton["axoness_fc4000_avgwind0_wo2"] = preds_4000_wo2
        sso.skeleton["axoness_fc8000_avgwind0_wo2"] = preds_8000_wo2
        sso.skeleton["axoness_fc4000_avgwind0"] = preds_4000
        sso.skeleton["axoness_fc8000_avgwind0"] = preds_8000
        # map current axon prediction to skel (DIFFERENT CNN) only to get the view <-> skel-node assignments (sso.skeleton["view_ixs"])
        sso.cnn_axoness_2_skel(pred_key_appendix="_v2", k=1, reload=False,
                               save_sso=False)
        # change svs to get old views
        new_svs = []
        for sv in sso.svs:
            new_svs.append(old_sd.get_segmentation_object(sv.id))
        sso._objects["sv"] = new_svs
        #now create majority vote for each if the predictions
        #now create majority vote for each if the predictions
        # use majority votes of unique views collected within max_dist
        # new predictions with the same CNN to check if the views are the same as when generating the GT
        views = sso.load_views(force_reload=True)
        assert len(views) == len(np.concatenate(sso.sample_locations()))
        probas = cnn_model.predict_proba(views, verbose=False)
        sso.attr_dict["axoness_preds_cnn_gt_v2"] = np.argmax(probas, axis=1)
        sso.cnn_axoness_2_skel(pred_key_appendix="_gt_v2", k=1, save_sso=False)
        # node_cnn_preds = cnn_preds[sso.skeleton["view_ixs"]]
        axoness_cnn_k1_gt_avg = sso.average_node_axoness_views(max_dist=12500,
                                return_res=True, pred_key="axoness_preds_cnn_gt_v2")
        sso.skeleton["axoness_cnn_k1_gt_v2_maj"] = axoness_cnn_k1_gt_avg
        # triplet network embedding
        latent_z = t_net.predict_proba(views)
        knn_preds = knn_clf.predict_proba(latent_z)
        knn_preds = np.argmax(knn_preds, axis=1)
        sso.attr_dict["axoness_cnn_latent"] = knn_preds
        # Approach 1, assumes sample locations have same ordering as view predictions are returned...
        sample_locs = np.concatenate(sso.sample_locations())
        node_latent = colorcode_vertices(sso.skeleton["nodes"] * sso.scaling,
                                         sample_locs, knn_preds, colors=[0, 1, 2], k=1)
        # Approach 2; assumes ordering of view prediction is the same as in view_ixs
        node_latent_v2 = knn_preds[sso.skeleton["view_ixs"]]
        assert np.all(node_latent == node_latent_v2)
        sso.skeleton["axoness_cnn_latent"] = node_latent
        axoness_latent_avg = sso.average_node_axoness_views(max_dist=12500,
                                return_res=True, pred_key="axoness_cnn_latent")
        sso.skeleton["axoness_cnn_latent_maj"] = axoness_latent_avg
        # skeleton based prediction
        for prop_key in ["axoness_fc4000_avgwind0", "axoness_fc8000_avgwind0",
                         "axoness_fc4000_avgwind0_wo2", "axoness_fc8000_avgwind0_wo2",
                         "axoness_cnn_k1_gt"]:
            sso.skeleton[prop_key + "_maj"] = sso.majority_vote(prop_key, max_dist=12500)
        # map node coordinate to index in skeleton arrays (valid for all skeleton entries with the same ordering as skeleton['nodes'], i.e. all majority votings and predictions
        skel_nodes = sso.skeleton["nodes"]
        coord2ix_dc = {}
        for i in range(len(skel_nodes)):
            coord2ix_dc[frozenset(skel_nodes[i])] = i
        # preds_4000wo2_dc = {}
        # preds_8000wo2_dc = {}
        # for i in range(len(skel_nodes)):
        #     preds_4000wo2_dc[frozenset(skel_nodes[i])] = preds_4000_wo2[i]
        #     preds_8000wo2_dc[frozenset(skel_nodes[i])] = preds_8000_wo2[i]
        for n in skel.getNodes():
            c = n.getComment()
            try:
                label = cnv_dict[c]
            except KeyError:
                continue
            all_coords.append(np.array(n.getCoordinate()))
            all_labels.append(label)
            ix = coord2ix_dc[frozenset(n.getCoordinate())]
            all_preds_skel_4000.append(sso.skeleton["axoness_fc4000_avgwind0"][ix])
            all_preds_skel_8000.append(sso.skeleton["axoness_fc8000_avgwind0"][ix])
            all_preds_skel_4000_wo2.append(sso.skeleton["axoness_fc4000_avgwind0_wo2"][ix])
            all_preds_skel_8000_wo2.append(sso.skeleton["axoness_fc8000_avgwind0_wo2"][ix])
            all_preds_latent.append(sso.skeleton["axoness_cnn_latent"][ix])
            all_preds_cnn.append(sso.skeleton["axoness_gt_v2"][ix])
            all_preds_cnn_previous.append(sso.skeleton["axoness_cnn_k1_gt"][ix])
            # get majority votings
            all_preds_skel_4000_maj.append(sso.skeleton["axoness_fc4000_avgwind0_maj"][ix])
            all_preds_skel_8000_maj.append(sso.skeleton["axoness_fc8000_avgwind0_maj"][ix])
            all_preds_skel_4000_wo2_maj.append(sso.skeleton["axoness_fc4000_avgwind0_wo2_maj"][ix])
            all_preds_skel_8000_wo2_maj.append(sso.skeleton["axoness_fc8000_avgwind0_wo2_maj"][ix])
            all_preds_latent_maj.append(sso.skeleton["axoness_cnn_latent_maj"][ix])
            all_preds_cnn_previous_maj.append(sso.skeleton["axoness_cnn_k1_gt_maj"][ix])
            all_preds_cnn_maj.append(sso.skeleton["axoness_cnn_k1_gt_v2_maj"][ix])
        print("Previous Views: {:0.4f}\t{:0.4f}\n"
              "New Views: {:0.4f}\t{:0.4f}\n"
              "KNN latent: {:0.4f}\t{:0.4f}".format(
              accuracy_score(np.array(all_labels), np.array(all_preds_cnn_previous)),
              accuracy_score(np.array(all_labels), np.array(all_preds_cnn_previous_maj)),
              accuracy_score(np.array(all_labels), np.array(all_preds_cnn)),
              accuracy_score(np.array(all_labels), np.array(all_preds_cnn_maj)),
              accuracy_score(np.array(all_labels), np.array(all_preds_latent)),
              accuracy_score(np.array(all_labels), np.array(all_preds_latent_maj))))
    print(np.sum(sso_sizes) / 1e9, np.sum(sso_sizes) * 9 * 9 * 20 / 1e9, np.sum(sso_lenghts) / 1e6, )
    print "Collected %d labeled nodes." % len(all_labels)
    model_performance_predonly(all_preds_skel_4000, all_labels, model_dir=dest_folder, prefix="skel_4000")
    model_performance_predonly(all_preds_skel_8000, all_labels, model_dir=dest_folder, prefix="skel_8000")
    model_performance_predonly(all_preds_cnn, all_labels, model_dir=dest_folder, prefix="cnn_k1")
    model_performance_predonly(all_preds_latent, all_labels, model_dir=dest_folder, prefix="latent")
    # majority
    model_performance_predonly(all_preds_skel_4000_maj, all_labels, model_dir=dest_folder, prefix="skel_4000_maj")
    model_performance_predonly(all_preds_skel_8000_maj, all_labels, model_dir=dest_folder, prefix="skel_8000_maj")
    model_performance_predonly(all_preds_cnn_maj, all_labels, model_dir=dest_folder, prefix="cnn_k1_maj")
    model_performance_predonly(all_preds_latent_maj, all_labels, model_dir=dest_folder, prefix="latent_maj")

    # withoout soma
    all_labels = np.array(all_labels)
    all_preds_skel_4000_wo2 = np.array(all_preds_skel_4000_wo2)
    all_preds_skel_8000_wo2 = np.array(all_preds_skel_8000_wo2)
    all_preds_skel_4000_wo2_maj = np.array(all_preds_skel_4000_wo2_maj)
    all_preds_skel_8000_wo2_maj = np.array(all_preds_skel_8000_wo2_maj)
    all_preds_skel_4000_wo2 = all_preds_skel_4000_wo2[all_labels != 2]
    all_preds_skel_4000_wo2_maj = all_preds_skel_4000_wo2_maj[all_labels != 2]
    all_preds_skel_8000_wo2 = all_preds_skel_8000_wo2[all_labels != 2]
    all_preds_skel_8000_wo2_maj = all_preds_skel_8000_wo2_maj[all_labels != 2]
    all_labels = all_labels[all_labels != 2]
    model_performance_predonly(all_preds_skel_4000_wo2, all_labels, model_dir=dest_folder, prefix="skel_4000_wo2")
    model_performance_predonly(all_preds_skel_8000_wo2, all_labels, model_dir=dest_folder, prefix="skel_8000_wo2")
    # majority
    model_performance_predonly(all_preds_skel_4000_wo2_maj, all_labels, model_dir=dest_folder, prefix="skel_4000_wo2_maj")
    model_performance_predonly(all_preds_skel_8000_wo2_maj, all_labels, model_dir=dest_folder, prefix="skel_8000_wo2_maj")

# model which is NOT trained on all samples, but on training samples
# as defined in axgt_splitting_v3.pkl
def get_axoness_model():
    model_p = "/wholebrain/scratch/pschuber/CNN_Training/nupa_cnn/axoness_old/" \
              "g4_axoness_v0_run3/g4_axoness_v0_run3-FINAL.mdl"
    m = NeuralNetworkInterface(model_p, imposed_batch_size=200, nb_labels=3)
    _ = m.predict_proba(np.zeros((1, 4, 2, 128, 256)))
    return m


def preproc_gt_skels(fname):
    dest_folder = "/wholebrain/scratch/pschuber/cmn_paper/data/axoness_comparison/test_ssv/"
    ssd_all = SuperSegmentationDataset("/wholebrain/scratch/areaxfs3/")
    currently_annotated = ["34299393.001.k", "8733319.001.k", "30030341.001.k",
                           "28985344.002.k", "12474080.001.k", "15065120.001.k",
                           "31798150.001.k",
                           "28531969.001.k", "31936512.001.k", "141995.004.k",
                           "1931265.001.k", "32946948.002.k", "18916097.001.k",
                           "24300036.002.k",
                           "782342.001.k", "15372930.001.k.zip",
                           "28418691.001.k", "8138368.001.k", "34053762.001.k",
                           "13463680.001.k", "14539904.001.k",
                           "23629521.001.k", "22081952.001.k", "5491713.001.k",
                           "10408322.001.k", "22892545.001.k", "13082627.001.k",
                           "32292865.001.k"]
    if not np.any([ca in fname for ca in currently_annotated]):
        return
    try:
        skel = load_skeleton(fname)["skeleton"]
    except KeyError:
        print("Skipped file %s." % fname)
        return
    print(fname)
    sso_id = int(re.findall("(\d+).\d+.k.zip", fname)[0])
    sso = ssd_all.get_super_segmentation_object(sso_id)

    # cache skeleton to make sure its nodes are aligned with their features
    skel_fname = dest_folder + os.path.split(fname)[1][:-6] + "skel.pkl"
    if 0:# os.path.isfile(skel_fname):
        sso.skeleton = load_pkl2obj(skel_fname)
    else:
        # Build SSO skeleton from k.zip data to retrieve prediction of RFC
        # trained only on dendrite and axon... Was necessary because SSV
        # skeletons were updated in the meantime..
        nodes = []
        edges = []
        radii = []
        nb_nodes = len(skel.getNodes())
        axoness_fc4000_avgwind0 = np.zeros(nb_nodes)
        axoness_fc8000_avgwind0 = np.zeros(nb_nodes)
        axoness_cnn_k1_gt = np.zeros(nb_nodes)
        node_lookup = {}
        for ii, n in enumerate(skel.getNodes()):
            nodes.append(n.getCoordinate())
            radii.append(n.data["radius"])
            axoness_cnn_k1_gt[ii] = int(n.data["axoness_cnn_k1_gt"])
            # compute those directly
            # axoness_fc4000_avgwind0[ii] = int(n.data["axoness_fc4000_avgwind0"])
            # axoness_fc8000_avgwind0[ii] = int(n.data["axoness_fc8000_avgwind0"])
            node_lookup[frozenset(nodes[-1])] = len(nodes) - 1
        for n1, n2 in skel.iter_edges():
            ix_n1 = node_lookup[frozenset(n1.getCoordinate())]
            ix_n2 = node_lookup[frozenset(n2.getCoordinate())]
            e = [ix_n1, ix_n2]
            edges.append(e)
        nodes = np.array(nodes)
        edges = np.array(edges, dtype=np.uint)
        radii = np.array(radii)
        sso.skeleton = dict(edges=edges, nodes=nodes, diameters=radii*2,
                            axoness_cnn_k1_gt=axoness_cnn_k1_gt)
        write_obj2pkl(skel_fname, sso.skeleton)
    # DONE....
    associate_objs_with_skel_nodes(sso)
    feats_4000_cache_fname = dest_folder + os.path.split(fname)[1][:-6] + "_4000.npy"
    if 1:#not os.path.isfile(feats_4000_cache_fname):
        feats_4000 = extract_skel_features(sso, feature_context_nm=4000)  # sso.skel_features(feature_context_nm=4000), avoid this call because it would trigger caching mechanism
        np.save(feats_4000_cache_fname, feats_4000)
    else:
        feats_4000 = np.load(feats_4000_cache_fname)
    feats_8000_cache_fname = dest_folder + os.path.split(fname)[1][:-6] + "_8000.npy"
    if 1:#not os.path.isfile(feats_8000_cache_fname):
        feats_8000 = extract_skel_features(sso, feature_context_nm=8000)  # sso.skel_features(feature_context_nm=8000), avoid this call because it would trigger caching mechanism
        np.save(feats_8000_cache_fname, feats_8000)
    else:
        feats_8000 = np.load(feats_8000_cache_fname)


def plot_bars(ind, values, title='', legend_labels=None, r_y=[0.2, 1.0],
            save_path=None, colorVals=None, r_x=None, xtick_rotation=0,
            xlabel='', ylabel='', l_pos="upper right", yaxis_mult=0.2,
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
        rects = ax.bar(ind+width/len(values)*(ii-len(values)/2)+width/len(values)/2, values[ii], width/len(values), color=colorVals[ii])
    for ii in range(len(values)):
        handles.append(patches.Patch(color=colorVals[ii],
                                     label=legend_labels[ii]))

    if legend:
        ax.legend(handles=handles, frameon=False,
                   prop={'size': ls}, loc='upper right')

    ax.set_xticks(ind)
    if plt.ylim(r_y) is not None:
        plt.ylim(r_y)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(yaxis_mult))
    if len(xtick_labels) > 0:
        # ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.set_xticklabels(xtick_labels, rotation=xtick_rotation)
    plt.tight_layout()
    if plt.xlim(r_x) is not None:
        plt.xlim(r_x)
    if save_path is None:
        plt.show(block=False)
    else:
        plt.savefig(save_path, dpi=400)


def plot_axoness_comparison():
    name = ["skel_4000", "skel_8000", "cnn_k1", "latent", "skel_4000_wo2", "skel_8000_wo2"]
    ordering = [3, 2, 0, -2, 1, -1]
    xtick_labels = np.array(["RFC-4", "RFC-8", "CMN", "Z", "RFC*-4", "RFC*-8"])[ordering]
    # the f-score value is the weighted f-score average of the three classes
    fscore_den = np.array([0.8437, 0.9066, 0.9450, 0.4026, 0.8701, 0.9272])[ordering]
    fscore_ax = np.array([0.5959, 0.6437, 0.9386, .6442, 0.8615, 0.9156])[ordering]
    fscore_so = np.array([0.2171, 0.2677, 0.9820, 0.6750, 0, 0])[ordering]
    fscore_overall_weighted = np.array([0.5666, 0.6211, 0.9552, 0.5621 , 0.8664, 0.9222])[ordering]
    fscore_overall_unweighted = np.mean(np.array([fscore_den[:], fscore_ax, fscore_so]), axis=0)
    fscore_overall_unweighted[-2] = np.mean([fscore_den[-2], fscore_ax[-2]])
    fscore_overall_unweighted[-1] = np.mean([fscore_den[-1], fscore_ax[-1]])

    plot_bars(np.arange(len(xtick_labels))+0.9, [fscore_so, fscore_den, fscore_ax, fscore_overall_weighted], legend=False, xtick_labels=xtick_labels, width=0.8, xtick_rotation=90,
            xlabel="model", ylabel="F1-Score", legend_labels=["soma", "dendrite", "axon", "avg."], r_x=[0, 7.5], colorVals=[[0.32, 0.32, 0.32, 1.], [0.6, 0.6, 0.6, 1], [0.841, 0.138, 0.133, 1.],
                           np.array([11, 129, 220, 255]) / 255.], r_y=[0.0, 1.0],
            save_path="/wholebrain/scratch/pschuber/cmn_paper/figures/axoness_comparison/node_bar.png",)# colorVals=['0.32', '0.66'])
    # plot_pr(fscore, np.arange(1, len(fscore)+1), xtick_labels=xtick_labels, legend=False,
    #         xlabel="", ylabel="F-Score", r=[0.8, 1.01], r_x=[0, len(fscore) + 1],
    #             save_path="/wholebrain/scratch/pschuber/cmn_paper/figures//glia_performances.png")
    # plot_pr_stacked([fscore_neg, fscore], np.arange(1, len(fscore) + 1), xtick_labels=xtick_labels,
    #         legend=True, legend_labels=["non-glia", "glia", "avg"], colorVals=["0.3", "0.6"],
    #         xlabel="", ylabel="F-Score", r=[0.8, 1.01], r_x=[0, len(fscore) + 1],
    #         save_path="/wholebrain/scratch/pschuber/cmn_paper/figures//glia_performances_stacked.png")

    # Majority voting with sliding window approahc of 12500nm maximum traversal path length from source node (views were collected along this traversal for majority vote)
    name_maj = ["skel_4000_maj", "skel_8000_maj", "cnn_k1_maj", "latent_maj", "skel_4000_wo2_maj", "skel_8000_wo2_maj"]
    xtick_labels_maj = np.array(["RFC-4", "RFC-8", "CMN", "Z", "RFC*-4", "RFC*-8"])[ordering]
    # the f-score value is the weighted f-score average of the three classes
    fscore_den_maj = np.array([0.9168, 0.9337, 0.9661, 0.3418, 0.9296, 0.9448])[ordering]
    fscore_ax_maj = np.array([0.6065, 0.6442, 0.9720, 0.6622, 0.9188, 0.9339])[ordering]
    fscore_so_maj = np.array([0.0798, 0.1977, 0.9688, 0.8818, 0, 0])[ordering]
    fscore_overall_weighted_maj = np.array([0.5528, 0.6088, 0.9687, 0.6118, 0.9249, 0.9401])[ordering]
    fscore_overall_unweighted_maj = np.mean(np.array([fscore_den_maj[:], fscore_ax_maj, fscore_so_maj]), axis=0)
    fscore_overall_unweighted_maj[-2] = np.mean([fscore_den_maj[-2], fscore_ax_maj[-2]])
    fscore_overall_unweighted_maj[-1] = np.mean([fscore_den_maj[-1], fscore_ax_maj[-1]])

    plot_bars(np.arange(len(xtick_labels))+0.9, [fscore_so_maj, fscore_den_maj, fscore_ax_maj, fscore_overall_weighted_maj], legend=False, xtick_labels=xtick_labels, width=0.8, xtick_rotation=90,
            xlabel="model", ylabel="F1-Score",legend_labels=["soma", "dendrite", "axon", "avg."], r_x=[0, 7.5], colorVals=[[0.32, 0.32, 0.32, 1.], [0.6, 0.6, 0.6, 1], [0.841, 0.138, 0.133, 1.],
                           np.array([11, 129, 220, 255]) / 255.], r_y=[0.0, 1.0],
            save_path="/wholebrain/scratch/pschuber/cmn_paper/figures/axoness_comparison/majority_vote_bar.png",)

    plot_bars(np.arange(len(xtick_labels))+0.9, [fscore_overall_weighted, fscore_overall_weighted_maj], legend=False, xtick_labels=xtick_labels, width=0.8, xtick_rotation=90,
            xlabel="model", ylabel="F1-Score", legend_labels=["node-wise", "majority"], r_x=[0, 7.5], colorVals=['0.32', '0.66'], r_y=[0.4, 1.0],
            save_path="/wholebrain/scratch/pschuber/cmn_paper/figures/axoness_comparison/majority_node_comparison.png",)

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
    if 1:
        eval_test_candidates()
    if 0:
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
            assert len(views) == len(np.concatenate(ssv.sample_locations()))
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