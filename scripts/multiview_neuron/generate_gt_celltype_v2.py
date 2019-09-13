# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max-Planck-Institute of Neurobiology, Munich, Germany
# Authors: Philipp Schubert, Joergen Kornfeld

from syconn.reps.super_segmentation import *
from syconn.handler import log_main
from syconn import global_params
from syconn.handler.multiviews import generate_rendering_locs
import numpy as np
import logging
import pandas
from sklearn.model_selection import train_test_split


if __name__ == "__main__":
    WD ="/wholebrain/songbird/j0126/areaxfs_v6/"
    global_params.wd = WD
    global_params.config['batch_proc_system'] = None
    # based on /wholebrain/songbird/j0126/areaxfs_v6/
    # INT and ? are label 8, GPe and GP are label 5
    # asymmetric: (0, 3, 4)
    # symmetric: (2, 6, 7, 8, 5)
    str2int_label = dict(STN=0, DA=1, MSN=2, LMAN=3, HVC=4, GP=5, TAN=6, GPe=5, INT=7, FS=8, GLIA=9)
    str2int_label["GP "] = 5  # typo
    csv_p = '/wholebrain/songbird/j0126/GT/celltype_gt/j0126_cell_type_gt_areax_fs6_v3.csv'
    df = pandas.io.parsers.read_csv(csv_p, header=None, names=['ID', 'type']).values
    ssv_ids = df[:, 0].astype(np.uint)
    if len(np.unique(ssv_ids)) != len(ssv_ids):
        raise ValueError('Multi-usage of IDs!')
    str_labels = df[:, 1]
    ssv_labels = np.array([str2int_label[el] for el in str_labels], dtype=np.uint16)
    classes, c_cnts = np.unique(ssv_labels, return_counts=True)
    if np.max(classes) > 7:
        raise ValueError('Now we got Glia or FS?!')
    log_main.setLevel(20)  # This is INFO level (to filter copied file messages)
    log_main.info('Successfully parsed "{}" with the following cell type class '
                  'distribution [labels, counts]: {}, {}'.format(csv_p, classes,
                                                                 c_cnts))
    log_main.info('Total #cells: {}'.format(np.sum(c_cnts)))
    gt_version = "ctgt_v4"
    new_ssd = SuperSegmentationDataset(working_dir=WD, version=gt_version)

    ssd = SuperSegmentationDataset(working_dir=WD)
    orig_ssvs = ssd.get_super_segmentation_object(ssv_ids)
    for ssv in orig_ssvs:
        if not (ssv.id in ssd.ssv_ids and os.path.isfile(ssv.attr_dict_path)):
            msg = 'GT file contains SSV with ID {} which is not part of the ' \
                  'used SSD.'.format(ssv.id)
            log_main.warning(msg)
            raise ValueError(msg)
    copy_ssvs2new_SSD_simple(orig_ssvs, new_version=gt_version,
                             target_wd=WD, safe=False)  # `safe=False` will overwrite existing data
    new_ssd.save_dataset_deep(extract_only=True, )

    pbar = tqdm.tqdm(total=len(ssv_ids))
    for ii, ssv_id in enumerate(ssv_ids):
        ssv = new_ssd.get_super_segmentation_object(ssv_id)
        ssv.save_attributes(["cellttype_gt"], [ssv_labels[ii]])
        # # previous run for "normal" bootstrap N-view predictions
        ssv._render_rawviews(4, verbose=True, force_recompute=True)  # TODO: Make sure that copied files are always the same if `force_recompute` is set to False!
        # # Large FoV bootstrapping
        # # downsample vertices and get ~3 locations per comp_window
        verts = ssv.mesh[1].reshape(-1, 3)
        comp_window = 40960  # nm  -> pixel size: 80 nm
        rendering_locs = generate_rendering_locs(verts, comp_window)
        views = render_sso_coords(ssv, rendering_locs, verbose=True, ws=(512, 512), add_cellobjects=True,
                                  return_rot_mat=False, comp_window=comp_window, nb_views=4)
        ssv.save_views(views, view_key="4_large_fov")

        pbar.update()
    label_dc = {ssv_ids[kk]: ssv_labels[kk] for kk in range(len(ssv_ids))}
    write_obj2pkl(new_ssd.path + "/{}_labels.pkl".format(gt_version),
                 label_dc)

    X_train, X_test, y_train, y_test = train_test_split(
        list(label_dc.keys()), list(label_dc.values()), test_size=0.50,
        random_state=0, stratify=list(label_dc.values()))
    split_dc = {"train": X_train, "valid": X_test, "test": []}
    write_obj2pkl(path=new_ssd.path + "/{}_splitting.pkl".format(gt_version),
                  objects=split_dc)
    raise()
    # test prediction, copy trained models, see paths to 'models' folder defined in global_params.config
    from syconn.handler.prediction import get_celltype_model_large_e3, \
        get_tripletnet_model_large_e3, get_celltype_model_e3
    from syconn.proc.stats import cluster_summary, projection_tSNE, model_performance
    from elektronn3.models.base import InferenceModel
    from syconn.reps.super_segmentation import SuperSegmentationDataset, SuperSegmentationObject
    import tqdm
    split_dc = load_pkl2obj(path=new_ssd.path + "/{}_splitting.pkl".format(gt_version))
    for m_name in ['celltype_GTv3_nclasscorrected_CV1_sgd_bs20_nbviews10',
                   'celltype_GTv3_nclasscorrected_CV2_adabound_bs20_nbviews10',
                   'celltype_GTv3_nclasscorrected_CV2_adabound_bs60_nbviews10',
                   'celltype_GTv3_nclasscorrected_CV2_adabound_bs60_nbviews10_biggercache',
                   'celltype_GTv3_nclasscorrected_CV2_sgd_bs20_nbviews10',
                   'celltype_GTv3_nclasscorrected_CV1_adam_bs20_nbviews10']:
        m_path = '/wholebrain/u/pschuber/e3_training/' + m_name
        m = InferenceModel(m_path)

        pred_key_appendix1 = "test_pred_v1_large"
        pred_key_appendix2 = m_name
        ssd = SuperSegmentationDataset(working_dir=WD, version=gt_version)
        # #
        # # prediction
        # m = get_celltype_model_e3()
        # m_large = get_celltype_model_large_e3()
        # m_tnet = get_tripletnet_model_large_e3()

        # use appropriate validation set, according to the cross validation used.
        # TODO: change back to predict validation set, currently evaluates on training set for
        #  sanity checks
        ssv_ids = split_dc['valid'] if 'CV2' in m_name else split_dc['train']
        pbar = tqdm.tqdm(total=len(ssv_ids))

        for ssv_id in ssv_ids:
            ssv = ssd.get_super_segmentation_object(ssv_id)
            ssv.nb_cpus = 20
            ssv._view_caching = True
            # ssv.predict_celltype_cnn(model=m_large, pred_key_appendix=pred_key_appendix1,
            #                          model_tnet=m_tnet)
            ssv.predict_celltype_cnn(model=m, pred_key_appendix=pred_key_appendix2, largeFoV=False,
                                     view_props={"overwrite": False, 'use_syntype': False,
                                                 'nb_views': 10})
            pbar.update()
        pbar.close()
        # analysis
        gt_l = []
        pred_l = []
        pred_proba = []
        pred_l_large = []
        pred_proba_large = []
        latent_morph_d = []
        latent_morph_l = []
        for ssv_id in ssv_ids:
            ssv = ssd.get_super_segmentation_object(ssv_id)
            ssv.load_attr_dict()
            gt_l.append(ssv.attr_dict["cellttype_gt"])

            # small FoV
            pred_l.append(ssv.attr_dict["celltype_cnn_e3" + pred_key_appendix2])
            preds_small = ssv.attr_dict["celltype_cnn_e3{}_probas".format(pred_key_appendix2)]
            preds_small = np.argmax(preds_small, axis=1)
            major_dec = np.zeros(10)
            for ii in range(len(major_dec)):
                major_dec[ii] = np.sum(preds_small == ii)
            major_dec /= np.sum(major_dec)
            pred_proba.append(major_dec)

            # # large FoV
            # pred_l_large.append(ssv.attr_dict["celltype_cnn_e3" + pred_key_appendix1])
            # probas_large = ssv.attr_dict["celltype_cnn_e3{}_probas".format(pred_key_appendix1)]
            # preds_large = np.argmax(probas_large, axis=1)
            # major_dec = np.zeros(10)
            # for ii in range(len(major_dec)):
            #     major_dec[ii] = np.sum(preds_large == ii)
            # major_dec /= np.sum(major_dec)
            # pred_proba_large.append(major_dec)

            # # morphology embedding
            # latent_morph_d.append(ssv.attr_dict["latent_morph_ct" + pred_key_appendix2])
            # latent_morph_l.append(len(latent_morph_d[-1]) * [gt_l[-1]])

        # train_d = np.concatenate(latent_morph_d)
        # train_l = np.concatenate(latent_morph_l)
        # pred_proba_large = np.array(pred_proba_large)
        pred_proba = np.array(pred_proba)
        gt_l = np.array(gt_l)
        str2int_label = dict(STN=0, DA=1, MSN=2, LMAN=3, HVC=4, GP=5, FS=6, TAN=7, INT=8)
        int2str_label = {v: k for k, v in str2int_label.items()}
        dest_p = f"/wholebrain/scratch/pschuber/celltype_comparison/{m_name}/"
        os.makedirs(dest_p, exist_ok=True)
        target_names = [int2str_label[kk] for kk in range(9)]
        #
        # # large
        # classes, c_cnts = np.unique(np.argmax(pred_proba_large, axis=1), return_counts=True)
        # log_main.info('Successful prediction [large FoV] with the following cell type class '
        #               'distribution [labels, counts]: {}, {}'.format(classes, c_cnts))
        # model_performance(pred_proba_large, gt_l, dest_p n_labels=9,
        #                   target_names=target_names, prefix="large_")
        #
        # standard
        classes, c_cnts = np.unique(pred_l, return_counts=True)
        log_main.info('Successful prediction [standard] with the following cell type class '
                      'distribution [labels, counts]: {}, {}'.format(classes, c_cnts))
        model_performance(pred_proba, gt_l, dest_p, n_labels=9,
                          target_names=target_names)
        #
        # # tSNE
        # tsne_kwargs = {"n_components": 3, "random_state": 1, "perplexity": 30,
        #                "n_iter": 500}
        # projection_tSNE(train_d, train_l, dest_path=dest_p + 'tsne.png', target_names=target_names, do_3d=True,
        #                 **tsne_kwargs)
        # # cluster_summary(train_d, train_l, train_d, train_l, prefix=gt_version,
        # #                 fold=dest_p + 'tsne.png')
