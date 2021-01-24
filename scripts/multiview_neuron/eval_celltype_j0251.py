# Copyright (c) 2016 - now
# Max-Planck-Institute of Neurobiology, Munich, Germany
# Authors: Philipp Schubert, Joergen Kornfeld

from syconn.reps.super_segmentation import *
from syconn.handler import log_main
from syconn import global_params
import numpy as np
import pandas


if __name__ == "__main__":
    WD ="/wholebrain/songbird/j0126/areaxfs_v6/"
    global_params.wd = WD
    global_params.config['batch_proc_system'] = None

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
        raise ValueError('')
    log_main.setLevel(20)  # This is INFO level (to filter copied file messages)
    log_main.info('Successfully parsed "{}" with the following cell type class '
                  'distribution [labels, counts]: {}, {}'.format(csv_p, classes,
                                                                 c_cnts))
    log_main.info('Total #cells: {}'.format(np.sum(c_cnts)))
    gt_version = "ctgt_v4"
    new_ssd = SuperSegmentationDataset(working_dir=WD, version=gt_version)
    # --------------------------------------------------------------------------
    # TEST PREDICTIONS OF TRAIN AND VALIDATION DATA
    from syconn.handler.prediction import get_celltype_model_e3
    from syconn.proc.stats import cluster_summary, projection_tSNE, model_performance
    from elektronn3.models.base import InferenceModel
    from syconn.reps.super_segmentation import SuperSegmentationDataset, SuperSegmentationObject
    import tqdm
    np.set_printoptions(precision=4)
    da_equals_tan = True
    # --------------------------------------------------------------------------
    # analysis of VALIDATION set
    for m_name in ['celltype_GTv4_syntype_CV{}_adam_nbviews20_longRUN_2ratios_BIG_bs40_10fold_eval0',
                   'celltype_GTv4_syntype_CV{}_adam_nbviews20_longRUN_2ratios_BIG_bs40_10fold_eval1',
                   'celltype_GTv4_syntype_CV{}_adam_nbviews20_longRUN_2ratios_BIG_bs40_10fold_eval2']:
        # CV1: valid dataset: split_dc['valid'], CV2: valid_dataset: split_dc['train']
        # Perform train data set eval as counter check
        gt_l = []
        certainty = []
        pred_l = []
        pred_proba = []
        pred_l_large = []
        pred_proba_large = []
        latent_morph_d = []
        latent_morph_l = []
        loaded_ssv_ids = []
        # pbar = tqdm.tqdm(total=len(new_ssd.ssv_ids))
        for cv in range(10):
            split_dc = load_pkl2obj(path=new_ssd.path + "/{}_splitting_cv{}_10fold.pkl".format(
                gt_version, cv))
            ssv_ids = split_dc['valid']

            loaded_ssv_ids.extend(ssv_ids)
            pred_key_appendix2 = m_name.format(str(cv))
            print('Loading cv-{}-data of model {}'.format(cv, pred_key_appendix2))
            m_path = '/wholebrain/u/pschuber/e3_training_10fold_eval/' + pred_key_appendix2
            m = InferenceModel(m_path, bs=80)
            for ssv_id in ssv_ids:
                ssv = new_ssd.get_super_segmentation_object(ssv_id)
                # predict
                ssv.nb_cpus = 20
                ssv._view_caching = True
                # ssv.predict_celltype_cnn(model=m_large, pred_key_appendix=pred_key_appendix1,
                #                          model_tnet=m_tnet)
                ssv.predict_celltype_cnn(model=m, pred_key_appendix=pred_key_appendix2,
                                         view_props={"overwrite": False, 'use_syntype': True,
                                                     'nb_views': 20, 'da_equals_tan': da_equals_tan})
                ssv.load_attr_dict()
                curr_l = ssv.attr_dict["cellttype_gt"]
                if da_equals_tan:
                    # adapt GT labels
                    if curr_l == 6: curr_l = 1  # TAN and DA are the same now
                    if curr_l == 7: curr_l = 6  # INT now has label 6
                gt_l.append(curr_l)

                # small FoV
                pred_l.append(ssv.attr_dict["celltype_cnn_e3" + pred_key_appendix2])
                preds_small = ssv.attr_dict["celltype_cnn_e3{}_probas".format(pred_key_appendix2)]
                major_dec = np.zeros(preds_small.shape[1])
                preds_small = np.argmax(preds_small, axis=1)
                # For printing with all classes (in case da_equals_tan is True)
                for ii in range(len(major_dec)):
                    major_dec[ii] = np.sum(preds_small == ii)
                major_dec /= np.sum(major_dec)
                pred_proba.append(major_dec)
                if pred_l[-1] != gt_l[-1]:
                    print(f'{pred_l[-1]}\t{gt_l[-1]}\t{ssv.id}\t{major_dec}')
                certainty.append(ssv.certainty_celltype("celltype_cnn_e3{}_probas".format(pred_key_appendix2)))
                # pbar.update(1)
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

        assert set(loaded_ssv_ids) == set(new_ssd.ssv_ids.tolist())
        # # WRITE OUT COMBINED RESULTS
        # train_d = np.concatenate(latent_morph_d)
        # train_l = np.concatenate(latent_morph_l)
        # pred_proba_large = np.array(pred_proba_large)
        pred_proba = np.array(pred_proba)
        certainty = np.array(certainty)
        gt_l = np.array(gt_l)

        int2str_label = {v: k for k, v in str2int_label.items()}
        dest_p = f"/wholebrain/scratch/pschuber/celltype_comparison_syntype/{m_name}_valid" \
                 f"{'DA_eq_TAN' if da_equals_tan else ''}/"
        os.makedirs(dest_p, exist_ok=True)
        target_names = [int2str_label[kk] for kk in range(8)]

        # SET TAN AND DA TO THE SAME CLASS
        if da_equals_tan:
            target_names[1] = 'Modulatory'
            target_names.remove('TAN')

        # # large
        # classes, c_cnts = np.unique(np.argmax(pred_proba_large, axis=1), return_counts=True)
        # log_main.info('Successful prediction [large FoV] with the following cell type class '
        #               'distribution [labels, counts]: {}, {}'.format(classes, c_cnts))
        # model_performance(pred_proba_large, gt_l, dest_p n_labels=9,
        #                   target_names=target_names, prefix="large_")
        # standard
        classes, c_cnts = np.unique(pred_l, return_counts=True)
        log_main.info('Successful prediction [standard] with the following cell type class '
                      'distribution [labels, counts]: {}, {}'.format(classes, c_cnts))
        perc_50 = np.percentile(certainty, 50)
        model_performance(pred_proba[certainty > perc_50], gt_l[certainty > perc_50],
                          dest_p + '/upperhalf/', n_labels=7, target_names=target_names,
                          add_text=f'Percentile-50: {perc_50}')
        model_performance(pred_proba[certainty <= perc_50], gt_l[certainty <= perc_50],
                          dest_p + '/lowerhalf/', n_labels=7, target_names=target_names,
                          add_text=f'Percentile-50: {perc_50}')
        model_performance(pred_proba, gt_l, dest_p, n_labels=7,
                          target_names=target_names)
