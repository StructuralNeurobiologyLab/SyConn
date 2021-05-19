# Copyright (c) 2016 - now
# Max-Planck-Institute of Neurobiology, Munich, Germany
# Authors: Philipp Schubert, Joergen Kornfeld

from syconn.reps.super_segmentation import *
from syconn.handler import log_main
from syconn.handler.prediction import int2str_converter
from syconn.cnn.TrainData import CelltypeViewsJ0251
from syconn import global_params
import numpy as np
import pandas


if __name__ == "__main__":
    WD = "/ssdscratch/pschuber/songbird/j0251/rag_flat_Jan2019_v3"
    global_params.wd = WD
    global_params.config['batch_proc_system'] = None
    nclasses = 11
    int2str_label = {ii: int2str_converter(ii, 'ctgt_j0251_v2') for ii in range(nclasses)}
    str2int_label = {int2str_converter(ii, 'ctgt_j0251_v2'): ii for ii in range(nclasses)}
    csv_p = "/wholebrain/songbird/j0251/groundtruth/celltypes/j0251_celltype_gt_v4.csv"

    # prepare GT
    df = pandas.io.parsers.read_csv(csv_p, header=None, names=['ID', 'type']).values
    ssv_ids = df[:, 0].astype(np.uint64)
    if len(np.unique(ssv_ids)) != len(ssv_ids):
        raise ValueError('Multi-usage of IDs!')
    str_labels = df[:, 1]
    ssv_labels = np.array([str2int_label[el] for el in str_labels], dtype=np.uint16)
    classes, c_cnts = np.unique(ssv_labels, return_counts=True)
    if np.max(classes) > nclasses:
        raise ValueError('Class mis-match!')
    log_main.setLevel(20)  # This is INFO level (to filter copied file messages)
    log_main.info('Successfully parsed "{}" with the following cell type class '
                  'distribution [labels, counts]: {}, {}'.format(csv_p, classes,
                                                                 c_cnts))
    log_main.info('Total #cells: {}'.format(np.sum(c_cnts)))
    ssd_kwargs = dict(working_dir=WD)
    ssd = SuperSegmentationDataset(**ssd_kwargs)
    ssv_label_dc = {ssvid: str2int_label[el] for ssvid, el in zip(ssv_ids, str_labels)}
    # --------------------------------------------------------------------------
    # TEST PREDICTIONS OF TRAIN AND VALIDATION DATA
    from syconn.handler.prediction import get_celltype_model_e3
    from syconn.proc.stats import cluster_summary, projection_tSNE, model_performance
    from elektronn3.models.base import InferenceModel
    from syconn.reps.super_segmentation import SuperSegmentationDataset, SuperSegmentationObject
    np.set_printoptions(precision=4)
    # --------------------------------------------------------------------------
    base_dir = '/wholebrain/scratch/pschuber/e3_trainings_cmn_celltypes_j0251/'
    # analysis of VALIDATION set
    for run_ix in range(3):
        # Perform train data set eval as counter check
        gt_l = []
        certainty = []
        pred_l = []
        pred_proba = []
        loaded_ssv_ids = []
        for cv in range(10):
            ccd = CelltypeViewsJ0251(None, None, cv_val=cv)
            split_dc = ccd.splitting_dict
            ssv_ids = split_dc['valid']

            loaded_ssv_ids.extend(ssv_ids)
            pred_key_appendix = f'celltype_CV' \
                                f'{cv}/celltype_cmn_j0251v2_adam_nbviews20_longRUN_2ratios_BIG_bs40_10fold_CV' \
                                f'{cv}_eval{run_ix}'
            print('Loading cv-{}-data of model {}'.format(cv, pred_key_appendix))
            m_path = base_dir + pred_key_appendix
            pred_key_appendix += '_cmn'
            m = InferenceModel(m_path, bs=80)
            for ssv_id in ssv_ids:
                ssv = ssd.get_super_segmentation_object(ssv_id)
                ssv.load_attr_dict()
                # predict
                ssv.nb_cpus = 20
                ssv._view_caching = True
                ssv.predict_celltype_multiview(model=m, pred_key_appendix=pred_key_appendix, onthefly_views=True,
                                               view_props={'use_syntype': True, 'nb_views': 20}, overwrite=False,
                                               save_to_attr_dict=False, verbose=True,
                                               model_props={'n_classes': nclasses, 'da_equals_tan': False})
                ssv.save_attr_dict()
                # GT
                curr_l = ssv_label_dc[ssv.id]
                gt_l.append(curr_l)

                pred_l.append(ssv.attr_dict["celltype_cnn_e3" + pred_key_appendix])
                preds_small = ssv.attr_dict["celltype_cnn_e3{}_probas".format(pred_key_appendix)]
                major_dec = np.zeros(preds_small.shape[1])
                preds_small = np.argmax(preds_small, axis=1)
                for ii in range(len(major_dec)):
                    major_dec[ii] = np.sum(preds_small == ii)
                major_dec /= np.sum(major_dec)
                pred_proba.append(major_dec)
                if pred_l[-1] != gt_l[-1]:
                    print(f'{pred_l[-1]}\t{gt_l[-1]}\t{ssv.id}\t{major_dec}')
                certainty.append(ssv.certainty_celltype("celltype_cnn_e3{}".format(pred_key_appendix)))

        assert len(set(loaded_ssv_ids)) == len(ssv_label_dc)
        # # WRITE OUT COMBINED RESULTS
        pred_proba = np.array(pred_proba)
        certainty = np.array(certainty)
        gt_l = np.array(gt_l)

        target_names = [int2str_label[kk] for kk in range(nclasses)]

        # standard
        classes, c_cnts = np.unique(pred_l, return_counts=True)
        log_main.info('Successful prediction [standard] with the following cell type class '
                      'distribution [labels, counts]: {}, {}'.format(classes, c_cnts))
        perc_50 = np.percentile(certainty, 50)
        model_performance(pred_proba[certainty > perc_50], gt_l[certainty > perc_50],
                          f'{base_dir}/eval{run_ix}_results/upperhalf/', n_labels=nclasses, target_names=target_names,
                          add_text=f'Percentile-50: {perc_50}')
        model_performance(pred_proba[certainty <= perc_50], gt_l[certainty <= perc_50],
                          f'{base_dir}/eval{run_ix}_results/lowerhalf/', n_labels=nclasses, target_names=target_names,
                          add_text=f'Percentile-50: {perc_50}')
        model_performance(pred_proba, gt_l, f'{base_dir}/eval{run_ix}_results/', n_labels=nclasses,
                          target_names=target_names)
