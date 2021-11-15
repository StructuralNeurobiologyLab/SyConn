import os
import numpy as np
import pickle as pkl
from syconn.mp.batchjob_utils import batchjob_script
from syconn.reps.super_segmentation import SuperSegmentationDataset
from syconn.handler.config import initialize_logging
from syconn.analysis.compartment_pts.evaluation.inference import predict_sso_thread_dnho
from syconn.analysis.compartment_pts.evaluation.evaluate_on_synapses import evaluate_syn_thread_dnho


if __name__ == '__main__':
    ssv_ids = [141995, 11833344, 28410880, 28479489]
    wd = "/wholebrain/scratch/areaxfs3/"
    # base_dir = ('/wholebrain/scratch/pschuber/e3_trainings/lcp_semseg_j0251_dnho/semseg_pts_'
    #             'nb15000_ctx15000_dnho_nclass4_ptconv_GN_strongerWeighted_noKernelSep_eval0/')
    #
    # base_dir = ('/wholebrain/scratch/pschuber/e3_trainings/lcp_semseg_j0251_dnho/semseg_pts_nb15000_ctx15000_'
    #             'dnho_nclass4_lcp_GN_noKernelSep_AdamW_dice_eval0/')

    architecture = [dict(ic=-1, oc=1, ks=32, nn=16, np=-1),
                    dict(ic=1, oc=1, ks=32, nn=16, np=2048),
                    dict(ic=1, oc=1, ks=32, nn=16, np=1024),
                    dict(ic=1, oc=1, ks=16, nn=16, np=256),
                    dict(ic=1, oc=2, ks=16, nn=16, np=64),
                    dict(ic=2, oc=2, ks=16, nn=16, np=16),
                    dict(ic=2, oc=2, ks=16, nn=16, np=8),
                    dict(ic=2, oc=2, ks=16, nn=4, np='d'),
                    dict(ic=4, oc=2, ks=16, nn=4, np='d'),
                    dict(ic=4, oc=1, ks=16, nn=4, np='d'),
                    dict(ic=2, oc=1, ks=32, nn=8, np='d'),
                    dict(ic=2, oc=1, ks=32, nn=8, np='d'),
                    dict(ic=2, oc=1, ks=32, nn=8, np='d')]
    base_dir = ('/wholebrain/scratch/pschuber/e3_trainings/lcp_semseg_j0251_dnho/semseg_pts_nb15000_ctx15000_'
                'dnho_nclass4_lcp_GN_noKernelSep_eval0/')
    red = 5
    pred_key = 'syn_dnho_cmn'
    log = initialize_logging('dnho_eval', f'{base_dir}/{pred_key}', overwrite=False)

    if 1:
        log.info(f'Predicting ssvs {ssv_ids} from working directory "{wd}".\n'
                 f'prediction key: {pred_key}, redundancy: {red}, model path: {base_dir}')
        duration = predict_sso_thread_dnho(ssv_ids, wd,
                                           model_p=base_dir + '/state_dict.pth', pred_key=f'dnho_cmn_new',
                                           redundancy=red, out_p=f'{base_dir}/{pred_key}', architecture=architecture)
        ssd = SuperSegmentationDataset(working_dir=wd)
        vol = np.sum([ssv.size for ssv in ssd.get_super_segmentation_object(ssv_ids)]) * np.prod(ssd.scaling) / 1e9  # in um^3
        total_inference_speed = vol / duration * 3600  # µm^3/h
        log.info(f'Processing speed for "{pred_key}": {total_inference_speed:.2f} µm^3/h')

    if 1:
        report = evaluate_syn_thread_dnho(base_dir, pred_key)
        log.info(report)
