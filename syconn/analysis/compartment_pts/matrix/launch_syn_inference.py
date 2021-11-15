import os
import numpy as np
import pickle as pkl
from syconn.mp.batchjob_utils import batchjob_script

if __name__ == '__main__':
    # params = []
    # red = 5
    #
    # training_dir = os.path.expanduser('~/working_dir/paper/dasbtnh/')
    # trainings = os.listdir(training_dir)
    # durations = []
    # for training in trainings:
    #     with open(training_dir + training + '/eval_red1_valiter1_batchsize-1/eval_red1_mv.pkl', 'rb') as f:
    #         data = pkl.load(f)
    #     scores = []
    #     models = []
    #     for key in data.keys():
    #         scores.append(data[key]['total']['mv_skel']['weighted avg']['f1-score'])
    #         models.append(int(key[6:]))
    #     model = models[int(np.argmax(scores))]
    #     pred_key = f'syn_e{model}_red{red}'
    #     params.append([dict(sso_ids=[141995, 11833344, 28410880, 28479489],
    #                         wd="/wholebrain/scratch/areaxfs3/",
    #                         model_p=training_dir + training + f'/models/state_dict_e{model}.pth',
    #                         model_args_p=training_dir + training + '/argscont.pkl',
    #                         pred_key=pred_key, redundancy=red, out_p=training_dir + training + f'/{pred_key}')])
    #     durations.append((training, len(data.keys()) * 30 - 30, model, params[-1][0]['out_p']))
    #
    # for duration in durations:
    #     print(str(duration) + '\n')
    # batchjob_script(params, 'launch_syn_inference', n_cores=10,
    #                 additional_flags='--mem=125000 --gres=gpu:1',
    #                 disable_batchjob=False, max_iterations=0,
    #                 batchjob_folder='/wholebrain/u/jklimesch/working_dir/batchjobs/syn_inference/',
    #                 remove_jobfolder=False, overwrite=True, exclude_nodes=[])

    params = []
    red = 5

    training_dir = os.path.expanduser('/wholebrain/scratch/pschuber/syconn_v2_paper/supplementals/'
                                      'compartment_pts/dnh_matrix_update_cmn_ads/models/')
    trainings = os.listdir(training_dir)
    durations = []
    for training in trainings:
        pred_key = 'syn_e_final'
        params.append([dict(sso_ids=[141995, 11833344, 28410880, 28479489],
                            wd="/wholebrain/scratch/areaxfs3/",
                            model_p=training_dir + training + '/state_dict.pth',
                            model_args_p=training_dir + training + '/argscont.pkl',
                            pred_key=pred_key, redundancy=red, out_p=training_dir + training + f'/{pred_key}')])

    batchjob_script(params, 'launch_syn_inference', n_cores=20,
                    additional_flags='--mem=125000 --gres=gpu:1',
                    disable_batchjob=True, max_iterations=0,
                    batchjob_folder='/wholebrain/u/pschuber/batchjobs/syn_inference/',
                    remove_jobfolder=False, overwrite=True, exclude_nodes=[])
