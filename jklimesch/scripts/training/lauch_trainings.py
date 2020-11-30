# do not remove open3d as import order of open3d and torch is important
import open3d as o3d
import torch
import numpy as np
from torch import nn
from datetime import date
import morphx.processing.clouds as clouds
import elektronn3
elektronn3.select_mpl_backend('Agg')
from neuronx.classes.argscontainer import ArgsContainer
from syconn.mp.batchjob_utils import batchjob_script

if __name__ == '__main__':
    today = date.today().strftime("%Y_%m_%d")
    params = []

    for ix, sample_num in enumerate([16384]):
        for chunk_size in [8000, 24000]:
            batch_size = 8
            name = today + '_{}'.format(chunk_size) + '_{}'.format(sample_num)
            argscont = ArgsContainer(save_root='/u/jklimesch/working_dir/batchjob_test/',
                                     train_path='/u/jklimesch/thesis/gt/cmn/dnh/voxeled/',
                                     sample_num=sample_num,
                                     name=name + f'_cp_cp_q_{ix}',
                                     random_seed=ix,
                                     class_num=3,
                                     train_transforms=[clouds.RandomVariation((-40, 40)), clouds.RandomRotate(apply_flip=True),
                                                       clouds.Center(), clouds.ElasticTransform(res=(40, 40, 40), sigma=(6, 6)),
                                                       clouds.RandomScale(distr_scale=0.1, distr='uniform'), clouds.Center()],
                                     batch_size=batch_size,
                                     input_channels=1,
                                     use_val=True,
                                     val_path='/u/jklimesch/thesis/gt/cmn/dnh/voxeled/evaluation/',
                                     val_freq=30,
                                     features={'hc': np.array([1])},
                                     chunk_size=chunk_size,
                                     max_step_size=50000*batch_size,
                                     hybrid_mode=True,
                                     splitting_redundancy=5,
                                     norm_type='gn',
                                     label_remove=[2],
                                     label_mappings=[(2, 0), (5, 1), (6, 2)],
                                     architecture=[{'ic': -1, 'oc': 1, 'ks': 16, 'nn': 32, 'np': -1},
                                                   {'ic': 1, 'oc': 1, 'ks': 16, 'nn': 32, 'np': 2048},
                                                   {'ic': 1, 'oc': 1, 'ks': 16, 'nn': 32, 'np': 1024},
                                                   {'ic': 1, 'oc': 1, 'ks': 16, 'nn': 32, 'np': 256},
                                                   {'ic': 1, 'oc': 2, 'ks': 16, 'nn': 32, 'np': 64},
                                                   {'ic': 2, 'oc': 2, 'ks': 16, 'nn': 16, 'np': 16},
                                                   {'ic': 2, 'oc': 2, 'ks': 16, 'nn': 8, 'np': 8},
                                                   {'ic': 2, 'oc': 2, 'ks': 16, 'nn': 4, 'np': 'd'},
                                                   {'ic': 4, 'oc': 2, 'ks': 16, 'nn': 4, 'np': 'd'},
                                                   {'ic': 4, 'oc': 1, 'ks': 16, 'nn': 8, 'np': 'd'},
                                                   {'ic': 2, 'oc': 1, 'ks': 16, 'nn': 16, 'np': 'd'},
                                                   {'ic': 2, 'oc': 1, 'ks': 16, 'nn': 16, 'np': 'd'},
                                                   {'ic': 2, 'oc': 1, 'ks': 16, 'nn': 16, 'np': 'd'}],
                                     target_names=['dendrite', 'neck', 'head'])
            params.append([argscont])

    batchjob_script(params, 'launch_neuronx_training', n_cores=10, additional_flags='--time=7-0 --qos=720h --gres=gpu:1',
                    disable_batchjob=False, batchjob_folder='/wholebrain/u/jklimesch/working_dir/batchjobs/dnh_trainings/',
                    remove_jobfolder=True, overwrite=True)
