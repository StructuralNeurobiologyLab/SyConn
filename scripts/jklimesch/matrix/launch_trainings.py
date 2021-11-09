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

architecture_512 = [{'ic': -1, 'oc': 1, 'ks': 16, 'nn': 32, 'np': -1},
                    {'ic': 1, 'oc': 1, 'ks': 16, 'nn': 32, 'np': 256},
                    {'ic': 1, 'oc': 2, 'ks': 16, 'nn': 32, 'np': 64},
                    {'ic': 2, 'oc': 2, 'ks': 16, 'nn': 16, 'np': 16},
                    {'ic': 2, 'oc': 2, 'ks': 16, 'nn': 8, 'np': 8},
                    {'ic': 2, 'oc': 2, 'ks': 16, 'nn': 4, 'np': 'd'},
                    {'ic': 4, 'oc': 2, 'ks': 16, 'nn': 4, 'np': 'd'},
                    {'ic': 4, 'oc': 1, 'ks': 16, 'nn': 8, 'np': 'd'},
                    {'ic': 2, 'oc': 1, 'ks': 16, 'nn': 16, 'np': 'd'}]

architecture_1024 = [{'ic': -1, 'oc': 1, 'ks': 16, 'nn': 32, 'np': -1},
                     {'ic': 1, 'oc': 1, 'ks': 16, 'nn': 32, 'np': 512},
                     {'ic': 1, 'oc': 1, 'ks': 16, 'nn': 32, 'np': 256},
                     {'ic': 1, 'oc': 2, 'ks': 16, 'nn': 32, 'np': 64},
                     {'ic': 2, 'oc': 2, 'ks': 16, 'nn': 16, 'np': 16},
                     {'ic': 2, 'oc': 2, 'ks': 16, 'nn': 8, 'np': 8},
                     {'ic': 2, 'oc': 2, 'ks': 16, 'nn': 4, 'np': 'd'},
                     {'ic': 4, 'oc': 2, 'ks': 16, 'nn': 4, 'np': 'd'},
                     {'ic': 4, 'oc': 1, 'ks': 16, 'nn': 8, 'np': 'd'},
                     {'ic': 2, 'oc': 1, 'ks': 16, 'nn': 16, 'np': 'd'},
                     {'ic': 2, 'oc': 1, 'ks': 16, 'nn': 16, 'np': 'd'}]

architecture_2048 = [{'ic': -1, 'oc': 1, 'ks': 16, 'nn': 32, 'np': -1},
                     {'ic': 1, 'oc': 1, 'ks': 16, 'nn': 32, 'np': 1024},
                     {'ic': 1, 'oc': 1, 'ks': 16, 'nn': 32, 'np': 512},
                     {'ic': 1, 'oc': 1, 'ks': 16, 'nn': 32, 'np': 256},
                     {'ic': 1, 'oc': 2, 'ks': 16, 'nn': 32, 'np': 64},
                     {'ic': 2, 'oc': 2, 'ks': 16, 'nn': 16, 'np': 16},
                     {'ic': 2, 'oc': 2, 'ks': 16, 'nn': 8, 'np': 8},
                     {'ic': 2, 'oc': 2, 'ks': 16, 'nn': 4, 'np': 'd'},
                     {'ic': 4, 'oc': 2, 'ks': 16, 'nn': 4, 'np': 'd'},
                     {'ic': 4, 'oc': 1, 'ks': 16, 'nn': 8, 'np': 'd'},
                     {'ic': 2, 'oc': 1, 'ks': 16, 'nn': 16, 'np': 'd'},
                     {'ic': 2, 'oc': 1, 'ks': 16, 'nn': 16, 'np': 'd'},
                     {'ic': 2, 'oc': 1, 'ks': 16, 'nn': 16, 'np': 'd'}]

architecture_large = [{'ic': -1, 'oc': 1, 'ks': 16, 'nn': 32, 'np': -1},
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
                      {'ic': 2, 'oc': 1, 'ks': 16, 'nn': 16, 'np': 'd'}]

if __name__ == '__main__':
    today = date.today().strftime("%Y_%m_%d")
    params = []

    contexts = [1, 2, 4, 8, 12, 16, 20, 24, 28, 32]
    points = [512, 1024, 2048, 4096, 8192, 16384, 32768]
    # matrix = [[True, True, True, True, True, True, True, True, True, True],
    #           [True, True, True, True, True, True, True, True, True, True],
    #           [False, True, True, True, True, True, True, True, True, True],
    #           [False, False, True, True, True, True, True, True, True, True],
    #           [False, False, False, True, True, True, True, True, True, True],
    #           [False, False, False, False, True, True, True, True, True, True],
    #           [False, False, False, False, False, False, False, True, True, True]]
    matrix = [[True, True, True, True, True, True, True, True, True, True],
              [True, True, True, True, True, True, True, True, True, True],
              [False, True, True, True, True, True, True, True, True, True],
              [False, False, True, True, True, True, True, True, True, True],
              [False, False, False, True, True, True, True, True, True, True],
              [False, False, False, False, True, True, True, True, True, True],
              [False, False, False, False, False, False, False, True, True, True]]
    stop_epochs = [1500, 1500, 2000, 3000, 3000, 3000, 3000, 3000, 3000, 3000]

    for i in [1, 2, 3]:
        for sample_num in [8192]:
            for chunk_size in [4000]:

                architecture = architecture_large
                if sample_num < 1024:
                    architecture = architecture_512
                if sample_num < 2048:
                    architecture = architecture_1024
                if sample_num == 2048:
                    architecture = architecture_2048

                batch_size = 4
                if sample_num <= 16384:
                    batch_size = 8
                if sample_num <= 8192:
                    batch_size = 16
                if sample_num < 2048:
                    batch_size = 32

                name = today + '_{}'.format(chunk_size) + '_{}'.format(sample_num)
                argscont = ArgsContainer(save_root='/u/jklimesch/working_dir/batchjobs/',
                                         train_path='/u/jklimesch/working_dir/gt/cmn/dnh/voxeled/',
                                         sample_num=sample_num,
                                         name=name + f'_{i}',
                                         random_seed=i,
                                         class_num=3,
                                         train_transforms=[clouds.RandomVariation((-40, 40)),
                                                           clouds.RandomRotate(apply_flip=True),
                                                           clouds.Center(),
                                                           clouds.ElasticTransform(res=(40, 40, 40), sigma=(6, 6)),
                                                           clouds.RandomScale(distr_scale=0.1, distr='uniform'),
                                                           clouds.Center()],
                                         batch_size=batch_size,
                                         input_channels=1,
                                         use_val=True,
                                         val_path='/u/jklimesch/working_dir/gt/cmn/dnh/voxeled/evaluation/',
                                         val_freq=30,
                                         features={'hc': np.array([1])},
                                         chunk_size=chunk_size,
                                         stop_epoch=3000,
                                         max_step_size=100000000,
                                         hybrid_mode=True,
                                         splitting_redundancy=5,
                                         norm_type='gn',
                                         label_remove=[2],
                                         label_mappings=[(2, 0), (5, 1), (6, 2)],
                                         val_label_mappings=[(5, 1), (6, 2)],
                                         val_label_remove=[-2, 1, 2, 3, 4],
                                         architecture=architecture,
                                         target_names=['dendrite', 'neck', 'head'])
                params.append([argscont])

    batchjob_script(params, 'launch_neuronx_training', n_cores=10,
                    additional_flags='--mem=125000 --gres=gpu:1',
                    disable_batchjob=False, max_iterations=0,
                    batchjob_folder='/wholebrain/u/jklimesch/working_dir/batchjobs/dnh_trainings_new/',
                    remove_jobfolder=False, overwrite=True, exclude_nodes=[])
