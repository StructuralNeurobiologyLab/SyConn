# do not remove open3d as import order of open3d and torch is important
import numpy as np
from datetime import date
import morphx.processing.clouds as clouds
import elektronn3

elektronn3.select_mpl_backend('Agg')
from neuronx.classes.argscontainer import ArgsContainer
from syconn.mp.batchjob_utils import batchjob_script


if __name__ == '__main__':
    today = date.today().strftime("%Y_%m_%d")
    params = []

    contexts = [1, 2, 4, 8, 12, 16, 20, 24, 28, 32]
    points = [512, 1024, 2048, 4096, 8192, 16384, 32768]

    stop_epochs = [1500, 1500, 2000, 3000, 3000, 3000, 3000, 3000, 3000, 3000]

    for i in [1, 2, 3]:
        for sample_num in [8192]:
            for chunk_size in [4000]:

                batch_size = 4

                name = today + '_{}'.format(chunk_size) + '_{}'.format(sample_num)
                argscont = ArgsContainer(save_root='/wholebrain/scratch/pschuber/compartments_j0251/models_refined01/',
                                         train_path='/wholebrain/scratch/pschuber/compartments_j0251/hybrid_clouds_refined01/train/',
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
                                         model='randla_net',
                                         val_path='/wholebrain/scratch/pschuber/compartments_j0251/hybrid_clouds_refined01/test/',
                                         val_freq=30,
                                         features={'hc': np.array([1])},
                                         chunk_size=chunk_size,
                                         stop_epoch=3000,
                                         max_step_size=200e3,
                                         hybrid_mode=True,
                                         splitting_redundancy=5,
                                         label_remove=[2],
                                         label_mappings=[(2, 0), (5, 1), (6, 2)],
                                         val_label_mappings=[(5, 1), (6, 2)],
                                         val_label_remove=[-2, 1, 2, 3, 4],
                                         target_names=['dendrite', 'neck', 'head'])
                params.append([argscont])

    batchjob_script(params, 'launch_neuronx_training', n_cores=10,
                    additional_flags='--mem=125000 --gres=gpu:1',
                    disable_batchjob=False, max_iterations=0,
                    batchjob_folder='/wholebrain/scratch/pschuber/batchjobs/dnh_trainings_Apr_2021/',
                    remove_jobfolder=False, overwrite=True, exclude_nodes=[])
