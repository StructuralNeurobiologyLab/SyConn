# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max-Planck-Institute for Medical Research, Heidelberg, Germany
# Max-Planck-Institute of Neurobiology, Martinsried, Germany
# Authors: Sven Dorkenwald, Philipp Schubert, Joergen Kornfeld

# adjusted neuro3d.py
try:
    import cPickle as pkl
except ImportError:
    import pickle as pkl
import os

home = os.path.expanduser("~/")
data_class = 'BatchCreatorImage'  # <String>: Name of Data Class in TrainData or <tuple>: (path_to_file, class_name)

background_processes = 3
data_path = '/u/sdorkenw/groundtruth_paper/bird/synapse2/sparse_3d/' # Path to data dir
label_path = '/u/sdorkenw/groundtruth_paper/bird/synapse2/sparse_3d/' #'/home/sdorkenw/Data/synapse_bird/' # Path to label dir
d_files = pkl.load(open(data_path + "d_path.pkl"))   # list of tupels of file name and filed name in h5 dataset
l_files = pkl.load(open(label_path + "l_path.pkl"))
cube_prios = pkl.load(open(data_path + "prios.pkl"))

d_files = [(data_path + d_file[0], d_file[1]) for d_file in d_files]
l_files = [(data_path + l_file[0], l_file[1]) for l_file in l_files]

data_init_kwargs = dict(d_path=data_path, l_path=label_path,
                        d_files=d_files, l_files=l_files,
                        cube_prios=cube_prios)

data_batch_args = dict(grey_augment_channels=[0], warp=0,
                       warp_args={'sample_aniso': True, 'perspective': True},
                       ret_ll_mask=True)

n_steps = 50 * 1000 * 1000
max_runtime = 4 * 24 * 3600
history_freq = 2000
monitor_batch_size = 10
optimiser = 'SGD'
optimiser_params = dict(lr=0.001, mom=0.9)
batch_size = 1
schedules = {'lr': {'dec': 0.995}}
save_path = home + '/CNN_Training/MIGA/'
save_name = "bird_1036_v2"


def create_model():
    from elektronn2 import neuromancer
    in_sh = (None, 1, 111+16*5, 111+16*5, 13+4*5)
    inp = neuromancer.Input(in_sh, 'b,f,x,y,z', name='raw')

    out = neuromancer.Conv(inp, 12, (6, 6, 1), (2, 2, 1))
    out = neuromancer.Conv(out, 24, (4, 4, 1), (2, 2, 1))
    out = neuromancer.Conv(out, 36, (4, 4, 4), (2, 2, 2))
    out = neuromancer.Conv(out, 48, (4, 4, 2), (2, 2, 2))
    out = neuromancer.Conv(out, 48, (4, 4, 2), (1, 1, 1))
    out = neuromancer.Conv(out, 4, (1, 1, 1), (1, 1, 1),
                           activation_func='lin', name='sy')
    probs = neuromancer.Softmax(out)

    target = neuromancer.Input_like(probs, override_f=1, name='target')
    loss_pix = neuromancer.MultinoulliNLL(probs, target, target_is_sparse=True)

    loss = neuromancer.AggregateLoss(loss_pix, name='loss')
    errors = neuromancer.Errors(probs, target, target_is_sparse=True)

    model = neuromancer.model_manager.getmodel()
    model.designate_nodes(
        input_node=inp,
        target_node=target,
        loss_node=loss,
        prediction_node=probs,
        prediction_ext=[loss, errors, probs]
    )

    return model


if __name__ == "__main__":
    from elektronn2 import neuromancer

    model = create_model()

    # from elektronn2.utils.d3viz import visualise_model
    #
    # if not os.path.exists(home+'/models/'):
    #     os.makedirs(home+'/models/')
    #
    # visualise_model(model, home+'/models/'+save_name)
