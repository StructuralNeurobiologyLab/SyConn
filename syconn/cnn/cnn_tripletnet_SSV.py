# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max-Planck-Institute of Neurobiology, Munich, Germany
# Authors: Philipp Schubert, Joergen Kornfeld

import os
import syconn

save_path = '~/CNN_Training/SyConn/triplet_net_SSV/'
nb_views = 1
save_name = "wholecell_orthoviews_v6"

# preview_data_path = None
# preview_kwargs    = dict(export_class='all', max_z_pred=5)
# initial_prev_h   = 0.5                  # hours: time after which first preview is made
# prev_save_h      = 1.0
home = os.path.expanduser("~/")
data_class = (os.path.split(syconn.__file__)[0] + '/cnn/TrainData.py',
              'TripletData_SSV')
background_processes = 10

n_steps = 5200000
max_runtime = 4 * 24 * 3600 # in seconds
history_freq = 100
monitor_batch_size = 6
optimiser = 'SGD'
data_batch_args = {}
data_init_kwargs = {"downsample": 2}#{"nb_views": nb_views}
optimiser_params = dict(lr=0.0001, mom=0.9, wd=0.5e-3)#, beta2=0.99)
batch_size = 6
dr = 0.1
schedules = {"lr": {"dec": 0.98}}
alpha = 1e-7#1e-6

def create_model():
    from elektronn2 import neuromancer

    act = 'relu'
    in_sh = (batch_size, 4, nb_views*3, 512, 512)
    inp = neuromancer.Input(in_sh, 'b,f,z,y,x', name='raw')

    out0 = neuromancer.Conv(inp, 13, (1, 5, 5), (1, 2, 2), activation_func=act, dropout_rate=dr)
    out0 = neuromancer.Conv(out0, 19, (1, 5, 5), (1, 2, 2), activation_func=act, dropout_rate=dr)
    out0 = neuromancer.Conv(out0, 25, (1, 4, 4), (1, 2, 2), activation_func=act, dropout_rate=dr)
    out0 = neuromancer.Conv(out0, 25, (1, 4, 4), (1, 2, 2), activation_func=act, dropout_rate=dr)
    out0 = neuromancer.Conv(out0, 30, (1, 2, 2), (1, 2, 2), activation_func=act, dropout_rate=dr)
    out0 = neuromancer.Conv(out0, 30, (1, 1, 1), (1, 2, 2), activation_func=act, dropout_rate=dr)
    out = neuromancer.Conv(out0, 31, (1, 1, 1), (1, 1, 1), activation_func=act, dropout_rate=dr)
    out0, out1, out2 = neuromancer.split(out, axis="z", n_out=3)
    out0 = neuromancer.Reshape(out0, shape=(inp.shape[0], out0.shape.stripbatch_prod, 1), tags="b,f,z")
    out1 = neuromancer.Reshape(out1, shape=(inp.shape[0], out1.shape.stripbatch_prod, 1), tags="b,f,z")
    out2 = neuromancer.Reshape(out2, shape=(inp.shape[0], out2.shape.stripbatch_prod, 1), tags="b,f,z")
    out = neuromancer.Concat([out0, out1, out2], axis="z")
    out = neuromancer.Perceptron(out, 40, flatten=False,  dropout_rate=dr)
    out = neuromancer.Perceptron(out, 10, flatten=False,  dropout_rate=dr)
    out0, out1, out2 = neuromancer.split(out, axis="z", n_out=3)
    d_small = neuromancer.EuclideanDistance(out0, out1)
    d_big = neuromancer.EuclideanDistance(out0, out2)
    loss = neuromancer.RampLoss(d_small, d_big, margin=0.1)
    # reg = neuromancer.loss.L2Reg(out0, out1, out2)
    # loss = neuromancer.AggregateLoss([loss, reg], mixing_weights=[1, alpha])
    loss = neuromancer.AggregateLoss([loss,])
    model = neuromancer.model_manager.getmodel()
    # model = neuromancer.model.modelload("/wholebrain/scratch/pschuber/CNN_Training/nupa_cnn/t_net/ssv6_tripletnet_v9/ssv6_tripletnet_v9-FINAL.mdl")
    model.designate_nodes(input_node=inp, target_node=None, loss_node=loss,
                          prediction_node=out,
                          prediction_ext=[loss, loss, out])

    # params = neuromancer.model.params_from_model_file("/wholebrain/scratch/pschuber/CNN_Training/nupa_cnn/t_net/ssv6_tripletnet_v9/ssv6_tripletnet_v9-FINAL.mdl")
    # params = dict(filter(lambda x: x[0].startswith('conv'), params.items()))
    # model.set_param_values(params)
    return model

# if __name__ == "__main__":
    # model = create_model()
    # "Test" if model is saveable
    # model.save("/tmp/"+save_name)