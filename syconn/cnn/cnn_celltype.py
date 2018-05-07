# -*- coding: utf-8 -*-
# Neuromancer Toolkit
# Copyright (c) 2016 Philipp J. Schubert
# All rights reserved+
import os
import syconn

save_path = '~/CNN_Training/SyConn/celltype/'
# preview_data_path = None
# preview_kwargs    = dict(export_class='all', max_z_pred=5)
# initial_prev_h   = 0.5                  # hours: time after which first preview is made
# prev_save_h      = 1.0
home = os.path.expanduser("~/")
data_class = (os.path.split(syconn.__file__)[0] + '/cnn/TrainData.py',
              'SSVCelltype')
background_processes = 8

n_steps = 200000
max_runtime = 4 * 24 * 3600 # in seconds
history_freq = 200
monitor_batch_size = 48
optimiser = 'Adam'
nb_views = 20
save_name = "g1_%dviews_v3_rawonly" % nb_views
data_batch_args = {}
data_init_kwargs = {"nb_views": nb_views, "nb_cpus": 1, "raw_only": True}
optimiser_params = dict(lr=10e-4, mom=0.9, wd=0.5e-3, beta2=0.99)
batch_size = 12
schedules = {"lr": {"dec": 0.98}}#{'lr': {'updates': [(2000, 10e-4), (3000, 8e-4), (5000, 6e-4), (7500, 4e-4), (15000, 2e-4)]}}
dr = 0.08

def create_model():
    from elektronn2 import neuromancer
    import theano

    act = 'relu'
    in_sh = (batch_size, 1, nb_views, 128, 256)
    inp = neuromancer.Input(in_sh, 'b,f,z,y,x', name='raw')
    out0 = neuromancer.Conv(inp, 13, (1, 5, 5), (1, 2, 2), activation_func=act, dropout_rate=dr)
    out0 = neuromancer.Conv(out0, 19, (1, 5, 5), (1, 2, 2), activation_func=act, dropout_rate=dr)
    out0 = neuromancer.Conv(out0, 25, (1, 4, 4), (1, 2, 2), activation_func=act, dropout_rate=dr)
    out0 = neuromancer.Conv(out0, 25, (1, 4, 4), (1, 2, 2), activation_func=act, dropout_rate=dr)
    out0 = neuromancer.Conv(out0, 30, (1, 2, 2), (1, 2, 2), activation_func=act, dropout_rate=dr)
    out0 = neuromancer.Conv(out0, 30, (1, 1, 1), (1, 2, 2), activation_func=act, dropout_rate=dr)
    out = neuromancer.Conv(out0, 31, (1, 1, 1), (1, 1, 1), activation_func=act, dropout_rate=dr)

    out = neuromancer.Perceptron(out, 50, flatten=True, dropout_rate=dr)
    out = neuromancer.Perceptron(out, 30, flatten=True, dropout_rate=dr)
    out = neuromancer.Perceptron(out, 4, activation_func='lin')
    out = neuromancer.Softmax(out)
    target = neuromancer.Input_like(out, override_f=1, name='target')
    weights = neuromancer.ValueNode((4,), 'f', value=(1, 1, 2, 2))
    loss = neuromancer.MultinoulliNLL(out, target, name='nll_',
                                      target_is_sparse=True,class_weights=weights)

    # Objective
    loss = neuromancer.AggregateLoss(loss)
    # Monitoring  / Debug outputs
    errors = neuromancer.Errors(out, target, target_is_sparse=True)

    model = neuromancer.model_manager.getmodel()
    model.designate_nodes(input_node=inp, target_node=target, loss_node=loss,
                          prediction_node=out,
                          prediction_ext=[loss, errors, out])
    return model