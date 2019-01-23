# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max-Planck-Institute of Neurobiology, Munich, Germany
# Authors: Philipp Schubert, Joergen Kornfeld

import os
import syconn

save_path = '~/CNN_Training/SyConn/glia_views/'
save_name = "g0_v0"

preview_data_path = None
preview_kwargs    = dict(export_class='all', max_z_pred=5)
initial_prev_h   = 0.5                  # hours: time after which first preview is made
prev_save_h      = 1.0
home = os.path.expanduser("~/")
data_class = (os.path.split(syconn.__file__)[0] + '/cnn/TrainData.py',
              'GliaViews')
background_processes = 4

n_steps = 350000
max_runtime = 4 * 24 * 3600 # in seconds
history_freq = 300
monitor_batch_size = 240
optimiser = 'Adam'
data_batch_args = {}
data_init_kwargs = {"nb_views": 2, "reduce_context": 0,
                    "reduce_context_fact": 1}
optimiser_params = dict(lr=10e-4, mom=0.9, wd=0.5e-3, beta2=0.99)
batch_size = 40
schedules = {"lr": {"dec": 0.99}}#{'lr': {'updates': [(10000, 8e-4), (40000, 7e-4), (100000, 6e-4), (180000, 4e-4), (300000, 2e-4)]}}
dr = 0.1

x = 128
y = 256

if data_init_kwargs["reduce_context"] > 0:
    assert data_init_kwargs["reduce_context_fact"] == 1
    x = 128 - data_init_kwargs["reduce_context"]
    y = 256 - 2*data_init_kwargs["reduce_context"]

if data_init_kwargs["reduce_context_fact"] > 1:
    assert data_init_kwargs["reduce_context"] == 0
    x = 128 / data_init_kwargs["reduce_context_fact"]
    y = 256 / data_init_kwargs["reduce_context_fact"]

def create_model():
    from elektronn2 import neuromancer

    act = 'relu'
    in_sh = (batch_size, 1, data_init_kwargs["nb_views"], int(x), int(y))
    inp = neuromancer.Input(in_sh, 'b,f,z,y,x', name='raw')

    out0 = neuromancer.Conv(inp, 13, (1, 5, 5), (1, 2, 2), activation_func=act, dropout_rate=dr)
    out0 = neuromancer.Conv(out0, 17, (1, 5, 5), (1, 2, 2), activation_func=act, dropout_rate=dr)
    out0 = neuromancer.Conv(out0, 21, (1, 4, 4), (1, 2, 2), activation_func=act, dropout_rate=dr)
    out0 = neuromancer.Conv(out0, 25, (1, 4, 4), (1, 2, 2), activation_func=act, dropout_rate=dr)
    out0 = neuromancer.Conv(out0, 29, (1, 2, 2), (1, 2, 2), activation_func=act, dropout_rate=dr)
    out0 = neuromancer.Conv(out0, 30, (1, 1, 1), (1, 2, 2), activation_func=act, dropout_rate=dr)
    out = neuromancer.Conv(out0, 31, (1, 1, 1), (1, 1, 1), activation_func=act, dropout_rate=dr)

    out = neuromancer.Perceptron(out, 50, flatten=True, dropout_rate=dr)
    out = neuromancer.Perceptron(out, 30, flatten=True, dropout_rate=dr)
    out = neuromancer.Perceptron(out, 2, activation_func='lin')
    out = neuromancer.Softmax(out)
    target = neuromancer.Input_like(out, override_f=1, name='target')
    weights = neuromancer.ValueNode((2, ), 'f', value=(4, 1))
    loss = neuromancer.MultinoulliNLL(out, target, name='nll_',
                                      target_is_sparse=True,
                                      class_weights=weights)
    # Objective
    loss = neuromancer.AggregateLoss(loss)
    # Monitoring  / Debug outputs
    errors = neuromancer.Errors(out, target, target_is_sparse=True)

    model = neuromancer.model_manager.getmodel()
    model.designate_nodes(input_node=inp, target_node=target, loss_node=loss,
                          prediction_node=out,
                          prediction_ext=[loss, errors, out])
    return model