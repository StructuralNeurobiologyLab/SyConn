#!/usr/bin/env python3

# Copyright (c) 2017 - now
# Max Planck Institute of Neurobiology, Munich, Germany
# Authors: Philipp Schubert

"""
Workflow of spinal semantic segmentation based on multiviews (2D semantic segmentation).

It learns how to differentiate between spine head, spine neck and spine shaft.
Caution! The input dataset was not manually corrected.
"""
from syconn import global_params
from syconn.cnn.TrainData import CelltypeViewsE3
import argparse
import _pickle
import os
import torch
from torch import nn
from torch import optim
from elektronn3.models.simple import StackedConv2Scalar, StackedConv2ScalarWithLatentAdd
from elektronn3.data.transforms import RandomFlip
from elektronn3.data import transforms
from elektronn3.training.schedulers import SGDR
from elektronn3.training.metrics import channel_metric
from elektronn3.training import metrics


def get_model():
    model = StackedConv2ScalarWithLatentAdd(in_channels=4, n_classes=9, n_scalar=1)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a network.')
    parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('-n', '--exp-name', default="celltype_e3_axonGTv3_SGDR_LatentAdd_run4",
                        help='Manually set experiment name')
    parser.add_argument(
        '-m', '--max-steps', type=int, default=500000,
        help='Maximum number of training steps to perform.'
    )
    parser.add_argument(
        '-r', '--resume', metavar='PATH',
        help='Path to pretrained model state dict or a compiled and saved '
             'ScriptModule from which to resume training.'
    )
    args = parser.parse_args()
    if not args.disable_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print('Running on device: {}'.format(device))
    # Don't move this stuff, it needs to be run this early to work
    import elektronn3
    elektronn3.select_mpl_backend('agg')
    from elektronn3.training import Backup
    from elektronn3.training.trainer_scalarinput import Trainer

    torch.manual_seed(0)

    # USER PATHS
    save_root = os.path.expanduser('~/e3training/')

    max_steps = args.max_steps
    lr = 0.008
    lr_stepsize = 500
    lr_dec = 0.995
    batch_size = 10

    model = get_model()
    if 0: #torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        batch_size = batch_size * torch.cuda.device_count()
        # dim = 0 [20, xxx] -> [10, ...], [10, ...] on 2 GPUs
        model = nn.DataParallel(model)
    model.to(device)
    n_classes = 9
    data_init_kwargs = {"raw_only": False, "nb_views": 20, 'train_fraction': 0.95,
                        'nb_views_renderinglocations': 4,
                        "reduce_context": 0, "reduce_context_fact": 1, 'ctgt_key': "ctgt_v2", 'random_seed': 0,
                        "binary_views": False, "n_classes": n_classes, 'class_weights': [1] * n_classes}

    if args.resume is not None:  # Load pretrained network
        try:  # Assume it's a state_dict for the model
            model.load_state_dict(torch.load(os.path.expanduser(args.resume)))
        except _pickle.UnpicklingError as exc:
            # Assume it's a complete saved ScriptModule
            model = torch.jit.load(os.path.expanduser(args.resume), map_location=device)

    # Specify data set
    transform = transforms.Compose([RandomFlip(ndim_spatial=2), ])
    train_dataset = CelltypeViewsE3(train=True, transform=transform, **data_init_kwargs)
    valid_dataset = CelltypeViewsE3(train=False, transform=transform, **data_init_kwargs)

    # Set up optimization
    optimizer = optim.SGD(
        model.parameters(),
        weight_decay=0.5e-4,
        lr=lr,
        # amsgrad=True
    )
    # lr_sched = optim.lr_scheduler.StepLR(optimizer, lr_stepsize, lr_dec)
    schedulers = {'lr': SGDR(optimizer, 20000, 3)}
    # All these metrics assume a binary classification problem. If you have
    #  non-binary targets, remember to adapt the metrics!
    val_metric_keys = []
    val_metric_vals = []
    for c in range(n_classes):
        kwargs = dict(c=c, num_classes=n_classes)
        val_metric_keys += [f'val_accuracy_c{c}', f'val_precision_c{c}', f'val_recall_c{c}', f'val_DSC_c{c}']
        val_metric_vals += [channel_metric(metrics.accuracy, **kwargs), channel_metric(metrics.precision, **kwargs),
                            channel_metric(metrics.recall, **kwargs), channel_metric(metrics.dice_coefficient, **kwargs),]
    valid_metrics = dict(zip(val_metric_keys, val_metric_vals))

    # criterion = LovaszLoss().to(device)
    criterion = nn.CrossEntropyLoss().to(device)

    # Create and run trainer
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        batchsize=batch_size,
        num_workers=0,
        save_root=save_root,
        exp_name=args.exp_name,
        schedulers=schedulers,
        valid_metrics=valid_metrics,
        ipython_shell=False,
        mixed_precision=False,  # Enable to use Apex for mixed precision training
    )

    # Archiving training script, src folder, env info
    bk = Backup(script_path=__file__, save_path=trainer.save_path).archive_backup()

    trainer.run(max_steps)
