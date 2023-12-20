#!/usr/bin/env python3

# Copyright (c) 2017 - now
# Max Planck Institute of Neurobiology, Munich, Germany
# Authors: Philipp Schubert

"""
Workflow of spinal semantic segmentation based on multiviews (2D semantic segmentation).

It learns how to differentiate between spine head, spine neck and spine shaft.
Caution! The input dataset was not manually corrected.
"""
from syconn.cnn.TrainData import AxonsViewsE3
import argparse
import os
import torch
from torch import nn
from torch import optim
from elektronn3.training import Trainer, Backup, metrics, Padam
from elektronn3.models.simple import StackedConv2Scalar
from elektronn3.data.transforms import RandomFlip
from elektronn3.data import transforms
from elektronn3.training.schedulers import SGDR
from elektronn3.training.metrics import channel_metric


def get_model():
    """
    This function creates and returns a model of type StackedConv2Scalar with 4 input channels
    and 3 output channels.
    
    Returns:
        model (StackedConv2Scalar): The created model.
    """
    model = StackedConv2Scalar(4, 3)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a network.')
    parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('-n', '--exp-name', default="axon_e3_axonGT_run3_SGDR",
                        help='Manually set experiment name')
    parser.add_argument(
        '-m', '--max-steps', type=int, default=5000000,
        help='Maximum number of training steps to perform.'
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
    from elektronn3.training import Trainer, Backup

    torch.manual_seed(0)

    # USER PATHS
    save_root = os.path.expanduser('~/e3training/')

    max_steps = args.max_steps
    lr = 0.006
    lr_stepsize = 500
    lr_dec = 0.995
    batch_size = 5

    model = get_model()
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        batch_size = batch_size * torch.cuda.device_count()
        # dim = 0 [20, xxx] -> [10, ...], [10, ...] on 2 GPUs
        model = nn.DataParallel(model)
    model.to(device)
    n_classes = 3
    data_init_kwargs = {"channels_to_load": (0, 1, 2, 3), "nb_views": 2, }
    # Specify data set
    transform = transforms.Compose([RandomFlip(ndim_spatial=2), ])
    train_dataset = AxonsViewsE3(train=True, transform=transform, **data_init_kwargs)
    valid_dataset = AxonsViewsE3(train=False, transform=transform, **data_init_kwargs)
    
    # Set up optimization
    optimizer = optim.SGD(
        model.parameters(),
        weight_decay=0.5e-4,
        lr=lr,
        # amsgrad=True
    )
    # lr_sched = optim.lr_scheduler.StepLR(optimizer, lr_stepsize, lr_dec)
    lr_sched = optim.lr_scheduler.StepLR(optimizer, lr_stepsize, lr_dec)
    schedulers = {'lr': SGDR(optimizer, 10000, 2)}
    
    # metrics for non binary classification
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
