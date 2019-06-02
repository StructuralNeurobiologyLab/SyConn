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
from syconn.cnn.TrainData import MultiviewDataCached
import argparse
import os
import torch
from torch import nn
from torch import optim
from torch.utils.data.dataset import random_split
from elektronn3.training.loss import DiceLoss, LovaszLoss
from elektronn3.models.fcn_2d import *
from elektronn3.models.unet import UNet
from elektronn3.data.transforms import RandomFlip
from elektronn3.data import transforms

from icecream import ic

def get_model():
    vgg_model = VGGNet(model='vgg13', requires_grad=True, in_channels=4)
    model = FCNs(base_net=vgg_model, n_class=4)
    # model = UNet(in_channels=4, out_channels=4, n_blocks=5, start_filts=32,
    #              up_mode='resize', merge_mode='concat', planar_blocks=(),
    #              activation='relu', batch_norm=True, dim=2,)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a network.')
    parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('-n', '--exp-name', default="axonseg-FCN-Dice-resizeconv-80nmGT",
                        help='Manually set experiment name')
    parser.add_argument(
        '-m', '--max-steps', type=int, default=500000,
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
    lr = 0.0048
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

    # Specify data set
    transform = transforms.Compose([RandomFlip(ndim_spatial=2), ])
    global_params.gt_path_axonseg = '/wholebrain/scratch/areaxfs3/ssv_semsegaxoness/gt_h5_files_80nm_1024'
    
    train_dataset = MultiviewDataCached(base_dir=global_params.gt_path_axonseg+'/train', 
                                        train=True, inp_key='raw', target_key='label', 
                                        transform=transform, num_read_limit=5)
    valid_dataset = MultiviewDataCached(base_dir=global_params.gt_path_axonseg+'/val', 
                                        train=False, inp_key='raw', target_key='label', 
                                        transform=transform, num_read_limit=5)
    # Set up optimization
    optimizer = optim.Adam(
        model.parameters(),
        weight_decay=0.5e-4,
        lr=lr,
        amsgrad=True
    )
    lr_sched = optim.lr_scheduler.StepLR(optimizer, lr_stepsize, lr_dec)

    # criterion = LovaszLoss().to(device)
    criterion = DiceLoss().to(device)

    # Create and run trainer
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        batchsize=batch_size,
        num_workers=2,
        save_root=save_root,
        exp_name=args.exp_name,
        schedulers={"lr": lr_sched},
        ipython_shell=False,
        mixed_precision=False,  # Enable to use Apex for mixed precision training
    )

    # Archiving training script, src folder, env info
    bk = Backup(script_path=__file__, save_path=trainer.save_path).archive_backup()

    trainer.run(max_steps)
