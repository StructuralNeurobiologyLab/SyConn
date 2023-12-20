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
import zipfile
import torch
from elektronn3 import logger

from torch import nn
from torch import optim
try:
    from elektronn3.modules.loss import DiceLoss
except ImportError:
    from elektronn3.training.loss import DiceLoss
from elektronn3.training import metrics
from elektronn3.models.fcn_2d import *
from elektronn3.data.transforms import RandomFlip
from elektronn3.data import transforms


def get_model():
    """
    This function creates and returns a model for semantic segmentation of spine based on multiviews.
    The model is a Fully Convolutional Network (FCN) with a VGGNet as the base network.
    
    Returns:
        model (torch.nn.Module): The FCN model with VGGNet as the base network.
    """
    vgg_model = VGGNet(model='vgg13', requires_grad=True, in_channels=4)
    model = FCNs(base_net=vgg_model, n_class=6)
    # model = UNet(in_channels=4, out_channels=6, n_blocks=5, start_filts=32,
    #              merge_mode='concat', planar_blocks=(), #up_mode='resize',
    #              activation='relu', batch_norm=True, dim=2,)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a network.')
    parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('-n', '--exp-name',
                        default="axonseg-FCN-Dice-40nmGT-BOUTON_v4_biggerbatch_2gpus_run3_resumed",
                        help='Manually set experiment name')
    parser.add_argument(
        '-m', '--max-steps', type=int, default=5000000,
        help='Maximum number of training steps to perform.'
    )
    parser.add_argument('--num-repeat', type=int, default=1,
                    help='Specify how many times each datapoint be used before corresponding h5 file is released')
    parser.add_argument('-r', '--resume', metavar='PATH',help='Path to pretrained model state dict or a compiled and saved '
                                                                'ScriptModule from which to resume training.')
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
    lr = 0.0012
    lr_stepsize = 500
    lr_dec = 0.995
    batch_size = 4

    model = get_model()

    optimizer_state_dict = None
    lr_sched_state_dict = None
    if args.resume is not None:  # Load pretrained network
        pretrained = os.path.expanduser(args.resume)
        _warning_str = 'Loading model without optimizer state. Prefer state dicts'
        if zipfile.is_zipfile(pretrained):  # Zip file indicates saved ScriptModule
            logger.warning(_warning_str)
            model = torch.jit.load(pretrained, map_location=device)
        else:  # Either state dict or pickled model
            state = torch.load(pretrained)
            if isinstance(state, dict):
                model.load_state_dict(state['model_state_dict'])
                optimizer_state_dict = state.get('optimizer_state_dict')
                lr_sched_state_dict = state.get('lr_sched_state_dict')
                if optimizer_state_dict is None:
                    logger.warning('optimizer_state_dict not found.')
                if lr_sched_state_dict is None:
                    logger.warning('lr_sched_state_dict not found.')
            elif isinstance(state, nn.Module):
                logger.warning(_warning_str)
                model = state
            else:
                raise ValueError(f'Can\'t load {pretrained}.')

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        batch_size = batch_size * torch.cuda.device_count()
        # dim = 0 [20, xxx] -> [10, ...], [10, ...] on 2 GPUs
        model = nn.DataParallel(model)

    model.to(device)
    # pdb.set_trace()

    # Set up optimization
    optimizer = optim.Adam(
        model.parameters(),
        weight_decay=0.5e-4,
        lr=lr,
        amsgrad=True
    )
    # optimizer = optim.SGD(
    #     model.parameters(),
    #     weight_decay=0.5e-4,
    #     lr=lr,
    # )

    # lr_sched = optim.lr_scheduler.ExponentialLR(optimizer, 0.9998)
    lr_sched = optim.lr_scheduler.StepLR(optimizer, lr_stepsize, lr_dec)

    if optimizer_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)
    if lr_sched_state_dict is not None:
        lr_sched.load_state_dict(lr_sched_state_dict)

    # Specify data set
    transform = transforms.Compose([RandomFlip(ndim_spatial=2), ])
    # global_params.config['compartments']['gt_path_axonseg'] = '/wholebrain/scratch/areaxfs3/ssv_semsegaxoness/gt_h5_files_80nm_1024'
    # global_params.config['compartments']['gt_path_axonseg'] = '/wholebrain/scratch/areaxfs3/ssv_semsegaxoness/gt_h5_files_80nm_1024_with_BOUTONS'
    global_params.config['compartments']['gt_path_axonseg'] = \
        '/wholebrain/scratch/areaxfs3/ssv_semsegaxoness/all_bouton_data_40nm/'
    # num_workers must be <=1 for MultiviewDataCached class
    train_dataset = MultiviewDataCached(base_dir=global_params.config['compartments']['gt_path_axonseg']+'',
                                        train=True, inp_key='raw', target_key='label',
                                        transform=transform, num_read_limit=args.num_repeat)
    valid_dataset = MultiviewDataCached(base_dir=global_params.config['compartments']['gt_path_axonseg']+'',
                                        train=False, inp_key='raw', target_key='label',
                                        transform=transform, num_read_limit=1)
    # ic(train_dataset.__len__(), valid_dataset.__len__())
    # train_dataset = ModMultiviewData(train=True, transform=transform, base_dir=global_params.config['compartments']['gt_path_axonseg'])
    # valid_dataset = ModMultiviewData(train=False, transform=transform, base_dir=global_params.config['compartments']['gt_path_axonseg'])

    # criterion = LovaszLoss().to(device)
    criterion = DiceLoss().to(device)

    valid_metrics = {
    # 'val_accuracy': metrics.bin_accuracy,
    'val_precision': metrics.bin_precision,
    'val_recall': metrics.bin_recall,
    # 'val_DSC': metrics.bin_dice_coefficient,
    'val_IoU': metrics.bin_iou,
    # 'val_AP': metrics.bin_average_precision,  # expensive
    # 'val_AUROC': metrics.bin_auroc,  # expensive
    }

    # Create and run trainer
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        batchsize=batch_size,
        num_workers=1,
        save_root=save_root,
        exp_name=args.exp_name,
        schedulers={"lr": lr_sched},
        valid_metrics=valid_metrics,
        ipython_shell=False,
        mixed_precision=False,  # Enable to use Apex for mixed precision training
    )

    # Archiving training script, src folder, env info
    bk = Backup(script_path=__file__, save_path=trainer.save_path).archive_backup()

    trainer.run(max_steps)
