# ELEKTRONN3 - Neural Network Toolkit
#
# Copyright (c) 2019 - now
# Max Planck Institute of Neurobiology, Munich, Germany
# Authors: Philipp Schubert
from syconn.cnn.TrainData import CellCloudDataTriplet

import os
import torch
import argparse
import random
import numpy as np
# Don't move this stuff, it needs to be run this early to work
import elektronn3
elektronn3.select_mpl_backend('Agg')
import morphx.processing.clouds as clouds
from torch import nn
from elektronn3.models.convpoint import ModelNet40, ModelNetBig, ModelNetAttention
from elektronn3.training import Trainer3dTriplet, Backup
from elektronn3.training import SWA
import torch.nn.functional as F
import numpy as np
from elektronn3.training.schedulers import CosineAnnealingWarmRestarts

# Dimension of latent space
Z_DIM = 25


class TripletNet(nn.Module):
    """
    adapted from https://github.com/andreasveit/triplet-network-pytorch/blob/master/tripletnet.py
    """
    def __init__(self, rep_net):
        super().__init__()
        self.rep_net = rep_net

    def forward(self, feat, inp):
        if not self.train:
            assert feat.dim() == 3, 'Expecting feature shape (B, N, C) and ' \
                                    'input shape (B, N, 3) during inference.'
            return self.rep_net(feat, inp)
        assert feat.dim() == 4, 'Expecting feature shape (B, 3, N, C) and ' \
                                'input shape (B, 3, N, 3) during inference.'
        z_0 = self.rep_net(feat[0], inp[0])
        z_1 = self.rep_net(feat[1], inp[1])
        z_2 = self.rep_net(feat[2], inp[2])
        if self.train:
            dist_a = F.pairwise_distance(z_0, z_1, 2)
            dist_b = F.pairwise_distance(z_0, z_2, 2)
        else:
            dist_a = None
            dist_b = None
        return dist_a, dist_b, z_0, z_1, z_2


if __name__ == '__main__':
    # PARSE PARAMETERS #

    parser = argparse.ArgumentParser(description='Train a network.')
    parser.add_argument('--na', type=str, help='Experiment name',
                        default=None)
    parser.add_argument('--sr', type=str, help='Save root', default=None)
    parser.add_argument('--bs', type=int, default=16, help='Batch size')
    parser.add_argument('--sp', type=int, default=50000, help='Number of sample points')
    parser.add_argument('--scale_norm', type=int, default=30000, help='Scale factor for normalization')
    parser.add_argument('--cl', type=int, default=5, help='Number of classes')
    parser.add_argument('--co', action='store_true', help='Disable CUDA')
    parser.add_argument('--seed', default=0, help='Random seed')
    parser.add_argument('--ana', default=0, help='Cloudset size of previous analysis')
    parser.add_argument(
        '-j', '--jit', metavar='MODE', default='disabled',  # TODO: does not work
        choices=['disabled', 'train', 'onsave'],
        help="""Options:
    "disabled": Completely disable JIT tracing;
    "onsave": Use regular Python model for training, but trace it on-demand for saving training state;
    "train": Use traced model for training and serialize it on disk"""
    )

    args = parser.parse_args()

    # SET UP ENVIRONMENT #

    random_seed = args.seed
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

    # define parameters
    use_cuda = not args.co
    name = args.na
    batch_size = args.bs
    npoints = args.sp
    scale_norm = args.scale_norm
    size = args.ana
    save_root = args.sr

    lr = 1e-3
    lr_stepsize = 1000
    lr_dec = 0.995
    max_steps = 150000
    margin = 0.2

    # celltype specific
    cval = -1  # unsupervised learning -> use all available cells for training!
    cellshape_only = False
    use_syntype = True

    if name is None:
        name = f'celltype_pts_tnet_scale{scale_norm}_nb{npoints}_' \
               f'cv{cval}_nDim{Z_DIM}'
        if cellshape_only:
            name += '_cellshapeOnly'
        if not use_syntype:
            name += '_noSyntype'
    if use_cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print(f'Running on device: {device}')

    # set paths
    if save_root is None:
        save_root = '~/e3_training_convpoint/'
    save_root = os.path.expanduser(save_root)

    # CREATE NETWORK AND PREPARE DATA SET #
    input_channels = 1

    # # Model selection
    # model = ModelNet40(input_channels, Z_DIM)

    model = ModelNetBig(input_channels, Z_DIM)
    name += '_big'

    # model = ModelNetAttention(input_channels, Z_DIM, npoints=npoints)
    # name += '_attention'

    model = TripletNet(model)

    if use_cuda:
        model.to(device)

    example_input = torch.ones(1, 1, 32, 64, 64)
    enable_save_trace = False if args.jit == 'disabled' else True
    if args.jit == 'onsave':
        # Make sure that tracing works
        tracedmodel = torch.jit.trace(model, example_input.to(device))
    elif args.jit == 'train':
        if getattr(model, 'checkpointing', False):
            raise NotImplementedError(
                'Traced models with checkpointing currently don\'t '
                'work, so either run with --disable-trace or disable '
                'checkpointing.')
        tracedmodel = torch.jit.trace(model, example_input.to(device))
        model = tracedmodel

    # Transformations to be applied to samples before feeding them to the network
    train_transform = clouds.Compose([clouds.RandomVariation((-20, 20)),  # in nm
                                      clouds.Normalization(scale_norm),
                                      clouds.Center(),
                                      clouds.RandomRotate()])
    valid_transform = clouds.Compose([clouds.Normalization(scale_norm),
                                      clouds.Center()])

    train_ds = CellCloudDataTriplet(npoints=npoints, transform=train_transform, cv_val=cval,
                                    cellshape_only=cellshape_only, use_syntype=use_syntype)
    valid_ds = CellCloudDataTriplet(npoints=npoints, transform=valid_transform, train=False,
                                    cv_val=cval, cellshape_only=cellshape_only,
                                    use_syntype=use_syntype)

    # PREPARE AND START TRAINING #

    # set up optimization
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,  # Learning rate is set by the lr_sched below
        momentum=0.9,
        weight_decay=0.5e-4,
    )
    # optimizer = SWA(optimizer)  # Enable support for Stochastic Weight Averaging
    # lr_sched = torch.optim.lr_scheduler.StepLR(optimizer, lr_stepsize, lr_dec)
    # lr_sched = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99992)
    lr_sched = CosineAnnealingWarmRestarts(optimizer, T_0=5000, T_mult=1.5)
    # lr_sched = torch.optim.lr_scheduler.CyclicLR(
    #     optimizer,
    #     base_lr=1e-4,
    #     max_lr=1e-2,
    #     step_size_up=2000,
    #     cycle_momentum=True,
    #     mode='exp_range',
    #     gamma=0.99994,
    # )
    criterion = nn.MarginRankingLoss(margin=margin).to(device)
    if use_cuda:
        criterion.cuda()

    # Create trainer
    trainer = Trainer3dTriplet(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        train_dataset=train_ds,
        batchsize=batch_size,
        num_workers=10,
        save_root=save_root,
        enable_save_trace=enable_save_trace,
        exp_name=name,
        schedulers={"lr": lr_sched},
    )

    # Archiving training script, src folder, env info
    bk = Backup(script_path=__file__,
                save_path=trainer.save_path).archive_backup()

    # Start training
    trainer.run(max_steps)
