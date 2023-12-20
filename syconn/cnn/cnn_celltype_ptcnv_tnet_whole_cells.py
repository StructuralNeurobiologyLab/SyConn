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
from elektronn3.models.convpoint import ModelNet40
from elektronn3.training import Trainer3dTriplet, Backup
import torch.nn.functional as F
import numpy as np

# Dimension of latent space
Z_DIM = 10


class TripletNet(nn.Module):
    """
    This class is an adaptation from the Triplet Network implementation in PyTorch. 
    It is a type of neural network that takes 3 input vectors and outputs 3 vectors.
    The network is trained to minimize the distance between the first vector and the 
    second one, while maximizing the distance between the first vector and the third one.
    
    Args:
        rep_net: The base network to be used for representation learning.
        
    Adapted from: https://github.com/andreasveit/triplet-network-pytorch/blob/master/tripletnet.py
    """
    def __init__(self, rep_net):
        """
        Initializes the TripletNet class.
        
        Args:
            rep_net: The base network to be used for representation learning.
        """
        super().__init__()
        self.rep_net = rep_net

    def forward(self, x0, x1, x2):
        """
        Defines the computation performed at every call.
        
        Args:
            x0: The anchor input to the network.
            x1: The positive input to the network, which is similar to x0.
            x2: The negative input to the network, which is dissimilar to x0.
        
        Returns:
            dist_a: The distance between the anchor and the positive input.
            dist_b: The distance between the anchor and the negative input.
            z_0: The output of the base network for the anchor input.
            z_1: The output of the base network for the positive input.
            z_2: The output of the base network for the negative input.
        """
        if not self.training:
            assert x1 is None and x2 is None
            return self.rep_net(x0[0], x0[1])
        assert x1 is not None, x2 is not None
        z_0 = self.rep_net(x0[0], x0[1])
        z_1 = self.rep_net(x1[0], x1[1])
        z_2 = self.rep_net(x2[0], x2[1])
        dist_a = F.pairwise_distance(z_0, z_1, 2)
        dist_b = F.pairwise_distance(z_0, z_2, 2)
        return dist_a, dist_b, z_0, z_1, z_2


if __name__ == '__main__':
    # PARSE PARAMETERS

    parser = argparse.ArgumentParser(description='Train a network.')
    parser.add_argument('--na', type=str, help='Experiment name',
                        default=None)
    parser.add_argument('--sr', type=str, help='Save root', default=None)
    parser.add_argument('--bs', type=int, default=16, help='Batch size')
    parser.add_argument('--sp', type=int, default=25000, help='Number of sample points')
    parser.add_argument('--scale_norm', type=int, default=1500, help='Scale factor for normalization')
    parser.add_argument('--co', action='store_true', help='Disable CUDA')
    parser.add_argument('--seed', default=0, help='Random seed')
    parser.add_argument('--ctx', default=15000, help='Context size in nm', type=float)
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
    ctx = args.ctx

    lr = 5e-4
    lr_stepsize = 250
    lr_dec = 0.995
    max_steps = 500000
    margin = 0.2
    dr = 0.3

    # celltype specific
    cval = -1  # unsupervised learning -> use all available cells for training!
    cellshape_only = False
    use_syntype = True
    onehot = True
    track_running_stats = False
    use_norm = 'gn'
    act = 'swish'
    dataset = 'j0251'
    use_myelin = True

    if name is None:
        name = f'celltype_pts_tnet_scale{scale_norm}_nb{npoints}_ctx{ctx}_{act}_nDim{Z_DIM}_drawWholeCell'
        if cellshape_only:
            name += '_cellshapeOnly'
        if not use_syntype:
            name += '_noSyntype'
        if use_myelin:
            name += '_myelin'
    if onehot:
        input_channels = 4
        if use_syntype:
            input_channels += 1
        if use_myelin:
            input_channels += 1
    else:
        input_channels = 1
        name += '_flatinp'

    if use_cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    if use_norm is False:
        name += '_noBN'
        if track_running_stats:
            name += '_trackRunStats'
    else:
        name += f'_{use_norm}'

    if dataset == 'j0126':
        ssd_kwargs = dict(working_dir='/ssdscratch/pschuber/songbird/j0126/areaxfs_v10_v4b_base_20180214_full_'
                                      'agglo_cbsplit')
    elif dataset == 'j0251':
        ssd_kwargs = dict(working_dir='/ssdscratch/pschuber/songbird/j0251/rag_flat_Jan2019_v2/')
    else:
        raise NotImplementedError
    name += f'_{dataset}'

    print(f'Running on device: {device}')

    # set paths
    if save_root is None:
        save_root = '~/e3_training_convpoint/'
    save_root = os.path.expanduser(save_root)

    # CREATE NETWORK AND PREPARE DATA SET #

    # # Model selection
    model = ModelNet40(input_channels, Z_DIM, dropout=dr, use_norm=use_norm,
                       track_running_stats=track_running_stats, act=act)
    name += '_moreAug4'

    model = TripletNet(model)
    model = nn.DataParallel(model)

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
    train_transform = clouds.Compose([clouds.RandomVariation((-40, 40), distr='normal'),  # in nm
                                      clouds.Center(500, distr='uniform'),
                                      clouds.Normalization(scale_norm),
                                      clouds.RandomRotate(apply_flip=True),
                                      clouds.ElasticTransform(res=(40, 40, 40), sigma=6),
                                      clouds.RandomScale(distr_scale=0.1, distr='uniform')])

    train_ds = CellCloudDataTriplet(npoints=npoints, transform=train_transform, cv_val=cval,
                                    cellshape_only=cellshape_only, use_syntype=use_syntype, onehot=onehot,
                                    batch_size=batch_size, ctx_size=ctx, ssd_kwargs=ssd_kwargs,
                                    map_myelin=use_myelin, draw_local=True, draw_local_dist=np.inf)

    # PREPARE AND START TRAINING #

    # set up optimization
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # optimizer = torch.optim.SGD(
    #     model.parameters(),
    #     lr=lr,  # Learning rate is set by the lr_sched below
    #     momentum=0.9,
    #     weight_decay=0.5e-4,
    # )
    # optimizer = SWA(optimizer)  # Enable support for Stochastic Weight Averaging
    lr_sched = torch.optim.lr_scheduler.StepLR(optimizer, lr_stepsize, lr_dec)
    # lr_sched = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99992)
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
        batchsize=1,
        num_workers=10,
        save_root=save_root,
        enable_save_trace=enable_save_trace,
        exp_name=name,
        schedulers={"lr": lr_sched},
        dataloader_kwargs=dict(collate_fn=lambda x: x[0])
    )

    # Archiving training script, src folder, env info
    bk = Backup(script_path=__file__,
                save_path=trainer.save_path).archive_backup()

    # Start training
    trainer.run(max_steps)
