# ELEKTRONN3 - Neural Network Toolkit
#
# Copyright (c) 2019 - now
# Max Planck Institute of Neurobiology, Munich, Germany
# Authors: Philipp Schubert
from syconn.cnn.TrainData import CellCloudDataJ0251

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
from elektronn3.training import Trainer3d, Backup, metrics
import distutils.util

# PARSE PARAMETERS #
parser = argparse.ArgumentParser(description='Train a network.')
parser.add_argument('--na', type=str, help='Experiment name',
                    default=None)
parser.add_argument('--sr', type=str, help='Save root', default=None)
parser.add_argument('--bs', type=int, default=10, help='Batch size')
parser.add_argument('--sp', type=int, default=50000, help='Number of sample points')
parser.add_argument('--scale_norm', type=int, default=2000, help='Scale factor for normalization')
parser.add_argument('--co', action='store_true', help='Disable CUDA')
parser.add_argument('--seed', default=0, help='Random seed', type=int)
parser.add_argument('--ctx', default=20000, help='Context size in nm', type=int)
parser.add_argument('--use_bias', default=1, help='Use bias parameter in Convpoint layers.',
                    type=distutils.util.strtobool)
parser.add_argument('--use_syntype', default=1, help='Use synapse type',
                    type=distutils.util.strtobool)
parser.add_argument('--cellshape_only', default=0, help='Use only cell surface points',
                    type=distutils.util.strtobool)
parser.add_argument(
    '-j', '--jit', metavar='MODE', default='disabled',  # TODO: does not work
    choices=['disabled', 'train', 'onsave'],
    help="""Options:
"disabled": Completely disable JIT tracing;
"onsave": Use regular Python model for training, but trace it on-demand for saving training state;
"train": Use traced model for training and serialize it on disk"""
)
parser.add_argument('--cval', default=None, help='Cross-validation split indicator.', type=int)


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
save_root = args.sr
cval = args.cval
ctx = args.ctx
use_bias = args.use_bias
use_syntype = args.use_syntype
cellshape_only = args.cellshape_only
lr = 5e-4
lr_stepsize = 100
lr_dec = 0.99
max_steps = 500000

# celltype specific
eval_nr = random_seed  # number of repetition
dr = 0.3
track_running_stats = False
use_norm = 'gn'
num_classes = 11
onehot = True
act = 'relu'
use_myelin = False
if name is None:
    name = f'celltype_pts_j0251v2_scale{scale_norm}_nb{npoints}_ctx{ctx}_{act}'
    if cellshape_only:
        name += '_cellshapeOnly'
    else:
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
if cellshape_only:
    input_channels = 1
if use_norm is False:
    name += '_noBN'
    if track_running_stats:
        name += '_trackRunStats'
else:
    name += f'_{use_norm}'

if not use_bias:
    name += '_noBias'

if use_cuda:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print(f'Running on device: {device}')

# set paths
if save_root is None:
    save_root = '~/e3_training_convpoint/'
save_root = os.path.expanduser(save_root)

# CREATE NETWORK AND PREPARE DATA SET

# Model selection
model = ModelNet40(input_channels, num_classes, dropout=dr, use_norm=use_norm,
                   track_running_stats=track_running_stats, act=act, use_bias=use_bias)

if cval is not None:
    name += f'_CV{cval}'
else:
    name += f'_AllGT'
name += f'_eval{eval_nr}'
model = nn.DataParallel(model)

if use_cuda:
    model.to(device)

example_input = (torch.ones(batch_size, npoints, input_channels).to(device),
                 torch.ones(batch_size, npoints, 3).to(device))
enable_save_trace = False if args.jit == 'disabled' else True
if args.jit == 'onsave':
    # Make sure that tracing works
    tracedmodel = torch.jit.trace(model, example_input)
elif args.jit == 'train':
    if getattr(model, 'checkpointing', False):
        raise NotImplementedError(
            'Traced models with checkpointing currently don\'t '
            'work, so either run with --disable-trace or disable '
            'checkpointing.')
    tracedmodel = torch.jit.trace(model, example_input)
    model = tracedmodel

# Transformations to be applied to samples before feeding them to the network
train_transform = clouds.Compose([clouds.RandomVariation((-40, 40), distr='normal'),  # in nm
                                  clouds.Center(),
                                  clouds.Normalization(scale_norm),
                                  clouds.RandomRotate(apply_flip=True),
                                  clouds.ElasticTransform(res=(40, 40, 40), sigma=6),
                                  clouds.RandomScale(distr_scale=0.1, distr='uniform')])
valid_transform = clouds.Compose([clouds.Center(), clouds.Normalization(scale_norm)])

train_ds = CellCloudDataJ0251(npoints=npoints, transform=train_transform, cv_val=cval,
                              cellshape_only=cellshape_only, use_syntype=use_syntype,
                              onehot=onehot, batch_size=batch_size, ctx_size=ctx, map_myelin=use_myelin)
# valid_ds = CellCloudDataJ0251(npoints=npoints, transform=valid_transform, train=False,
#                               cv_val=cval, cellshape_only=cellshape_only,
#                               use_syntype=use_syntype, onehot=onehot, batch_size=batch_size,
#                               ctx_size=ctx, map_myelin=use_myelin)
valid_ds = None

# PREPARE AND START TRAINING #

# set up optimization
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# optimizer = torch.optim.SGD(
#     model.parameters(),
#     lr=lr,  # Learning rate is set by the lr_sched below
#     momentum=0.9,
#     weight_decay=0.5e-5,
# )

# optimizer = SWA(optimizer)  # Enable support for Stochastic Weight Averaging
lr_sched = torch.optim.lr_scheduler.StepLR(optimizer, lr_stepsize, lr_dec)
# extra weight for HVC and LMAN
# STN=0, DA=1, MSN=2, LMAN=3, HVC=4, GP=5, TAN=6, INT=7
criterion = torch.nn.CrossEntropyLoss()  # weight=torch.Tensor([1]*num_classes))
if use_cuda:
    criterion.cuda()

valid_metrics = {  # mean metrics
    'val_accuracy_mean': metrics.Accuracy(),
    'val_precision_mean': metrics.Precision(),
    'val_recall_mean': metrics.Recall(),
    'val_DSC_mean': metrics.DSC(),
    'val_IoU_mean': metrics.IoU(),
}

# Create trainer
# it seems pytorch 1.1 does not support batch_size=None to enable batched dataloader, instead
# using batch size 1 with custom collate_fn
trainer = Trainer3d(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    device=device,
    train_dataset=train_ds,
    valid_dataset=valid_ds,
    batchsize=1,
    num_workers=20,
    valid_metrics=valid_metrics,
    save_root=save_root,
    enable_save_trace=enable_save_trace,
    exp_name=name,
    schedulers={"lr": lr_sched},
    num_classes=num_classes,
    # example_input=example_input,
    dataloader_kwargs=dict(collate_fn=lambda x: x[0]),
    nbatch_avg=10,
)

# Archiving training script, src folder, env info
bk = Backup(script_path=__file__,
            save_path=trainer.save_path).archive_backup()

# Start training
trainer.run(max_steps)
