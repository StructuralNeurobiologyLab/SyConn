# ELEKTRONN3 - Neural Network Toolkit
#
# Copyright (c) 2019 - now
# Max Planck Institute of Neurobiology, Munich, Germany
# Authors: Philipp Schubert
from syconn.cnn.TrainData import CloudDataSemseg

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
from elektronn3.models.pointnetpp import PointNet2
from elektronn3.training import Trainer3d, Backup, metrics
try:
    from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
except ModuleNotFoundError as e:
    print(e)
    from elektronn3.training.schedulers import CosineAnnealingWarmRestarts

# PARSE PARAMETERS #
parser = argparse.ArgumentParser(description='Train a network.')
parser.add_argument('--na', type=str, help='Experiment name',
                    default=None)
parser.add_argument('--sr', type=str, help='Save root', default=None)
parser.add_argument('--bs', type=int, default=4, help='Batch size')
parser.add_argument('--sp', type=int, default=15000, help='Number of sample points')
parser.add_argument('--scale_norm', type=int, default=5000, help='Scale factor for normalization')
parser.add_argument('--co', action='store_true', help='Disable CUDA')
parser.add_argument('--seed', default=0, help='Random seed', type=int)
parser.add_argument('--use_bias', default=True, help='Use bias parameter in Convpoint layers.', type=bool)
parser.add_argument('--ctx', default=10000, help='Context size in nm', type=float)
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
save_root = args.sr
ctx = args.ctx
use_bias = args.use_bias

lr = 2e-3
lr_stepsize = 100
lr_dec = 0.992
max_steps = 300000

# celltype specific
eval_nr = random_seed  # number of repetition
cellshape_only = True
use_syntype = False
dr = 0.2
track_running_stats = False
use_norm = 'gn'
# 'dendrite': 0, 'axon': 1, 'soma': 2, 'bouton': 3, 'terminal': 4, 'neck': 5, 'head': 6
num_classes = 7
use_subcell = True
if cellshape_only:
    use_subcell = False
    use_syntype = False
act = 'relu'

if name is None:
    name = f'semseg_pts_scale{scale_norm}_nb{npoints}_ctx{ctx}_{act}_nclass' \
           f'{num_classes}_PointNet2_boarderMask'
    if cellshape_only:
        name += '_cellshapeOnly'
    if use_syntype:
        name += '_Syntype'
if not cellshape_only and use_subcell:
    input_channels = 5 if use_syntype else 4
else:
    input_channels = 1
if use_norm is False:
    name += '_noBN'
else:
    name += f'_{use_norm}'
if track_running_stats:
    name += '_trackRunStats'

if use_cuda:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

if not use_bias:
    name += '_noBias'

print(f'Running on device: {device}')

# set paths
if save_root is None:
    save_root = '~/e3_training_convpoint/'
save_root = os.path.expanduser(save_root)

# CREATE NETWORK AND PREPARE DATA SET

# Model selection
model = PointNet2(num_classes + 1, input_channels)

name += f'_eval{eval_nr}'
# model = nn.DataParallel(model)

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
train_transform = clouds.Compose([clouds.RandomVariation((-30, 30), distr='normal'),  # in nm
                                  clouds.Center(),
                                  clouds.Normalization(scale_norm),
                                  clouds.RandomRotate(apply_flip=True),
                                  clouds.ElasticTransform(res=(40, 40, 40), sigma=6),
                                  clouds.RandomScale(distr_scale=0.1, distr='uniform')])
valid_transform = clouds.Compose([clouds.Center(), clouds.Normalization(scale_norm)])

# mask boarder points with 'num_classes' and set its weight to 0
train_ds = CloudDataSemseg(npoints=npoints, transform=train_transform, use_subcell=use_subcell,
                           batch_size=batch_size, ctx_size=ctx, mask_boarders_with_id=num_classes)
valid_ds = CloudDataSemseg(npoints=npoints, transform=valid_transform, train=False, use_subcell=use_subcell,
                           batch_size=batch_size, ctx_size=ctx)

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
# lr_sched = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99992)
# lr_sched = CosineAnnealingWarmRestarts(optimizer, T_0=4000, T_mult=2)
# lr_sched = torch.optim.lr_scheduler.CyclicLR(
#     optimizer,
#     base_lr=1e-4,
#     max_lr=1e-2,
#     step_size_up=2000,
#     cycle_momentum=True,
#     mode='exp_range',
#     gamma=0.99994,
# )
# set weight of the masking label at context boarders to 0
class_weights = torch.tensor([1] * num_classes + [0], dtype=torch.float32, device=device)
criterion = torch.nn.CrossEntropyLoss(weight=class_weights).to(device)

valid_metrics = {  # mean metrics
    'val_accuracy_mean': metrics.Accuracy(),
    'val_precision_mean': metrics.Precision(),
    'val_recall_mean': metrics.Recall(),
    'val_DSC_mean': metrics.DSC(),
    'val_IoU_mean': metrics.IoU(),
}
if num_classes > 2:
    # Add separate per-class accuracy metrics only if there are more than 2 classes
    valid_metrics.update({
        f'val_IoU_c{i}': metrics.Accuracy(i)
        for i in range(num_classes)
    })

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
    num_workers=5,
    valid_metrics=valid_metrics,
    save_root=save_root,
    enable_save_trace=enable_save_trace,
    exp_name=name,
    schedulers={"lr": lr_sched},
    num_classes=num_classes + 1,
    # example_input=example_input,
    dataloader_kwargs=dict(collate_fn=lambda x: x[0]),
    nbatch_avg=5
)

# Archiving training script, src folder, env info
bk = Backup(script_path=__file__,
            save_path=trainer.save_path).archive_backup()

# Start training
trainer.run(max_steps)
