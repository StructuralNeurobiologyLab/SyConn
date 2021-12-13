# ELEKTRONN3 - Neural Network Toolkit
#
# Copyright (c) 2019 - now
# Max Planck Institute of Neurobiology, Munich, Germany
# Authors: Philipp Schubert, Andrei Mancu
from merge_dataloader import CloudFalseMergeLoader

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
from elektronn3.models.convpoint import SegSmall
from elektronn3.models.lcp_adapt import ConvAdaptSeg
from lightconvpoint.utils.network import get_search, get_conv
from elektronn3.modules.loss import FocalLoss
from elektronn3.training import Trainer3d, Backup, metrics

# PARSE PARAMETERS #
parser = argparse.ArgumentParser(description='Train a network.')
parser.add_argument('--na', type=str, help='Experiment name',
                    default=None)
parser.add_argument('--sr', type=str, help='Save root', default=None)
parser.add_argument('--model', type=str, default='lcp', help='Model to use: segsmall/lcp/randla')
parser.add_argument('--r', type=int, default=1000, help='Radius of merger positive labeling  neighborhood')
parser.add_argument('--opt', type=str, default='Adam', help='Chosen optimizer: Adam/SGD')
parser.add_argument('--lr', type=str, default='StepLR', help='Chosen learning rate: StepLR/ExponentialLR/CyclicLR/ConstantLR')
parser.add_argument('--conv', type=str, default='ConvPoint', help='Convolution type for lcp')
parser.add_argument('--arch', type=str, default=None, help='Architecture type of model')
parser.add_argument('--resume', type=str, default=None,  help='Path to pretrained model state dict from which to resume training.')
parser.add_argument('--bs', type=int, default=4, help='Batch size')
parser.add_argument('--sp', type=int, default=10000, help='Number of sample points')
parser.add_argument('--scale_norm', type=int, default=5000, help='Scale factor for normalization')
parser.add_argument('--co', action='store_true', help='Disable CUDA')
parser.add_argument('--seed', default=0, help='Random seed', type=int)
parser.add_argument('--use_bias', default=True, help='Use bias parameter in Convpoint layers.', type=bool)
parser.add_argument('--ctx', default=20000, help='Context size in nm', type=float)
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
modelselect = args.model
radius = args.r
opt = args.opt
learning_rate = args.lr
conv = args.conv
arch = args.arch
resume = args.resume
batch_size = args.bs
npoints = args.sp
scale_norm = args.scale_norm
save_root = args.sr
ctx = args.ctx
use_bias = args.use_bias

lr = 2e-3
lr_stepsize = 100
lr_dec = 0.995
max_steps = 290000

# celltype specific
eval_nr = random_seed  # number of repetition
use_syntype = False
dr = 0.2
track_running_stats = False
use_norm = 'gn'

# 'no_merge': 0, 'merge_error': 1
num_classes = 2

act = 'relu'

if name is None:
    name = f'{modelselect}_r{radius}'

if not cellshape_only and use_subcell:
    input_channels = 5 if use_syntype else 4
else:
    input_channels = 1

if use_cuda:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print(f'Running on device: {device}')

# set save path
if save_root is None:
    save_root = f'/wholebrain/scratch/amancu/mergeError/trainings/{modelselect}/see/'

# Architecture select for LightConvPoint
if arch == 'archLrg':
    architecture = [{'ic': -1, 'oc': 1, 'ks': 16, 'nn': 32, 'np': -1},
                      {'ic': 1, 'oc': 1, 'ks': 16, 'nn': 32, 'np': 2048},
                      {'ic': 1, 'oc': 1, 'ks': 16, 'nn': 32, 'np': 1024},
                      {'ic': 1, 'oc': 1, 'ks': 16, 'nn': 32, 'np': 256},
                      {'ic': 1, 'oc': 2, 'ks': 16, 'nn': 32, 'np': 64},
                      {'ic': 2, 'oc': 2, 'ks': 16, 'nn': 16, 'np': 16},
                      {'ic': 2, 'oc': 2, 'ks': 16, 'nn': 8, 'np': 8},
                      {'ic': 2, 'oc': 2, 'ks': 16, 'nn': 4, 'np': 'd'},
                      {'ic': 4, 'oc': 2, 'ks': 16, 'nn': 4, 'np': 'd'},
                      {'ic': 4, 'oc': 1, 'ks': 16, 'nn': 8, 'np': 'd'},
                      {'ic': 2, 'oc': 1, 'ks': 16, 'nn': 16, 'np': 'd'},
                      {'ic': 2, 'oc': 1, 'ks': 16, 'nn': 16, 'np': 'd'},
                      {'ic': 2, 'oc': 1, 'ks': 16, 'nn': 16, 'np': 'd'}]
else:
    architecture = None

# Model selection
model = None
if modelselect == 'lcp':
    search = 'SearchQuantized'
    convol = dict(layer=conv, kernel_separation=False)
    layer = convol['layer']
    name += f'_{layer}_{search}_{arch}'
    act = nn.ReLU
    model = ConvAdaptSeg(input_channels, num_classes, get_conv(convol), get_search(search), kernel_num=64,
                         architecture=architecture, activation=act, norm='gn')
if modelselect == 'randla':
    from elektronn3.models.randla_net import RandLANet
    model = RandLANet(input_channels, num_classes, dropout_p=dr)

print(f'Using model {modelselect}')

model.to(device)

if args.resume is not None:  # Load pretrained network params
    model.load_state_dict(torch.load(os.path.expanduser(resume), map_location=device)['model_state_dict'])
    name += '_run2'


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
# valid_transform = clouds.Compose([])

# mask boarder points with 'num_classes' and set its weight to 0
train_ds = CloudFalseMergeLoader(radius=radius, npoints=npoints, transform=train_transform,
                                 batch_size=batch_size, ctx_size=ctx)
valid_ds = CloudFalseMergeLoader(radius=radius, npoints=npoints, transform=valid_transform, train=False,
                                 batch_size=batch_size, ctx_size=ctx)

# PREPARE AND START TRAINING #

# set up optimizer
if opt == 'Adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    name += '_Adam'
elif opt == 'SGD':
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,  # Learning rate is set by the lr_sched below
        momentum=0.9,
        weight_decay=0.5e-5,
    )
    name += '_SGD'

# set up learning rate scheduler
if learning_rate == 'StepLR':
    lr_sched = torch.optim.lr_scheduler.StepLR(optimizer, lr_stepsize, lr_dec)
    name += '_StepLR'
if learning_rate == 'ConstantLR':
    lr_sched = torch.optim.lr_scheduler.StepLR(optimizer, lr_stepsize, 1)
    name += '_ConstantLR'
elif learning_rate == 'ExponentialLR':
    lr_sched = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99998)
    name += '_ExponentialLR'
elif learning_rate == 'CyclicLR':
    lr_sched = torch.optim.lr_scheduler.CyclicLR(
        optimizer,
        base_lr=1e-4,
        max_lr=1e-2,
        step_size_up=2000,
        cycle_momentum=True,
        mode='exp_range',
        gamma=0.99997,
    )
    name += '_CyclicLR'

# adapt class weights for the merge error task
weights = [1,2]

# uncomment for desired  loss!

# name += f'_weights{weights[0]},{weights[1]}_FocalLoss'
name += f'_weights{weights[0]},{weights[1]}_CrossEntropy'

class_weights = torch.tensor(weights, dtype=torch.float32, device=device)

# criterion = FocalLoss(weight=class_weights, ignore_index=num_classes).to(device)
criterion = torch.nn.CrossEntropyLoss(weight=class_weights, ignore_index=num_classes).to(device)
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

# for LCP models
if modelselect == 'lcp':
    trainer = Trainer3d(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        train_dataset=train_ds,
        valid_dataset=valid_ds,
        batchsize=1,
        num_workers=4,
        valid_metrics=valid_metrics,
        save_root=save_root,
        enable_save_trace=enable_save_trace,
        exp_name=name,
        schedulers={"lr": lr_sched},
        num_classes=num_classes,
        # example_input=example_input,
        dataloader_kwargs=dict(collate_fn=lambda x: x[0]),
        nbatch_avg=1,
        tqdm_kwargs={'disable': False},
        lcp_flag=True
    )
# for randla
else:
    trainer = Trainer3d(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        train_dataset=train_ds,
        valid_dataset=valid_ds,
        batchsize=1,
        num_workers=4,
        valid_metrics=valid_metrics,
        save_root=save_root,
        enable_save_trace=enable_save_trace,
        exp_name=name,
        schedulers={"lr": lr_sched},
        num_classes=num_classes,
        dataloader_kwargs=dict(collate_fn=lambda x: x[0]),
        nbatch_avg=5,
        tqdm_kwargs={'disable': False},
    )

# Archiving training script, src folder, env info
bk = Backup(script_path=__file__,
            save_path=trainer.save_path).archive_backup()

# Start training
trainer.run(max_steps)
