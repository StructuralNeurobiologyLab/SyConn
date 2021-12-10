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
from elektronn3.models.lcp_adapt import ConvAdaptSeg
from elektronn3.modules.loss import LovaszLoss, DiceLossFancy
from lightconvpoint.utils.network import get_search, get_conv
from elektronn3.training import Trainer3d, Backup, metrics

torch.backends.cudnn.enabled = True
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
batch_size = args.bs
npoints = args.sp
scale_norm = args.scale_norm
save_root = args.sr
ctx = args.ctx

lr = 1e-3
lr_stepsize = 1000
lr_dec = 0.99
max_steps = 500000

normalize_pts = True

# celltype specific
eval_nr = random_seed  # number of repetition
cellshape_only = False
use_syntype = False
# ads: axon dendrite soma
# abt: axon bouton terminal
# fine: 'dendrite': 0, 'axon': 1, 'soma': 2, 'bouton': 3, 'terminal': 4, 'neck': 5, 'head': 6
gt_type = 'fine'
num_classes = {'ads': 3, 'abt': 3, 'dnh': 3, 'fine': 7, 'dnho': 4, 'do': 2}
ignore_l = num_classes[gt_type]  # num_classes is also used as ignore label
remap_dicts = {'ads': {3: 1, 4: 1, 5: 0, 6: 0},
               'abt': {0: ignore_l, 2: ignore_l, 5: ignore_l, 6: ignore_l, 1: 0, 3: 1, 4: 2},
               'dnh': {1: ignore_l, 2: ignore_l, 3: ignore_l, 4: ignore_l, 5: 1, 6: 2},
               'dnho': {4: 3, 2: 3, 1: 3, 5: 1, 6: 2},
               'fine': {},
               'do': {4: 1, 3: 1, 2: 1, 5: 0, 6: 0}}
weights = dict(ads=[1, 1, 1], abt=[1, 2, 2], dnh=[1, 2, 2], fine=[1, 1, 1, 2, 8, 4, 8], dnho=[2, 4, 4, 1],
               do=[2, 1])

use_subcell = True
if cellshape_only:
    use_subcell = False
    use_syntype = False

if name is None:
    name = f'semseg_pts_nb{npoints}_ctx{ctx}_{gt_type}_nclass' \
           f'{num_classes[gt_type]}_lcp_GN_noKernelSep_AdamW_CE_large_v2'
    if not normalize_pts:
        name += '_NonormPts'
    if cellshape_only:
        name += '_cellshapeOnly'
    if use_syntype:
        name += '_Syntype'
if not cellshape_only and use_subcell:
    input_channels = 5 if use_syntype else 4
else:
    input_channels = 1

if use_cuda:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print(f'Running on device: {device}')

# set paths
if save_root is None:
    save_root = '/wholebrain/scratch/pschuber/e3_trainings/lcp_semseg_j0251_fine'
save_root = os.path.expanduser(save_root)

# CREATE NETWORK AND PREPARE DATA SET

# Model selection
search = 'SearchQuantized'
conv = dict(layer='ConvPoint', kernel_separation=False, normalize_pts=normalize_pts)
act = nn.ReLU
# architecture = None
architecture = [dict(ic=-1, oc=1, ks=48, nn=32, np=-1),
                dict(ic=1, oc=1, ks=48, nn=32, np=2048),
                dict(ic=1, oc=1, ks=32, nn=16, np=1024),
                dict(ic=1, oc=2, ks=32, nn=16, np=256),
                dict(ic=2, oc=2, ks=32, nn=16, np=128),
                dict(ic=2, oc=2, ks=16, nn=16, np=64),
                dict(ic=2, oc=2, ks=16, nn=16, np=32),
                dict(ic=2, oc=2, ks=16, nn=4, np='d'),
                dict(ic=4, oc=2, ks=16, nn=4, np='d'),
                dict(ic=4, oc=1, ks=32, nn=4, np='d'),
                dict(ic=3, oc=1, ks=32, nn=8, np='d'),
                dict(ic=2, oc=1, ks=32, nn=8, np='d'),
                dict(ic=2, oc=1, ks=48, nn=8, np='d')]
model = ConvAdaptSeg(input_channels, num_classes[gt_type], get_conv(conv), get_search(search), kernel_num=64,
                     architecture=architecture, activation=act, norm='gn')

name += f'_eval{eval_nr}'
# model = nn.DataParallel(model)

model.to(device)
example_input = (torch.ones(batch_size, input_channels, npoints).to(device),
                 torch.ones(batch_size, 3, npoints).to(device))

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
train_transform = clouds.Compose([clouds.RandomVariation((-20, 20), distr='normal'),  # in nm
                                  clouds.Center(),
                                  # clouds.Normalization(scale_norm),
                                  clouds.RandomRotate(apply_flip=True),
                                  clouds.ElasticTransform(res=(40, 40, 40), sigma=6),
                                  clouds.RandomScale(distr_scale=0.05, distr='uniform')])
valid_transform = clouds.Compose([clouds.Center(),
                                  # clouds.Normalization(scale_norm)
                                  ])

# mask boarder points with 'num_classes' and set its weight to 0
if gt_type == 'dnho' or gt_type == 'do':  # no additional validation data
    train_dir = '/wholebrain/songbird/j0126/GT/spgt_semseg/kzips/pkl_files/'
    valid_dir = '/wholebrain/songbird/j0126/GT/spgt_semseg/kzips/pkl_files/'
else:  # no additional validation data
    train_dir = '/wholebrain/songbird/j0251/groundtruth/compartment_gt/2021_11_rev1/train/hc_out_2021_11_fine/'
    valid_dir = '/wholebrain/songbird/j0251/groundtruth/compartment_gt/2021_11_rev1/train/hc_out_2021_11_fine/'

train_ds = CloudDataSemseg(npoints=npoints, transform=train_transform, use_subcell=use_subcell,
                           batch_size=batch_size, ctx_size=ctx, mask_borders_with_id=ignore_l,
                           source_dir=train_dir, remap_dict=remap_dicts[gt_type])
valid_ds = None
# valid_ds = CloudDataSemseg(npoints=npoints, transform=valid_transform, train=False, use_subcell=use_subcell,
#                            batch_size=batch_size, ctx_size=ctx, mask_borders_with_id=ignore_l,
#                            source_dir=valid_dir, remap_dict=remap_dicts[gt_type])
# PREPARE AND START TRAINING #

# set up optimization
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
# optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# optimizer = torch.optim.SGD(
#     model.parameters(),
#     lr=lr,  # Learning rate is set by the lr_sched below
#     momentum=0.9,
#     weight_decay=0.5e-5,
# )

# optimizer = SWA(optimizer)  # Enable support for Stochastic Weight Averaging
lr_sched = torch.optim.lr_scheduler.StepLR(optimizer, lr_stepsize, lr_dec)
# lr_sched = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99992)
# lr_sched = torch.optim.lr_scheduler.CyclicLR(
#     optimizer,
#     base_lr=1e-4,
#     max_lr=5e-3,
#     step_size_up=2000,
#     step_size_down=20000,
#     cycle_momentum=True,
#     mode='exp_range',
#     gamma=0.99994,
# )
# set weight of the masking label at context boarders to 0
class_weights = torch.tensor(weights[gt_type], dtype=torch.float32, device=device)
assert ignore_l == len(weights[gt_type])  # ignore index needs to be the lasst class
# criterion = DiceLossFancy(weights=class_weights, ignore_index=ignore_l).to(device)  # add zero weight for ignore index
criterion = torch.nn.CrossEntropyLoss(weight=class_weights, ignore_index=ignore_l).to(device)

valid_metrics = {  # mean metrics
    'val_accuracy_mean': metrics.Accuracy(),
    'val_precision_mean': metrics.Precision(),
    'val_recall_mean': metrics.Recall(),
    'val_DSC_mean': metrics.DSC(),
    'val_IoU_mean': metrics.IoU(),
}
if num_classes[gt_type] > 2:
    # Add separate per-class accuracy metrics only if there are more than 2 classes
    valid_metrics.update({
        f'val_IoU_c{i}': metrics.Accuracy(i)
        for i in range(num_classes[gt_type])
    })

# Create trainer
trainer = Trainer3d(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    device=device,
    train_dataset=train_ds,
    valid_dataset=valid_ds,
    batchsize=None,
    num_workers=10,
    valid_metrics=valid_metrics,
    save_root=save_root,
    enable_save_trace=enable_save_trace,
    exp_name=name,
    schedulers={"lr": lr_sched},
    num_classes=ignore_l,
    # example_input=example_input,
    dataloader_kwargs=dict(persistent_workers=True),
    nbatch_avg=4,
    tqdm_kwargs=dict(disable=False),
    lcp_flag=True
)

# Archiving training script, src folder, env info
bk = Backup(script_path=__file__,
            save_path=trainer.save_path).archive_backup()

# Start training
trainer.run(max_steps)