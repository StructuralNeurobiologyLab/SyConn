#!/usr/bin/env python3

# ELEKTRONN3 - Neural Network Toolkit
#
# Copyright (c) 2017 - now
# Max Planck Institute of Neurobiology, Munich, Germany
# Authors: Martin Drawitsch, Philipp Schubert

import argparse
import logging
import os
import random
import zipfile

import torch
from torch import nn
from torch import optim
import numpy as np
"""
Used the following to create the GT:
    create_h5_from_kzip('/myeling_gt/myelin_3_0_8.010.k.zip', kd_p, mag=4, apply_mops_seg=['binary_opening', 'binary_closing'])
    create_h5_from_kzip('/myeling_gt/myelin_2_4_1.007.k.zip', kd_p, mag=4, apply_mops_seg=['binary_opening', 'binary_closing'])
    create_h5_from_kzip('/myeling_gt/myelin_0_0_1.007.k.zip', kd_p, mag=4, apply_mops_seg=['binary_opening', 'binary_closing'])
"""
parser = argparse.ArgumentParser(description='Train a network.')
parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
parser.add_argument('-n', '--exp-name', default='myelin_unet_gtv3_BN_smaller',
                    help='Manually set experiment name')
parser.add_argument(
    '-s', '--epoch-size', type=int, default=1000,
    help='How many training samples to process between '
         'validation/preview/extended-stat calculation phases.'
)
parser.add_argument(
    '-m', '--max-steps', type=int, default=1500000,
    help='Maximum number of training steps to perform.'
)
parser.add_argument(
    '-j', '--jit', metavar='MODE', default='onsave',
    choices=['disabled', 'train', 'onsave'],
    help="""Options:
"disabled": Completely disable JIT tracing;
"onsave": Use regular Python model for training, but trace it on-demand for saving training state;
"train": Use traced model for training and serialize it on disk"""
)
parser.add_argument(
    '-t', '--max-runtime', type=int, default=3600 * 24 * 4,  # 4 days
    help='Maximum training time (in seconds).'
)
parser.add_argument(
    '-r', '--resume', metavar='PATH',
    help='Path to pretrained model state dict or a compiled and saved '
         'ScriptModule from which to resume training.'
)
parser.add_argument('--seed', type=int, default=0, help='Base seed for all RNGs.')
parser.add_argument(
    '--deterministic', action='store_true',
    help='Run in fully deterministic mode (at the cost of execution speed).'
)
parser.add_argument('--sr', type=str, default=os.path.expanduser('~/e3training/'), help='Save root.')
args = parser.parse_args()

# Set up all RNG seeds, set level of determinism
random_seed = args.seed
torch.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)
deterministic = args.deterministic
if deterministic:
    torch.backends.cudnn.deterministic = True
else:
    torch.backends.cudnn.benchmark = True  # Improves overall performance in *most* cases

# Don't move this stuff, it needs to be run this early to work
import elektronn3
elektronn3.select_mpl_backend('Agg')
logger = logging.getLogger('elektronn3log')

from elektronn3.data import PatchCreator, transforms, utils, get_preview_batch
from elektronn3.training import Trainer, Backup, metrics
from elektronn3.training import SWA
from elektronn3.modules import DiceLoss, CombinedLoss
from elektronn3.models.unet import UNet


if not args.disable_cuda and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
logger.info(f'Running on device: {device}')

out_channels = 2
model = UNet(
    out_channels=out_channels,
    n_blocks=4,
    start_filts=32,
    planar_blocks=(0, 2),
    activation='relu',
    normalization='batch',
).to(device)

# Example for a model-compatible input.
example_input = torch.randn(1, 1, 48, 144, 144)
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

# USER PATHS
save_root = args.sr
os.makedirs(save_root, exist_ok=True)

data_root = os.path.expanduser('/wholebrain/scratch/pschuber/')
input_h5data = [
    (os.path.join(data_root, f'myelin_{i}.h5'), 'raw')
    for i in [0, 2, 3, 0, 2, 3]
]
target_h5data = [
    (os.path.join(data_root, f'myelin_{i}.h5'), 'label')
    for i in [0, 2, 3, 0, 2, 3]
]

# use training data for validation too
valid_indices = [3, 4, 5]

max_steps = args.max_steps
max_runtime = args.max_runtime

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

# Transformations to be applied to samples before feeding them to the network
common_transforms = [
    transforms.SqueezeTarget(dim=0),
]
train_transform = transforms.Compose(common_transforms + [
    transforms.RandomFlip(ndim_spatial=3),
    transforms.RandomGrayAugment(channels=[0], prob=0.3),
    transforms.RandomGammaCorrection(gamma_std=0.25, gamma_min=0.25, prob=0.3),
    transforms.AdditiveGaussianNoise(sigma=0.1, channels=[0], prob=0.3),
    transforms.RandomBlurring({'probability': 0.1})
])
valid_transform = transforms.Compose(common_transforms + [])

# Specify data set
aniso_factor = 2  # Anisotropy in z dimension. E.g. 2 means half resolution in z dimension.
common_data_kwargs = {  # Common options for training and valid sets.
    'aniso_factor': aniso_factor,
    'patch_shape': (32, 144, 144),
    'in_memory': True  # Uncomment to avoid disk I/O (if you have enough host memory for the data)
}
train_dataset = PatchCreator(
    input_sources=[input_h5data[i] for i in range(len(input_h5data)) if i not in valid_indices],
    target_sources=[target_h5data[i] for i in range(len(input_h5data)) if i not in valid_indices],
    train=True,
    epoch_size=args.epoch_size,
    warp_prob=0.2,
    warp_kwargs={
        'sample_aniso': aniso_factor != 1,
        'perspective': True,
        'warp_amount': 0.5,
    },
    transform=train_transform,
    **common_data_kwargs
)
valid_dataset = None if not valid_indices else PatchCreator(
    input_sources=[input_h5data[i] for i in range(len(input_h5data)) if i in valid_indices],
    target_sources=[target_h5data[i] for i in range(len(input_h5data)) if i in valid_indices],
    train=False,
    epoch_size=10,  # How many samples to use for each validation run
    warp_prob=0,
    warp_kwargs={'sample_aniso': aniso_factor != 1},
    transform=valid_transform,
    **common_data_kwargs
)

# optimizer = optim.SGD(
#     model.parameters(),
#     lr=0.001,
#     momentum=0.9,
#     weight_decay=0.5e-4,
# )
# optimizer = SWA(optimizer)  # Enable support for Stochastic Weight Averaging

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=2e-3,
    weight_decay=0.5e-4,
)

lr_stepsize = 200
lr_dec = 0.997
lr_sched = optim.lr_scheduler.StepLR(optimizer, lr_stepsize, lr_dec)
if optimizer_state_dict is not None:
    optimizer.load_state_dict(optimizer_state_dict)
if lr_sched_state_dict is not None:
    lr_sched.load_state_dict(lr_sched_state_dict)

# All these metrics assume a binary classification problem. If you have
#  non-binary targets, remember to adapt the metrics!
valid_metrics = {
    'val_accuracy': metrics.bin_accuracy,
    'val_precision': metrics.bin_precision,
    'val_recall': metrics.bin_recall,
    'val_DSC': metrics.bin_dice_coefficient,
    'val_IoU': metrics.bin_iou,
}

weight = torch.tensor((1, 2))
# crossentropy = nn.CrossEntropyLoss(weight)
criterion = DiceLoss(weight=weight)
# criterion = CombinedLoss([crossentropy, dice], weight=[0.5, 0.5], device=device)

# Create trainer
trainer = Trainer(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    device=device,
    train_dataset=train_dataset,
    valid_dataset=valid_dataset,
    batch_size=2,
    num_workers=4,
    save_root=save_root,
    exp_name=args.exp_name,
    example_input=example_input,
    schedulers={'lr': lr_sched},
    valid_metrics=valid_metrics,
    enable_save_trace=enable_save_trace,
    enable_videos=False,
    out_channels=out_channels,
)

# Archiving training script, src folder, env info
Backup(script_path=__file__,save_path=trainer.save_path).archive_backup()

# Start training
trainer.run(max_steps=max_steps, max_runtime=max_runtime)

# How to re-calculate mean, std and class_weights for other datasets:
#  dataset_mean = utils.calculate_means(train_dataset.inputs)
#  dataset_std = utils.calculate_stds(train_dataset.inputs)
#  class_weights = torch.tensor(utils.calculate_class_weights(train_dataset.targets))
