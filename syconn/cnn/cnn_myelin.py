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

parser = argparse.ArgumentParser(description='Train a network.')
parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
parser.add_argument('-n', '--exp-name', default='myelin_unet_4b_32f_gtv1', help='Manually set '
                                                                           'experiment name')
parser.add_argument(
    '-s', '--epoch-size', type=int, default=1000,
    help='How many training samples to process between '
         'validation/preview/extended-stat calculation phases.'
)
parser.add_argument(
    '-m', '--max-steps', type=int, default=500000,
    help='Maximum number of training steps to perform.'
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

model = UNet(
    n_blocks=4,
    start_filts=32,
    planar_blocks=(0,),
    activation='relu',
    batch_norm=True,
    # conv_mode='valid',
    # up_mode='resizeconv_nearest',  # Enable to avoid checkerboard artifacts
    adaptive=True  # Experimental. Disable if results look weird.
).to(device)

# USER PATHS
save_root = os.path.expanduser('~/e3training/')
os.makedirs(save_root, exist_ok=True)

data_root = os.path.expanduser('~/')
input_h5data = [
    (os.path.join(data_root, f'myelin_{i}.h5'), 'raw')
    for i in [0, 2]
]
target_h5data = [
    (os.path.join(data_root, f'myelin_{i}.h5'), 'label')
    for i in [0, 2]
]
valid_indices = []

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
])
valid_transform = transforms.Compose(common_transforms + [])

# Specify data set
aniso_factor = 2  # Anisotropy in z dimension. E.g. 2 means half resolution in z dimension.
common_data_kwargs = {  # Common options for training and valid sets.
    'aniso_factor': aniso_factor,
    'patch_shape': (48, 96, 96),
    # 'offset': (8, 20, 20),
    'num_classes': 2,
    # 'in_memory': True  # Uncomment to avoid disk I/O (if you have enough host memory for the data)
}
train_dataset = PatchCreator(
    input_h5data=[input_h5data[i] for i in range(len(input_h5data)) if i not in valid_indices],
    target_h5data=[target_h5data[i] for i in range(len(input_h5data)) if i not in valid_indices],
    train=True,
    epoch_size=args.epoch_size,
    warp_prob=0.2,
    warp_kwargs={
        'sample_aniso': aniso_factor != 1,
        'perspective': True,
        'warp_amount': 0.1,
    },
    transform=train_transform,
    **common_data_kwargs
)
valid_dataset = None if not valid_indices else PatchCreator(
    input_h5data=[input_h5data[i] for i in range(len(input_h5data)) if i in valid_indices],
    target_h5data=[target_h5data[i] for i in range(len(input_h5data)) if i in valid_indices],
    train=False,
    epoch_size=10,  # How many samples to use for each validation run
    warp_prob=0,
    warp_kwargs={'sample_aniso': aniso_factor != 1},
    transform=valid_transform,
    **common_data_kwargs
)

optimizer = optim.SGD(
    model.parameters(),
    lr=0.001,  # Learning rate is set by the lr_sched below
    momentum=0.9,
    weight_decay=0.5e-4,
)
optimizer = SWA(optimizer)  # Enable support for Stochastic Weight Averaging

lr_stepsize = 500
lr_dec = 0.995
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


crossentropy = nn.CrossEntropyLoss()
dice = DiceLoss(apply_softmax=True)
criterion = CombinedLoss([crossentropy, dice], weight=[0.5, 0.5], device=device)

# Create trainer
trainer = Trainer(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    device=device,
    train_dataset=train_dataset,
    valid_dataset=valid_dataset,
    batchsize=1,
    num_workers=1,
    save_root=save_root,
    exp_name=args.exp_name,
    schedulers={'lr': lr_sched},
    valid_metrics=valid_metrics,
    enable_videos=True,
    offset=train_dataset.offset,
    apply_softmax_for_prediction=True,
    num_classes=train_dataset.num_classes,
)

# Archiving training script, src folder, env info
Backup(script_path=__file__,save_path=trainer.save_path).archive_backup()

# Start training
trainer.run(max_steps=max_steps, max_runtime=max_runtime)


# How to re-calculate mean, std and class_weights for other datasets:
#  dataset_mean = utils.calculate_means(train_dataset.inputs)
#  dataset_std = utils.calculate_stds(train_dataset.inputs)
#  class_weights = torch.tensor(utils.calculate_class_weights(train_dataset.targets))
