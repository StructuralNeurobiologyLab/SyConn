#!/usr/bin/env python3

# ELEKTRONN3 - Neural Network Toolkit
#
# Copyright (c) 2017 - now
# Max Planck Institute of Neurobiology, Munich, Germany
# Authors: Martin Drawitsch, Philipp Schubert

import argparse
import os
import zipfile
import re
import torch
from torch import nn
from torch import optim

# TODO: Make torch and numpy RNG seed configurable

parser = argparse.ArgumentParser(description='Train a network.')
parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
parser.add_argument('-n', '--exp-name',
                    default='syntype_unet_sameConv_BN_fancydice_gtv2_bs4_bal_RES',
                    help='Manually set experiment name')
parser.add_argument(
    '-s', '--epoch-size', type=int, default=500,
    help='How many training samples to process between '
         'validation/preview/extended-stat calculation phases.'
)
parser.add_argument(
    '-m', '--max-steps', type=int, default=4000000,
    help='Maximum number of training steps to perform.'
)
parser.add_argument(
    '-t', '--max-runtime', type=int, default=3600 * 24 * 5,  # 4 days
    help='Maximum training time (in seconds).'
)
parser.add_argument(
    '-r', '--resume', metavar='PATH',
    help='Path to pretrained model state dict or a compiled and saved '
         'ScriptModule from which to resume training.'
)
parser.add_argument(
    '-j', '--jit', metavar='MODE', default='onsave',
    choices=['disabled', 'train', 'onsave'],
    help="""Options:
"disabled": Completely disable JIT tracing;
"onsave": Use regular Python model for training, but trace it on-demand for saving training state;
"train": Use traced model for training and serialize it on disk"""
)
args = parser.parse_args()

if not args.disable_cuda and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print(f'Running on device: {device}')

# Don't move this stuff, it needs to be run this early to work
import elektronn3
elektronn3.select_mpl_backend('Agg')
import numpy as np
from elektronn3.data import PatchCreator, transforms, utils, get_preview_batch
from elektronn3.training import Trainer, Backup, metrics, Padam, handlers
from elektronn3.models.unet import UNet
from elektronn3.modules.loss import DiceLoss, DiceLossFancy
from elektronn3.training.metrics import channel_metric

torch.backends.cudnn.benchmark = True  # Improves overall performance in *most* cases

# model = UNet_valid_blocks(
#     out_channels=5,
#     n_blocks=5,
#     start_filts=64,
#     planar_blocks=(1,2),
#     activation='relu',
#     batch_norm=True,
#     valid_blocks = (0,1,2),
#     adaptive=False
# ).to(device)
# offset = model.offset

model = UNet(
    in_channels=1,
    out_channels=4,
    n_blocks=4,
    start_filts=28,
    planar_blocks=(0,),
    activation='relu',
    batch_norm=True,
    # conv_mode='valid',
    #up_mode='resizeconv_nearest',  # Enable to avoid checkerboard artifacts
    adaptive=False  # Experimental. Disable if results look weird.
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
save_root = os.path.expanduser('~/e3_training/')
os.makedirs(save_root, exist_ok=True)
# data_root = os.path.expanduser('/ssdscratch/pschuber/songbird/j0126/GT/synapsetype_gt/')
data_root = os.path.expanduser(
    '/wholebrain/songbird/j0126/GT/synapsetype_gt/')

gt_dir = data_root + '/Segmentierung_von_Synapsentypen_v4/'
fnames = sorted([gt_dir + f for f in os.listdir(gt_dir) if f.endswith('.h5')])

# Get synapse GT based on cell type connectivity
gt_dir = data_root + '/synssv_reconnects_nosomamerger_v2/'
fnames_files = sorted([gt_dir + f for f in os.listdir(gt_dir) if f.endswith('.h5')])
fnames_files = np.array(fnames_files)
fnames_files_ls = np.array([int(re.findall('(\d)_', fname)[0]) for fname in fnames_files])
# do not use STN, DA and TAN as GT samples
fnames_files = np.concatenate([fnames_files[fnames_files_ls == cl][:110] for cl in [2, 3, 4, 5,
                                                                                    7]]).tolist()
# Add all STN samples
fnames_files_stn = sorted([gt_dir + f for f in os.listdir(gt_dir) if (f.endswith('.h5') and '0_cube'
                           in f)])

print(f'{len(fnames_files_stn)} STN, {len(fnames_files)} additional CT and'
      f' {len(fnames)} dense cube samples.')
fnames += fnames_files + fnames_files_stn

input_h5data = [(f, 'raw') for f in fnames + fnames[-1:]]
target_h5data = [(f, 'label') for f in fnames + fnames[-1:]]
valid_indices = [len(target_h5data) - 1]

# Class weights for imbalanced dataset, the last one is used as ignore label
class_weights = torch.tensor([0.33, 0.33, 0.33, 0]).to(device)
# class_weights = torch.tensor([1, 2, 2, 0]).to(device)

max_steps = args.max_steps
max_runtime = args.max_runtime

if args.resume is not None:  # Load pretrained network
    pretrained = os.path.expanduser(args.resume)
    _warning_str = 'Loading model without optimizer state. Prefer state dicts'
    if zipfile.is_zipfile(pretrained):  # Zip file indicates saved ScriptModule
        print(_warning_str)
        model = torch.jit.load(pretrained, map_location=device)
    else:  # Either state dict or pickled model
        state = torch.load(pretrained)
        if isinstance(state, dict):
            try:
                model.load_state_dict(state['model_state_dict'])
            except RuntimeError:
                print('Converting state dict (probably stored as DataParallel.')
                for k in list(state.keys()):
                    state['module' + k] = state[k]
                    del state[k]
            optimizer_state_dict = state.get('optimizer_state_dict')
            lr_sched_state_dict = state.get('lr_sched_state_dict')
            if optimizer_state_dict is None:
                print('optimizer_state_dict not found.')
            if lr_sched_state_dict is None:
                print('lr_sched_state_dict not found.')
        elif isinstance(state, nn.Module):
            print(_warning_str)
            model = state
        else:
            raise ValueError(f'Can\'t load {pretrained}.')

drop_func = transforms.DropIfTooMuchBG(bg_id=3, threshold=0.9)
# Transformations to be applied to samples before feeding them to the network
common_transforms = [
    transforms.SqueezeTarget(dim=0),  # Workaround for neuro_data_cdhw
    #transforms.Normalize(mean=dataset_mean, std=dataset_std),
]
train_transform = transforms.Compose(common_transforms + [
    transforms.RandomGrayAugment(channels=[0], prob=0.2),
    transforms.RandomGammaCorrection(gamma_std=0.25, gamma_min=0.25, prob=0.2),
    transforms.AdditiveGaussianNoise(sigma=0.05, channels=[0], prob=0.1),
    transforms.RandomBlurring({'probability': 0.1}),
    transforms.RandomFlip(ndim_spatial=3),
    drop_func
])
valid_transform = transforms.Compose(common_transforms + [])

# Specify data set
aniso_factor = 2  # Anisotropy in z dimension. E.g. 2 means half resolution in z dimension.
common_data_kwargs = {  # Common options for training and valid sets.
    'aniso_factor': aniso_factor,
    'patch_shape': (48, 144, 144),
    'num_classes': 4,
    # 'offset': (20, 46, 46),
}

if len(fnames) > 100:  # artificial GT based on cell types is in use
    cube_prios = None
else:
    cube_prios = [1] * (len(input_h5data) - len(valid_indices))

type_args = list(range(len(input_h5data)))
train_dataset = PatchCreator(
    input_h5data=[input_h5data[i] for i in type_args if i not in valid_indices],
    target_h5data=[target_h5data[i] for i in type_args if i not in valid_indices],
    train=True,
    epoch_size=args.epoch_size,
    cube_prios=cube_prios,
    warp_prob=0.2,
    in_memory=True,
    warp_kwargs={
        'sample_aniso': aniso_factor != 1,
        'perspective': True,
        'warp_amount': 0.05,
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
    in_memory=True,
    **common_data_kwargs
)
# Use first validation cube for previews. Can be set to any other data source.
preview_batch = get_preview_batch(
    # use first cube - might also be training data but always previews the same slice!
    h5data=input_h5data[3],
    preview_shape=(48, 144, 144),
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=2e-3,
    weight_decay=0.5e-4,
)

lr_stepsize = 200
lr_dec = 0.997
lr_sched = optim.lr_scheduler.StepLR(optimizer, lr_stepsize, lr_dec)
schedulers = {'lr': lr_sched}
# All these metrics assume a binary classification problem. If you have
#  non-binary targets, remember to adapt the metrics!
valid_metrics = {
}

criterion = DiceLossFancy(apply_softmax=True, weights=class_weights,
                          ignore_index=3)

# Create trainer
trainer = Trainer(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    device=device,
    train_dataset=train_dataset,
    valid_dataset=valid_dataset,
    batchsize=4,
    num_workers=2,
    save_root=save_root,
    exp_name=args.exp_name,
    example_input=example_input,
    enable_save_trace=enable_save_trace,
    schedulers=schedulers,#{"lr": optim.lr_scheduler.StepLR(optimizer, 1000, 0.995)},
    valid_metrics=valid_metrics,
    #preview_batch=preview_batch,
    #preview_interval=5,
    enable_videos=False,  # Uncomment to get rid of videos in tensorboard
    offset=train_dataset.offset,
    apply_softmax_for_prediction=True,
    num_classes=train_dataset.num_classes,
    ipython_shell=False,
    # TODO: Tune these:
    #preview_tile_shape=(48, 96, 96),
    #preview_overlap_shape=(48, 96, 96),
    #sample_plotting_handler = handlers._tb_log_sample_images_Synapse,
    #mixed_precision=True,  # Enable to use Apex for mixed precision training
)

# Archiving training script, src folder, env info
Backup(script_path=__file__,save_path=trainer.save_path).archive_backup()

# Start training
trainer.run(max_steps=max_steps, max_runtime=max_runtime)


# How to re-calculate mean, std and class_weights for other datasets:
#  dataset_mean = utils.calculate_means(train_dataset.inputs)
#  dataset_std = utils.calculate_stds(train_dataset.inputs)
#  class_weights = torch.tensor(utils.calculate_class_weights(train_dataset.targets))
