#!/usr/bin/env python3

# ELEKTRONN3 - Neural Network Toolkit
#
# Copyright (c) 2017 - now
# Max Planck Institute of Neurobiology, Munich, Germany
# Authors: Martin Drawitsch, Philipp Schubert

import argparse
import os
import _pickle

import torch
from torch import nn
from torch import optim

# TODO: Make torch and numpy RNG seed configurable

parser = argparse.ArgumentParser(description='Train a network.')
parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
parser.add_argument('-n', '--exp-name', default=None, help='Manually set experiment name')
parser.add_argument(
    '-s', '--epoch-size', type=int, default=200,
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

from elektronn3.data import PatchCreator, transforms, utils, get_preview_batch
from elektronn3.training import Trainer, Backup, metrics, Padam, handlers
from elektronn3.models.unet import UNet
from elektronn3.modules.loss import DiceLoss
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
    #conv_mode='valid',
    #up_mode='resizeconv_nearest',  # Enable to avoid checkerboard artifacts
    adaptive=True  # Experimental. Disable if results look weird.
).to(device)

# Example for a model-compatible input.
example_input = torch.randn(1, 1, 40, 144, 144)

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
data_root = os.path.expanduser('/wholebrain/songbird/j0126/GT/synapsetype_gt/')

gt_dir = data_root + '/Segmentierung_von_Synapsentypen_v1/'
fnames = sorted([gt_dir + f for f in os.listdir(gt_dir) if f.endswith('.h5')])
gt_dir = data_root + '/synssv_reconnects_nosomamerger/'
fnames += sorted([gt_dir + f for f in os.listdir(gt_dir)[:900] if f.endswith('.h5')])

input_h5data = [(f, 'raw') for f in fnames + fnames[-1:]]
target_h5data = [(f, 'label') for f in fnames + fnames[-1:]]
valid_indices = [3]

# Class weights for imbalanced dataset, the last one is used as ignore label
class_weights = torch.tensor([0.3, 0.3, 0.3, 0]).to(device)

max_steps = args.max_steps
max_runtime = args.max_runtime

if args.resume is not None:  # Load pretrained network
    try:  # Assume it's a state_dict for the model
        model.load_state_dict(torch.load(os.path.expanduser(args.resume)))
    except _pickle.UnpicklingError as exc:
        # Assume it's a complete saved ScriptModule
        model = torch.jit.load(os.path.expanduser(args.resume), map_location=device)

# Transformations to be applied to samples before feeding them to the network
common_transforms = [
    transforms.SqueezeTarget(dim=0),  # Workaround for neuro_data_cdhw
    #transforms.Normalize(mean=dataset_mean, std=dataset_std),
    transforms.DropIfTooMuchBG(bg_id=3, threshold=0.9)
]
train_transform = transforms.Compose(common_transforms + [
    transforms.RandomGrayAugment(channels=[0], prob=0.3),
    transforms.RandomGammaCorrection(gamma_std=0.25, gamma_min=0.25, prob=0.3),
    transforms.AdditiveGaussianNoise(sigma=0.05, channels=[0], prob=0.1),
    transforms.RandomBlurring({'probability': 0.1})
])
valid_transform = transforms.Compose(common_transforms + [])

# Specify data set
aniso_factor = 2  # Anisotropy in z dimension. E.g. 2 means half resolution in z dimension.
common_data_kwargs = {  # Common options for training and valid sets.
    'aniso_factor': aniso_factor,
    'patch_shape': (40, 144, 144),
    'num_classes': 4,
}

type_args = list(range(len(input_h5data)))
train_dataset = PatchCreator(
    input_h5data=[input_h5data[i] for i in type_args if i not in valid_indices],
    target_h5data=[target_h5data[i] for i in type_args if i not in valid_indices],
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
# Use first validation cube for previews. Can be set to any other data source.
preview_batch = get_preview_batch(
    # use first cube - might also be training data but always previews the same slice!
    h5data=input_h5data[0],
    preview_shape=(40, 144, 144),
)

optimizer = Padam(
    model.parameters(),
    lr=2e-3,
    weight_decay=0.5e-4,
    partial=1/4,
)

lr_stepsize = 200
lr_dec = 0.995
lr_sched = optim.lr_scheduler.StepLR(optimizer, lr_stepsize, lr_dec)
schedulers = {'lr': lr_sched}
# All these metrics assume a binary classification problem. If you have
#  non-binary targets, remember to adapt the metrics!
valid_metrics = {
}

criterion = DiceLoss(apply_softmax=True, weight=class_weights)

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
