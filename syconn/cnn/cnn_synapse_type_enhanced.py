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


class HybridDiceLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dice_syntype = DiceLossFancy(
            apply_softmax=True, weights=torch.tensor([0.33, 0.33, 0.33, 0]).to(device),
            ignore_index=3)
        weights = torch.tensor([1.] * 10 + [0., ]).to(device)
        self.dice_celltype = DiceLossFancy(
            apply_softmax=True, weights=weights, ignore_index=10)
        self.dice_celltype2 = DiceLossFancy(
            apply_softmax=True, weights=weights, ignore_index=10)
        self.mse = torch.nn.MSELoss()  # use `reduce` kwarg?

    def forward(self, output, target, downscale_fact=0.1):
        # target shape: B, C, Z, Y, X
        # output shape: B, OUT_C, Z, Y, X
        # C: vector field 0-2, syntype label 3, celltype labels 4-5
        # OUT_C: vec. field 0-2, syntype 3-6, celltype 7-17 and 18-28
        vec_field_d = output[:, :3]
        vec_field_t = target[:, :3]
        # scale target vectors
        loss_vec = self.mse(vec_field_d, vec_field_t.to(torch.float32) * downscale_fact)
        syntype_d = output[:, 3:7]
        syntype_l = target[:, 3]
        loss_syntype = self.dice_syntype(syntype_d, syntype_l)
        celltype_d = output[:, 7:18]
        celltype_l = target[:, 4]
        loss_celltype = self.dice_celltype(celltype_d, celltype_l)
        celltype_d2 = output[:, 18:]
        celltype_l2 = target[:, 5]
        loss_celltype2 = self.dice_celltype2(celltype_d2, celltype_l2)
        if torch.isnan(loss_vec):
            raise ValueError('Vectorial loss is NaN.')
        if torch.isnan(loss_syntype):
            raise ValueError('Synapsetype loss is NaN.')
        if torch.isnan(loss_celltype):
            raise ValueError('Celltype loss is NaN.')
        if torch.isnan(loss_celltype2):
            raise ValueError('Celltype 2 loss is NaN.')
        return loss_vec + loss_syntype + loss_celltype + loss_celltype2


# TODO: Make torch and numpy RNG seed configurable
parser = argparse.ArgumentParser(description='Train a network.')
parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
parser.add_argument('-n', '--exp-name',
                    default='syntype_unet_sameConv_BN_fancydice_gt3_enhanced_bs4_inmem_10xmsefact',
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

torch.backends.cudnn.benchmark = True  # Improves overall performance in *most* cases

model = UNet(
    in_channels=1,
    out_channels=29,  # vec. field 0-2, syntype 3-6, celltype 7-17 and 18-28
    n_blocks=4,
    start_filts=28,
    planar_blocks=(0,),
    activation='relu',
    batch_norm=True,
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
data_root = os.path.expanduser('/ssdscratch/pschuber/songbird/j0126/GT/synapsetype_gt/')

gt_dir = data_root + '/synssv_reconnects_nosomamerger_enhanced/'
fnames_files = sorted([gt_dir + f for f in os.listdir(gt_dir) if f.endswith('.h5')])
random_ixs = np.arange(len(fnames_files))
np.random.seed(0)
np.random.shuffle(fnames_files)
fnames_files = np.array(fnames_files)[random_ixs].tolist()
fnames = fnames_files[:950]

input_h5data = [(f, 'raw') for f in fnames + fnames[-5:]]
target_h5data = [(f, 'label') for f in fnames + fnames[-5:]]
valid_indices = np.arange(len(target_h5data) - 5, len(target_h5data))

max_steps = args.max_steps
max_runtime = args.max_runtime

if args.resume is not None:  # Load pretrained network
    try:  # Assume it's a state_dict for the model
        model.load_state_dict(torch.load(os.path.expanduser(args.resume)))
    except _pickle.UnpicklingError as exc:
        # Assume it's a complete saved ScriptModule
        model = torch.jit.load(os.path.expanduser(args.resume), map_location=device)

drop_func = transforms.DropIfTooMuchBG(bg_id=3, threshold=0.9)
# Transformations to be applied to samples before feeding them to the network
common_transforms = [
    #transforms.Normalize(mean=dataset_mean, std=dataset_std),
]
train_transform = transforms.Compose(common_transforms + [
    transforms.RandomGrayAugment(channels=[0], prob=0.3),
    transforms.RandomGammaCorrection(gamma_std=0.25, gamma_min=0.25, prob=0.3),
    transforms.AdditiveGaussianNoise(sigma=0.05, channels=[0], prob=0.1),
    transforms.RandomBlurring({'probability': 0.1}),
    drop_func
])
valid_transform = transforms.Compose(common_transforms + [])

# Specify data set
aniso_factor = 2  # Anisotropy in z dimension. E.g. 2 means half resolution in z dimension.
common_data_kwargs = {  # Common options for training and valid sets.
    'aniso_factor': aniso_factor,
    'patch_shape': (48, 144, 144),
    'num_classes': 6,
    # 'offset': (20, 46, 46),
    'target_discrete_ix': [3, 4, 5]
}

type_args = list(range(len(input_h5data)))
train_dataset = PatchCreator(
    input_h5data=[input_h5data[i] for i in type_args if i not in valid_indices],
    target_h5data=[target_h5data[i] for i in type_args if i not in valid_indices],
    train=True,
    epoch_size=args.epoch_size,
    cube_prios=None,
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
valid_dataset = None if (valid_indices is None) or (len(valid_indices) == 0) else PatchCreator(
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

optimizer = Padam(
    model.parameters(),
    lr=2e-3,
    weight_decay=0.5e-4,
    partial=1/4,
)

lr_stepsize = 200
lr_dec = 0.997
lr_sched = optim.lr_scheduler.StepLR(optimizer, lr_stepsize, lr_dec)
schedulers = {'lr': lr_sched}
# All these metrics assume a binary classification problem. If you have
#  non-binary targets, remember to adapt the metrics!
valid_metrics = {
}

criterion = HybridDiceLoss()

# Create trainer
trainer = Trainer(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    device=device,
    train_dataset=train_dataset,
    valid_dataset=valid_dataset,
    batchsize=4,
    num_workers=8,
    save_root=save_root,
    exp_name=args.exp_name,
    example_input=example_input,
    enable_save_trace=enable_save_trace,
    schedulers=schedulers,
    valid_metrics=valid_metrics,
    enable_videos=False,  # Uncomment to get rid of videos in tensorboard
    offset=train_dataset.offset,
    apply_softmax_for_prediction=True,
    num_classes=train_dataset.num_classes,
    ipython_shell=False,

)

# Archiving training script, src folder, env info
Backup(script_path=__file__, save_path=trainer.save_path).archive_backup()

# Start training
trainer.run(max_steps=max_steps, max_runtime=max_runtime)
