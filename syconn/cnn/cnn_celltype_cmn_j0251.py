#!/usr/bin/env python3

# Copyright (c) 2017 - now
# Max Planck Institute of Neurobiology, Munich, Germany
# Authors: Philipp Schubert

"""
"""
from syconn.cnn.TrainData import CelltypeViewsE3
import argparse
import zipfile
import numpy as np
import os
import torch
from torch import nn
from torch import optim
from elektronn3.models.simple import Conv3DLayer, StackedConv2Scalar
from elektronn3.data.transforms import RandomFlip
from elektronn3.data import transforms


class StackedConv2ScalarWithLatentAdd(nn.Module):
    """
    This class is a PyTorch module for a convolutional neural network model. The model is a stack of 
    Conv3DLayer layers with a final fully connected layer. The model also includes a latent addition 
    operation.
    
    Args:
        in_channels (int): The number of input channels.
        n_classes (int): The number of output classes.
        dropout_rate (float, optional): The dropout rate for each Conv3DLayer. Defaults to 0.08.
        act (str, optional): The activation function to use. Can be either 'relu' or 'leaky_relu'. 
            Defaults to 'relu'.
        n_scalar (int, optional): The number of scalar inputs to the fully connected layer. 
            Defaults to 1.
    """
    def __init__(self, in_channels, n_classes, dropout_rate=0.08, act='relu',
                 n_scalar=1):
        """
        docstring
        """
        super().__init__()
        if act == 'relu':
            act = nn.ReLU()
        elif act == 'leaky_relu':
            act = nn.LeakyReLU()
        self.seq = nn.Sequential(
            Conv3DLayer(in_channels, 20, (1, 5, 5), pooling=(1, 2, 2),
                        dropout_rate=dropout_rate, act=act),
            Conv3DLayer(20, 30, (1, 5, 5), pooling=(1, 2, 2),
                        dropout_rate=dropout_rate, act=act),
            Conv3DLayer(30, 40, (1, 4, 4), pooling=(1, 2, 2),
                        dropout_rate=dropout_rate, act=act),
            Conv3DLayer(40, 50, (1, 4, 4), pooling=(1, 2, 2),
                        dropout_rate=dropout_rate, act=act),
            Conv3DLayer(50, 60, (1, 2, 2), pooling=(1, 2, 2),
                        dropout_rate=dropout_rate, act=act),
            Conv3DLayer(60, 70, (1, 1, 1), pooling=(1, 2, 2),
                        dropout_rate=dropout_rate, act=act),
            Conv3DLayer(70, 70, (1, 1, 1), pooling=(1, 1, 1),
                        dropout_rate=dropout_rate, act=act),
        )  # given: torch.Size([1, 4, 20, 128, 256]), returns torch.Size([1, 31, 20, 1, 3])
        self.fc = nn.Sequential(
            nn.Linear(4200 + n_scalar, 100),
            act,
            nn.Linear(100, 50),
            act,
            nn.Linear(50, n_classes),
        )

    def forward(self, *args):
        """
        Defines the forward pass of the model. The input is passed through each Conv3DLayer in 
        sequence, then reshaped and concatenated with the scalar input before being passed through 
        the fully connected layer.
        
        Args:
            *args: The input to the model. This should be a tuple containing two elements: a tensor 
                representing the input data and a tensor representing the scalar input.
        
        Returns:
            The output of the model.
        """
        x, scal = args
        x = self.seq(x)
        x = x.view(x.size()[0], -1)
        x = torch.cat((x, scal), 1)
        x = self.fc(x)
        return x


def get_model():
    """
    This function creates and returns an instance of the StackedConv2ScalarWithLatentAdd model.
    
    Returns:
        StackedConv2ScalarWithLatentAdd: An instance of the model.
    """
    model = StackedConv2ScalarWithLatentAdd(in_channels=4, n_classes=11, n_scalar=2)
    # model = StackedConv2Scalar(in_channels=4, n_classes=8)
    return model


if __name__ == "__main__":
    lr = 1e-3
    lr_stepsize = 750
    lr_dec = 0.99
    batch_size = 20
    n_classes = 11
    parser = argparse.ArgumentParser(description='Train a network.')
    parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('-n', '--exp-name',
                        default='',
                        help='Manually set experiment name')
    parser.add_argument(
        '-m', '--max-steps', type=int, default=100e3,
        help='Maximum number of training steps to perform.'
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
    parser.add_argument('--sr', type=str, help='Save root', default=None)

    parser.add_argument('--seed', default=0, help='Random seed', type=int)
    parser.add_argument('--cval', default=None, help='Cross-validation split indicator.', type=int)

    args = parser.parse_args()
    if not args.disable_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print('Running on device: {}'.format(device))

    # Don't move this stuff, it needs to be run this early to work
    import elektronn3
    elektronn3.select_mpl_backend('agg')
    from elektronn3.training import Backup
    from elektronn3.training.trainer_scalarinput import Trainer

    cval = args.cval
    save_root = args.sr
    random_seed = args.seed
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    eval_nr = random_seed  # number of repetition

    if args.exp_name == '':
        name = f"celltype_cmn_j0251v2_adam_nbviews20_longRUN_2ratios_BIG_bs40_10fold"
    else:
        name = args.exp_name
    if cval is not None:
        name += f'_CV{cval}'
    else:
        name += f'_AllGT'
    name += f'_eval{eval_nr}'
    torch.manual_seed(eval_nr)

    # USER PATHS
    if save_root is None:
        save_root = os.path.expanduser('~/e3_training_10fold_eval/')
    max_steps = args.max_steps
    data_init_kwargs = {"raw_only": False, "nb_views": 20,
                        'nb_views_renderinglocations': 4, "cv_val": cval,
                        "reduce_context": 0, "reduce_context_fact": 1,
                        'random_seed': random_seed, "binary_views": False,
                        "n_classes": n_classes, 'class_weights': [1] * n_classes}

    model = get_model()
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        batch_size = batch_size * torch.cuda.device_count()
        # dim = 0 [20, xxx] -> [10, ...], [10, ...] on 2 GPUs
        model = nn.DataParallel(model)
    model.to(device)

    example_input = (torch.randn(1, 4, data_init_kwargs['nb_views'], 128, 256).to(device),
                     torch.randn(1, 2).to(device))
    # enable_save_trace = False if args.jit == 'disabled' else True
    # if args.jit == 'onsave':
    #     # Make sure that tracing works
    #     tracedmodel = torch.jit.trace(model, example_input)
    # elif args.jit == 'train':
    #     if getattr(model, 'checkpointing', False):
    #         raise NotImplementedError(
    #             'Traced models with checkpointing currently don\'t '
    #             'work, so either run with --disable-trace or disable '
    #             'checkpointing.')
    #     tracedmodel = torch.jit.trace(model, example_input)
    #     model = tracedmodel

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

    # Specify data set
    use_syntype_scal = True
    transform = transforms.Compose([RandomFlip(ndim_spatial=2), ])
    train_dataset = CelltypeViewsE3(is_j0251=True,
        train=True, transform=transform, use_syntype_scal=use_syntype_scal, **data_init_kwargs)
    valid_dataset = CelltypeViewsE3(is_j0251=True,
        train=False, transform=transform, use_syntype_scal=use_syntype_scal, **data_init_kwargs)

    # # Set up optimization
    optimizer = optim.Adam(
        model.parameters(),
        weight_decay=0.5e-4,
        lr=lr,
        amsgrad=True
    )
    lr_sched = optim.lr_scheduler.StepLR(optimizer, lr_stepsize, lr_dec)
    schedulers = {'lr': lr_sched}
    # All these metrics assume a binary classification problem. If you have
    #  non-binary targets, remember to adapt the metrics!
    val_metric_keys = []
    val_metric_vals = []
    valid_metrics = {}  # dict(zip(val_metric_keys, val_metric_vals))

    criterion = nn.CrossEntropyLoss().to(device)

    # Create and run trainer
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        batch_size=batch_size,
        num_workers=0,
        save_root=save_root,
        exp_name=name,
        schedulers=schedulers,
        valid_metrics=valid_metrics,
        ipython_shell=False,
    )

    # Archiving training script, src folder, env info
    bk = Backup(script_path=__file__, save_path=trainer.save_path).archive_backup()

    trainer.run(max_steps)
