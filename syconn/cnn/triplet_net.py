#!/usr/bin/env python3

# Copyright (c) 2017 - now
# Max Planck Institute of Neurobiology, Munich, Germany
# Authors: Philipp Schubert
import argparse
import os
from elektronn3.models.fcn_2d import *
from elektronn3.data.transforms import RandomFlip
from elektronn3.data import transforms
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F


def passthrough(x, **kwargs):
    return x


class RepresentationNetwork(nn.Module):
    """
    Encoder network
    """
    def __init__(self, n_in_channels, n_out_channels=10, dr=.0,
                 leaky_relu=True):
        if dr > 0:
            DropOut = lambda: nn.Dropout3d(dr)
        else:
            DropOut = passthrough
        act = nn.LeakyReLU if leaky_relu else nn.ReLU
        super().__init__()
        self.dr = dr
        self.n_out_channels = n_out_channels
        self.conv = nn.Sequential(
            nn.Conv2d(n_in_channels, 15, (5, 5)), act(),
            nn.Conv2d(15, 19, (5, 5)),act(),
            nn.MaxPool3d((2, 2)),
            nn.Conv2d(19, 25, (4, 4)), act(),
            DropOut(),
            nn.Conv2d(25, 25, (4, 4)), act(),
            nn.MaxPool3d((2, 2)),
            nn.Conv2d(25, 30, (2, 2)), act(),
            DropOut(),
            nn.Conv2d(30, 32, (2, 2)), act(),
            nn.MaxPool3d((2, 2)),
            nn.Conv2d(32, 32, 1),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(100, 50), act())
        self.fc2 = nn.Linear(50, n_out_channels)

    def forward(self, x):
        x = self.conv(x)  # representation network
        x = self.fc1(x)
        x = F.dropout(x, p=self.dr, training=self.training)
        return self.fc2(x)


# Discriminator
class D_net_gauss(nn.Module):
    """
    adapted from https://blog.paperspace.com/adversarial-autoencoders-with-pytorch/
    """
    # z_dim has to be equal to n_out_channels in TrupletNet
    def __init__(self, z_dim=10, dr=.0):
        super().__init__()
        if dr > 0:
            DropOut = lambda: nn.Dropout3d(dr)
        else:
            DropOut = passthrough
        self.fc = nn.Sequential(nn.Linear(z_dim, 250), nn.ReLU(),
                                DropOut(),
                                nn.Linear(250, 100), nn.ReLU(),
                                DropOut(),
                                nn.Linear(100, 1))

    def forward(self, x):
        x = self.fc(x)
        return F.sigmoid(x)


class TripletNet(nn.Module):
    """
    adapted from https://github.com/andreasveit/triplet-network-pytorch/blob/master/tripletnet.py
    """
    def __init__(self, n_in_channels, n_out_channels=10, dr=.0,
                 leaky_relu=True):
        super().__init__()
        self.rep_net = RepresentationNetwork(n_in_channels, n_out_channels,
                                             dr, leaky_relu)

    def forward(self, x, y, z):
        z_0 = self.rep_net(x)
        z_1 = self.rep_net(y)
        z_2 = self.rep_net(z)
        dist_a = F.pairwise_distance(z_0, z_1, 2)
        dist_b = F.pairwise_distance(z_0, z_2, 2)
        return dist_a, dist_b, z_0, z_1, z_2


def get_model():
    return TripletNet(n_in_channels=4,
                      n_out_channels=10, dr=0.1).to(device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a network.')
    parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('-n', '--exp-name', default="FCN-VGG13--BlurryBoundary--NewGT", help='Manually set experiment name')
    parser.add_argument(
        '-m', '--max-steps', type=int, default=500000,
        help='Maximum number of training steps to perform.'
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

    from elektronn3.training import Backup
    from elektronn3.training.trainer_tnet import TripletNetTrainer
    from syconn.cnn.TrainData import MultiviewData_TNet

    torch.manual_seed(0)


    # USER PATHS
    save_root = os.path.expanduser('~/e3training/')

    max_steps = args.max_steps
    lr = 0.004
    lr_stepsize = 500
    lr_dec = 0.99
    batch_size = 20
    margin = 0.2

    model = get_model()
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        batch_size = batch_size * torch.cuda.device_count()
        model = nn.DataParallel(model)
    model.to(device)

    model_discr = D_net_gauss(dr=0.1)
    if torch.cuda.device_count() > 1:
        model_discr = nn.DataParallel(model_discr)
    model_discr.to(device)

    # Specify data set
    transform = transforms.Compose([RandomFlip(ndim_spatial=2), ])
    train_dataset = MultiviewData_TNet(train=True, transform=transform)
    valid_dataset = MultiviewData_TNet(train=False, transform=transform)

    # Set up optimization
    optimizer = optim.Adam(
        model.parameters(),
        weight_decay=0.5e-4,
        lr=lr,
        amsgrad=True
    )
    optimizer_disc = optim.Adam(
        model.parameters(),
        weight_decay=0.5e-4,
        lr=lr,
        amsgrad=True
    )
    lr_sched = optim.lr_scheduler.StepLR(optimizer, lr_stepsize, lr_dec)

    criterion = nn.MarginRankingLoss(margin).to(device)

    # Create and run trainer
    trainer = TripletNetTrainer(
        model=[model, model_discr],
        criterion=criterion,
        optimizer=[optimizer, optimizer_disc],
        device=device,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        batchsize=batch_size,
        num_workers=2,
        save_root=save_root,
        exp_name=args.exp_name,
        schedulers={"lr": lr_sched},
        ipython_on_error=False
    )

    # Archiving training script, src folder, env info
    bk = Backup(script_path=__file__,save_path=trainer.save_path).archive_backup()

    trainer.train(max_steps)
