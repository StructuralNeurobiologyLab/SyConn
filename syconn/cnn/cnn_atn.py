#!/usr/bin/env python3

# Copyright (c) 2017 - now
# Max Planck Institute of Neurobiology, Munich, Germany
# Authors: Philipp Schubert
import argparse
import os
from elektronn3.data.transforms import RandomFlip
from elektronn3.data import transforms
from elektronn3.models.simple import StackedConv2Scalar
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from elektronn3.training.schedulers import SGDR

# Dimension of latent space
Z_DIM = 25


class RepresentationNetwork(nn.Module):
    """
    This class defines the encoder network, a subclass of PyTorch's nn.Module.
    
    Args:
        n_in_channels (int): The number of input channels.
        n_out_channels (int): The number of output channels.
        dr (float): The dropout rate.
        leaky_relu (bool): If True, use LeakyReLU activation function, else use ReLU.
    """
    def __init__(self, n_in_channels, n_out_channels, dr=.0,
                 leaky_relu=True):
        DropOut = lambda: nn.Dropout2d(dr)
        act = nn.LeakyReLU if leaky_relu else nn.ReLU
        super().__init__()
        self.dr = dr
        self.n_out_channels = n_out_channels
        self.conv = nn.Sequential(
            nn.Conv2d(n_in_channels, 13, (5, 5)), nn.MaxPool2d((2, 2)), act(),
            nn.Conv2d(13, 19, (5, 5)), nn.MaxPool2d((2, 2)), act(),
            nn.Conv2d(19, 25, (4, 4)), DropOut(), act(),
            nn.Conv2d(25, 25, (4, 4)), DropOut(), nn.MaxPool2d((2, 2)), act(),
            nn.Conv2d(25, 30, (2, 2)), DropOut(), nn.MaxPool2d((2, 2)), act(),
            nn.Conv2d(30, 30, (2, 2)), nn.MaxPool2d((2, 2)), act(),
            nn.Conv2d(30, 31, 1), act(),
        )
        self.fc = nn.Sequential(
            # nn.AdaptiveMaxPool1d(200),  # flexible to various input sizes
            nn.Linear(6076, 50), act(),  # 6076 is hard-coded for this architecture and input size: 128, 256
            # nn.Linear(372, 50), act(),  # 372 is hard-coded for this architecture and input size: 128, 256
            # DropOut(),
            nn.Linear(50, n_out_channels)
        )

    def forward(self, x):
        """
        Defines the forward pass of the RepresentationNetwork.
        
        Args:
            x (torch.Tensor): The input tensor.
            
        Returns:
            torch.Tensor: The output tensor.
        """
        x = self.conv(x)  # representation network
        x = x.view(1, x.size()[0], -1)  # x.view(1, x.size()[0], -1) #flatten and # add auxiliary axis
        x = self.fc(x)
        x = x.view(x.size()[1:])
        return x  #.squeeze()  # get rid of auxiliary axis needed for AdaptiveMaxPool


class RepNetwork_v2(nn.Module):
    """
    This class is a wrapper to extend input dimension to match requirements of 
    `StackedConv2Scalar`. It is a subclass of the PyTorch nn.Module class.
    """
    def __init__(self):
        super().__init__()
        self.base_net = StackedConv2Scalar(in_channels=4, n_classes=Z_DIM, dropout_rate=0.05, act='relu')

    def forward(self, x):
        """
        Defines the forward pass of the RepNetwork_v2.
        
        Args:
            x (torch.Tensor): The input tensor.
            
        Returns:
            torch.Tensor: The output tensor.
        """
        x = x.unsqueeze(-3)
        return self.base_net(x)


class D_net_gauss(nn.Module):
    """
    This class defines a network for a Gaussian discriminator. It is a subclass of the PyTorch 
    nn.Module class. It is adapted from 
    https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/aae/aae.py.
    
    Args:
        z_dim (int): The dimension of the latent space.
    """
    def __init__(self, z_dim):
        super(D_net_gauss, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(z_dim * 3, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, z):
        """
        Defines the forward pass of the D_net_gauss.
        
        Args:
            z (torch.Tensor): The input tensor.
            
        Returns:
            torch.Tensor: The output tensor.
        """
        x = self.fc(z)
        return x


class TripletNet(nn.Module):
    """
    This class defines a Triplet Network, a subclass of the PyTorch nn.Module class. 
    It is adapted from https://github.com/andreasveit/triplet-network-pytorch/blob/master/tripletnet.py.
    
    Args:
        rep_net (nn.Module): The representation network to be used.
    """
    def __init__(self, rep_net):
        super().__init__()
        self.rep_net = rep_net

    def forward(self, x, y, z):
        """
        Defines the forward pass of the TripletNet.
        
        Args:
            x (torch.Tensor): The first input tensor.
            y (torch.Tensor): The second input tensor.
            z (torch.Tensor): The third input tensor.
            
        Returns:
            tuple: The distances and representations of the input tensors.
        """
        z_0 = self.rep_net(x)
        z_1 = self.rep_net(y)
        z_2 = self.rep_net(z)
        if self.train:
                dist_a = F.pairwise_distance(z_0, z_1, 2)
                dist_b = F.pairwise_distance(z_0, z_2, 2)
        else:
                dist_a = None
                dist_b = None
        return dist_a, dist_b, z_0, z_1, z_2


def get_model():
    """
    This function creates and returns a Triplet Network model.
    
    Returns:
        nn.Module: The Triplet Network model.
    """
    device = torch.device('cuda')
    RepNet = RepresentationNetwork(n_in_channels=4, n_out_channels=Z_DIM, dr=0.1,
                                   leaky_relu=False)
    # RepNet = RepNetwork_v2()
    return TripletNet(RepNet).to(device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a network.')
    parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('-n', '--exp-name', default="TN-25-Neighbors-3_SGDR_run2",
                        help='Manually set experiment name')
    parser.add_argument(
        '-m', '--max-steps', type=int, default=500000,
        help='Maximum number of training steps to perform.'
    )
    parser.add_argument(
        '-r', '--resume', metavar='PATH',
        help='Path to pretrained model state dict from which to resume training.'
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
    from syconn.cnn.TrainData import MultiviewData_TNet_online

    torch.manual_seed(0)

    # USER PATHS
    save_root = os.path.expanduser('~/e3training/')

    max_steps = args.max_steps
    lr = 0.0003
    lr_discr = 0.0001
    lr_stepsize = 500
    lr_dec = 0.985
    batch_size = 10
    margin = 0.2
    model = get_model()
    # if torch.cuda.device_count() > 1:
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     batch_size = batch_size * torch.cuda.device_count()
    #     model = nn.DataParallel(model)
    model.to(device)
    if args.resume is not None:  # Load pretrained network params
        model.load_state_dict(torch.load(os.path.expanduser(args.resume)))

    model_discr = D_net_gauss(Z_DIM)
    if torch.cuda.device_count() > 1:
        model_discr = nn.DataParallel(model_discr)
    model_discr.to(device)

    # Specify data set
    n_classes = 9
    data_init_kwargs = {"raw_only": False, "nb_views": 2, 'train_fraction': 0.95,
                        'nb_views_renderinglocations': 4, 'view_key': "4_large_fov",
                        "reduce_context": 0, "reduce_context_fact": 1, 'ctgt_key': "ctgt_v2", 'random_seed': 0,
                        "binary_views": False, "n_classes": n_classes, 'class_weights': [1] * n_classes}
    transform = transforms.Compose([RandomFlip(ndim_spatial=2), ])
    train_dataset = MultiviewData_TNet_online(train=True, transform=transform, allow_close_neigh=0,
                                              ctv_kwargs=data_init_kwargs, allow_axonview_gt=False)
    valid_dataset = MultiviewData_TNet_online(train=False, transform=transform, allow_close_neigh=0,
                                              ctv_kwargs=data_init_kwargs, allow_axonview_gt=False)
    # Set up optimization
    optimizer = optim.Adam(
        model.parameters(),
        weight_decay=0.5e-3,
        lr=lr,
        amsgrad=True
    )

    # optim. for discriminator model - true distr. vs. fake distr.
    optimizer_disc = optim.Adam(
        model_discr.parameters(),
        weight_decay=0.5e-3,
        lr=lr_discr,
        amsgrad=True
    )
    # lr_sched = optim.lr_scheduler.StepLR(optimizer, lr_stepsize, lr_dec)
    lr_sched = SGDR(optimizer, 20000, 3)
    # lr_sched = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    lr_discr_sched = optim.lr_scheduler.StepLR(optimizer_disc, lr_stepsize, lr_dec)
    # lr_discr_sched = optim.lr_scheduler.ReduceLROnPlateau(optimizer_disc, patience=10, factor=0.5)

    criterion = nn.MarginRankingLoss(margin=margin).to(device)
    criterion_discr = nn.BCELoss().to(device)

    # latent distribution
    # l_distr = Cauchy(torch.tensor([0.0]), torch.tensor([1.0]))
    # l_sample_func = lambda n, z: l_distr.rsample((n, z)).squeeze()
    l_sample_func = lambda n, z: torch.randn(n, z) #+ torch.randn(1)  # make loss mean invariant
    # bimodal normals
    # l_distr = lambda : Normal(torch.tensor([-3.0]), torch.tensor([1.0])) if np.random.randint(2) else Normal(torch.tensor([3.0]), torch.tensor([1.0]))
    # l_sample_func = lambda n, z: l_distr().rsample((n, z)).squeeze()
    # Create and run trainer
    trainer = TripletNetTrainer(
        model=[model, model_discr],
        criterion=[criterion, criterion_discr],
        optimizer=[optimizer, optimizer_disc],
        device=device,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        batchsize=batch_size,
        num_workers=2,
        save_root=save_root,
        exp_name=args.exp_name,
        schedulers={"lr": lr_sched, "lr_discr": lr_discr_sched},
        alpha=1e-6, alpha2=0.0,  # Adv. regularization will make up (alpha2 * 100)% of the total loss
        latent_distr=l_sample_func
    )

    # Archiving training script, src folder, env info
    bk = Backup(script_path=__file__,save_path=trainer.save_path).archive_backup()

    trainer.run(max_steps)
