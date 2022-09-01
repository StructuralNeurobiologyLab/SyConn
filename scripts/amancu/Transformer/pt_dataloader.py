from logging import root
from torch.utils.data import Dataset, DataLoader
import torch
import pytorch_lightning as pl

from syconn.handler.prediction_pts import pts_loader_semseg_train_transformer
from elektronn3.data.transforms import Identity
elektronn3_avail = True
from sklearn import model_selection

from typing import Callable
import glob
import os
import numpy as np


class NodeFalseMergeLoader(Dataset):
    def __init__(self, split=None, fnames=None, radius=3000, npoints=20000, transform: Callable = Identity(),
                 train=True, batch_size=64, ctx_size=20000, mask_borders_with_id=None, regression=False):

        self.split = split
        self.fnames = fnames
        self.radius = radius
        self.num_pts = npoints
        self.transform = transform
        self.train = train
        self._batch_size = batch_size
        self.ctx_size = ctx_size
        self.mask_borders_with_id = mask_borders_with_id
        self.regression = regression

    def __getitem__(self, item):

        item = np.random.randint(0, self.__len__)

        sample_pts, sample_feats, out_nodes, out_labels, offset = self.load_sample(item)

        pts = torch.from_numpy(sample_pts).float()
        feats = torch.from_numpy(sample_feats).float()
        nodes = torch.from_numpy(out_nodes).float()
        if self.regression:
            lbs = torch.from_numpy(out_labels).float()
        else:
            lbs = torch.from_numpy(out_labels).long()
        return {'pts': pts, 'features': feats, 'out_pts': nodes,'target': lbs, 'extra': os.path.basename(self.fnames[item]), 'offset': offset}

    def __len__(self):
        return len(self.fnames)

    def load_sample(self, item):
        """
        Deterministic data loader.

        Args:
            item: Index in `py:attr:~fnames`.

        Returns:
            Numpy arrays of points, point features, target points and target labels.
        """
        p = self.fnames[item]
        sample_feats, sample_pts, out_pts, out_labels, offset = \
            [*pts_loader_semseg_train_transformer([p], self._batch_size, self.num_pts,
                                      transform=self.transform, ctx_size=self.ctx_size,
                                      use_subcell=False,
                                      mask_borders_with_id=self.mask_borders_with_id, gt_type='merger', regression=self.regression)][0]
        return sample_pts, sample_feats, out_pts, out_labels, offset

class PtNodeDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 64, radius=3000, npoints=20000, transforms=None, root=None,
                limit_num_samples=None, num_workers=32, shuffle=True):

        self.hclouds = root
        if self.hclouds is None:
            self.hclouds = f'/wholebrain/scratch/amancu/mergeError/Nodes/TrainingGT/R{radius}_downsample300/'

        self.fnames = glob.glob(self.hclouds + '*.pkl')

        if self.fnames == [] or self.fnames is None:
            raise BaseException(f'There have been no Hybridcloud pickles found at this location: {self.hclouds}')

        self.batch_size = batch_size
        self.radius = radius
        self.npoints = npoints
        self.transforms = transforms
        self.root = root
        self.limit_num_samples = limit_num_samples
        self.num_workers = num_workers
        self.shuffle = shuffle

        # only use 90% for train-val, 10% is always test
        # from which train is 80% and val is 20%
        train_test_split = int(0.9 * len(self.fnames))
        self.train_split, self.val_split = model_selection.train_test_split(self.fnames[:train_test_split], test_size=0.2, shuffle=self.shuffle)
        self.test_split = self.fnames[train_test_split:]

        # Limit all predefined paths to the number of limited samples
        if self.limit_num_samples is not None:
            self.train_split = self.train_split[:self.limit_num_samples]
            self.val_split = self.val_split[:self.limit_num_samples]
            self.test_split = self.test_split[:self.limit_num_samples]

        print(f'Processed split:\nTrain\t{len(self.train_split)}\nVal\t{len(self.val_split)}\nTest\t{len(self.test_split)}')

        self.train_dataset = NodeFalseMergeLoader('train', self.train_split, self.radius, self.npoints, 
                                                transform=self.transforms["train"], root=root)

        self.validation_dataset = NodeFalseMergeLoader('val', self.val_split, self.radius, self.npoints, 
                                                transform=self.transforms["val"], root=root)

        self.test_dataset = NodeFalseMergeLoader('test', self.test_split, self.radius, self.npoints, 
                                                transform=self.transforms["test"], root=root)


    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.validation_dataset, batch_size=self.batch_size,
                          num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size,
                          num_workers=self.num_workers, pin_memory=True)
