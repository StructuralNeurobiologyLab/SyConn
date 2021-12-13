from torch.utils.data import Dataset
import torch
from syconn.handler.prediction_pts import pts_loader_semseg_train
from elektronn3.data.transforms import Identity
elektronn3_avail = True
from typing import Callable
import glob
import os
import numpy as np


class CloudFalseMergeLoader(Dataset):
    def __init__(self, source_dir=None, radius=50, npoints=20000, transform: Callable = Identity(),
                     train=True, batch_size=1, ctx_size=20000, mask_borders_with_id=None, source_node_labels=(0,1)):
        self.hclouds = f'/wholebrain/scratch/amancu/mergeError/ptclouds/R{radius}/Hybridcloud/'

        # get all Hybridcloud files
        self.fnames = fnames = glob.glob(self.hclouds + '*.pkl')

        self.train_limit = int(0.8*(len(self.fnames)))
        self.radius = radius
        self.num_pts = npoints
        self.transform = transform
        self.train = train
        self._batch_size = batch_size
        self.ctx_size = ctx_size
        self.mask_borders_with_id = mask_borders_with_id
        self.source_node_labels = source_node_labels

        appen = f'{self.train_limit} cell samples for training' if self.train else f'{len(self.fnames) - self.train_limit} cell samples for validation'
        print(f'Using radius {radius} and ' + appen)

    def __getitem__(self, item):

        # random file selector
        # get from training data
        if self.train:
            item = np.random.randint(0, self.train_limit)
        # get from validation data
        else:
            item = np.random.randint(self.train_limit, len(self.fnames))

        sample_pts, sample_feats, out_labels = self.load_sample(item)

        pts = torch.from_numpy(sample_pts).float()
        feats = torch.from_numpy(sample_feats).float()
        lbs = torch.from_numpy(out_labels).long()
        return {'pts': pts, 'features': feats, 'target': lbs, 'extra': os.path.basename(self.fnames[item])}

    def __len__(self):
        return self.train_limit if self.train else (len(self.fnames) - self.train_limit)

    def load_sample(self, item):
        """
        Deterministic data loader.

        Args:
            item: Index in `py:attr:~fnames`.

        Returns:
            Numpy arrays of points, point features, target points and target labels.
        """
        p = self.fnames[item]
        sample_feats, sample_pts, out_labels = \
            [*pts_loader_semseg_train([p], self._batch_size, self.num_pts,
                                      transform=self.transform, ctx_size=self.ctx_size,
                                      use_subcell=False,
                                      mask_borders_with_id=self.mask_borders_with_id, gt_type='merger', source_node_labels=self.source_node_labels)][0]
        return sample_pts, sample_feats, out_labels