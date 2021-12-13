import os.path

from torch.utils.data import Dataset
import torch
from syconn.handler.prediction_pts import pts_loader_semseg_train
from elektronn3.data.transforms import Identity

elektronn3_avail = True
from typing import Callable
import glob
import time
import os
import shutil


class FilterLoader(Dataset):
    def __init__(self, source_dir=None, radius=50, npoints=20000, transform: Callable = Identity(),
                 train=True, batch_size=1, ctx_size=15000, mask_borders_with_id=None, source_node_labels=(0, 1)):
        if source_dir is None:
            source_dir = f'/ssdscratch/songbird/j0251/rag_flat_Jan2019_v2'
        self.source_dir = source_dir
        self.hclouds = f'/wholebrain/scratch/amancu/mergeError/ptclouds/R{radius}/Hybridcloud/'

        # get all Hybridcloud files
        self.fnames = glob.glob(self.hclouds + '*.pkl')

        print(f'Using radius {radius} and {len(self.fnames)} cell samples for ' + (
            f'training' if train else f'validation'))
        self.radius = radius
        self.num_pts = npoints
        self.transform = transform
        self.train = train
        self._batch_size = batch_size
        self.ctx_size = ctx_size
        self.mask_borders_with_id = mask_borders_with_id
        self.source_node_lables = source_node_labels

    def __getitem__(self, item):

        try:
            sample_pts, sample_feats, out_labels = self.load_sample(item)
        except:
            print(f'[EXCEPTION] Moving... {self.fnames[item]}')
            shutil.move(self.fnames[item],
                        f'/wholebrain/scratch/amancu/mergeError/ptclouds/Dump/{self.radius}/' + os.path.basename(
                            self.fnames[item]))
            return

        # if sample_pts.shape[1] < 2300:
        #     print(f'Not enough points {sample_pts.shape[1]}. Moving... {self.fnames[item]}')
        #     shutil.move(self.fnames[item],
        #                 f'/wholebrain/scratch/amancu/mergeError/ptclouds/Dump/{self.radius}/' + os.path.basename(
        #                     self.fnames[item]))
        #     return
        # elapsed = time.time() - timer

        # if elapsed > 4.0:
        #     print(f'Moving... {self.fnames[item]}')
        #     shutil.move(self.fnames[item],
        #                 f'/wholebrain/scratch/amancu/mergeError/ptclouds/Dump/{self.radius}/' + os.path.basename(
        #                     self.fnames[item]))

        pts = torch.from_numpy(sample_pts).float()
        feats = torch.from_numpy(sample_feats).float()
        lbs = torch.from_numpy(out_labels).long()
        return {'pts': pts, 'features': feats, 'target': lbs}

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
        sample_feats, sample_pts, out_labels = \
            [*pts_loader_semseg_train([p], self._batch_size, self.num_pts,
                                      transform=self.transform, ctx_size=self.ctx_size,
                                      use_subcell=False,
                                      mask_borders_with_id=self.mask_borders_with_id, gt_type='merger',
                                      source_node_labels=self.source_node_lables)][0]
        return sample_pts, sample_feats, out_labels


class DeterministicLoader(Dataset):
    def __init__(self, source_dir=None, radius=50, npoints=20000, transform: Callable = Identity(),
                 train=True, batch_size=1, ctx_size=15000, mask_borders_with_id=None, source_node_labels=(0, 1)):
        if source_dir is None:
            source_dir = f'/ssdscratch/songbird/j0251/rag_flat_Jan2019_v2'
        self.source_dir = source_dir
        self.hclouds = f'/wholebrain/scratch/amancu/mergeError/ptclouds/R{radius}/Hybridcloud/'

        # get all Hybridcloud files
        self.fnames = glob.glob(self.hclouds + '*.pkl')

        if train:
            self.fnames = self.fnames[0:1]
        else:
            self.fnames = self.fnames[0:1]

        print(f'Using radius {radius} and {len(self.fnames)} cell samples for ' + (
            f'training' if train else f'validation'))
        filenames = [os.path.basename(x) for x in self.fnames]
        print(f'Using files: {filenames}')
        self.radius = radius
        self.num_pts = npoints
        self.transform = transform
        self.train = train
        self._batch_size = batch_size
        self.ctx_size = ctx_size
        self.mask_borders_with_id = mask_borders_with_id
        self.source_node_lables = source_node_labels

    def __getitem__(self, item):

        sample_pts, sample_feats, out_labels = self.load_sample(item)

        pts = torch.from_numpy(sample_pts).float()
        feats = torch.from_numpy(sample_feats).float()
        lbs = torch.from_numpy(out_labels).long()
        return {'pts': pts, 'features': feats, 'target': lbs, 'extra': os.path.basename(self.fnames[item])}

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
        sample_feats, sample_pts, out_labels = \
            [*pts_loader_semseg_train([p], self._batch_size, self.num_pts,
                                      transform=self.transform, ctx_size=self.ctx_size,
                                      use_subcell=False,
                                      mask_borders_with_id=self.mask_borders_with_id, gt_type='merger',
                                      source_node_labels=self.source_node_lables)][0]
        return sample_pts, sample_feats, out_labels