from torch.utils.data import Dataset
import torch
from syconn.handler.prediction_pts import pts_loader_semseg_train
from elektronn3.data.transforms import Identity
elektronn3_avail = True
from morphx.classes.hybridmesh import HybridCloud
from typing import Callable
import glob
import numpy as np

hybridCloudPath = '/wholebrain/scratch/amancu/mergeError/ptclouds/R50'

class CloudFalseMergeLoader(Dataset):
    def __init__(self, source_dir, radius, npoints=20000, transform: Callable = Identity(),
                     train=True, batch_size=1, ctx_size=15000, mask_boarders_with_id=None):
        if source_dir is None:
            source_dir = f'/ssdscratch/songbird/j0251/rag_flat_Jan2019_v2'
        self.source_dir = source_dir
        self.hclouds = f'/wholebrain/scratch/amancu/mergeError/ptclouds/R{radius}/Hybridcloud/'

        ssd = SuperSegmentationDataset(self.source_dir)

        # get all Hybridcloud files
        self.fnames = glob.glob(self.hclouds + '*.pkl')
        print(f'Using {len(self.fnames)} false mergers for training')
        self.radius = radius
        self.num_pts = npoints
        self.transform = transform
        self.train = train
        self._batch_size = batch_size
        self.ctx_size = ctx_size
        self.mask_boarders_with_id = mask_boarders_with_id


    def __getitem__(self, item):

        #random file selector

        item = np.random.randint(0, len(self.fnames))

        sample_pts, sample_feats, out_pts, out_labels = self.load_sample(item)

        pts = torch.from_numpy(sample_pts).float()
        feats = torch.from_numpy(sample_feats).float()
        out_pts = torch.from_numpy(out_pts).float()
        out_l = torch.from_numpy(out_labels).long()
        return {'pts': pts, 'features': feats, 'out_pts': out_pts, 'target': out_l}

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
        (sample_feats, sample_pts), (out_pts, out_labels) = \
            [*pts_loader_semseg_train([p], self._batch_size, self.num_pts,
                                      transform=self.transform, ctx_size=self.ctx_size,
                                      use_subcell=False,
                                      mask_boarders_with_id=self.mask_boarders_with_id, gt_type='merger', source_node_labels=True)][0]
        return sample_pts, sample_feats, out_pts, out_labels