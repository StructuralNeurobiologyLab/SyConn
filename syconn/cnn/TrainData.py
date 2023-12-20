# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max-Planck-Institute of Neurobiology, Munich, Germany
# Authors: Philipp Schubert, Sven Dorkenwald, Jörgen Kornfeld
# non-relative import needed for this file in order to be importable by
# ELEKTRONN2 architectures
try:
    import open3d as o3d
except ImportError:
    pass  # for sphinx build
import os
import re
import numpy as np
import pandas
from typing import Optional, Tuple, Dict, List, Union
import warnings
from syconn.handler.basics import load_pkl2obj, temp_seed, kd_factory
from syconn.handler.prediction import naive_view_normalization, naive_view_normalization_new, str2int_converter
from syconn.handler.prediction_pts import pts_loader_scalar, \
    pts_loader_local_skel, pts_loader_semseg_train
from syconn.reps.super_segmentation import SuperSegmentationDataset, SegmentationObject
from syconn.reps.super_segmentation_helper import syn_sign_ratio_celltype
from syconn.reps.segmentation import SegmentationDataset
from syconn import global_params
from syconn.handler import log_main as log_cnn
try:
    from vigra.filters import boundaryVectorDistanceTransform
    import vigra
except ImportError as e:
    log_cnn.error(str(e))
import threading
from sklearn.model_selection import train_test_split, StratifiedKFold
try:
    from torch.utils.data import Dataset
    import torch
    from elektronn3.data.transforms import Identity
    elektronn3_avail = True
except ImportError as e:
    elektronn3_avail = False
    Dataset = None
    Identity = None
from typing import Callable
from sklearn.utils.class_weight import compute_class_weight
import h5py
import glob
from scipy import spatial, ndimage
import time
import socket
# fix random seed.
np.random.seed(0)


# -------------------------------------- elektronn3 ----------------------------
if elektronn3_avail:

    class CellCloudData(Dataset):
        """
        This class is a loader for cell vertices. It is used to initialize the SuperSegmentationDataset which 
        contains the ground truth. It also samples random points (with features) from the cell vertices. The 
        features are set to ``dict(sv=0, mi=1, vc=2, syn_ssv=3)`` depending on the vertex type. The subset of 
        vertices is drawn randomly (uniformly).
        """
        def __init__(self, ssd_kwargs=None, npoints=20000, transform: Callable = Identity(),
                     train=True, cv_val=0, cellshape_only=False, ctx_size=20000, use_syntype=True,
                     onehot=True, batch_size=1, map_myelin: bool = False):
            """
            Initializes the CellCloudData object with the given parameters.
            
            Args:
                ssd_kwargs (dict): Kwargs to init the SuperSegmentationDataset which 
                    contains the ground truth.
                npoints (int): Number of points to be sampled.
                transform (Callable): Transformations which are applied in 
                    `__getitem__`.
                train (bool): If True, training data will be used with key 'train'. 
                    If False, validation data will be used with key 'valid'.
                cv_val (int): Cross validation value.
                cellshape_only (bool): If True, only cell shape will be considered.
                ctx_size (int): Context size.
                use_syntype (bool): If True, uses different features for symmetric 
                    and asymmetric synapses.
                onehot (bool): If True, one-hot encoding will be used.
                batch_size (int): Size of the batch.
                map_myelin (bool): If True, myelin will be mapped.
            """
            if not onehot:
                raise NotImplementedError
            super().__init__()
            if ssd_kwargs is not None:
                self.ssd_kwargs = ssd_kwargs
            elif 'wb' not in socket.gethostname():
                wd_path = os.path.expanduser('~/mnt/wb//wholebrain/songbird/j0126/areaxfs_v6/')
                self.ssd_kwargs = dict(working_dir=wd_path, version='ctgt_v4')
            else:
                self.ssd_kwargs = dict(working_dir='/wholebrain/songbird/j0126/areaxfs_v6/',
                                       version='ctgt_v4')
            self.ctx_size = ctx_size
            ssd = SuperSegmentationDataset(**self.ssd_kwargs)
            self.ssd = ssd
            self.sso_ids = None
            gt_dir = ssd.path
            self.map_myelin = map_myelin
            self.cv_val = cv_val
            log_cnn.info(f'Set {ssd} as GT source.')
            split_dc_path = f'{gt_dir}/ctgt_v4_splitting_cv0_10fold.pkl'
            if os.path.isfile(split_dc_path):
                if cv_val is -1:
                    log_cnn.critical(f'"cval_val" was set to -1. training will also '
                                     f'include validation data.')
                    split_dc = load_pkl2obj(split_dc_path)
                    split_dc['train'].extend(split_dc['valid'])
                else:
                    split_dc_path = f'{gt_dir}/ctgt_v4_splitting_cv{cv_val}_10fold.pkl'
                    split_dc = load_pkl2obj(split_dc_path)
                    # Do not use validation split during training. Use training samples for validation
                    # error instead (should still be informative due to missing augmentations)
                    split_dc['valid'] = split_dc['train']
                label_dc = load_pkl2obj(f'{gt_dir}/ctgt_v4_labels.pkl')
            else:
                split_dc = None
                label_dc = None
            self.train = train
            self.num_pts = npoints
            self._batch_size = batch_size
            self._curr_ssv_id = None
            self.cellshape_only = cellshape_only
            self.use_syntype = use_syntype
            self.onehot = onehot
            self.transform = transform
            if use_syntype:
                self._num_obj_types = 5
            else:
                self._num_obj_types = 4
            self.label_dc = label_dc
            self.splitting_dict = split_dc
            if self.splitting_dict is not None and self.label_dc is not None:
                self.sso_ids = self.splitting_dict['train'] if train else self.splitting_dict['valid']
                for ix in self.sso_ids:
                    if ix not in ssd.ssv_ids:
                        raise ValueError(f'SSO with ID {ix} is not part of {ssd}!')
                log_cnn.debug(f'Using splitting dict at "{split_dc_path}".')
                for k, v in self.splitting_dict.items():
                    classes, c_cnts = np.unique([self.label_dc[ix] for ix in
                                                 self.splitting_dict[k]], return_counts=True)
                    log_cnn.debug(f"{k} [labels, counts]: {classes}, {c_cnts}")

        def __getitem__(self, item):
            """
            Samples random points (with features) from the cell vertices. Features are set 
            to ``dict(sv=0, mi=1, vc=2, syn_ssv=3)`` depending on the vertex type. The 
            subset of vertices is drawn randomly (uniformly).
            
            Args:
                item (int): If ``self.train=True``, `item` will be overwritten by
                    ``np.random.randint(0, len(self.fnames))``.
            
            Returns:
                dict: Point array (N, 3), feature array (N, ), cell label (scalar). N
                is the number of points set during initialization.
            """
            item = np.random.randint(0, len(self.sso_ids))
            self._curr_ssv_id = self.sso_ids[item]
            pts, feats = self.load_ssv_sample(item)
            lbs = np.array([self.label_dc[self._curr_ssv_id]]*self._batch_size, dtype=np.int32)
            pts = torch.from_numpy(pts).float()
            lbs = torch.from_numpy(lbs[..., None]).long()
            feats = torch.from_numpy(feats).float()
            return {'pts': pts, 'features': feats, 'target': lbs}

        def __len__(self):
            """
            Returns the length of the dataset.
            
            Returns:
                int: Length of the dataset.
            """
            if self.train:
                # make use of the underlying LRU cache with high epoch size,
                # worker instances of the pytorch loader will reset after each epoch
                return len(self.sso_ids) * 20
            else:
                return max(len(self.sso_ids) // 5, 1)

        def load_ssv_sample(self, item: int, draw_local: bool = False, draw_local_dist: float = 1000):
            """
            Loads a sample from the SuperSegmentationDataset.
            
            Args:
                item (int): Cell ID.
                draw_local (bool): If True, two similar samples from the same location will 
                be sampled.
                draw_local_dist (float): Maximum distance to the location used for generating 
                the "similar" sample. Note that the location is drawn randomly from all 
                skeleton nodes within the traversed path (within `draw_local_dist`).
            
            Internal parameters:
                * `feat_dc`: Labels for the different point types:
                  ``dict(sv=0, mi=1, vc=2, syn_ssv=3, syn_ssv_sym=3, syn_ssv_asym=4)``
            
            Returns:
                tuple: If draw_local is True, two tuples of points and features: 
                [(pts0, feat0), (pts1, feat1)]. Else, point and feature array; if batch size 
                is 1, the first axis is removed.
            """
            if draw_local:
                sso_id, (sample_feats, sample_pts) = [*pts_loader_scalar(
                    self.ssd_kwargs, [self.sso_ids[item], ] * 2, self._batch_size * 2,
                    self.num_pts, transform=self.transform, ctx_size=self.ctx_size,
                    train=True, draw_local=True, cache=False, map_myelin=self.map_myelin,
                    use_syntype=self.use_syntype, cellshape_only=self.cellshape_only,
                    draw_local_dist=draw_local_dist)][0]
            else:
                sso_id, (sample_feats, sample_pts) = [*pts_loader_scalar(
                    self.ssd_kwargs, [self.sso_ids[item], ], self._batch_size,
                    self.num_pts, transform=self.transform, ctx_size=self.ctx_size,
                    use_syntype=self.use_syntype, cellshape_only=self.cellshape_only,
                    train=True, cache=False, map_myelin=self.map_myelin)][0]
            assert np.unique(sso_id) == self.sso_ids[item]
            if self._batch_size == 1 and not draw_local:
                return sample_pts[0], sample_feats[0]
            else:
                return sample_pts, sample_feats


    class CellCloudDataTriplet(CellCloudData):
        """
        This class is a loader for triplets of cell vertices. It is used to load 
        triplets of cell vertices for training purposes.
        """

        def __init__(self, draw_local: bool = True, draw_local_dist: float = 1000, **kwargs):
            """
            Initializes the CellCloudDataTriplet object with the given parameters.
            
            Args:
                draw_local (bool): If True, two similar samples from the same location will be 
                    sampled. False will learn similarities of cells, not local morphology.
                draw_local_dist (float): Maximum distance to the location used for generating 
                    the "similar" sample.
                **kwargs: Additional keyword arguments.
            """
            super().__init__(**kwargs)
            if self.sso_ids is None:
                bb = self.ssd.load_numpy_data('bounding_box') * self.ssd.scaling  # N, 2, 3
                bb = np.linalg.norm(bb[:, 1] - bb[:, 0], axis=1)
                self.sso_ids = self.ssd.ssv_ids[bb > 2 * self.ctx_size]
                print(f'Using {len(self.sso_ids)} SSVs from {self.ssd} for triplet training.')
            self._curr_ssv_id_altern = None
            self.draw_local = draw_local
            self.draw_local_dist = draw_local_dist

        def __getitem__(self, item):
            """
            Returns a triplet of cell vertices.
            
            Args:
                item (int): Index of the item to be fetched.
            
            Returns:
                tuple: A triplet of cell vertices.
            """
            while True:
                try:
                    item = np.random.randint(0, len(self.sso_ids))
                    self._curr_ssv_id_altern = self.sso_ids[item]
                    pts_altern, feats_altern = self.load_ssv_sample(item)
                    pts_altern = torch.from_numpy(pts_altern).float()
                    feats_altern = torch.from_numpy(feats_altern).float()
                    break
                except ValueError as e:
                    print(f'Exception occurred during CellCloudDataTriplet._getitem__ with SSV ID {self.sso_ids[item]}: {str(e)}')
            while True:
                try:
                    # draw base and similar sample from a different cell
                    while True:
                        ix = np.random.randint(0, len(self.sso_ids))
                        if self.sso_ids[ix] != self._curr_ssv_id_altern:
                            self._curr_ssv_id = self.sso_ids[ix]
                            break
                    if self.draw_local:
                        # consecutive samples belong together
                        pts, feats = self.load_ssv_sample(ix, draw_local=True, draw_local_dist=self.draw_local_dist)
                        pts0, pts1 = pts[0::2], pts[1::2]
                        feats0, feats1 = feats[0::2], feats[1::2]
                    else:
                        pts0, feats0 = self.load_ssv_sample(ix)  # base sample
                        pts1, feats1 = self.load_ssv_sample(ix)  # similar sample to base

                    x0 = {'pts': torch.from_numpy(pts0).float(), 'features':
                          torch.from_numpy(feats0).float()}
                    x1 = {'pts': torch.from_numpy(pts1).float(), 'features':
                          torch.from_numpy(feats1).float()}
                    x2 = {'pts': pts_altern, 'features': feats_altern}  # alternative sample
                    break
                except ValueError as e:
                    print(f'Exception occurred during CellCloudDataTriplet._getitem__ with SSV ID {self.sso_ids[item]}: {str(e)}')
            return x0, x1, x2

        def __len__(self):
            """
            Returns the length of the dataset.
            
            Returns:
                int: Length of the dataset.
            """
            return 20000


    class CellCloudDataJ0251(CellCloudData):
        """
        This class is a loader for cell vertices. It uses the same data for train and valid set.
        """
        def __init__(self, cv_val=None, **kwargs):
            """
            Initializes the CellCloudDataJ0251 object with the given parameters.
            
            Args:
                cv_val (int): Cross validation value.
                **kwargs: Additional keyword arguments.
            """
            ssd_kwargs = dict(working_dir='/ssdscratch/pschuber/songbird/j0251/rag_flat_Jan2019_v3/')

            super().__init__(ssd_kwargs=ssd_kwargs, cv_val=cv_val, **kwargs)
            # load GT
            assert self.train, "Other mode than 'train' is not implemented."
            self.csv_p = "/wholebrain/songbird/j0251/groundtruth/celltypes/j0251_celltype_gt_v4.csv"
            df = pandas.io.parsers.read_csv(self.csv_p, header=None, names=['ID', 'type']).values
            ssv_ids = df[:, 0].astype(np.uint64)
            if len(np.unique(ssv_ids)) != len(ssv_ids):
                ixs, cnt = np.unique(ssv_ids, return_counts=True)
                raise ValueError(f'Multi-usage of IDs! {ixs[cnt > 1]}')
            str_labels = df[:, 1]
            ssv_labels = np.array([str2int_converter(el, gt_type='ctgt_j0251_v2') for el in str_labels], dtype=np.uint16)
            if self.cv_val is not None and self.cv_val != -1:
                assert self.cv_val < 10
                kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
                for ii, (train_ixs, test_ixs) in enumerate(kfold.split(ssv_ids, y=ssv_labels)):
                    if ii == self.cv_val:
                        self.splitting_dict = {'train': ssv_ids[train_ixs], 'valid': ssv_ids[test_ixs]}
            else:
                self.splitting_dict = {'train': ssv_ids, 'valid': ssv_ids}  # use all data
                log_cnn.critical(f'Using all GT data for training!')
            self.label_dc = {k: v for k, v in zip(ssv_ids, ssv_labels)}
            self.sso_ids = self.splitting_dict['train']
            for k, v in self.splitting_dict.items():
                classes, c_cnts = np.unique([self.label_dc[ix] for ix in
                                             self.splitting_dict[k]], return_counts=True)
                log_cnn.debug(f"{k} [labels, counts]: {classes}, {c_cnts}")
                log_cnn.debug(f'{len(self.sso_ids)} SSV IDs in training set: {self.sso_ids}')

        def __len__(self):
            """
            Returns the length of the dataset.
            
            Returns:
                int: Length of the dataset.
            """
            if self.train:
                # make use of the underlying LRU cache with high epoch size,
                # worker instances of the pytorch loader will reset after each epoch
                return len(self.sso_ids) * 60
            else:
                return max(len(self.sso_ids) // 5, 1)


    class CellCloudGlia(Dataset):
        """
        This class is a loader for cell vertices. It is used to load and prepare data for training 
        and validation. It inherits from the Dataset class. Loader for cell vertices.
        """

        def __init__(self, npoints=20000, transform: Callable = Identity(),
                     train=True, batch_size=1, use_subcell=False, ctx_size=15000):
            """
            Initializes the CellCloudGlia class with the given parameters.
            
            Args:
                npoints (int): The number of points to be sampled from the cell vertices.
                transform (Callable): The transformations to be applied in `__getitem__`.
                train (bool): If True, training data will be used. If False, validation data 
                              will be used with key 'valid'.
                batch_size (int): The size of the batch to be used for training or validation.
                use_subcell (bool): If True, subcellular structures will be used.
                ctx_size (int): The size of the context to be used for training or validation.
            """
            if 'wb' not in socket.gethostname():
                wd_path = os.path.expanduser('~/mnt/wb//wholebrain/songbird/j0126/areaxfs_v6/')
            else:
                wd_path = '/wholebrain/songbird/j0126/areaxfs_v6/'
            ssd = SuperSegmentationDataset(wd_path)

            # Define specific subset with no glia merges
            nonglia_ssv_ids = np.array([
                10919937, 16096256, 23144450, 2734465, 34811392, 491527,
                15933443, 16113665, 24414208, 2854913, 37558272, 8339462,
                15982592, 18571264, 26501121, 33581058, 46319619
            ], dtype=np.uint64)
            # use celltype GT
            csv_p = '/wholebrain/songbird/j0126/GT/celltype_gt/j0126_cell_type_gt_areax_fs6_v3.csv'
            df = pandas.io.parsers.read_csv(csv_p, header=None, names=['ID', 'type']).values
            nonglia_ssv_ids = np.concatenate([df[:, 0].astype(np.uint64), nonglia_ssv_ids])
            self.ctx_size = ctx_size
            self.sso_params = {sso.id: dict(**sso.ssv_kwargs) for sso in ssd.get_super_segmentation_object(nonglia_ssv_ids)}
            ssd_glia = SuperSegmentationDataset(working_dir=wd_path, version='gliagt')
            self.sso_params.update({sso.id: dict(**sso.ssv_kwargs) for sso in ssd_glia.ssvs})
            log_cnn.info(f'Set {ssd} as GT source.')

            # get dataset statistics
            self.label_dc = {sso_id: 0 for sso_id in nonglia_ssv_ids}
            self._label_dc_neuron = self.label_dc.copy()
            self._label_dc_glia = {sso_id: 1 for sso_id in ssd_glia.ssv_ids}
            self.label_dc.update(self._label_dc_glia)
            classes, c_cnts = np.unique([self.label_dc[ix] for ix in
                                         self.label_dc], return_counts=True)
            size_cnt_glia = [sso.size for sso in ssd_glia.ssvs]
            size_cnt_neuron = [sso.size for sso in ssd.get_super_segmentation_object(nonglia_ssv_ids)]
            print(f"neuron (0): {np.sum(size_cnt_neuron)//1e9} GVx\n"
                  f"glia (1): {np.sum(size_cnt_glia)//1e9} GVx")
            print(f"[labels, cell count]: {classes}, {c_cnts}")

            if use_subcell:  # TODO: add syntype
                self._num_obj_types = 4
            else:
                self._num_obj_types = 1
            self.use_subcell = use_subcell
            self.train = train
            self.num_pts = npoints
            self._batch_size = batch_size
            self._curr_ssv_params = None
            self._curr_ssv_label = None
            self.transform = transform

        def __getitem__(self, item):
            """
            Samples random points (with features) from the cell vertices. The subset 
            of vertices is drawn randomly. Features are set to ``dict(sv=0, mi=1, 
            vc=2, syn_ssv=3)`` depending on the vertex type.
            
            Args:
                item (int): If `self.train=True`, `item` will be overwritten by a 
                random integer.
            
            Returns:
                dict: A dictionary containing the point array, feature array, cell 
                label, and output points. Point array (N, 3), feature array (N, ), 
                cell label (scalar). N is the number of points set during 
                initialization.
            """
            if np.random.randint(3) == 1:
                # reduce context and merge glia and neuron samples
                orig_ctx = self.ctx_size
                orig_npts = self.num_pts
                self.ctx_size *= 0.75
                self.num_pts = int(0.5 * self.num_pts)
                item = np.random.choice(list(self._label_dc_neuron.keys()))
                pts, feats, out_pts, out_l = self.load_ssv_sample(item)
                item = np.random.choice(list(self._label_dc_glia.keys()))
                pts_g, feats_g, out_pts_g, out_l_g = self.load_ssv_sample(item)
                conc_axis = 1 if self._batch_size > 1 else 0
                pts = np.concatenate([pts, pts_g], axis=conc_axis)
                feats = np.concatenate([feats, feats_g], axis=conc_axis)
                out_pts = np.concatenate([out_pts, out_pts_g], axis=conc_axis)
                out_l = np.concatenate([out_l, out_l_g], axis=conc_axis)
                self.ctx_size = orig_ctx
                self.num_pts = orig_npts
            else:
                item = np.random.choice(list(self.label_dc.keys()))
                pts, feats, out_pts, out_l = self.load_ssv_sample(item)
            pts = torch.from_numpy(pts).float()
            feats = torch.from_numpy(feats).float()
            out_pts = torch.from_numpy(out_pts).float()
            out_l = torch.from_numpy(out_l).long()
            return {'pts': pts, 'features': feats, 'out_pts': out_pts, 'target': out_l}

        def __len__(self):
            """
            Returns the length of the dataset. If `self.train=True`, the length is the number of super segmentation
            objects times 15. Otherwise, it's the maximum of the number of super segmentation objects divided by 10 and 1.
            
            Returns:
                int: The length of the dataset.
            """
            if self.train:
                # make use of the underlying LRU cache with high epoch size,
                # worker instances of the pytorch loader will reset after each epoch
                return len(self.sso_params) * 15
            else:
                return max(len(self.sso_params) // 10, 1)

        def load_ssv_sample(self, item: int):
            """
            Loads a sample from the super segmentation object.
            
            Args:
                item (int): The ID of the cell.
            
            Internal parameters:
                * `feat_dc`: Labels for the different point types:
                  ``dict(sv=0, mi=1, vc=2, syn_ssv=3, syn_ssv_sym=3, syn_ssv_asym=4)``
            
            Returns:
                tuple: A tuple containing the point array, feature array, output points, 
                and output labels. If batch size is 1, the first axis is removed.
            """
            self._curr_ssv_params = self.sso_params[item]
            self._curr_ssv_label = self.label_dc[self.sso_params[item]['ssv_id']]
            # Changed pts_loader_local_skel=True, 10Sep2020 PS
            sso_id, (sample_feats, sample_pts), (out_pts, out_labels) = \
                [*pts_loader_local_skel([self.sso_params[item]], [self._curr_ssv_label], self._batch_size,
                                        self.num_pts, transform=self.transform, use_subcell=self.use_subcell,
                                        train=True, ctx_size=self.ctx_size, recalc_skeletons=True)][0]
            if self._batch_size == 1:
                return sample_pts[0], sample_feats[0], out_pts[0], out_labels[0]
            else:
                return sample_pts, sample_feats, out_pts, out_labels


    class CloudDataSemseg(Dataset):
        """
        This class is used to load and prepare data for semantic segmentation. It inherits from the Dataset class.
        """
        def __init__(self, source_dir=None, npoints=12000, transform: Callable = Identity(),
                     train=True, batch_size=2, use_subcell=True, ctx_size=8000, mask_borders_with_id=None,
                     remap_dict: Optional[dict] = None):
            """
            Initializes the CloudDataSemseg class with the given parameters.
            
            Args:
                source_dir (str): The directory where the data is stored.
                npoints (int): The number of points to be sampled from the cell vertices.
                transform (Callable): The transformations to be applied in `__getitem__`.
                train (bool): If True, training data will be used. If False, validation data will be used.
                batch_size (int): The size of the batch to be used for training or validation.
                use_subcell (bool): If True, subcellular structures will be used.
                ctx_size (int): The size of the context to be used for training or validation.
                mask_borders_with_id (int): Used as label for context borders and more importantly
                    for the ultrastructure points. Ultrastructure points will be relabeled from -1 to
                    `mask_borders_with_id` after remapping cell surface points according to `remap_dict`.
                remap_dict (dict): Remap cell surface points labels from key to value according to 
                    `remap_dict`.
            """
            if source_dir is None:
                raise ValueError('"source_dir" must be given.')

            self.source_dir = source_dir
            self.fnames = np.array(glob.glob(f'{source_dir}/*.pkl'))

            print(f'Using {len(self.fnames)} cells for {"training" if train else "validation"}.')
            if use_subcell:  # TODO: add syntype
                self._num_obj_types = 4
            else:
                self._num_obj_types = 1
            self.ctx_size = ctx_size
            self.use_subcell = use_subcell
            self.train = train
            self.num_pts = npoints
            self._batch_size = batch_size
            self.transform = transform
            self.mask_borders_with_id = mask_borders_with_id
            self.remap_dict = remap_dict

        def __getitem__(self, item):
            """
            Loads a sample from the dataset.
            
            Args:
                item (int): The index of the sample to be loaded.
            
            Returns:
                dict: A dictionary containing the point array, feature array, and target labels.
            """
            item = np.random.randint(0, len(self.fnames))
            sample_pts, sample_feats, out_labels = self.load_sample(item)
            if self.remap_dict is not None:
                for k, v in self.remap_dict.items():
                    out_labels[out_labels == k] = v
            if -1 in out_labels:
                out_labels[out_labels == -1] = self.mask_borders_with_id
            pts = torch.from_numpy(sample_pts).float()
            feats = torch.from_numpy(sample_feats).float()
            lbs = torch.from_numpy(out_labels).long()
            return {'pts': pts, 'features': feats, 'target': lbs,
                    'extra': os.path.split(self.fnames[item])[1][:-4]}

        def __len__(self):
            """
            Returns the length of the dataset. If `self.train=True`, the length is the number of files times 100.
            Otherwise, it's the number of files times 10.
            
            Returns:
                int: The length of the dataset.
            """
            if self.train:
                return len(self.fnames) * 100
            else:
                return len(self.fnames) * 10

        def load_sample(self, item):
            """
            Loads a deterministic data sample.
            
            Args:
                item (int): The index of the sample in `py:attr:~fnames`.
            
            Returns:
                tuple: A tuple containing the point array, feature array, target 
                points and target labels.
            """
            p = self.fnames[item]
            sample_feats, sample_pts, out_labels = \
                pts_loader_semseg_train(p, self._batch_size, self.num_pts,
                                        transform=self.transform, ctx_size=self.ctx_size,
                                        use_subcell=self.use_subcell,
                                        mask_borders_with_id=self.mask_borders_with_id)
            return sample_pts, sample_feats, out_labels


    class MultiviewDataCached(Dataset):
        """
        This class is used to load multiview data. It reads data from h5 files and 
        stores them in memory for faster access. It also supports parallel reading 
        of data to further speed up the data loading process. Previously known as 
        Multiview data loader.
        """
        def __init__(self,
                    base_dir,
                    train=True,
                    inp_key='raw',
                    target_key='label',
                    transform: Callable = Identity(),
                    num_read_limit=5 # num_times each sample point should be used before corresponding h5py file is released
                    ):
            """
            Initializes the MultiviewDataCached object.
            
            Args:
                base_dir (str): The base directory where the data files are located.
                train (bool): If True, the object is set to training mode. If False, it is set to validation mode.
                inp_key (str): The key used to access the input data in the h5 files.
                target_key (str): The key used to access the target data in the h5 files.
                transform (Callable): A function or transform to apply to the data.
                num_read_limit (int): The number of times each sample point should be used before the corresponding h5py file is released.
            """
            super().__init__()
            #IMPORTANT while creating dataloader from this class, num_workers must be <=1
            cube_id = "train" if train else "valid"
            self.fnames_inp = sorted(glob.glob(base_dir + "/raw_{}*.h5".format(cube_id)))
            self.fnames_target = sorted(glob.glob(base_dir + "/label_{}*.h5".format(cube_id)))
            assert len(self.fnames_inp) == len(self.fnames_target)
            print('Using {} .h5 GT files for {}.'.format(
                len(self.fnames_inp), "training" if train else "validation"))
            self.inp_key = inp_key
            self.target_key = target_key
            self.transform = transform
            self.train = train

            if self.train:
                self.num_read_limit = num_read_limit
            else:
                self.num_read_limit = 1  # no need to repeat sample points in validation
            self.secondary = self.secondary_t = None
            self.read(0)
            self.primary, self.primary_t = self.secondary, self.secondary_t
            self.close_files()
            self.secondary = self.secondary_t = None

            self.num_samples_in_curr_file = self.primary.shape[0]
            self.index_array = np.array(list(range(self.num_samples_in_curr_file))*self.num_read_limit)
            np.random.shuffle(self.index_array)

            self.current_count = 0
            self.file_pointer = 1
            self.num_samples_in_already_read_files = 0
            self.thread_launched = False

        def __getitem__(self, index):
            """
            Returns the input and target data for a given index.
            
            Args:
                index (int): The index of the data to return.
            
            Returns:
                tuple: A tuple containing the input and target data.
            """
            # index = index - self.num_samples_in_already_read_files
            index = np.random.randint(0, len(self.index_array), 1)[0]
            if self.current_count > int(0.1*len(self.index_array)) and self.thread_launched == \
                    False : #adjust 0.5
                print("Launching parallel thread.")
                start = time.time()
                self.read_thread = threading.Thread(target=self.read, args=[self.file_pointer])
                self.read_thread.start()
                dt = time.time() - start
                print(f"parallel thread launched after {dt:.2f}")
                self.thread_launched = True

            if self.current_count == len(self.index_array) - 1:
                print("Joining parallel thread.")
                start = time.time()
                self.read_thread.join()
                dt = time.time() - start
                print(f"parallel thread joined after {dt:.2f}")

                temp, temp_t = self.primary[self.index_array[index]], self.primary_t[self.index_array[index]]
                self.num_samples_in_already_read_files += len(self.index_array)
                self.primary, self.primary_t = self.secondary, self.secondary_t
                self.close_files()
                self.secondary = self.secondary_t = None
                self.num_samples_in_curr_file = self.primary.shape[0]
                self.index_array = np.array(list(range(self.num_samples_in_curr_file))*self.num_read_limit)
                if self.file_pointer == 0: self.num_samples_in_already_read_files = 0
                self.file_pointer = (self.file_pointer+1)%len(self.fnames_inp)
                self.current_count = 0
                self.thread_launched = False
                return temp, np.squeeze(temp_t, axis=0)

            self.current_count += 1
            return self.transform(self.primary[self.index_array[index]], np.squeeze(
                self.primary_t[self.index_array[index]], axis=0))

        def read(self, file_pointer):
            """
            Reads data from the h5 files and stores them in memory.
            
            Args:
                file_pointer (int): The index of the file to read.
            """
            self.file_inp = h5py.File(os.path.expanduser(self.fnames_inp[file_pointer]), 'r')
            self.file_target = h5py.File(os.path.expanduser(self.fnames_target[file_pointer]), 'r')
            self.secondary = self.file_inp[self.inp_key][()]/255
            self.secondary = self.secondary.astype(np.float32)
            self.secondary_t = self.file_target[self.target_key][()].astype(np.int64)
            self.secondary, self.secondary_t = self.secondary, self.secondary_t
            print(f"read h5 file {self.fnames_inp[file_pointer]} contains {self.secondary.shape[0]} samples") #, {self.secondary_t.shape[0]} labels")

        def __len__(self):
            """
            Returns the number of data points.
            
            Returns:
                int: The number of data points.
            """
            return 10000 if self.train else 1000

        def close_files(self):
            """
            Closes the currently open h5 files.
            """
            self.file_inp.close()
            self.file_target.close()


    class MultiviewData(Dataset):
        """
        This class is used to load multiview spine data. It reads data from h5 files and stores them in memory.
        """
        def __init__(
                self,
                base_dir,
                train=True,
                inp_key='raw', target_key='label',
                transform: Callable = Identity()
        ):
            """
            Initializes the MultiviewData object.
            
            Args:
                base_dir (str): The base directory where the data files are located.
                train (bool): If True, the object is set to training mode. If False, it is set to validation mode.
                inp_key (str): The key used to access the input data in the h5 files.
                target_key (str): The key used to access the target data in the h5 files.
                transform (Callable): A function or transform to apply to the data.
            """
            super().__init__()
            if not os.path.isdir(base_dir):
                raise RuntimeError('Could not find specified base directory "{}".'.format(base_dir))
            self.train = train
            cube_id = "train" if train else "valid"
            fnames_inp = sorted(glob.glob(base_dir + "/raw_{}*.h5".format(cube_id)))
            fnames_target = sorted(glob.glob(base_dir + "/label_{}*.h5".format(cube_id)))
            print("Loading {} h5 raw files: {}".format(len(fnames_inp), fnames_inp))
            print("Loading {} h5 label files: {}".format(len(fnames_inp), fnames_target))
            assert len(fnames_inp) == len(fnames_target)
            self.inp = []
            self.target = []
            for ii in range(len(fnames_inp)):
                self.inp_file = h5py.File(os.path.expanduser(fnames_inp[ii]), 'r')
                self.target_file = h5py.File(os.path.expanduser(fnames_target[ii]), 'r')
                data = self.inp_file[inp_key][()]
                # 'normalize': division by 255
                self.inp.append(data.astype(np.float32) / 255.)
                data_t = self.target_file[target_key][()].astype(np.int64)
                self.target.append(data_t[:, 0])
                del data, data_t
                self.close_files()
            self.inp = np.concatenate(self.inp)
            self.target = np.concatenate(self.target)
            self.transform = transform
            print("Dataset ({}): {}\t{}".format(cube_id, self.inp.shape,
                                                np.unique(self.target, return_counts=True)))

        def __getitem__(self, index):
            """
            Returns the input and target data for a given index.
            
            Args:
                index (int): The index of the data to return.
            
            Returns:
                tuple: A tuple containing the input and target data.
            """
            inp = self.inp[index]
            target = self.target[index]
            inp, target = self.transform(inp, target)
            return inp, target

        def __len__(self):
            """
            Returns the number of data points.
            
            Returns:
                int: The number of data points.
            """
            if not self.train:
                return np.min([500, self.target.shape[0]])
            return np.min([2500, self.target.shape[0]])  # self.target.shape[0]  # this number determines the epoch size

        def close_files(self):
            """
            Closes the currently open h5 files.
            """
            self.inp_file.close()
            self.target_file.close()

    class AxonsViewsE3(Dataset):
        """
        This class is a wrapper for the AxonsViews data loader, providing an interface for data loading.
        """
        def __init__(
                self,
                train=True,
                transform: Callable = Identity(),
                **kwargs
        ):
            """
            Initializes the AxonsViewsE3 object.
            
            Args:
                train (bool): If True, the object is set to training mode. If False, it is set to validation mode.
                transform (Callable): A function or transform to apply to the data.
                **kwargs: Additional keyword arguments.
            """
            super().__init__()
            self.train = train
            self.transform = transform  # TODO: add gt paths to config
            self.av = AxonViews(None, None, naive_norm=False, working_dir='/wholebrain/scratch/areaxfs3/', **kwargs)

        def __getitem__(self, index):
            """
            Returns the input and target data for a given index.
            
            Args:
                index (int): The index of the data to return.
            
            Returns:
                tuple: A tuple containing the input and target data.
            """
            inp, target = self.av.getbatch(1, source='train' if self.train else 'valid')
            inp = naive_view_normalization_new(inp)
            inp, _ = self.transform(inp, None)  # Do not flip target label ^.^
            # target = np.eye(self.ctv.n_classes)[target.squeeze().astype(np.int32)]  # one-hot encoding
            return inp[0], target.squeeze().astype(np.int32)  # target should just be a scalar

        def __len__(self):
            """
            Returns the number of data points, also referred to as epoch size(s).
            
            Returns:
                int: The number of data points, indicating the epoch size(s).
            """
            if not self.train:
                return 1000
            return 5000


    class CelltypeViewsE3(Dataset):
        """
        This class is a wrapper for the CelltypeViews data loader. It loads views that need to be
        available. If `view_key` is specified, ensure they exist by running the appropriate rendering
        for every SSV in the GT, e.g. ``ssv._render_rawviews(4)`` for 4 views per location.
        """
        def __init__(
                self,
                train=True,
                transform: Callable = Identity(),
                use_syntype_scal=False,
                is_j0251=False,
                **kwargs
        ):
            """
            Initializes the CelltypeViewsE3 object.
            
            Args:
                train (bool): If True, the object is set to training mode. If False, it is set to validation mode.
                transform (Callable): A function or transform to apply to the data.
                use_syntype_scal (bool): If True, uses the synapse type scaler.
                is_j0251 (bool): If True, uses the J0251 version of the data loader.
                **kwargs: Additional keyword arguments.
            """
            super().__init__()
            self.train = train
            self.use_syntype_scal = use_syntype_scal
            self.transform = transform
            # TODO: add gt paths to config
            if not is_j0251:
                raise RuntimeError('This version is deprecated!')
                self.ctv = CelltypeViews(None, None, **kwargs)
            else:
                self.ctv = CelltypeViewsJ0251(None, None, **kwargs)

        def __getitem__(self, index):
            """
            Returns the input and target data for a given index.
            
            Args:
                index (int): The index of the data to return.
            
            Returns:
                dict: A dictionary containing the input and target data.
            """
            if self.use_syntype_scal:
                inp, target, syn_signs = self.ctv.getbatch_alternative(1, source='train' if self.train else 'valid')
                inp, _ = self.transform(inp, None)  # Do not flip target label ^.^
                # target should just be a scalar
                return {'inp': (inp[0], syn_signs[0].astype(np.float32)), 'target': target.squeeze().astype(np.int32)}
            else:
                inp, target = self.ctv.getbatch_alternative_noscal(1, source='train' if self.train else 'valid')
                inp, _ = self.transform(inp, None)  # Do not flip target label ^.^
                return {'inp': inp[0], 'target': target.squeeze().astype(np.int32)}

        def __len__(self):
            """
            Returns the number of data points, also referred to as epoch size(s).
            
            Returns:
                int: The number of data points, or epoch size(s).
            """
            if not self.train:
                if len(self.ctv.valid_d) == 0:
                    return 0
                return 2000
            return 20000

    
    class GliaViewsE3(Dataset):
        """
        This class is a wrapper for the GliaViews data loader.
        """
        def __init__(
                self,
                train=True,
                transform: Callable = Identity(),
                **kwargs
        ):
            """
            Initializes the GliaViewsE3 object.
            
            Args:
                train (bool): If True, the object is set to training mode. If False, it is set to validation mode.
                transform (Callable): A function or transform to apply to the data.
                **kwargs: Additional keyword arguments.
            """
            super().__init__()
            self.train = train
            self.transform = transform  # TODO: add gt paths to config
            self.gv = GliaViews(None, None, naive_norm=False,
                                av_working_dir='/wholebrain/scratch/areaxfs3/', **kwargs)

        def __getitem__(self, index):
            """
            Returns the input and target data for a given index.
            
            Args:
                index (int): The index of the data to return.
            
            Returns:
                tuple: A tuple containing the input and target data.
            """
            inp, target = self.gv.getbatch(1, source='train' if self.train else 'valid')
            inp = naive_view_normalization_new(inp)
            inp, _ = self.transform(inp, None)  # Do not flip target label ^.^
            # target = np.eye(self.ctv.n_classes)[target.squeeze().astype(np.int32)]  # one-hot encoding
            return inp[0], target.squeeze().astype(np.int32)  # target should just be a scalar

        def __len__(self):
            """
            Returns the number of data points, effectively determining the epoch size(s).
            
            Returns:
                int: The number of data points, which corresponds to the epoch size(s).
            """
            if not self.train:
                return 1000
            return 5000


    class MultiviewData_TNet_online(Dataset):
        """
        This class is a data loader for the Multiview Triplet Network. It loads the data, applies 
        transformations, and prepares it for training or validation. It also handles caching to 
        improve performance.
        """

        def __init__(
                self, working_dir='/wholebrain/scratch/areaxfs3/',
                train=True, epoch_size=40000, allow_close_neigh=0,
                transform: Callable = Identity(), allow_axonview_gt=True,
                ctv_kwargs=None,
        ):
            """
            Initializes the MultiviewData_TNet_online object.
            
            Args:
                working_dir (str): The directory where the data is stored.
                train (bool): If True, the data loader is in training mode. If False, it's in validation mode.
                epoch_size (int): The number of samples per epoch.
                allow_close_neigh (int): The number of close neighbors allowed.
                transform (Callable): The transformation function to apply to the data.
                allow_axonview_gt (bool): If True, allows the use of AxonView ground truth data.
                ctv_kwargs (dict): Additional keyword arguments for the CelltypeViews object.
            """
            if ctv_kwargs is None:
                ctv_kwargs = {}
            super().__init__()
            self.transform = transform
            self.epoch_size = epoch_size
            self.train = train
            if 2 > allow_close_neigh > 0:
                raise ValueError('allow_close_neigh must be at least 2.')
            self.allow_close_neigh = allow_close_neigh
            self.inp_locs = None
            # load GliaView Data and store all views in memory
            # inefficient because GV has to be loaded twice (train and valid)
            print("Loaded all data. Concatenating now.")
            # use these classes to load label and splitting dicts, CURRENTLY AxonGT is not supported anymore
            AV = AxonViews(None, None, raw_only=False, nb_views=2,
                           naive_norm=False, load_data=False,
                           working_dir='/wholebrain/scratch/areaxfs3/')
            if not allow_axonview_gt:  # set repsective data set to empty lists
                AV.splitting_dict["train"] = []
                AV.splitting_dict["valid"] = []
                AV.splitting_dict["test"] = []

            CTV = CelltypeViews(None, None, load_data=False, **ctv_kwargs)
            self.view_key = CTV.view_key
            # now link actual data
            self.ssd = CTV.ssd  # SuperSegmentationDataset(global_params.config.working_dir, version='tnetgt')

            if train:  # use all available data!
                self.inp = [self.ssd.get_super_segmentation_object(ix) for ix in AV.splitting_dict["train"]] + \
                           [self.ssd.get_super_segmentation_object(ix) for ix in CTV.splitting_dict["train"]] + \
                            [self.ssd.get_super_segmentation_object(ix) for ix in AV.splitting_dict["valid"]] + \
                            [self.ssd.get_super_segmentation_object(ix) for ix in CTV.splitting_dict["valid"]]
                self._inp_ssv_ids = self.inp.copy()
                if self.allow_close_neigh:
                    self.inp_locs = []
                    for ssv in self.inp:
                        ssv.load_attr_dict()
                        self.inp_locs.append(np.concatenate(ssv.sample_locations(verbose=True)))
            else:  # valid
                self.inp = [self.ssd.get_super_segmentation_object(ix) for ix in AV.splitting_dict["valid"]] + \
                           [self.ssd.get_super_segmentation_object(ix) for ix in CTV.splitting_dict["valid"]]
                self._inp_ssv_ids = self.inp.copy()
                if self.allow_close_neigh:
                    self.inp_locs = []
                    for ssv in self.inp:
                        ssv.load_attr_dict()
                        self.inp_locs.append(np.concatenate(ssv.sample_locations(verbose=True)))
                self.inp = [ssv.load_views(view_key=self.view_key) for ssv in self.inp]
            ixs = np.arange(len(self.inp))
            np.random.shuffle(ixs)
            self.inp = np.array(self.inp)[ixs]
            if self.allow_close_neigh:
                self.inp_locs = np.array(self.inp_locs)[ixs]
            print("Dataset: {}\t{}".format("train" if train else "valid",
                                           self.inp.shape))
            self._cache_use = 0
            self._cache_use_dist = 0
            self._max_cache_usages = 200
            self._max_cache_usages_dist = 200
            self._cache = None
            self._cache_dist = None
            self._cached_ssv_ix = None

        def __getitem__(self, index):
            """
            Retrieves a sample from the dataset at the specified index.
            
            Args:
                index (int): The index of the sample to retrieve.
            
            Returns:
                np.array: The sample data.
            """
            start = time.time()
            summary = ""
            if self._cache is None or self._cache_use > self._max_cache_usages:
                # use random seed locally; overwrite index -> always draw randomly
                with temp_seed(None):
                    index = np.random.randint(0, len(self.inp))
                self._cache_use = 0
                # similar pair views
                if self.train:
                    ssv = self.inp[index]
                    ssv.disable_locking = True
                    views = ssv.load_views(view_key=self.view_key)
                else:
                    views = self.inp[index]
                # 50% more because of augmentations
                self._max_cache_usages = np.max([200, int(len(views) * 1.5)])
                # get random different SSV
                self._cache = views
                self._cached_ssv_ix = index
                if self.allow_close_neigh:
                    self._cached_loc_tree = spatial.cKDTree(self.inp_locs[index])
            else:
                views = self._cache
                self._cache_use += 1
            dtime_similar = time.time() - start
            if dtime_similar > 10:
                summary += "Found similar views after {:.1}s.".format(dtime_similar)
            start = time.time()
            if self._cache_dist is None or self._cache_use_dist > self._max_cache_usages_dist:
                while True:
                    # use random seed locally; overwrite index -> always draw randomly
                    with temp_seed(None):
                        dist_ix = np.random.randint(0, len(self.inp))
                    if dist_ix != self._cached_ssv_ix:
                        break
                if self.train:
                    ssv = self.inp[dist_ix]
                    ssv.disable_locking = True
                    views_dist = ssv.load_views(view_key=self.view_key)
                else:
                    views_dist = self.inp[dist_ix]
                self._cache_dist = views_dist
                # fact 2 because only drawing 1 view, and additional 50% because of augmentations
                self._max_cache_usages_dist = np.max([200, len(views_dist) * 3])
            else:
                views_dist = self._cache_dist
                self._cache_use_dist += 1
            dtime_distant = time.time() - start
            if dtime_distant > 10 or dtime_similar > 10:
                summary += "Found distant views after {:.1}s.".format(dtime_distant)
            start = time.time()
            # similar views are from same, randomly picked location
            mview_ix = np.random.randint(0, len(views))
            views_sim = views[mview_ix]
            views_sim = self.transform(views_sim, target=None)[0]
            views_sim = views_sim.swapaxes(1, 0)
            if len(views_sim) > 2:
                view_ixs = np.arange(len(views_sim))
                views_sim = views_sim[view_ixs][:2]
            if self.allow_close_neigh and np.random.rand(1)[0] > 0.25:  # only use neighbors as similar views with 0.25 chance
                dists, close_neigh_ixs = self._cached_loc_tree.query(self.inp_locs[self._cached_ssv_ix][mview_ix], k=self.allow_close_neigh)  # itself and two others
                neigh_ix = close_neigh_ixs[np.random.randint(1, self.allow_close_neigh)]  # only use neighbors not itself
                views_sim[1] = views[neigh_ix, :, np.random.randint(0, 2)]  # chose any of the two views
            # single unsimilar view is from different SSV and randomly picked location
            mview_ix = np.random.randint(0, len(views_dist))
            # choose random view locations and random view (out of the two) and add the two axes back to the shape
            view_dist = views_dist[mview_ix][None, :, np.random.randint(0, 2)]
            view_dist = self.transform(view_dist, target=None)[0]
            dtime_transf = time.time() - start
            if dtime_distant > 10 or dtime_similar > 10 or dtime_transf > 10:
                summary += "Transformation finished after {:.1}s.".format(dtime_transf)
                print(summary)
            return naive_view_normalization_new(np.concatenate([views_sim, view_dist]))

        def __len__(self):
            """
            Returns the length of the dataset.
            
            Returns:
                int: The length of the dataset.
            """
            if self.train:
                return 5000
            else:
                return 20

        def close_files(self):
            """
            Closes all open files associated with this dataset.
            """
            return


# -------------------------------------- ELEKTRONN2 ----------------------------
class Data(object):
    """
    This class is a base object for loading and preparing data. It is copied 
    from ELEKTRONN2 due to import issues. Note: Refactoring and removal of 
    dependency on this class is pending.
    """
    def __init__(self, n_lab=None):
        """
        Initializes the Data object.
        
        Args:
            n_lab (int, optional): The number of labels in the dataset.
        """
        self._pos           = 0
        # self.train_d = None
        # self.train_l = None
        # self.valid_d = None
        # self.valid_l = None
        # self.test_d = None
        # self.test_l = None

        if isinstance(self.train_d, np.ndarray):
            self._training_count = self.train_d.shape[0]
            if n_lab is None:
                self.n_lab = np.unique(self.train_l).size
            else:
                self.n_lab = n_lab
        elif isinstance(self.train_d, list):
            self._training_count = len(self.train_d)
            if n_lab is None:
                unique = [np.unique(l) for l in self.train_l]
                self.n_lab = np.unique(np.hstack(unique)).size
            else:
                self.n_lab = n_lab

        if self.example_shape is None:
            self.example_shape = self.train_d[0].shape
        self.n_ch = self.example_shape[0]

        self.rng = np.random.RandomState(np.uint32((time.time()*0.0001 - int(time.time()*0.0001))*4294967295))
        self.pid = os.getpid()
        log_cnn.info(self.__repr__())
        self._perm = self.rng.permutation(self._training_count)

    def _reseed(self):
        """
        Reseeds the random number generator if the process ID has changed.
        """
        current_pid = os.getpid()
        if current_pid!=self.pid:
            self.pid = current_pid
            self.rng.seed(np.uint32((time.time()*0.0001 - int(time.time()*0.0001))*4294967295+self.pid))
            log_cnn.debug("Reseeding RNG in Process with PID: {}".format(self.pid))

    def __repr__(self):
        """
        Returns a string representation of the Data object.
        
        Returns:
            str: A string representation of the Data object.
        """
        return "%i-class Data Set: #training examples: %i and #validation: %i" \
        %(self.n_lab, self._training_count, len(self.valid_d))

    def getbatch(self, batch_size, source='train'):
        """
        Retrieves a batch of samples from the specified source.
        
        Args:
            batch_size (int): The number of samples in the batch.
            source (str): The source of the data ('train', 'valid', or 'test').
        
        Returns:
            tuple: A tuple containing the data and labels for the batch.
        """
        if source=='train':
            if (self._pos+batch_size) < self._training_count:
                self._pos += batch_size
                slice = self._perm[self._pos-batch_size:self._pos]
            else: # get new permutation
                self._perm = self.rng.permutation(self._training_count)
                self._pos = 0
                slice = self._perm[:batch_size]

            if isinstance(self.train_d, np.ndarray):
                return (self.train_d[slice], self.train_l[slice])

            elif isinstance(self.train_d, list):
                data  = np.array([self.train_d[i] for i in slice])
                label = np.array([self.train_l[i] for i in slice])
                return (data, label)

        elif source=='valid':
            data  = self.valid_d[:batch_size]
            label = self.valid_l[:batch_size]
            return (data, label)

        elif source=='test':
            data  = self.test_d[:batch_size]
            label = self.test_l[:batch_size]
            return (data, label)

    def createCVSplit(self, data, label, n_folds=3, use_fold=2, shuffle=False, random_state=None):
        """
        Creates a cross-validation split of the data.
        
        Args:
            data (np.array): The data to split.
            label (np.array): The labels for the data.
            n_folds (int): The number of folds for the cross-validation split.
            use_fold (int): The fold to use for validation.
            shuffle (bool): If True, shuffles the data before splitting.
            random_state (int): The seed for the random number generator.
        """
        try:  # sklearn >=0.18 API
            # (see http://scikit-learn.org/dev/whats_new.html#model-selection-enhancements-and-api-changes)
            import sklearn.model_selection
            kfold = sklearn.model_selection.KFold(
                n_splits=n_folds, shuffle=shuffle, random_state=random_state
            )
            cv = kfold.split(data)
        except:  # sklearn <0.18 API # TODO: We can remove this after a while.
            import sklearn.cross_validation
            cv = sklearn.cross_validation.KFold(
                len(data), n_folds, shuffle=shuffle, random_state=random_state
            )
        for fold, (train_i, valid_i) in enumerate(cv):
            if fold==use_fold:
                self.valid_d = data[valid_i]
                self.valid_l = label[valid_i]
                self.train_d = data[train_i]
                self.train_l = label[train_i]


class MultiViewData(Data):
    def __init__(self, working_dir, gt_type, nb_cpus=20,
                 label_dict=None, view_kwargs=None, naive_norm=True,
                 load_data=True, train_fraction=None, random_seed=0,
                 splitting_dict=None):
        """
        Initializes the MultiViewData class which is a subclass of Data. It is used to handle
        multi-view data for training, validation and testing.
        
        Args:
            working_dir (str): The directory where the data is stored.
            gt_type (str): The type of ground truth data.
            nb_cpus (int, optional): The number of CPUs to use. Default is 20.
            label_dict (dict, optional): A dictionary mapping labels to their respective data.
            view_kwargs (dict, optional): A dictionary of arguments for the view.
            naive_norm (bool, optional): If True, applies naive normalization to the data. Default is True.
            load_data (bool, optional): If True, loads the data. Default is True.
            train_fraction (float, optional): The fraction of data to be used for training.
            random_seed (int, optional): The seed for the random number generator. Default is 0.
            splitting_dict (dict, optional): A dictionary defining the data splitting.
        """
        self.splitting_dict = splitting_dict
        if view_kwargs is None:
            view_kwargs = dict(raw_only=False,
                               nb_cpus=nb_cpus, ignore_missing=True,
                               force_reload=False)
        self.gt_dir = working_dir + "/ssv_%s/" % gt_type
        splitting_dc_path = self.gt_dir + "%s_splitting.pkl" % gt_type
        label_dc_path = self.gt_dir + "%s_labels.pkl" % gt_type
        if label_dict is None:
            self.label_dict = load_pkl2obj(label_dc_path)
        if (not os.path.isfile(splitting_dc_path) and self.splitting_dict is None) or train_fraction is \
                not None:
            if train_fraction is None:
                msg = f'Did not find splitting dictionary at {splitting_dc_path} ' \
                      f'and train data fraction was not defined.'
                log_cnn.error(msg)
                raise ValueError(msg)
            # TODO: Use  stratified splitting for per-class balanced splitting
            ssv_ids = np.array(list(self.label_dict.keys()), dtype=np.uint64)
            ssv_labels = np.array(list(self.label_dict.values()), dtype=np.uint64)
            avail_classes, c_count = np.unique(ssv_labels, return_counts=True)
            n_classes = len(avail_classes)
            for c, cnt in zip(avail_classes, c_count):
                if cnt < 2:
                    log_cnn.warn('Class {} has support of {}. Using same SSV multiple times to '
                                 'satisfy "train_test_split" condition.'.format(c, cnt))
                    curr_c_ssvs = ssv_ids[ssv_labels == c][:1]
                    ssv_ids = np.concatenate([ssv_ids, curr_c_ssvs])
                    ssv_labels = np.concatenate([ssv_labels, [c]])
            if train_fraction == 1:
                self.splitting_dict = {"train": ssv_ids, "valid": [],
                                       "test": []}
            else:
                if int(train_fraction) * len(ssv_ids) < n_classes:
                    train_fraction = 1. - float(n_classes + 1) / len(ssv_ids)
                    print("Train data fraction was set to {} due to splitting restrictions "
                          "(at least one sample per class in validation set).".format(train_fraction))
                train_ids, valid_ids = train_test_split(ssv_ids, shuffle=True, random_state=random_seed, stratify=ssv_labels,
                                                        train_size=train_fraction)
                self.splitting_dict = {"train": train_ids, "valid": valid_ids,
                                       "test": []}
                print('Validation set: {}\t{}'.format(self.splitting_dict['valid'],
                                                  [self.label_dict[ix] for ix in self.splitting_dict['valid']]))
        else:
            if train_fraction is not None:
                raise ValueError('Value fraction can only be set if splitting dict is not available.')
            if self.splitting_dict is None:
                self.splitting_dict = load_pkl2obj(splitting_dc_path)

        self.ssd = SuperSegmentationDataset(working_dir, version=gt_type)
        if not load_data:
            self.test_d = np.zeros((1, 1))
            self.test_l = np.zeros((1, 1))
            self.train_d = np.zeros((1, 1))
            self.train_l = np.zeros((1, 1))
            self.valid_d = np.zeros((1, 1))
            self.valid_l = np.zeros((1, 1))
            self.example_shape = self.train_d[:1].shape
            super(MultiViewData, self).__init__()
            return
        for k, v in self.splitting_dict.items():
            classes, c_cnts = np.unique([self.label_dict[ix] for ix in
                                         self.splitting_dict[k]], return_counts=True)
            print(f"{k} [labels, counts]: {classes}, {c_cnts}")
        # train data
        # get views of each SV
        self.train_d = [self.ssd.get_super_segmentation_object(ix).load_views(**view_kwargs)
            for ix in self.splitting_dict["train"]]
        # create labels for SV views according to SSV label, assuming pure SSV compartments
        self.train_l = np.concatenate([[self.label_dict[ix]] * len(self.train_d[ii])
        for ii, ix in enumerate(self.splitting_dict["train"])]).astype(np.uint16)[:, None]
        # concatenate raw data
        self.train_d = np.concatenate(self.train_d)
        if naive_norm:
            self.train_d = naive_view_normalization(self.train_d)
        # set example shape for parent class 'Data'
        self.example_shape = self.train_d[0].shape
        # valid data
        self.valid_d = [self.ssd.get_super_segmentation_object(ix).load_views(**view_kwargs)
            for ix in self.splitting_dict["valid"]]
        self.valid_l = np.concatenate([[self.label_dict[ix]] * len(self.valid_d[ii])
        for ii, ix in enumerate(self.splitting_dict["valid"])]).astype(np.uint16)[:, None]
        self.valid_d = np.concatenate(self.valid_d)
        if naive_norm:
            self.valid_d = naive_view_normalization(self.valid_d)
        # test data
        if len(self.splitting_dict["test"]) > 0:
            self.test_d = [self.ssd.get_super_segmentation_object(ix).load_views(**view_kwargs)
                for ix in self.splitting_dict["test"]]
            self.test_l = np.concatenate(
                [[self.label_dict[ix]] * len(self.test_d[ii])
                 for ii, ix in enumerate(self.splitting_dict["test"])]).astype(
                np.uint16)[:, None]
            self.test_d = np.concatenate(self.test_d)
            if naive_norm:
                self.test_d = naive_view_normalization(self.test_d)
        else:
            self.test_d = np.zeros_like(self.valid_d)[:1]
            self.test_l = np.zeros_like(self.valid_l)[:1]
        print("GT splitting:", self.splitting_dict)
        print("\nlabels (train) - {}".format(np.unique(self.train_l,
                                                       return_counts=True)))
        print("\nlabels (valid) - {}".format(np.unique(self.valid_l,
                                                       return_counts=True)))
        super(MultiViewData, self).__init__()


class AxonViews(MultiViewData):
    def __init__(self, inp_node, out_node, gt_type="axgt", working_dir=None,
                 nb_views=2, reduce_context=0, channels_to_load=(0, 1, 2, 3),
                 reduce_context_fact=1, binary_views=False, raw_only=False,
                 nb_cpus=20, naive_norm=True, **kwargs):
        """
        Initializes the AxonViews class which is a subclass of MultiViewData. It is used to handle
        axon view data for training, validation and testing.
        
        Args:
            inp_node (str): The input node for the data.
            out_node (str): The output node for the data.
            gt_type (str, optional): The type of ground truth data. Default is "axgt".
            working_dir (str, optional): The directory where the data is stored.
            nb_views (int, optional): The number of views. Default is 2.
            reduce_context (int, optional): The context to be reduced. Default is 0.
            channels_to_load (tuple, optional): The channels to load. Default is (0, 1, 2, 3).
            reduce_context_fact (int, optional): The factor by which to reduce the context. Default is 1.
            binary_views (bool, optional): If True, converts the views to binary. Default is False.
            raw_only (bool, optional): If True, only raw data is used. Default is False.
            nb_cpus (int, optional): The number of CPUs to use. Default is 20.
            naive_norm (bool, optional): If True, applies naive normalization to the data. Default is True.
            kwargs (dict): Additional keyword arguments.
        """
        if working_dir is None:
            working_dir = global_params.config.working_dir
        super(AxonViews, self).__init__(working_dir, gt_type,
                                        nb_cpus=nb_cpus, naive_norm=naive_norm, **kwargs)
        print("Initialized AxonViews:", self.__repr__())
        self.nb_views = nb_views
        self.reduce_context = reduce_context
        self.reduce_context_fact = reduce_context_fact
        self.channels_to_load = channels_to_load
        self.binary_views = binary_views
        self.raw_only = raw_only
        if self.raw_only and self.train_d.shape[1] > 1:
            self.train_d = self.train_d[:, :1]
            self.valid_d = self.valid_d[:, :1]
            if len(self.test_d) > 0:
                self.test_d = self.test_d[:, :1]
        self.example_shape = self.train_d[0].shape

    def getbatch(self, batch_size, source='train'):
        """
        Retrieves a batch of data.
        
        Args:
            batch_size (int): The size of the batch to retrieve.
            source (str, optional): The source of the data. Can be 'train', 'valid' or 'test'. Default is 'train'.
        
        Returns:
            tuple: A tuple containing the data and labels for the batch.
        """
        # TODO: keep in mind that super().get_batch does not shuffle for validation data -> btach_size for validation should be sufficiently big
        # if source == 'valid':
        #     nb = len(self.valid_l)
        #     ixs = np.arange(nb)
        #     np.random.shuffle(ixs)
        #     self.valid_d = self.valid_d[ixs]
        #     self.valid_l = self.valid_l[ixs]
        d, l = super(AxonViews, self).getbatch(batch_size, source)
        view_shuffle = np.arange(0, d.shape[2])
        np.random.shuffle(view_shuffle)
        if self.reduce_context > 0:
            d = d[:, :, :, (self.reduce_context/2):(-self.reduce_context/2),
                self.reduce_context:-self.reduce_context]
        if self.reduce_context_fact > 1:
            d = d[:, :, :, ::self.reduce_context_fact, ::self.reduce_context_fact]
        d = d[:, :, view_shuffle[:self.nb_views]]
        if self.binary_views:
            d[d < 1.0] = 0
        return tuple([d, l])


class CelltypeViews(MultiViewData):
    """
    This class is a subclass of MultiViewData and is used to handle data related to cell types. It 
    includes methods for initializing the class and getting batches of data. It uses naive view 
    normalization (i.e., `/ 255. - 0.5`).
    """
    def __init__(self, inp_node, out_node, raw_only=False, nb_views=20, nb_views_renderinglocations=2,
                 reduce_context=0, binary_views=False, reduce_context_fact=1, n_classes=4,
                 class_weights=(2, 2, 1, 1), load_data=False, nb_cpus=1, ctgt_key="ctgt",
                 train_fraction=None, random_seed=0, view_key=None, splitting_dict=None):
        """
        Initializes the CelltypeViews class with the given parameters. It sets up the working 
        directory, checks for the existence of the directory, sets the view key, number of views, 
        number of CPUs, raw_only flag, reduce_context value, maximum number of cache uses, current 
        cache uses, number of classes, class weights, view cache, label cache, syn_sign cache, 
        sample weights, reduce_context_fact, binary_views, and example shape. It also initializes 
        the superclass with the working directory, ctgt_key, train_fraction, naive_norm, and 
        load_data parameters.
        
        Args:
            inp_node : Input node for the data.
            out_node : Output node for the data.
            raw_only (bool) : If True, only raw data is used.
            nb_views (int) : Number of sampled views used for prediction of cell type.
            nb_views_renderinglocations (int) : Number of views per rendering location.
            reduce_context (int) : Context to be reduced.
            binary_views (bool) : If True, views are binary.
            reduce_context_fact (int) : Factor by which context is reduced.
            n_classes (int) : Number of classes in the data.
            class_weights (tuple) : Weights for each class.
            load_data (bool) : If True, data is loaded.
            nb_cpus (int) : Number of CPUs to be used.
            ctgt_key (str) : Key for the ctgt data.
            train_fraction : Fraction of data to be used for training.
            random_seed (int) : Seed for random number generation.
            view_key (str) : Key for the view data.
            splitting_dict : Dictionary for splitting the data.
        """
        global_params.wd = "/wholebrain/songbird/j0126/areaxfs_v6/"
        assert "areaxfs_v6" in global_params.config.working_dir
        assert os.path.isdir(global_params.config.working_dir)
        if view_key is None:
            self.view_key = "raw{}".format(nb_views_renderinglocations)
        else:
            self.view_key = view_key
        self.nb_views = nb_views
        self.nb_cpus = nb_cpus
        self.raw_only = raw_only
        self.reduce_context = reduce_context
        # RandomFlip augmentation etc.
        self.max_nb_cache_uses = 4000 * 2
        self.current_cache_uses = 0
        assert n_classes == len(class_weights)
        self.n_classes = n_classes
        self.class_weights = np.array(class_weights)
        self.view_cache = {'train': None, 'valid': None, 'test': None}
        self.label_cache = {'train': None, 'valid': None, 'test': None}
        self.syn_sign_cache = {'train': None, 'valid': None, 'test': None}
        self.sample_weights = {'train': None, 'valid': None, 'test': None}
        self.reduce_context_fact = reduce_context_fact
        self.binary_views = binary_views
        self.example_shape = (nb_views, 4, 2, 128, 256)
        print("Initializing CelltypeViews:", self.__dict__)  # TODO: add gt paths to config
        super().__init__(global_params.config.working_dir, ctgt_key, train_fraction=train_fraction,
                         naive_norm=False, load_data=load_data, random_seed=random_seed)
        if splitting_dict is None:
            ssv_splits = self.splitting_dict
        else:
            prev_splits = [(k, np.unique(v, return_counts=False)) for
                           k, v in self.splitting_dict.items()]
            now_splits = [(k, np.unique(v, return_counts=False)) for
                          k, v in splitting_dict.items()]
            log_cnn.critical('Splitting dict was passed explicitly. Overwriting '
                             'default splitting of super-class. Support '
                             f'previous: {prev_splits}'
                             f'Support now: {now_splits}.')
            ssv_splits = splitting_dict
            self.splitting_dict = splitting_dict
        self.train_d = np.array(ssv_splits["train"])
        self.valid_d = np.array(ssv_splits["valid"])
        self.test_d = np.array(ssv_splits["test"])
        ssv_gt_dict = self.label_dict
        self.train_l = np.array([ssv_gt_dict[ix] for ix in self.train_d], np.int16)[:, None]
        self.test_l = np.array([ssv_gt_dict[ix] for ix in self.test_d], np.int16)[:, None]
        self.valid_l = np.array([ssv_gt_dict[ix] for ix in self.valid_d], np.int16)[:, None]
        self.train_d = self.train_d[:, None]
        self.valid_d = self.valid_d[:, None]
        self.test_d = self.test_d[:, None]
        super(MultiViewData, self).__init__()
        for k, v in self.splitting_dict.items():
            classes, c_cnts = np.unique([self.label_dict[ix] for ix in
                                         self.splitting_dict[k]], return_counts=True)
            print(f"{k} [labels, counts]: {classes}, {c_cnts}")

    def getbatch_alternative(self, batch_size, source='train'):
        """
        This method is used to get a batch of data. Preliminary tests showed inferior performance 
        of models trained with sampling batches with this method compared to "getbatch" below. 
        This might be due to less stochasticity (bigger cache).
        
        Args:
            batch_size (int) : Size of the batch to be fetched.
            source (str) : Source of the data ('train', 'valid', or 'test').
        
        Returns:
            tuple: A tuple containing the data, labels, and syn_signs.
        """
        self._reseed()
        if source == 'valid':
            nb = len(self.valid_l)
            ixs = np.arange(nb)
            np.random.shuffle(ixs)
            self.valid_d = self.valid_d[ixs]
            self.valid_l = self.valid_l[ixs]
        # NOTE: also performs 'naive_view_normalization'
        if self.view_cache[source] is None or self.current_cache_uses == self.max_nb_cache_uses:
            sample_fac = np.max([int(self.nb_views / 20), 1])  # draw more ssv if number of views
            # is high
            nb_ssv = self.n_classes * sample_fac
            sample_ixs = []
            l = []
            labels2draw = np.arange(self.n_classes)
            np.random.shuffle(labels2draw)  # change order
            for cnt, i in enumerate(labels2draw):
                curr_nb_samples = max(nb_ssv // self.n_classes * self.class_weights[i], 1)
                if source == 'valid' and cnt > 2:
                    break
                try:
                    if source == "train":
                        sample_ixs.append(np.random.choice(self.train_d[self.train_l == i],
                                                           curr_nb_samples, replace=True).tolist())
                        l += [[i] * curr_nb_samples]
                    elif source == "valid":
                        sample_ixs.append(np.random.choice(self.valid_d[self.valid_l == i],
                                                           curr_nb_samples, replace=True).tolist())
                        l += [[i] * curr_nb_samples]
                    elif source == "test":
                        sample_ixs.append(np.random.choice(self.test_d[self.test_l == i],
                                                           curr_nb_samples, replace=True).tolist())
                        l += [[i] * curr_nb_samples]
                    else:
                        raise NotImplementedError
                except ValueError:  # requested class does not exist in current dataset
                    pass
            ssos = []
            sample_ixs = np.concatenate(sample_ixs)
            l = np.concatenate(l)
            for ix in sample_ixs:
                sso = self.ssd.get_super_segmentation_object(ix)
                sso.nb_cpus = self.nb_cpus
                ssos.append(sso)
            self.view_cache[source] = [sso.load_views(view_key=self.view_key) for sso in ssos]
            # pre- and postsynapse type ratios
            start = time.time()
            self.syn_sign_cache[source] = np.array(
                [[syn_sign_ratio_celltype(sso), syn_sign_ratio_celltype(sso, comp_types=[0, ])]
                 for sso in ssos])
            for ii in range(len(self.view_cache[source])):
                views = self.view_cache[source][ii]
                views = naive_view_normalization_new(views)
                views = views.swapaxes(1, 0).reshape((4, -1, 128, 256))
                self.view_cache[source][ii] = views
            self.label_cache[source] = l
            # draw big cache batch from current SSOs from which training batches are drawn
            self.current_cache_uses = 0
            self.sample_weights[source] = compute_class_weight('balanced',
                                                               np.unique(self.label_cache[source]),
                                                               self.label_cache[source])
        ixs = np.random.choice(len(self.view_cache[source]), batch_size, replace=True)
        d, l, syn_signs = transform_celltype_data_views_alternative(
            [self.view_cache[source][ix] for ix in ixs], [self.label_cache[source][ix] for ix in ixs],
            [self.syn_sign_cache[source][ix] for ix in ixs], batch_size, self.nb_views)
        if self.reduce_context > 0:
            d = d[:, :, :, (self.reduce_context/2):(-self.reduce_context/2),
                self.reduce_context:-self.reduce_context]
        if self.reduce_context_fact > 1:
            d = d[:, :, :, ::self.reduce_context_fact, ::self.reduce_context_fact]
        if self.binary_views:
            d[d < 1.0] = 0
        self.current_cache_uses += batch_size
        if self.raw_only:
            return d[:, :1], l
        return tuple([d, l, syn_signs])

    def getbatch_alternative_noscal(self, batch_size, source='train'):
        """
        This method is used to get a batch of data without scaling. Preliminary tests showed 
        inferior performance of models trained with sampling batches with this method compared 
        to "getbatch" below. This might be due to less stochasticity (bigger cache).
        
        Args:
            batch_size (int) : Size of the batch to be fetched.
            source (str) : Source of the data ('train', 'valid', or 'test').
        
        Returns:
            tuple: A tuple containing the data and labels.
        """
        self._reseed()
        if source == 'valid':
            nb = len(self.valid_l)
            ixs = np.arange(nb)
            np.random.shuffle(ixs)
            self.valid_d = self.valid_d[ixs]
            self.valid_l = self.valid_l[ixs]
        # NOTE: also performs 'naive_view_normalization'
        if self.view_cache[source] is None or self.current_cache_uses == self.max_nb_cache_uses:
            sample_fac = np.max([int(self.nb_views / 10), 2])  # draw more ssv if number of views
            # is high
            nb_ssv = self.n_classes * sample_fac
            sample_ixs = []
            l = []
            labels2draw = np.arange(self.n_classes)
            np.random.shuffle(labels2draw)  # change order
            for i in labels2draw:
                curr_nb_samples = max(nb_ssv // self.n_classes * self.class_weights[i], 1)
                try:
                    if source == "train":
                        sample_ixs.append(np.random.choice(self.train_d[self.train_l == i],
                                                           curr_nb_samples, replace=True).tolist())
                        l += [[i] * curr_nb_samples]
                    elif source == "valid":
                        sample_ixs.append(np.random.choice(self.valid_d[self.valid_l == i],
                                                           curr_nb_samples, replace=True).tolist())
                        l += [[i] * curr_nb_samples]
                    elif source == "test":
                        sample_ixs.append(np.random.choice(self.test_d[self.test_l == i],
                                                           curr_nb_samples, replace=True).tolist())
                        l += [[i] * curr_nb_samples]
                    else:
                        raise NotImplementedError
                except ValueError:  # requested class does not exist in current dataset
                    pass
            ssos = []
            sample_ixs = np.concatenate(sample_ixs)
            l = np.concatenate(l)
            for ix in sample_ixs:
                sso = self.ssd.get_super_segmentation_object(ix)
                sso.nb_cpus = self.nb_cpus
                ssos.append(sso)
            self.view_cache[source] = [sso.load_views(view_key=self.view_key) for sso in ssos]
            for ii in range(len(self.view_cache[source])):
                views = self.view_cache[source][ii]
                views = naive_view_normalization_new(views)
                views = views.swapaxes(1, 0).reshape((4, -1, 128, 256))
                self.view_cache[source][ii] = views
            self.label_cache[source] = l
            print(f'Cache support: {np.unique(l, return_counts=True)}')
            # draw big cache batch from current SSOs from which training batches are drawn
            self.current_cache_uses = 0
            self.sample_weights[source] = compute_class_weight('balanced',
                                                               np.unique(self.label_cache[source]),
                                                               self.label_cache[source])
        ixs = np.random.choice(len(self.view_cache[source]), batch_size, replace=False)
        d, l = transform_celltype_data_views_alternative_noscal(
            [self.view_cache[source][ix] for ix in ixs], [self.label_cache[source][ix] for ix in ixs],
            batch_size, self.nb_views)
        if self.reduce_context > 0:
            d = d[:, :, :, (self.reduce_context/2):(-self.reduce_context/2),
                self.reduce_context:-self.reduce_context]
        if self.reduce_context_fact > 1:
            d = d[:, :, :, ::self.reduce_context_fact, ::self.reduce_context_fact]
        if self.binary_views:
            d[d < 1.0] = 0
        self.current_cache_uses += batch_size
        if self.raw_only:
            return d[:, :1], l
        return tuple([d, l])

    def getbatch(self, batch_size, source='train'):
        self._reseed()
        if source == 'valid':
            nb = len(self.valid_l)
            ixs = np.arange(nb)
            np.random.shuffle(ixs)
            self.valid_d = self.valid_d[ixs]
            self.valid_l = self.valid_l[ixs]
        # NOTE: also performs 'naive_view_normalization'
        if self.view_cache[source] is None or self.current_cache_uses == self.max_nb_cache_uses:
            sample_fac = np.max([int(self.nb_views / 20), 1])  # draw more ssv if number of views is high
            nb_ssv = self.n_classes * sample_fac  # 1 for each class

            sample_ixs = []
            l = []
            labels2draw = np.arange(self.n_classes)
            class_sample_weight = self.class_weights
            np.random.shuffle(labels2draw)  # change order
            for i in labels2draw:
                curr_nb_samples = nb_ssv // self.n_classes * class_sample_weight[i]
                try:
                    if source == "train":
                        sample_ixs.append(np.random.choice(self.train_d[self.train_l == i],
                                                           curr_nb_samples, replace=True).tolist())
                        l += [[i] * curr_nb_samples]
                    elif source == "valid":
                        sample_ixs.append(np.random.choice(self.valid_d[self.valid_l == i],
                                                           curr_nb_samples, replace=True).tolist())
                        l += [[i] * curr_nb_samples]
                    elif source == "test":
                        sample_ixs.append(np.random.choice(self.test_d[self.test_l == i],
                                                           curr_nb_samples, replace=True).tolist())
                        l += [[i] * curr_nb_samples]
                    else:
                        raise NotImplementedError
                except ValueError:  # requested class does not exist in current dataset
                    pass
            ssos = []
            sample_ixs = np.concatenate(sample_ixs)
            l = np.concatenate(l)
            for ix in sample_ixs:
                sso = self.ssd.get_super_segmentation_object(ix)
                sso.nb_cpus = self.nb_cpus
                ssos.append(sso)
            self.view_cache[source] = [sso.load_views(view_key=self.view_key) for sso in ssos]
            self.label_cache[source] = l
            self.sample_weights[source] = compute_class_weight('balanced',
                                                               np.unique(l), l)
            self.current_cache_uses = 0
        ixs = np.arange(len(self.view_cache[source]))
        with temp_seed(None):
            # draw sample accoridng to their class weights
            np.random.shuffle(ixs)
        self.view_cache[source] = [self.view_cache[source][ix] for ix in ixs]
        self.label_cache[source] = self.label_cache[source][ixs]
        d, l = transform_celltype_data_views(self.view_cache[source], self.label_cache[source],
                                             batch_size, self.nb_views,
                                             norm_func=naive_view_normalization_new)
        if self.reduce_context > 0:
            d = d[:, :, :, (self.reduce_context/2):(-self.reduce_context/2),
                self.reduce_context:-self.reduce_context]
        if self.reduce_context_fact > 1:
            d = d[:, :, :, ::self.reduce_context_fact, ::self.reduce_context_fact]
        if self.binary_views:
            d[d < 1.0] = 0
        self.current_cache_uses += 1
        if self.raw_only:
            return d[:, :1], l
        return tuple([d, l])


class CelltypeViewsJ0251(CelltypeViews):
    def __init__(self, inp_node, out_node, raw_only=False, nb_views=20, nb_views_renderinglocations=2,
                 reduce_context=0, binary_views=False, reduce_context_fact=1, n_classes=4,
                 class_weights=(2, 2, 1, 1), load_data=False, nb_cpus=1,
                 random_seed=0, view_key=None, cv_val=None):
        """
        Initializes the CelltypeViewsJ0251 object. This class is a subclass of CelltypeViews and is used for handling
        cell type views for the J0251 dataset. It sets up the working directory, view key, number of views, number of
        CPUs, raw only flag, reduce context, max number of cache uses, number of classes, class weights, view cache,
        label cache, syn sign cache, sample weights, reduce context factor, binary views, example shape, and cv value.
        
        Args:
            inp_node : Input node for the neural network.
            out_node : Output node for the neural network.
            raw_only (bool): If True, only raw data is used. Default is False.
            nb_views (int): Number of sampled views used for prediction of cell type. Default is 20.
            nb_views_renderinglocations (int): Number of views per rendering location. Default is 2.
            reduce_context (int): The amount of context to reduce. Default is 0.
            binary_views (bool): If True, binary views are used. Default is False.
            reduce_context_fact (int): The factor by which to reduce context. Default is 1.
            n_classes (int): The number of classes. Default is 4.
            class_weights (tuple): The weights for each class. Default is (2, 2, 1, 1).
            load_data (bool): If True, data is loaded. Default is False.
            nb_cpus (int): The number of CPUs to use. Default is 1.
            random_seed (int): The seed for the random number generator. Default is 0.
            view_key (str): The key for the view. Default is None.
            cv_val : The cross-validation value. Default is None.
        """

        global_params.wd = "/ssdscratch/pschuber/songbird/j0251/rag_flat_Jan2019_v3/"
        ctgt_key = None  # use standard ssv store
        assert "rag_flat_Jan2019_v3" in global_params.config.working_dir
        assert os.path.isdir(global_params.config.working_dir)
        if view_key is None:
            self.view_key = "raw{}".format(nb_views_renderinglocations)
        else:
            self.view_key = view_key
        self.nb_views = nb_views
        self.nb_cpus = nb_cpus
        self.raw_only = raw_only
        self.reduce_context = reduce_context
        # RandomFlip augmentation etc.
        self.max_nb_cache_uses = 4000 * 2
        self.current_cache_uses = 0
        assert n_classes == len(class_weights)
        self.n_classes = n_classes
        self.class_weights = np.array(class_weights)
        self.view_cache = {'train': None, 'valid': None, 'test': None}
        self.label_cache = {'train': None, 'valid': None, 'test': None}
        self.syn_sign_cache = {'train': None, 'valid': None, 'test': None}
        self.sample_weights = {'train': None, 'valid': None, 'test': None}
        self.reduce_context_fact = reduce_context_fact
        self.binary_views = binary_views
        self.example_shape = (nb_views, 4, 2, 128, 256)
        self.cv_val = cv_val
        # load GT
        self.csv_p = "/wholebrain/songbird/j0251/groundtruth/celltypes/j0251_celltype_gt_v4.csv"
        df = pandas.io.parsers.read_csv(self.csv_p, header=None, names=['ID', 'type']).values
        ssv_ids = df[:, 0].astype(np.uint64)
        if len(np.unique(ssv_ids)) != len(ssv_ids):
            ixs, cnt = np.unique(ssv_ids, return_counts=True)
            raise ValueError(f'Multi-usage of IDs! {ixs[cnt > 1]}')
        str_labels = df[:, 1]
        ssv_labels = np.array([str2int_converter(el, gt_type='ctgt_j0251_v2') for el in str_labels], dtype=np.uint16)
        if self.cv_val is not None and self.cv_val != -1:
            assert self.cv_val < 10
            kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
            for ii, (train_ixs, test_ixs) in enumerate(kfold.split(ssv_ids, y=ssv_labels)):
                if ii == self.cv_val:
                    self.splitting_dict = {'train': ssv_ids[train_ixs], 'valid': ssv_ids[test_ixs]}
        else:
            self.splitting_dict = {'train': ssv_ids, 'valid': ssv_ids}  # use all data
            log_cnn.critical(f'Using all GT data for training!')
        self.label_dict = {k: v for k, v in zip(ssv_ids, ssv_labels)}
        self.sso_ids = self.splitting_dict['train']
        for k, v in self.splitting_dict.items():
            classes, c_cnts = np.unique([self.label_dict[ix] for ix in
                                         self.splitting_dict[k]], return_counts=True)
            log_cnn.debug(f"{k} [labels, counts]: {classes}, {c_cnts}")
            log_cnn.debug(f'{len(self.sso_ids)} SSV IDs in training set: {self.sso_ids}')

        dc_split_prev = dict(self.splitting_dict)
        dc_label_prev = dict(self.label_dict)

        print("Initializing CelltypeViewsJ0251:", self.__dict__)  # TODO: add gt paths to config
        super(CelltypeViews, self).__init__(global_params.config.working_dir, ctgt_key, train_fraction=None,
                         naive_norm=False, load_data=load_data, random_seed=random_seed,
                         splitting_dict=dc_split_prev, label_dict=dc_label_prev)
        # check that super left dicts unmodified
        assert self.splitting_dict == dc_split_prev
        assert self.label_dict == dc_label_prev

        self.train_d = np.array(self.splitting_dict["train"])
        self.valid_d = np.array(self.splitting_dict["valid"])
        ssv_gt_dict = self.label_dict
        self.train_l = np.array([ssv_gt_dict[ix] for ix in self.train_d], np.int16)[:, None]
        self.valid_l = np.array([ssv_gt_dict[ix] for ix in self.valid_d], np.int16)[:, None]
        self.train_d = self.train_d[:, None]
        self.valid_d = self.valid_d[:, None]
        super(MultiViewData, self).__init__()
        for k, v in self.splitting_dict.items():
            classes, c_cnts = np.unique([self.label_dict[ix] for ix in
                                         self.splitting_dict[k]], return_counts=True)
            print(f"{k} [labels, counts]: {classes}, {c_cnts}")


class GliaViews(Data):
    def __init__(self, inp_node, out_node, raw_only=True, nb_views=2,
                 reduce_context=0, binary_views=False, reduce_context_fact=1,
                 naive_norm=True, av_working_dir=None):
        """
        Initializes the GliaViews object. This class is a subclass of Data and is used for handling glia views. It sets
        up the number of views, raw only flag, reduce context, reduce context factor, binary views, and initializes the
        MultiViewData and AxonViews objects.
        
        Args:
            inp_node : Input node for the neural network.
            out_node : Output node for the neural network.
            raw_only (bool): If True, only raw data is used. Default is True.
            nb_views (int): Number of views. Default is 2.
            reduce_context (int): The amount of context to reduce. Default is 0.
            binary_views (bool): If True, binary views are used. Default is False.
            reduce_context_fact (int): The factor by which to reduce context. Default is 1.
            naive_norm (bool): If True, naive normalization is used. Default is True.
            av_working_dir (str): The working directory for the AxonViews object. Default is None.
        """
        self.nb_views = nb_views
        self.raw_only = raw_only
        self.reduce_context = reduce_context
        self.reduce_context_fact = reduce_context_fact
        self.binary_views = binary_views
        print("Initializing GliaViews:", self.__dict__)
        # get glia gt
        GV = MultiViewData("/wholebrain/scratch/areaxfs3/", "gliagt",
                           view_kwargs=dict(view_key="raw{}".format(nb_views)),
                           naive_norm=naive_norm)
        # get axon GT
        AV = AxonViews(inp_node, out_node, raw_only=True, nb_views=nb_views,
                       naive_norm=naive_norm, working_dir=av_working_dir)
        # set label to non-glia
        AV.train_l[:] = 0
        AV.valid_l[:] = 0
        AV.test_l[:] = 0
        self.train_d = np.concatenate([AV.train_d, GV.train_d])
        self.train_l = np.concatenate([AV.train_l, GV.train_l])
        self.valid_d = np.concatenate([AV.valid_d, GV.valid_d])
        self.valid_l = np.concatenate([AV.valid_l, GV.valid_l])
        self.test_d = np.concatenate([AV.test_d, GV.test_d])
        self.test_l = np.concatenate([AV.test_l, GV.test_l])
        print("\nlabels (train) - 0:%d\t1:%d" % (
            np.sum(self.train_l == 0),
            np.sum(self.train_l == 1)))
        print("labels (valid) - 0:%d\t1:%d" % (
            np.sum(self.valid_l == 0),
            np.sum(self.valid_l == 1)))
        self.example_shape = self.train_d[0].shape
        super(GliaViews, self).__init__()

    def getbatch(self, batch_size, source='train'):
        """
        Generates a batch of data for training, validation or testing. The function applies various transformations to
        the data such as reducing context, binary views, and raw only. It also flips the data along the x and y axes.
        
        Args:
            batch_size (int): The size of the batch to be generated.
            source (str): The source of the data, can be 'train', 'valid', or 'test'. Default is 'train'.
        
        Returns:
            tuple: A tuple containing the data and labels for the batch.
        """
        # TODO: keep in mind that super().get_batch does not shuffle validation data ->
        #  batch_size for validation should be sufficiently big
        # if source == 'valid':
        #     nb = len(self.valid_l)
        #     ixs = np.arange(nb)
        #     np.random.shuffle(ixs)
        #     self.valid_d = self.valid_d[ixs]
        #     self.valid_l = self.valid_l[ixs]
        d, l = super(GliaViews, self).getbatch(batch_size, source)
        view_shuffle = np.arange(0, d.shape[2])
        np.random.shuffle(view_shuffle)
        if self.reduce_context > 0:
            d = d[:, :, :, (self.reduce_context/2):(-self.reduce_context/2),
                self.reduce_context:-self.reduce_context]
        if self.reduce_context_fact > 1:
            d = d[:, :, :, ::self.reduce_context_fact, ::self.reduce_context_fact]
        d = d[:, :, view_shuffle]
        flipx, flipy = np.random.randint(0, 2, 2)
        if flipx:
            d = d[..., ::-1, :]
        if flipy:
            d = d[..., ::-1]
        if self.binary_views:
            d[d < 1.0] = 0
        if self.raw_only and d.shape[1] > 1:
            d = d[:, :1]
        return tuple([d, l])


def transform_celltype_data_views(sso_views, labels, batch_size, nb_views,
                                  norm_func=None):
    """
    Transforms cell type data views. The function normalizes the views, reshapes the views, and fills up missing
    samples with a random set of those.
    
    Args:
        sso_views : The views of the super segmentation object.
        labels : The labels for the data.
        batch_size (int): The size of the batch to be generated.
        nb_views (int): Number of views.
        norm_func : The normalization function to be used. Default is None.
    
    Returns:
        tuple: A tuple containing the transformed views and labels.
    """
    if norm_func is None:
        norm_func = naive_view_normalization
    orig_views = np.zeros((batch_size, 4, nb_views, 128, 256), dtype=np.float32)
    new_labels = np.zeros((batch_size, 1), dtype=np.int16)
    cnt = 0
    # sample_fac_sv = np.max([int(nb_views / 10), 1]) # draw more SV if #views is high
    for ii, views in enumerate(sso_views):
        # sso.load_attr_dict()
        # sample_svs = np.random.choice(list(sso.svs), np.min([nb_views*sample_fac_sv, len(sso.sv_ids)]), replace=False)
        # views = np.concatenate(start_multiprocess_obj("load_views", [[sv, ] for sv in sample_svs], nb_cpus=nb_cpus))
        views = norm_func(views)
        views = views.swapaxes(1, 0).reshape((4, -1, 128, 256))
        curr_nb_samples = int(np.min([np.floor(views.shape[1]/nb_views), batch_size-cnt, batch_size//len(sso_views)]))
        curr_nb_samples = np.max([curr_nb_samples, 1])
        if curr_nb_samples == 0:
            continue
        view_sampling = np.random.choice(views.shape[1], curr_nb_samples*nb_views, replace=False)
        orig_views[cnt:(curr_nb_samples+cnt)] = views[:, view_sampling].reshape((4, curr_nb_samples,
                                                                                 nb_views, 128, 256)).swapaxes(1, 0)
        new_labels[cnt:(curr_nb_samples+cnt)] = labels[ii]
        cnt += curr_nb_samples
        if cnt == batch_size:
            break
    if cnt == 0:
        print("--------------------------------------" 
              "Number of views in batch is zero. " 
              "Missing views and labels were filled with 0."
              "--------------------------------------")
    elif cnt < batch_size:
        # print "--------------------------------------" \
        #       "%d/%d samples were collected initially. Filling up missing" \
        #       "samples with random set of those." \
        #       "--------------------------------------" % (cnt, batch_size)
        while cnt != batch_size:
            curr_nb_samples = 1
            random_ix = np.random.choice(np.arange(cnt), 1, replace=False)
            orig_views[cnt:(curr_nb_samples + cnt)] = orig_views[random_ix]
            new_labels[cnt:(curr_nb_samples + cnt)] = new_labels[random_ix]
            cnt += curr_nb_samples
    return tuple([orig_views, new_labels])


def transform_celltype_data_views_alternative(sso_views, labels, syn_signs, batch_size, nb_views):
    """
    Transforms the cell type data views by sampling views from the given sso_views and labels. 
    The function also handles cases where the number of views in a batch is zero or less than the batch size.
    
    Args:
        sso_views (np.array): Array of views from super supervoxel objects.
        labels (np.array): Array of labels corresponding to the sso_views.
        syn_signs (np.array): Array of synaptic signs corresponding to the sso_views.
        batch_size (int): The size of the batch to be returned.
        nb_views (int): The number of views to be sampled for each batch.
    
    Returns:
        tuple: A tuple containing the original views, new labels, and new synaptic signs.
    """
    orig_views = np.zeros((batch_size, 4, nb_views, 128, 256), dtype=np.float32)
    new_labels = np.zeros((batch_size, 1), dtype=np.int16)
    new_synsigns = np.zeros((batch_size, 2), dtype=np.float32)
    cnt = 0
    # sample_fac_sv = np.max([int(nb_views / 10), 1]) # draw more SV if #views is high
    for ii, views in enumerate(sso_views):
        curr_nb_samples = int(np.min([np.floor(views.shape[1]/nb_views), batch_size-cnt, batch_size//len(sso_views)]))
        curr_nb_samples = np.max([curr_nb_samples, 1])
        if curr_nb_samples == 0:
            continue
        view_sampling = np.random.choice(np.arange(views.shape[1]),
                                         curr_nb_samples*nb_views, replace=True)
        orig_views[cnt:(curr_nb_samples+cnt)] = views[:, view_sampling].reshape((4, curr_nb_samples,
                                                                                 nb_views, 128, 256)).swapaxes(1, 0)
        new_labels[cnt:(curr_nb_samples+cnt)] = labels[ii]
        new_synsigns[cnt:(curr_nb_samples+cnt)] = syn_signs[ii]
        cnt += curr_nb_samples
        if cnt == batch_size:
            break
    if cnt == 0:
        print("--------------------------------------" 
              "Number of views in batch is zero. " 
              "Missing views and labels were filled with 0."
              "--------------------------------------")
    elif cnt < batch_size:
        # print "--------------------------------------" \
        #       "%d/%d samples were collected initially. Filling up missing" \
        #       "samples with random set of those." \
        #       "--------------------------------------" % (cnt, batch_size)
        while cnt != batch_size:
            curr_nb_samples = 1
            random_ix = np.random.choice(np.arange(cnt), 1, replace=False)
            orig_views[cnt:(curr_nb_samples + cnt)] = orig_views[random_ix]
            new_labels[cnt:(curr_nb_samples + cnt)] = new_labels[random_ix]
            new_synsigns[cnt:(curr_nb_samples + cnt)] = new_synsigns[random_ix]
            cnt += curr_nb_samples
    return tuple([orig_views, new_labels, new_synsigns])


def transform_celltype_data_views_alternative_noscal(sso_views, labels, batch_size,
                                                 nb_views):
    """
    Transforms the cell type data views without scaling. The function samples views from the given sso_views and labels. 
    It also handles cases where the number of views in a batch is zero or less than the batch size.
    
    Args:
        sso_views (np.array): Array of views from super supervoxel objects.
        labels (np.array): Array of labels corresponding to the sso_views.
        batch_size (int): The size of the batch to be returned.
        nb_views (int): The number of views to be sampled for each batch.
    
    Returns:
        tuple: A tuple containing the original views and new labels.
    """
    orig_views = np.zeros((batch_size, 4, nb_views, 128, 256), dtype=np.float32)
    new_labels = np.zeros((batch_size, 1), dtype=np.int16)
    cnt = 0
    # sample_fac_sv = np.max([int(nb_views / 10), 1]) # draw more SV if #views is high
    for ii, views in enumerate(sso_views):
        curr_nb_samples = int(np.min([np.floor(views.shape[1]/nb_views), batch_size-cnt, batch_size//len(sso_views)]))
        curr_nb_samples = np.max([curr_nb_samples, 1])
        if curr_nb_samples == 0:
            continue
        view_sampling = np.random.choice(np.arange(views.shape[1]),
                                         curr_nb_samples*nb_views, replace=False)
        orig_views[cnt:(curr_nb_samples+cnt)] = views[:, view_sampling].reshape((4, curr_nb_samples,
                                                                                 nb_views, 128, 256)).swapaxes(1, 0)
        new_labels[cnt:(curr_nb_samples+cnt)] = labels[ii]
        cnt += curr_nb_samples
        if cnt == batch_size:
            break
    if cnt == 0:
        print("--------------------------------------" 
              "Number of views in batch is zero. " 
              "Missing views and labels were filled with 0."
              "--------------------------------------")
    elif cnt < batch_size:
        # print "--------------------------------------" \
        #       "%d/%d samples were collected initially. Filling up missing" \
        #       "samples with random set of those." \
        #       "--------------------------------------" % (cnt, batch_size)
        while cnt != batch_size:
            curr_nb_samples = 1
            random_ix = np.random.choice(np.arange(cnt), 1, replace=False)
            orig_views[cnt:(curr_nb_samples + cnt)] = orig_views[random_ix]
            new_labels[cnt:(curr_nb_samples + cnt)] = new_labels[random_ix]
            cnt += curr_nb_samples
    return tuple([orig_views, new_labels])


def transform_celltype_data(ssos, labels, batch_size, nb_views, nb_cpus=1,
                            view_key=None, norm_func=None):
    """
    Transforms the cell type data by sampling views from the given ssos and labels. 
    The function also handles cases where the number of views in a batch is zero or less than the batch size.
    
    Args:
        ssos (list): List of super supervoxel objects.
        labels (np.array): Array of labels corresponding to the ssos.
        batch_size (int): The size of the batch to be returned.
        nb_views (int): The number of views to be sampled for each batch.
        nb_cpus (int): The number of CPUs to be used for multiprocessing.
        view_key (str): The key to be used to load views from the ssos.
        norm_func (function): The function to be used for normalizing the views.
    
    Returns:
        tuple: A tuple containing the original views and new labels.
    """
    if norm_func is None:
        norm_func = naive_view_normalization
    orig_views = np.zeros((batch_size, 4, nb_views, 128, 256), dtype=np.float32)
    new_labels = np.zeros((batch_size, 1), dtype=np.int16)
    cnt = 0
    # sample_fac_sv = np.max([int(nb_views / 10), 1]) # draw more SV if #views is high
    for ii, sso in enumerate(ssos):
        # sso.load_attr_dict()
        sso.nb_cpus = nb_cpus
        # sample_svs = np.random.choice(list(sso.svs), np.min([nb_views*sample_fac_sv, len(sso.sv_ids)]), replace=False)
        # views = np.concatenate(start_multiprocess_obj("load_views", [[sv, ] for sv in sample_svs], nb_cpus=nb_cpus))
        views = np.concatenate(sso.load_views(view_key=view_key))
        sso.clear_cache()
        views = norm_func(views)
        views = views.swapaxes(1, 0).reshape((4, -1, 128, 256))
        curr_nb_samples = int(np.min([np.floor(views.shape[1]/nb_views), batch_size-cnt, batch_size//4]))
        if curr_nb_samples == 0:
            continue
        view_sampling = np.random.choice(np.arange(views.shape[1]),
                                         curr_nb_samples*nb_views, replace=False)
        orig_views[cnt:(curr_nb_samples+cnt)] = views[:, view_sampling].reshape((4, curr_nb_samples, nb_views, 128, 256)).swapaxes(1, 0)
        new_labels[cnt:(curr_nb_samples+cnt)] = labels[ii]
        cnt += curr_nb_samples
        if cnt == batch_size:
            break
    if cnt == 0:
        print("--------------------------------------" 
              "Number of views in batch is zero. " 
              "Missing views and labels were filled with 0."
              "--------------------------------------")
    elif cnt < batch_size:
        # print "--------------------------------------" \
        #       "%d/%d samples were collected initially. Filling up missing" \
        #       "samples with random set of those." \
        #       "--------------------------------------" % (cnt, batch_size)
        while cnt != batch_size:
            curr_nb_samples = 1
            random_ix = np.random.choice(np.arange(cnt), 1, replace=False)
            orig_views[cnt:(curr_nb_samples + cnt)] = orig_views[random_ix]
            new_labels[cnt:(curr_nb_samples + cnt)] = new_labels[random_ix]
            cnt += curr_nb_samples
    return tuple([orig_views, new_labels])


class TripletData_N(Data):
    """
    A class representing a dataset for training a neural network. It
    utilizes neighboring location for small distance sample.
    """
    def __init__(self, input_node, target_node):
        """
        Initializes the TripletData_N object.
        
        Args:
            input_node (str): The name of the input node in the neural network.
            target_node (str): The name of the target node in the neural network.
        """
        self.sds = SegmentationDataset("sv", working_dir="/wholebrain/scratch/areaxfs3/",
                                       version=0)
        ssds = SuperSegmentationDataset(working_dir="/wholebrain/scratch/areaxfs3/")
        ssds.load_mapping_dict()
        rev_dc = {}
        for k, v in ssds.mapping_dict.items():
            for el in v:
                rev_dc[el] = k
        self.s_ids = np.concatenate(ssds.mapping_dict.values())
        bb = ssds.load_numpy_data("bounding_box")
        # sizes as diagonal of bounding box in um (SV size will be size of corresponding SSV)
        bb_size = np.linalg.norm((bb[:, 1] - bb[: ,0])*self.sds.scaling, axis=1) / 1e3
        ssds_sizes = {}
        for i in range(len(ssds.ssv_ids)):
            ssds_sizes[ssds.ssv_ids[i]] = bb_size[i]
        sizes = np.array([ssds_sizes[rev_dc[ix]] for ix in self.s_ids], dtype=np.float32)
        # assign weights: 0 for SV below 8, 1 for 8 to 18, 2 for 19 to 28, truncated at 5 for 49 to 58
        self.s_weights = (sizes - 8.) / 10.
        self.s_weights *= np.array(self.s_weights > 0, dtype=np.int32)
        self.s_weights = np.ceil(self.s_weights)
        self.s_weights[self.s_weights > 6] = 6
        print("Data Summary:\nweight\t#samples")
        for i in range(7):
            print("%d\t%d" % (i, np.sum(self.s_weights == i)))
        print("Using %d SV (weight bigger than 0)." % (np.sum(self.s_weights > 0)))
        example = self.sds.get_segmentation_object(self.s_ids[1])
        self.example_shape = example.load_views()[0].shape
        self.train_d = np.array(self.s_ids)
        self.valid_d = np.zeros((0, ))
        self.test_d = np.zeros((0, ))
        print("Samples (train):", self.train_d.shape)
        print("Samples (valid):", self.valid_d.shape)
        print("Samples (test):", self.test_d.shape)
        self.train_l = np.zeros((len(self.train_d), 1))
        self.test_l = np.zeros((len(self.test_d), 1))
        self.valid_l = np.zeros((len(self.valid_d), 1))
        super(TripletData_N, self).__init__()
        print("Initializing SSV Data:", self.__repr__())

    def getbatch(self, batch_size, source='train'):
        """
        Returns a batch of data for training or testing.
        
        Args:
            batch_size (int): The size of the batch to be returned.
            source (str): The source of the data ('train', 'valid', or 'test').
        
        Returns:
            tuple: A tuple containing the data and the labels for the batch.
        """
        if source != "train":
            print("Does not have valid and test datasets, returning batch " \
                  "from training pool.")
        self._reseed()
        nb_per_weight = batch_size // 6
        assert nb_per_weight * 6 == batch_size, "Batch size must be multiple of 6"
        chosen_ixs = []
        for i in range(1, 7):
            curr_w = self.s_weights[self.s_weights == i]
            curr_w /= curr_w.sum()
            chosen_ixs += np.random.choice(self.s_ids[self.s_weights == i], nb_per_weight, p=curr_w, replace=False).tolist()
        sos = [self.sds.get_segmentation_object(ix) for ix in chosen_ixs]
        out_d = transform_tripletN_data_so(sos)
        rotate_by_pi_x = np.random.randint(0, 2)
        rotate_by_pi_y = np.random.randint(0, 2)
        if rotate_by_pi_y:
            out_d = out_d[..., ::-1]
        if rotate_by_pi_x:
            out_d = out_d[..., ::-1, :]
        return tuple([out_d, None])


class TripletData_SSV(Data):
    """
    A class that represents a dataset for training a neural network. It uses orthogonal 
    views of SSVs to generate a set of three views: one as reference, one as similar, 
    and one as different view.
    """
    def __init__(self, input_node, target_node, nb_cpus=1,
                 raw_only=False, downsample=1):
        """
        Initializes the TripletData_SSV object.
        
        Args:
            input_node (str): The name of the input node in the neural network.
            target_node (str): The name of the target node in the neural network.
            nb_cpus (int): The number of CPUs to be used for multiprocessing.
            raw_only (bool): If True, only raw data will be returned.
            downsample (int): The factor by which to downsample the data.
        """
        self.nb_cpus = nb_cpus
        self.raw_only = raw_only
        self.sds = SegmentationDataset("sv", working_dir="/wholebrain/scratch/areaxfs3/")
        self.ssds = SuperSegmentationDataset(working_dir="/wholebrain/scratch/areaxfs3/")
        self.downsample = downsample
        self.example_shape = (4, 3, 1024//downsample, 1024//downsample)

        self.train_d = np.array((0, ))
        self.valid_d = np.zeros((0, ))
        self.test_d = np.zeros((0, ))
        self.train_l = np.zeros((len(self.train_d), 1))
        self.test_l = np.zeros((len(self.test_d), 1))
        self.valid_l = np.zeros((len(self.valid_d), 1))

        self.sso_ids = self.ssds.ssv_ids
        bb = self.ssds.load_numpy_data("bounding_box")
        for sso_ix in self.valid_d:
            bb[self.sso_ids == sso_ix] = 0  # bb below 8, i.e. also 0, will be ignored during training
        for sso_ix in self.test_d:
            bb[self.sso_ids == sso_ix] = 0  # bb below 8, i.e. also 0, will be ignored during training

        # sizes as diagonal of bounding box in um (SV size will be size of corresponding SSV)
        bb_size = np.linalg.norm((bb[:, 1] - bb[: ,0])*self.sds.scaling, axis=1) / 1e3
        ssds_sizes = np.zeros((len(self.ssds.ssv_ids)))
        for i in range(len(self.ssds.ssv_ids)):
            ssds_sizes[i] = bb_size[i]
        # assign weights: 0 for SV below 8, 1 for 8 to 18, 2 for 19 to 28, truncated at 5 for 49 to 58
        self.sso_weights = (ssds_sizes - 8.) / 10.
        self.sso_weights *= np.array(self.sso_weights > 0, dtype=np.int32)
        self.sso_weights = np.ceil(self.sso_weights)
        self.sso_weights[self.sso_weights > 6] = 6
        self.sso_ids = np.array(self.sso_ids, dtype=np.uint32)
        super(TripletData_SSV, self).__init__()
        print("Initializing SSV Data:", self.__repr__())

    def getbatch(self, batch_size, source='train'):
        """
        Returns a batch of data for training or testing.
        
        Args:
            batch_size (int): The size of the batch to be returned.
            source (str): The source of the data ('train', 'valid', or 'test').
        
        Returns:
            tuple: A tuple containing the data and the labels for the batch.
        """
        while True:
            try:
                self._reseed()
                # the number of views (one for reference and one set for similar N-view set, i.e. different views from the same ssv
                nb_per_weight = batch_size // 6
                assert nb_per_weight * 6 == batch_size, "Batch size must be multiple of 6"
                ssos = []
                for i in range(1, 7):
                    existing_views = []
                    while True:
                        curr_w = self.sso_weights[self.sso_weights == i]
                        curr_w /= curr_w.sum()
                        sampled_sso = np.random.choice(
                            self.sso_ids[self.sso_weights == i], nb_per_weight * 10, # increase number of sample SSVs to minimize chance of resampling
                            p=curr_w, replace=False).tolist()
                        for ix in sampled_sso:
                            sso = self.ssds.get_super_segmentation_object(ix)
                            sso.enable_locking = False
                            sso.nb_cpus = self.nb_cpus
                            try:
                                _ = sso.load_views("ortho")
                                existing_views.append(sso)
                            except KeyError:
                                self.sso_weights[self.sso_ids == ix] = 0  # sso will not be sampled
                                # print("WARNING: Couldn't find 'ortho' views of SSO %d (weight: %d)" % (ix, i))
                            if len(existing_views) == nb_per_weight or\
                                    len(self.sso_ids[self.sso_weights == i]) == 0:
                                break
                        if len(existing_views) == nb_per_weight or\
                                len(self.sso_ids[self.sso_weights == i]) == 0:
                            break
                        print("Couldn't retrieve enough samples for batch, "
                              "chance of sampling SSVs multiple times.")
                    ssos += existing_views
                # get random view pair of each ssos for small distance sample
                out_d = np.zeros((len(ssos), 4, 2, 1024//self.downsample, 1024//self.downsample), dtype=np.float32)
                for ii in range(len(ssos)):
                    views = ssos[ii].load_views("ortho").swapaxes(1, 0)
                    perm_ixs_same_ssv = np.arange(views.shape[1])
                    np.random.shuffle(perm_ixs_same_ssv)
                    views = views[:, perm_ixs_same_ssv[:2]]
                    out_d[ii] = views[..., ::self.downsample, ::self.downsample] / 255 - 0.5
                # shape [bs, 4, 2, x, y]
                for ii in range(len(out_d)):
                    rotate_by_pi_x = np.random.randint(0, 2)
                    rotate_by_pi_y = np.random.randint(0, 2)
                    if rotate_by_pi_y:
                        out_d[ii] = out_d[ii, ..., ::-1]
                    if rotate_by_pi_x:
                        out_d[ii] = out_d[ii, ..., ::-1, :]
                # add different view of different sso for big distance as third view
                out_d = transform_tripletN_data_SSV(out_d)
                # shape [bs, 4, 3, x, y]
                if self.raw_only:
                    return out_d[:, :1], None
                return out_d, None
            except RuntimeError as e:
                print("RuntimeError occured during 'getbatch'. Trying again.\n"
                      "%s\n"  % e)


class TripletData_SSV_nviews(Data):
    """
    This class, a child of the Data class, represents Super Supervoxel (SSV) data using a 
    specified number of 'nb_views', randomly sampled views. It loads, prepares the data for 
    training, validation, and testing, and handles the generation of training batches.
    """
    def __init__(self, input_node, target_node, nb_views=20, nb_cpus=1,
                 raw_only=False):
        """
        Initializes the TripletData_SSV_nviews object. It loads the SSV splits and ground truth 
        dictionary, sets the number of views, CPUs, and raw_only flag. It also initializes the 
        SegmentationDataset and SuperSegmentationDataset, sets the example shape, and prepares 
        the data for training, validation, and testing.
        
        Args:
            input_node: The input node for the data.
            target_node: The target node for the data.
            nb_views (int, optional): The number of views to be used. Default is 20.
            nb_cpus (int, optional): The number of CPUs to be used. Default is 1.
            raw_only (bool, optional): If True, only raw data is used. Default is False.
        """
        ssv_splits = load_pkl2obj("/wholebrain/scratch/pschuber/NeuroPatch/gt/ssv_ctgt_splitted_ids_cleaned.pkl")
        ssv_gt_dict = load_pkl2obj("/wholebrain/scratch/pschuber/NeuroPatch/gt/ssv_ctgt.pkl")
        self.nb_views = nb_views
        self.nb_cpus = nb_cpus
        self.raw_only = raw_only
        self.sds = SegmentationDataset("sv", working_dir="/wholebrain/scratch/areaxfs/", version="0")
        self.ssds = SuperSegmentationDataset(working_dir="/wholebrain/scratch/areaxfs/", version="6")

        self.example_shape = (nb_views, 4, 2, 128, 256)
        self.train_d = ssv_splits["train"]
        self.valid_d = ssv_splits["valid"]
        self.test_d = ssv_splits["test"]
        self.train_l = np.array([ssv_gt_dict[ix] for ix in self.train_d], np.int16)[:, None]
        self.test_l = np.array([ssv_gt_dict[ix] for ix in self.test_d], np.int16)[:, None]
        self.valid_l = np.array([ssv_gt_dict[ix] for ix in self.valid_d], np.int16)[:, None]
        self.train_d = self.train_d[:, None]
        self.valid_d = self.valid_d[:, None]
        self.test_d = self.test_d[:, None]

        self.sso_ids = self.ssds.ssv_ids
        bb = self.ssds.load_numpy_data("bounding_box")
        for sso_ix in self.valid_d:
            ix = self.sso_ids.index(sso_ix)
            bb[ix] = 0  # bb below 8, i.e. also 0, will be ignored during training
        for sso_ix in self.test_d:
            ix = self.sso_ids.index(sso_ix)
            bb[ix] = 0  # bb below 8, i.e. also 0, will be ignored during training

        # sizes as diagonal of bounding box in um (SV size will be size of corresponding SSV)
        bb_size = np.linalg.norm((bb[:, 1] - bb[:, 0])*self.sds.scaling, axis=1) / 1e3
        ssds_sizes = np.zeros((len(self.ssds.ssv_ids)))
        for i in range(len(self.ssds.ssv_ids)):
            ssds_sizes[i] = bb_size[i]
        # assign weights: 0 for SV below 8, 1 for 8 to 18, 2 for 19 to 28, truncated at 5 for 49 to 58
        self.sso_weights = (ssds_sizes - 8.) / 10.
        self.sso_weights *= np.array(self.sso_weights > 0, dtype=np.int32)
        self.sso_weights = np.ceil(self.sso_weights)
        self.sso_weights[self.sso_weights > 6] = 6
        self.sso_ids = np.array(self.sso_ids, dtype=np.uint32)
        super(TripletData_SSV_nviews, self).__init__()
        print("Initializing SSV Data:", self.__repr__())

    def getbatch(self, batch_size, source='train'):
        """
        Generates a batch of data for training. It reseeds the random number generator, selects 
        indices based on weights, loads the corresponding SSVs, and transforms the data into the 
        required format. If only raw data is required, it returns the raw data only. Otherwise, 
        it returns both the raw and the transformed data.
        
        Args:
            batch_size (int): The size of the batch to be generated.
            source (str, optional): The source of the data ('train', 'valid', or 'test'). Default is 'train'.
        
        Returns:
            np.array: The batch of data.
        """
        while True:
            try:
                self._reseed()
                # the number of views (one for reference and one set for similar N-view set, i.e.
                # different views from the same ssv
                nb_per_weight = batch_size // 6
                assert nb_per_weight * 6 == batch_size, "Batch size must be multiple of 6"
                chosen_ixs = []
                for i in range(1, 7):
                    curr_w = self.sso_weights[self.sso_weights == i]
                    curr_w /= curr_w.sum()
                    chosen_ixs += np.random.choice(
                        self.sso_ids[self.sso_weights == i], nb_per_weight,
                        p=curr_w, replace=False).tolist()
                ssos = []
                for ix in chosen_ixs:
                    sso = self.ssds.get_super_segmentation_object(ix)
                    sso.enable_locking = False
                    sso.nb_cpus = self.nb_cpus
                    ssos.append(sso)
                l = np.zeros((len(ssos), )) # dummy labels
                # get pair of view set for small distance
                out_d, l = transform_celltype_data(ssos, l, batch_size, self.nb_views * 2)
                # shape [bs, 4, nb_views * 2, x, y9
                # add different set of views for big distance to pair of similar set of views
                out_d = transform_tripletN_data_SSV(out_d)
                if self.raw_only:
                    return out_d[:, :1], None
                return out_d, None
            except RuntimeError as e:
                print("RuntimeError occured during 'getbatch'. Trying again.\n"
                      "%s\n" % e)


def transform_tripletN_data_SSV(orig_views):
    """
    Transforms the original views into a new format. The new format has the same shape as the 
    original views, but with 50% more views. If the number of views was initially 6, then it is 
    assumed that 3 and 3 views are used for the similar pair and 3 random views will be added 
    from a random different sample in this batch.
    
    Args:
        orig_views (np.array): The original views. The shape is (batch size, nb channels, 
        nb views, x, y).
    
    Returns:
        np.array: The transformed views with the same shape as orig_views, but with 50% more 
        views. If the number of views was initially 6, it is assumed that 3 and 3 views are 
        used for the similar pair and 3 random views will be added from a random different 
        sample in this batch.
    """
    # split into view to be compared to similar view and very likely view from different SV/SSV
    bigger_dist_d = np.array(orig_views, dtype=np.float32)  # copy
    # perm_ixs = np.arange(orig_views.shape[0])
    # np.random.shuffle(perm_ixs)
    # instead of shuffling (finite chance to have view of same sso as different view) indices are shifted
    perm_ixs = np.roll(np.arange(orig_views.shape[0]), shift=2)
    # view will be compared to a randomly assigned view (likely to be different)
    bigger_dist_d = bigger_dist_d[perm_ixs] # shift array elements by shift_val
    perm_ixs_same_ssv = np.arange(orig_views.shape[2])
    np.random.shuffle(perm_ixs_same_ssv)
    orig_views = orig_views[:, :, perm_ixs_same_ssv]
    nb_views_per_set = len(perm_ixs_same_ssv) // 2
    # orig views contains nb_views * 2 views, which will be used as two equally sized sets for reference and similar pair
    out_d = np.concatenate([orig_views,
                            bigger_dist_d[:, :, :nb_views_per_set]], axis=2)
    return out_d.astype(np.float32)


def transform_tripletN_data_so(sos):
    """
    Transforms the given Super Objects (SOs) into a new format. The new format has three views: 
    one original view, one similar view, and one likely different view. The views are then 
    concatenated along the channel axis.
    
    Args:
        sos (list): The list of Super Objects (SOs) to be transformed.
    
    Returns:
        np.array: The transformed SOs.
    """
    # split into view to be compared to similar view and probably different view
    orig_views = np.zeros((len(sos), 4, 2, 128, 256))
    shift_val = np.min([int(len(sos) / 3), 10])
    cnt = 0
    for ii, so in enumerate(sos):
        try:
            views = so.load_views()
        except KeyError: #use previous views
            continue
        view_ixs = np.arange(len(views))
        np.random.shuffle(view_ixs)
        curr_nb_samples = np.min([shift_val, len(views), len(sos)-cnt])
        orig_views[cnt:(curr_nb_samples+cnt)] = views[view_ixs[:curr_nb_samples]]
        cnt += curr_nb_samples
        if cnt == len(sos):
            break
    if cnt != len(sos):
        warnings.warn("Number of views in batch is not equal to batch size. "
                      "Missing views were filled with 0.")
    view_flip = np.random.randint(0, 2)
    small_dist_d = orig_views[:, :, int(1-view_flip)]
    bigger_dist_d = np.array(small_dist_d, dtype=np.float32)  # copy
    perm_ixs = np.roll(np.arange(small_dist_d.shape[0]), shift=shift_val)
    # view will be compared to a randomly assigned view (likely to be different)
    bigger_dist_d = bigger_dist_d[perm_ixs] # shift array elements by shift_val

    out_d = np.concatenate([orig_views[:, :, view_flip][:, :, None],
                            small_dist_d[:, :, None],
                            bigger_dist_d[:, :, None]], axis=2)
    return out_d.astype(np.float32)


def transform_tripletN_data(d, channels_to_load, view_striding):
    """
    Transforms the given data into a new format. The new format has three views: one original 
    view, one similar view, and one likely different view. The views are then concatenated along 
    the channel axis. The number of views and the channels to load can be specified.
    
    Args:
        d (np.array): The data to be transformed.
        channels_to_load (list): The list of channels to load.
        view_striding (int): The stride for the views.
    
    Returns:
        np.array: The transformed data.
    """
    # split into view to be compared to similar view and probably different view
    comp_d = np.concatenate([v[0].load()[None,] for v in d])
    small_dist_d = np.concatenate([v[1].load()[None,] for v in d])
    big_dist_d = np.array(comp_d)  # copy
    perm_ixs = np.roll(np.arange(d.shape[0]), 1)
    big_dist_d = big_dist_d[perm_ixs]  # rotate array by 1, i.e. each original
    # view will be compared to a randomly assigned view (likely to be different)
    # change channels
    channels_to_load = list(channels_to_load)
    out_d_1 = comp_d[:, channels_to_load[0]][:, None]
    out_d_2 = small_dist_d[:, channels_to_load[0]][:, None]
    out_d_3 = big_dist_d[:, channels_to_load[0]][:, None]
    for ch in channels_to_load[1:]:
        out_d_1 = np.concatenate([out_d_1, comp_d[:, ch][:, None]], axis=1)
        out_d_2 = np.concatenate([out_d_2, small_dist_d[:, ch][:, None]],
                                 axis=1)
        out_d_3 = np.concatenate([out_d_3, big_dist_d[:, ch][:, None]], axis=1)

    # change number views
    if view_striding != 1:
        assert view_striding in [1, 2, 3]
        out_d_1 = out_d_1[:, :, ::view_striding, :, :]
        out_d_2 = out_d_2[:, :, ::view_striding, :, :]
        out_d_3 = out_d_3[:, :, ::view_striding, :, :]

    # sample views
    view_sampling = np.random.choice(comp_d.shape[2], 3, replace=False)
    out_d_1 = out_d_1[:, :, view_sampling[0]][:, :, None]
    out_d_2 = out_d_2[:, :, view_sampling[1]][:, :, None]
    out_d_3 = out_d_3[:, :, view_sampling[2]][:, :, None]

    out_d = np.concatenate([out_d_1, out_d_2, out_d_3], axis=2)
    return out_d


def transform_tripletN_data_predonly(d, channels_to_load, view_striding):
    """
    Transforms the given data into a new format for prediction only. The new format has three 
    views: one original view, one similar view, and one likely different view. The views are 
    then concatenated along the channel axis. The number of views and the channels to load can 
    be specified.
    
    Args:
        d (np.array): The data to be transformed.
        channels_to_load (list): The list of channels to load.
        view_striding (int): The stride for the views.
    
    Returns:
        np.array: The transformed data.
    """
    # split into view to be compared to similar view and probably different view
    comp_d = np.concatenate([v[0].load()[None,] for v in d])
    small_dist_d = np.zeros(comp_d.shape, dtype=np.float32)
    big_dist_d = np.zeros(comp_d.shape, dtype=np.float32)
    perm_ixs = np.roll(np.arange(d.shape[0]), 1)
    big_dist_d = big_dist_d[perm_ixs]  # rotate array by 1, i.e. each original
    # view will be compared to a randomly assigned view (likely to be different)
    # change channels
    channels_to_load = list(channels_to_load)
    out_d_1 = comp_d[:, channels_to_load[0]][:, None]
    out_d_2 = small_dist_d[:, channels_to_load[0]][:, None]
    out_d_3 = big_dist_d[:, channels_to_load[0]][:, None]
    for ch in channels_to_load[1:]:
        out_d_1 = np.concatenate([out_d_1, comp_d[:, ch][:, None]], axis=1)
        out_d_2 = np.concatenate([out_d_2, small_dist_d[:, ch][:, None]],
                                 axis=1)
        out_d_3 = np.concatenate([out_d_3, big_dist_d[:, ch][:, None]], axis=1)

    # change number views
    if view_striding != 1:
        assert view_striding in [1, 2, 3]
        out_d_1 = out_d_1[:, :, ::view_striding, :, :]
        out_d_2 = out_d_2[:, :, ::view_striding, :, :]
        out_d_3 = out_d_3[:, :, ::view_striding, :, :]

    # sample views
    view_sampling = np.random.choice(comp_d.shape[2], 3, replace=False)
    out_d_1 = out_d_1[:, :, view_sampling[0]][:, :, None]
    out_d_2 = out_d_2[:, :, view_sampling[1]][:, :, None]
    out_d_3 = out_d_3[:, :, view_sampling[2]][:, :, None]
    out_d = np.concatenate([out_d_1, out_d_2, out_d_3], axis=2)
    return out_d


def add_gt_sample(ssv_id, label, gt_type, set_type="train"):
    """
    Adds a ground truth sample to the specified set. This function is currently unused.
    
    Args:
        ssv_id (int): The ID of the Super Supervoxel (SSV) to be added.
        label (int): The label of the SSV.
        gt_type (str): The type of ground truth (e.g., 'axgt').
        set_type (str, optional): The type of set to which the SSV should be 
        added ('train', 'valid', or 'test'). Default is 'train'.
    """
    # retrieve SSV from original SSD (which is used in Knossos-Plugin) and
    # copy its data to the yet empty SSV in the axgt SSD.
    ssd_axgt = SuperSegmentationDataset(version=gt_type, working_dir=global_params.config.working_dir)
    ssd = SuperSegmentationDataset(working_dir=global_params.config.working_dir)
    ssv = ssd.get_super_segmentation_object(ssv_id)
    ssv_axgt = ssd_axgt.get_super_segmentation_object(ssv_id)
    ssv.copy2dir(ssv_axgt.ssv_dir)
    # add entries to label and splitting dict
    base_dir = "{}/ssv_{}/".format(global_params.config.working_dir, gt_type)
    splitting = load_pkl2obj("{}/axgt_splitting.pkl".format(base_dir))
    labels = load_pkl2obj("{}/axgt_labels.pkl".format(base_dir))
    splitting[set_type].append(ssv_id)
    labels[ssv_id] = label


def fetch_single_synssv_typseg(syn_ssv: SegmentationObject,
                               syntype_label: Optional[int] = None,
                               raw_offset: Tuple[int, int, int] = (50, 50, 25),
                               pad_offset: int = 0, pad_value: int = 0,
                               ignore_offset: int = 0, ignore_value: int = -1,
                               n_closings: int = 0, n_dilations: int = 0)\
        -> Tuple[np.ndarray, np.ndarray]:
    """
    Retrieves the type segmentation data (0: background, 1: asymmetric, 2: symmetric)
    of a single 'syn_ssv' object. This function is used for sparse acquisition of 
    synapse type ground truth.
    
    Args:
        syn_ssv (SegmentationObject): The synapse supervoxel object used to fetch 
            the segmentation data.
        syntype_label (Optional[int], optional): If None, uses ``syn_sign`` stored 
            in ``syn_ssv.attr_dict`` and transforms the object segmentation into 
            the respective label (1: symmetric, 2: asymmetric). Defaults to None.
        raw_offset (Tuple[int, int, int], optional): Offset used for fetching the 
            raw data. Raw cube shape will be the segmentation cube shape + 
            2*raw_offset. Defaults to (50, 50, 25).
        pad_offset (int, optional): Number of voxels padded with 0-value around 
            the synapse segmentation. If `n_closings` is given, `pad_offset` will 
            be set to ``max([pad_offset, n_closings])``. Defaults to 0.
        pad_value (int, optional): Value used for padding. Defaults to 0.
        ignore_offset (int, optional): Number of voxels padded with `ignore_value` 
            around the padded synapse segmentation. Defaults to 0.
        ignore_value (int, optional): Value used for ignore-padding. Defaults to -1.
        n_closings (int, optional): Number of closings performed on the 
            segmentation. Defaults to 0.
        n_dilations (int, optional): Number of dilations performed before closing. 
            Defaults to 0.
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: Volumetric raw and segmentation data.
    """
    pad_offset = max([pad_offset, n_closings])
    raw_offset = np.array(raw_offset) + pad_offset + ignore_offset
    coord_raw = syn_ssv.bounding_box[0] - raw_offset
    size_raw = syn_ssv.bounding_box[1] - syn_ssv.bounding_box[0] + 2 * raw_offset
    segmentation = syn_ssv.voxels.astype(np.uint16)
    segmentation = np.pad(segmentation, pad_offset, 'constant',
                          constant_values=pad_value)  # volumetric binary mask
    if n_dilations > 0:
        segmentation = ndimage.binary_dilation(segmentation.astype(np.bool),
                                               iterations=n_dilations).astype(np.uint16)
    if n_closings > 0:
        segmentation = ndimage.binary_closing(segmentation.astype(np.bool),
                                              iterations=n_closings).astype(np.uint16)
    segmentation = np.pad(segmentation, ignore_offset, 'constant',
                          constant_values=ignore_value)
    kd = kd_factory(global_params.config.kd_seg_path)
    raw = kd.from_raw_cubes_to_matrix(size_raw, coord_raw)
    if syntype_label is None:
        syn_sign = syn_ssv.lookup_in_attribute_dict('syn_sign')
        if syn_sign is None:
            raise ValueError(f'Key "syn_sign" does not exist in AttributeDict of'
                             f' {str(syn_ssv)}.')
        syntype_label = 1 if syn_ssv.attr_dict["syn_sign"] == 1 else 2
    segmentation[segmentation == 1] = syntype_label
    return raw, segmentation


def fetch_single_synssv_typseg_enhanced(
        syn_ssv: SegmentationObject, pre_synapse,
        syntype_label: Optional[int] = None,
        raw_offset: Tuple[int, int, int] = (50, 50, 25), pad_offset: int = 0,
        pad_value: int = 0, ignore_offset: int = 0, ignore_value: int = -1,
        n_closings: int = 0, n_dilations: int = 0)\
        -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Retrieves the type segmentation data (0: background, 1: asymmetric, 2: symmetric) of a single 
    'syn_ssv' object. This function is used for sparse acquisition of synapse type ground truth.
    
    Args:
        syn_ssv (SegmentationObject): The synapse supervoxel object used to fetch the segmentation data.
        pre_synapse (int): ID of the presynaptic SSV object.
        syntype_label (Optional[int], optional): If None, uses ``syn_sign`` stored in ``syn_ssv.attr_dict``
            and transforms the object segmentation into the respective label (1: symmetric, 2: asymmetric).
            Defaults to None.
        raw_offset (Tuple[int, int, int], optional): Offset used for fetching the raw data. Raw cube shape 
            will be the segmentation cube shape + 2*raw_offset. Defaults to (50, 50, 25).
        pad_offset (int, optional): Number of voxels padded with 0-value around the synapse segmentation. 
            If `n_closings` is given, `pad_offset` will be set to ``max([pad_offset, n_closings])``. 
            Defaults to 0.
        pad_value (int, optional): Value used for padding. Defaults to 0.
        ignore_offset (int, optional): Number of voxels padded with `ignore_value` around the padded 
            synapse segmentation. Defaults to 0.
        ignore_value (int, optional): Value used for ignore-padding. Defaults to -1.
        n_closings (int, optional): Number of closings performed on the segmentation. Defaults to 0.
        n_dilations (int, optional): Number of dilations performed before closing. Defaults to 0.
    
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Volumetric raw and vector field pointing to the nearest 
        boundary pixels of the pre-synaptic cell and the mask of the entire synapse. The vector field is 
        set to 0 where no synapse is present.
    """
    pad_offset = max([pad_offset, n_closings])
    raw_offset = np.array(raw_offset) + pad_offset + ignore_offset
    coord_raw = syn_ssv.bounding_box[0] - raw_offset
    size_raw = syn_ssv.bounding_box[1] - syn_ssv.bounding_box[0] + 2 * raw_offset
    segmentation = syn_ssv.voxels.astype(np.uint16)
    segmentation = np.pad(segmentation, pad_offset, 'constant',
                          constant_values=pad_value)  # volumetric binary mask
    if n_dilations > 0:
        segmentation = ndimage.binary_dilation(segmentation.astype(np.bool),
                                               iterations=n_dilations).astype(np.uint16)
    if n_closings > 0:
        segmentation = ndimage.binary_closing(segmentation.astype(np.bool),
                                              iterations=n_closings).astype(np.uint16)
    segmentation = np.pad(segmentation, ignore_offset, 'constant',
                          constant_values=ignore_value)
    kd = kd_factory(global_params.config.kd_seg_path)
    raw = kd.from_raw_cubes_to_matrix(size_raw, coord_raw)

    # get the SSV IDs ordering given the PCA vector (ID1 points towards ID2)
    seg_cell = kd.from_overlaycubes_to_matrix(np.array(segmentation.shape),
                                              syn_ssv.bounding_box[0] -
                                              pad_offset - ignore_offset)
    syn_ssv.load_attr_dict()
    ssv_ids = syn_ssv.attr_dict['neuron_partners']
    if pre_synapse not in ssv_ids:
        raise ValueError('Pre-synaptic SSV ID is not present in the '
                         '"neuron_partners" attribute if the given synapse:'
                         f'{ssv_ids}.')
    ssd = SuperSegmentationDataset()
    sv_ids_present = np.unique(seg_cell)
    for ssv in ssd.get_super_segmentation_object(ssv_ids):
        for ix in ssv.sv_ids:
            if ix not in sv_ids_present:
                continue
            seg_cell[seg_cell == ix] = ssv.id

    # Find if vector connecting the two cells (3rd PC of the synapse) is
    # pointing towards cell1 or cell2 (< 0 means pointing away, > 0 pointing towards it)
    ssv1_seg = (seg_cell == pre_synapse).astype(np.float32)
    # erode SSV supervoxel
    iterations = 13
    ssv1_seg_tmp = ndimage.binary_erosion(ssv1_seg, iterations=iterations)
    while 1 not in ssv1_seg_tmp:
        iterations -= 1
        if iterations < 0:
            raise ValueError
        ssv1_seg_tmp = ndimage.binary_erosion(ssv1_seg, iterations=iterations)
    ssv1_seg = ssv1_seg_tmp
    assert 1 in ssv1_seg
    ssv1_seg = vigra.VigraArray((ssv1_seg != 1).astype(np.float32),
                                axistags=vigra.defaultAxistags('xyz'))
    # get the vectorial distances to the pre-synaptic supervoxels for all other voxels.
    vecdist = boundaryVectorDistanceTransform(ssv1_seg, boundary='OuterBoundary')
    vecdist = np.array(vecdist, dtype=np.float32)
    vecdist[(segmentation != 1)] = 0
    if syntype_label is None:
        syn_sign = syn_ssv.lookup_in_attribute_dict('syn_sign')
        if syn_sign is None:
            raise ValueError(f'Key "syn_sign" does not exist in AttributeDict of'
                             f' {str(syn_ssv)}.')
        syntype_label = 1 if syn_ssv.attr_dict["syn_sign"] == 1 else 2
    segmentation[segmentation == 1] = syntype_label
    return raw, vecdist, segmentation


def parse_gt_usable_synssv(mask_celltypes: bool = True,
                           synprob_thresh: float = 0.9):
    """
    Parses ground truth usable synapse supervoxels.
    
    Args:
        mask_celltypes (bool, optional): If True, filters inh. and exc. cells based 
            on celltype predictions. If False, returned synapse types are -1. 
            Defaults to True.
        synprob_thresh (float, optional): Minimum probability of synapse objects to 
            be not filtered. Defaults to 0.9.
    
    Returns:
        Tuple[List, List]: Two lists. One contains the 'syn_ssv' used to fetch the 
        raw and segmentation data and the other the synapse type (1: asymmetric, 
        2: symmetric).
    """
    syn_objs_total, syn_type_total = [], []
    sd_syn_ssv = SegmentationDataset('syn_ssv', working_dir=global_params.config.working_dir)
    syn_cts = sd_syn_ssv.load_numpy_data('partner_celltypes')
    syn_axs = sd_syn_ssv.load_numpy_data('partner_axoness')
    syn_prob = sd_syn_ssv.load_numpy_data('syn_prob')
    m_prob = syn_prob >= synprob_thresh
    # set bouton predictions to axon label
    syn_axs[syn_axs == 3] = 1
    syn_axs[syn_axs == 4] = 1
    # dict(STN=0, modulatory=1, MSN=2, LMAN=3, HVC=4, GP=5, INT=6)
    # asymmetric STN, HVC, LMAN synapses
    if mask_celltypes:
        m_exc = (syn_cts == 0) | (syn_cts == 3) | (syn_cts == 4)
    else:
        # use alle cells
        m_exc = np.ones_like(syn_cts, dtype=np.bool)
    # excitatory cell is pre-synaptic, high probability synapse, synapse must be on a
    # dendrite or soma
    pre_mask = np.any(m_exc & (syn_axs == 1), axis=1) & m_prob & \
               np.any((syn_axs == 0) | (syn_axs == 2), axis=1)
    pre_syns = sd_syn_ssv.get_segmentation_object(sd_syn_ssv.ids[pre_mask])
    syn_objs_total += pre_syns
    syn_type_total += [1] * len(pre_syns)

    # symmetric MSN, INT, TAN, FS synapses
    if mask_celltypes:
        m_inh = (syn_cts == 2) | (syn_cts == 5) | (syn_cts == 6)
    else:
        # this time set it to zero - all celltypes were already taken into account above
        m_inh = np.zeros_like(syn_cts, dtype=np.bool)
    # inhibitory cell is pre-synaptic, high probability synapse, synapse must be on a
    # dendrite or soma
    pre_mask = np.any(m_inh & (syn_axs == 1), axis=1) & m_prob & \
               np.any((syn_axs == 0) | (syn_axs == 2), axis=1)
    pre_syns = sd_syn_ssv.get_segmentation_object(sd_syn_ssv.ids[pre_mask])
    syn_objs_total += pre_syns
    syn_type_total += [2] * len(pre_syns)

    # TODO: Care about false negatives when using non-synaptic locations as GT due to additional
    #  raw offset -> Probably better to add true negative examples manually.
    # m_non_syn = sd_syn_ssv.get_segmentation_object(sd_syn_ssv.ids[(syn_prob < 0.1) \
    # & (sd_syn_ssv.sizes > 200)])
    # syn_objs_total += m_non_syn
    # syn_type_total += [0] * len(m_non_syn)

    if mask_celltypes is False:
        return syn_objs_total, np.ones_like(syn_type_total) * -1
    log_cnn.info('Gathered the following synapses: {}'.format(
        np.unique(syn_type_total, return_counts=True)))
    return syn_objs_total, syn_type_total
