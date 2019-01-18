# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max-Planck-Institute of Neurobiology, Munich, Germany
# Authors: Philipp Schubert, Sven Dorkenwald, Jörgen Kornfeld
# non-relative import needed for this file in order to be importable by
# ELEKTRONN2 architectures
import matplotlib
matplotlib.use("Agg", warn=False, force=True)
from elektronn2.data.traindata import Data
import numpy as np
import warnings
from syconn.config.global_params import wd
from syconn.handler.basics import load_pkl2obj, temp_seed
from syconn.handler.prediction import naive_view_normalization, naive_view_normalization_new
from syconn.reps.super_segmentation import SuperSegmentationDataset
from syconn.reps.segmentation import SegmentationDataset
from syconn.config import global_params
import os
try:
    from torch.utils.data import Dataset
    from elektronn3.data.transforms import Identity
    elektronn3_avail = True
except ImportError as e:
    elektronn3_avail = False
    Dataset = None
    Identity = None
from typing import Callable
import h5py
from scipy import spatial
import time
# fix random seed.
np.random.seed(0)


# -------------------------------------- elektronn3 ----------------------------
if elektronn3_avail:
    class MultiviewDataSpines(Dataset):
        """
        Multiview spine data loader.
        """
        def __init__(
                self,
                inp_path=None,
                target_path=None,
                train=True,
                inp_key='raw', target_key='label',
                transform: Callable = Identity()
        ):
            super().__init__()
            cube_id = "train" if train else "valid"
            if inp_path is None or target_path is None:
                base_dir = os.path.expanduser("~") + "/spine_gt_multiview/"
                inp_path = os.path.expanduser('{}raw_{}_v2.h5'.format(base_dir, cube_id))
                target_path = os.path.expanduser('{}label_{}_v2.h5'.format(base_dir, cube_id))
            self.inp_file = h5py.File(os.path.expanduser(inp_path), 'r')
            self.target_file = h5py.File(os.path.expanduser(target_path), 'r')
            self.inp = self.inp_file[inp_key].value
            self.inp = self.inp[:, :4].astype(np.float32) / 255.
            self.target = self.target_file[target_key].value.astype(np.int64)
            self.target = self.target[:, 0]
            self.close_files()
            self.transform = transform
            print("Dataset ({}): {}\t{}".format(cube_id, self.inp.shape,
                                                np.unique(self.target,
                                                return_counts=True)))

        def __getitem__(self, index):
            inp = self.inp[index]
            target = self.target[index]
            inp, target = self.transform(inp, target)
            return inp, target

        def __len__(self):
            return self.target.shape[0]

        def close_files(self):
            self.inp_file.close()
            self.target_file.close()


    class MultiviewData_TNet_online(Dataset):
        """
        Multiview triplet net data loader.
        """

        def __init__(
                self, working_dir='/wholebrain/scratch/areaxfs3/',
                train=True, epoch_size=40000, allow_close_neigh=0,
                transform: Callable = Identity(),
        ):
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
            # use these classes to load label and splitting dicts
            AV = AxonViews(None, None, raw_only=False, nb_views=2,
                           naive_norm=False, load_data=False)

            CTV = CelltypeViews(load_data=False, naive_norm=False)
            # now link actual data
            self.ssd = SuperSegmentationDataset(working_dir, version='tnetgt')

            if train:
                self.inp = [self.ssd.get_super_segmentation_object(ix) for ix in AV.splitting_dict["train"]] + \
                           [self.ssd.get_super_segmentation_object(ix) for ix in CTV.splitting_dict["train"]]
                self._inp_ssv_ids = self.inp.copy()
                if self.allow_close_neigh:
                    self.inp_locs = []
                    for ssv in self.inp:
                        ssv.load_attr_dict()
                        self.inp_locs.append(np.concatenate(ssv.sample_locations(verbose=True)))
            else:
                self.inp = [self.ssd.get_super_segmentation_object(ix) for ix in AV.splitting_dict["valid"]] + \
                           [self.ssd.get_super_segmentation_object(ix) for ix in CTV.splitting_dict["valid"]]
                self._inp_ssv_ids = self.inp.copy()
                if self.allow_close_neigh:
                    self.inp_locs = []
                    for ssv in self.inp:
                        ssv.load_attr_dict()
                        self.inp_locs.append(np.concatenate(ssv.sample_locations(verbose=True)))
                self.inp = [ssv.load_views(view_key="raw2") for ssv in self.inp]
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
                    views = ssv.load_views(view_key="raw2")
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
                    views_dist = ssv.load_views(view_key="raw2")
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
            if self.train:
                return self.epoch_size
            else:
                return 1000

        def close_files(self):
            return


# -------------------------------------- ELEKTRONN2 ----------------------------
class MultiViewData(Data):
    def __init__(self, working_dir, gt_type, nb_cpus=20,
                 label_dict=None, view_kwargs=None, naive_norm=True,
                 load_data=True, train_fraction=None):
        if view_kwargs is None:
            view_kwargs = dict(raw_only=False, cache_default_views=True,
                               nb_cpus=nb_cpus, ignore_missing=True,
                               force_reload=False)
        self.gt_dir = working_dir + "/ssv_%s/" % gt_type
        if label_dict is None:
            self.label_dict = load_pkl2obj(self.gt_dir +
                                           "%s_labels.pkl" % gt_type)
        if not os.path.isfile(self.gt_dir + "%s_splitting.pkl" % gt_type):
            print("Splitting file not found. Splitting data accoring to {:.2f}"
                  " (train) - {:.2f} (valid).".format(train_fraction, 1 - train_fraction))
            if train_fraction is None:
                train_fraction = 0.85
            with temp_seed(0):
                ixs = np.arange(len(self.label_dict))
                np.random.shuffle(ixs)
            ssv_ids = list(self.label_dict.keys())
            split_ix = int(len(ixs) * train_fraction)
            splits = (ssv_ids[:split_ix], ssv_ids[split_ix:])
            self.splitting_dict = {"train": splits[0], "valid": splits[1],
                                   "test": []}
            print('Validation set: {}\t{}'.format(self.splitting_dict['valid'],
                                              [self.label_dict[ix] for ix in self.splitting_dict['valid']]))
        else:
            if train_fraction is not None:
                raise ValueError('Value fraction can only be set if splitting dict is not available.')
            self.splitting_dict = load_pkl2obj(self.gt_dir +
                                               "%s_splitting.pkl" % gt_type)
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
    def __init__(self, inp_node, out_node, gt_type="axgt", working_dir=wd,
                 nb_views=2, reduce_context=0, channels_to_load=(0, 1, 2, 3),
                 reduce_context_fact=1, binary_views=False, raw_only=False,
                 nb_cpus=20, **kwargs):
        super(AxonViews, self).__init__(working_dir, gt_type,
                                        nb_cpus=nb_cpus, **kwargs)
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
        if source == 'valid':
            nb = len(self.valid_l)
            ixs = np.arange(nb)
            np.random.shuffle(ixs)
            self.valid_d = self.valid_d[ixs]
            self.valid_l = self.valid_l[ixs]
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
    def __init__(self, inp_node, out_node, raw_only=False, nb_views=20,
                 reduce_context=0, binary_views=False, reduce_context_fact=1,
                 load_data=False, nb_cpus=1):
        # TODO: current cache size is not related to max_cache_uses nor batch_size
        self.view_key = "raw{}".format(2)
        self.nb_views = nb_views
        self.nb_cpus = nb_cpus
        self.raw_only = raw_only
        self.reduce_context = reduce_context
        self.max_nb_cache_uses = 2000
        self.current_cache_uses = 0
        self.view_cache = {'train': None, 'valid': None, 'test': None}
        self.label_cache = {'train': None, 'valid': None, 'test': None}
        self.reduce_context_fact = reduce_context_fact
        self.binary_views = binary_views
        self.example_shape = (nb_views, 4, 2, 128, 256)
        print("Initializing CelltypeViews:", self.__dict__)
        super().__init__(global_params.wd, "ctgt", train_fraction=0.95,
                         naive_norm=False, load_data=load_data)
        ssv_splits = self.splitting_dict
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

    def getbatch_alternative(self, batch_size, source='train'):
        """
        Preliminary tests showed inferior performance of models trained with sampling
        batches with this method compared to "getbatch" below. Might be due to less
        stochasticity (bigger cache)

        Parameters
        ----------
        batch_size :
        source :

        Returns
        -------

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
            sample_fac = np.max([int(self.nb_views / 20), 1])  # draw more ssv if number of views is high
            nb_ssv = 12 * sample_fac  # 1 for each class
            sample_ixs = []
            l = []
            labels2draw = np.arange(4)
            class_sample_weight = np.array([2, 2, 1, 1])
            np.random.shuffle(labels2draw)  # change order
            for i in labels2draw:
                curr_nb_samples = nb_ssv // 4 * class_sample_weight[i]  # sample more EA and MSN
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
            self.view_cache[source] = [sso.load_views(view_key="raw{}".format(2)) for sso in ssos]
            self.label_cache[source] = l
            # draw big cache batch from current SSOs from which training batches are drawn
            self.view_cache[source], self.label_cache[source] = transform_celltype_data_views(
                self.view_cache[source], self.label_cache[source], 2000, self.nb_views, norm_func=naive_view_normalization_new)
            self.current_cache_uses = 0
        ixs = np.arange(len(self.view_cache[source]))
        with temp_seed(None):
            np.random.shuffle(ixs)
        ixs = ixs[:batch_size]
        d, l = self.view_cache[source][ixs], self.label_cache[source][ixs]
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
            nb_ssv = 12 * sample_fac  # 1 for each class
            sample_ixs = []
            l = []
            labels2draw = np.arange(4)
            class_sample_weight = np.array([2, 2, 1, 1])
            np.random.shuffle(labels2draw)  # change order
            for i in labels2draw:
                curr_nb_samples = nb_ssv // 4 * class_sample_weight[i]  # sample more EA and MSN
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
            self.view_cache[source] = [sso.load_views(view_key="raw{}".format(2)) for sso in ssos]
            self.label_cache[source] = l
            self.current_cache_uses = 0
        ixs = np.arange(len(self.view_cache[source]))
        with temp_seed(None):
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


class GliaViews(Data):
    def __init__(self, inp_node, out_node, raw_only=True, nb_views=2,
                 reduce_context=0, binary_views=False, reduce_context_fact=1,
                 naive_norm=True):
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
                       naive_norm=naive_norm)
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
        if source == 'valid':
            nb = len(self.valid_l)
            ixs = np.arange(nb)
            np.random.shuffle(ixs)
            self.valid_d = self.valid_d[ixs]
            self.valid_l = self.valid_l[ixs]
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
        curr_nb_samples = int(np.min([np.floor(views.shape[1]/nb_views), batch_size-cnt, batch_size//4]))
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
    """Using neighboring location for small distance sample"""
    def __init__(self, input_node, target_node):
        self.sds = SegmentationDataset("sv", working_dir="/wholebrain/scratch/areaxfs3/",
                                       version=0)
        ssds = SuperSegmentationDataset(working_dir="/wholebrain/scratch/areaxfs3/")
        ssds.load_mapping_dict()
        rev_dc = {}
        for k, v in ssds.mapping_dict.items():
            for el in v:
                rev_dc[el] = k
        self.s_ids = np.concatenate(ssds.mapping_dict.values())
        bb = ssds.load_cached_data("bounding_box")
        # sizes as diagonal of bounding box in um (SV size will be size of corresponding SSV)
        bb_size = np.linalg.norm((bb[:, 1] - bb[: ,0])*self.sds.scaling, axis=1) / 1e3
        ssds_sizes = {}
        for i in range(len(ssds.ssv_ids)):
            ssds_sizes[ssds.ssv_ids[i]] = bb_size[i]
        sizes = np.array([ssds_sizes[rev_dc[ix]] for ix in self.s_ids], dtype=np.float32)
        # assign weights: 0 for SV below 8, 1 for 8 to 18, 2 for 19 to 28, truncated at 5 for 49 to 58
        self.s_weights = (sizes - 8.) / 10.
        self.s_weights *= np.array(self.s_weights > 0, dtype=np.int)
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
    """Uses orthogonal views of SSVs to generate set of three views:
    one as reference, one as similar and one as different view."""
    def __init__(self, input_node, target_node, nb_cpus=1,
                 raw_only=False, downsample=1):
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
        bb = self.ssds.load_cached_data("bounding_box")
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
        self.sso_weights *= np.array(self.sso_weights > 0, dtype=np.int)
        self.sso_weights = np.ceil(self.sso_weights)
        self.sso_weights[self.sso_weights > 6] = 6
        self.sso_ids = np.array(self.sso_ids, dtype=np.uint32)
        super(TripletData_SSV, self).__init__()
        print("Initializing SSV Data:", self.__repr__())

    def getbatch(self, batch_size, source='train'):
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
                        print("Couldn't retrieve enough samples for batch, chance of sampling SSVs multiple times.")
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
    """Uses 'nb_views', randomly sampled views as SSV representation"""
    def __init__(self, input_node, target_node, nb_views=20, nb_cpus=1,
                 raw_only=False):
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
        bb = self.ssds.load_cached_data("bounding_box")
        for sso_ix in self.valid_d:
            ix = self.sso_ids.index(sso_ix)
            bb[ix] = 0  # bb below 8, i.e. also 0, will be ignored during training
        for sso_ix in self.test_d:
            ix = self.sso_ids.index(sso_ix)
            bb[ix] = 0  # bb below 8, i.e. also 0, will be ignored during training

        # sizes as diagonal of bounding box in um (SV size will be size of corresponding SSV)
        bb_size = np.linalg.norm((bb[:, 1] - bb[: ,0])*self.sds.scaling, axis=1) / 1e3
        ssds_sizes = np.zeros((len(self.ssds.ssv_ids)))
        for i in range(len(self.ssds.ssv_ids)):
            ssds_sizes[i] = bb_size[i]
        # assign weights: 0 for SV below 8, 1 for 8 to 18, 2 for 19 to 28, truncated at 5 for 49 to 58
        self.sso_weights = (ssds_sizes - 8.) / 10.
        self.sso_weights *= np.array(self.sso_weights > 0, dtype=np.int)
        self.sso_weights = np.ceil(self.sso_weights)
        self.sso_weights[self.sso_weights > 6] = 6
        self.sso_ids = np.array(self.sso_ids, dtype=np.uint32)
        super(TripletData_SSV_nviews, self).__init__()
        print("Initializing SSV Data:", self.__repr__())

    def getbatch(self, batch_size, source='train'):
        while True:
            try:
                self._reseed()
                # the number of views (one for reference and one set for similar N-view set, i.e. different views from the same ssv
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
                      "%s\n"  % e)


def transform_tripletN_data_SSV(orig_views):
    """
    Parameters
    ----------
    orig_views : np.array
        shape: (batch size, nb channels, nb views, x, y)
    Returns
    -------
    np.array
        same shape as orig_views, but with with 50% more views, i.e. if nb views
        was initially 6, then it is assumend that 3 and 3 views are used for the similar pair
        and 3 random views will be added from a random different sample in this batch
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

    Parameters
    ----------
    ssv_id : int
        Supersupervoxel ID
    gt_type : str
        e.g. 'axgt'
    set_type : str
        either one of: 'train', 'valid', 'test'
    Returns
    -------

    """
    # retrieve SSV from original SSD (which is used in Knossos-Plugin) and
    # copy its data to the yet empty SSV in the axgt SSD.
    ssd_axgt = SuperSegmentationDataset(version=gt_type, working_dir=wd)
    ssd = SuperSegmentationDataset(working_dir=wd)
    ssv = ssd.get_super_segmentation_object(ssv_id)
    ssv_axgt = ssd_axgt.get_super_segmentation_object(ssv_id)
    ssv.copy2dir(ssv_axgt.ssv_dir)
    # add entries to label and splitting dict
    base_dir = "{}/ssv_{}/".format(wd, gt_type)
    splitting = load_pkl2obj("{}/axgt_splitting.pkl".format(base_dir))
    labels = load_pkl2obj("{}/axgt_labels.pkl".format(base_dir))
    splitting[set_type].append(ssv_id)
    labels[ssv_id] = label