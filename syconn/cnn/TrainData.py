# NeuroPatch
# Copyright (c) 2016 Philipp J. Schubert
# All rights reserved
# non-relative import needed for this file in order to be importable by
# ELEKTRONN2 architectures
import matplotlib
matplotlib.use("Agg")
from elektronn2.data.traindata import Data
import numpy as np
import warnings
from syconn.config.global_params import wd
from syconn.handler.basics import load_pkl2obj
from syconn.handler.compression import lz4stringtoarr
from syconn.reps.super_segmentation import SuperSegmentationDataset
from syconn.reps.segmentation import SegmentationDataset
from syconn.mp.shared_mem import start_multiprocess_obj


class MultiViewData(Data):
    def __init__(self, working_dir, gt_type, raw_only=False, nb_cpus=20):
        self.gt_dir = working_dir + "/ssv_%s/" % (gt_type)
        self.label_dict = load_pkl2obj(self.gt_dir + "%s_labels.pkl" % gt_type)
        self.splitting_dict = load_pkl2obj(self.gt_dir + "%s_splitting.pkl" % gt_type)
        self.ssd = SuperSegmentationDataset(working_dir, version=gt_type)

        if 1 and gt_type == "axgt" and "areaxfs3" in working_dir:
            print("Added contradictory sample to AxonGT.")
            self.label_dict[24479744] = 0
            self.splitting_dict["train"].append(24479744)

        # train data
        # get views of each SV
        self.train_d = [self.ssd.get_super_segmentation_object(ix).load_views(
            raw_only=raw_only, cache_default_views=True, nb_cpus=nb_cpus, ignore_missing=True, force_reload=False)
            for ix in self.splitting_dict["train"]]
        # create labels for SV views according to SSV label, assuming pure SSV compartments
        self.train_l = np.concatenate([[self.label_dict[ix]] * len(self.train_d[ii])
        for ii, ix in enumerate(self.splitting_dict["train"])]).astype(np.uint16)[:, None]
        # concatenate raw data
        self.train_d = np.concatenate(self.train_d)
        # perform pseudo-normalization (proper normaalization: how to store mean and std for inference?)
        if not (np.all(0 <= self.train_d) and np.all(self.train_d <= 1.0)):
            self.train_d = self.train_d.astype(np.float32) / 255 - 0.5
            print("Performing normalization of train.-views. Double check that "
                  "all view data ranges from 0...255 originally.")
        else:
            self.train_d = self.train_d.astype(np.float32) - 0.5
        # set example shape for parent class 'Data'
        self.example_shape = self.train_d[0].shape
        # valid data
        self.valid_d = [self.ssd.get_super_segmentation_object(ix).load_views(
            raw_only=raw_only, cache_default_views=True, nb_cpus=nb_cpus, ignore_missing=True, force_reload=False)
            for ix in self.splitting_dict["valid"]]
        self.valid_l = np.concatenate([[self.label_dict[ix]] * len(self.valid_d[ii])
        for ii, ix in enumerate(self.splitting_dict["valid"])]).astype(np.uint16)[:, None]
        self.valid_d = np.concatenate(self.valid_d)
        # perform pseudo-normalization (proper normaalization: how to store mean and std for inference?)
        if not (np.all(0 <= self.valid_d) and np.all(self.valid_d <= 1.0)):
            self.valid_d = self.valid_d.astype(np.float32) / 255 - 0.5
            print("Performing normalization of valid.-views. Double check that "
                  "all view data ranges from 0...255 originally.")
        else:
            self.valid_d = self.valid_d.astype(np.float32) - 0.5
        # test data
        if len(self.splitting_dict["test"]) > 0:
            self.test_d = [self.ssd.get_super_segmentation_object(ix).load_views(
                raw_only=raw_only, cache_default_views=True, nb_cpus=nb_cpus, ignore_missing=True, force_reload=False)
                for ix in self.splitting_dict["test"]]
            self.test_l = np.concatenate(
                [[self.label_dict[ix]] * len(self.test_d[ii])
                 for ii, ix in enumerate(self.splitting_dict["test"])]).astype(
                np.uint16)[:, None]
            self.test_d = np.concatenate(self.test_d)
        else:
            self.test_d = np.zeros((0, ), dtype=np.float32)
            self.test_l = np.zeros((0, ), dtype=np.uint16)
        # perform pseudo-normalization (proper normaalization: how to store mean and std for inference?)
        if not (np.all(0 <= self.test_d) and np.all(self.test_d <= 1.0)):
            self.test_d = self.test_d.astype(np.float32) / 255 - 0.5
            print("Performing normalization of test.-views. Double check that "
                  "all view data ranges from 0...255 originally.")
        else:
            self.test_d = self.test_d.astype(np.float32) - 0.5
        print("GT splitting:", self.splitting_dict)
        print "\nlabels (train) - 0:%d\t1:%d\t2:%d" % (
            np.sum(self.train_l == 0),
            np.sum(self.train_l == 1),
            np.sum(self.train_l == 2))
        print "labels (valid) - 0:%d\t1:%d\t2:%d" % (
            np.sum(self.valid_l == 0),
            np.sum(self.valid_l == 1),
            np.sum(self.valid_l == 2))
        super(MultiViewData, self).__init__()


class AxonViews(MultiViewData):
    def __init__(self, input_node, target_node, gt_type="axgt", working_dir=wd,
                 nb_views=2, reduce_context=0, channels_to_load=(0, 1, 2, 3),
                 reduce_context_fact=1, binary_views=False, raw_only=False, nb_cpus=20):
        super(AxonViews, self).__init__(working_dir, gt_type, raw_only, nb_cpus=nb_cpus)
        self.nb_views = nb_views
        self.reduce_context = reduce_context
        self.reduce_context_fact = reduce_context_fact
        self.channels_to_load = channels_to_load
        self.binary_views = binary_views
        self.nb_views = nb_views
        self.raw_only = raw_only
        self.example_shape = (nb_views, 4, 2, 128, 256)
        print "Initializing AxonViews:", self.__repr__()
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
        return d, l


class GliaViews(Data):
    def __init__(self, input_node, target_node, channels_to_load=(0, ),
                 nb_views=2, glia_only=False, augmentation=False, clahe=False,
                 train_all=False, squeeze=True, reduce_context=0, binary_views=False,
                 reduce_context_fact=1):
        self.nb_views = nb_views
        self.reduce_context = reduce_context
        self.reduce_context_fact = reduce_context_fact
        self.channels_to_load = channels_to_load
        self.glia_only = glia_only
        self.clahe = clahe
        self.binary_views = binary_views
        self.augmentation = augmentation
        print "Initializing GliaViews:", self.__dict__
        ax_gt_dir = wd + "/gt/gt_axoness/"

        if clahe:
            # load glia views
            raise(NotImplementedError)
            self.glia_dict = load_pkl2obj(
                wd + "/gt/gt_gliacells/views/glia_dict_v2_withclahe.pkl")
        else:
            # load glia views
            self.glia_dict = load_pkl2obj(
                wd + "/gt/gt_gliacells/views/glia_dict_v2.pkl")

        nonglia_train_d, nonglia_train_l, nonglia_valid_d, nonglia_valid_l, \
        nonglia_test_d, nonglia_test_l = load_axon_gt(channels_to_load, nb_views,
                                                      version="6with2axonmergers", squeeze=squeeze,
                                                      train_all=train_all,
                                                      new_squeeze=False)
        nb_nonglia_train = len(nonglia_train_d)
        nb_nonglia_valid = len(nonglia_valid_d)
        nb_nonglia_test = len(nonglia_test_d)
        nb_ch = len(channels_to_load)
        print "%d training and %d validation samples for non glia. " \
              "Decompressing glia samples." % (nb_nonglia_train, nb_nonglia_valid)
        nb_glia = 0
        for k in self.glia_dict.keys():
            decomp_arr = lz4stringtoarr(self.glia_dict[k], shape=(-1, nb_ch, nb_views, 128, 256), dtype=np.float64)[:, :, :nb_views].astype(np.float32)
            self.glia_dict[k] = decomp_arr
            nb_glia += len(decomp_arr)
        glia_samples = np.zeros((nb_glia, nb_ch, nb_views, 128, 256), dtype=np.float32)
        cnt = 0
        for decomp_arr in self.glia_dict.itervalues():
            glia_samples[cnt:(cnt+len(decomp_arr))] = decomp_arr
            cnt += len(decomp_arr)
        glia_boarder = int(nb_glia * 0.9)
        nb_valid = nb_nonglia_valid+(nb_glia-glia_boarder)
        nb_train = nb_nonglia_train+glia_boarder
        self.train_d = np.zeros((nb_train, nb_ch, 2, 128, 256), dtype=np.float32)
        self.train_l = np.zeros((nb_train, 1), dtype=np.int16)
        self.train_d[:nb_nonglia_train] = nonglia_train_d
        self.train_l[:nb_nonglia_train] = nonglia_train_l
        del nonglia_train_d
        del nonglia_train_l

        self.valid_d = np.zeros((nb_valid, nb_ch, 2, 128, 256), dtype=np.float32)
        self.valid_l = np.zeros((nb_valid, 1), dtype=np.int16)
        self.valid_d[:nb_nonglia_valid] = nonglia_valid_d
        self.valid_l[:nb_nonglia_valid] = nonglia_valid_l
        del nonglia_valid_d
        del nonglia_valid_l

        self.test_d = np.zeros((0, nb_ch, nb_views, 128, 256), dtype=np.float32)
        self.test_l = np.zeros((0, 1), dtype=np.int16)

        self.train_d[nb_nonglia_train:] = glia_samples[:glia_boarder]
        self.train_l[nb_nonglia_train:] = 3

        self.valid_d[nb_nonglia_valid:] = glia_samples[glia_boarder:]
        self.valid_l[nb_nonglia_valid:] = 3
        if self.glia_only:
            self.train_l[self.train_l != 3] = 0
            self.train_l[self.train_l == 3] = 1
            self.valid_l[self.valid_l != 3] = 0
            self.valid_l[self.valid_l == 3] = 1
            print "\nlabels (train) - 0:%d\t1:%d" % (np.sum(self.train_l==0),
                                                         np.sum(self.train_l == 1))
            print "labels (valid) - 0:%d\t1:%d" % (np.sum(self.valid_l==0),
                                                         np.sum(self.valid_l == 1))
        else:
            print "\nlabels (train) - 0:%d\t1:%d\t2:%d\t3:%d" % (
            np.sum(self.train_l == 0),
            np.sum(self.train_l == 1),
            np.sum(self.train_l == 2),
            np.sum(self.train_l == 3))
            print "labels (valid) - 0:%d\t1:%d\t2:%d\t3:%d" % (
            np.sum(self.valid_l == 0),
            np.sum(self.valid_l == 1),
            np.sum(self.valid_l == 2),
            np.sum(self.valid_l == 3))
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
        d = d[:, :, view_shuffle[:self.nb_views]]
        if self.augmentation:
            d = _augmentViews(d)
        if self.binary_views:
            d[d < 1.0] = 0
        return d, l


class SSVCelltype(Data):
    """Uses N-views to represent SSVs and perform supervised classification of cell types"""
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
        super(SSVCelltype, self).__init__()
        print "Initializing SSV Data:", self.__repr__()

    def getbatch(self, batch_size, source='train'):
        self._reseed()
        if source == 'valid':
            nb = len(self.valid_l)
            ixs = np.arange(nb)
            np.random.shuffle(ixs)
            self.valid_d = self.valid_d[ixs]
            self.valid_l = self.valid_l[ixs]
        sample_fac = np.max([int(self.nb_views / 20), 1]) # draw more ssv if number of views is high
        nb_ssv = batch_size * sample_fac
        sample_ixs = []
        l = []
        for i in range(4):
            curr_nb_samples = nb_ssv // 4
            if source == "train":
                sample_ixs.append(np.random.choice(self.train_d[self.train_l==i],
                                               curr_nb_samples, replace=True).tolist())
                l += [[i]*curr_nb_samples]
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
        ssos = []
        sample_ixs = np.concatenate(zip(*sample_ixs))
        l = np.concatenate(zip(*l))
        for ix in sample_ixs:
            sso = self.ssds.get_super_segmentation_object(ix)
            sso.nb_cpus = self.nb_cpus
            ssos.append(sso)
        out_d, l = transform_celltype_data(ssos, l, batch_size, self.nb_views)
        if self.raw_only:
            return out_d[:, :1], l
        return out_d, l


def transform_celltype_data(ssos, labels, batch_size, nb_views, nb_cpus=1):
    orig_views = np.zeros((batch_size, 4, nb_views, 128, 256), dtype=np.float32)
    new_labels = np.zeros((batch_size, 1), dtype=np.int16)
    cnt = 0
    sample_fac_sv = np.max([int(nb_views / 10), 1]) # draw more SV if #views is high
    for ii, sso in enumerate(ssos):
        sso.load_attr_dict()
        sample_svs = np.random.choice(list(sso.svs), np.min([nb_views*sample_fac_sv, len(sso.sv_ids)]), replace=False)
        views = np.concatenate(start_multiprocess_obj("load_views", [[sv, ] for sv in sample_svs], nb_cpus=nb_cpus))
        # views = np.concatenate(sso.load_views())
        sso.clear_cache()
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
        print "--------------------------------------" \
              "Number of views in batch is zero. " \
              "Missing views and labels were filled with 0." \
              "--------------------------------------"
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
    return orig_views, new_labels


class TripletData_N(Data):
    """Using neighboring location for small distance sample"""
    def __init__(self, input_node, target_node):
        self.sds = SegmentationDataset("sv", working_dir="/wholebrain/scratch/areaxfs/",
                                       version=0)
        rev_dc = load_pkl2obj("/wholebrain/scratch/pschuber/NeuroPatch/datasets/rev_cc_dict_ssv6.pkl")
        ssds = SuperSegmentationDataset(working_dir="/wholebrain/scratch/areaxfs/", version="6")
        ssds.load_mapping_dict()
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
        print "Data Summary:\nweight\t#samples"
        for i in range(7):
            print "%d\t%d" % (i, np.sum(self.s_weights == i))
        print "Using %d SV (weight bigger than 0)." % (np.sum(self.s_weights > 0))
        example = self.sds.get_segmentation_object(self.s_ids[1])
        self.example_shape = example.load_views()[0].shape
        self.train_d = np.array(self.s_ids)
        self.valid_d = np.zeros((0, ))
        self.test_d = np.zeros((0, ))
        print "Samples (train):", self.train_d.shape
        print "Samples (valid):", self.valid_d.shape
        print "Samples (test):", self.test_d.shape
        self.train_l = np.zeros((len(self.train_d), 1))
        self.test_l = np.zeros((len(self.test_d), 1))
        self.valid_l = np.zeros((len(self.valid_d), 1))
        super(TripletData_N, self).__init__()
        print "Initializing SSV Data:", self.__repr__()

    def getbatch(self, batch_size, source='train'):
        if source != "train":
            print "Does not have valid and test datasets, returning batch " \
                  "from training pool."
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
        return out_d, None


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
        print "Initializing SSV Data:", self.__repr__()

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
                            except ValueError:
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
        print "Initializing SSV Data:", self.__repr__()

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


def _augmentViews(data, crop=(4, 8)):
    """
    Creates new data, by cropping/shifting data.
    """
    for i in range(data.shape[0]):
        v = data[i]
        crop0a = np.random.randint(0, crop[0])
        crop0b = np.random.randint(0, crop[0])
        crop1a = np.random.randint(0, crop[1])
        crop1b = np.random.randint(0, crop[1])
        new = np.ones_like(v)
        new[:, :, crop0a:-crop0b, crop1a:-crop1b] = v[:, :, crop0a:-crop0b, crop1a:-crop1b]
        data[i] = new
    return data


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