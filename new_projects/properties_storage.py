# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Philipp Schubert, Sven Dorkenwald, Joergen Kornfeld, Maria Kawula

import numpy as np
import h5py
import os
import sys
from collections import defaultdict

try:
    from lz4.block import compress, decompress
except ImportError:
    from lz4 import compress, decompress


class PropertiesStorage:
    """
    Customized dictionary to store compressed numpy arrays, but with a
    intuitive user interface, i.e. compression will happen in background.
    kwarg 'cache_decomp' can be enabled to cache decompressed arrays
    additionally (save decompressing time when accessing items frequently).
    """

    def __init__(self, path, write=False):
        self.path = path

        if os.path.isfile(path) and write:
            os.remove(path)

        self.f = h5py.File(path, 'a')

    def __getitem__(self, item):
        try:
            return self.f[item]
        except KeyError:
            pass

    def save_properties(self, cs_props):

        grp = self.f.create_group(str(0))
        for key, val in cs_props[0].items():
            grp.create_dataset(str(key), data=val, compression="lzf")

        grp = self.f.create_group(str(1))
        for key, val in cs_props[1].items():
            if len(np.shape(val)) != 4:
                val = np.concatenate(val)
            grp.create_dataset(str(key), data=val, compression="lzf")

        grp = self.f.create_group(str(2))
        for key, val in cs_props[2].items():
            grp.create_dataset(str(key), data=val)
        del cs_props
        self.f.close()

    def read_properties(self, obj_keys):
        obj_keys = np.intersect1d(obj_keys, list(self.f['0'].keys()))

        out_dict = [{}, defaultdict(list), {}]
        for obj_key in obj_keys:
            out_dict[0][int(obj_key)] = self.f['0'][obj_key][:]
            out_dict[1][int(obj_key)] = self.f['1'][obj_key][:]
            out_dict[2][int(obj_key)] = self.f['2'][obj_key][()]

        return out_dict


class H5Dictionary:

    def __init__(self, path, write=False):
        self.path = path

        if os.path.isfile(path) and write:
            os.remove(path)
            self.f = h5py.File(path, 'a')
        if write is False:
            self.f = h5py.File(path, 'r')
        else:
            self.f = h5py.File(path, 'a')

    def save_properties(self, cs_props, compression=False):
        for key, key_str in zip(cs_props.keys(), map(str, cs_props.keys())):
            if compression:
                self.f.create_dataset(key_str, data=cs_props[key], compression="gzip")
            else:
                self.f.create_dataset(key_str, data=cs_props[key])
        self.f.close()

    def read_properties(self, obj_keys=None):
        data = {}
        if obj_keys is None:
            obj_keys = self.f.keys()
        else:
            obj_keys = np.intersect1d(obj_keys, list(self.f.keys()))

        for key in obj_keys:
            data[int(key)] = self.f[key][()]

        # self.f.close()
        return data















