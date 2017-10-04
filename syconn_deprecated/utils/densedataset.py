# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max-Planck-Institute for Medical Research, Heidelberg, Germany
# Authors: Sven Dorkenwald, Philipp Schubert, Joergen Kornfeld

import numpy as np

import densedataset_helper as ddh
from ..multi_proc import multi_proc_main as mpm
import segmentationdataset
import time


def apply_rallignment():
    pass


class DenseDataset(object):
    def __init__(self, path, path_segmentation, sv_mapping=None):
        self._path = path
        self._path_segmentation = path_segmentation
        self._ssv_dict = {}
        if sv_mapping:
            self._initialize_from_sv_mapping(sv_mapping)
        else:
            self._initialize()

        # Never stored - only loaded on request
        self._sv_dataset = None

    @property
    def ssv_ids(self):
        return self.ssv_dict.keys()

    @property
    def ssv_dict(self):
        return self._ssv_dict

    @property
    def sv_dataset(self):
        if not self._sv_dataset:
            self._sv_dataset = \
                segmentationdataset.load_dataset(self._path_segmentation)
        return self._sv_dataset

    def _initialize(self):
        for sv_id in self.ssv_ids:
            ssv = SuperSuperVoxelObject(sv_id)
            ssv._object_ids["sv"] = [sv_id]
            ssv._path_to_object_datasets["sv"] = self._path_segmentation
            self._ssv_dict[sv_id] = ssv

    def _initialize_from_sv_mapping(self, sv_mapping):
        for sv_id in sv_mapping.values():
            ssv = SuperSuperVoxelObject(sv_id)
            ssv._path_to_object_datasets["sv"] = self._path_segmentation
            self._ssv_dict[sv_id] = ssv

        for sv_id in sv_mapping.keys():
            self._ssv_dict[sv_mapping[sv_id]]._object_ids["sv"].append(sv_id)


class SuperSuperVoxelObject(object):
    def __init__(self, id):
        self._id = id
        self._object_ids = dict(mi=[], sj=[], vc=[], sv=[])
        self._cell_type = None
        self._compartment_type = None
        self._path_to_object_datasets = {}

        # Never stored - only loaded on request
        self._object_datasets = {}
        self._objects = {}
        self._voxels = []

    @property
    def id(self):
        return self._id

    @property
    def sv_ids(self):
        return self._object_ids["sv"]

    @property
    def sj_ids(self):
        return self._object_ids["sj"]

    @property
    def mi_ids(self):
        return self._object_ids["mi"]

    @property
    def vc_ids(self):
        return self._object_ids["vc"]

    @property
    def svs(self):
        return self._get_objects("sv")

    @property
    def sjs(self):
        return self._get_objects("sj")

    @property
    def mis(self):
        return self._get_objects("mi")

    @property
    def vcs(self):
        return self._get_objects("vc")

    @property
    def voxels(self):
        if not self._voxels:
            self._voxels = []
            for sv in self.svs:
                self._voxels += list(sv.voxels)
        return self._voxels

    @property
    def cell_type(self):
        return self._cell_type

    @property
    def compartment_type(self):
        return self._compartment_type

    @property
    def state(self):
        return

    def _get_objects(self, key):
        if key not in self._objects or not self._object[key]:
            self._objects[key] = []
            if key not in self._object_datasets:
                self._load_dataset(key)

            for sv_id in range(self.sv_ids):
                self._objects[key].append(
                    self._object_datasets[key].object_dict[sv_id])

        return self._objects[key]

    def _load_dataset(self, key):
        assert key in self._path_to_object_datasets
        self._object_datasets[key] = \
            segmentationdataset.load_dataset(self._path_to_object_datasets[key])

    def clear_cache(self):
        self._object_datasets = {}
        self._objects = {}
        self._voxels = []

