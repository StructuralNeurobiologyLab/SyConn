# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Philipp Schubert, Sven Dorkenwald, Joergen Kornfeld

import numpy as np
from collections import defaultdict
from typing import Any
from ..backend import log_backend
try:
    from lz4.block import compress, decompress
except ImportError:
    from lz4 import compress, decompress

from ..handler.compression import lz4string_listtoarr, arrtolz4string_list
from ..handler.basics import kd_factory
from ..backend import StorageClass


class AttributeDict(StorageClass):
    """
    General purpose dictionary class inherited from
    :class:`syconn.backend.base.StorageClass`.
    """
    def __init__(self, inp_p, **kwargs):
        super(AttributeDict, self).__init__(inp_p, **kwargs)

    def __getitem__(self, item):
        try:
            return self._dc_intern[item]
        except KeyError:
            self._dc_intern[item] = {}
            return self._dc_intern[item]

    def __setitem__(self, key, value):
        self._dc_intern[key] = value

    def update(self, other, **kwargs):
        self._dc_intern.update(other, **kwargs)

    def copy_intern(self):
        return dict(self._dc_intern)


class CompressedStorage(StorageClass):
    """
    Customized dictionary to store compressed numpy arrays, but with a
    intuitive user interface, i.e. compression will happen in background.
    kwarg 'cache_decomp' can be enabled to cache decompressed arrays
    additionally (save decompressing time when accessing items frequently).
    """
    def __init__(self, inp, **kwargs):
        super(CompressedStorage, self).__init__(inp, **kwargs)

    def __getitem__(self, item):
        try:
            return self._cache_dc[item]
        except KeyError:
            pass
        value_intern = self._dc_intern[item]
        sh = value_intern["sh"]
        dt = np.dtype(value_intern["dt"])
        decomp_arr = lz4string_listtoarr(value_intern["arr"], dtype=dt, shape=sh)
        if self._cache_decomp:
            self._cache_dc[item] = decomp_arr
        return decomp_arr

    def __setitem__(self, key, value):
        if type(value) is not np.ndarray:
            msg = "CompressedStorage supports np.array values only."
            log_backend.error(msg)
            raise ValueError(msg)
        if self._cache_decomp:
            self._cache_dc[key] = value
        sh = list(value.shape)
        sh[0] = -1
        value_intern = {"arr": arrtolz4string_list(value), "sh": tuple(sh),
                        "dt": value.dtype.str}
        self._dc_intern[key] = value_intern


class VoxelStorageL(StorageClass):
    """
    Customized dictionary to store compressed numpy arrays, but with a
    intuitive user interface, i.e. compression will happen in background.
    kwarg 'cache_decomp' can be enabled to cache decompressed arrays
    additionally (save decompressing time).
    """

    def __init__(self, inp, **kwargs):
        super(VoxelStorageL, self).__init__(inp, **kwargs)

    def __getitem__(self, item):
        """
        Parameters
        ----------
        item : int/str

        Returns
        -------
        list of np.array, list of np.array
            Decompressed voxel masks with corresponding offsets
        """
        try:
            return self._cache_dc[item], self._dc_intern[item]["off"]
        except KeyError:
            pass
        value_intern = self._dc_intern[item]
        dt = np.dtype(value_intern["dt"])
        sh = value_intern["sh"]
        offsets = value_intern["off"]
        comp_arrs = value_intern["arr"]
        decomp_arrs = []
        for i in range(len(sh)):
            decomp_arrs.append(lz4string_listtoarr(comp_arrs[i], dt, sh[i]))
        if self._cache_decomp:
            self._cache_dc[item] = decomp_arrs
        return decomp_arrs, offsets

    def __setitem__(self, key, values):
        """

        Parameters
        ----------
        key : int/str
            E.g. SO ID.
        values : list of np.array
            E.g. voxel masks
        """
        voxel_masks, offsets = values
        assert np.all([voxel_masks[0].dtype == v.dtype for v in voxel_masks])
        assert len(voxel_masks) == len(offsets)
        if self._cache_decomp:
            self._cache_dc[key] = voxel_masks
        sh = [v.shape for v in voxel_masks]
        for i in range(len(sh)):
            curr_sh = list(sh[i])
            curr_sh[0] = -1
            sh[i] = curr_sh
        value_intern = {"arr": [arrtolz4string_list(v) for v in voxel_masks],
                        "sh": sh, "dt": voxel_masks[0].dtype.str,
                        "off": offsets}
        self._dc_intern[key] = value_intern

    def append(self, key, voxel_mask, offset):
        value_intern = self._dc_intern[key]
        dt = np.dtype(value_intern["dt"])
        sh = value_intern["sh"]
        offsets = value_intern["off"] + [offset]
        comp_arrs = value_intern["arr"]

        assert dt == voxel_mask.dtype.str

        curr_sh = list(voxel_mask.shape)
        curr_sh[0] = -1
        sh.append(curr_sh)

        value_intern = {"arr": comp_arrs + [arrtolz4string_list(voxel_mask)],
                        "sh": sh, "dt": dt, "off": offsets}
        self._dc_intern[key] = value_intern


def VoxelStorage(inp, **kwargs):
    """
    Deprecated storage for voxel data.

    Args:
        inp:
        **kwargs:

    Returns:

    """
    obj = VoxelStorageClass(inp, **kwargs)
    if 'meta' in obj._dc_intern:  # TODO: Remove asap as soon as we switch to VoxelStorageDyn
        obj = VoxelStorageDyn(inp, **kwargs)
    # # TODO: activate as soon as synapse-detection pipelines is refactored.
    # else:
    #     log_backend.warning('VoxelStorage is deprecated. Please switch to'
    #                         ' VoxelStorageDyn.')
    return obj


class VoxelStorageClass(VoxelStorageL):
    """
    Customized dictionary to store compressed numpy arrays, but with a
    intuitive user interface, i.e. compression will happen in background.
    kwarg 'cache_decomp' can be enabled to cache decompressed arrays
    additionally (save decompressing time).

    No locking provided in this class!
    """

    def __init__(self, inp, **kwargs):
        if "disable_locking" in kwargs:
            assert kwargs["disable_locking"], "Locking must be disabled " \
                                              "in this class. Use VoxelDictL " \
                                              "to enable locking."
        super(VoxelStorageL, self).__init__(inp, **kwargs)


class VoxelStorageDyn(CompressedStorage):
    """
    Similar to `VoxelStorageL` but does not store the voxels explicitly,
    but the information necessary to query the voxels of an object.

    If ``voxel_mode = True`` getter method will operate on underlying data set
    to retrieve voxels of an object. `__setitem__` throws `RuntimeError`.
    `__getitem__` will return a list of 3D binary cubes with ones at the
    object's locations (key: object ID). Note: The item ID has to match the
    object ID in the segmentation.

    Otherwise (``voxel_mode = False``) `__getitem__` and `__setitem__` allow
    manipulation of the object's bounding box. In this case `voxeldata_path`
    has to be given or already be existent in loaded dictionary. Expects the
    source path of a KnossoDataset (see knossos_utils), like
        kd = KnossoDataset()
        kd.initialize_from_knossos_path(SOURCE_PATH)
    `__setitem__` requires the object ID as key and an 3 dimensional array with
     all bounding boxes defining the object (N, 2, 3). Those BBs are then used to
     query the object voxels. The bounding box is expected to be two 3D
     coordinates which define the lower and the upper limits.


    """

    def __init__(self, inp, voxel_mode=True, voxeldata_path=None, **kwargs):
        super().__init__(inp, **kwargs)
        self.voxel_mode = voxel_mode
        if not 'meta' in self._dc_intern:
            # add meta information about underlying voxel data set to internal dictionary
            self._dc_intern['meta'] = dict(voxeldata_path=voxeldata_path)
        if not 'size' in self._dc_intern:
            self._dc_intern['size'] = defaultdict(int)
        if not 'rep_coord' in self._dc_intern:
            self._dc_intern['rep_coord'] = dict()
        if voxeldata_path is not None:
            old_p = self._dc_intern['meta']['voxeldata_path']
            new_p = voxeldata_path
            if old_p != new_p:
                log_backend.warn('Overwriting `voxeldata_path` in `VoxelStorag'
                                 'eDyn` object (stored at "{}") '
                                 'from `{}` to `{}`.'.format(inp, old_p, new_p))
                self._dc_intern['meta']['voxeldata_path'] = voxeldata_path
        voxeldata_path = self._dc_intern['meta']['voxeldata_path']
        if voxel_mode:
            if voxeldata_path is None:
                msg = '`voxel_mode` is True but no path to' \
                      ' voxeldata given / found.'
                log_backend.error(msg)
                raise ValueError(msg)
            kd = kd_factory(voxeldata_path)
            self.voxeldata = kd

    def __setitem__(self, key: str, value: Any):
        if self.voxel_mode:
            raise RuntimeError('`VoxelStorageDyn.__setitem__` may only '
                               'be used when `voxel_mode=False`.')
        else:
            return super().__setitem__(key, value)

    def __getitem__(self, item):
        if self.voxel_mode:
            res = []
            bbs = super().__getitem__(item)
            for bb in bbs:  # iterate over all bounding boxes
                size = bb[1] - bb[0]
                off = bb[0]
                curr_mask = self.voxeldata.from_overlaycubes_to_matrix(
                    size, off, show_progress=False, verbose=False) == item
                res.append(curr_mask)
            return res, bbs[:, 0]  # N, 3 --> all offset
        else:
            return super().__getitem__(item)

    def object_size(self, item):
        if not self.voxel_mode:
            log_backend.warn('`object_size` sould only be called during `voxel_mode=True`.')
        if item not in self._dc_intern:
            raise KeyError('KeyError: Could not find key "{}" in `self._dc_intern`.`'.format(item))
        return self._dc_intern['size'][item]

    def increase_object_size(self, item, value):
        if self.voxel_mode:
            log_backend.warn('`increase_object_size` sould only be called when `voxel_mode=False`.')
        self._dc_intern['size'][item] += value

    def object_repcoord(self, item):
        if not self.voxel_mode:
            log_backend.warn('`object_repcoord` sould only be called when `voxel_mode=True`.')
        if item not in self._dc_intern:
            raise KeyError('KeyError: Could not find key "{}" in `self._dc_intern`.`'.format(item))
        return self._dc_intern['rep_coord'][item]

    def set_object_repcoord(self, item, value):
        if self.voxel_mode:
            log_backend.warn('`set_object_repcoord` sould only be called when `voxel_mode=False`.')
        self._dc_intern['rep_coord'][item] = value

    def get_voxeldata(self, item):
        old_vx_mode = self.voxel_mode
        self.voxel_mode = True
        if self._dc_intern['meta']['voxeldata_path'] is None:
            msg = '`voxel_mode` is True but no path to' \
                  ' voxeldata given / found.'
            log_backend.error(msg)
            raise ValueError(msg)
        kd = kd_factory(self._dc_intern['meta']['voxeldata_path'])
        self.voxeldata = kd
        res = self[item]
        self.voxel_mode = old_vx_mode
        return res

    def get_boundingdata(self, item):
        old_vx_mode = self.voxel_mode
        self.voxel_mode = False
        res = self[item]
        self.voxel_mode = old_vx_mode
        return res

    def keys(self):
        # do not return 'meta' and other helper items in self._dc_intern, only object IDs
        # TODO: make this a generator
        obj_elements = list([k for k in self._dc_intern.keys() if (type(k) is str and k.isdigit())
                             or (type(k) is not str)])
        return obj_elements


class MeshStorage(StorageClass):
    """
    Customized dictionary to store compressed numpy arrays, but with a
    intuitive user interface, i.e. compression will happen in background.
    kwarg 'cache_decomp' can be enabled to cache decompressed arrays
    additionally (save decompressing time).
    """

    def __init__(self, inp, load_colarr=False, **kwargs):
        self.load_colarr = load_colarr
        super(MeshStorage, self).__init__(inp, **kwargs)

    def __getitem__(self, item):
        """

        Parameters
        ----------
        item : int/str

        Returns
        -------
        List[np.arrays]
            [indices, vertices, normals, colors/labels]
        """
        try:
            return self._cache_dc[item]
        except KeyError:
            pass
        mesh = self._dc_intern[item]
        # if no normals were given in file / cache append empty array
        if len(mesh) == 2:
            mesh.append([""])
        # if no colors/labels were given in file / cache append empty array
        if len(mesh) == 3:
            mesh.append([""])
        decomp_arrs = [lz4string_listtoarr(mesh[0], dtype=np.uint32),
                       lz4string_listtoarr(mesh[1], dtype=np.float32),
                       lz4string_listtoarr(mesh[2], dtype=np.float32),
                       lz4string_listtoarr(mesh[3], dtype=np.uint8)]
        if not self.load_colarr:
            decomp_arrs = decomp_arrs[:3]
        if self._cache_decomp:
            self._cache_dc[item] = decomp_arrs
        return decomp_arrs

    def __setitem__(self, key, mesh):
        """

        Parameters
        ----------
        key : int/str
        mesh : List[np.array]
            [indices, vertices, normals, colors/labels]
        """
        if len(mesh) == 2:
            mesh.append(np.zeros((0, ), dtype=np.float32))
        if len(mesh) == 3:
            mesh.append(np.zeros((0, ), dtype=np.uint8))
        if self._cache_decomp:
            self._cache_dc[key] = mesh
        if len(mesh[2]) > 0 and len(mesh[1]) != len(mesh[2]):
            log_backend.warning('Lengths of vertex array and length of normal'
                                ' array differ!')
        # test if lengths of vertex and color array are identical or test
        # if vertex array length is equal to 3x label array length. Arrays are flattened.
        if len(mesh[3]) > 0 and not (len(mesh[1]) == len(mesh[3]) or
                                     len(mesh[1]) == len(mesh[3]) * 3):
            log_backend.warning('Lengths of vertex array and length of color/'
                                'label array differ!')
        comp_ind = arrtolz4string_list(mesh[0].astype(dtype=np.uint32))
        comp_vert = arrtolz4string_list(mesh[1].astype(dtype=np.float32))
        comp_norm = arrtolz4string_list(mesh[2].astype(dtype=np.float32))
        comp_col = arrtolz4string_list(mesh[3].astype(dtype=np.uint8))
        self._dc_intern[key] = [comp_ind, comp_vert, comp_norm, comp_col]


class SkeletonStorage(StorageClass):
    """
    Stores skeleton dictionaries (keys: "nodes", "diameters", "edges") as
    compressed numpy arrays.
    """

    def __init__(self, inp, **kwargs):
        super(SkeletonStorage, self).__init__(inp, **kwargs)

    def __getitem__(self, item):
        """

        Parameters
        ----------
        item : int/str

        Returns
        -------
        dict
        """
        try:
            return self._cache_dc[item]
        except KeyError:
            pass
        comp_arrs = self._dc_intern[item]
        skeleton = {"nodes": lz4string_listtoarr(comp_arrs[0], dtype=np.uint32),
                    "diameters": lz4string_listtoarr(comp_arrs[1], dtype=np.float32),
                    "edges": lz4string_listtoarr(comp_arrs[2], dtype=np.uint32)}
        if self._cache_decomp:
            self._cache_dc[item] = skeleton
        return skeleton

    def __setitem__(self, key, skeleton):
        """

        Parameters
        ----------
        key : int/str
        skeleton : dict
            keys: nodes diameters edges
        """
        if self._cache_decomp:
            self._cache_dc[key] = skeleton
        comp_n = arrtolz4string_list(skeleton["nodes"].astype(dtype=np.uint32))
        comp_d = arrtolz4string_list(skeleton["diameters"].astype(dtype=np.float32))
        comp_e = arrtolz4string_list(skeleton["edges"].astype(dtype=np.uint32))
        self._dc_intern[key] = [comp_n, comp_d, comp_e]
