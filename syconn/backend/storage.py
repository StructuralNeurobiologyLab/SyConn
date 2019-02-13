# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Philipp Schubert, Sven Dorkenwald, Joergen Kornfeld
from ..backend import log_backend
try:
    from lz4.block import compress, decompress
except ImportError:
    from lz4 import compress, decompress
try:  # TODO: check in global_params.py
    import fasteners
    LOCKING = True
except ImportError:
    print("fasteners could not be imported. Locking will be disabled by default."
          "Please install fasteners to enable locking (pip install fasteners).")
    LOCKING = False
import numpy as np

from ..handler.compression import lz4string_listtoarr, arrtolz4string_list
from ..backend import StorageClass


class AttributeDict(StorageClass):
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


class VoxelStorage(VoxelStorageL):
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
                                              "in this class. Use VoxelDictL" \
                                              "to enable locking."
        super(VoxelStorageL, self).__init__(inp, **kwargs)


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