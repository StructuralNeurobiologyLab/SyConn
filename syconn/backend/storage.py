# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Philipp Schubert, Sven Dorkenwald, Joergen Kornfeld
import os.path
import shutil
from collections import defaultdict
from typing import Any, Tuple, Optional, Union, List, Iterator, Dict

from ..backend import StorageClass
from ..backend import log_backend
from ..handler.basics import kd_factory
from ..handler.compression import lz4string_listtoarr, arrtolz4string_list

import h5py
import numpy as np

try:
    from lz4.block import compress, decompress
except ImportError:
    from lz4 import compress, decompress


class AttributeDict(StorageClass):
    """
    General purpose dictionary class inherited from
    :class:`syconn.backend.base.StorageClass`.
    """

    def __init__(self, inp_p, **kwargs):
        super().__init__(inp_p, **kwargs)

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

    def __init__(self, inp: str, **kwargs):
        super().__init__(inp, **kwargs)

    def __getitem__(self, item: Union[int, str]):
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

    def __setitem__(self, key: Union[int, str], value: np.ndarray):
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

    def __delitem__(self, key):
        del self._dc_intern[key]
        if key in self._cache_dc:
            del self._cache_dc[key]


class VoxelStorageL(StorageClass):
    """
    Customized dictionary to store compressed numpy arrays, but with a
    intuitive user interface, i.e. compression will happen in background.
    kwarg 'cache_decomp' can be enabled to cache decompressed arrays
    additionally (save decompressing time).
    """

    def __init__(self, inp: str, **kwargs):
        super().__init__(inp, **kwargs)

    def __getitem__(self, item: Union[int, str]):
        """

        Args:
            item:

        Returns:
            Decompressed voxel masks with corresponding offsets.
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

    def __setitem__(self, key: Union[int, str],
                    values: Tuple[List[np.ndarray], List[np.ndarray]]):
        """

        Args:
            key: E.g. SO ID.
            values: E.g. voxel masks

        Returns:

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

    def append(self, key: int, voxel_mask: np.ndarray, offset: np.ndarray):
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
    Alias for :class:`~VoxelStorageDyn`.

    Args:
        inp:
        **kwargs:

    Returns:

    """
    obj = VoxelStorageDyn(inp, **kwargs)
    return obj


class VoxelStorageClass(VoxelStorageL):
    """
    Customized dictionary to store compressed numpy arrays, but with a
    intuitive user interface, i.e. compression will happen in background.
    kwarg 'cache_decomp' can be enabled to cache decompressed arrays
    additionally (save decompressing time).

    No locking provided in this class!
    """

    def __init__(self, inp: str, **kwargs):
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
    manipulation of the object's bounding boxes. In this case `voxeldata_path`
    has to be given or already be existent in loaded dictionary. Expects the
    source path of a KnossoDataset (see knossos_utils), like:

        kd = KnossoDataset()
        kd.initialize_from_knossos_path(SOURCE_PATH)

    or

        kd = kd_factory(SOURCE_PATH)

    `__setitem__` requires the object ID as key and an 3 dimensional array with
     all bounding boxes defining the object (N, 2, 3). Those BBs are then used to
     query the object voxels. The bounding box is expected to be two 3D
     coordinates which define the lower and the upper limits.


    """

    def __init__(self, inp: str, voxel_mode: bool = True,
                 voxeldata_path: Optional[str] = None, **kwargs):
        if not inp.endswith('.pkl'):
            inp = inp + '.pkl'
        super().__init__(inp, **kwargs)
        self.voxel_mode = voxel_mode
        if 'meta' not in self._dc_intern:
            # add meta information about underlying voxel data set to internal dictionary
            self._dc_intern['meta'] = dict(voxeldata_path=voxeldata_path)
        if 'size' not in self._dc_intern:
            self._dc_intern['size'] = defaultdict(int)
        if 'rep_coord' not in self._dc_intern:
            self._dc_intern['rep_coord'] = dict()
        if 'voxel_cache' not in self._dc_intern:
            self._dc_intern['voxel_cache'] = dict()
        if voxeldata_path is not None:
            old_p = self._dc_intern['meta']['voxeldata_path']
            new_p = voxeldata_path
            if old_p != new_p:
                log_backend.warn('Overwriting `voxeldata_path` in `VoxelStorageDyn` object (stored at "{}") '
                                 'from `{}` to `{}`.'.format(inp, old_p, new_p))
                self._dc_intern['meta']['voxeldata_path'] = voxeldata_path
        voxeldata_path = self._dc_intern['meta']['voxeldata_path']
        if voxel_mode:
            if voxeldata_path is None:
                msg = '`voxel_mode` is True but no path to voxeldata given / found.'
                log_backend.error(msg)
                raise ValueError(msg)
            kd = kd_factory(voxeldata_path)
            self.voxeldata = kd
        self._cache_dc = VoxelStorageLazyLoading(inp.replace('.pkl', '.npz'))

    def __setitem__(self, key: int, value: Any):
        if self.voxel_mode:
            raise RuntimeError('`VoxelStorageDyn.__setitem__` may only be used when `voxel_mode=False`.')
        else:
            return super().__setitem__(key, value)

    def __getitem__(self, item: int):
        return self.get_voxelmask_offset(item)

    def get_voxelmask_offset(self, item: int, overlap: int = 0):
        if self.voxel_mode:
            res = []
            bbs = super().__getitem__(item)
            for bb in bbs:  # iterate over all bounding boxes
                size = bb[1] - bb[0] + 2 * overlap
                off = bb[0] - overlap
                curr_mask = self.voxeldata.load_seg(size=size, offset=off, mag=1) == item
                res.append(curr_mask.swapaxes(0, 2))
            return res, bbs[:, 0]  # (N, 3) --> all offset
        else:
            return super().__getitem__(item)

    def iter_voxelmask_offset(self, item: int, overlap: int = 0) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        bbs = super().__getitem__(item)
        for bb in bbs:  # iterate over all bounding boxes
            size = bb[1] - bb[0] + 2 * overlap
            off = bb[0] - overlap
            curr_mask = self.voxeldata.load_seg(size=size, offset=off, mag=1) == item
            yield curr_mask.swapaxes(0, 2), bb[0]

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

    def push(self):
        if len(self._cache_dc) > 0:
            self._cache_dc.push()
        super().push()

    def set_voxel_cache(self, key: int, voxel_coords: np.ndarray):
        """
        This is only used to store the voxels during the synapse extraction step. This method operates independent of
        :func:`~__setitem__`.

        Args:
            key: Segment ID.
            voxel_coords: Voxel coordinates.
        """
        self._cache_dc[key] = voxel_coords

    def get_voxel_cache(self, key: int):
        """
        Voxels corresponding to item `key` must have been added to store via :func:`~set_voxel_cache`.
        This implementation operates independent of :func:`~get_voxeldata`.

        Args:
            key: Segment ID.

        Returns:
            Voxel coordinates.
        """
        return self._cache_dc[key]

    def get_voxeldata(self, item: int) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Get the object binary mask as list of 3D cubes with the respective offsets
        (in voxels). All in xyz.

        Args:
            item: Object ID.

        Returns:
            List of 3D binary masks and offsets (in voxels; xyz).
        """
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

    def get_voxel_data_cubed(self, item: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the object binary mask as dense 3D array (xyz).

        Args:
            item: Object ID.

        Returns:
            3D mask, cube offset in voxels (xyz).
        """
        bin_arrs, block_offsets = self[item]
        min_off = np.min(block_offsets, axis=0)
        block_extents = np.array([off + np.array(bin_arr.shape) for bin_arr, off in zip(bin_arrs, block_offsets)],
                                 dtype=np.int32)
        max_extent = np.max(block_extents, axis=0)
        size = max_extent - min_off
        block_offsets -= min_off
        voxel_arr = np.zeros(size, dtype=np.bool)
        for bin_arr, off in zip(bin_arrs, block_offsets):
            sh = off + np.array(bin_arr.shape, dtype=np.int32)
            voxel_arr[off[0]:sh[0], off[1]:sh[1], off[2]:sh[2]] = bin_arr
        return voxel_arr, min_off

    def get_boundingdata(self, item: int) -> List[np.ndarray]:
        """
        Get the object bounding boxes.
        Args:
            item: Object ID.

        Returns:
            List of bounding boxes (in voxels; xyz).
        """
        old_vx_mode = self.voxel_mode
        self.voxel_mode = False
        res = self[item]
        self.voxel_mode = old_vx_mode
        return res

    def keys(self):
        # do not return 'meta' and other helper items in self._dc_intern, only object IDs
        # TODO: make this a generator, check usages beforehand!
        obj_elements = list([k for k in self._dc_intern.keys() if (type(k) is str and k.isdigit())
                             or (type(k) is not str)])
        return obj_elements


class VoxelStorageLazyLoading:
    """
    Similar to `VoxelStorage` but uses lazy loading via numpy npz files.

    Notes:
        * Once  written, npz storages will not support modification via ``__setitem__``.
        * Key of types other than int are not supported. Internally, keys are converted to string,
          as required by npz, and then always converted to int for "external" use (e.g. :attr:`~keys`).
        * Call :attr:`~close` when opening an existing npz file.
    """

    def __init__(self, path: str, overwrite: bool = False):
        if not path.endswith('.npz'):
            path = path + '.npz'
        self.path = path
        self._dc_intern = {}
        if os.path.isfile(path):
            if overwrite:
                os.remove(path)
            else:
                self.pull()

    def pull(self):
        self._dc_intern = np.load(self.path)

    def push(self):
        np.savez_compressed(self.path, **self._dc_intern)

    def __setitem__(self, key: int, value: np.ndarray):
        """

        Args:
            key: Segment ID.
            value: Voxel coordinates.
        """
        # npz only allows string keys
        self._dc_intern[str(key)] = value

    def __getitem__(self, item: int) -> np.ndarray:
        """
        Voxels corresponding to `item` (supervoxel ID).

        Args:
            item: Segment ID.

        Returns:
            Voxel coordinates belonging to ID `item`.
        """
        # npz only allows string keys
        return self._dc_intern[str(item)]

    def __contains__(self, item: int) -> bool:
        """
        npz only allows string IDs.

        Args:
            item: Integer key.

        Returns:
            True if item in storage.
        """
        return str(item) in self._dc_intern

    def __len__(self):
        return len(self._dc_intern)

    def keys(self):
        for k in self._dc_intern.keys():
            yield int(k)

    def close(self):
        if isinstance(self._dc_intern, np.lib.npyio.NpzFile):
            self._dc_intern.close()


class MeshStorage(StorageClass):
    """
    Customized dictionary to store compressed numpy arrays, but with a
    intuitive user interface, i.e. compression will happen in background.
    kwarg 'cache_decomp' can be enabled to cache decompressed arrays
    additionally (save decompressing time).
    """

    def __init__(self, inp, load_colarr=False, compress=False, **kwargs):
        self.load_colarr = load_colarr
        self.compress = compress
        super().__init__(inp, **kwargs)

    def __getitem__(self, item: Union[int, str]) -> List[np.ndarray]:
        """

        Args:
            item: Key.

        Returns:
            Flat arrays: (indices, vertices, [normals, [colors/labels]])
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

    def __setitem__(self, key: int, mesh: List[np.ndarray]):
        """

        Parameters
        ----------
        key : int/str
        mesh : List[np.array]
            [indices, vertices, normals, colors/labels]
        """
        if len(mesh) == 2:
            mesh.append(np.zeros((0,), dtype=np.float32))
        if len(mesh) == 3:
            mesh.append(np.zeros((0,), dtype=np.uint8))
        if self._cache_decomp:
            self._cache_dc[key] = mesh
        if len(mesh[1]) != len(mesh[2]) > 0:
            log_backend.warning('Lengths of vertex array and length of normal'
                                ' array differ!')
        # test if lengths of vertex and color array are identical or test
        # if vertex array length is equal to 3x label array length. Arrays are flattened.
        if len(mesh[3]) > 0 and not (len(mesh[1]) == len(mesh[3]) or
                                     len(mesh[1]) == len(mesh[3]) * 3):
            log_backend.warning('Lengths of vertex array and length of color/'
                                'label array differ!')
        if self.compress:
            transf = arrtolz4string_list
        else:
            def transf(x): return x
        comp_ind = transf(mesh[0].astype(dtype=np.uint32))
        comp_vert = transf(mesh[1].astype(dtype=np.float32))
        comp_norm = transf(mesh[2].astype(dtype=np.float32))
        comp_col = transf(mesh[3].astype(dtype=np.uint8))
        self._dc_intern[key] = [comp_ind, comp_vert, comp_norm, comp_col]


class SkeletonStorage(StorageClass):
    """
    Stores skeleton dictionaries (keys: "nodes", "diameters", "edges") as compressed numpy arrays.
    """

    def __init__(self, inp, **kwargs):
        super().__init__(inp, **kwargs)

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
        if len(comp_arrs) > 3:
            for k, v in comp_arrs[3].items():
                skeleton[k] = v
        if self._cache_decomp:
            self._cache_dc[item] = skeleton
        return skeleton

    def __setitem__(self, key, skeleton):
        """

        Parameters
        ----------
        key : int/str
        skeleton : dict
            keys: nodes diameters edges and other attributes (uncompressed).
        """
        if self._cache_decomp:
            self._cache_dc[key] = skeleton
        comp_n = arrtolz4string_list(skeleton["nodes"].astype(dtype=np.uint32))
        comp_d = arrtolz4string_list(skeleton["diameters"].astype(dtype=np.float32))
        comp_e = arrtolz4string_list(skeleton["edges"].astype(dtype=np.uint32))
        entry = [comp_n, comp_d, comp_e, dict()]
        if len(skeleton) > 3:
            for k, v in skeleton.items():
                if k in ['nodes', 'diameters', 'edges']:
                    continue
                entry[3][k] = v
        self._dc_intern[key] = entry


class BinarySearchStore:
    def __init__(self, fname: str, id_array: Optional[np.ndarray] = None,
                 attr_arrays: Optional[Dict[str, np.ndarray]] = None, overwrite: bool = False,
                 n_shards: Optional[int] = None, rdcc_nbytes: int = 5*2**20):
        """
        Data structure to store properties (values) of a corresponding ID array (keys). Internally a binary search
        is used that uses a sorted representation of keys and values to enable sparse look-ups with a much lower
        memory complexity than python dictionaries.
        Maximum ID is the last element of :attr:`~id_array`.

        Args:
            fname: File name.
            id_array: (Unsorted) ID array.
            attr_arrays: (Unsorted) attribute arrays, must have the same ordering as ID array.
            overwrite: Overwrite existing array files.
            n_shards: Number of shards/chunks the ID and attribute arrays are split into. Defaults to 5.
            rdcc_nbytes: Size of h5 chunks in bytes. Default is 5 MiB.
        """
        self.fname = fname
        self._h5_file = None
        if id_array is not None:
            if attr_arrays is None:
                raise ValueError('ID array is given, but no attribute array(s).')
            if isinstance(fname, str) and os.path.isfile(fname):
                if not overwrite:
                    raise FileExistsError(f'BinarySearchStore at "{fname}" already exists and overwrite is False."')
                else:
                    os.remove(fname)
            if n_shards is None:
                n_shards = 5
            if isinstance(fname, str):
                os.makedirs(os.path.split(self.fname)[0], exist_ok=True)
            # sort keys / ID array
            ixs = np.argsort(id_array)
            id_array = id_array[ixs]
            bucket_ranges = []
            h5_file = h5py.File(fname, 'w', libver='latest', rdcc_nbytes=rdcc_nbytes)
            grp = h5_file.create_group("ids")
            for ii, id_sub in enumerate(np.array_split(id_array, n_shards)):
                bucket_ranges.append((id_sub[0], id_sub[-1]))
                grp.create_dataset(f'{ii}', data=id_sub)
            for k, v in attr_arrays.items():
                v_sorted = v[ixs]
                grp = h5_file.create_group(k)
                grp.attrs['shape'] = v_sorted.shape
                grp.attrs['dtype'] = np.dtype(v_sorted.dtype).str
                for ii, attr_sub in enumerate(np.array_split(v_sorted, n_shards)):
                    grp.create_dataset(f'{ii}', data=attr_sub)
            del ixs
            h5_file.attrs['bucket_ranges'] = bucket_ranges
            h5_file.close()
        else:
            if isinstance(fname, str) and not os.path.isfile(fname):
                raise FileNotFoundError(f'Could not find BinarySearchStore at "{self.fname}".')

    @property
    def n_shards(self) -> int:
        """
        Number of shards/chunks the ID and attribute arrays are split into.
        Returns:

        """
        with h5py.File(self.fname, 'r', libver='latest') as f:
            n_shards = len(f.attrs['bucket_ranges'])
        return n_shards

    @property
    def id_array(self) -> np.ndarray:
        """

        Returns:
            Flat ID array.
        """
        ids = []
        with h5py.File(self.fname, 'r', libver='latest') as f:
            for bucket_id in range(len(f.attrs['bucket_ranges'])):
                ids.append(f[f'ids/{bucket_id}'][()])
        return np.concatenate(ids)

    def _get_bucket_ids(self, obj_ids: np.ndarray) -> np.ndarray:
        bucket_ids = np.ones(obj_ids.shape, dtype=np.int32) * -1
        for ii, bucket_range in enumerate(self._h5_file.attrs['bucket_ranges']):
            bucket_ids[(bucket_range[0] <= obj_ids) & (obj_ids <= bucket_range[1])] = ii
        if -1 in bucket_ids:
            raise ValueError(f'IDs {obj_ids[bucket_ids == -1]} not in {self.fname}.')
        return bucket_ids

    def get_attributes(self, obj_ids: np.ndarray, attr_key: str) -> np.ndarray:
        """
        Query attributes of given `obj_ids`. Note that this will not raise an Exception if a ID does not exist in the
        store, as the lookup uses binary search.

        Args:
            obj_ids: Object IDs to query.
            attr_key: Value type obtained from the store.

        Returns:
            Value array.
        """
        self._h5_file = h5py.File(self.fname, 'r', libver='latest')
        if attr_key not in self._h5_file.keys():
            raise KeyError(f'Key "{attr_key}" does not exist.')
        bucket_ids = self._get_bucket_ids(obj_ids)
        grp = self._h5_file[f'{attr_key}']
        sh = [len(obj_ids)]
        if len(grp.attrs['shape']) > 1:
            sh += list(grp.attrs['shape'])[1:]
        data = np.zeros(sh, dtype=grp.attrs['dtype'])
        for bucket_id in np.unique(bucket_ids):
            ids = self._h5_file[f'ids/{bucket_id}'][()]
            bucket_mask = bucket_ids == bucket_id
            queries = obj_ids[bucket_mask]
            ixs_sort = np.argsort(queries)
            indices = np.searchsorted(ids, queries[ixs_sort])
            d = grp[f'{bucket_id}'][list(indices)]
            # undo sorting using argsort of argsort to match slicing mask on the left
            data[bucket_mask] = d[np.argsort(ixs_sort)]
        self._h5_file.close()
        self._h5_file = None
        return data


def bss_get_attr_helper(args):
    """
    Helper function to query attributes from a BinarySearchStore instance.

    Args:
        args: BinarySearchStore, query_ids, attribute key.

    Returns:
        Query result.
    """
    bss, samples, key = args
    return bss.get_attributes(samples, key)
