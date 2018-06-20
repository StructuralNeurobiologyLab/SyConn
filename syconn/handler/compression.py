# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Sven Dorkenwald, Philipp Schubert, Joergen Kornfeld

try:
    from lz4.block import compress, decompress
except ImportError:
    from lz4 import compress, decompress
import time
try:
    import fasteners
    LOCKING = True
except ImportError:
    print("fasteners could not be imported. Locking will be disabled by default."
          "Please install fasteners to enable locking (pip install fasteners).")
    LOCKING = False
from .basics import load_pkl2obj, write_obj2pkl
import numpy as np
import h5py
import os
import shutil
import warnings
__all__ = ["arrtolz4string", "lz4stringtoarr", "load_lz4_meshdict_items",
           "load_lz4_compressed", "add_lz4_meshdict_items", "init_lz4_meshdict",
           "save_lz4_compressed", "load_compressed", "load_from_h5py",
           "save_to_h5py"]


# ---------------------------- lz4
# ------------------------------------------------------------------------------
class LZ4DictBase(dict):
    """
    Customized dictionary to store compressed numpy arrays, but with a 
    intuitive user interface, i.e. compression will happen in background.
    kwarg 'cache_decomp' can be enabled to cache decompressed arrays 
    additionally (save decompressing time when accessing items frequently).
    """
    def __init__(self, inp_p, cache_decomp=False, read_only=True,
                 max_delay=100, timeout=1000, disable_locking=not LOCKING,
                 max_nb_attempts=10):
        super(LZ4DictBase, self).__init__()
        self.read_only = read_only
        self.a_lock = None
        self.max_delay = max_delay
        self.timeout = timeout
        self.disable_locking = disable_locking
        self._cache_decomp = cache_decomp
        self._max_nb_attempts = max_nb_attempts
        self._cache_dc = {}
        self._dc_intern = {}
        self._path = inp_p
        if inp_p is not None:
            if isinstance(inp_p, str):
                self.load_pkl(inp_p)
            else:
                raise("Unsupported initialization type %s for LZ4Dict." %
                      type(inp_p), NotImplementedError)

    def __delitem__(self, key):
        try:
            del self[key]
        except KeyError:
            raise AttributeError("No such attribute: ", key)

    def __len__(self):
        return self._dc_intern.__len__()

    def __eq__(self, other):
        if not isinstance(other, LZ4Dict):
            return False
        return self._dc_intern.__eq__(other._dc_intern)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __contains__(self, item):
        return self._dc_intern.__contains__(item)

    def __iter__(self):
        return iter(self._dc_intern)

    def __repr__(self):
        return self._dc_intern.__repr__()

    def update(self, other, **kwargs):
        raise NotImplementedError

    def copy(self):
        raise NotImplementedError

    def items(self):
        return [(k, self[k]) for k in self._dc_intern.keys()]

    def values(self):
        return [self[k] for k in self._dc_intern.keys()]

    def keys(self):
        return self._dc_intern.keys()

    def iteritems(self):
        for k in self.keys():
            yield k, self[k]

    def itervalues(self):
        for k in self.keys():
            yield self[k]

    def iterkeys(self):
        for k in self.keys():
            yield k

    def save2pkl(self, dest_path=None):
        if dest_path is None:
            dest_path = self._path
        write_obj2pkl(dest_path + ".tmp", self._dc_intern)
        shutil.move(dest_path + ".tmp", dest_path)
        if not self.read_only and not self.disable_locking:
            self.a_lock.release()

    def load_pkl(self, source_path=None):
        if source_path is None:
            source_path = self._path
        fold, fname = os.path.split(source_path)
        lock_path = fold + "/." + fname + ".lk"
        if not os.path.isdir(os.path.split(source_path)[0]):
            try:
                os.makedirs(os.path.split(source_path)[0])
            except:
                pass
        # acquires lock until released when saving or after loading if self.read_only
        if not self.disable_locking:
            gotten = False
            nb_attempts = 1
            while True:
                self.a_lock = fasteners.InterProcessLock(lock_path)
                start = time.time()
                gotten = self.a_lock.acquire(blocking=True, delay=0.1,
                                             max_delay=self.max_delay,
                                             timeout=self.timeout / self._max_nb_attempts)
                # if not gotten and maximum attempts not reached yet keep trying
                if not gotten and nb_attempts < 10:
                    nb_attempts += 1
                else:
                    break
            if not gotten:
                raise RuntimeError("Unable to acquire file lock for %s after"
                               "%0.0fs." % (source_path, time.time()-start))
        if os.path.isfile(source_path):
            try:
                self._dc_intern = load_pkl2obj(source_path)
            except EOFError:
                warnings.warn("Could not load LZ4Dict (%s). 'save2pkl' will"
                              " overwrite broken .pkl file." % self._path,
                              RuntimeWarning)
                self._dc_intern = {}
        else:
            self._dc_intern = {}
        if self.read_only and not self.disable_locking:
            self.a_lock.release()


class AttributeDict(LZ4DictBase):
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


class LZ4Dict(LZ4DictBase):
    """
    Customized dictionary to store compressed numpy arrays, but with a 
    intuitive user interface, i.e. compression will happen in background.
    kwarg 'cache_decomp' can be enabled to cache decompressed arrays 
    additionally (save decompressing time when accessing items frequently).
    """
    def __init__(self, inp, **kwargs):
        super(LZ4Dict, self).__init__(inp, **kwargs)

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
        assert isinstance(value, np.ndarray), "LZ4Dict supports np.array " \
                                              "values only."
        if self._cache_decomp:
            self._cache_dc[key] = value
        sh = list(value.shape)
        sh[0] = -1
        value_intern = {"arr": arrtolz4string_list(value), "sh": tuple(sh),
                        "dt": value.dtype.str}
        self._dc_intern[key] = value_intern


class VoxelDictL(LZ4DictBase):
    """
    Customized dictionary to store compressed numpy arrays, but with a 
    intuitive user interface, i.e. compression will happen in background.
    kwarg 'cache_decomp' can be enabled to cache decompressed arrays 
    additionally (save decompressing time).
    """

    def __init__(self, inp, **kwargs):
        super(VoxelDictL, self).__init__(inp, **kwargs)

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
                        "sh": sh, "dt": voxel_masks[0].dtype.str, "off": offsets}
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


class MeshDict(LZ4DictBase):
    """
    Customized dictionary to store compressed numpy arrays, but with a 
    intuitive user interface, i.e. compression will happen in background.
    kwarg 'cache_decomp' can be enabled to cache decompressed arrays 
    additionally (save decompressing time).
    """

    def __init__(self, inp, **kwargs):
        super(MeshDict, self).__init__(inp, **kwargs)


    def __getitem__(self, item):
        """
        
        Parameters
        ----------
        item : int/str

        Returns
        -------
        list of np.arrays
            [indices, vertices]
        """
        try:
            return self._cache_dc[item]
        except KeyError:
            pass
        mesh = self._dc_intern[item]
        # if no normals were given in file / cache append empty array
        if len(mesh) == 2:
            mesh.append([""])
        decomp_arrs = [lz4string_listtoarr(mesh[0], dtype=np.uint32),
                       lz4string_listtoarr(mesh[1], dtype=np.float32),
                       lz4string_listtoarr(mesh[2], dtype=np.float32)]
        if self._cache_decomp:
            self._cache_dc[item] = decomp_arrs
        return decomp_arrs

    def __setitem__(self, key, mesh):
        """
        
        Parameters
        ----------
        key : int/str
        mesh : list of np.array
            [indices, vertices, optional: normals]
        """
        if len(mesh) == 2:
            mesh.append(np.zeros((0, ), dtype=np.float32))
        if self._cache_decomp:
            self._cache_dc[key] = mesh
        comp_ind = arrtolz4string_list(mesh[0].astype(dtype=np.uint32))
        comp_vert = arrtolz4string_list(mesh[1].astype(dtype=np.float32))
        comp_norm = arrtolz4string_list(mesh[2].astype(dtype=np.float32))
        self._dc_intern[key] = [comp_ind, comp_vert, comp_norm]


class VoxelDict(VoxelDictL):
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
        super(VoxelDictL, self).__init__(inp, **kwargs)


class SkeletonDict(LZ4DictBase):
    """
    Stores skeleton dictionaries (keys: "nodes", "diameters", "edges") as
    compressed numpy arrays.
    """

    def __init__(self, inp, **kwargs):
        super(SkeletonDict, self).__init__(inp, **kwargs)

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
        skeleton : list of np.array
            [indices, vertices]
        """
        if self._cache_decomp:
            self._cache_dc[key] = skeleton
        comp_n = arrtolz4string_list(skeleton["nodes"].astype(dtype=np.uint32))
        comp_d = arrtolz4string_list(skeleton["diameters"].astype(dtype=np.float32))
        comp_e = arrtolz4string_list(skeleton["edges"].astype(dtype=np.uint32))
        self._dc_intern[key] = [comp_n, comp_d, comp_e]


def arrtolz4string(arr):
    """
    Converts (multi-dimensional) array to lz4 compressed string.

    Parameters
    ----------
    arr : np.array

    Returns
    -------
    str
        lz4 compressed string
    """
    if isinstance(arr, list):
        arr = np.array(arr)
    if len(arr) == 0:
        return ""
    try:
        comp_arr = compress(arr.tobytes())
    except OverflowError:
        warnings.warn(OverflowError, "Overflow occurred when compression array."
                                     "Use 'arrtolz4string_list' instead.")
        comp_arr = arrtolz4string_list(arr)

    return comp_arr


def lz4stringtoarr(string, dtype=np.float32, shape=None):
    """
    Converts lz4 compressed string to 1d array.

    Parameters
    ----------
    string : str
    dtype : np.dtype
    shape : tuple

    Returns
    -------
    np.array
        1d array
    """
    if string == "":
        return np.zeros((0, ), dtype=dtype)
    try:
        arr_1d = np.frombuffer(decompress(string), dtype=dtype)
    except TypeError: # python3 compatibility
        arr_1d = np.frombuffer(decompress(str.encode(string)), dtype=dtype)
    if shape is not None:
        arr_1d = arr_1d.reshape(shape)
    return arr_1d


def arrtolz4string_list(arr):
    """
    Converts (multi-dimensional) array to list of lz4 compressed strings.

    Parameters
    ----------
    arr : np.array

    Returns
    -------
    list of str
        lz4 compressed string
    """
    if isinstance(arr, list):
        arr = np.array(arr)
    if len(arr) == 0:
        return [""]
    try:
        str_lst = [compress(arr.tobytes())]
    except OverflowError:
        half_ix = len(arr) / 2
        str_lst = arrtolz4string_list(arr[:half_ix]) + \
                   arrtolz4string_list(arr[half_ix:])
    return str_lst


def lz4string_listtoarr(str_lst, dtype=np.float32, shape=None):
    """
    Converts lz4 compressed strings to array.

    Parameters
    ----------
    str_lst : list of str
    dtype : np.dtype
    shape : tuple

    Returns
    -------
    np.array
        1d array
    """
    if len(str_lst) == 0:
        return np.zeros((0, ), dtype=dtype)
    arr_lst = []
    for string in str_lst:
        arr_lst.append(lz4stringtoarr(string, dtype=dtype, shape=shape))
    return np.concatenate(arr_lst)


def multi_lz4stringtoarr(args):
    return lz4string_listtoarr(*args)


def save_lz4_compressed(p, arr, dtype=np.float32):
    """
    Saves array as lz4 compressed string. Due to overflow in python2 added
    error handling by recursive splitting.

    Parameters
    ----------
    p : str
    arr : np.array
    dtype : np.dtype

    Returns
    -------
    None
    """
    arr = arr.astype(dtype)
    try:
        text_file = open(p, "wb")
        text_file.write(arrtolz4string(arr))
        text_file.close()
    except OverflowError:
        # save dummy (emtpy) file
        text_file = open(p, "wb")
        text_file.write("")
        text_file.close()
        half_ix = len(arr) / 2
        new_p1 = p[:-4] + "_1" + p[-4:]
        new_p2 = p[:-4] + "_2" + p[-4:]
        save_lz4_compressed(new_p1, arr[:half_ix])
        save_lz4_compressed(new_p2, arr[half_ix:])


def load_lz4_compressed(p, shape=(-1, 20, 2, 128, 256), dtype=np.float32):
    """
    Shape must be known in order to load (multi-dimensional) array from binary
    string. Due to overflow in python2 added recursive loading.

    Parameters
    ----------
    p : path to lz4 file
    shape : tuple
    dtype : np.dtype

    Returns
    -------
    np.array
    """
    with open(p, "rb") as text_file:
        decomp_arr = lz4stringtoarr(text_file.read(), dtype=dtype, shape=shape)
    # assume original array was split due to overflow error
    if len(decomp_arr) == 0:
        new_p1 = p[:-4] + "_1" + p[-4:]
        new_p2 = p[:-4] + "_2" + p[-4:]
        decomp_arr1 = load_lz4_compressed(new_p1, shape=shape, dtype=dtype)
        decomp_arr2 = load_lz4_compressed(new_p2, shape=shape, dtype=dtype)
        decomp_arr = np.concatenate([decomp_arr1, decomp_arr2])
    return decomp_arr


# def init_lz4_meshdict(sv_ixs, meshes):
#     res = {}
#     for m, ix in zip(meshes, sv_ixs):
#         res[ix] = [arrtolz4string(m[0].astype(np.uint32)),
#                    arrtolz4string(m[1].astype(np.float32))]
#     return res
#
#
# def load_lz4_meshdict_items(dc):
#     return [(lz4stringtoarr(dc[k][0], dtype=np.uint32),
#              lz4stringtoarr(dc[k][1], dtype=np.float32)) for k in dc.keys()]
def init_lz4_meshdict(sv_ixs, meshes):
    res = {}
    for m, ix in zip(meshes, sv_ixs):
        res[ix] = {"ind": arrtolz4string(m[0]), "vert": arrtolz4string(m[1])}
    return res


def load_lz4_meshdict_items(dc):
    dtype = np.float
    # HACK: check if vertices were saved as integer or float
    if np.sum(lz4stringtoarr(dc[dc.keys()[0]]["vert"], dtype=np.float32).astype(np.uint)) == 0:
        dtype = np.uint
    return [(lz4stringtoarr(dc[k]["ind"], dtype=np.uint),
             lz4stringtoarr(dc[k]["vert"], dtype=dtype)) for k in dc.keys()]


def add_lz4_meshdict_items(dc, sv_ixs, meshes):
    dc.update(init_lz4_meshdict(sv_ixs, meshes))


# ---------------------------- HDF5
# ------------------------------------------------------------------------------
def load_from_h5py(path, hdf5_names=None, as_dict=False):
    """
    Loads data from a h5py File

    Parameters
    ----------
    path: str
    hdf5_names: list of str
        if None, all keys will be loaded
    as_dict: boolean
        if False a list is returned

    Returns
    -------
    data: dict or np.array

    """
    if as_dict:
        data = {}
    else:
        data = []
    try:
        f = h5py.File(path, 'r')
        if hdf5_names is None:
            hdf5_names = f.keys()
        for hdf5_name in hdf5_names:
            if as_dict:
                data[hdf5_name] = f[hdf5_name].value
            else:
                data.append(f[hdf5_name].value)
    except:
        raise Exception("Error at Path: %s, with labels:" % path, hdf5_names)
    f.close()
    return data


def save_to_h5py(data, path, hdf5_names=None, overwrite=False, compression=True):
    """
    Saves data to h5py File.

    Parameters
    ----------
    data: list or dict of np.arrays
        if list, hdf5_names has to be set.
    path: str
        forward-slash separated path to file
    hdf5_names: list of str
        has to be the same length as data
    overwrite : bool
        determines whether existing files are overwritten
    compression : bool
        True: compression='gzip' is used which is recommended for sparse and
        ordered data

    Returns
    -------
    nothing

    """
    if (not type(data) is dict) and hdf5_names is None:
        raise TypeError("hdf5names has to be set, when data is a list")
    if os.path.isfile(path) and overwrite:
        os.remove(path)
    f = h5py.File(path, "w")
    if type(data) is dict:
        for key in data.keys():
            if compression:
                f.create_dataset(key, data=data[key], compression="gzip")
            else:
                f.create_dataset(key, data=data[key])
    else:
        if len(hdf5_names) != len(data):
            f.close()
            raise ValueError("Not enough or too many hdf5-names given!")
        for nb_data in range(len(data)):
            if compression:
                f.create_dataset(hdf5_names[nb_data], data=data[nb_data],
                                 compression="gzip")
            else:
                f.create_dataset(hdf5_names[nb_data], data=data[nb_data])
    f.close()


def load_compressed(p):
    f = np.load(p)
    assert len(f.keys()) == 1, "More than one key in .npz file"
    return f[f.keys()[0]]