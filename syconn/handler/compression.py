# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Philipp Schubert, Sven Dorkenwald, Joergen Kornfeld
try:
    from lz4.block import compress, decompress
except ImportError:
    from lz4 import compress, decompress
try:
    import fasteners
    LOCKING = True
except ImportError:
    print("fasteners could not be imported. Locking will be disabled by default."
          "Please install fasteners to enable locking (pip install fasteners).")
    LOCKING = False
import numpy as np
import h5py
import os

from ..handler import log_handler
__all__ = ['arrtolz4string', 'lz4stringtoarr', 'load_lz4_compressed',
           'save_lz4_compressed', 'load_compressed', 'load_from_h5py',
           'save_to_h5py', 'lz4string_listtoarr', 'arrtolz4string_list']


def arrtolz4string(arr):
    """
    Converts (multi-dimensional) array to lz4 compressed string.

    Parameters
    ----------
    arr : np.array

    Returns
    -------
    byte
        lz4 compressed string
    """
    if isinstance(arr, list):
        arr = np.array(arr)
    if len(arr) == 0:
        return ""
    try:
        comp_arr = compress(arr.tobytes())
    except OverflowError:
        log_handler.warning(OverflowError, "Overflow occurred when compression array."
                                           "Use 'arrtolz4string_list' instead.")
        comp_arr = arrtolz4string_list(arr)

    return comp_arr


def lz4stringtoarr(string, dtype=np.float32, shape=None):
    """
    Converts lz4 compressed string to 1d array.

    Parameters
    ----------
    string : byte
    dtype : type
    shape : tuple

    Returns
    -------
    np.array
        1d array
    """
    if len(string) == 0:
        return np.zeros((0, ), dtype=dtype)
    try:
        arr_1d = np.frombuffer(decompress(string), dtype=dtype)
    except TypeError:  # python3 compatibility
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
    # catch Value error which is thrown in py3 lz4 version
    except (OverflowError, ValueError):
        half_ix = len(arr) // 2
        str_lst = arrtolz4string_list(arr[:half_ix]) + \
                   arrtolz4string_list(arr[half_ix:])
    return str_lst


def lz4string_listtoarr(str_lst, dtype=np.float32, shape=None):
    """
    Converts lz4 compressed strings to array.

    Parameters
    ----------
    str_lst : List[str]
    dtype : type
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
    except (OverflowError, ValueError):
        # save dummy (emtpy) file
        text_file = open(p, "wb")
        text_file.write("")
        text_file.close()
        half_ix = len(arr) // 2
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
    dtype : type

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
    except Exception as e:
        msg = "Error ({}) raised when loading h5-file at path:" \
              " {}, with labels: {}".format(e, path, hdf5_names)
        log_handler.error(msg)
        raise Exception(e)
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
    hdf5_names: List[str]
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
            msg = "Not enough or too many hdf5-names ({}) given when during" \
                  " h5-file load attempt!".format(hdf5_names)
            log_handler.error(msg)
            raise ValueError(msg)
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