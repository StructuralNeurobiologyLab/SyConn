# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Philipp Schubert, Sven Dorkenwald, Joergen Kornfeld
from ..handler import log_handler
import os
from typing import List, Tuple, Optional, Iterable, Union, Dict
import h5py
import numpy as np
try:
    from lz4.block import compress, decompress
except ImportError:
    from lz4 import compress, decompress
from lz4.block import LZ4BlockError
try:
    import fasteners
    LOCKING = True
except ImportError:
    print("fasteners could not be imported. Locking will be disabled by default."
          "Please install fasteners to enable locking (pip install fasteners).")
    LOCKING = False

__all__ = ['arrtolz4string', 'lz4stringtoarr', 'load_lz4_compressed',
           'save_lz4_compressed', 'load_from_h5py',
           'save_to_h5py', 'lz4string_listtoarr', 'arrtolz4string_list']


def arrtolz4string(arr: np.ndarray) -> bytes:
    """
    Converts (multi-dimensional) array to list of lz4 compressed strings.

    Args:
        arr: Input array.

    Returns:
        lz4 compressed string.
    """
    if isinstance(arr, list):
        arr = np.array(arr)
    if len(arr) == 0:
        return b""
    try:
        comp_arr = compress(arr.tobytes())
    except OverflowError:
        log_handler.warning(OverflowError, "Overflow occurred when compression array."
                                           "Use 'arrtolz4string_list' instead.")
        comp_arr = arrtolz4string_list(arr)

    return comp_arr


def lz4stringtoarr(string: bytes, dtype: np.dtype = np.float32,
                   shape: Optional[Tuple[int]] = None):
    """
    Converts lz4 compressed string to 1d array.

    Args:
        string: Serialized array.
        dtype: Data type of original array.
        shape: Shape of original array.

    Returns:
        N-dimensional numpy array.
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


def arrtolz4string_list(arr: np.ndarray) -> List[bytes]:
    """
    Converts (multi-dimensional) array to list of lz4 compressed strings.

    Args:
        arr: Input array.

    Returns:
        lz4 compressed string.
    """
    if isinstance(arr, list):
        arr = np.array(arr)
    if len(arr) == 0:
        return [b""]
    try:
        str_lst = [compress(arr.tobytes())]
    # catch Value error which is thrown in py3 lz4 version
    except (OverflowError, ValueError, LZ4BlockError):
        half_ix = len(arr) // 2
        str_lst = arrtolz4string_list(arr[:half_ix]) + arrtolz4string_list(arr[half_ix:])
    return str_lst


def lz4string_listtoarr(str_lst: List[bytes], dtype: np.dtype = np.float32,
                        shape: Optional[Tuple[int]] = None) -> np.ndarray:
    """
    Converts lz4 compressed strings to array.

    Args:
        str_lst: Binary string representation of the array.
        dtype: Data type of the serialized array.
        shape: Shape of the serialized array.

    Returns:
        1d numpy array.
    """
    if len(str_lst) == 0:
        return np.zeros((0, ), dtype=dtype)
    arr_lst = []
    for string in str_lst:
        arr_lst.append(lz4stringtoarr(string, dtype=dtype, shape=shape))
    return np.concatenate(arr_lst)


def multi_lz4stringtoarr(args: tuple) -> np.ndarray:
    """
    Helper function for multiprocessing.

    Args:
        args: see :func:`~syconn.handler.compression.lz4string_listtoarr`.

    Returns:
        1d numpy array.
    """
    return lz4string_listtoarr(*args)


def save_lz4_compressed(p: str, arr: np.ndarray, dtype: np.dtype = np.float32):
    """
    Saves array as lz4 compressed string. Due to overflow in python2 added
    error handling by recursive splitting.

    Args:
        p: Path to the destination file.
        arr: Numpy array.
        dtype: Data type in which the array should be stored.
    """
    arr = arr.astype(dtype)
    try:
        text_file = open(p, "wb")
        text_file.write(arrtolz4string(arr))
        text_file.close()
    except (OverflowError, ValueError):
        # save dummy (emtpy) file
        text_file = open(p, "wb")
        text_file.write(b"")
        text_file.close()
        half_ix = len(arr) // 2
        new_p1 = p[:-4] + "_1" + p[-4:]
        new_p2 = p[:-4] + "_2" + p[-4:]
        save_lz4_compressed(new_p1, arr[:half_ix])
        save_lz4_compressed(new_p2, arr[half_ix:])


def load_lz4_compressed(p: str, shape: Tuple[int] = (-1, 20, 2, 128, 256),
                        dtype: np.dtype = np.float32):
    """
    Shape must be known in order to load (multi-dimensional) array from binary
    string. Due to overflow in python2 added recursive loading.

    Args:
        p: path to lz4 file
        shape: tuple
        dtype: type

    Returns: np.array

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
def load_from_h5py(path: str, hdf5_names: Optional[Iterable[str]] = None,
                   as_dict: bool = False)\
        -> Union[Dict[str, np.ndarray], List[np.ndarray]]:
    """
    Loads data from a h5py File.

    Args:
        path: Path to .h5 file.
        hdf5_names: If None, all keys will be loaded.
        as_dict: If True, returns a dictionary.

    Returns:
        The data stored at `path` either as list of arrays
        (ordering as `hdf5_names`) or as dictionary.
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
                data[hdf5_name] = f[hdf5_name][()]
            else:
                data.append(f[hdf5_name][()])
    except Exception as e:
        msg = "Error ({}) raised when loading h5-file at path:" \
              " {}, with labels: {}".format(e, path, hdf5_names)
        log_handler.error(msg)
        raise Exception(e)
    f.close()
    return data


def save_to_h5py(data: Union[Dict[str, np.ndarray], List[np.ndarray]],
                 path: str, hdf5_names: Optional[List[str]] = None,
                 overwrite: bool = False,
                 compression: bool = True):
    """
    Saves data to h5py File.

    Args:
        data: If list, hdf5_names has to be set.
        path: Forward-slash separated path to file.
        hdf5_names: Keys used to store arrays in `data`.
            Has to be the same length as `data`.
        overwrite: Determines whether existing files are overwritten.
        compression: If True, ``compression='gzip'`` is used which is
            recommended for sparse and ordered data.
    """
    if (not type(data) is dict) and hdf5_names is None:
        raise TypeError("hdf5names has to be set when data is a list")
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
            msg = "Not enough or too many hdf5-names ({}) given during" \
                  " h5-file save attempt!".format(hdf5_names)
            log_handler.error(msg)
            raise ValueError(msg)
        for nb_data in range(len(data)):
            if compression:
                f.create_dataset(hdf5_names[nb_data], data=data[nb_data],
                                 compression="gzip")
            else:
                f.create_dataset(hdf5_names[nb_data], data=data[nb_data])
    f.close()
