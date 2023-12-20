# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Philipp Schubert, Sven Dorkenwald, Joergen Kornfeld
import os
from typing import List, Tuple, Optional, Iterable, Union, Dict

import h5py
import numpy as np

from ..handler import log_handler

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
    Converts (multi-dimensional) array to an LZ4 compressed byte string.
    
    This function takes a NumPy array or a list and compresses it using the LZ4
    algorithm, returning the compressed data as a byte string. If the input is a
    list, it is first converted to a NumPy array. If the array is empty, an empty
    byte string is returned. In case of an OverflowError during compression, a
    warning is logged, and the 'arrtolz4string_list' function is used instead.
    
    Args:
        arr: The NumPy array to be compressed.
    
    Returns:
        A byte string containing the LZ4 compressed data.
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
    Converts an LZ4 compressed byte string back to a NumPy array.
    
    This function decompresses a byte string that was previously compressed using
    the LZ4 algorithm and reconstructs the original NumPy array. The data type and
    shape of the original array must be provided. If the input string is empty, an
    empty NumPy array is returned. If the shape is provided, the decompressed data
    is reshaped accordingly.
    
    Args:
        string: The LZ4 compressed byte string to be decompressed.
        dtype: The data type of the original array, to ensure data integrity.
        shape: The original array's shape, used to reshape the decompressed data.
    
    Returns:
        An N-dimensional numpy array reconstructed from the compressed data.
    """
    if len(string) == 0:
        return np.zeros((0,), dtype=dtype)
    try:
        arr_1d = np.frombuffer(decompress(string), dtype=dtype)
    except TypeError:  # python3 compatibility
        arr_1d = np.frombuffer(decompress(str.encode(string)), dtype=dtype)
    if shape is not None:
        arr_1d = arr_1d.reshape(shape)
    return arr_1d


def arrtolz4string_list(arr: np.ndarray) -> List[bytes]:
    """
    Converts a NumPy array to a list of LZ4 compressed byte strings.
    
    This function is similar to 'arrtolz4string' but returns a list of compressed byte
    strings instead of a single byte string, to handle large arrays that cannot be
    compressed in a single step. If the input is a list, it is first converted to a
    NumPy array. If the array is empty, a list containing a single empty byte string
    is returned.
    
    Args:
        arr: The NumPy array or a list to be compressed.
    
    Returns:
        A list of LZ4 compressed byte strings.
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


def lz4string_listtoarr(str_lst: Union[List[bytes], np.ndarray], dtype: np.dtype = np.float32,
                        shape: Optional[Tuple[int]] = None) -> np.ndarray:
    """
    Converts a list of LZ4 compressed byte strings back to a NumPy array.
    
    This function takes a list of byte strings, each compressed using the LZ4
    algorithm, and concatenates their decompressed data to reconstruct the original
    NumPy array. If the input is already a NumPy array, it is returned as is. If the
    list is empty, an empty NumPy array is returned. If the shape is provided, the
    decompressed data is reshaped accordingly.
    
    Args:
        str_lst: A list of LZ4 compressed byte strings or a NumPy array. If numpy
                 array, do nothing.
        dtype: The data type of the original array. If not provided, the dtype
               inferred from the compressed data will be used.
        shape: The shape of the original array, used for reshaping the decompressed
               data into the specified shape. If not provided, the shape is inferred
               from the compressed data.
    
    Returns:
        A 1d numpy array reconstructed from the compressed data.
    """
    if type(str_lst) is np.ndarray:
        return str_lst
    if len(str_lst) == 0:
        return np.zeros((0,), dtype=dtype)
    arr_lst = []
    for string in str_lst:
        arr_lst.append(lz4stringtoarr(string, dtype=dtype, shape=shape))
    return np.concatenate(arr_lst)


def multi_lz4stringtoarr(args: tuple) -> np.ndarray:
    """
    Helper function for multiprocessing that converts LZ4 compressed byte strings to a NumPy array.
    
    This function is intended for use with multiprocessing to decompress multiple
    LZ4 compressed byte strings in parallel. It accepts a tuple with arguments for
    the 'lz4string_listtoarr' function and returns the resulting 1d NumPy array.
    
    Args:
        args: Tuple with arguments for :func:`~syconn.handler.compression.lz4string_listtoarr`.
    
    Returns:
        1d numpy array reconstructed from the compressed data.
    """
    return lz4string_listtoarr(*args)


def save_lz4_compressed(p: str, arr: np.ndarray, dtype: np.dtype = np.float32):
    """
    Saves a NumPy array as an LZ4 compressed file.
    
    This function compresses a NumPy array using the LZ4 algorithm and saves the
    compressed data to a file. If an OverflowError or ValueError occurs during
    compression, the array is split into two halves and the function is called
    recursively to save each half separately, addressing issues specifically
    related to python2.
    
    Args:
        p: The file path where the compressed data will be saved.
        arr: The NumPy array to be compressed and saved.
        dtype: The data type to which the array should be cast before compression.
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
    Loads a NumPy array from an LZ4 compressed file.
    
    This function reads LZ4 compressed data from a file and decompresses it to
    reconstruct the original NumPy array. The shape and data type of the original
    array must be provided. In cases of decompression issues, possibly due to
    overflow in Python 2, it attempts to recursively load and concatenate split
    parts of the array to return the complete array.
    
    Args:
        p: The file path of the LZ4 compressed file.
        shape: The shape of the original array, provided as a tuple.
        dtype: The data type of the original array, provided as a type.
    
    Returns:
        The decompressed NumPy array as np.array.
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
                   as_dict: bool = False) \
        -> Union[Dict[str, np.ndarray], List[np.ndarray]]:
    """
    Loads data from an HDF5 file.
    
    This function reads data from an HDF5 file and returns it either as a list of
    NumPy arrays or as a dictionary mapping dataset names to their corresponding
    arrays. If 'hdf5_names' is None, all datasets in the file are loaded. If 'as_dict'
    is True, the data is returned as a dictionary; otherwise, it is returned as a list.
    
    Args:
        path: The file path of the HDF5 file.
        hdf5_names: An iterable of dataset names to load; if None, all datasets are loaded.
        as_dict: Whether to return the data as a dictionary.
    
    Returns:
        The data stored at `path` either as a list of NumPy arrays (ordering as
        `hdf5_names`) or as a dictionary.
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
    Saves data to an HDF5 file.
    
    This function writes data to an HDF5 file, using dataset names provided in
    'hdf5_names' if 'data' is a list. If 'data' is a dictionary, its keys are used
    as dataset names. If 'overwrite' is True, existing files at the specified path
    are overwritten. If 'compression' is True, gzip compression is used, which is
    beneficial for sparse and ordered data.
    
    Args:
        data: The data to be saved, either as a list of NumPy arrays or as a
              dictionary. If list, 'hdf5_names' must be provided and have the same
              length as 'data'.
        path: Forward-slash separated path to the file where the HDF5 file will be
              saved.
        hdf5_names: The names of the datasets to be saved; required if 'data' is a
                    list and must be the same length as 'data'.
        overwrite: Whether to overwrite existing files.
        compression: Whether to use gzip compression, recommended for sparse and
                     ordered data.
    
    Raises:
        TypeError: If 'data' is a list and 'hdf5_names' is not provided.
        ValueError: If the length of 'hdf5_names' does not match the length of 'data'.
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
