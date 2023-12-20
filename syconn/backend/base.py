# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Philipp Schubert, Sven Dorkenwald, Joergen Kornfeld

import os
import shutil
import time
from pickle import UnpicklingError

from .. import global_params
from ..extraction import log_extraction
from ..handler.basics import write_obj2pkl, load_pkl2obj

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

__all__ = ['FSBase', 'BTBase']


class StorageBase(dict):
    """
    Interface class for data Input/Output operations in SyConn. This class is a dictionary-like 
    object that provides an interface for data storage and retrieval. It is designed to work with 
    compressed data and provides methods for data caching and decompression.
    
    Attributes:
        _cache_decomp (bool): Flag indicating whether to cache decompressed arrays.
        _cache_dc (dict): Cache for decompressed arrays.
        _dc_intern (dict): Internal dictionary for data storage.
    """

    def __init__(self, cache_decomp):
        """
        Initializes the StorageBase object.
        
        Args:
            cache_decomp (bool): Flag indicating whether to cache decompressed arrays.
        """
        super(StorageBase, self).__init__()
        self._cache_decomp = cache_decomp
        self._cache_dc = {}
        self._dc_intern = {}

    def __getitem__(self, key):
        """
        Retrieves an item from the storage. This method needs to be implemented in subclasses.
        
        Args:
            key (str): The key of the item to retrieve.
        """
        raise NotImplementedError

    def __setitem__(self, key, value):
        """
        Sets an item in the storage. This method needs to be implemented in subclasses.
        
        Args:
            key (str): The key of the item to set.
            value (any): The value of the item to set.
        """
        raise NotImplementedError

    def __delitem__(self, key):
        """
        Deletes an item from the storage. This method needs to be implemented in subclasses.
        
        Args:
            key (str): The key of the item to delete.
        """
        raise NotImplementedError

    def __del__(self):
        """
        Deletes the storage object. This method needs to be implemented in subclasses.
        """
        raise NotImplementedError

    def __len__(self):
        """
        Returns the number of items in the storage.
        
        Returns:
            int: The number of items in the storage.
        """
        return self._dc_intern.__len__()

    def __eq__(self, other):
        """
        Checks if the storage is equal to another storage.
        
        Args:
            other (StorageBase): The other storage to compare with.
        
        Returns:
            bool: True if the storages are equal, False otherwise.
        """
        if not isinstance(other, StorageBase):
            return False
        return self._dc_intern.__eq__(other._dc_intern)

    def __ne__(self, other):
        """
        Checks if the storage is not equal to another storage.
        
        Args:
            other (StorageBase): The other storage to compare with.
        
        Returns:
            bool: True if the storages are not equal, False otherwise.
        """
        return not self.__eq__(other)

    def __contains__(self, item):
        """
        Checks if an item is in the storage.
        
        Args:
            item (str): The key of the item to check.
        
        Returns:
            bool: True if the item is in the storage, False otherwise.
        """
        return self._dc_intern.__contains__(item)

    def __iter__(self):
        """
        Returns an iterator over the keys in the storage.
        
        Returns:
            iterator: An iterator over the keys in the storage.
        """
        return iter(self._dc_intern)

    def __repr__(self):
        """
        Returns a string representation of the storage.
        
        Returns:
            str: A string representation of the storage.
        """
        return self._dc_intern.__repr__()

    def update(self, other, **kwargs):
        """
        Updates the storage with the items from another storage. This method needs to be implemented 
        in subclasses.
        
        Args:
            other (StorageBase): The other storage to update from.
            **kwargs: Additional keyword arguments.
        """
        raise NotImplementedError

    def copy(self):
        """
        Returns a copy of the storage. This method needs to be implemented in subclasses.
        
        Returns:
            StorageBase: A copy of the storage.
        """
        raise NotImplementedError

    def items(self):
        """
        Returns an iterator over the items (key-value pairs) in the storage.
        
        Returns:
            iterator: An iterator over the items in the storage.
        """
        for k in self._dc_intern.keys():
            yield k, self[k]

    def values(self):
        """
        Returns an iterator over the values in the storage.
        
        Returns:
            iterator: An iterator over the values in the storage.
        """
        for k in self._dc_intern.keys():
            yield self[k]

    def keys(self):
        """
        Returns an iterator over the keys in the storage.
        
        Returns:
            iterator: An iterator over the keys in the storage.
        """
        return self._dc_intern.keys()

    def push(self, dest=None):
        """
        Pushes the data to a destination. This method needs to be implemented in subclasses.
        
        Args:
            dest (str, optional): The destination to push the data to.
        """
        raise NotImplementedError

    def pull(self, source=None):
        """
        Pulls the data from a source. This method needs to be implemented in subclasses.
        
        Args:
            source (str, optional): The source to pull the data from.
        """
        raise NotImplementedError


# ---------------------------- BT
# ------------------------------------------------------------------------------
class BTBase(StorageBase):
    """
    BTBase is a subclass of StorageBase that provides an interface for data storage and retrieval 
    in a BTree-like structure. It is designed to work with compressed data and provides methods 
    for data caching and decompression.
    
    Attributes:
        identifier (str): Identifier for the BTBase instance.
        cache_decomp (bool): Flag indicating whether to cache decompressed arrays.
        read_only (bool): Flag indicating whether the BTBase instance is read-only.
        disable_locking (bool): Flag indicating whether to disable locking.
    """
    def __init__(self, identifier, cache_decomp=False, read_only=True,
                 disable_locking=False):
        """
        Initializes the BTBase object.
        
        Args:
            identifier (str): Identifier for the BTBase instance.
            cache_decomp (bool, optional): Flag indicating whether to cache decompressed arrays.
            read_only (bool, optional): Flag indicating whether the BTBase instance is read-only.
            disable_locking (bool, optional): Flag indicating whether to disable locking.
        """
        # likely 'cache_decomp' not necessary, but needed to match interface of LZ4Dicts
        super(BTBase, self).__init__(cache_decomp=False)
        pass

    def __eq__(self, other):
        """
        Checks if the BTBase instance is equal to another BTBase instance.
        
        Args:
            other (BTBase): The other BTBase instance to compare with.
        
        Returns:
            bool: True if the BTBase instances are equal, False otherwise.
        """
        if not isinstance(other, BTBase):
            return False
        return self._dc_intern.__eq__(other._dc_intern)


# ---------------------------- lz4
# ------------------------------------------------------------------------------
class FSBase(StorageBase):
    """
    This class is a customized dictionary that stores compressed numpy arrays. The compression process 
    happens in the background, providing an intuitive user interface. The 'cache_decomp' kwarg can be 
    enabled to cache decompressed arrays, saving decompression time when accessing items frequently.
    """

    def __init__(self, inp_p: str, cache_decomp: bool = False,
                 read_only: bool = True, max_delay: int = 100,
                 timeout: int = 1000, disable_locking: bool = True,
                 max_nb_attempts: int = 100):
        """
        Initializes the FSBase object.
        
        Args:
            inp_p (str): Path to the file.
            cache_decomp (bool, optional): If True, caches deserialized arrays. 
                                           Defaults to False.
            read_only (bool, optional): If True and locking is enabled, no semaphore 
                                        will be placed. Defaults to True.
            max_delay (int, optional): Delay between attempts. Defaults to 100.
            timeout (int, optional): Throws `RuntimeError` after `timeout` seconds. 
                                     Defaults to 1000.
            disable_locking (bool, optional): If True, disables file locking. 
                                              Defaults to True.
            max_nb_attempts (int, optional): Maximum number of total attempts. 
                                             Defaults to 100.
        """
        super(FSBase, self).__init__(cache_decomp)
        if not LOCKING and not disable_locking:
            log_extraction.warning('Locking could not be enabled due to missing "fasteners" package.')
            disable_locking = True
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
            if type(inp_p) is str:
                self.pull(inp_p)
            else:
                msg = "Unsupported initialization type {} for 'FSBase'.".format(type(inp_p))
                log_extraction.error(msg)
                raise NotImplementedError(msg)

    def __delitem__(self, key):
        """
        Deletes the item with the specified key from the dictionary.
        """
        try:
            del self[key]
        except KeyError:
            msg = "No such attribute {} in dict at {}. Existing keys:" \
                  " {}.".format(key, self._path, list(self.keys()))
            log_extraction.error(msg)
            raise AttributeError(msg)

    def __del__(self):
        """
        Deletes the object and releases any acquired locks.
        """
        if self.a_lock is not None and self.a_lock.acquired:
            self.a_lock.release()
        del self._dc_intern, self._cache_dc

    def __len__(self):
        """
        Returns the number of items in the dictionary.
        """
        return self._dc_intern.__len__()

    def __eq__(self, other):
        """
        Checks if the current object is equal to the other object.
        """
        if not isinstance(other, FSBase):
            return False
        return self._dc_intern.__eq__(other._dc_intern)

    def __ne__(self, other):
        """
        Checks if the current object is not equal to the other object.
        """
        return not self.__eq__(other)

    def __contains__(self, item):
        """
        Checks if the dictionary contains the specified item.
        """
        return self._dc_intern.__contains__(item)

    def __iter__(self):
        """
        Returns an iterator for the dictionary.
        """
        return iter(self._dc_intern)

    def __repr__(self):
        """
        Returns a string representation of the dictionary.
        """
        return self._dc_intern.__repr__()

    def update(self, other, **kwargs):
        """
        Updates the dictionary with the key-value pairs from other. This method is not implemented.
        """
        raise NotImplementedError

    def copy(self):
        """
        Returns a copy of the dictionary. This method is not implemented.
        """
        raise NotImplementedError

    def items(self):
        """
        Returns a generator that yields the key-value pairs in the dictionary.
        """
        for k in self._dc_intern.keys():
            yield k, self[k]

    def values(self):
        """
        Returns a generator that yields the values in the dictionary.
        """
        for k in self._dc_intern.keys():
            yield self[k]

    def keys(self):
        """
        Returns the keys in the dictionary.
        """
        return self._dc_intern.keys()

    def push(self, dest: str = None):
        """
        Pushes data to the specified destination.
        
        Args:
            dest (str, optional): The storage destination. Defaults to None.
        """
        if dest is None:
            dest = self._path
        if dest is None:  # support virtual / temporary SSO objects
            log_extraction.warning('"push" called but Storage object was initialized '
                                   'with "None". Content will not be written.')
            return
        write_obj2pkl(dest, self._dc_intern)
        if not self.read_only and not self.disable_locking:
            self.a_lock.release()

    def pull(self, source: str = None):
        """
        Fetches data from the specified source.
        
        Args:
            source (str, optional): The source location. Defaults to None.
        """
        if source is None:
            source = self._path
        fold, fname = os.path.split(source)
        lock_path = fold + "/." + fname + ".lk"
        # only create directory if read_only is false. -> support virtual SSO
        if not os.path.isdir(fold) and not self.read_only:
            try:
                os.makedirs(fold)
            except OSError as e:  # if to jobs create the folder at the same time
                log_extraction.warning("Tried to create folder of dict {}, but it already existed. "
                                       "Multiple jobs might work on same chunk almost"
                                       " simultaneously. Error {}.".format(self._path, e))
                pass
        # acquires lock until released when saving or after loading if self.read_only
        if not self.disable_locking:
            self.a_lock = fasteners.InterProcessLock(lock_path)
            nb_attempts = 1
            start = time.time()
            while True:
                try:
                    gotten = self.a_lock.acquire(blocking=True, delay=0.1,
                                                 max_delay=self.max_delay,
                                                 timeout=self.timeout / self._max_nb_attempts)
                except ValueError:
                    gotten = False
                # if not gotten and maximum attempts not reached yet keep trying
                if not gotten and (nb_attempts < self._max_nb_attempts):
                    nb_attempts += 1
                else:
                    break
            if not gotten:
                msg = "Unable to acquire file lock for {} after {:.0f}s.".format(source, time.time() - start)
                log_extraction.warning(msg)
                raise RuntimeError(msg)
        if os.path.isfile(source):
            try:
                self._dc_intern = load_pkl2obj(source)
            except (UnpicklingError, EOFError) as e:
                log_extraction.warning("Could not load LZ4Dict ({}). 'push' will"
                                       " overwrite broken .pkl file: {}.".format(self._path, e))
                self._dc_intern = {}
        else:
            self._dc_intern = {}
        if self.read_only and not self.disable_locking:
            self.a_lock.release()
