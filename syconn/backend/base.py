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
import os
import warnings
import time
import shutil
from ..handler.basics import write_obj2pkl, load_pkl2obj
__all__ = ['LZ4DictBase', 'BTBase']
# TODO: adapt to new class interface all-over syconn


class StorageBase(dict):
    """
    Base class for data interface with all attributes and methods necessary:
    input parameter for init:
    identifier, read_only=True, disable_locking=False
    """
    def __init__(self, cache_decomp):
        super(StorageBase, self).__init__()
        self._cache_decomp = cache_decomp
        self._cache_dc = {}
        self._dc_intern = {}

    def __delitem__(self, key):
        raise NotImplementedError

    def __del__(self):
        raise NotImplementedError

    def __len__(self):
        return self._dc_intern.__len__()

    def __eq__(self, other):
        if not isinstance(other, LZ4DictBase):
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

    # TODO: make iter interface standard for above items, values,
    # keys generators and remove explicit iter interface
    def iteritems(self):
        for k in self.keys():
            yield k, self[k]

    def itervalues(self):
        for k in self.keys():
            yield self[k]

    def iterkeys(self):
        for k in self.keys():
            yield k

    def push(self, dest=None):
        raise NotImplementedError

    def pull(self, source=None):
        raise NotImplementedError


# ---------------------------- BT
# ------------------------------------------------------------------------------
class BTBase(StorageBase):
    def __init__(self, inp_p, cache_decomp=False, read_only=True):
        super(BTBase, self).__init__(cache_decomp)
        pass

    def __eq__(self, other):
        if not isinstance(other, BTBase):
            return False
        return self._dc_intern.__eq__(other._dc_intern)


# ---------------------------- lz4
# ------------------------------------------------------------------------------
class LZ4DictBase(StorageBase):
    """
    Customized dictionary to store compressed numpy arrays, but with a
    intuitive user interface, i.e. compression will happen in background.
    kwarg 'cache_decomp' can be enabled to cache decompressed arrays
    additionally (save decompressing time when accessing items frequently).
    """
    def __init__(self, inp_p, cache_decomp=False, read_only=True,
                 max_delay=100, timeout=1000, disable_locking=not LOCKING,
                 max_nb_attempts=10):
        super(LZ4DictBase, self).__init__(cache_decomp)
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

    def __del__(self):
        if self.a_lock is not None and self.a_lock.acquired:
            self.a_lock.release()

    def __len__(self):
        return self._dc_intern.__len__()

    def __eq__(self, other):
        if not isinstance(other, LZ4DictBase):
            return False
        return self._dc_intern.__eq__(other._dc_intern)

    # TODO: make above items, values, keys generators
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