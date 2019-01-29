# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max-Planck-Institute of Neurobiology, Munich, Germany
# Authors: Philipp Schubert, Sven Dorkenwald, Jörgen Kornfeld

try:
    import cPickle as pkl
except ImportError:
    import pickle as pkl
import getpass
from multiprocessing import cpu_count, Process
import multiprocessing
import multiprocessing.pool
import os
import shutil
import subprocess
import sys
import time
import tqdm
from . import log_mp


home_dir = os.environ['HOME'] + "/"
path_to_scripts_default = os.path.dirname(__file__)
subp_work_folder = "%s/SUBP/" % home_dir
username = getpass.getuser()
python_path = sys.executable

# with py 3.6 the pool class was refactored and NoDaemonProcess impl. are not that straight forward
#  anymore, see: https://stackoverflow.com/questions/6974695/python-process-pool-non-daemonic?rq=1
if not (sys.version_info[0] == 3 and sys.version_info[1] > 5):
    # found NoDaemonProcess on stackexchange by Chris Arndt - enables
    # hierarchical multiprocessing
    class NoDaemonProcess(Process):
        # make 'daemon' attribute always return False
        def _get_daemon(self):
            return False

        def _set_daemon(self, value):
            pass

        daemon = property(_get_daemon, _set_daemon)


    # We sub-class multi_proc.pool.Pool instead of multi_proc.Pool
    # because the latter is only a wrapper function, not a proper class.
    class MyPool(multiprocessing.pool.Pool):
        Process = NoDaemonProcess
else:
    class NoDaemonProcess(multiprocessing.Process):
        @property
        def daemon(self):
            return False

        @daemon.setter
        def daemon(self, value):
            pass

    class NoDaemonContext(type(multiprocessing.get_context())):
        Process = NoDaemonProcess

    # We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
    # because the latter is only a wrapper function, not a proper class.
    class MyPool(multiprocessing.pool.Pool):
        def __init__(self, *args, **kwargs):
            kwargs['context'] = NoDaemonContext()
            super(MyPool, self).__init__(*args, **kwargs)


def start_multiprocess(func, params, debug=False, verbose=False, nb_cpus=None):
    """

    Parameters
    ----------
    func : function
    params : function parameters
    debug : boolean
    verbose : bool
    nb_cpus : int

    Returns
    -------
    result: list
        list of function returns
    """
    if nb_cpus is None:
        nb_cpus = cpu_count()

    if debug:
        nb_cpus = 1

    nb_cpus = min(nb_cpus, len(params), cpu_count())

    if verbose:
        log_mp.debug("Computing %d parameters with %d cpus." % (len(params), nb_cpus))

    start = time.time()
    if nb_cpus > 1:
        pool = MyPool(nb_cpus)
        result = pool.map(func, params)
        pool.close()
        pool.join()
    else:
        result = list(map(func, params))

    if verbose:
        log_mp.debug("Time to compute: {:.1f} min".format((time.time() - start) / 60.))

    return result


def start_multiprocess_imap(func, params, debug=False, verbose=False,
                            nb_cpus=None, show_progress=True):
    """
    Multiprocessing method which supports progress bar (therefore using
    imap instead of map). # TODO: support generator params

    Parameters
    ----------
    func : function
    params : Iterable
        function parameters
    debug : boolean
    verbose : bool
    nb_cpus : int
    show_progress : bool

    Returns
    -------
    result: list
        list of function returns
    """
    if nb_cpus is None:
        nb_cpus = cpu_count()

    nb_cpus = min(nb_cpus, len(params), cpu_count())

    if debug:
        nb_cpus = 1

    if verbose:
        log_mp.debug("Computing %d parameters with %d cpus." % (len(params), nb_cpus))

    start = time.time()
    if nb_cpus > 1:
        with MyPool(nb_cpus) as pool:
            if show_progress:
                result = list(tqdm.tqdm(pool.imap(func, params), total=len(params),
                                        ncols=80, leave=False, unit='jobs',
                                        unit_scale=True, dynamic_ncols=False,
                                        mininterval=0.5))
            else:
                result = list(pool.imap(func, params))
    else:
        if show_progress:
            pbar = tqdm.tqdm(total=len(params), ncols=80, leave=False,
                             mininterval=0.5, unit='jobs', unit_scale=True,
                             dynamic_ncols=False)
            result = []
            for p in params:
                result.append(func(p))
                pbar.update(1)
            pbar.close()
        else:
            result = []
            for p in params:
                result.append(func(p))
    if verbose:
        log_mp.debug("Time to compute: {:.1f} min".format((time.time() - start) / 60.))

    return result


def start_multiprocess_obj(func_name, params, debug=False, verbose=False,
                           nb_cpus=None):
    """

    Parameters
    ----------
    func_name : str
    params : List[List]
        each element in params must be object with attribute func_name
        (+ optional: kwargs)
    debug : boolean
    verbose : bool
    nb_cpus : int

    Returns
    -------
    result: List
        list of function returns
    """
    if nb_cpus is None:
        nb_cpus = cpu_count()

    if debug:
        nb_cpus = 1

    nb_cpus = min(nb_cpus, len(params), cpu_count())

    if verbose:
        log_mp.debug("Computing %d parameters with %d cpus." % (len(params), nb_cpus))
    for el in params:
        el.insert(0, func_name)
    start = time.time()
    if nb_cpus > 1:
        pool = MyPool(nb_cpus)
        result = pool.map(multi_helper_obj, params)
        pool.close()
        pool.join()
    else:
        result = list(map(multi_helper_obj, params))
    if verbose:
        log_mp.debug("Time to compute: {:.1f} min".format((time.time() - start) / 60.))
    return result


def multi_helper_obj(args):
    """
    Generic helper emthod for multiprocessed jobs. Calls the given object
    method.

    Parameters
    ----------
    args : iterable
        object, method name, optional: kwargs

    Returns
    -------

    """
    attr_str = args[0]
    obj = args[1]
    if len(args) == 3:
        kwargs = args[2]
    else:
        kwargs = {}
    attr = getattr(obj, attr_str)
    # check if attr is callable, i.e. a method to be called
    if not hasattr(attr, '__call__'):
        return attr
    return attr(**kwargs)
