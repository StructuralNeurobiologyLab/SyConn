# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max-Planck-Institute of Neurobiology, Munich, Germany
# Authors: Philipp Schubert, Sven Dorkenwald, Jörgen Kornfeld
import multiprocessing.pool
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from typing import Callable, List, Union

import numpy as np
import tqdm

from . import log_mp

MyPool = multiprocessing.Pool


def parallel_process(array: Union[list, np.ndarray], function: Callable, n_jobs: int,
                     use_kwargs: bool = False, front_num: int = 0, show_progress: bool = True) -> list:
    """From http://danshiebler.com/2016-09-14-parallel-progress-bar/
     A parallel version of the map function with a progress bar.

    Args:
        array (array-like): An array to iterate over.
        function (function): A python function to apply to the elements of
            array n_jobs (int, default=16): The number of cores to use
        use_kwargs (boolean, default=False): Whether to consider the
            elements of array as dictionaries of keyword arguments to function
        front_num (int, default=3): The number of iterations to run
            serially before kicking off the parallel job.
            Useful for catching bugs.
        n_jobs:
        show_progress: show progress

    Returns:
        [function(array[0]), function(array[1]), ...]
    """
    # We run the first few iterations serially to catch bugs
    if front_num > 0:
        front = [function(**a) if use_kwargs else function(a) for a
                 in array[:front_num]]
    else:
        front = []
    # Assemble the workers
    pool = ProcessPoolExecutor(max_workers=n_jobs)
    try:
        # Pass the elements of array into function
        if use_kwargs:
            futures = [pool.submit(function, **a) for a in array[front_num:]]
        else:
            futures = [pool.submit(function, a) for a in array[front_num:]]
        kwargs = {
            'total': len(futures),
            'unit': 'job',
            'unit_scale': True,
            'leave': False,
            'ncols': 80,
            'dynamic_ncols': False,
            'miniters': 1,
            'mininterval': 1
        }
        # Print out the progress as tasks complete
        for f in tqdm.tqdm(as_completed(futures), disable=show_progress, **kwargs):
            pass
    finally:
        pool.shutdown()
    out = []
    # Get the results from the futures.
    for i, future in enumerate(futures):
        try:
            out.append(future.result())
        except Exception as e:
            msg = "In function '{}': {}".format(str(function), e)
            log_mp.error(msg)
            raise Exception(e)
            out.append(e)
    return front + out


def start_multiprocess(func: Callable, params: list, debug: bool = False,
                       verbose: bool = False, nb_cpus: int = None):
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
        log_mp.debug("Computing %d parameters with %d cpus." %
                     (len(params), nb_cpus))

    start = time.time()
    if nb_cpus > 1:
        pool = MyPool(nb_cpus)
        try:
            result = pool.map(func, params)
        finally:
            pool.close()
            pool.join()
    else:
        result = list(map(func, params))

    if verbose:
        log_mp.debug("Time to compute: {:.1f} min".format((time.time() -
                                                           start) / 60.))

    return result


def start_multiprocess_imap(func: Callable, params, debug=False, verbose=False,
                            nb_cpus=None, show_progress=True,
                            ignore_cpu_cnt=False):
    """

    Args:
        func:
        params:
        debug:
        verbose:
        nb_cpus:
        show_progress:
        ignore_cpu_cnt:

    Returns:
        list of function returns.
    """
    if nb_cpus is None:
        nb_cpus = cpu_count()
    if ignore_cpu_cnt:
        cpu_cnt = 999999999
    else:
        cpu_cnt = cpu_count()
    nb_cpus = min(nb_cpus, len(params), cpu_cnt)

    if debug:
        nb_cpus = 1

    if verbose:
        log_mp.debug("Computing %d parameters with %d cpus." %
                     (len(params), nb_cpus))

    start = time.time()
    if nb_cpus > 1:
        result = parallel_process(params, func, nb_cpus, show_progress=show_progress)
    else:
        if show_progress:
            pbar = tqdm.tqdm(total=len(params), ncols=80, leave=False,
                             miniters=1, mininterval=1, unit='job',
                             unit_scale=True, dynamic_ncols=False)
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
        log_mp.debug("Time to compute: {:.1f} min".format((time.time() -
                                                           start) / 60.))

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
        log_mp.debug("Computing %d parameters with %d cpus." %
                     (len(params), nb_cpus))
    for el in params:
        el.insert(0, func_name)
    start = time.time()
    if nb_cpus > 1:
        pool = MyPool(nb_cpus)
        try:
            result = pool.map(multi_helper_obj, params)
        finally:
            pool.close()
            pool.join()
    else:
        result = list(map(multi_helper_obj, params))
    if verbose:
        log_mp.debug("Time to compute: {:.1f} min".format((time.time() -
                                                           start) / 60.))
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
