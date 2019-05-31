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
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing.pool
import time
import tqdm
from . import log_mp

MyPool = multiprocessing.Pool


def parallel_process(array, function, n_jobs, use_kwargs=False, front_num=0):
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
             Useful for catching bugs
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
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
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
        for f in tqdm.tqdm(as_completed(futures), **kwargs):
            pass
    out = []
    # Get the results from the futures.
    for i, future in enumerate(futures):
        try:
            out.append(future.result())
        except Exception as e:
            msg = "In function '{}': {}".format(str(function), e)
            log_mp.error(msg)
            out.append(e)
    return front + out


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
        log_mp.debug("Computing %d parameters with %d cpus." %
                     (len(params), nb_cpus))

    start = time.time()
    if nb_cpus > 1:
        pool = MyPool(nb_cpus)
        result = pool.map(func, params)
        pool.close()
        pool.join()
    else:
        result = list(map(func, params))

    if verbose:
        log_mp.debug("Time to compute: {:.1f} min".format((time.time() -
                                                           start) / 60.))

    return result


def start_multiprocess_imap(func, params, debug=False, verbose=False,
                            nb_cpus=None, show_progress=True):
    """
    # TODO: support generator params; currently length is required for pbar
    Multiprocessing method which supports progress bar (therefore using
    imap instead of map).

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
        log_mp.debug("Computing %d parameters with %d cpus." %
                     (len(params), nb_cpus))

    start = time.time()
    if nb_cpus > 1:
        with MyPool(nb_cpus) as pool:
            if show_progress:
                result = parallel_process(params, func, nb_cpus)
            else:
                result = list(pool.map(func, params))
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


def parallel_threads(array, function, n_jobs, use_kwargs=False, front_num=0):
    """From http://danshiebler.com/2016-09-14-parallel-progress-bar/
        A parallel version of the map function with a progress bar.

        Args:
            array (array-like): An array to iterate over.
            function (function): A python function to apply to the elements of array
            n_jobs (int, default=16): The number of cores to use
            use_kwargs (boolean, default=False): Whether to consider the elements of array as dictionaries of
                keyword arguments to function
            front_num (int, default=3): The number of iterations to run serially before kicking off the parallel job.
                Useful for catching bugs
        Returns:
            [function(array[0]), function(array[1]), ...]
    """
    print("is in Parallel thread")
    #We run the first few iterations serially to catch bugs
    if front_num > 0:
        print("is in Parallel thread 1")
        front = [function(**a) if use_kwargs else function(a) for a in array[:front_num]]
    else:
        front = []
        print("is in Parallel thread 2")

    #Assemble the workers
    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        #Pass the elements of array into function

        if use_kwargs:
            futures = [executor.submit(function, **a) for a in array[front_num:]]
            print("is in Parallel thread 3")

        else:
            futures = [executor.submit(function, a) for a in array[front_num:]]
        kwargs = {
            'total': len(futures),
            'unit': 'job',
            'unit_scale': True,
            'leave': False,
            'ncols': 80,
            'dynamic_ncols': False
        }
        print("is in Parallel thread 4")

        #Print out the progress as tasks complete
        for f in tqdm.tqdm(as_completed(futures), **kwargs):
            pass
    out = []
    print("is in Parallel thread 6")

    #Get the results from the futures.
    for i, future in enumerate(futures):

        try:
            out.append(future.result())
        except Exception as e:
            out.append(e)

    print("finished in Parallel thread")

    return front + out


def start_multithread_imap(func, params, debug=False, verbose=False,
                           nb_cpus=None, show_progress=True):
    """
    Parameters
    ----------
    func : function : to be multithreaded
    params : Array: Iterable function parameters
    debug : boolean
    verbose : bool :
    nb_cpus : int : number of parallel processes at a time
    show_progress : bool

    Returns
    -------
    result: list
        list of function returns
    """
    if nb_cpus is None:
        nb_cpus = cpu_count()
    nb_cpus = min(nb_cpus, cpu_count(), len(params))
    if debug:
        nb_cpus = 1

    if verbose:
        log_mp.debug("Computing %d parameters with %d cpus." % (len(params), nb_cpus))
    start = time.time()

    if nb_cpus > 1:
        # TODO: fix show_progress flag
        # if show_progress:
        """
        pool = MyPool(nb_cpus)
        if show_progress:
            result = list(tqdm.tqdm(pool.imap(func, params), total=len(params), ncols=80, leave=False,
                                    unit='jobs', unit_scale=True, dynamic_ncols=False, mininterval=1))
        else:
            result = pool.imap(func, params)
        pool.close()
        pool.join()

        """
        if show_progress:
            result = parallel_threads(params, func, nb_cpus)
        else:
            result = list(pool.map(func, params))

    else:
        if show_progress:
            pbar = tqdm.tqdm(total=len(params), ncols=80, leave=False,
                             unit='job', unit_scale=True, dynamic_ncols=False)
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
        log_mp.debug("Computing %d parameters with %d cpus." %
                     (len(params), nb_cpus))
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
