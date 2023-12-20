# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max-Planck-Institute of Neurobiology, Munich, Germany
# Authors: Philipp Schubert, Sven Dorkenwald, Jörgen Kornfeld
import multiprocessing.pool
import time
import dill
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from typing import Callable, List, Union

import numpy as np
import tqdm

from . import log_mp

MyPool = multiprocessing.Pool


def parallel_process(array: Union[list, np.ndarray], function: Callable, n_jobs: int,
                     use_kwargs: bool = False, front_num: int = 0, show_progress: bool = True,
                     use_dill: bool = False) -> list:
    """
    Executes a function in parallel over an array with a progress bar.
    
    Args:
        array (Union[list, np.ndarray]): An array to iterate over.
        function (Callable): A python function to apply to the elements of array.
        n_jobs (int): The number of cores to use.
        use_kwargs (bool, optional): If True, elements of array are considered as dictionaries of 
            keyword arguments to function. Defaults to False.
        front_num (int, optional): The number of iterations to run serially before kicking off the 
            parallel job. Useful for catching bugs. Defaults to 0.
        show_progress (bool, optional): If True, shows progress bar. Defaults to True.
        use_dill (bool, optional): If True, uses dill to serialize data. Defaults to False.
    
    Returns:
        list: Returns a list of results from applying the function to the array.
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
        if use_dill:
            futures = [pool.submit(_run_dill_encoded, (dill.dumps((function, a)))) for a in array[front_num:]]
        elif use_kwargs:
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
        for f in tqdm.tqdm(as_completed(futures), disable=not show_progress, **kwargs):
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


def _run_dill_encoded(payload):
    """
    Decodes a dill encoded payload and executes the function with the arguments.
    
    Args:
        payload (bytes): A dill encoded payload containing a function and its arguments.
    
    Returns:
        Any: The result of the function execution.
    """
    fun, args = dill.loads(payload)
    return fun(args)


def start_multiprocess(func: Callable, params: list, debug: bool = False,
                       verbose: bool = False, nb_cpus: int = None):
    """
    Executes a function in parallel with multiple parameters.
    
    Args:
        func (Callable): The function to execute.
        params (list): The parameters for the function.
        debug (bool, optional): If True, uses only one CPU. Defaults to False.
        verbose (bool, optional): If True, prints debug information. Defaults 
        to False.
        nb_cpus (int, optional): The number of CPUs to use. Defaults to None.
    
    Returns:
        list: A list of results from the function execution.
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
                            ignore_cpu_cnt=False, desc: str = None,
                            use_dill: bool = False):
    """
    Executes a function in parallel with multiple parameters using imap.
    
    Args:
        func (Callable): The function to execute.
        params (list): The parameters for the function.
        debug (bool, optional): If True, uses only one CPU. Defaults to False.
        verbose (bool, optional): If True, prints debug information. Defaults to False.
        nb_cpus (int, optional): The number of CPUs to use. Defaults to None.
        show_progress (bool, optional): If True, shows a progress bar. Defaults to True.
        ignore_cpu_cnt (bool, optional): If True, ignores CPU count limit. Defaults to False.
        desc (str, optional): Task description. Used for progress bar. Defaults to None.
        use_dill (bool, optional): If True, uses dill to serialize data. Defaults to False.
    
    Returns:
        list: A list of results from the function execution.
    """
    if nb_cpus is None:
        nb_cpus = cpu_count()
    if ignore_cpu_cnt:
        cpu_cnt = 999999999
    else:
        cpu_cnt = cpu_count()
    if desc is None:
        if hasattr(func, '__name__'):
            desc = f'{func.__name__}'
        else:
            desc = str(func)
    nb_cpus = min(nb_cpus, len(params), cpu_cnt)

    if debug:
        nb_cpus = 1

    if verbose:
        log_mp.debug("Computing %d parameters with %d cpus." %
                     (len(params), nb_cpus))

    start = time.time()
    if nb_cpus > 1:
        result = parallel_process(params, func, nb_cpus, show_progress=show_progress, use_dill=use_dill)
    else:
        if show_progress:
            pbar = tqdm.tqdm(total=len(params), ncols=80, leave=False,
                             miniters=1, mininterval=1, unit='job',
                             unit_scale=True, dynamic_ncols=False,
                             desc=desc)
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
    Executes a function of an object in parallel with multiple parameters.
    
    Args:
        func_name (str): The name of the function to execute.
        params (list): The parameters for the function. Each element in params 
        must be an object with attribute func_name (+ optional: kwargs).
        debug (bool, optional): If True, uses only one CPU. Defaults to False.
        verbose (bool, optional): If True, prints debug information. Defaults to 
        False.
        nb_cpus (int, optional): The number of CPUs to use. Defaults to None.
    
    Returns:
        list: A list of results from the function execution. Each element in the 
        list is a function return.
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
    Helper method for multiprocessed jobs. Calls the given object method.
    
    Args:
        args (Iterable): Contains object, method name, and optional kwargs.
    
    Returns:
        Any: The result of the method execution.
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
