# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max-Planck-Institute for Medical Research, Heidelberg, Germany
# Authors: Sven Dorkenwald, Philipp Schubert, Jörgen Kornfeld

try:
    import cPickle as pkl
# TODO: switch to Python3 at some point and remove above
except Exception:
    import pickle as pkl
import getpass
from multiprocessing import cpu_count, Process
import multiprocessing.pool
import os
import shutil
import subprocess
import sys
import time
import tqdm


home_dir = os.environ['HOME'] + "/"
path_to_scripts_default = os.path.dirname(__file__)
subp_work_folder = "%s/SUBP/" % home_dir
username = getpass.getuser()
python_path = sys.executable


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

    if nb_cpus is None:
        nb_cpus = max(cpu_count(), 1)

    if debug:
        nb_cpus = 1

    if verbose:
        print("Computing %d parameters with %d cpus." % (len(params), nb_cpus))

    start = time.time()
    if nb_cpus > 1:
        pool = MyPool(nb_cpus)
        result = pool.map(func, params)
        pool.close()
        pool.join()
    else:
        result = map(func, params)

    if verbose:
        print("\nTime to compute:", time.time() - start)

    return result


def start_multiprocess_imap(func, params, debug=False, verbose=False,
                            nb_cpus=None, show_progress=True):
    """

    Parameters
    ----------
    func : function
    params : function parameters
    debug : boolean
    verbose : bool
    nb_cpus : int
    show_progress : bool

    Returns
    -------
    result: list
        list of function returns
    """
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

    if nb_cpus is None:
        nb_cpus = max(cpu_count(), 1)

    if debug:
        nb_cpus = 1


    if verbose:
        print("Computing %d parameters with %d cpus." % (len(params), nb_cpus))

    start = time.time()
    if nb_cpus > 1:
        pool = MyPool(nb_cpus)
        if show_progress:
            result = list(tqdm.tqdm(pool.imap(func, params), total=len(params), ncols=80, leave=False,
                             unit='jobs', unit_scale=True, dynamic_ncols=False))
        else:
            result = pool.imap(func, params)
        pool.close()
        pool.join()
    else:
        if show_progress:
            pbar = tqdm.tqdm(total=len(params), ncols=80, leave=False, mininterval=1,
                                    unit='jobs', unit_scale=True, dynamic_ncols=False)
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
        print("\nTime to compute:", time.time() - start)

    return result


def start_multiprocess_obj(func_name, params, debug=False, verbose=False,
                           nb_cpus=None):
    """

    Parameters
    ----------
    func_name : str
    params : list of list
        each element in params must be object with attribute func_name 
        (+ optional: kwargs)
    debug : boolean
    verbose : bool
    nb_cpus : int

    Returns
    -------
    result: list
        list of function returns
    """
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

    if nb_cpus is None:
        nb_cpus = max(cpu_count(), 1)

    if debug:
        nb_cpus = 1

    if verbose:
        print("Computing %d parameters with %d cpus." % (len(params), nb_cpus))
    for el in params:
        el.insert(0, func_name)
    start = time.time()
    if nb_cpus > 1:
        pool = MyPool(nb_cpus)
        result = pool.map(multi_helper_obj, params)
        pool.close()
        pool.join()
    else:
        result = map(multi_helper_obj, params)

    if verbose:
        print("\nTime to compute:", time.time() - start)

    return result



def SUBP_script(params, name, suffix="", delay=0):
    """
    Runs multiple subprocesses on one node at the same time - no load
    balancing, all jobs get executed right away (or with a specified delay)

    Parameters
    ----------
    params: list
        list of all paramter sets to be processed
    name: str
        name of job - specifies script with QSUB_%s % name
    suffix: str
        suffix for folder names - enables the execution of multiple subp jobs
        for the same function
    delay: int
        delay between executions in seconds

    Returns
    -------
    path_to_out: str
        path to the output directory

    """
    if os.path.exists(subp_work_folder + "/%s_folder%s/" % (name, suffix)):
        shutil.rmtree(subp_work_folder + "/%s_folder%s/" % (name, suffix))

    path_to_script = path_to_scripts_default + "/QSUB_%s.py" % (name)
    path_to_storage = subp_work_folder + "/%s_folder%s/storage/" % (name,
                                                                    suffix)
    path_to_out = subp_work_folder + "/%s_folder%s/out/" % (name, suffix)

    if not os.path.exists(path_to_storage):
        os.makedirs(path_to_storage)
    if not os.path.exists(path_to_out):
        os.makedirs(path_to_out)

    processes = []
    for ii in range(len(params)):
        this_storage_path = path_to_storage + "job_%d.pkl" % ii
        this_out_path = path_to_out + "job_%d.pkl" % ii

        with open(this_storage_path, "wb") as f:
            for param in params[ii]:
                pkl.dump(param, f)

        p = subprocess.Popen("%s %s %s %s" % (python_path, path_to_script,
                                              this_storage_path, this_out_path),
                             shell=True)
        processes.append(p)
        time.sleep(delay)

    for p in processes:
        p.wait()

    return path_to_out


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