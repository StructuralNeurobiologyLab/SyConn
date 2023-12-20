# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max-Planck-Institute of Neurobiology, Munich, Germany
# Authors: Philipp Schubert

import os
import shutil
import subprocess


def example_run():
    """
    This function performs a full test run by executing the 'start.py' script from the examples 
    directory. It uses a subprocess call to prevent the threaded ('forked') parts in the pipeline 
    from freezing. The function creates a working directory, runs the script, and then removes 
    the directory. It also handles the environment variable 'syconn_wd' if it exists. 
    
    Returns:
        int: The return code of the subprocess if it is 0, else it returns the error.
    """
    example_cube_id = 1
    working_dir = f"~/SyConn/tests/example_cube{example_cube_id}_{os.getpid()}/"
    example_wd = os.path.expanduser(working_dir) + "/"
    shutil.rmtree(example_wd, ignore_errors=True)

    startpy_fname = os.path.dirname(os.path.realpath(__file__)) + '/../examples/start.py'

    process = subprocess.Popen(
        ["python", startpy_fname, f"--working_dir={working_dir}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    out, err = process.communicate()

    if os.environ.get('syconn_wd') is not None:
        del os.environ['syconn_wd']
    shutil.rmtree(example_wd, ignore_errors=True)
    return process.returncode if process.returncode == 0 else err


def test_example_run():
    """
    This function tests the 'example_run' function. It calls the 'example_run' function and raises a 
    RuntimeError if the return value is not 0.
    
    Raises:
        RuntimeError: If the return value of 'example_run' is not 0.
    """
    ret = example_run()
    if ret != 0:
        raise RuntimeError(ret)


if __name__ == '__main__':
    test_example_run()
