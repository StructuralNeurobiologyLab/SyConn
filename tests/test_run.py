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
    Full test run. Executes examples/start.py. Subprocess call is required to prevent the threaded ('forked') parts
    in the pipeline from freezing.
    """
    example_cube_id = 1
    working_dir = f"~/SyConn/tests/example_cube{example_cube_id}_{os.getpid()}/"
    example_wd = os.path.expanduser(working_dir) + "/"
    shutil.rmtree(example_wd, ignore_errors=True)

    startpy_fname = os.path.dirname(os.path.realpath(__file__)) + '/../examples/start.py'

    process = subprocess.Popen(
        ["python", startpy_fname, f"--working_dir={working_dir}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
    )
    out, _ = process.communicate()

    if os.environ.get('syconn_wd') is not None:
        del os.environ['syconn_wd']
    shutil.rmtree(example_wd, ignore_errors=True)
    return process.returncode


def test_example_run():
    assert example_run() == 0


if __name__ == '__main__':
    test_example_run()
