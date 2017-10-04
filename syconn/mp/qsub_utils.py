# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max-Planck-Institute for Medical Research, Heidelberg, Germany
# Authors: Sven Dorkenwald, Philipp Schubert, Jörgen Kornfeld

import cPickle as pkl
import getpass
import numpy as np
import os
import re
import shutil
import string
import subprocess
import sys
import time

import utils

__QSUB__ = True
try:
    with open(os.devnull, 'w') as devnull:
        subprocess.check_call('qstat', shell=True,
                                stdout=devnull, stderr=devnull)
except subprocess.CalledProcessError:
    print "QSUB not found, switching to single node multiprocessing."
    __QSUB__ = False

home_dir = os.environ['HOME'] + "/"
path_to_scripts_default = os.path.dirname(__file__)
qsub_work_folder = "%s/QSUB/" % home_dir
username = getpass.getuser()
python_path = sys.executable


def QSUB_script(params, name, queue=None, pe=None, n_cores=1, priority=0,
                sge_additional_flags='', suffix="", job_name="default",
                script_folder=None, n_max_co_processes=None):
    """
    QSUB handler - takes parameter list like normal multiprocessing job and
    runs them on the specified cluster

    IMPORTANT NOTE: the user has to make sure that queues exist and work; we
    suggest to generate multiple queues handling different workloads

    Parameters
    ----------
    params: list
        list of all parameter sets to be processed
    name: str
        name of job - specifies script with QSUB_%s % name
    queue: str or None
        queue name
    pe: str
        parallel environment name
    n_cores: int
        number of cores per job submission
    priority: int
        -1024 .. 1023, job priority, higher is more important
    sge_additional_flags: str
        additional command line flags to be passed to qsub
    suffix: str
        suffix for folder names - enables the execution of multiple qsub jobs
        for the same function
    job_name: str
        unique name for job - or just 'default' which gets changed into a
        random name automatically
    script_folder: str or None
        directory in which the QSUB_* file is located
    n_max_co_processes: int or None
        limits the number of processes that are executed on the cluster at the 
        same time; None: no limit
        
    Returns
    -------
    path_to_out: str
        path to the output directory

    """
    if job_name == "default":
        letters = string.ascii_lowercase
        job_name = "".join([letters[l] for l in
                            np.random.randint(0, len(letters), 10)])
        print "Random job_name created: %s" % job_name
    else:
        print "WARNING: running multiple jobs via qsub is only supported " \
              "with non-default job_names"

    if len(job_name) > 10:
        print "WARNING: Your job_name is longer than 10. job_names have " \
              "to be distinguishable with only using their first 10 characters."

    if script_folder is not None:
        path_to_scripts = script_folder
    else:
        path_to_scripts = path_to_scripts_default

    if os.path.exists(qsub_work_folder+"/%s_folder%s/" % (name, suffix)):
        shutil.rmtree(qsub_work_folder+"/%s_folder%s/" % (name, suffix))

    path_to_script = path_to_scripts + "/QSUB_%s.py" % (name)
    path_to_storage = qsub_work_folder+"/%s_folder%s/storage/" % (name, suffix)
    path_to_sh = qsub_work_folder+"/%s_folder%s/sh/" % (name, suffix)
    path_to_log = qsub_work_folder+"/%s_folder%s/log/" % (name, suffix)
    path_to_err = qsub_work_folder+"/%s_folder%s/err/" % (name, suffix)
    path_to_out = qsub_work_folder+"/%s_folder%s/out/" % (name, suffix)

    if pe is not None:
        sge_queue_option = "-pe %s %d" % (pe, n_cores)
    elif queue is not None:
        sge_queue_option = "-q %s" % queue
    else:
        raise Exception("No queue or parallel environment defined")

    if not os.path.exists(path_to_storage):
        os.makedirs(path_to_storage)
    if not os.path.exists(path_to_sh):
        os.makedirs(path_to_sh)
    if not os.path.exists(path_to_log):
        os.makedirs(path_to_log)
    if not os.path.exists(path_to_err):
        os.makedirs(path_to_err)
    if not os.path.exists(path_to_out):
        os.makedirs(path_to_out)

    print "Number of jobs:", len(params)

    time_start = time.time()
    last_diff_rp = 0
    for i_job in range(len(params)):
        if n_max_co_processes is not None:
            while last_diff_rp == 0:
                nb_rp = number_of_running_processes(job_name)
                last_diff_rp = n_max_co_processes - nb_rp

                if last_diff_rp == 0:
                    progress = float(i_job - n_max_co_processes) / len(params) * 100
                    print 'Progress: %.2f%% in %.2fs' % \
                          (progress, time.time() - time_start)
                    time.sleep(5.)

            last_diff_rp -= 1

        this_storage_path = path_to_storage+"job_%d.pkl" % i_job
        this_sh_path = path_to_sh+"job_%d.sh" % i_job
        this_out_path = path_to_out+"job_%d.pkl" % i_job
        job_log_path = path_to_log + "job_%d.log" % i_job
        job_err_path = path_to_err + "job_%d.log" % i_job

        with open(this_sh_path, "w") as f:
            f.write("#!/bin/bash\n")
            f.write("{0} {1} {2} {3}".format(python_path,
                                             path_to_script,
                                             this_storage_path,
                                             this_out_path))

        with open(this_storage_path, "wb") as f:
            for param in params[i_job]:
                pkl.dump(param, f)

        os.chmod(this_sh_path, 0744)

        subprocess.call("qsub {0} -o {1} -e {2} -N {3} -p {4} {5} {6}".format(
            sge_queue_option,
            job_log_path,
            job_err_path,
            job_name,
            priority,
            sge_additional_flags,
            this_sh_path), shell=True)

    print "All jobs are submitted: %s" % name
    while True:
        if show_progress(job_name, len(params), time.time() - time_start):
            break
        time.sleep(5.)

    return path_to_out


def number_of_running_processes(job_name):
    """
    Calculates the number of running jobs using qstat

    Parameters
    ----------
    job_name: str
        job_name as shown in qstats

    Returns
    -------
    nb_jobs: int
        number of running jobs

    """
    process = subprocess.Popen("qstat -u %s" % username,
                               shell=True, stdout=subprocess.PIPE)
    nb_lines = 0
    for line in iter(process.stdout.readline, ''):
        if job_name[:10] in line:
            nb_lines += 1
    return nb_lines


def show_progress(job_name, n_jobs_total, time_diff):
    """
    Prints progress for specific qsub job

    Parameters
    ----------
    job_name: str
        job_name as shown in qstats
    n_jobs_submitted: int
        number of submitted jobs
    time_diff: float
        time since starting the job

    Returns
    -------
    finished: bool
        True of no jobs are running anymore; False otherwise
    """
    nb_rp = number_of_running_processes(job_name)

    if nb_rp == 0:
        sys.stdout.write('\rAll jobs were finished in %.2fs\n' % time_diff)
        return True
    else:
        progress = 100 * (n_jobs_total - utils.negative_to_zero(nb_rp)) / \
                   float(n_jobs_total)
        sys.stdout.write('\rProgress: %.2f%% in %.2fs' % (progress, time_diff))
        sys.stdout.flush()
        return False


def delete_jobs_by_name(job_name):
    """
    Deletes a group of jobs that have the same name

    Parameters
    ----------
    job_name: str
        job_name as shown in qstats

    Returns
    -------

    """
    process = subprocess.Popen("qstat -u %s" % username, shell=True,
                               stdout=subprocess.PIPE)
    job_ids = []
    for line in iter(process.stdout.readline, ''):
        if job_name[:10] in line:
            job_ids.append(re.findall("[\d]+", line)[0])

    command = "qdel "
    for job_id in job_ids:
        command += job_id + ", "
    command = command[:-2]

    subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
