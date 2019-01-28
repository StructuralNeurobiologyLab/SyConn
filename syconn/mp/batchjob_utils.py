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
import glob
import numpy as np
import os
import io
import re
import shutil
import string
import subprocess
import tqdm
import sys
import time

from ..handler.basics import temp_seed
from ..handler.logger import initialize_logging
from .. import global_params
from ..mp.mp_utils import start_multiprocess_imap
from . import log_mp

BATCH_PROC_SYSTEM = global_params.BATCH_PROC_SYSTEM


def batchjob_enabled():
    if 'example_cube' in global_params.config.working_dir:  # disable QSUB/SLURM for example_run.py
        return False
    if BATCH_PROC_SYSTEM is None:
        return False
    try:
        if BATCH_PROC_SYSTEM == 'QSUB':
            cmd_check = 'qstat'
        elif BATCH_PROC_SYSTEM == 'SLURM':
            cmd_check = 'squeue'
        else:
            raise NotImplementedError
        with open(os.devnull, 'w') as devnull:
            subprocess.check_call(cmd_check, shell=True,
                                  stdout=devnull, stderr=devnull)
    except subprocess.CalledProcessError as e:
        print("BatchJobSystem '{}' specified but failed with error '{}' not found,"
              " switching to single node multiprocessing.".format(BATCH_PROC_SYSTEM, e))
        return False
    return True


home_dir = os.environ['HOME'] + "/"
path_to_scripts_default = global_params.batchjob_script_folder
qsub_work_folder = "%s/%s/" % (home_dir, BATCH_PROC_SYSTEM)
username = getpass.getuser()
python_path_global = sys.executable


def QSUB_script(params, name, queue=None, pe=None, n_cores=1, priority=0,
                additional_flags='', suffix="", job_name="default",
                script_folder=None, n_max_co_processes=None, resume_job=False,
                sge_additional_flags=None, iteration=1, max_iterations=3,
                params_orig_id=None, python_path=None, disable_mem_flag=False):
    """
    TODO: change `queue` and `pe` to be set globally in global_params. All wrappers around QSUB_script should then only have a flage like 'use_batchjob'

    QSUB handler - takes parameter list like normal multiprocessing job and
    runs them on the specified cluster

    IMPORTANT NOTE: the user has to make sure that queues exist and work; we
    suggest to generate multiple queues handling different workloads

    Parameters
    ----------
    params: List
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
    additional_flags: str
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
        same time; None: use global_params.NCORE_TOTAL // number of cores per job (n_cores)
    iteration : int
        This counter stores how often QSUB_script was called for the same job
         submission. E.g. if jobs fail during a submission, it will be repeated
         max_iterations times.
    sge_additional_flags : str
    max_iterations : int
    resume_job : bool
        If True, will only process jobs without an output pkl file.
    params_orig_id : np.array or None
        If given, has to have same length as params, and specifies the previous
        job ID for each parameter. Only required if resuming a job.
    python_path : str
        Default is sys.executable
    disable_mem_flag : bool
        If True, memory flag will not be set, otherwise it will be set to the
         fraction of the cores per job to the total number of cores per node
        
    Returns
    -------
    path_to_out: str
        path to the output directory

    """
    if n_cores is None:
        n_cores = 1
    if not batchjob_enabled():
        return batchjob_fallback(params, name, n_cores, suffix, n_max_co_processes,
                                 script_folder, python_path)
    if resume_job:
        return resume_QSUB_script(
            params, name, queue=queue, pe=pe, max_iterations=max_iterations,
            priority=priority, additional_flags=additional_flags, script_folder=None,
            job_name=job_name, suffix=suffix,
            sge_additional_flags=sge_additional_flags, iteration=iteration,
            n_max_co_processes=n_max_co_processes,  n_cores=n_cores)
    if python_path is None:
        python_path = python_path_global
    job_folder = qsub_work_folder+"/%s_folder%s/" % (name, suffix)
    if os.path.exists(job_folder):
        shutil.rmtree(job_folder, ignore_errors=True)
    log_batchjob = initialize_logging("{}".format(name + suffix),
                                      log_dir=job_folder)
    n_max_co_processes = np.min([global_params.NCORE_TOTAL // n_cores,
                                 len(params)])
    log_batchjob.info('Starting BatchJob script "{}" with {} tasks using {}'
                      ' parallel jobs, each using {} core(s).'.format(
        name, len(params), n_max_co_processes, n_cores))
    if sge_additional_flags is not None:
        log_batchjob.info('"sge_additional_flags" kwarg will soon be replaced'
                          ' with "additional_flags". Please adapt method'
                          ' calls accordingly.')
        if additional_flags is not '':
            message = 'Multiple flags set. Please use only' \
                      ' "additional_flags" kwarg.'
            log_batchjob.error(message)
            raise ValueError(message)
        else:
            additional_flags = sge_additional_flags
    if job_name == "default":
        with temp_seed(hash(time.time()) % (2 ** 32 - 1)):
            letters = string.ascii_lowercase
            job_name = "".join([letters[l] for l in
                                np.random.randint(0, len(letters), 10 if BATCH_PROC_SYSTEM == 'QSUB' else 8)])
            log_batchjob.info("Random job_name created: %s" % job_name)
    else:
        log_batchjob.warning("WARNING: running multiple jobs via qsub is only supported "
                             "with non-default job_names")

    if len(job_name) > 10:
        log_batchjob.warning("WARNING: Your job_name is longer than 10. job_names have "
                             "to be distinguishable with only using their first 10 characters.")

    if script_folder is not None:
        path_to_scripts = script_folder
    else:
        path_to_scripts = path_to_scripts_default

    path_to_script = path_to_scripts + "/QSUB_%s.py" % name
    path_to_storage = "%s/storage/" % job_folder
    path_to_sh = "%s/sh/" % job_folder
    path_to_log = "%s/log/" % job_folder
    path_to_err = "%s/err/" % job_folder
    path_to_out = "%s/out/" % job_folder

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

    if BATCH_PROC_SYSTEM == 'SLURM':
        if '-V ' in additional_flags:
            log_batchjob.warning(
                '"additional_flags" contained "-V" which is a QSUB/SGE specific flag,'
                ' but SLURM was set as batch system. Converting "-V" to "--export=ALL".')
            additional_flags.replace('-V ', '--export=ALL ')
        if not '--mem=' in additional_flags and not disable_mem_flag:
            # Node memory limit is 250,000M and not 250G! -> max memory per core is 250000M/20, leave safety margin
            mem_lim = int(
                global_params.MEM_PER_NODE * n_cores / global_params.NCORES_PER_NODE)
            additional_flags += ' --mem={}M'.format(mem_lim)
            log_batchjob.info(
                'Memory requirements were not set explicitly. Setting to 250,000 MB'
                ' * n_cores / {} = {} MB'.format(global_params.NCORES_PER_NODE,
                                                 mem_lim))

    log_batchjob.info("Number of jobs for {}-script: {}".format(name, len(params)))
    pbar = tqdm.tqdm(total=len(params))

    # memory of finished jobs to calculate increments
    n_jobs_finished = 0
    last_diff_rp = 0
    sleep_time = 10
    for i_job in range(len(params)):
        if params_orig_id is not None:
            job_id = params_orig_id[i_job]
        else:
            job_id = i_job
        if n_max_co_processes is not None:
            while last_diff_rp == 0:
                nb_rp = number_of_running_processes(job_name)
                last_diff_rp = n_max_co_processes - nb_rp

                if last_diff_rp == 0:
                    n_jobs_done = len(glob.glob(path_to_out + "*.pkl"))
                    diff = n_jobs_done - n_jobs_finished
                    pbar.update(diff)
                    n_jobs_finished = n_jobs_done
                    time.sleep(sleep_time)
            last_diff_rp -= 1
            sleep_time = 1

        this_storage_path = path_to_storage + "job_%d.pkl" % job_id
        this_sh_path = path_to_sh + "job_%d.sh" % job_id
        this_out_path = path_to_out + "job_%d.pkl" % job_id
        job_log_path = path_to_log + "job_%d.log" % job_id
        job_err_path = path_to_err + "job_%d.log" % job_id

        with open(this_sh_path, "w") as f:
            f.write("#!/bin/bash\n")
            f.write('export syconn_wd="{4}"\n{0} {1} {2} {3}'.format(
                python_path, path_to_script, this_storage_path,
                this_out_path, global_params.config.working_dir))

        with open(this_storage_path, "wb") as f:
            for param in params[i_job]:
                pkl.dump(param, f)

        os.chmod(this_sh_path, 0o744)
        if BATCH_PROC_SYSTEM == 'QSUB':
            if pe is not None:
                sge_queue_option = "-pe %s %d" % (pe, n_cores)
            elif queue is not None:
                sge_queue_option = "-q %s" % queue
            else:
                raise Exception("No queue or parallel environment defined")
            cmd_exec = "qsub {0} -o {1} -e {2} -N {3} -p {4} {5} {6}".format(
                sge_queue_option, job_log_path, job_err_path, job_name,
                priority, additional_flags, this_sh_path)
            subprocess.call(cmd_exec, shell=True)
        elif BATCH_PROC_SYSTEM == 'SLURM':
            if pe is not None:
                queue_option = "--ntasks-per-node %d" % n_cores
            elif queue is not None:
                queue_option = "--partition=%s" % queue
            else:
                raise Exception("No queue or parallel environment defined")
            if priority is not None and priority != 0:
                log_batchjob.warning('Priorities are not supported with SLURM.')
            # added '--quiet' flag to prevent submission messages, errors will still be printed
            # (https://slurm.schedmd.com/sbatch.html), DOES NOT WORK
            cmd_exec = "sbatch {0} --quiet --output={1} --error={2}" \
                       " --job-name={3} {4} {5}".format(
                queue_option,
                job_log_path,
                job_err_path,
                job_name,
                additional_flags,
                this_sh_path)
            subprocess.call(cmd_exec, shell=True)
        else:
            raise NotImplementedError

    log_batchjob.info("All jobs are submitted: %s" % name)

    while True:
        nb_rp = number_of_running_processes(job_name)
        # check actually running files
        if nb_rp == 0:
            break
        n_jobs_done = len(glob.glob(path_to_out + "*.pkl"))
        diff = n_jobs_done - n_jobs_finished
        pbar.update(diff)
        n_jobs_finished = n_jobs_done
        time.sleep(sleep_time)
    pbar.close()
    log_batchjob.info("All batch jobs have finished: %s" % name)

    # Submit singleton job to send status email after jobs have been completed
    this_sh_path = path_to_sh + "singleton.sh"
    with open(this_sh_path, "w") as f:
        f.write("#!/bin/bash\n")
        f.write("")
    job_log_path = path_to_log + "singleton.log"
    job_err_path = path_to_err + "singleton.log"
    cmd_exec = "sbatch {0} --output={1} --error={2} \
     --quiet --job-name={3} --mail-type=END {4} ".format(
        '--ntasks-per-node 1',
        job_log_path,
        job_err_path,
        job_name,  # has to be the same as the above job name
        this_sh_path)
    subprocess.call(cmd_exec, shell=True)

    out_files = glob.glob(path_to_out + "*.pkl")
    # only stop if first iteration and script was not resumed (params_orig_id is None)
    if len(out_files) == 0 and iteration == 1 and params_orig_id is None:
        msg = 'All submitted jobs have failed. Re-submission will not be initiated.' \
              ' Please check your submitted code.'
        log_batchjob.error(msg)
        raise Exception(msg)
    if len(out_files) < len(params):
        log_batchjob.error("%d jobs appear to have failed." % (len(params) - len(out_files)))
        checklist = np.zeros(len(params), dtype=np.bool)

        for p in out_files:
            checklist[int(re.findall("[\d]+", p)[-1])] = True

        msg = "Missing: {}".format(np.where(~checklist)[0])
        log_batchjob.error(msg)
        if iteration >= max_iterations:
            raise RuntimeError(msg)

        missed_params = [params[ii] for ii in range(len(params)) if not checklist[ii]]
        orig_job_ids = np.arange(len(params))[~checklist]
        assert len(missed_params) == len(orig_job_ids)
        # set number cores per job higher which will at the same time increase
        # the available amount of memory per job, ONLY VALID IF '--mem' was not specified explicitly!
        n_cores += 1  # increase number of cores per job by at least 1
        n_cores = np.max([np.min([global_params.NCORES_PER_NODE, float(n_max_co_processes) /
                                  len(missed_params) * n_cores]), n_cores])
        n_cores = np.min([n_cores, global_params.NCORES_PER_NODE])
        if n_cores == global_params.NCORES_PER_NODE:
            if not '--mem=' in additional_flags:
                additional_flags += ' --mem=0'
            else:
                m = re.search('(?<=--mem=)\w+', additional_flags)
                additional_flags.replace(m.group(0), '--mem=0')
        # if all jobs failed, increase number of cores
        return QSUB_script(
            missed_params, name, queue=queue, pe=pe, max_iterations=max_iterations,
            priority=priority, additional_flags=additional_flags, script_folder=None,
            job_name="default", suffix=suffix+"_iter"+str(iteration),
            sge_additional_flags=sge_additional_flags, iteration=iteration+1,
            n_max_co_processes=n_max_co_processes,  n_cores=n_cores,
            params_orig_id=orig_job_ids)

    return path_to_out


def resume_QSUB_script(params, name, queue=None, pe=None, n_cores=1, priority=0,
                        additional_flags='', suffix="", job_name="default",
                        script_folder=None, n_max_co_processes=None,
                        sge_additional_flags=None, iteration=0, max_iterations=3):
    """
    QSUB handler - takes parameter list like normal multiprocessing job and
    runs them on the specified cluster

    IMPORTANT NOTE: the user has to make sure that queues exist and work; we
    suggest to generate multiple queues handling different workloads

    Parameters
    ----------
    params: List
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
    additional_flags: str
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
    iteration : int
        This counter stores how often QSUB_script was called for the same job
         submission. E.g. if jobs fail during a submission, it will be repeated
         max_iterations times.
    sge_additional_flags : str
    max_iterations : int

    Returns
    -------
    path_to_out: str
        path to the output directory

    """
    job_folder = qsub_work_folder + "/%s_folder%s/" % (name, suffix)
    if not os.path.exists(job_folder):
        raise RuntimeError('Job folder has to exist, in order to '
                           'resume unfinished job.')
    log_batchjob = initialize_logging("{}_resumed".format(name + suffix),
                                      log_dir=job_folder)
    log_batchjob.info('RESUMING BatchJob script {} with {} tasks.'
                      .format(name, len(params)))
    path_to_out = "%s/out/" % job_folder

    out_files = glob.glob(path_to_out + "*.pkl")
    if len(out_files) < len(params):
        log_batchjob.error("%d jobs appear to have failed. Restarting."
                           "" % (len(params) - len(out_files)))
        checklist = np.zeros(len(params), dtype=np.bool)

        for p in out_files:
            checklist[int(re.findall("[\d]+", p)[-1])] = True

        missed_params = [params[ii] for ii in range(len(params)) if not checklist[ii]]
        orig_job_ids = np.arange(len(params))[~checklist]
        return QSUB_script(
            missed_params, name, queue=queue, pe=pe, max_iterations=max_iterations,
            priority=priority, script_folder=None, job_name=job_name,
            suffix=suffix + "_resumed", additional_flags=additional_flags,
            sge_additional_flags=sge_additional_flags, iteration=iteration,
            n_max_co_processes=n_max_co_processes, n_cores=n_cores,
            params_orig_id=orig_job_ids)
    else:
        log_batchjob.info('All jobs had already been finished successfully.')

    return path_to_out


def batchjob_fallback(params, name, n_cores=1, suffix="", n_max_co_processes=None,
                      script_folder=None, python_path=None):
    """
    Fallback method in case no batchjob submission system is available.

    Parameters
    ----------
    params :
    name :
    n_cores :
    suffix :
    job_name :
    script_folder :
    python_path :

    Returns
    -------

    """
    if python_path is None:
        python_path = python_path_global
    job_folder = qsub_work_folder + "/%s_folder%s/" % (name, suffix)
    if os.path.exists(job_folder):
        shutil.rmtree(job_folder, ignore_errors=True)
    log_batchjob = initialize_logging("{}_fallback".format(name + suffix),
                                      log_dir=job_folder)
    if n_max_co_processes is None:
        n_max_co_processes = global_params.NCORES_PER_NODE
    n_max_co_processes = np.min([global_params.NCORES_PER_NODE // n_cores, n_max_co_processes])
    n_max_co_processes = np.min([n_max_co_processes, len(params)])
    log_batchjob.debug('Starting BatchJobFallback script "{}" with {} tasks using {}'
                       ' parallel jobs, each using {} core(s).'.format(
        name, len(params), n_max_co_processes, n_cores))

    if script_folder is not None:
        path_to_scripts = script_folder
    else:
        path_to_scripts = path_to_scripts_default

    path_to_script = path_to_scripts + "/QSUB_%s.py" % name
    path_to_storage = "%s/storage/" % job_folder
    path_to_sh = "%s/sh/" % job_folder
    path_to_log = "%s/log/" % job_folder
    path_to_err = "%s/err/" % job_folder
    path_to_out = "%s/out/" % job_folder

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

    multi_params = []
    for i_job in range(len(params)):
        job_id = i_job
        this_storage_path = path_to_storage + "job_%d.pkl" % job_id
        this_sh_path = path_to_sh + "job_%d.sh" % job_id
        this_out_path = path_to_out + "job_%d.pkl" % job_id
        with open(this_sh_path, "w") as f:
            f.write('#!/bin/bash\n')
            f.write('export syconn_wd="{4}"\n{0} {1} {2} {3}'.format(
                python_path, path_to_script, this_storage_path,
                this_out_path, global_params.config.working_dir))
        with open(this_storage_path, "wb") as f:
            for param in params[i_job]:
                pkl.dump(param, f)
        os.chmod(this_sh_path, 0o744)

        cmd_exec = "sh {}".format(this_sh_path)
        multi_params.append(cmd_exec)
    start_multiprocess_imap(fallback_exec, multi_params, debug=False,
                            nb_cpus=n_max_co_processes)
    return path_to_out


def fallback_exec(cmd_exec):
    """
    Helper function to execute commands using subprocess.
    """
    ps = subprocess.Popen(cmd_exec, shell=True, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE)
    out, err = ps.communicate()
    if 'error' in err.decode().lower():
        log_mp.error(out.decode())
        log_mp.error(err.decode())


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
    if BATCH_PROC_SYSTEM == 'QSUB':
        cmd_stat = "qstat -u %s" % username
    elif BATCH_PROC_SYSTEM == 'SLURM':
        cmd_stat = "squeue -u %s" % username
    else:
        raise NotImplementedError
    process = subprocess.Popen(cmd_stat, shell=True,
                               stdout=subprocess.PIPE)
    nb_lines = 0
    for line in io.TextIOWrapper(process.stdout, encoding="utf-8"):
        if job_name[:10 if BATCH_PROC_SYSTEM == 'QSUB' else 8] in line:
            nb_lines += 1
    return nb_lines


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
    if BATCH_PROC_SYSTEM == 'QSUB':
        cmd_stat = "qstat -u %s" % username
    elif BATCH_PROC_SYSTEM == 'SLURM':
        cmd_stat = "squeue -u %s" % username
    else:
        raise NotImplementedError
    process = subprocess.Popen(cmd_stat, shell=True,
                               stdout=subprocess.PIPE)
    job_ids = []
    for line in iter(process.stdout.readline, ''):
        curr_line = str(line)
        if job_name[:10] in curr_line:
            job_ids.append(re.findall("[\d]+", curr_line)[0])

    if BATCH_PROC_SYSTEM == 'QSUB':
        cmd_del = "qdel "
        for job_id in job_ids:
            cmd_del += job_id + ", "
        command = cmd_del[:-2]

        subprocess.Popen(command, shell=True,
                         stdout=subprocess.PIPE)
    elif BATCH_PROC_SYSTEM == 'SLURM':
        cmd_del = "scancel -n {}".format(job_name)
        subprocess.Popen(cmd_del, shell=True,
                         stdout=subprocess.PIPE)
    else:
        raise NotImplementedError


def negative_to_zero(a):
    if a > 0:
        return a
    else:
        return 0
