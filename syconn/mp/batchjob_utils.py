# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max-Planck-Institute of Neurobiology, Munich, Germany
# Authors: Philipp Schubert, Sven Dorkenwald, Jörgen Kornfeld
import dill  # supports pickling of lambda expressions
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
import datetime
from typing import Dict, Optional
import shutil
import string
import subprocess
import tqdm
import sys
import time
from multiprocessing import cpu_count
from logging import Logger

from ..handler.basics import temp_seed
from ..handler.config import initialize_logging
from .. import global_params
from .mp_utils import start_multiprocess_imap
from . import log_mp


def batchjob_enabled():
    """
    Checks if active batch processing system is actually working.

    Returns:
        True if either SLURM or QSUB is active.
    """
    batch_proc_system = global_params.config['batch_proc_system']
    if batch_proc_system is None or batch_proc_system == 'None':
        return False
    try:
        if batch_proc_system == 'QSUB':
            cmd_check = 'qstat'
        elif batch_proc_system == 'SLURM':
            cmd_check = 'squeue'
        else:
            raise NotImplementedError
        with open(os.devnull, 'w') as devnull:
            subprocess.check_call(cmd_check, shell=True,
                                  stdout=devnull, stderr=devnull)
    except subprocess.CalledProcessError as e:
        print("BatchJobSystem '{}' specified but failed with error '{}' not found,"
              " switching to single node multiprocessing.".format(batch_proc_system, e))
        return False
    return True


path_to_scripts_default = global_params.config.batchjob_script_folder
username = getpass.getuser()
python_path_global = sys.executable


def batchjob_script(params: list, name: str,
                    batchjob_folder: Optional[str] = None,
                    n_cores: int = 1, additional_flags: str = '',
                    suffix: str = "", job_name: str = "default",
                    script_folder: Optional[str] = None,
                    n_max_co_processes: Optional[int] = None,
                    max_iterations: int = 5,
                    python_path: Optional[str] = None,
                    disable_batchjob: bool = False,
                    use_dill: bool = False,
                    remove_jobfolder: bool = False,
                    log: Logger = None, sleep_time: int = 20,
                    show_progress=True):
    """
    Submits batch jobs to process a list of parameters `params` with a python
    script on the specified environment (either None, SLURM or QSUB; run
    ``global_params.config['batch_proc_system']`` to get the active system).

    Notes:
        * The memory available for each job is coupled to the number of cores
          per job (`n_cores`).

    Todo:
        * Add sbatch array support -> faster submission
        * Make script specification more generic

    Args:
        params: List of all parameter sets to be processed.
        name: Name of batch job submitted via the batch processing system.
        batchjob_folder: Directory which contains all submission relevant files,
            e.g. bash scripts, logs, output files, .. Defaults to
            ``"{}/{}_folder{}/".format(global_params.config.qsub_work_folder, name, suffix)``.
        n_cores: Number of cores used for each job.
        additional_flags: Used to set additional parameters for each job. To
            allocate one GPU for each worker use: ``additional_flags=--gres=gpu:1``.
        suffix: Suffix added to `batchjob_folder`.
        job_name: Name of the jobs submitted via the batch processing system.
            Defaults to a random string of 8 letters.
        script_folder: Directory where to look for the script which is executed.
            Looks for ``QSUB_{name}.py``.
        n_max_co_processes: Not needed / not monitored anymore.
        max_iterations: Maximum number of retries of failed jobs.
        python_path: Path to python binary.
        disable_batchjob: Use single node multiprocessing.
        use_dill: Use dill to enable pickling of lambda expressions.
        remove_jobfolder: Remove `batchjob_folder` after successful termination.
        log: Logger.
        sleep_time: Sleep duration before checking batch job states again.
        show_progress: Only used if ``disabled_batchjob=True``.
    """
    starttime = datetime.datetime.today().strftime("%m.%d")
    # Parameter handling
    if n_cores is None:
        n_cores = 1
    if python_path is None:
        python_path = python_path_global
    if batchjob_folder is None:
        batchjob_folder = "{}/{}_folder{}/".format(
            global_params.config.qsub_work_folder, name, suffix)
    if os.path.exists(batchjob_folder):
        shutil.rmtree(batchjob_folder, ignore_errors=True)
    if log is None:
        log_batchjob = initialize_logging("{}".format(name + suffix),
                                          log_dir=batchjob_folder)
    else:
        log_batchjob = log
    if n_max_co_processes is None:
        n_max_co_processes = np.min([global_params.config.ncore_total // n_cores,
                                     len(params)])
    n_max_co_processes = np.max([n_max_co_processes, 1])
    if script_folder is not None:
        path_to_scripts = script_folder
    else:
        path_to_scripts = path_to_scripts_default

    # Check if any fallback is required
    if disable_batchjob or not batchjob_enabled():
        return batchjob_fallback(params, name, n_cores, suffix,
                                 script_folder, python_path, show_progress=show_progress,
                                 remove_jobfolder=remove_jobfolder, log=log)
    if global_params.config['batch_proc_system'] != 'SLURM':
        log_mp.warn('"batchjob_script" currently does not support any other '
                    'batch processing system than SLURM. Falling back to '
                    'deprecated "QSUB_script".')
        return QSUB_script(params, name, n_cores=n_cores,
                           additional_flags=additional_flags, suffix=suffix,
                           job_name=job_name, script_folder=script_folder,
                           n_max_co_processes=n_max_co_processes,
                           max_iterations=max_iterations,
                           python_path=python_path,
                           disable_batchjob=disable_batchjob, use_dill=use_dill,
                           remove_jobfolder=remove_jobfolder, log=log)

    mem_lim = int(global_params.config['mem_per_node'] /
                  global_params.config['ncores_per_node'])
    if '--mem' in additional_flags:
        raise ValueError('"--mem" must not be set via the "additional_flags"'
                         ' kwarg.')
    additional_flags += ' --mem-per-cpu={}M'.format(mem_lim)

    # Start SLURM job
    log_batchjob.info(
        'Started BatchJob script "{}" with {} tasks using {} parallel jobs, each'
        ' using {} core(s).'.format(name, len(params), n_max_co_processes, n_cores))
    if job_name == "default":
        with temp_seed(hash(time.time()) % (2 ** 32 - 1)):
            letters = string.ascii_lowercase
            job_name = "".join([letters[l] for l in
                                np.random.randint(0, len(letters), 8)])
            log_batchjob.info("Random job_name created: %s" % job_name)
    if len(job_name) > 8:
        msg = "job_name is longer than 8 characters. This is untested."
        log_batchjob.error(msg)
        raise ValueError(msg)

    # Create folder structure
    path_to_script = path_to_scripts + "/QSUB_%s.py" % name
    path_to_storage = "%s/storage/" % batchjob_folder
    path_to_sh = "%s/sh/" % batchjob_folder
    path_to_log = "%s/log/" % batchjob_folder
    path_to_err = "%s/err/" % batchjob_folder
    path_to_out = "%s/out/" % batchjob_folder
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

    # Submit jobs
    log_batchjob.info("Number of jobs for {}-script: {}".format(name, len(params)))
    pbar = tqdm.tqdm(total=len(params), miniters=1, mininterval=1, leave=False)
    dtime_sub = 0
    start_all = time.time()
    job_exec_dc = {}
    job2slurm_dc = {}  # stores mapping of internal to SLURM job ID
    slurm2job_dc = {}  # stores mapping of SLURM to internal job ID
    for job_id in range(len(params)):
        this_storage_path = path_to_storage + "job_%d.pkl" % job_id
        this_sh_path = path_to_sh + "job_%d.sh" % job_id
        this_out_path = path_to_out + "job_%d.pkl" % job_id
        job_log_path = path_to_log + "job_%d.log" % job_id
        job_err_path = path_to_err + "job_%d.log" % job_id

        with open(this_sh_path, "w") as f:
            f.write("#!/bin/bash -l\n")
            f.write('export syconn_wd="{4}"\n{0} {1} {2} {3}'.format(
                python_path, path_to_script, this_storage_path,
                this_out_path, global_params.config.working_dir))

        with open(this_storage_path, "wb") as f:
            for param in params[job_id]:
                if use_dill:
                    dill.dump(param, f)
                else:
                    pkl.dump(param, f)

        os.chmod(this_sh_path, 0o744)
        cmd_exec = "{0} --output={1} --error={2} --job-name={3} {4}".format(
            additional_flags, job_log_path, job_err_path, job_name, this_sh_path)

        job_exec_dc[job_id] = cmd_exec
        start = time.time()
        process = subprocess.Popen(f'sbatch --cpus-per-task={n_cores} {cmd_exec}',
                                   shell=True, stdout=subprocess.PIPE)
        out_str = io.TextIOWrapper(process.stdout, encoding="utf-8").read()
        slurm_id = int(re.findall(r'(\d+)', out_str)[0])
        job2slurm_dc[job_id] = slurm_id
        slurm2job_dc[slurm_id] = job_id
        dtime_sub += time.time() - start
        time.sleep(0.1)

    # wait for jobs to be in SLURM memory
    time.sleep(sleep_time)
    # requeue failed jobs for `max_iterations`-times
    js_dc = jobstates_slurm(job_name, starttime)
    requeue_dc = {k: 0 for k in job2slurm_dc}  # use internal job IDs!
    nb_completed_compare = 0
    while True:
        nb_failed = 0
        # get internal job ids from current job dict
        job_ids = np.array(list(slurm2job_dc.values()))
        # get states of slurm jobs with the same ordering as 'job_ids'
        try:
            job_states = np.array([js_dc[k] for k in slurm2job_dc.keys()])
        except KeyError as e:  # sometimes new SLURM job is not yet in the SLURM cache.
            log_batchjob.warning(f'Did not find state of worker {e}\nFetching worker states '
                                 f'again, SLURM cache might have been delayed.')
            time.sleep(1.5*sleep_time)
            js_dc = jobstates_slurm(job_name, starttime)
            job_states = np.array([js_dc[k] for k in slurm2job_dc.keys()])
        # all jobs which are not running, completed or pending have failed for
        # some reason (states: failed, out_out_memory, ..).
        for j in job_ids[(job_states != 'COMPLETED') & (job_states != 'PENDING')
                         & (job_states != 'RUNNING')]:
            if requeue_dc[j] == max_iterations:
                nb_failed += 1
                continue
            # restart job
            requeue_dc[j] += 1
            # increment number of cores by one.
            job_cmd = f'sbatch --cpus-per-task={requeue_dc[j] + n_cores} {job_exec_dc[j]}'
            process = subprocess.Popen(job_cmd, shell=True,
                                       stdout=subprocess.PIPE)
            out_str = io.TextIOWrapper(process.stdout, encoding="utf-8").read()
            slurm_id = int(re.findall(r'(\d+)', out_str)[0])
            slurm_id_orig = job2slurm_dc[j]
            del slurm2job_dc[slurm_id_orig]
            job2slurm_dc[j] = slurm_id
            slurm2job_dc[slurm_id] = j
            log_batchjob.info(f'Requeued job {j}. SLURM IDs: {slurm_id} (new), '
                              f'{slurm_id_orig} (old).')
        nb_completed = np.sum(job_states == 'COMPLETED')
        pbar.update(nb_completed - nb_completed_compare)
        nb_completed_compare = nb_completed
        nb_finished = nb_completed + nb_failed
        # check actually running files
        if nb_finished == len(params):
            break
        time.sleep(sleep_time)
        js_dc = jobstates_slurm(job_name, starttime)
    pbar.close()

    dtime_all = time.time() - start_all
    dtime_all = time.strftime("%Hh:%Mmin:%Ss", time.gmtime(dtime_all))
    log_batchjob.info(f"All jobs ({name}, {job_name}) have finished after "
                      f"{dtime_all} ({dtime_sub:.1f} s submission): "
                      f"{nb_completed} completed, {nb_failed} failed.")
    out_files = glob.glob(path_to_out + "*.pkl")
    if len(out_files) < len(params):
        msg = f'Batch processing error during execution of {name} in job ' \
              f'\"{job_name}\": Found {len(out_files)}, expected {len(params)}.'
        log_batchjob.error(msg)
        raise ValueError(msg)
    if remove_jobfolder:
        shutil.rmtree(batchjob_folder)
    return path_to_out


def jobstates_slurm(job_name: str, start_time: str) -> Dict[int, str]:
    """
    Generates a dictionary which stores the state of every job belonging to
    `job_name`.

    Args:
        job_name:
        start_time: The following formats are allowed: MMDD[YY] or MM/DD[/YY]
            or MM.DD[.YY], e.g. ``datetime.datetime.today().strftime("%m.%d")``.

    Returns:
        Dictionary with the job states. (key: job ID, value: state)
    """
    cmd_stat = f"sacct -b --name {job_name} -u {username} -S {start_time}"
    process = subprocess.Popen(cmd_stat, shell=True,
                               stdout=subprocess.PIPE)
    job_states = dict()
    for line in io.TextIOWrapper(process.stdout, encoding="utf-8"):
        str_parsed = re.findall(r"(\d+)[\s,\t]+([A-Z]+)", line)
        if len(str_parsed) == 1:
            str_parsed = str_parsed[0]
            job_states[int(str_parsed[0])] = str_parsed[1]
    return job_states


def QSUB_script(params, name, queue=None, pe=None, n_cores=1, priority=0,
                additional_flags='', suffix="", job_name="default",
                script_folder=None, n_max_co_processes=None, resume_job=False,
                sge_additional_flags=None, iteration=1, max_iterations=5,
                params_orig_id=None, python_path=None, disable_mem_flag=False,
                disable_batchjob=False, send_notification=False, use_dill=False,
                remove_jobfolder=False, log=None, show_progress=True,
                allow_resubm_all_fail=False):
    """

    Todo:
        * Remove folders of failed job submission attempts in case the last one
          is successful.
        * Use JobArrays

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
        additional command line flags to be passed to QSUB/SLURM. Changes to
        "--cpus-per-task=X" will be overwritten by "--cpus-per-task={n_cores}"
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
        same time; None: use global_params.config.ncore_total // number of cores per job (n_cores)
    iteration : int
        This counter stores how often QSUB_script was called for the same job
         submission. E.g. if jobs fail during a submission, it will be repeated
         max_iterations times. Starts at 1!
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
    disable_batchjob : bool
        Overwrites global batchjob settings and will run multiple, independent bash jobs
        on multiple CPUs instead.
    send_notification : bool
        Sends an notification email after completion. Currently does not contain any
        information about the job name, required time or CPU/MEM usage.
        TODO: use SLURM JobArrays to enable detailed notification emails
    remove_jobfolder: bool
        Remove the created folder after successfully processing.
    use_dill : bool
    show_progress : bool
        Currently only applies for `batchjob_fallback`
    allow_resubm_all_fail : bool
        Will resubmit failed jobs even if all failed. Useful for e.g. unexpected memory
        requirements.
    log: Logger
        Logger.

    Returns
    -------
    path_to_out: str
        path to the output directory

    """
    if n_cores is None:
        n_cores = 1
    if disable_batchjob or not batchjob_enabled():
        return batchjob_fallback(params, name, n_cores, suffix,
                                 script_folder, python_path, 
                                 remove_jobfolder=remove_jobfolder,
                                 show_progress=show_progress, log=log)
    if queue is None:
        queue = global_params.config['batch_queue']
    if pe is None:
        pe = global_params.config['batch_pe']
    if resume_job:
        raise NotImplementedError('Resume jobs requires refactoring.')
        return resume_QSUB_script(
            params, name, queue=queue, pe=pe, max_iterations=max_iterations,
            priority=priority, additional_flags=additional_flags, script_folder=None,
            job_name=job_name, suffix=suffix,
            sge_additional_flags=sge_additional_flags, iteration=iteration,
            n_max_co_processes=n_max_co_processes,  n_cores=n_cores, use_dill=use_dill)
    if python_path is None:
        python_path = python_path_global
    job_folder = "{}/{}_folder{}/".format(global_params.config.qsub_work_folder,
                                          name, suffix)

    # TODO: replace QSUB_script by batchjob_script package-wide
    return batchjob_script(params, name, job_folder, n_cores, additional_flags,
                    suffix, job_name, script_folder,
                    n_max_co_processes, max_iterations,
                    python_path, disable_batchjob, use_dill,
                    remove_jobfolder, log, 20, show_progress)

    if iteration == 1 and os.path.exists(job_folder):
        shutil.rmtree(job_folder, ignore_errors=True)
    if log is None:
        log_batchjob = initialize_logging("{}".format(name + suffix),
                                          log_dir=job_folder)
    else:
        log_batchjob = log
    if n_max_co_processes is None:
        n_max_co_processes = np.min([global_params.config.ncore_total // n_cores,
                                     len(params)])
    n_max_co_processes = np.max([n_max_co_processes, 1])
    log_batchjob.info('Started BatchJob script "{}" with {} tasks using {}'
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
                                np.random.randint(0, len(letters), 10 if
                                global_params.config['batch_proc_system'] == 'QSUB' else 8)])
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

    if global_params.config['batch_proc_system'] == 'SLURM':
        if '-V ' in additional_flags:
            log_batchjob.warning(
                '"additional_flags" contained "-V" which is a QSUB/SGE specific flag,'
                ' but SLURM was set as batch system. Converting "-V" to "--export=ALL".')
            additional_flags = additional_flags.replace('-V ', '--export=ALL ')
        if not '--mem=' in additional_flags and not disable_mem_flag:
            # Node memory limit is 250,000M and not 250G! ->
            # max memory per core is 250000M/20, leave safety margin
            mem_lim = int(
                global_params.config['mem_per_node'] * n_cores /
                global_params.config['ncores_per_node'])
            additional_flags += ' --mem={}M'.format(mem_lim)
            log_batchjob.info(
                'Memory requirements were not set explicitly. Setting to 250,000 MB'
                ' * n_cores / {} = {} MB'.format(global_params.config['ncores_per_node'],
                                                 mem_lim))
        if n_cores > 1:
            # remove existing cpus per task
            if '--cpus-per-task' in additional_flags:
                m = re.search(r'(?<=--cpus-per-task=)\w+', additional_flags)
                additional_flags = additional_flags.replace('--cpus-per-task=' + m.group(0), '')
            additional_flags += f" --cpus-per-task={n_cores}"

    log_batchjob.info("Number of jobs for {}-script: {}".format(name, len(params)))
    pbar = tqdm.tqdm(total=len(params), miniters=1, mininterval=1)

    dtime_sub = 0
    start_all = time.time()
    # memory of finished jobs to calculate increments
    n_jobs_finished = 0
    last_diff_rp = 0
    sleep_time = 5
    for i_job in range(len(params)):
        if params_orig_id is not None:
            job_id = params_orig_id[i_job]
        else:
            job_id = i_job
        this_storage_path = path_to_storage + "job_%d.pkl" % job_id
        this_sh_path = path_to_sh + "job_%d.sh" % job_id
        this_out_path = path_to_out + "job_%d.pkl" % job_id
        job_log_path = path_to_log + "job_%d.log" % job_id
        job_err_path = path_to_err + "job_%d.log" % job_id

        with open(this_sh_path, "w") as f:
            f.write("#!/bin/bash -l\n")
            f.write('export syconn_wd="{4}"\n{0} {1} {2} {3}'.format(
                python_path, path_to_script, this_storage_path,
                this_out_path, global_params.config.working_dir))

        with open(this_storage_path, "wb") as f:
            for param in params[i_job]:
                if use_dill:
                    dill.dump(param, f)
                else:
                    pkl.dump(param, f)

        os.chmod(this_sh_path, 0o744)
        if global_params.config['batch_proc_system'] == 'QSUB':
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
        elif global_params.config['batch_proc_system'] == 'SLURM':
            cmd_exec = "sbatch {0} --output={1} --error={2}" \
                       " --job-name={3} {4}".format(
                additional_flags, job_log_path, job_err_path,
                job_name, this_sh_path)
            if priority is not None and priority != 0:
                log_batchjob.warning('Priorities are not supported with SLURM.')
            start = time.time()
            subprocess.call(cmd_exec, shell=True)
            dtime_sub += time.time() - start
        else:
            raise NotImplementedError
        time.sleep(0.05)

    log_batchjob.info("All jobs are submitted: %s" % name)
    while True:
        nb_rp = number_of_running_processes(job_name)
        # check actually running files
        if nb_rp == 0:
            break
        fname_ids = set([int(re.findall(r"[\d]+", p)[-1]) for p in
                     glob.glob(path_to_out + "*.pkl")])
        n_jobs_done = len(fname_ids)
        if iteration > 1:
            n_jobs_done = len(set(params_orig_id).intersection(fname_ids))
        diff = n_jobs_done - n_jobs_finished
        pbar.update(diff)
        n_jobs_finished = n_jobs_done
        time.sleep(sleep_time)
    pbar.close()
    log_batchjob.info("All batch jobs have finished after {:.2f} s ({:.2f} "
                      "s submission): {}".format(time.time()-start_all, dtime_sub, name))

    # Submit singleton job to send status email after jobs have been completed
    if send_notification:
        this_sh_path = path_to_sh + "singleton.sh"
        with open(this_sh_path, "w") as f:
            f.write("#!/bin/bash -l\n")
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
    if len(out_files) == 0 and iteration == 1 and params_orig_id is None and not \
            allow_resubm_all_fail:
        # check if out-of-memory event occurred.
        err_files = glob.glob(path_to_err + "*.log")
        found_oom_event = False
        for fname in err_files:
            strng = "".join(open(fname).readlines()).lower()
            if 'killed' in strng or 'oom' in strng or 'out-of-memory' in strng:
                found_oom_event = True
        if not found_oom_event:
            msg = 'All submitted jobs have failed. Re-submission will not be ' \
                  'initiated. Please check your submitted code.'
            log_batchjob.error(msg)
            raise RuntimeError(msg)
    if len(out_files) < len(params):
        log_batchjob.error("%d jobs appear to have failed." % (len(params) - len(out_files)))
        checklist = np.zeros(len(params), dtype=np.bool)
        if iteration == 1:
            params_orig_id = np.arange(len(params))
        for p in out_files:
            job_id = int(re.findall(r"[\d]+", p)[-1])
            if job_id not in params_orig_id:
                continue
            index = np.nonzero(params_orig_id == job_id)[0]  # still an array, "[0]" only gives
            # us the first dimension
            checklist[index[0]] = True

        missed_params = [params[ii] for ii in range(len(params)) if not checklist[ii]]
        orig_job_ids = params_orig_id[~checklist]
        assert len(missed_params) == len(orig_job_ids)
        msg = "Missing: {}".format(orig_job_ids)
        log_batchjob.error(msg)
        if iteration >= max_iterations:
            raise RuntimeError(msg)

        # set number cores per job higher which will at the same time increase
        # the available amount of memory per job, ONLY VALID IF '--mem' was not specified explicitly!
        n_cores += 1  # increase number of cores per job by at least 2
        # TODO: activate again
        # n_cores = np.max([np.min([global_params.config['ncores_per_node'], float(n_max_co_processes) //
        #                           len(missed_params)]), n_cores])
        n_cores = np.min([n_cores, global_params.config['ncores_per_node']])
        n_cores = int(n_cores)
        # remove existing memory and cpus-per-task flags:
        if '--mem=' in additional_flags:
            m = re.search('(?<=--mem=)\w+', additional_flags)
            additional_flags = additional_flags.replace('--mem=' + m.group(0), '')
        if '--cpus-per-task' in additional_flags:
            m = re.search('(?<=--cpus-per-task=)\w+', additional_flags)
            additional_flags = additional_flags.replace('--cpus-per-task=' + m.group(0), '')
        if n_cores == global_params.config['ncores_per_node']:
            if not '--mem=' in additional_flags:
                additional_flags += ' --mem=0'
            else:
                m = re.search('(?<=--mem=)\w+', additional_flags)
                additional_flags = additional_flags.replace('--mem=' + m.group(0), '--mem=0')
        return QSUB_script(
            missed_params, name, queue=queue, pe=pe, max_iterations=max_iterations,
            priority=priority, additional_flags=additional_flags, script_folder=None,
            job_name="default", suffix=suffix, remove_jobfolder=remove_jobfolder,
            sge_additional_flags=sge_additional_flags, iteration=iteration+1,
            n_max_co_processes=n_max_co_processes,  n_cores=n_cores,
            params_orig_id=orig_job_ids, use_dill=use_dill)
    if remove_jobfolder:
        shutil.rmtree(job_folder)
    return path_to_out


def resume_QSUB_script(params, name, queue=None, pe=None, n_cores=1, priority=0,
                       additional_flags='', suffix="", job_name="default",
                       script_folder=None, n_max_co_processes=None, use_dill=False,
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
    job_folder = "{}/{}_folder{}/".format(global_params.config.qsub_work_folder,
                                          name, suffix)
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
            checklist[int(re.findall(r"[\d]+", p)[-1])] = True

        missed_params = [params[ii] for ii in range(len(params)) if not checklist[ii]]
        orig_job_ids = np.arange(len(params))[~checklist]
        return QSUB_script(
            missed_params, name, queue=queue, pe=pe, max_iterations=max_iterations,
            priority=priority, script_folder=None, job_name=job_name,
            suffix=suffix + "_resumed", additional_flags=additional_flags,
            sge_additional_flags=sge_additional_flags, iteration=iteration,
            n_max_co_processes=n_max_co_processes, n_cores=n_cores,
            params_orig_id=orig_job_ids, use_dill=use_dill)
    else:
        log_batchjob.info('All jobs had already been finished successfully.')

    return path_to_out


def batchjob_fallback(params, name, n_cores=1, suffix="",
                      script_folder=None, python_path=None,
                      remove_jobfolder=False, show_progress=True, log=None):
    """
    # TODO: utilize log and error files ('path_to_err', path_to_log')
    Fallback method in case no batchjob submission system is available. Always uses
    ``n_max_co_processes = cpu_count()``.

    Parameters
    ----------
    params : List[Any]
    name : str
    n_cores : int
        CPUs per job.
    suffix : str
    script_folder : str
    python_path : str
    remove_jobfolder : bool
    show_progress : bool
    log: Logger
        Logger.

    Returns
    -------
    str
        Path to output.
    """
    if python_path is None:
        python_path = python_path_global
    job_folder = "{}/{}_folder{}/".format(global_params.config.qsub_work_folder,
                                          name, suffix)
    if os.path.exists(job_folder):
        shutil.rmtree(job_folder, ignore_errors=True)
    if log is None:
        log_batchjob = initialize_logging("{}".format(name + suffix),
                                          log_dir=job_folder)
    else:
        log_batchjob = log
    n_max_co_processes = cpu_count()
    n_max_co_processes = np.min([cpu_count() // n_cores, n_max_co_processes])
    n_max_co_processes = np.min([n_max_co_processes, len(params)])
    n_max_co_processes = np.max([n_max_co_processes, 1])
    log_batchjob.debug('Started BatchJobFallback script "{}" with {} tasks using {}'
                       ' parallel jobs, each using {} core(s).'.format(
        name, len(params), n_max_co_processes, n_cores))
    start = time.time()

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
            f.write('#!/bin/bash -l\n')
            f.write('export syconn_wd="{4}"\n{0} {1} {2} {3}'.format(
                python_path, path_to_script, this_storage_path,
                this_out_path, global_params.config.working_dir))
        with open(this_storage_path, "wb") as f:
            for param in params[i_job]:
                pkl.dump(param, f)
        os.chmod(this_sh_path, 0o744)

        cmd_exec = "sh {}".format(this_sh_path)
        multi_params.append(cmd_exec)
    out_str = start_multiprocess_imap(fallback_exec, multi_params, debug=False,
                                      nb_cpus=n_max_co_processes,
                                      show_progress=show_progress)
    out_files = glob.glob(path_to_out + "*.pkl")
    if len(out_files) < len(params):
        # report errors
        msg = 'Critical errors occurred during "{}". {}/{} Batchjob fallback worker ' \
              'failed.\n{}'.format(name, len(params) - len(out_files),
                                   len(params), out_str)
        log_mp.error(msg)
        log_batchjob.error(msg)
        raise ValueError(msg)
    elif len("".join(out_str)) == 0:
        if remove_jobfolder:
            shutil.rmtree(job_folder, ignore_errors=True)
    else:
        msg = 'Warnings/errors occurred during ' \
              '"{}".:\n{} See logs at {} for details.'.format(name, out_str,
                                                              job_folder)
        log_mp.warning(msg)
        log_batchjob.warning(msg)
    log_batchjob.debug('Finished "{}" after {:.2f}s.'.format(
        name, time.time() - start))
    return path_to_out


def fallback_exec(cmd_exec):
    """
    Helper function to execute commands via ``subprocess.Popen``.
    """
    ps = subprocess.Popen(cmd_exec, shell=True, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE)
    out, err = ps.communicate()
    out_str = ""
    reported = False
    if 'error' in out.decode().lower() or 'error' in err.decode().lower() \
    or 'killed' in out.decode().lower() or 'killed' in err.decode().lower() \
    or 'segmentation fault' in out.decode().lower() \
    or 'segmentation fault' in err.decode().lower():
        reported = True
        out_str = out.decode() + err.decode()
    if not reported and ('warning' in out.decode().lower() or
                         'warning' in err.decode().lower()):
        out_str = out.decode() + err.decode()
    return out_str


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
    if global_params.config['batch_proc_system'] == 'QSUB':
        cmd_stat = "qstat -u %s" % username
    elif global_params.config['batch_proc_system'] == 'SLURM':
        cmd_stat = "squeue -u %s" % username
    else:
        raise NotImplementedError
    process = subprocess.Popen(cmd_stat, shell=True,
                               stdout=subprocess.PIPE)
    nb_lines = 0
    for line in io.TextIOWrapper(process.stdout, encoding="utf-8"):
        if job_name[:10 if global_params.config['batch_proc_system'] == 'QSUB'
        else 8] in line:
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
    if global_params.config['batch_proc_system'] == 'QSUB':
        cmd_stat = "qstat -u %s" % username
    elif global_params.config['batch_proc_system'] == 'SLURM':
        cmd_stat = "squeue -u %s" % username
    else:
        raise NotImplementedError
    process = subprocess.Popen(cmd_stat, shell=True,
                               stdout=subprocess.PIPE)
    job_ids = []
    for line in iter(process.stdout.readline, ''):
        curr_line = str(line)
        if job_name[:10] in curr_line:
            job_ids.append(re.findall(r"[\d]+", curr_line)[0])

    if global_params.config['batch_proc_system'] == 'QSUB':
        cmd_del = "qdel "
        for job_id in job_ids:
            cmd_del += job_id + ", "
        command = cmd_del[:-2]

        subprocess.Popen(command, shell=True,
                         stdout=subprocess.PIPE)
    elif global_params.config['batch_proc_system'] == 'SLURM':
        cmd_del = "scancel -n {}".format(job_name)
        subprocess.Popen(cmd_del, shell=True,
                         stdout=subprocess.PIPE)
    else:
        raise NotImplementedError
