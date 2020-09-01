# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max-Planck-Institute of Neurobiology, Munich, Germany
# Authors: Philipp Schubert, Sven Dorkenwald, Jörgen Kornfeld
import dill  # supports pickling of lambda expressions

from . import log_mp
from .mp_utils import start_multiprocess_imap
from .. import global_params
from ..handler.basics import temp_seed, str_delta_sec
from ..handler.config import initialize_logging

import pickle as pkl
import threading
import getpass
import glob
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
import socket
import time
import numpy as np
from multiprocessing import cpu_count
from logging import Logger


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
                    max_iterations: int = 5,
                    python_path: Optional[str] = None,
                    disable_batchjob: bool = False,
                    use_dill: bool = False,
                    remove_jobfolder: bool = False,
                    log: Logger = None, sleep_time: Optional[int] = None,
                    show_progress=True,
                    overwrite=False):
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
        max_iterations: Maximum number of retries of failed jobs.
        python_path: Path to python binary.
        disable_batchjob: Use single node multiprocessing.
        use_dill: Use dill to enable pickling of lambda expressions.
            remove_jobfolder: Remove `batchjob_folder` after successful termination.
        remove_jobfolder:
        log: Logger.
        sleep_time: Sleep duration before checking batch job states again.
        show_progress: Only used if ``disabled_batchjob=True``.
        overwrite:
    """
    starttime = datetime.datetime.today().strftime("%m.%d")
    # Parameter handling
    if n_cores is None:
        n_cores = 1
    if sleep_time is None:
        sleep_time = 5
    if python_path is None:
        python_path = python_path_global

    if job_name == "default":
        with temp_seed(hash(time.time()) % (2 ** 32 - 1)):
            letters = string.ascii_lowercase
            job_name = "".join([letters[le] for le in np.random.randint(0, len(letters), 8)])

    if batchjob_folder is None:
        batchjob_folder = f"{global_params.config.qsub_work_folder}/{name}{suffix}_{job_name}/"
    if os.path.exists(batchjob_folder):
        if not overwrite:
            raise FileExistsError(f'Batchjob folder already exists at "{batchjob_folder}". Please'
                                  f' make sure it is safe for deletion, then set overwrite=True')
        shutil.rmtree(batchjob_folder, ignore_errors=True)
    batchjob_folder = batchjob_folder.rstrip('/')
    # Check if fallback is required
    if disable_batchjob or not batchjob_enabled():
        return batchjob_fallback(params, name, n_cores, suffix, script_folder, python_path,
                                 show_progress=show_progress, remove_jobfolder=remove_jobfolder,
                                 log=log, overwrite=True, job_folder=batchjob_folder)

    if log is None:
        log_batchjob = initialize_logging("{}".format(name + suffix), log_dir=batchjob_folder)
    else:
        log_batchjob = log
    if script_folder is not None:
        path_to_scripts = script_folder
    else:
        path_to_scripts = path_to_scripts_default
    path_to_script = f'{path_to_scripts}/QSUB_{name}.py'
    if not os.path.exists(path_to_script):
        if os.path.exists(f'{path_to_scripts}/batchjob_{name}.py'):
            path_to_script = f'{path_to_scripts}/batchjob_{name}.py'
        else:
            raise FileNotFoundError(f'Specified script does not exist: {path_to_script}')

    if global_params.config['batch_proc_system'] != 'SLURM':
        msg = ('"batchjob_script" currently does not support any other batch processing '
               'system than SLURM.')
        log_mp.error(msg)
        raise NotImplementedError(msg)
    cpus_per_node = global_params.config['ncores_per_node']
    mem_lim = int(global_params.config['mem_per_node'] /
                  cpus_per_node)
    if '--mem' in additional_flags:
        raise ValueError('"--mem" must not be set via the "additional_flags"'
                         ' kwarg.')
    additional_flags += ' --mem-per-cpu={}M'.format(mem_lim)

    # Start SLURM job
    if len(job_name) > 8:
        msg = "job_name is longer than 8 characters. This is untested."
        log_batchjob.error(msg)
        raise ValueError(msg)
    log_batchjob.info(
        'Started BatchJob script "{}" ({}) (suffix="{}") with {} tasks, each'
        ' using {} core(s).'.format(name, job_name, suffix, len(params), n_cores))

    # Create folder structure
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
        if job_id == 0:
            log_batchjob.debug(f'Starting jobs with command "{cmd_exec}".')
        job_exec_dc[job_id] = cmd_exec
        job_cmd = f'sbatch --cpus-per-task={n_cores} {cmd_exec}'
        start = time.time()
        max_relaunch_cnt = 0
        while True:
            process = subprocess.Popen(job_cmd, shell=True, stdout=subprocess.PIPE)
            out_str, err = process.communicate()
            if process.returncode != 0:
                if max_relaunch_cnt == 5:
                    msg = f'Could not launch job with ID {job_id} and command "{job_cmd}".'
                    log_batchjob.error(msg)
                    raise RuntimeError(msg)
                log_batchjob.warning(f'Could not launch job with ID {job_id} with command "{job_cmd}"'
                               f'for the {max_relaunch_cnt}. time.'
                               f'Attempting again in 5s. Error raised: {err}')
                max_relaunch_cnt += 1
                time.sleep(5)
            else:
                break

        slurm_id = int(re.findall(r'(\d+)', out_str.decode())[0])
        job2slurm_dc[job_id] = slurm_id
        slurm2job_dc[slurm_id] = job_id
        dtime_sub += time.time() - start
        time.sleep(0.01)

    # wait for jobs to be in SLURM memory
    time.sleep(10)
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
                                 f'again, SLURM cache may not have been updated yet.')
            time.sleep(5)
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
            if requeue_dc[j] == 20:
                log_batchjob.warning(f'About to re-submit job {j} ({job2slurm_dc[j]}) '
                                     f'which already was assigned the maximum number '
                                     f'of available CPUs.')
            requeue_dc[j] = min(requeue_dc[j] + 1, cpus_per_node - n_cores)  # n_cores is the base number of cores
            new_core_init = requeue_dc[j] - 1  # do not increase if failed first time
            # increment number of cores by one.
            job_cmd = f'sbatch --cpus-per-task={new_core_init + n_cores} {job_exec_dc[j]}'
            max_relaunch_cnt = 0
            while True:
                process = subprocess.Popen(job_cmd, shell=True, stdout=subprocess.PIPE)
                out_str, err = process.communicate()
                if process.returncode != 0:
                    if max_relaunch_cnt == 5:
                        raise RuntimeError(f'Could not launch job with ID {j} ({job2slurm_dc[j]}) '
                                           f'and command "{job_cmd}".')
                    log_batchjob.warning(f'Could not re-launch job with ID {j} ({job2slurm_dc[j]}) '
                                         f'with command "{job_cmd}" for the {max_relaunch_cnt}. '
                                         f'time. Attempting again in 5s. Error raised: {err}')
                    max_relaunch_cnt += 1
                    time.sleep(5)
                else:
                    break
            slurm_id = int(re.findall(r'(\d+)', out_str.decode())[0])
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
    dtime_all = str_delta_sec(dtime_all)
    log_batchjob.info(f"All jobs ({name}, {job_name}) have finished after "
                      f"{dtime_all} ({dtime_sub:.1f}s submission): "
                      f"{nb_completed} completed, {nb_failed} failed.")
    out_files = glob.glob(path_to_out + "job_*.pkl")
    if len(out_files) < len(params):
        msg = f'Batch processing error during execution of {name} in job ' \
              f'\"{job_name}\": Found {len(out_files)}, expected {len(params)}.'
        log_batchjob.error(msg)
        raise ValueError(msg)
    if remove_jobfolder:
        _delete_folder_daemon(batchjob_folder, log_batchjob, job_name)
    return path_to_out


def _delete_folder_daemon(dirname, log, job_name, timeout=60):

    def _delete_folder(dn, lg, to=60):
        start = time.time()
        e = ''
        while timeout > time.time() - start:
            try:
                shutil.rmtree(dn)
                break
            except OSError as e:
                e = str(e)
                time.sleep(5)
        if time.time() - start > timeout:
            lg.warning(f'Deletion of job folder "{dn}" timed out after {to}s. OSError: {e}')
            shutil.rmtree(dn, ignore_errors=True)
            if os.path.exists(dn):
                dn_del = f"{os.path.dirname(dn)}/DEL/{os.path.basename(dn)}_DEL"
                if os.path.exists(os.path.dirname(dn_del)):
                    shutil.rmtree(os.path.dirname(dn_del), ignore_errors=True)
                os.makedirs(os.path.dirname(dn_del), exist_ok=True)
                shutil.move(dn, dn_del)

    t = threading.Thread(name=f'jobfold_delete_{job_name}', target=_delete_folder,
                         args=(dirname, log))
    t.setDaemon(True)
    t.start()


def jobstates_slurm(job_name: str, start_time: str,
                    max_retry: int = 10) -> Dict[int, str]:
    """
    Generates a dictionary which stores the state of every job belonging to
    `job_name`.

    Args:
        job_name:
        start_time: The following formats are allowed: MMDD[YY] or MM/DD[/YY]
            or MM.DD[.YY], e.g. ``datetime.datetime.today().strftime("%m.%d")``.
        max_retry: Number of retries for ``sacct`` SLURM query if failing (5s
            sleep in-between).

    Returns:
        Dictionary with the job states. (key: job ID, value: state)
    """
    # TODO: test!
    cmd_stat = f"sacct -b --name {job_name} -u {username} -S {start_time}"
    job_states = dict()
    cnt_retry = 0
    while True:
        process = subprocess.Popen(cmd_stat, shell=True,
                                   stdout=subprocess.PIPE)
        out, err = process.communicate()
        if process.returncode != 0:
            log_mp.warning(f'Delaying SLURM job state queries due to an error. '
                           f'Attempting again in 5s. {err}')
            time.sleep(5)
            cnt_retry += 1
            if cnt_retry == max_retry:
                log_mp.error(f'Could not query job states from SLURM: {err}\n'
                             f'Aborting due to maximum number of retries.')
                break
            continue
        for line in out.decode().split('\n'):
            str_parsed = re.findall(r"(\d+)[\s,\t]+([A-Z]+)", line)
            if len(str_parsed) == 1:
                str_parsed = str_parsed[0]
                job_states[int(str_parsed[0])] = str_parsed[1]
        break
    return job_states


def batchjob_fallback(params, name, n_cores=1, suffix="", script_folder=None, python_path=None, remove_jobfolder=False,
                      show_progress=True, log=None, overwrite=False, job_folder=None):
    """
    # TODO: utilize log and error files ('path_to_err', path_to_log')
    Fallback method in case no batchjob submission system is available.

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
    overwrite:
    job_folder:

    Returns
    -------
    str
        Path to output.
    """
    if python_path is None:
        python_path = python_path_global
    if job_folder is None:
        job_folder = "{}/{}_folder{}/".format(global_params.config.qsub_work_folder,
                                              name, suffix)
    if os.path.exists(job_folder):
        if not overwrite:
            raise FileExistsError(f'Batchjob folder already exists at "{job_folder}". '
                                  f'Please make sure it is safe for deletion, then set overwrite=True')
        shutil.rmtree(job_folder, ignore_errors=True)
    job_folder = job_folder.rstrip('/')
    if log is None:
        log_batchjob = initialize_logging("{}".format(name + suffix),
                                          log_dir=job_folder)
    else:
        log_batchjob = log
    n_max_co_processes = cpu_count()
    n_max_co_processes = np.min([cpu_count() // n_cores, n_max_co_processes])
    n_max_co_processes = np.min([n_max_co_processes, len(params)])
    n_max_co_processes = np.max([n_max_co_processes, 1])
    log_batchjob.info(f'Started BatchJobFallback script "{name}" with {len(params)} tasks'
                      f' using {n_max_co_processes} parallel jobs, each using {n_cores} core(s).')
    start = time.time()

    if script_folder is not None:
        path_to_scripts = script_folder
    else:
        path_to_scripts = path_to_scripts_default
    path_to_script = f'{path_to_scripts}/QSUB_{name}.py'
    if not os.path.exists(path_to_script):
        if os.path.exists(f'{path_to_scripts}/batchjob_{name}.py'):
            path_to_script = f'{path_to_scripts}/batchjob_{name}.py'
        else:
            raise FileNotFoundError(f'Specified script does not exist: {path_to_script}')

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
                                      show_progress=show_progress, nb_cpus=n_max_co_processes)
    out_files = glob.glob(path_to_out + "*.pkl")
    if len(out_files) < len(params):
        # report errors
        msg = 'Critical errors occurred during "{}". {}/{} Batchjob fallback worker ' \
              'failed.\n{}'.format(name, len(params) - len(out_files),
                                   len(params), out_str)
        log_mp.error(msg)
        log_batchjob.error(msg)
        raise ValueError(msg)
    elif len("".join(out_str)) != 0:
        msg = 'Warnings/errors occurred during ' \
              '"{}".:\n{} See logs at {} for details.'.format(name, out_str, job_folder)
        log_mp.warning(msg)
        log_batchjob.warning(msg)
    if remove_jobfolder:
        # nfs might be slow and leaves .nfs files behind (possibly from the slurm worker)
        try:
            shutil.rmtree(job_folder)
        except OSError:
            job_folder_old = f"{os.path.dirname(job_folder)}/DEL/{os.path.basename(job_folder)}_DEL"
            log_batchjob.warning(f'Deletion of job folder "{job_folder}" was not complete. Moving to '
                                 f'{job_folder_old}')
            if os.path.exists(os.path.dirname(job_folder_old)):
                shutil.rmtree(os.path.dirname(job_folder_old), ignore_errors=True)
            os.makedirs(os.path.dirname(job_folder_old), exist_ok=True)
            if os.path.exists(job_folder_old):
                shutil.rmtree(job_folder_old, ignore_errors=True)
            shutil.move(job_folder, job_folder_old)
    log_batchjob.debug('Finished "{}" after {:.2f}s.'.format(name, time.time() - start))
    return path_to_out


def fallback_exec(cmd_exec):
    """
    Helper function to execute commands via ``subprocess.Popen``.
    """
    ps = subprocess.Popen(cmd_exec, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
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
        if job_name[:10 if global_params.config['batch_proc_system'] == 'QSUB' else 8] in line:
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
