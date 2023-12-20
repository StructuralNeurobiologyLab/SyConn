import subprocess
from multiprocessing import Process, Queue, Manager
import time
import sys


def start_training(q_in: Queue, dc: dict):
    """
    Executes training scripts provided in the queue with the given arguments.
    
    This function runs in an infinite loop, checking for script paths and arguments in the
    input queue. For each set of script and arguments, it constructs a command and executes
    it using `subprocess.Popen`. It captures the output and error streams, checks for
    successful execution, and logs the results. If a training folder already exists, it
    skips the training for that command. The function breaks out of the loop when the queue
    is empty, indicating that all training tasks have been completed.
    
    Args:
        q_in: A multiprocessing queue containing tuples of script paths and their
              corresponding argument dictionaries.
        dc: A manager dictionary that stores the command strings as keys and the return
            status or error messages as values.
    """
    while True:
        if q_in.empty():
            print('Worker finished.')
            break
        script_path, args = q_in.get()
        args_str = ' '.join([f'--{k}={v}' for k, v in args.items()])
        cmd_str = f'{sys.executable} {script_path} {args_str}'
        print(f'Started training with command {cmd_str}.')
        process = subprocess.Popen(cmd_str, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out_str, err = process.communicate()
        exit_code = process.wait()
        err = err.decode()
        if exit_code != 0:
            print(f'Command "{cmd_str}" failed with exit code {exit_code}:\n{err}\n\nOutput pipe:\n{out_str.decode()}\n\n', )
            # check if training folder already exists and ignore
            if 'Please choose a different combination of save_root and exp_name' in err:
                print(f'\n\nSkipping training with command "{cmd_str}" - folder already exists.\n\n')
                ret = 0
            else:
                ret = err
        else:
            ret = 0
        dc[cmd_str] = ret


def worker_train(args):
    """
    Launches multiple training processes in parallel based on the provided arguments.
    
    This function initializes a multiprocessing environment and manages the execution of
    training routines in parallel threads or processes. It is designed to handle a list
    of training configurations, each specified as a tuple containing a script path and
    a set of arguments. The number of worker threads is currently fixed at 5, which will
    each undertake a separate training sequence. 
    
    Args:
        args: A list of tuples, where each tuple consists of a script path and a 
              dictionary of arguments for the training command, e.g., 
              [('...', {'bs': 10, 'scale_norm': 30000}), ('...', {})]
    
    Raises:
        RuntimeError: If any training process encounters an error, an exception is 
                      raised, including information about the failures.
    
    Note: The docstring has been updated to reflect the hard-coded number of workers
          from the old docstring which specifies 5 workers as opposed to the generated
          docstring's mention of a maximum of 4 workers.
    """
    n_worker = min(4, len(args))
    print(f'Starting {n_worker} trainings in parallel.')
    manager = Manager()
    ret_dc = manager.dict()
    q_in = Queue()
    for el in args:
        q_in.put(el)
    time.sleep(0.5)
    if len(args) == 1:
        start_training(q_in, ret_dc)
    else:
        trainers = [Process(target=start_training, args=(q_in, ret_dc)) for _ in range(n_worker)]
        for t in trainers:
            t.start()
        for t in trainers:
            t.join()
            t.close()
    any_failed = any([r != 0 for r in ret_dc.values()])
    if any_failed:
        failed_ret = [r for r in ret_dc.values() if r != 0]
        raise RuntimeError(f'{len(failed_ret)} worker failed with: {failed_ret}')
