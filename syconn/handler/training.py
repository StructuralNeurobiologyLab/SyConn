import subprocess
from multiprocessing import Process, Queue, Manager
import time


def start_training(q_in: Queue, dc: dict):
    while True:
        if q_in.empty():
            print('Worker finished.')
            break
        script_path, args = q_in.get()
        args_str = ' '.join([f'--{k}={v}' for k, v in args.items()])
        cmd_str = f'python {script_path} {args_str}'
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
    Launch ``len(args)`` trainings. Currently `n_workers` is hard-coded to 5, i.e. this method with launch 5 threads
    each running one training.

    Args:
        args: List of tuples of script path and arguments, i.e.
            [('...', dict(bs=10, scale_norm=30000), ('...', dict())]

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
