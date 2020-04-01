import sys
import subprocess
from syconn.handler import basics
from multiprocessing import Process, Queue


def start_training(q_in: Queue):
    while True:
        if q_in.empty():
            print('Worker finished.')
            break
        script_path, args = q_in.get()
        args_str = ' '.join([f'--{k}={v}' for k, v in args.items()])
        cmd_str = f'python {script_path} {args_str}'
        print(f'Started training with command {cmd_str}.')
        subprocess.run(cmd_str, shell=True)


def worker_train(args):
    """
    Launch ``len(args)`` trainings.

    Args:
        args: List of tuples of script path and arguments, i.e.
            [('...', dict(bs=10, scale_norm=30000), ('...', dict())]

    """
    n_worker = 3
    q_in = Queue()
    for el in args:
        q_in.put(el)
    trainers = [Process(target=start_training, args=(q_in, )) for _ in range(n_worker)]
    for t in trainers:
        t.start()
    for t in trainers:
        t.join()
        t.close()
