import subprocess
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
        process = subprocess.Popen(cmd_str, shell=True, stdout=subprocess.PIPE)
        out_str, err = process.communicate()
        if process.returncode != 0:
            raise RuntimeError(str(err) + str(out_str))


def worker_train(args):
    """
    Launch ``len(args)`` trainings. Currently `n_workers` is hard-coded to 5, i.e. this method with launch 5 threads
    each running one training.

    Args:
        args: List of tuples of script path and arguments, i.e.
            [('...', dict(bs=10, scale_norm=30000), ('...', dict())]

    """
    n_worker = 5
    q_in = Queue()
    for el in args:
        q_in.put(el)
    trainers = [Process(target=start_training, args=(q_in, )) for _ in range(n_worker)]
    for t in trainers:
        t.start()
    for t in trainers:
        t.join()
        t.close()
