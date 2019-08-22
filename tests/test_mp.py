# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Philipp Schubert, Joergen Kornfeld

from multiprocessing import cpu_count, Pool
import numpy as np
import time


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def test_func(x):
    for i in range(100):
        _ = np.linalg.norm(np.sqrt(x ** 2) * 5 / x * x ** 2 - x + x, axis=0)
    return


if __name__ == '__main__':
    data = np.arange(5000000).reshape(10000, 500) + 1

    n_worker = cpu_count()
    data = list(chunks(data, n_worker))

    start = time.time()
    for d in data:
        test_func(d)
    dt_single = time.time() - start

    start = time.time()
    print("Starting MP with {} workers.".format(n_worker))
    pool = Pool(processes=n_worker)
    pool.map(test_func, data)
    dt_mp = time.time() - start

    print('Single process:\t{:.4f} s\n'
          'Multi processes:\t{:.4f} s'.format(dt_single, dt_mp))
