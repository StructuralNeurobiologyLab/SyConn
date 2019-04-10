from knossos_utils import chunky
from syconn import global_params
from syconn.handler.basics import kd_factory
import numpy as np
from syconn.handler.basics import chunkify
from syconn.mp import batchjob_utils as qu, mp_utils as sm
from sys import getsizeof

def check_zeros():
    a = np.zeros(shape=(3, 3), dtype=np.bool)
    a[1, 1] = 1
    b = np.zeros(shape=(3, 3), dtype=np.bool)
    b[1, 1] = 1
    b[1, 0] = 1
    res = np.sum(a & b)
    print("res =", res)

def main():
    check_zeros();


if __name__ == "__main__":
    main()