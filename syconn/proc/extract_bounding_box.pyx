# distutils: language = c++
from cython.view cimport array as cvarray
from libc.stdint cimport uint64_t, uint32_t
cimport cython
from libcpp.map cimport map
from cython.operator import dereference, postincrement
import timeit
from libc.stdlib cimport rand
import numpy as np
from libcpp.vector cimport vector as cpp_vector


ctypedef fused n_type:
    uint64_t
    uint32_t

def extract_bounding_box(n_type[:, :, :] chunk):
    box = dict()

    for x in range(chunk.shape[0]):
        for y in range(chunk.shape[1]):
            for z in range(chunk.shape[2]):

                key = chunk[x, y, z]
                coord = np.array([x, y, z])[None, ]  # shape: 1, 3
                if key in box:
                    box[key][:1] = np.min([box[key][:1], coord], axis=0)
                    box[key][1:] = np.max([box[key][1:], coord], axis=0)
                else:
                    box[key] = np.concatenate([coord, coord])  # use coordinate as min and max
    return box


# TODO move to test_extract_bounding_box.py
def create_toy_data(int size1, int size2, int size3, int moduloo):
    np.random.seed(0)
    cdef uint64_t[:, :, :] matrix = cvarray(shape = (size1, size2, size3),
                                     itemsize = sizeof(uint64_t), format = 'Q')
    for i in range(size1):
        for j in range(size2):
            matrix[i, j] = np.random.randint(moduloo, size=1)
    return matrix


def main():
    cdef uint64_t [:, :, :] chunk
    chunk = create_toy_data(15, 15, 15, 10)
    extract_bounding_box(chunk)
    print("done")

# main()