# distutils: language = c++
from cython.view cimport array as cvarray
from libc.stdint cimport uint64_t, uint32_t, int
cimport cython
from libcpp.map cimport map
from cython.operator import dereference, postincrement
import timeit
from libc.stdlib cimport rand
import numpy as np
from libcpp.vector cimport vector as cpp_vector


ctypedef fused n_type:
    uint64_t
    uint32_

def extract_bounding_box(n_type[:, :, :] chunk):
    box = dict()

    for x in range(chunk.shape[0]):
        for y in range(chunk.shape[1]):
            for z in range(chunk.shape[2]):

                
                box[chunk[x,y,z]].xMin = min(box[chunk[x,y,z]].xMin, x)
                box[chunk[x,y,z]].xMax = max(box[chunk[x,y,z]].xMax, x)

                box[chunk[x,y,z]].yMin = min(box[chunk[x,y,z]].yMin, y)
                box[chunk[x,y,z]].yMax = min(box[chunk[x,y,z]].yMax, y)

                box[chunk[x,y,z]].zMin = min(box[chunk[x,y,z]].zMin, z)
                box[chunk[x,y,z]].zMax = min(box[chunk[x,y,z]].zMax, z)
    return box


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

main()