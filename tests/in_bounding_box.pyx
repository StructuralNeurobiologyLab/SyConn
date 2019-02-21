# distutils: language = c++
cimport cython
from libcpp cimport bool
from libc.stdint cimport int, float
from libcpp.vector cimport vector
from cython.view cimport array as cvarray
import timeit
from libc.stdlib cimport rand
import numpy as np
from numpy import multiply



def in_bounding_box(float[:,:] coords, float[:, :] bounding_box):

    cdef float edge_sizes[3]
    edge_sizes[:] =[bounding_box[1,0]/2, bounding_box[1,1]/2, bounding_box[1,2]/2 ]

    for i in range(coords.shape[0]):
        coords[i, 0] = coords[i, 0] - bounding_box[0, 0]
        coords[i, 1] = coords[i, 1] - bounding_box[0, 1]
        coords[i, 2] = coords[i, 2] - bounding_box[0, 2]


    cdef vector[bool] inlier
    cdef bool x_cond, y_cond, z_cond

    for i in range(coords.shape[0]):
        x_cond = (coords[i, 0] > -edge_sizes[0]) & (coords[i, 0] < edge_sizes[0])
        y_cond = (coords[i, 1] > -edge_sizes[1]) & (coords[i, 1] < edge_sizes[1])
        z_cond = (coords[i, 2] > -edge_sizes[2]) & (coords[i, 2] < edge_sizes[2])
        inlier.push_back(x_cond & y_cond & z_cond)

    return inlier



def create_toy_data(int size1, int size2, int moduloo):
    np.random.seed(0)
    cdef float[:, :] matrix = cvarray(shape=(size1, size2), itemsize = sizeof(float), format = 'f')
    for i in range(size1):
        for j in range(size2):
            matrix[i, j] = np.random.randint(moduloo, size=1)
    return matrix


def wrapper(func, *args, **kwargs):
    def wrapped():
        return func(*args, **kwargs)
    return wrapped





def main():
    cdef float [:, :] coords
    cdef float[:, :] bounding_box
    coords = create_toy_data(15, 3, 9)
    bounding_box = create_toy_data(2, 3, 10)

    print(in_bounding_box(coords,bounding_box))
    #print(np.array(bounding_box))

    '''
    print("in_bounding_box__Maria")
    wrapped = wrapper(in_bounding_box, coords, bounding_box)
    print(timeit.timeit(wrapped, number=1000000))
    '''

main()