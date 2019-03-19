# distutils: language = c++
cimport cython
from libcpp cimport bool
##from libc.stdint cimport int, float, double
from libcpp.vector cimport vector
import numpy as np


def in_bounding_box(double[:,:] coords, bounding_box):

    cdef double edge_sizes[3]
    edge_sizes[:] =[bounding_box[1][0]/2, bounding_box[1][1]/2, bounding_box[1][2]/2]
    cdef double centers[3]
    centers[:] =[bounding_box[0][0], bounding_box[0][1], bounding_box[0][2]]

    for i in range(coords.shape[0]):
        coords[i, 0] = coords[i, 0] - centers[0]
        coords[i, 1] = coords[i, 1] - centers[1]
        coords[i, 2] = coords[i, 2] - centers[2]


    cdef vector[bool] inlier
    cdef bool x_cond, y_cond, z_cond

    for i in range(coords.shape[0]):
        x_cond = (coords[i, 0] > -edge_sizes[0]) & (coords[i, 0] < edge_sizes[0])
        y_cond = (coords[i, 1] > -edge_sizes[1]) & (coords[i, 1] < edge_sizes[1])
        z_cond = (coords[i, 2] > -edge_sizes[2]) & (coords[i, 2] < edge_sizes[2])
        inlier.push_back(x_cond & y_cond & z_cond)

    return inlier