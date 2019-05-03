# distutils: language = c++

cimport cython
from libcpp cimport bool
from libc.stdint cimport int, float
from libcpp.vector cimport vector

ctypedef fused pyf:
    float
    double


def in_bounding_box(pyf[:,:] coords, pyf[:, :] bounding_box):

    cdef float edge_sizes[3]
    edge_sizes[:] = [bounding_box[1,0]/2, bounding_box[1,1]/2, bounding_box[1,2]/2 ]

    cdef vector[bool] inlier
    cdef bool x_cond, y_cond, z_cond
    cdef pyf x, y, z

    for i in range(coords.shape[0]):
        x = coords[i, 0] - bounding_box[0, 0]
        y = coords[i, 1] - bounding_box[0, 1]
        z = coords[i, 2] - bounding_box[0, 2]
        x_cond = (x > -edge_sizes[0]) & (x < edge_sizes[0])
        y_cond = (y > -edge_sizes[1]) & (y < edge_sizes[1])
        z_cond = (z > -edge_sizes[2]) & (z < edge_sizes[2])
        inlier.push_back(x_cond & y_cond & z_cond)

    return inlier