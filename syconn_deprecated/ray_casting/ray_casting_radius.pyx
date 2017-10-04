#!python

# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max-Planck-Institute for Medical Research, Heidelberg, Germany
# Authors: Sven Dorkenwald, Philipp Schubert, Joergen Kornfeld

#cython: boundscheck=False
import numpy as np
from libc.math cimport isnan, sqrt
cimport numpy as np
from libc.stdlib cimport rand
cdef extern from "limits.h":
    int INT_MAX
DTYPE = np.float
DTYPE_i = np.int

def ray_casting_radius(np.ndarray[np.int64_t, ndim=1] node_pos, int nb,
                      np.ndarray[np.float64_t, ndim=1] skel_interp,
                      np.ndarray[np.float64_t, ndim=1]orth_plane, unsigned int ix,
                      np.ndarray[np.int64_t, ndim=1] scaling,
                      float threshold, np.ndarray[np.int64_t, ndim=1] prop_offset,
                      np.ndarray[np.uint8_t, ndim=3] mem, bint end_node,
                      float max_dist_multi):
    """calculates the radius for the given node, using position of node node_pos, its index ix to get its
    corresponding value in the "global" orth. plane array and skeleton interpolation. Nb is the number of
    rays used to estimate the radius.
    """
    cdef float val, val_tmp, angle, angle2
    cdef unsigned int cnt2
    cdef np.ndarray[np.int64_t, ndim=1] pixel_pos, old_pos
    cdef np.ndarray[np.float64_t, ndim=1] dir_vector, pos, hull_point, cross_vec
    radii = []
    membrane_points = []
    vals2 = []
    vals = []
    if not end_node:
        angles = np.linspace(0, 2*np.pi*5, nb*5)
    else:
        angles = np.linspace(0, 2*np.pi*10, nb*10)
        #print "End Node", ix
    #get the average radius of several rays defined by dir_vector
    for i in range(len(angles)):
        dir_vector = np.dot(rotation_matrix(skel_interp, angles[i]), orth_plane)
        if np.linalg.norm(dir_vector) == 0.:
            #print "dir_vector has zero length", ix
            break
        if i >= nb:
            cross_vec = np.cross(orth_plane, skel_interp)
            # scatter in 60 degree
            if not end_node:
                angle2 = rand()/float(INT_MAX)*2*np.pi/3 - np.pi/3
            # scatter in 270 degree
            else:
                angle2 = rand()/float(INT_MAX)*2*np.pi - np.pi
                angle2 = np.sign(angle2)*90 + angle2
                #print "End Node[%d] angle2: %0.2f" % (ix, angle2)
            dir_vector = np.dot(rotation_matrix(cross_vec, angle2), dir_vector)
        dir_vector /= np.linalg.norm(dir_vector)*4
        pixel_pos = node_pos-prop_offset
        pos = np.array(pixel_pos, dtype=np.float64)
        vals = []
        val = 0
        cnt2 = 0
        while val < threshold:
            cnt2 += 1
            if cnt2 > 2000:
                print "Maximum iterations reached!"
                pos = None
                break
            old_pos = np.array(pixel_pos, dtype=DTYPE_i)
            val_tmp = mem[pixel_pos[0], pixel_pos[1], pixel_pos[2]]
            val += val_tmp
            vals.append(val_tmp)
            while (pixel_pos[0]==old_pos[0] and pixel_pos[1]==old_pos[1] and pixel_pos[2]==old_pos[2]):
                pos += dir_vector
                pixel_pos = np.array(np.round(pos), dtype=DTYPE_i)
            if (pixel_pos[0] < 0) or (pixel_pos[0] >= mem.shape[0]-1) or (pixel_pos[1] < 0) or (pixel_pos[1] >= mem.shape[1]-1)\
                or (pixel_pos[2] < 0) or (pixel_pos[2] >= mem.shape[2]-1):
                pos = None
                #print "outside box", pos+prop_offset
                break
        if pos is None:
            continue
        radii.append(sqrt(np.sum((pos*scaling-(node_pos-prop_offset)*scaling)**2)) / 10.)
        # CHANGED FOR SHRINKED HULL !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # hull_point = node_pos + (pos-(node_pos-prop_offset))/3
        hull_point = pos+prop_offset
        membrane_points.append(hull_point)
    vals2.append(vals)
    # setting arbitrary radius
    if radii == []:
        radii.append(0.)
    #print "[%d] Found radius of %s" % (ix, str(np.array(radii).mean()))
    radius = np.median(np.array(radii))
    nb_mem_points = len(membrane_points)
    resulting_points = []
    for i in range(nb_mem_points):
        curr_point = membrane_points[i]-prop_offset
        point_dist = sqrt(np.sum((curr_point*scaling-
                                  (node_pos-prop_offset)*scaling)**2)) / 10.
        if point_dist < max_dist_multi*radius:
            resulting_points.append(membrane_points[i])
    return radius, ix, resulting_points, vals2

def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    theta = np.asarray(theta)
    axis = axis/np.sqrt(np.dot(axis, axis))
    a = np.cos(theta/2)
    b, c, d = -axis*np.sin(theta/2)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                     [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                     [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])