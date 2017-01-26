# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max-Planck-Institute for Medical Research, Heidelberg, Germany
# Authors: Sven Dorkenwald, Philipp Schubert, Joergen Kornfeld

import numpy as np
from scipy import spatial
from numpy import array as arr
from math import pow, sqrt, ceil
from scipy import ndimage
from scipy.spatial import ConvexHull
from scipy.spatial.qhull import QhullError
import warnings
import re


def switch_array_entries(this_array, entries):
    entry_0 = this_array[entries[0]]
    this_array[entries[0]] = this_array[entries[1]]
    this_array[entries[1]] = entry_0
    return this_array


def negative_to_zero(a):
    """
    Sets negative values of array a to zero.
    :param a: numpy array
    :return: array a with non negativ values.
    """
    if a > 0:
        return a
    else:
        return 0


def get_orth_plane(node_com):
    """
    Calculates orthogonal plane and skeleton interpolation for each node.
    :param node_com: Spatially ordered list of nodes
    :return: orthogonal vector and vector representing skeleton interpolation
    at each node
    """
    lin_interp = np.zeros((len(node_com), 3), dtype=np.float)
    if len(node_com) < 2:
        return np.zeros((len(node_com), 3), dtype=np.float), lin_interp
    lin_interp[1:-1] = node_com[2:] - node_com[:-2]
    lin_interp[0] = node_com[1] - node_com[0]
    lin_interp[-1] = node_com[-1] - node_com[-2]
    n = np.linalg.norm(lin_interp, axis=1)
    lin_interp[..., 0][n != 0] = (lin_interp[..., 0] / n)[n != 0]
    lin_interp[..., 1][n != 0] = (lin_interp[..., 1] / n)[n != 0]
    lin_interp[..., 2][n != 0] = (lin_interp[..., 2] / n)[n != 0]
    x = lin_interp[:, 0]
    y = lin_interp[:, 1]
    z = lin_interp[:, 2]
    orth_plane = np.zeros((len(node_com), 3), dtype=np.float)
    inner_prod = np.zeros((len(node_com), 1), dtype=np.float)
    for i in range(len(node_com)):
        if not np.allclose(z[i], 0.0):
            orth_plane[i, :] = arr([x[i], y[i], -1.0*(x[i]**2+y[i]**2)/z[i]])
        else:
            if not np.allclose(y[i], 0.0):
                orth_plane[i, :] = arr([x[i], -1.0*(x[i]**2+z[i]**2)/y[i],
                                        z[i]])
            else:
                if not np.allclose(x[i], 0.0):
                    orth_plane[i, :] = arr([-1.0*(y[i]**2+z[i]**2)/x[i],
                                            y[i], z[i]])
                else:
                    print "WARNING: Problem finding orth. plane. ", i, lin_interp[i]
        inner_prod[i] = np.inner(orth_plane[i], lin_interp[i])
        n = np.linalg.norm(orth_plane[i])
        if n != 0:
            orth_plane[i] /= n
    assert np.allclose(inner_prod, 0, atol=1e-6), "Planes are not orthogonal!"
    return orth_plane, lin_interp


def rotation_matrix(axis, theta):
    """Get rotation matrix along axis and angle

    Parameters
    ----------
    axis: np.array
        rotation-axis
    theta: float
        angle to rotate

    Returns
    -------
    np.array
        rotation matrix associated with counterclockwise rotation about
        the given axis by theta radians.
    """
    axis = np.asarray(axis)
    theta = np.asarray(theta)
    axis = axis / np.sqrt(np.dot(axis, axis))
    a = np.cos(theta / 2)
    b, c, d = -axis * np.sin(theta / 2)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                     [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                     [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])


def pre_process_volume(vol):
    """
    Processes raw data to get membrane shapes.
    :param vol: array raw data
    :return: array membrane
    """
    thres = 120
    vol = ndimage.filters.gaussian_filter(vol, sigma=0.6, mode='wrap')
    binary = vol > thres
    edges = ndimage.filters.generic_gradient_magnitude(binary,
                                                       ndimage.filters.sobel)
    edges = (edges * 255).astype(np.uint8)
    return edges


def unit_normal(a, b, c):
    """
    Calculates the unit normal vector of a given polygon defined by the
    points a,b and c.
    :param a, b, c: Each is an array of length 3
    :return: unit normal vector
    """
    x = np.linalg.det([[1, a[1], a[2]], [1, b[1], b[2]], [1, c[1], c[2]]])
    y = np.linalg.det([[a[0], 1, a[2]], [b[0], 1, b[2]], [c[0], 1, c[2]]])
    z = np.linalg.det([[a[0], a[1], 1], [b[0], b[1], 1], [c[0], c[1], 1]])
    magnitude = (x**2 + y**2 + z**2)**.5
    return x / magnitude, y / magnitude, z / magnitude


def poly_area(poly):
    """
    Calculates the area of a given polygon.
    :param poly: list of points
    :return: area of input polygon
    """
    if len(poly) < 3:
        return 0
    total = [0, 0, 0]
    n = len(poly)
    for i in range(n):
        vi1 = poly[i]
        vi2 = poly[(i+1) % n]
        prod = np.cross(vi1, vi2)
        total[0] += prod[0]
        total[1] += prod[1]
        total[2] += prod[2]
    result = np.dot(total, unit_normal(poly[0], poly[1], poly[2]))
    return abs(result / 2)


def convex_hull_area(pts):
    """
    Calculates the surface area from a given point cloud using simplices of
    its convex hull. For the estimation of the synapse contact area, divide by
    a factor of two, in order to get the area of only one face (we assume that
    the contact site is sufficiently thin represented by the points).
    :param pts: np.array of coordinates in nm (scaled)
    :return: Area of the point cloud (nm^2)
    """
    if len(pts) < 4:
        return 0
    area = 0
    try:
        ch = ConvexHull(pts)
        triangles = ch.points[ch.simplices]
        for triangle in triangles:
            area += poly_area(triangle)
    except QhullError as e:
        # warnings.warn("%s encountered during calculation of convex hull "
        #               "area with %d points. Returning 0 nm^2." %
        #               (e, len(pts)), RuntimeWarning)
        pass
    return area


def cell_object_coord_parser(voxel_tree):
    """Extracts unique voxel coords from object tree list for cell objects
    'mitos', 'vc' and 'sj'.

    :param voxel_tree: annotation object containing voxels of cell objects
     ['mitos', 'vc', 'sj']
    :returns: coord arrays for 'mitos', 'vc' and 'sj'
    """
    mito_coords= []
    vc_coords = []
    sj_coords = []
    for node in voxel_tree.getNodes():
        comment = node.getComment()
        if 'mitos' in comment:
            mito_coords.append(node.getCoordinate())
        elif 'vc' in comment:
            vc_coords.append(node.getCoordinate())
        elif 'sj' in comment:
            sj_coords.append(node.getCoordinate())
        else:
            print "Couldn't understand comment:", comment
    return arr(mito_coords), arr(vc_coords), arr(sj_coords)


def helper_samllest_dist(args):
    """
    Returns the smallest distance of index ixs in dists.
    :param ixs: list of in Indices of objects
    :param annotation_ids: array of shape (m, )
    :param dists: array of shape (m, )
    :return: smallest distance of ixs found in dists.
    """
    ixs, annotation_ids, dists = args
    smallest_dists = np.ones((len(ixs, ))) * np.inf
    for i, ix in enumerate(ixs):
        smallest_dists[i] = np.min(dists[annotation_ids == ix])
    return ixs, smallest_dists


def get_box_coords(coord_array, min_pos, max_pos, ret_bool_array=False):
    """
    Reduce coord_array to coordinates in bounding box defined by
    global variable min_pos max_pos.
    :param coord_array: array of coordinates
    :return:
    """
    if len(coord_array) == 0:
        return np.zeros((0, 3))
    bool_1 = np.all(coord_array >= min_pos, axis=1) & \
             np.all(coord_array <= max_pos, axis=1)
    if ret_bool_array:
        return bool_1
    return coord_array[bool_1]


def get_normals(hull, number_fitnodes=12):
    """
    Calculate normals from given hull points using local convex hull fitting.
    Orientation of normals is found using local center of mass.
    :param hull: 3D coordinates of points representing cell hull
    :type hull: np.array
    :return: normals for each hull point
    """
    normals = np.zeros_like(hull, dtype=np.float)
    hull_tree = spatial.cKDTree(hull)
    dists, nearest_nodes_ixs = hull_tree.query(hull, k=number_fitnodes,
                                                distance_upper_bound=1000)
    for ii, nearest_ixs in enumerate(nearest_nodes_ixs):
        nearest_nodes = hull[nearest_ixs[dists[ii] != np.inf]]
        ch = ConvexHull(nearest_nodes, qhull_options='QJ Pp')
        triangles = ch.points[ch.simplices]
        normal = np.zeros((3), dtype=np.float)
        # average normal
        for triangle in triangles:
            cnt = 0
            n_help = unit_normal(triangle[0], triangle[1], triangle[2])
            if not np.any(np.isnan(n_help)):
                normal += np.abs(n_help)
        normal /= np.linalg.norm(normal)
        normal_sign = (hull[ii] - np.mean(nearest_nodes, axis=0))/\
                      np.abs(hull[ii] - np.mean(nearest_nodes, axis=0))
        normals[ii] = normal * normal_sign
    return normals


def calc_overlap(point_list_a, point_list_b, max_dist):
    """
    Calculates the portion of points in list b being similar (distance max_dist)
    to points from list a.
    :param point_list_a:
    :param point_list_b:
    :param max_dist:
    :return: Portion of similar points over number of points of list b and vice
    versa, overlap area in nm^2, centercoord of overlap area and coord_list of
    overlap points in point_list_b
    """
    point_list_a = arr(point_list_a)
    point_list_b = arr(point_list_b)
    tree_a = spatial.cKDTree(point_list_a)
    near_ids = tree_a.query_ball_point(point_list_b, max_dist)
    total_id_list = list(set([id for sublist in near_ids for id in sublist]))
    overlap_area = convex_hull_area(point_list_a[total_id_list]) / 1.e6
    nb_unique_neighbors = np.sum([1 for sublist in near_ids if len(sublist) > 0])
    portion_b = nb_unique_neighbors / float(len(point_list_b))
    tree_b = spatial.cKDTree(point_list_b)
    near_ids = tree_b.query_ball_point(point_list_a, max_dist)
    nb_unique_neighbors = np.sum([1 for sublist in near_ids if len(sublist) > 0])
    total_id_list = list(set([id for sublist in near_ids for id in sublist]))
    portion_a = nb_unique_neighbors / float(len(point_list_a))
    near_ixs = [ix for sublist in near_ids for ix in sublist]
    center_coord = np.mean(arr(point_list_b)[arr(near_ixs)], axis=0)
    return portion_b, portion_a, overlap_area, center_coord,\
           point_list_b[total_id_list]


def tuple_to_string(coordinate, sep=', ', dl='(', dr=')'):
    return dl + sep.join([str(x) for x in coordinate]) + dr


def coordinate_from_string(coord_string):
    coordinate_expression = '[(\[]{0,1}\s*(?P<x>-?\d+)\s*[,;]\s*(' \
                            '?P<y>-?\d+)\s*[,;]\s*(?P<z>-?\d+)\s*[)\]{0,1}]'
    try:
        x = int(re.search(coordinate_expression, coord_string).group('x'))
        y = int(re.search(coordinate_expression, coord_string).group('y'))
        z = int(re.search(coordinate_expression, coord_string).group('z'))
    except (AttributeError, ValueError, TypeError):
        return (None, None, None)

    return (x, y, z)


def coordinate_to_ewkt(coordinate, scale='dataset'):
    if isinstance(scale, str):
        scale = (1.0, 1.0, 1.0)

    coordinate = (coordinate[0] * scale[0],
                  coordinate[1] * scale[1],
                  coordinate[2] * scale[2])

    return "POINT(%s)" % (" ".join([str(x) for x in coordinate]),)


def has_equal_dimensions(c):
    """
    Return True if container types in iterable c have equal number of of
    elements, False otherwise.

    Example
    -------

    >>> a = set(range(0, 10))
    >>> b = range(0, 10)
    >>> has_equal_dimensions([a, b])
    True
    >>> a.add(100)
    >>> has_equal_dimensions([a, b])
    False
    """

    lens = [len(x) for x in c]
    if True in [bool(x - y) for x, y in zip(lens, lens[1:])]:
        return False
    else:
        return True


def average_coordinate(c):
    """
    Return the average coordinate (center of gravity) for an iterable of
    coordinates.

    Parameters
    ----------

    c : iterable of coordinates
        Coordinates are represented as lists and must have the same number of
        dimensions.

    Returns
    -------

    avg_coordinate : iterable

    Example
    -------

    >>> average_coordinate([[1, 2, 3], [4, 5, 6]])
    [2.5, 3.5, 4.5]
    >>> average_coordinate([])
    []
    """

    if not has_equal_dimensions(c):
        raise Exception('All coordinates must have equal number of dimensions '
            'to calculate average.')

    avg_coordinate = [sum([float(y) for y in x]) / len(x) for x in zip(*c)]

    return avg_coordinate


def euclidian_distance(c1, c2):
    return sqrt(pow((c2[0] - c1[0]), 2) +
     pow((c2[1] - c1[1]), 2) +
     pow((c2[2] - c1[2]), 2))


def interpolate_between(c1, c2):
    """
    Return list of coordinates from c1 to c2, including them. Coordinates
    are spaced by (approximately) one length unit.
    """

    delta = FloatCoordinate(c1) - FloatCoordinate(c2)
    dist = int(ceil(euclidian_distance(c1, c2)))
    step = delta / dist

    return [list(step * cur_step + c1) for cur_step in range(0, dist + 1)]


class Coordinate(list):
    """
    Represents a coordinate of arbitrary dimensionality.
    """
    def __init__(self, c):
        """
        Parameters
        ----------

        c : Iterable of numeric types
            Represents the coordinate.
        """
        list.__init__(self, c)

    def __eq__(self, other):
        if sum([x == y for x, y in zip(self, other)]) == len(self):
            return True
        else:
            return False

    def __add__(self, other):
        return type(self)([x + y for x, y in zip(self, other)])

    def __sub__(self, other):
        return type(self)([x - y for x, y in zip(self, other)])

    def __mul__(self, other):
        try:
            other[0]
            # element-wise multiplication
            return type(self)([x * y for x, y in zip(self, other)])
        except TypeError:
            # scalar multiplication
            return type(self)([x * other for x in self])

    def __div__(self, other):
        try:
            # scalar multiplication
            return type(self)([x / other for x in self])
        except TypeError:
            # element-wise multiplication
            return type(self)([x / y for x, y in zip(self, other)])


class FloatCoordinate(Coordinate):
    """
    Represent a coordinate of arbitrary dimensionality, using floats.
    """

    def __init__(self, c):
        c = [float(x) for x in c]
        super(FloatCoordinate, self).__init__(c)


