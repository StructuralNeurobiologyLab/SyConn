
import numpy as np
from scipy import spatial
from numpy import array as arr
from heraca.processing import ray_casting
import time
try:
    from DatasetUtils import knossosDataset as KnossosDataset
except:
    from dataset_utils import knossosDataset as KnossosDataset
from scipy import ndimage, sparse
import networkx
from scipy.spatial import ConvexHull
import re
import cPickle as pickle
__author__ = 'pschuber'


def get_orth_plane(node_com):
    """
    Calculates orthogonal plane and skeleton interpolation for each node.
    :param node_com: Spatially ordered list of nodes
    :return: orthogonal vector and vector representing skeleton interpolation
    at each node
    """
    lin_interp = np.zeros((len(node_com), 3), dtype=np.float)
    lin_interp[1:-1] = node_com[2:]-node_com[:-2]
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
    """
    :param axis: array Rotation-axis
    :param theta: float Angle to rotate
    :return: rotation matrix associated with counterclockwise rotation about
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
    x = np.linalg.det([[1, a[1], a[2]],
         [1, b[1], b[2]],
         [1, c[1], c[2]]])
    y = np.linalg.det([[a[0], 1, a[2]],
         [b[0], 1, b[2]],
         [c[0], 1, c[2]]])
    z = np.linalg.det([[a[0], a[1], 1],
         [b[0], b[1], 1],
         [c[0], c[1], 1]])
    magnitude = (x**2 + y**2 + z**2)**.5
    return (x/magnitude, y/magnitude, z/magnitude)


def poly_area(poly):
    """
    Calculates the area of a given polygon.
    :param poly: list of points
    :return: area of input polygon
    """
    if len(poly) < 3: # not a plane - no area
        return 0
    total = [0, 0, 0]
    N = len(poly)
    for i in range(N):
        vi1 = poly[i]
        vi2 = poly[(i+1) % N]
        prod = np.cross(vi1, vi2)
        total[0] += prod[0]
        total[1] += prod[1]
        total[2] += prod[2]
    result = np.dot(total, unit_normal(poly[0], poly[1], poly[2]))
    return abs(result/2)


def convex_hull_area(pts):
    """
    Calculates the surface area from a given point cloud using simplices of
    its convex hull. For the estimation of the synapse contact area, divide by
    a factor of two, in order to get the area of only one face (we assume that
    the contact site is sufficiently thin represented by the points).
    :param pts: np.array of coordinates in nm (scaled)
    :return: Area of the point cloud (nm^2)
    """
    area = 0
    ch = ConvexHull(pts)
    triangles = ch.points[ch.simplices]
    for triangle in triangles:
        area += poly_area(triangle)
    return area


def cell_object_coord_parser(voxel_tree):
    """
    Extracts unique voxel coords from object tree list for cell objects
    'mitos', 'p4' and 'az'.
    :param voxel_tree: annotation object containing voxels of cell objects
     ['mitos', 'p4', 'az']
    :return: coord arrays for 'mitos', 'p4' and 'az'
    """
    mito_coords= []
    p4_coords = []
    az_coords = []
    for node in voxel_tree.getNodes():
        comment = node.getComment()
        if 'mitos' in comment:
            mito_coords.append(node.getCoordinate())
        elif 'p4' in comment:
            p4_coords.append(node.getCoordinate())
        elif 'az' in comment:
            az_coords.append(node.getCoordinate())
        else:
            print "Couldn't understand comment:", comment
    print "Found %d mitos, %d az and %d p4." % (len(mito_coords), len(p4_coords),
                                                len(az_coords))
    return arr(mito_coords), arr(p4_coords), arr(az_coords)


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
        smallest_dists[i] = np.min(dists[annotation_ids==ix])

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
