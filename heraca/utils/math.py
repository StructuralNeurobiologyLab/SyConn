
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


def get_radii_hull(args):
    """Wrapper-function for point cloud extraction from membrane prediction.
    Gets a bounding box with nodes, loads the membrane prediction for these
    and then calculates the radius and hull at each skeleton node.
    """
    # node attribute contains skel_interpolation, orthogonal plane and
    # bool if node is end node
    box, node_attr = args[0]
    q = args[1]
    scaling = args[2]
    mem_path = args[3]
    nb_rays = args[4]
    thresh_factor = args[5]
    max_dist_mult = args[6]
    kd = KnossosDataset()
    kd.initialize_from_knossos_path(mem_path)
    mem_shape = kd.boundary
    ray_buffer = arr([2000, 2000, 2000])/scaling
    prop_offset = np.max([np.min(box, axis=0) - ray_buffer,
                          [0,0,0]], axis=0).astype(np.int)
    prop_size = np.min([np.max(box, axis=0) + ray_buffer, mem_shape],
                       axis=0) - prop_offset
    assert np.prod(prop_size) < 10e9, "Bounding box too big!"
    mem = kd.from_raw_cubes_to_matrix(prop_size.astype(np.int32),
                                      prop_offset.astype(np.int32),
                                      show_progress=False)
    # thresholding membrane!!
    mem[mem <= 0.4*mem.max()] = 0
    #mem = pre_process_volume(mem)
    mem = mem.astype(np.uint8)
    threshold = mem.max() * thresh_factor
    #iterate over every node
    avg_radius_list = []
    all_points = []
    ids = []
    val_list = []
    todo_list = zip(list(box), [nb_rays] * len(box), list(node_attr))
    for el in todo_list:
        try:
            radius, ix, membrane_points, vals = ray_casting.ray_casting(
                el[0], el[1], el[2][0], el[2][1], el[2][2],
                scaling, threshold, prop_offset, mem, el[2][3], max_dist_mult)
        except IndexError, e:
            print "Problem at ray_casting part.", el
            print e
            print mem.shape
        all_points.append(arr(membrane_points, dtype=np.float32))
        avg_radius_list.append(radius)
        ids.append(ix)
        val_list.append(vals)
    q.put(ids)
    del(mem)
    return avg_radius_list, ids, all_points, val_list


def node_id2key(segdataobject, node_ids, filter_size):
    """
    Maps list indices in node_ids to keys of SegmentationObjects. Filters
    objects bigger than filter_size.
    :param segdataobject: SegmentationDataset of object type currently processed
    :param node_ids: List of list containing annotated object ids for each node
    :param filter_size: int minimum number of voxels of object
    :return: List of objects keys
    """

    for node in node_ids:
        for obj in node:
            if segdataobject.sizes[obj] < filter_size:
                node[node.index(obj)] = -1
            else:
                key = segdataobject.ids[obj]
                node[node.index(obj)] = key
    node_ids = [filter(lambda a: a != -1, node) for node in node_ids]
    return node_ids


def outlier_detection(point_list, min_num_neigh, radius):
    """
    Finds hull outlier using point density criterion.
    :param point_list: List of coordinates
    :param min_num_neigh: int Minimum number of neighbors, s.t. hull-point survives.
    :param radius: int Radius in nm to look for neighbors
    :return: Cleaned point cloud
    """
    print "Starting outlier detection."
    if arr(point_list).ndim != 2:
        points = arr([point for sublist in point_list for point in sublist])
    else:
        points = arr(point_list)
    tree = spatial.cKDTree(points)
    nb_points = float(len(points))
    print "Old #points:\t%d" % nb_points
    new_points = np.ones((len(points), )).astype(np.bool)
    for ii, coord in enumerate(points):
        neighbors = tree.query_ball_point(coord, radius)
        num_neighbors = len(neighbors)
        new_points[ii] = num_neighbors>=min_num_neigh
    print "Found %d outlier." % np.sum(~new_points)
    return arr(new_points)


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


def helper_get_voxels(obj):
    """
    Helper function to receive object voxels.
    :param obj: SegmentationObject
    :return: array voxels
    """
    try:
        voxels = obj.voxels
    except KeyError:
        return np.array([])
    return voxels


def helper_get_hull_voxels(obj):
    """
    Helper function to receive object hull voxels.
    :param obj: SegmentationObject
    :return: array hull voxels
    """

    return obj.hull_voxels


def hull2text(hull_coords, normals, path):
    """
    Writes hull coordinates and normals to xyz file. Each line corresponds to
    coordinates and normal vector of one point x y z n_x n_y n_z.
    :param hull_coords: array
    :param normals: array
    :param path: str
    """
    print "Writing hull to .xyz file.", path
    # add ray-end-points to nml and to txt file (incl. normals)
    file = open(path, 'wb')
    for i in range(hull_coords.shape[0]):
        end_point = hull_coords[i]
        normal = normals[i]
        file.write("%d %d %d %0.4f %0.4f %0.4f\n" %
                   (end_point[0], end_point[1], end_point[2], normal[0],
                    normal[1], normal[2]))
    file.close()


def obj_hull2text(id_list, hull_coords_list, path):
    """
    Writes object hull coordinates and corresponding object ids to txt file.
    Each line corresponds to id and coordinate vector of one object:
     id x y z
    :param id_list: array
    :param hull_coords_list: array
    :param path: str
    """
    print "Writing object hull to .txt file.", path
    # add ray-end-points to nml and to txt file (incl. normals)
    file = open(path, 'wb')
    for i in range(len(hull_coords_list)):
        coord = hull_coords_list[i]
        file.write("%d %d %d\n" % (coord[0], coord[1], coord[2]))
    file.close()
    if id_list == []:
        return
    file = open(path[:-4]+'_id.txt', 'wb')
    for i in range(len(hull_coords_list)):
        id = id_list[i]
        file.write("%d\n" % id)
    file.close()


def syns_btw_annos(anno_a, anno_b, max_hull_dist, concom_dist):
    """
    Computes contact sites between two annotation objects and returns hull
     points of both skeletons near contact site.
    :param anno_a: Annotation object A
    :param anno_b: Annotation object B
    :param max_hull_dist: Maximum distance between skeletons in nm
    :return: List of hull coordinates for each contact site.
    """
    hull_a = anno_a._hull_coords
    hull_b = anno_b._hull_coords
    tree_a = spatial.cKDTree(hull_a)
    tree_b = spatial.cKDTree(hull_b)
    if len(hull_a) == 0 or len(hull_b) == 0:
        print "One skeleton hull is empty!! Skipping pair."
        return [], []
    contact_ids = tree_a.query_ball_tree(tree_b, max_hull_dist)
    num_neighbours = arr([len(sublist) for sublist in contact_ids])
    contact_coords_a = hull_a[num_neighbours>0]
    contact_ids_b = set([id for sublist in contact_ids for id in sublist])
    contact_coords_b = hull_b[list(contact_ids_b)]
    if contact_coords_a.ndim == 1:
        contact_coords_a = contact_coords_a[None, :]
    if contact_coords_b.ndim == 1:
        contact_coords_b = contact_coords_a[None, :]
    contact_coords = np.concatenate((contact_coords_a, contact_coords_b), axis=0)
    if contact_coords.shape[0] >= 0.3*(len(hull_a)+len(hull_b)):
        print "Found too many contact_coords, " \
              "assuming similar skeleton comparison."
        return [], []
    if contact_coords.shape[0] == 0:
        return [], []
    pdists = spatial.distance.pdist(contact_coords)
    pdists[pdists > concom_dist] = 0
    pdists = sparse.csr_matrix(spatial.distance.squareform(pdists))
    nb_cc, labels = sparse.csgraph.connected_components(pdists)
    cs_list = []
    for label in set(labels):
        curr_label_ixs = labels == label
        cs_list.append(contact_coords[curr_label_ixs])
    tree_a_b = spatial.cKDTree(np.concatenate((hull_a, hull_b), axis=0))
    contact_site_coord_ids = []
    min_id_b = len(hull_a)
    for cs in cs_list:
        ids_temp = tree_a_b.query(cs, 1)[1]
        in_b = arr(ids_temp>=min_id_b, dtype=np.bool)
        contact_site_coord_ids.append(in_b)
    return cs_list, contact_site_coord_ids


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
