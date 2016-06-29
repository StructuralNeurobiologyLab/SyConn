import os
from heraca.utils.datahandler import DataHandlerObject, create_skel, get_paths_of_skelID, \
load_mapped_skeleton

def compute_skeleton_hull(tracing_id, recmpute=False):
    """
    :param tracing_id: tracing id
    :type tracing_id: int
    :return hull coordinates, hull normals
    """
    mapped_skel_p = get_paths_of_skelID([tracing_id])[0]
    if os.path.isfile(mapped_skel_p) and not recompute:
        skel = load_mapped_skeleton(mapped_skel_p)
    else:
        dh = DataHandlerObject(load_obj=False)
        skel = creat_skel(dh, tracing_id, id=tracing_id)
        skel.hull_sampling(detect_outlier=detect_outlier, thresh=thresh,
                           nb_neighbors=nb_neighbors,
                           neighbor_radius=neighbor_radius,
                           max_dist_mult=max_dist_mult)
    return skel.hull_coords, skel.hull_normals


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
