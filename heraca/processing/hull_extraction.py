import os
from utils.datahandler import DataHandler, create_skel, get_paths_of_skelID, \
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
        dh = datahandler.DataHandler(load_obj=False)
        skel = creat_skel(dh, tracing_id, id=tracing_id)
        skel.hull_sampling(detect_outlier=detect_outlier, thresh=thresh,
                           nb_neighbors=nb_neighbors,
                           neighbor_radius=neighbor_radius,
                           max_dist_mult=max_dist_mult)
    return skel.hull_coords, skel.hull_normals
