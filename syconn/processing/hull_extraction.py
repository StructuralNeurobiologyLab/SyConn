import os
from syconn.utils.datahandler import DataHandler, create_skel,\
    get_paths_of_skelID, load_mapped_skeleton


def compute_skeleton_hull(tracing_id, wd, **kwargs):
    """
    :param tracing_id: tracing id
    :type tracing_id: int
    :return hull coordinates, hull normals
    """
    mapped_skel_p = get_paths_of_skelID([tracing_id])[0]
    if os.path.isfile(mapped_skel_p):
        skel = load_mapped_skeleton(mapped_skel_p)
    else:
        dh = DataHandler(wd)
        skel = create_skel(dh, tracing_id, id=tracing_id)
        skel.hull_sampling(**kwargs)
    return skel.hull_coords, skel.hull_normals
