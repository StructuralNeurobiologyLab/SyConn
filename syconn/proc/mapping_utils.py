# SyConnFS
# Copyright (c) 2016 Philipp J. Schubert
# All rights reserved

import numpy as np
from scipy import spatial
import warnings
from multiprocessing import Pool
from syconnmp.shared_mem import start_multiprocess
from numba import jit
from ..handler.basics import chunkify


# ------------------------------------------------------------------------------
# Hull mapping -----------------------------------------------------------------

# ------------------------------------------------------------------------------
# Parameters
map_radius = 1200
nb_vox = 500
max_sj_dist = 125.
obj_min_votes = {'mito': 235, 'vc': 210, 'sj': 346} #changed vc value (orig: 191)
obj_min_size = {'mito': 2786, 'vc': 1594, 'sj': 250}
neighbor_radius = 220
nb_voting_neighbors = 100


def map_objects(raw_mesh, skel_coords, obj):
    """
    Map objects of mesh using point cloud method (SyConn). Either path to
    object dataset or dataset itsself has to be given.

    Parameters
    ----------
    raw_mesh : MeshObject
    skel_coords : np.array
        Coordinates of (cell/mesh) skeleton
    obj : SegmentationDataset
        Cell objects

    Returns
    -------
    list
        mapped object IDs
    """
    global local_max_sj_dist
    local_max_sj_dist = max_sj_dist / raw_mesh.max_dist
    vert_resh = np.array(raw_mesh.vert_resh)
    normals_resh = np.array(raw_mesh.normals_resh)

    objtype = obj._type
    mapped_obj_ids = []
    premapped_obj_ids = map_with_kdtree(obj, skel_coords)
    print "%d %s candidates." % (len(premapped_obj_ids), objtype)
    curr_objects = [obj.object_dict[key] for key in premapped_obj_ids]
    for co in curr_objects:
        co._type = objtype
    pool = Pool(processes=4)
    curr_object_voxels = pool.map(get_voxels, curr_objects)
    pool.close()
    pool.join()
    curr_object_voxels = [raw_mesh.transform_external_coords(curr_vx) for
                          curr_vx in curr_object_voxels]
    # separate data in chunks
    chunk_objids = chunkify(premapped_obj_ids, 10)
    chunk_objvoxels = chunkify(curr_object_voxels, 10)
    params = [(vert_resh, normals_resh, ch_objvx, ch_objids, objtype)
              for ch_objvx, ch_objids in zip(chunk_objvoxels, chunk_objids)]
    # process chunks
    chunks = start_multiprocess(mapping_helper, params, nb_cpus=4, debug=False)
    for chunk in chunks:
        for el in chunk:
            curr_obj_id = el[0]
            is_in_hull = el[1]
            if is_in_hull:
                mapped_obj_ids.append(curr_obj_id)
    return mapped_obj_ids


def mapping_helper(args):
    """
    Multiprocessing helper function for mapping objects.

    Parameters
    ----------
    args : tuple of (np.array, np.array, np.array, np.array, str)
        (vertices [N, 3], normals [N, 3], object voxels [M, L], object ids [M,],
        object type

    Returns
    -------
    np.array (bool)
        Objects with IDs in object ids (args[3]) are mapped or not.
    """
    vert_resh = args[0]
    normals_resh = args[1]
    curr_object_voxels = args[2]
    premapped_obj_ids = args[3]
    objtype = args[4]
    sjtrue = (objtype == 'sj')
    min_votes = obj_min_votes[objtype]
    tree = spatial.cKDTree(vert_resh)
    res = []
    for i in range(len(curr_object_voxels)):
        curr_obj_id = premapped_obj_ids[i]
        rand_voxels = curr_object_voxels[i]
        if len(rand_voxels) == 0:
            res.append((curr_obj_id, False))
            continue
        is_in_hull = 0
        if sjtrue:
            n_hullnodes_dists, _ = tree.query(rand_voxels, k=20)
            for ii in range(len(rand_voxels)):
                is_in_hull += check_number_nn(n_hullnodes_dists[ii])
                # no chance of getting over min votes
                if len(rand_voxels)-ii+is_in_hull < min_votes:
                    break
        else:
            _, skel_hull_ixs = tree.query(rand_voxels, k=nb_voting_neighbors)
            for ii in range(len(skel_hull_ixs)):
                vx_near_cellixs = skel_hull_ixs[ii]
                is_in_hull += check_hull_normals(rand_voxels[ii],
                                                 vert_resh[vx_near_cellixs],
                                                 normals_resh[vx_near_cellixs])
                # no chance of getting over min votes
                if len(rand_voxels)-ii+is_in_hull < min_votes:
                    break
        res.append((curr_obj_id, is_in_hull >= min_votes))
    return res


def get_voxels(obj):
    """
    Get hull voxels of cell object.

    Parameters
    ----------
    obj : SegmentationObject

    Returns
    -------
    np.array
        Randomly drawn object voxels of size nb_vox
    """
    try:
        if obj.size < obj_min_size[obj._type]:
            return np.zeros((0, 3))
        if obj.size > 4 * obj_min_size[obj._type]:
            voxels = obj.voxels[::4]
        else:
            voxels = obj.voxels
        rand_ixs = np.arange(len(voxels))
        np.random.shuffle(rand_ixs)
        rand_ixs = rand_ixs[:nb_vox]
        return voxels[rand_ixs]
    except (KeyError, IOError) as e:
        warnings.warn("Object not found during voxel collection.",
                      RuntimeWarning)
        print e
        return np.array([])


@jit
def check_hull_normals(obj_coord, hull_coords, dir_vecs):
    """
    Check if object coordinates are inside hull using normals.

    Parameters
    ----------
    obj_coord : np.array
    hull_coords : np.array
    dir_vecs : np.array

    Returns
    -------
    np.array (bool)
    """
    norm = np.linalg.norm(dir_vecs, axis=1)
    dir_vecs /= norm[:, None]
    obj_coord = obj_coord[None, :]
    left_side = np.inner(obj_coord, dir_vecs)
    right_side = np.sum(dir_vecs * hull_coords, axis=1)
    sign = np.sign(left_side - right_side)
    return np.sum(sign) < 0


@jit
def check_number_nn(n_hullnodes_dists):
    """
    Check if mean of distances is below local_max_sj_dist
    (specified at top).

    Parameters
    ----------
    n_hullnodes_dists : np.array

    Returns
    -------
    bool
    """
    mean_dists = np.mean(n_hullnodes_dists)
    return mean_dists < local_max_sj_dist


def map_with_kdtree(obj, skel_coords, max_radius=map_radius):
    """
    Map objects to skeleton coordinates using a kd-tree. Maximum distance is
    given by map_radius at top.
    Parameters
    ----------
    obj : SegmentationDataset
    skel_coords : np.array
    max_radius: int

    Returns
    -------
    np.array
    IDs of mapped objects
    """
    obj_coords = np.array(obj.rep_coords)
    obj_ids = np.array(obj.ids)
    tree = spatial.cKDTree(obj_coords)
    mapped_ixs = []
    for coord in skel_coords:
        mapped_ixs += list(tree.query_ball_point(coord, max_radius))
    if len(mapped_ixs) == 0:
        return np.zeros((0, ))
    mapped_obj_ids = np.unique(obj_ids[np.array(mapped_ixs)])
    return mapped_obj_ids