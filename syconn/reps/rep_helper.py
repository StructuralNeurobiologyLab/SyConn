# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max-Planck-Institute of Neurobiology, Munich, Germany
# Authors: Philipp Schubert, Joergen Kornfeld

import numpy as np
from scipy import spatial
from collections import Counter
from typing import Tuple, Optional, Union, List, Dict, Any

from ..reps import log_reps
from .. import global_params


def knossos_ml_from_svixs(sv_ixs: Union[np.ndarray, List],
                          coords: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
                          comments: Optional[Union[List[str], np.ndarray]] = None) -> str:
    """
    Generate a KNOSSOS merge list of an array of supervoxels with optional
    coordinates and comments.

    Args:
        sv_ixs: Supervoxel IDs.
        coords: Representative coordinates of each supervoxel (in voxels).
        comments: Comments for each supervoxel.

    Returns:
        A KNOSSOS compatible merge list in string representation.
    """
    txt = ""
    if comments is not None:
        assert len(comments) == len(sv_ixs)
    if coords is None:
        coords = [None] * len(sv_ixs)
    for kk, ix, c in zip(np.arange(len(sv_ixs)), sv_ixs, coords):
        txt += "%d 0 0 " % kk
        txt += "%d " % ix
        if c is None:
            txt += "\n%d %d %d\n\n" % (0, 0, 0)
        else:
            txt += "\n%d %d %d\n\n" % (c[0], c[1], c[2])
        if comments is not None:
            txt += str(comments[kk])
        txt += "\n"
    return txt


def knossos_ml_from_ccs(cc_ixs: Union[List[int], np.ndarray],
                        ccs: List[List[int]],
                        coords: Optional[np.ndarray] = None,
                        comments: Optional[List[str]] = None) -> str:
    """
    Converts list of connected components (i.e. list of SV IDs) into knossos
    merge list string.

    Args:
        cc_ixs: Connected component IDs, i.e. super-supervoxel IDs.
        ccs: Supervoxel IDs for every connected component.
        coords: Coordinates to each connected component (in voxels).
        comments: Comments for each connected component.

    Returns:
        A KNOSSOS compatible merge list in string representation.
    """
    if coords is None:
        coords = [None] * len(cc_ixs)
    assert len(coords) == len(ccs)
    if comments is not None:
        assert len(comments) == len(ccs)
    txt = ""
    for i in range(len(cc_ixs)):
        if len(ccs[i]) == 0:
            continue
        txt += "%d 0 0 " % cc_ixs[i]
        for ix in ccs[i]:
            txt += "%d " % ix
        c = coords[i]
        if c is None:
            c = np.zeros((3))
        txt += "\n%d %d %d\n\n" % (c[0], c[1], c[2])
        if comments is not None:
            txt += str(comments[i])
        txt += "\n"
    return txt


def knossos_ml_from_sso(sso: 'SuperSegmentationObject',
                        comment: Optional[str] = None):
    """
    Converts the sueprvoxels which are part of the `sso` into a KNOSSOS
    compatible merge list string.

    Args:
        sso: :class:`~syconn.reps.super_segmentation_object.SuperSegmentationObject` object.
        comment: Comment.

    Returns:
        A KNOSSOS compatible merge list in string representation.

    """
    cc_ix = sso.id
    txt = "%d 0 0 " % cc_ix
    cc_svixs = sso.attr_dict["sv"]
    for ix in cc_svixs:
        txt += "%d " % ix
    c = sso.rep_coord
    if c is None:
        if sso.svs[0].mesh_exists:
            c = sso.svs[0].mesh[1][:3]
        else:
            c = None
    if c is None:
        c = np.zeros((3))
    txt += "\n%d %d %d\n\n" % (c[0], c[1], c[2])
    if comment is not None:
        txt += str(comment)
    txt += "\n"
    return txt


def subfold_from_ix(ix, n_folders, old_version=False):
    """
    # TODO: remove 'old_version' as soon as possible, currently there is one usage

    Parameters
    ----------
    ix : int
    n_folders: int

    Returns
    -------
    str
    """
    assert n_folders in [10**i for i in range(6)]
    if not global_params.config.use_new_subfold:
        return subfold_from_ix_OLD(ix, n_folders, old_version)
    order = int(np.log10(n_folders))
    subfold = "/"
    div_base = 1e3  # TODO: make this a parameter
    ix = int(ix // div_base % n_folders)  # carve out the middle part
    id_str = '{num:0{w}d}'.format(num=ix, w=order)

    for idx in range(0, order, 2):
        subfold += "%s/" % id_str[idx: idx + 2]

    return subfold


def subfold_from_ix_OLD(ix, n_folders, old_version=False):
    """
    # TODO: remove 'old_version' as soon as possible, currently there is one usage

    Parameters
    ----------
    ix : int
    n_folders: int

    Returns
    -------
    str
    """
    assert n_folders in [10**i for i in range(6)]

    order = int(np.log10(n_folders))

    id_str = "00000" + str(ix)

    subfold = "/"

    for f_order in range(0, order, 2):
        idx = len(id_str) - order + f_order
        subfold += "%s/" % id_str[idx: idx + 2]

    if old_version:
        subfold = subfold.replace('/0', '/').replace('//', '/0/')

    return subfold


def ix_from_subfold(subfold, n_folders):
    """

    Parameters
    ----------
    subfold : str

    Returns
    -------
    int
    """
    if not global_params.config.use_new_subfold:
        return ix_from_subfold_OLD(subfold, n_folders)

    parts = subfold.strip("/").split("/")
    order = int(np.log10(n_folders))
    # TODO: ' + "000"' needs to be adapted if `div_base` is made variable in `subfold_from_ix`
    if order % 2 == 0:
        return int("".join("%.2d" % int(part) for part in parts) + "000")
    else:
        return int("".join("%.2d" % int(part) for part in parts[:-1]) + parts[-1] + "000")


def ix_from_subfold_OLD(subfold, n_folders):
    """

    Parameters
    ----------
    subfold : str

    Returns
    -------
    int
    """

    parts = subfold.strip("/").split("/")

    order = int(np.log10(n_folders))

    if order % 2 == 0:
        return int("".join("%.2d" % int(part) for part in parts))
    else:
        return int("".join("%.2d" % int(part) for part in parts[:-1]) + parts[-1])


def subfold_from_ix_SSO(ix):
    """

    Parameters
    ----------
    ix : int

    Returns
    -------
    str
    """

    # raise NotImplementedError("Outdated")
    return "/%d/%d/%d/" % (ix % 1e2, ix % 1e4, ix)


def get_unique_subfold_ixs(n_folders):
    """
    Returns unique IDs each associated with a unique storage dict

    Parameters
    ----------
    n_folders : int

    Returns
    -------
    np.ndarray
    """
    if global_params.config.use_new_subfold:
        # TODO: this needs to be adapted as soon as `div_base` is changed in `subfold_from_ix`
        storage_location_ids = [int(str(ix) + "000") for ix in np.arange(n_folders)]
    else:
        storage_location_ids = np.arange(n_folders)
    return storage_location_ids


def colorcode_vertices(vertices, rep_coords, rep_values, colors=None,
                       nb_cpus=-1, k=1, return_color=True):
    """
    Assigns all vertices the kNN majority label from rep_coords/rep_values and
    if return_color is True assigns those a color. Helper function to colorcode
    a set of coordinates (vertices) by known labels (rep_coords, rep_values).

    Parameters
    ----------
    vertices : np.array
        [N, 3]
    rep_coords : np.array
        [M ,3]
    rep_values : np.array
        [M, 1] int values to be color coded for each vertex; used as indices
        for colors
    colors : list
        color for each rep_value
    nb_cpus : int
    k : int
        Number of nearest neighbors (average prediction)
    return_color : bool
        If false it returns the majority vote for each index

    Returns
    -------
    np. array [N, 4]
        rgba values for every vertex from 0 to 255
    """
    if colors is None:
        colors = np.array(np.array([[0.6, 0.6, 0.6, 1], [0.841, 0.138, 0.133, 1.],
                           [0.32, 0.32, 0.32, 1.]]) * 255, dtype=np.uint)
    else:
        if np.max(colors) <= 1.0:
            colors = np.array(colors) * 255
        colors = np.array(colors, dtype=np.uint)
        if len(colors) < np.max(rep_values) + 1:
            msg = 'Length of colors has to be equal to "np.max(rep_values)+1"' \
                  '. Note that currently only consecutive labels are supported.'
            log_reps.error(msg)
            raise ValueError(msg)
    hull_tree = spatial.cKDTree(rep_coords)
    if k > len(rep_coords):
        k = rep_coords
    dists, ixs = hull_tree.query(vertices, n_jobs=nb_cpus, k=k)
    hull_rep = np.zeros((len(vertices)), dtype=np.int)
    for i in range(len(ixs)):
        curr_reps = np.array(rep_values)[ixs[i]]
        if np.isscalar(curr_reps):
            curr_reps = np.array([curr_reps])
        curr_maj = Counter(curr_reps).most_common(1)[0][0]
        hull_rep[i] = curr_maj
    if not return_color:
        return hull_rep
    vert_col = colors[hull_rep]
    return vert_col


def assign_rep_values(target_coords, rep_coords, rep_values,
                      nb_cpus=-1, return_ixs=False):
    """
    Assigns values corresponding to representative coordinates to every target
    coordinate.

    Parameters
    ----------
    target_coords : np.array
        [N, 3]
    rep_coords : np.array
        [M ,3]
    rep_values : np.array
        [M, Z] any type of values for each rep_coord.
    nb_cpus : int
    return_ixs : bool
        returns indices of k-closest rep_coord for every target coordinate

    Returns
    -------
    np. array [N, Z]
        representation values for every vertex
    """
    if not type(rep_values) is np.ndarray:
        rep_values = np.array(rep_values)
    if not rep_values.ndim == 2:
        msg = "Number of dimensions of representation values " \
              "have to be exactly 2."
        log_reps.exception(msg)
        raise ValueError(msg)
    hull_tree = spatial.cKDTree(rep_coords)
    dists, ixs = hull_tree.query(target_coords, n_jobs=nb_cpus, k=1)
    hull_rep = np.zeros((len(target_coords), rep_values.shape[1]))
    for i in range(len(ixs)):
        curr_reps = rep_values[ixs[i]]
        hull_rep[i] = curr_reps
    if return_ixs:
        return hull_rep, ixs
    return hull_rep


def surface_samples(coords: np.ndarray,
                    bin_sizes: Tuple[int, int, int] = (2000, 2000, 2000),
                    max_nb_samples: int = 5000,
                    r: int = 1000) -> np.ndarray:
    """'TODO: optimization required -- maybe use simple downsampling instead of histogram
    Sample locations from density grid given by coordinates and bin sizes.
    At each grid center, collects coordinates within the given radius to
    calculate the center of mass which yields the sample location.
    
    Parameters
    ----------
    coords : np.array
    bin_sizes : np.array
    max_nb_samples : int or None
    r : int

    Returns
    -------
    np.array
    """
    offset = np.min(coords, axis=0)
    bin_sizes = np.array(bin_sizes, dtype=np.float)
    coords -= offset
    query_tree = spatial.cKDTree(coords)
    nb_bins = np.ceil(np.max(coords, axis=0) / bin_sizes).astype(np.int)
    nb_bins = np.max([[1, 1, 1], nb_bins], axis=0)
    H, edges = np.histogramdd(coords, bins=nb_bins)
    nb_smaples = np.min([np.sum(H != 0), max_nb_samples])
    if max_nb_samples is not None and nb_smaples > max_nb_samples:  # only use location with highest coordinate density
        thresh_val = np.sort(H.flatten())[::-1][nb_smaples]
        H[H <= thresh_val] = 0
    # get vertices closest to grid bins with density != 0
    max_dens_locs = (np.array(np.where(H != 0)).swapaxes(1, 0) + 0.5)\
                    * bin_sizes
    dists, ixs = query_tree.query(max_dens_locs)
    samples = coords[ixs]
    # get vertices around grid vertex and use mean as new surface sample
    # this point will always ly in the inside of convex shape (i.e. inside the
    # process)
    close_ixs = query_tree.query_ball_point(samples, r=r)
    for i, ixs in enumerate(close_ixs):
        samples[i] = np.mean(coords[ixs], axis=0) + offset
    return samples


def find_object_properties(cube: np.ndarray) -> \
        Tuple[Dict[int, Any], Dict[int, Any], Dict[int, Any]]:
    """
    Interface method for :func:`~_find_object_properties` (python fallback) and
    :func:`~find_object_properties_C` (cython).

    Args:
        cube: The segmentation cube as a 3D array.

    Returns:
        A list of dictionaries with supervoxel IDs as keys: representative
        coordinate pointing to an object voxel, bounding box, size (in voxels).
    """
    try:
        from . find_object_properties_C import find_object_propertiesC
        return find_object_propertiesC(cube)
    except ImportError:
        return _find_object_properties(cube)


def _find_object_properties(cube: np.ndarray) -> \
        Tuple[Dict[int, Any], Dict[int, Any], Dict[int, Any]]:
    """
    Extracts representative coordinate, bounding box and size for each segmentation objects
    within `cube`. Ignores ID=0.
    TODO: find a way to use bincount and find_objects for very large IDs -> use
     ID remapping to make segmentation IDs contiguous

    Args:
        cube:

    Returns:
        A list of dictionaries with supervoxel IDs as keys: representative
        coordinate pointing to an object voxel, bounding box, size (in voxels).

    """
    from scipy.ndimage import find_objects
    mask = cube != 0
    if np.prod(cube.shape) == 0 or np.sum(mask) == 0:
        return {}, {}, {}
    # get sizes
    min_id = np.min(cube[mask]) - 1  # -1 to not set the lowest ID to background
    log_reps.debug("Cube size: {}, min/max ID: {}/{}".format(cube.shape, min_id + 1,
                                                             np.max(cube)))
    cube[mask] = cube[mask] - min_id
    cnts = np.bincount(cube.flatten().astype(np.int64))
    ids = np.nonzero(cnts)[0]
    # get bounding boxes
    res = find_objects(
        cube)  # returns a list of bounding boxes, first entry will correspond to ID=1

    bbs = {}
    rep_coords = {}
    sizes = {}
    for ii in ids:
        if ii == 0:
            continue
        obj_id = int(ii + min_id)
        # sls = res[int(ii-1)]  # Old version which does not scale to huge IDs
        sls = res[int(ii - 1)]
        min_vec = np.array([sl.start for sl in sls],
                           dtype=np.int)  # -1 because bounding boxes start at element with ID 1
        max_vec = np.array([sl.stop for sl in sls], dtype=np.int)
        rand_obj_coord = np.transpose(np.nonzero(cube[sls] == ii))[0]
        bbs[obj_id] = np.array([min_vec, max_vec], dtype=np.int)
        sizes[obj_id] = cnts[ii]
        rep_coords[
            obj_id] = min_vec + rand_obj_coord
    return rep_coords, bbs, sizes
