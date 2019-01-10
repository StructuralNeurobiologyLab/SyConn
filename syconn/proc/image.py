# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Sven Dorkenwald, Philipp Schubert, Joergen Kornfeld

import numpy as np
__cv2__ = True
try:
    from cv2 import createCLAHE
    from cv2 import equalizeHist
except ImportError as e:
    print("Could not import cv2.", e)
    __cv2__ = False
    createCLAHE = None
    equalizeHist = None
from scipy import spatial, sparse, ndimage
from sklearn.decomposition import PCA
import tqdm

from ..proc import log_proc


def find_contactsite(coords_a, coords_b, max_hull_dist=1):
    """
    Computes contact sites between supver voxels and returns contact site voxel

    Parameters
    ----------
    coords_a : np.array
    coords_b : np.array
    max_hull_dist : int
        Maximum distance between voxels in coods_a and coords_b

    Returns
    -------
    np.array
        contact site coordinates
    """
    assert max_hull_dist >= 1
    if len(coords_a) == 0 or len(coords_b) == 0:
        return np.zeros((0, 3))
    tree_a = spatial.cKDTree(coords_a)
    tree_b = spatial.cKDTree(coords_b)
    for i in range(1, max_hull_dist+1):
        contact_ids = tree_a.query_ball_tree(tree_b, i)
        num_neighbours = np.array([len(sublist) for sublist in contact_ids])
        if np.sum(num_neighbours>0) >= 1:
            break
    contact_coords_a = coords_a[num_neighbours>0]
    contact_ids_b = set([id for sublist in contact_ids for id in sublist])
    contact_coords_b = coords_b[list(contact_ids_b)]
    if contact_coords_a.ndim == 1:
        contact_coords_a = contact_coords_a[None, :]
    if contact_coords_b.ndim == 1:
        contact_coords_b = contact_coords_a[None, :]
    contact_coords = np.concatenate((contact_coords_a, contact_coords_b), axis=0)
    return np.array(contact_coords).astype(np.uint)


def fast_check_sing_comp(sv, max_dist=5):
    """
    Fast check if super voxel is single connected component by subsampling
    Parameters
    ----------
    sv : np.array
    max_dist : int

    Returns
    -------
    bool
        True if single connected component
    """
    if len(sv) == 0:
        return True
    pdists = spatial.distance.pdist(sv[::4])
    pdists[pdists > max_dist] = 0
    pdists = sparse.csr_matrix(spatial.distance.squareform(pdists))
    nb_cc, labels = sparse.csgraph.connected_components(pdists)
    return nb_cc == 1


def conn_comp(sv, max_dist):
    sv = np.array(sv, dtype=np.float32)
    pdists = spatial.distance.pdist(sv)
    pdists[pdists > max_dist] = 0
    pdists = sparse.csr_matrix(spatial.distance.squareform(pdists))
    nb_cc, labels = sparse.csgraph.connected_components(pdists)
    return nb_cc, labels


def single_conn_comp(sv, max_dist=2, ref_coord=None, return_bool=False):
    """
    Returns single connected component of coordinates.
    Parameters
    ----------
    sv : np.array
    max_dist : int
    ref_coord : np.array
    return_bool : bool

    Returns
    -------
    np.array
    """
    # if fast_check_sing_comp(sv):
    #     return sv
    nb_cc, labels = conn_comp(sv, max_dist)
    if ref_coord is None:
        max_comp = np.argmax([np.count_nonzero(labels==i) for i in range(nb_cc)])
        if return_bool:
            return labels == max_comp
        return sv[labels == max_comp]
    else:
        min_dist_ix = np.argmin(np.linalg.norm(sv-ref_coord, axis=1))
        min_label = labels[min_dist_ix]
        if return_bool:
            return labels == min_label
        return sv[labels == min_label]


def single_conn_comp_img(img, background=1.0):
    """
    Returns connected component in image which is located at the center.
    TODO: add 'max component' option

    Parameters
    ----------
    img : np.array
    background : float

    Returns
    -------
    np.array
    """
    orig_shape = img.shape
    img = np.squeeze(img)
    labeled, nr_objects = ndimage.label(img != background)
    new_img = np.ones_like(img) * background
    center = np.array(img.shape) / 2
    ixs = [labeled == labeled[tuple(center)]]
    new_img[ixs] = img[ixs]
    return new_img.reshape(orig_shape)


def rgb2gray(rgb):
    if isinstance(rgb, list):
        rgb = np.array(rgb)
    rgb = normalize_img(rgb, max_val=1).astype(np.float32)
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


def apply_equalhist(arr):
    """
    If cv2 is available applies histogram normalization on array.

    Parameters
    ----------
    arr : np.array


    Returns
    -------

    """
    if not __cv2__:
        try:
            import cv2
        except ImportError as e:
            raise ImportError("cv2 not properly installed:\n %s" % str(e))
    if arr.shape[-1] != 1:
        arr = arr[..., None]
    if arr.dtype != np.uint8:
        arr = normalize_img(arr, max_val=255).astype(np.uint8)
    return normalize_img(equalizeHist(arr), max_val=1)


def apply_clahe(arr, clipLimit=4.0, tileGridSize=(8, 8), ret_normalized=True):
    """
    If cv2 is available applies clahe filter on array.

    Parameters
    ----------
    arr : np.array
    clipLimit : float
    tileGridSize : tuple of int
    ret_normalized : bool

    Returns
    -------
    np.array
    """
    if not __cv2__:
        try:
            import cv2
        except ImportError as e:
            raise ImportError("cv2 not properly installed:\n %s" % str(e))
    if arr.ndim == 2:
        arr = arr[..., None]
    if 0 < np.max(arr) <= 1:
        arr = normalize_img(arr, max_val=255)
    if arr.dtype.kind not in ('u', 'i'):
        arr = arr.astype(np.uint8)
    arr = apply_clahe_plain(arr, clipLimit, tileGridSize)
    # arr = equalize_adapthist(arr, clip_limit=clipLimit)
    if not ret_normalized:
        return normalize_img(arr, max_val=255).astype(np.uint8)
    return arr


def apply_clahe_plain(arr, clipLimit, tileGridSize):
    clahe = createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    return clahe.apply(arr)


def normalize_img(img, max_val=255):
    """
    Parameters
    ----------
    img : np.array
    max_val : int or float

    Returns
    -------
    np.array
        Normalized image
    """
    img = img.astype(np.float32)
    img -= img.min()
    if img.max() != 0:
        img /= img.max()
    return img.astype(np.float32) * max_val


def apply_pca(sv, pca=None):
    """
    Apply principal component analysis and return rotated supervoxel

    Parameters
    ----------
    sv : np.array [N x 3]
        super voxel
    pca : PCA
        prefitted pca

    Returns
    -------
        super voxel coordinates rotated in principle component system
    """
    if pca is None:
        pca = PCA(n_components=3, random_state=0)
        sv = pca.fit_transform(sv)
    else:
        sv = pca.transform(sv)
    return sv


def get_pc(sv):
    """
    Calculates principle components and return eigenvectors of
    covariance matrix.

    Parameters
    ----------
    sv : np.array [N x 3]
        coordinates of voxels in super voxel

    Returns
    -------
    np.array [3, 3]
        eigenvectors of input data
    """
    assert sv.ndim == 2, "Insert array of N x 3"
    assert sv.shape[1] == 3, "Insert array of N x 3"
    cov_mat = np.cov([sv[:, 0], sv[:, 1], sv[:, 2]])
    eig_val_cov, eig_vec_cov = np.linalg.eig(cov_mat)
    return eig_vec_cov


def remove_outlier(sv, edge_size):
    """
    Removes outlier in array sv beyond [0, edge_sizes]

    Parameters
    ----------
    sv : np.array
    edge_size : int

    Returns
    -------
    np.array
    """
    inlier = (sv[:, 0] >= 0) & (sv[:, 0] < edge_size) & (sv[:, 1] >= 0) & \
              (sv[:, 1] < edge_size) & (sv[:, 2] >= 0) & (sv[:, 2] < edge_size)
    nb_outlier = np.sum(~inlier)
    if (float(nb_outlier) / len(sv)) > 0.5:
        log_proc.warn("Found %d/%d outlier after PCA while preprocessing"
                      "supervoexl. Removing %d%% of voxels" % (nb_outlier,
                      len(sv), int(float(nb_outlier)/len(sv)*100)))
    new_sv = sv[inlier]
    assert np.all(np.min(new_sv, axis=0) >= 0), \
        "%s" % np.min(new_sv, axis=0)
    assert np.all(np.max(new_sv, axis=0) < edge_size),\
        "Mins: %s \nMaxs: %s" % (np.min(new_sv, axis=0),
                                 np.max(new_sv, axis=0))
    return new_sv


def normalize_vol(sv, edge_size, center_coord):
    """
    returns cube with given edge size and sv centered at center coordinate

    Paraemters
    ----------
    sv :  np.array [N x 3]
        coordinates of voxels in supervoxel
    edge_size : int
        edge size of returned cube
    center_coord : np.array

    Returns
    -------
        np.array
    """
    translation = np.ones(3) * edge_size / 2. - center_coord
    assert isinstance(edge_size, np.int)
    sv = sv.astype(np.float32)
    sv = sv + translation    # centralize
    sv = remove_outlier(sv, edge_size)
    return sv.astype(np.uint)


def multi_dilation(overlay, n_dilations, use_find_objects=False,
                   background_only=True):
    """
    Wrapper function for dilation

    Parameters
    ----------
    overlay
    n_dilations
    use_find_objects
    background_only

    Returns
    -------

    """
    return multi_mop(ndimage.binary_dilation, overlay, n_dilations,
                     use_find_objects, background_only)


def multi_mop(mop_func, overlay, n_iters, use_find_objects=False,
              background_only=True, mop_kwargs=None, verbose=False):
    """
    Generic function for binary morphological image operations with multi-label
     content.

    Parameters
    ----------
    mop_func
    overlay
    n_iters
    use_find_objects
    background_only : bool
        only works in combination with 'use_find_objects'
    mop_kwargs
    verbose

    Returns
    -------

    """
    if mop_kwargs is None:
        mop_kwargs = {}
    # TODO: Currently mop_kwargs are not generic because of explicit 'iterations' kwarg in mop_func call
    if n_iters == 0:
        return overlay
    if use_find_objects:
        return _multi_mop_findobjects(mop_func, overlay, n_iters, background_only,
                                      mop_kwargs=mop_kwargs, verbose=verbose)
    unique_ixs = np.unique(overlay)
    for ix in unique_ixs:
        if ix == 0:
            continue
        binary_mask = (overlay == ix).astype(np.int)
        binary_mask = mop_func(binary_mask, iterations=n_iters, **mop_kwargs)
        overlay[binary_mask == 1] = ix
    return overlay


def _multi_mop_findobjects(mop_func, overlay, n_iters, background_only=True,
                           verbose=False, mop_kwargs=None):
    """
    Generic function for binary morphological image operations with multi-label content
    using 'find_objects' from scipy.ndimage to reduce processed volume.

    Parameters
    ----------
    mop_func : func
        e.g. binary_dilation, binary_erosion etc
    overlay
    n_iters
    background_only
    verbose
    mop_kwargs

    Returns
    -------

    """
    if mop_kwargs is None:
        mop_kwargs = {}
    # TODO: Currently mop_kwargs are not generic because of explicit 'iterations' kwarg in mop_func call
    objslices = ndimage.find_objects(overlay)
    unique_ixs = np.unique(overlay[overlay != 0])
    if verbose:
        pbar = tqdm.tqdm(total=len(unique_ixs))
    for ix in unique_ixs:
        if verbose:
            pbar.update(1)
        obj_slice = objslices[ix-1]
        new_obj_slices = []
        for sl in obj_slice:
            new_start = sl.start - n_iters if sl.start >= 0 + n_iters else sl.start
            new_end = sl.stop + n_iters
            new_obj_slices.append(slice(new_start, new_end, None))
        sub_vol = overlay[new_obj_slices]
        binary_mask = (sub_vol == ix).astype(np.int)
        if verbose:
            nb_occ = np.sum(binary_mask)
        res = mop_func(binary_mask, iterations=n_iters, **mop_kwargs)
        if verbose:
            if np.sum(binary_mask) == 0 and nb_occ != 0:
                log_proc.debug("Object with ID={} and size={} is not present after"
                               " erosion with N={}.".format(ix, nb_occ, n_iters))
        # only dilate/erode background/the objects itself
        if "erosion" in mop_func.__name__:
            overlay[new_obj_slices][binary_mask == 1] = res[binary_mask == 1] * ix
        elif "dilation" in mop_func.__name__:
            proc_mask = (binary_mask == 1) | (sub_vol == 0)  # dilate only background
            overlay[new_obj_slices][proc_mask] = res[proc_mask] * ix
        else:
            msg = "Only erosion or dilation allowed. Attempted to use morphological " \
                  "operation '{}'.".format(mop_func.__name__)
            log_proc.error(msg)
            raise NotImplementedError(msg)
    if verbose:
        pbar.close()
    return overlay


def _multi_dilation_findobjects(overlay, n_dilations, background_only,
                                verbose=False):
    """
    Wrapper function for dilation

    Parameters
    ----------
    overlay
    n_dilations
    background_only
    verbose

    Returns
    -------

    """
    return _multi_mop_findobjects(ndimage.binary_dilation, overlay, n_dilations,
                                  background_only, verbose)


def multi_dilation_backgroundonly(overlay, n_dilations):
    """
    Same as 'multi_dilation' but only dilates regions into global background

    Parameters
    ----------
    overlay
    n_dilations

    Returns
    -------

    """
    return multi_mop_backgroundonly(ndimage.binary_dilation, overlay, n_dilations)


def multi_mop_backgroundonly(mop_func, overlay, n_dilations):
    """
    Same as 'multi_mop' but only dilates regions into global background

    Parameters
    ----------
    mop_func
    overlay
    n_dilations

    Returns
    -------

    """
    return _multi_mop_findobjects(mop_func, overlay, n_dilations, background_only=True)
