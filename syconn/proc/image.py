# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Sven Dorkenwald, Philipp Schubert, Joergen Kornfeld
import numpy as np

from ..proc import log_proc

__cv2__ = True
try:
    from cv2 import createCLAHE
    from cv2 import equalizeHist
except ImportError as e:
    print("Could not import cv2.", e)
    __cv2__ = False
    createCLAHE = None
    equalizeHist = None
from sklearn.decomposition import PCA
from scipy import spatial, sparse, ndimage
import tqdm
from typing import List, Optional, Union
# TODO: use mocking (try-except needed for sphinx build)
try:
    import fill_voids
except ImportError:
    pass


def find_contactsite(coords_a, coords_b, max_hull_dist=1):
    """
    This function computes the contact sites between two sets of supervoxels and 
    returns the coordinates of the contact site voxels. It uses a k-dimensional 
    tree for efficient nearest neighbor queries. The function iterates over a 
    range of distances until it finds at least one neighbor.
    
    Args:
        coords_a (np.array): An array of coordinates representing the first set 
        of supervoxels.
        coords_b (np.array): An array of coordinates representing the second set 
        of supervoxels.
        max_hull_dist (int, optional): The maximum distance between voxels in 
        coords_a and coords_b. Defaults to 1.
    
    Returns:
        np.array: An array of coordinates representing the contact site voxels.
    """
    assert max_hull_dist >= 1
    if len(coords_a) == 0 or len(coords_b) == 0:
        return np.zeros((0, 3))
    tree_a = spatial.cKDTree(coords_a)
    tree_b = spatial.cKDTree(coords_b)
    for i in range(1, max_hull_dist + 1):
        contact_ids = tree_a.query_ball_tree(tree_b, i)
        num_neighbours = np.array([len(sublist) for sublist in contact_ids])
        if np.sum(num_neighbours > 0) >= 1:
            break
    contact_coords_a = coords_a[num_neighbours > 0]
    contact_ids_b = set([id for sublist in contact_ids for id in sublist])
    contact_coords_b = coords_b[list(contact_ids_b)]
    if contact_coords_a.ndim == 1:
        contact_coords_a = contact_coords_a[None, :]
    if contact_coords_b.ndim == 1:
        contact_coords_b = contact_coords_a[None, :]
    contact_coords = np.concatenate((contact_coords_a, contact_coords_b), axis=0)
    return np.array(contact_coords).astype(np.int32)


def fast_check_sing_comp(sv, max_dist=5):
    """
    This function performs a fast check to determine if a supervoxel is a single 
    connected component by subsampling. It uses a pairwise distance function to 
    compute the Euclidean distance between all pairs of points in the supervoxel. 
    If the number of connected components is one, the function returns True.
    
    Args:
        sv (np.array): An array of coordinates representing the supervoxel.
        max_dist (int, optional): The maximum distance for two points to be 
        considered connected. Defaults to 5.
    
    Returns:
        bool: True if the supervoxel is a single connected component, False otherwise.
    """
    if len(sv) == 0:
        return True
    pdists = spatial.distance.pdist(sv[::4])
    pdists[pdists > max_dist] = 0
    pdists = sparse.csr_matrix(spatial.distance.squareform(pdists))
    nb_cc, labels = sparse.csgraph.connected_components(pdists)
    return nb_cc == 1


def conn_comp(sv, max_dist):
    """
    This function computes the number of connected components and their labels in a supervoxel. It uses a pairwise 
    distance function to compute the Euclidean distance between all pairs of points in the supervoxel. The function 
    then creates a sparse matrix from the distance matrix and uses a connected components algorithm to find the 
    connected components.
    
    Args:
        sv (np.array): An array of coordinates representing the supervoxel.
        max_dist (int): The maximum distance for two points to be considered connected.
    
    Returns:
        tuple: A tuple containing the number of connected components and an array of labels.
    """
    sv = np.array(sv, dtype=np.float32)
    pdists = spatial.distance.pdist(sv)
    pdists[pdists > max_dist] = 0
    pdists = sparse.csr_matrix(spatial.distance.squareform(pdists))
    nb_cc, labels = sparse.csgraph.connected_components(pdists)
    return nb_cc, labels


def single_conn_comp(sv, max_dist=2, ref_coord=None, return_bool=False):
    """
    This function returns the single connected component of a set of coordinates. It first checks if the supervoxel 
    is a single connected component. If it is, the function returns the supervoxel. If not, the function computes the 
    connected components and their labels. The function then returns the largest connected component or the component 
    closest to a reference coordinate.
    
    Args:
        sv (np.array): An array of coordinates representing the supervoxel.
        max_dist (int, optional): The maximum distance for two points to be considered connected. Defaults to 2.
        ref_coord (np.array, optional): A reference coordinate. Defaults to None.
        return_bool (bool, optional): If True, return a boolean array where True values correspond to the connected 
                                       component. Defaults to False.
    
    Returns:
        np.array: An array of coordinates representing the single connected component.
    """
    # if fast_check_sing_comp(sv):
    #     return sv
    nb_cc, labels = conn_comp(sv, max_dist)
    if ref_coord is None:
        max_comp = np.argmax([np.count_nonzero(labels == i) for i in range(nb_cc)])
        if return_bool:
            return labels == max_comp
        return sv[labels == max_comp]
    else:
        min_dist_ix = np.argmin(np.linalg.norm(sv - ref_coord, axis=1))
        min_label = labels[min_dist_ix]
        if return_bool:
            return labels == min_label
        return sv[labels == min_label]


def single_conn_comp_img(img, background=1.0):
    """
    This function returns the connected component in an image that is located at the center. It first labels the 
    connected components in the image. The function then creates a new image with the same shape as the input image 
    and sets the pixels corresponding to the center component to their original values. TODO: add 'max component' 
    option
    
    Args:
        img (np.array): An array representing the input image.
        background (float, optional): The value representing the background in the image. Defaults to 1.0.
    
    Returns:
        np.array: An array representing the image with only the center connected component.
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
    This function applies histogram normalization on an array if the OpenCV 
    library is available. It first checks if the array is two-dimensional and 
    adds an extra dimension if necessary. The function then normalizes the 
    array to have a maximum value of 255 and applies histogram equalization.
    
    Args:
        arr (np.array): An array representing the input data.
    
    Returns:
        np.array: An array representing the data after histogram normalization.
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
    This function applies the Contrast Limited Adaptive Histogram Equalization (CLAHE) filter on an array if the 
    OpenCV library is available. It first checks if the array is two-dimensional and adds an extra dimension if 
    necessary. The function then normalizes the array to have a maximum value of 255 and applies the CLAHE filter.
    
    Args:
        arr (np.array): An array representing the input data.
        clipLimit (float, optional): The threshold for contrast limiting. Defaults to 4.0.
        tileGridSize (tuple, optional): The size of the grid for the CLAHE algorithm. Defaults to (8, 8).
        ret_normalized (bool, optional): If True, return the normalized array. Defaults to True.
    
    Returns:
        np.array: An array representing the data after applying the CLAHE filter.
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
    This function normalizes an image to have a maximum value specified by the user. It first converts the image to 
    floating point values. The function then subtracts the minimum value of the image and divides by the maximum 
    value. Finally, the function multiplies the image by the maximum value specified by the user.
    
    Args:
        img (np.array): An array representing the input image.
        max_val (int or float, optional): The maximum value for the normalized image. Defaults to 255.
    
    Returns:
        np.array: An array representing the normalized image.
    """
    img = img.astype(np.float32)
    img -= img.min()
    if img.max() != 0:
        img /= img.max()
    return img.astype(np.float32) * max_val


def apply_pca(sv, pca=None):
    """
    This function applies Principal Component Analysis (PCA) to a supervoxel and returns the 
    supervoxel coordinates rotated in the principal component system. If a PCA object is 
    provided, the function uses it to transform the supervoxel. If not, the function creates 
    a new PCA object and fits it to the supervoxel.
    
    Args:
        sv (np.array): An array of coordinates representing the supervoxel.
        pca (PCA, optional): A pre-fitted PCA object. Defaults to None.
    
    Returns:
        np.array: An array of coordinates representing the supervoxel rotated in the principal 
        component system.
    """
    if pca is None:
        pca = PCA(n_components=3, random_state=0)
        sv = pca.fit_transform(sv)
    else:
        sv = pca.transform(sv)
    return sv


def remove_outlier(sv, edge_size):
    """
    Removes outliers from a given array that are beyond the specified edge size. Outliers are defined as
    elements that fall outside the range [0, edge_size).

    Args:
        sv (np.array): The input array from which outliers will be removed.
        edge_size (int): The upper limit for determining outliers in the array.

    Returns:
        np.array: The input array with outliers removed.
    """
    inlier = (sv[:, 0] >= 0) & (sv[:, 0] < edge_size) & (sv[:, 1] >= 0) & \
             (sv[:, 1] < edge_size) & (sv[:, 2] >= 0) & (sv[:, 2] < edge_size)
    nb_outlier = np.sum(~inlier)
    if (float(nb_outlier) / len(sv)) > 0.5:
        log_proc.warn("Found %d/%d outlier after PCA while preprocessing"
                      "supervoexl. Removing %d%% of voxels" % (nb_outlier,
                                                               len(sv), int(float(nb_outlier) / len(sv) * 100)))
    new_sv = sv[inlier]
    assert np.all(np.min(new_sv, axis=0) >= 0), \
        "%s" % np.min(new_sv, axis=0)
    assert np.all(np.max(new_sv, axis=0) < edge_size), \
        "Mins: %s \nMaxs: %s" % (np.min(new_sv, axis=0),
                                 np.max(new_sv, axis=0))
    return new_sv


def normalize_vol(sv, edge_size, center_coord):
    """
    Returns a cube with a given edge size and a supervoxel centered at the center coordinate.

    Args:
        sv (np.array): A 3D array representing the coordinates of voxels in the supervoxel.
        edge_size (int): The edge size of the returned cube.
        center_coord (np.array): The center coordinate for the supervoxel.

    Returns:
        np.array: A cube with the given edge size and the supervoxel centered at the center coordinate.
    """
    translation = np.ones(3) * edge_size / 2. - center_coord
    assert isinstance(edge_size, np.int32)
    sv = sv.astype(np.float32)
    sv = sv + translation  # center
    sv = remove_outlier(sv, edge_size)
    return sv.astype(np.int64)


def multi_dilation(overlay, n_dilations, use_find_objects=False,
                   background_only=True):
    """
    A wrapper function for dilation, which is a morphological operation that expands or thickens objects
    in a binary image.
    
    Args:
        overlay: The input image.
        n_dilations: The number of times dilation is applied.
        use_find_objects (bool, optional): If True, uses 'find_objects' from scipy.ndimage to reduce the
            processed volume and apply the operation per object. Defaults to False.
        background_only (bool, optional): If True, changes only apply to the background (0). Defaults to True.
    
    Returns:
        The dilated image.
    """
    return multi_mop(ndimage.binary_dilation, overlay, n_dilations,
                     use_find_objects, background_only)


def multi_mop(mop_func, overlay, n_iters, use_find_objects=False,
              mop_kwargs=None, verbose=False):
    """
    A generic function for binary morphological image operations with multi-label content. Currently supported
    operations include binary dilation, binary erosion, binary closing, and binary fill holes.
    
    Args:
        mop_func: The morphological operation function to be applied.
        overlay: The input image.
        n_iters: The number of times the morphological operation is applied.
        use_find_objects (bool, optional): If True, uses 'find_objects' from scipy.ndimage to reduce the
            processed volume and apply the operation per object. Defaults to False.
        mop_kwargs (dict, optional): Additional keyword arguments passed to the morphological operation function.
            Defaults to None.
        verbose (bool, optional): If True, displays a progress bar. Defaults to False.
    
    Returns:
        The image after applying the morphological operation.
    """
    if mop_kwargs is None:
        mop_kwargs = {}
    # TODO: Currently mop_kwargs are not generic because of explicit 'iterations' kwarg in mop_func call
    if n_iters == 0:
        return overlay
    if use_find_objects:
        return _multi_mop_findobjects(mop_func, overlay, n_iters, verbose=verbose,
                                      mop_kwargs=mop_kwargs)
    unique_ixs = np.unique(overlay)
    for ix in unique_ixs:
        if ix == 0:
            continue
        binary_mask = (overlay == ix).astype(np.int32)
        # TODO: use padding
        binary_mask = mop_func(binary_mask, iterations=n_iters, **mop_kwargs)
        overlay[binary_mask == 1] = ix
    return overlay


def _multi_mop_findobjects(mop_func, overlay, n_iters, verbose=False,
                           mop_kwargs=None):
    """
    A generic function for binary morphological image operations with multi-label content using 'find_objects'
    from scipy.ndimage to reduce the processed volume and apply the operation per object. Currently supported
    operations include binary dilation, binary erosion, binary closing, and binary fill holes.
    
    Args:
        mop_func: The morphological operation function to be applied.
        overlay: The input image.
        n_iters: The number of times the morphological operation is applied.
        verbose (bool, optional): If True, displays a progress bar. Defaults to False.
        mop_kwargs (dict, optional): Additional keyword arguments passed to the morphological operation function.
            Defaults to None.
    
    Returns:
        The image after applying the morphological operation.
    """
    # TODO: use mask kwarg of morphology operations to ommit subvol copies
    if mop_kwargs is None:
        mop_kwargs = {}
    if 'iterations' in mop_kwargs:
        n_iters = mop_kwargs['iterations']
    # TODO: Currently mop_kwargs are not generic because of explicit 'iterations' kwarg in mop_func call
    objslices = ndimage.find_objects(overlay)
    unique_ixs = np.unique(overlay[overlay != 0])
    if "fill_holes" in mop_func.__name__:
        for ix in unique_ixs:
            sub_vol = overlay[objslices[int(ix - 1)]]
            binary_mask = (sub_vol == ix).astype(np.int32)
            fill_voids.fill(binary_mask, in_place=True)
            proc_mask = sub_vol == 0  # modify only background
            overlay[objslices[int(ix - 1)]][proc_mask] = binary_mask[proc_mask] * ix
        return overlay
    if verbose:
        pbar = tqdm.tqdm(total=len(unique_ixs))
    for ix in unique_ixs:
        if verbose:
            pbar.update(1)
        obj_slice = objslices[int(ix - 1)]
        sub_vol = overlay[obj_slice]
        # pad with zeros to prevent boundary artifacts in the original data array
        if "closing" in mop_func.__name__ or "dilation" in mop_func.__name__:
            sub_vol = np.pad(sub_vol, n_iters)
        binary_mask = (sub_vol == ix).astype(np.int32)
        if verbose:
            nb_occ = np.sum(binary_mask)
        if "fill_holes" in mop_func.__name__:
            res = mop_func(binary_mask, **mop_kwargs)
        else:
            res = mop_func(binary_mask, iterations=n_iters, **mop_kwargs)
        # remove overlap
        if "closing" in mop_func.__name__ or "dilation" in mop_func.__name__:
            res = res[n_iters:-n_iters, n_iters:-n_iters,
                  n_iters:-n_iters]
            binary_mask = binary_mask[n_iters:-n_iters, n_iters:-n_iters,
                          n_iters:-n_iters]
            sub_vol = sub_vol[n_iters:-n_iters, n_iters:-n_iters,
                      n_iters:-n_iters]
        # only dilate/erode background/the objects itself
        if "erosion" in mop_func.__name__ or "binary_opening" in mop_func.__name__:
            if verbose:
                if np.sum(binary_mask) == 0 and nb_occ != 0:
                    log_proc.debug("Object with ID={} and size={} is not present after"
                                   " erosion with N={}.".format(ix, nb_occ, n_iters))
            overlay[obj_slice][binary_mask == 1] = res[binary_mask == 1] * ix
        elif ("dilation" in mop_func.__name__) or ("closing" in mop_func.__name__) or \
                ("fill_holes" in mop_func.__name__):
            proc_mask = (binary_mask == 1) | (sub_vol == 0)  # dilate only background
            overlay[obj_slice][proc_mask] = res[proc_mask] * ix
        else:
            msg = "Only erosion or dilation allowed. Attempted to use morphological " \
                  "operation '{}'.".format(mop_func.__name__)
            log_proc.error(msg)
            raise NotImplementedError(msg)
    if verbose:
        pbar.close()
    return overlay


def multi_dilation_backgroundonly(overlay, n_dilations, mop_kwargs=None):
    """
    Similar to the multi_dilation function, but processes each object in the overlay independently. Changes only
    apply to the background (0), meaning objects will not dilate into other objects.
    
    Args:
        overlay: The input 3D volume of type uint.
        n_dilations: The number of times dilation is applied.
        mop_kwargs (dict, optional): Additional keyword arguments passed to the morphological operation function.
            Defaults to None.
    
    Returns:
        The dilated image.
    """
    return multi_mop_backgroundonly(ndimage.binary_dilation, overlay,
                                    n_dilations, mop_kwargs=mop_kwargs)


def multi_mop_backgroundonly(mop_func, overlay, iterations, mop_kwargs=None):
    """
    Similar to the multi_mop function, but processes each object in the overlay independently. 
    Changes only apply to the background (0), meaning objects will not dilate into other objects. 
    The original regions/segmentation is only affected in case of erosion.
    
    Notes:
        * For ``binary_closing`` it is advised to pass ``structure=np.ones((2, 2, 2))``
          in order to fill gaps at the array boundaries. See https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.ndimage.morphology.binary_closing.html for an example.
    
    Args:
        mop_func: The morphological operation function to be applied. One of ``binary_closing``, 
            ``binary_dilation``, ``binary_erosion``, ``binary_fill_holes`` (see ``scipy.ndimage``).
        overlay: The input image. 3D volume of type uint.
        iterations: The number of times the morphological operation is applied.
        mop_kwargs (dict, optional): Additional keyword arguments passed to the morphological 
            operation function. Defaults to None.
    
    Returns:
        The image after applying the morphological operation.
    """
    return _multi_mop_findobjects(mop_func, overlay, iterations,
                                  mop_kwargs=mop_kwargs)


def apply_morphological_operations(vol: np.ndarray, morph_ops: List[str],
                                   mop_kwargs: Optional[dict] = None) \
        -> np.ndarray:
    """
    Applies a series of morphological operations on the input volume. The operations are specified by a list of string 
    identifiers that must match scipy.ndimage functions. The operations are applied in the order they appear in the list.
    
    Args:
        vol (np.ndarray): The input 3D array on which the morphological operations are to be applied.
        morph_ops (List[str]): A list of string identifiers specifying the morphological operations to be applied.
        mop_kwargs (Optional[dict]): A dictionary of keyword arguments to be passed to the morphological operations.
    
    Returns:
        np.ndarray: The input volume after the specified morphological operations have been applied.
    """
    if len(morph_ops) == 0:
        return vol
    # count zusammenhaengende, gleiche operationen und erhoehe n_iters entsprechend.
    morph_ops, morph_cnt = _count_subsequent_mops(morph_ops)
    for mop, mop_cnt in zip(morph_ops, morph_cnt):
        func = getattr(ndimage, mop)
        vol = _multi_mop_findobjects(func, vol, n_iters=mop_cnt, mop_kwargs=mop_kwargs)
    return vol


def _count_subsequent_mops(mops: List[str]) -> tuple:
    """
    Counts the number of times each morphological operation appears consecutively in the list. The function returns two 
    lists: one with the unique morphological operations and one with the counts of consecutive appearances.
    
    Args:
        mops (List[str]): A list of string identifiers specifying the morphological operations.
    
    Returns:
        tuple: A tuple of two lists. The first list contains the unique morphological operations and the second list 
        contains the counts of consecutive appearances of each operation in the original list.
    """
    mops_new = [mops[0]]
    mops_cnt = [1]
    for m in mops[1:]:
        if m == mops_new[-1]:
            mops_cnt[-1] += 1
        else:
            mops_new.append(m)
            mops_cnt.append(1)
    return mops_new, mops_cnt


def get_aniso_struct(scaling: Union[tuple, np.ndarray]):
    """
    Generates a kernel for morphological operations. The kernel is cross-like with anisotropic dilations in the xy plane. 
    The size of the kernel is determined by the provided scaling factor.
    
    Args:
        scaling (Union[tuple, np.ndarray]): A tuple or array specifying the voxel size in nm.
    
    Returns:
        Kernel: A kernel for morphological operations, taking into account the voxel size.
    """
    struct = np.zeros((5, 5))
    struct[2, 2] = 1
    aniso = scaling[2] // scaling[0]
    assert scaling[1] // scaling[0] == 1
    assert aniso >= 1
    struct2d = ndimage.binary_dilation(struct, iterations=aniso)
    struct = np.concatenate([struct[..., None], struct2d[..., None], struct[..., None]], axis=2)
    return struct
