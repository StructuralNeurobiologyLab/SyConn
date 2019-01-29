# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Philipp Schubert, Joergen Kornfeld
import numpy as np
from numba import jit
from scipy import spatial
from knossos_utils.skeleton_utils import load_skeleton

from ..proc.graphs import bfs_smoothing


def parse_skelnodes_labels_to_mesh(kzip_path, sso, gt_type, n_voting=40):
    """

    Parameters
    ----------
    kzip_path : str
        path to skeleton file with annotated skeleton nodes
    sso : SuperSegmentationObject
        object which corresponds to skeleton in kzip
    gt_type : str
    n_voting : int
        Number of nodes collected during BFS for majority voting
        (smoothing of vertex labels)

    Returns
    -------

    """
    # # Load mesh
    indices, vertices, normals = sso.mesh
    vertices = vertices.reshape((-1, 3))
    # load skeleton
    skel = load_skeleton(kzip_path)
    if len(skel) > 1:
        raise ValueError("Ill-defined skeleton key."
                         "Annotated k.zip contains more than one skeleton. ")
    skel = list(skel.values())[0]
    skel_nodes = list(skel.getNodes())

    node_coords = np.array([n.getCoordinate() * sso.scaling for n in skel_nodes])
    node_labels = np.array([str2intconverter(n.getComment(), gt_type) for n in skel_nodes], dtype=np.int)
    node_coords = node_coords[(node_labels != -1)]
    node_labels = node_labels[(node_labels != -1)]

    # create KD tree from skeleton node coordinates
    tree = spatial.cKDTree(node_coords)
    # transfer labels from skeleton to mesh
    dist, ind = tree.query(vertices, k=1)
    vertex_labels = node_labels[ind]  # retrieving labels of vertices
    vertex_labels = bfs_smoothing(vertices, vertex_labels, n_voting=n_voting)
    return vertex_labels


def generate_palette(nr_classes, return_rgba=True):
    """
    Creates a RGB(A) palette for N classes. Background label will be highest class label + 1.

    Parameters
    ----------
    nr_classes : int
    return_rgba : bool
        If True returned array has shape (N, 4) instead of (N, 3)

    Returns
    -------
    np.array
        Unique color array for N input classes
    """
    classes_ids = np.arange(nr_classes)  # reserve additional class id for background
    classes_rgb = id2rgb_array_contiguous(classes_ids)  # convention: do not use 1, 1, 1; will be background value
    if return_rgba:
        classes_rgb = np.concatenate([classes_rgb, np.ones(classes_rgb.shape[:-1])[..., None] * 255], axis=1)
    return classes_rgb.astype(np.uint8)


@jit
def remap_rgb_labelviews(rgb_view, palette):
    """

    Parameters
    ----------
    rgb_view :
    palette :

    Returns
    -------

    """
    label_view_flat = rgb_view.flatten().reshape((-1, 3))
    background_label = len(palette)
    # convention: Use highest ID as background
    remapped_label_views = np.ones((len(label_view_flat), ), dtype=np.uint16) * background_label
    for kk in range(len(label_view_flat)):
        if np.all(label_view_flat[kk] == 255):  # background
            continue
        for i in range(len(palette)):
            if (label_view_flat[kk, 0] == palette[i, 0]) and \
               (label_view_flat[kk, 1] == palette[i, 1]) and \
               (label_view_flat[kk, 2] == palette[i, 2]):
                remapped_label_views[kk] = i
                break
    return remapped_label_views.reshape(rgb_view.shape[:-1])


# create function that converts information in string type to the information in integer type
def str2intconverter(comment, gt_type):
    if gt_type == "axgt":
        if comment == "gt_axon":
            return 1
        elif comment == "gt_dendrite":
            return 0
        elif comment == "gt_soma":
            return 2
        else:
            return -1
    elif gt_type == "spgt":
        if "head" in comment:
            return 1
        elif "neck" in comment:
            return 0
        elif "shaft" in comment:
            return 2
        elif "other" in comment:
            return 3
        else:
            return -1
    else: raise ValueError("Given groundtruth type is not valid.")


def img_rand_coloring(img):
    if img.ndim == 3 and img.shape[2] > 1:
        raise ValueError("Input image must not contain rgb values")
    colored_img = np.zeros(list(img.shape) + [3], dtype=np.uint8)
    rnd_col_dc = {}
    for ix in np.unique(img):
        rand_col = np.random.randint(0, 256, 3)
        rnd_col_dc[ix] = rand_col
    # set background to white
    rnd_col_dc[np.max(img)] = np.array([255, 255, 255])
    orig_shape = img.shape
    img = img.flatten()
    orig_shape_col = colored_img.shape
    colored_img = colored_img.flatten().reshape(-1, 3)
    for ii in range(len(img)):
        colored_img[ii] = rnd_col_dc[img[ii]]
    colored_img = colored_img.reshape(orig_shape_col)
    return colored_img


def id2rgb(vertex_id):
    """
    Transforms ID value of single sso vertex into the unique RGD colour.

    Parameters
    ----------
    vertex_id : int

    Returns
    -------
    np.array
        RGB values [1, 3]
    """
    red = vertex_id % 256
    green = (vertex_id/256) % 256
    blue = (vertex_id/256/256) % 256
    colour = np.array([red, green, blue], dtype=np.uint8)
    return colour.squeeze()


def id2rgb_array(id_arr):
    """
    Transforms ID values into the array of RGBs labels based on 'idtorgb'.
    Note: Linear retrieval time. For small N preferable.

    Parameters
    ----------
    id_arr : np.array
        ID values [N, 1]

    Returns
    -------
    np.array
        RGB values.squeezed [N, 3]
    """

    if np.max(id_arr) > 256**3:
        raise ValueError("Overflow in vertex ID array.")
    if id_arr.ndim == 1:
        id_arr = id_arr[:, None]
    elif id_arr.ndim == 2:
        assert id_arr.shape[1] == 1, "ValueError: unsupported shape"
    else:
        raise ValueError("Unsupported shape")
    rgb_arr = np.apply_along_axis(id2rgb, 1, id_arr)
    return rgb_arr.squeeze()


@jit
def id2rgb_array_contiguous(id_arr):
    """
    # TODO: Add rgba implementation to render huge cells with shared context in EGL
    Transforms ID values into the array of RGBs labels based on the assumption
    that 'id_arr' is contiguous index array from 0...len(id_arr).
    Same mapping as 'id2rgb_array'.
    Note: Constant retrieval time. For large N preferable.

    Parameters
    ----------
    id_arr : np.array
        ID values [N, 1]

    Returns
    -------
    np.array
        RGB values.squeezed [N, 3]
    """
    if id_arr.squeeze().ndim > 1:
        raise ValueError("Unsupported index array shape.")
    nb_ids = len(id_arr.squeeze())
    if nb_ids >= 256**3:
        raise ValueError("Overflow in vertex ID array.")
    x1 = np.arange(256).astype(np.uint8)
    x2 = np.arange(256).astype(np.uint8)
    x3 = np.arange(256).astype(np.uint8)
    xx1, xx2, xx3 = np.meshgrid(x1, x2, x3, sparse=False, copy=False)
    rgb_arr = np.concatenate([xx3.flatten()[:, None], xx1.flatten()[:, None],
                              xx2.flatten()[:, None]], axis=-1)[:nb_ids]
    return rgb_arr


def rgb2id(rgb):
    """
    Transforms unique RGB values into soo vertex ID.

    Parameters
    ----------
    rgb: np.array
        RGB values [1, 3]

    Returns
    -------
    np.array
        ID values [1, 1]
    """
    red = rgb[0]
    green = rgb[1]
    blue = rgb[2]
    vertex_id = red + green*256 + blue*(256**2)
    return np.array([vertex_id], dtype=np.uint32)


@jit
def rgb2id_array(rgb_arr):
    """
    Transforms RGB values into IDs based on 'rgb2id'.

    Parameters
    ----------
    rgb_arr : np.array
        RGB values [N, 3]

    Returns
    -------
    np.array
        ID values [N, ]
    """
    if rgb_arr.ndim > 1:
        assert rgb_arr.shape[-1] == 3, "ValueError: unsupported shape"
    else:
        raise ValueError("Unsupported shape")
    rgb_arr_flat = rgb_arr.flatten().reshape((-1, 3))
    mask_arr = (rgb_arr_flat[:, 0] == 255) & (rgb_arr_flat[:, 1] == 255) & \
               (rgb_arr_flat[:, 2] == 255)
    id_arr = np.zeros((len(rgb_arr_flat)), dtype=np.uint32)
    for ii in range(len(rgb_arr_flat)):
        if mask_arr[ii]:
            continue
        rgb = rgb_arr_flat[ii]
        id_arr[ii] = rgb[0] + rgb[1]*256 + rgb[2]*(256**2)
    background_ix = np.max(id_arr) + 1  # convention: The highest index value in index view will correspond to the background
    id_arr[mask_arr] = background_ix
    return id_arr.reshape(rgb_arr.shape[:-1])
