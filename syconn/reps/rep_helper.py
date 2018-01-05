# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max-Planck-Institute for Medical Research, Heidelberg, Germany
# Authors: Sven Dorkenwald, Philipp Schubert, Joergen Kornfeld

import h5py
from shutil import copytree
import numpy as np
import re
from ..handler.basics import read_txt_from_zip
from scipy import spatial
import warnings
from collections import Counter


def parse_cc_dict_from_kml(kml_path):
    """

    Parameters
    ----------
    kml_path : str

    Returns
    -------
    dict
    """
    txt = open(kml_path, "rb").read()
    cc_dict = {}
    for line in txt.splitlines()[::4]:
        line_nb = np.array(re.findall("(\d+)", line), dtype=np.uint)
        curr_ixs = line_nb[3:]
        cc_ix = line_nb[0]
        curr_ixs = curr_ixs[curr_ixs != 0]
        cc_dict[cc_ix] = curr_ixs
    return cc_dict


def parse_cc_dict_from_kzip(k_path):
    """

    Parameters
    ----------
    k_path : str

    Returns
    -------
    dict
    """
    txt = read_txt_from_zip(k_path, "mergelist.txt")
    cc_dict = {}
    for line in txt.splitlines()[::4]:
        line_nb = np.array(re.findall("(\d+)", line), dtype=np.uint)
        curr_ixs = line_nb[3:]
        cc_ix = line_nb[0]
        curr_ixs = curr_ixs[curr_ixs != 0]
        cc_dict[cc_ix] = curr_ixs
    return cc_dict


def knossos_ml_from_svixs(sv_ixs, coords=None, comments=None):
    """

    Parameters
    ----------
    sv_ixs : np.array or list
    coords : np.array or list
    comments : np.array or lost

    Returns
    -------
    str
    """
    # create pseudo mergle list, to fit input types of write_knossos_mergelist
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


def knossos_ml_from_ccs(cc_ixs, ccs, coords=None, comments=None):
    """
    Converts list of connected components (i.e. list of SV IDs) into knossos
    merge list string.

    Parameters
    ----------
    cc_ixs : list of int
    ccs : list of lists of int
    coords : list of np.array
    comments : list of lists of str

    Returns
    -------
    str
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


def knossos_ml_from_sso(sso, comment=None):
    """
    Converts list of connected components (i.e. list of SV IDs) into knossos
    merge list string.

    Parameters
    ----------
    sso : SuperSegmentationObject
        Connected component
    comment : None or list of str

    Returns
    -------
    str
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


def negative_to_zero(a):
    """
    Sets negative values of array a to zero.
    :param a: numpy array
    :return: array a with non negativ values.
    """
    if a > 0:
        return a
    else:
        return 0


def get_rel_path(obj_name, filename, suffix=""):
    """
    Returns path from ChunkDataset folder to SegmentationDataset folder.

    Parameters
    ----------
    obj_name: str
        ie. hdf5name
    filename: str
        Filename of the prediction in the chunkdataset
    suffix: str
        suffix of name

    Returns
    -------
    rel_path: str
        relative path from ChunkDataset folder to UltrastructuralDataset folder

    """
    if len(suffix) > 0 and not suffix[0] == "_":
        suffix = "_" + suffix
    return "/obj_" + obj_name + "_" + \
           filename + suffix + "/"


# def subfold_from_ix(ix, n_folders=0):
#     """
#
#     Parameters
#     ----------
#     ix : int
#
#     Returns
#     -------
#     str
#     """
#     id_str = "00000" + str(ix)
#     subfold = "/%d/%d/%d/" % \
#            (int(id_str[-5:-3]), int(id_str[-3:-1]), int(id_str[-1]))
#     return subfold

def subfold_from_ix(ix, n_folders, old_version=False):
    """

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

    # if order % 2 == 1:
    #     subfold += "%s/" % id_str[-order]

    if old_version:
        subfold = subfold.replace('/0', '/').replace('//', '/0/')

    return subfold


def subfold_from_ix_2nd_stage(ix):
    """

    Parameters
    ----------
    ix : int

    Returns
    -------
    str
    """
    raise ImportError('Check correct functionality before using it')
    # return "/%d/%d/%d/" % ((ix // 1e3) % 1e2, (ix // 1e1) % 1e2, ix % 10)
    id_str = "00000" + str(ix)
    return "/%d/%d/" % (int(id_str[-4:-2]), int(id_str[-2:]))


def ix_from_subfold(subfold):
    """

    Parameters
    ----------
    subfold : str

    Returns
    -------
    int
    """

    parts = subfold.strip("/").split("/")
    return int("".join("%.2d" % part for part in parts))


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


def colorcode_vertices(vertices, rep_coords, rep_values, colors=None,
                       nb_cpus=-1, k=1):
    """

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

    Returns
    -------
    np. array [N, 4]
        rgba values for every vertex from 0 to 255
    """
    if colors is None:
        colors = np.array(np.array([[0.6, 0.6, 0.6, 1], [0.841, 0.138, 0.133, 1.],
                           [0.32, 0.32, 0.32, 1.]]) * 255, dtype=np.uint)
    else:
        colors = np.array(colors, dtype=np.uint)
    assert len(colors) >= np.max(rep_values) + 1
    hull_tree = spatial.cKDTree(rep_coords)
    dists, ixs = hull_tree.query(vertices, n_jobs=nb_cpus, k=k)
    hull_ax = np.zeros((len(vertices)), dtype=np.int)
    for i in range(len(ixs)):
        curr_ax = Counter(np.array(rep_values)[ixs[i]]).most_common(1)[0][0]
        hull_ax[i] = curr_ax
    vert_col = colors[hull_ax]
    return vert_col


def colorcode_vertices_color(vertices, rep_coords, colors,
                       nb_cpus=-1):
    """
    Colorcodes the vertices given a set of coordinates with corresponding
    colors. Separates space in Voronoi-fashion.
    
    Parameters
    ----------
    vertices : np.array
        [N, 3]
    rep_coords : np.array
        [M ,3]
    colors : np.array
        color for each rep_coord
    nb_cpus : int
    k : int
        Number of nearest neighbors (average prediction)

    Returns
    -------
    np. array [N, 4]
        color values for every vertex
    """
    hull_tree = spatial.cKDTree(rep_coords)
    _, ixs = hull_tree.query(vertices, n_jobs=nb_cpus)
    vert_col = colors[ixs]
    return vert_col



def surface_samples(coords, bin_sizes=(2000, 2000, 2000), max_nb_samples=5000,
                    r=1000):
    """
    Sample locations from density grid given by coordinates and bin sizes.
    At each grid center, collects coordinates within r to calculate center of
    mass for sample location.
    
    Parameters
    ----------
    coords : np.array
    bin_sizes : np.array
    max_nb_samples : int
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
    if nb_smaples > max_nb_samples:
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