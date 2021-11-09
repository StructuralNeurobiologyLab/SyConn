# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Sven Dorkenwald, Philipp Schubert, Joergen Kornfeld

import itertools
import copy
from collections import Counter
import time
from typing import Optional, List, Tuple, Dict, Union, Iterable, TYPE_CHECKING, Iterator

import numpy as np
import tqdm
from numba import jit
from plyfile import PlyData, PlyElement
from scipy import spatial
from scipy.ndimage import zoom
from scipy.ndimage.morphology import binary_erosion
from sklearn.decomposition import PCA
from zmesh import Mesher
try:
    from vigra.filters import gaussianGradient
except ImportError:
    pass  # for sphinx build
try:
    import open3d as o3d
except ImportError:
    pass  # for sphinx build

from .image import apply_pca
from .. import global_params
from ..backend.storage import AttributeDict, MeshStorage, VoxelStorage, VoxelStorageDyn, VoxelStorageLazyLoading
from ..handler.basics import write_data2kzip, data2kzip
from ..mp.mp_utils import start_multiprocess_obj, start_multiprocess_imap
from ..proc import log_proc
from ..reps.segmentation_helper import load_so_meshes_bulk
from syconn.extraction.in_bounding_boxC import in_bounding_box

from skimage.measure import mesh_surface_area

try:
    # set matplotlib backend to offscreen
    import matplotlib
    matplotlib.use('agg')
    from vigra.filters import boundaryDistanceTransform, gaussianSmoothing
except ImportError as e:
    boundaryDistanceTransform, gaussianSmoothing = None, None
    log_proc.error('ImportError. Could not import VIGRA. '
                   'Mesh generation will not be possible. {}'.format(e))
try:
    import openmesh
except ImportError as e:
    log_proc.error('ImportError. Could not import openmesh. '
                   'Writing meshes as `.obj` files will not be'
                   ' possible. {}'.format(e))
if TYPE_CHECKING:
    from ..reps import segmentation
    from ..reps import super_segmentation_object

__all__ = ['MeshObject', 'get_object_mesh', 'merge_meshes', 'calc_contact_syn_mesh',
           'get_random_centered_coords', 'write_mesh2kzip', 'write_meshes2kzip', 'gen_mesh_voxelmask',
           'compartmentalize_mesh', 'mesh_chunk', 'mesh_creator_sso', 'merge_meshes_incl_norm',
           'mesh_area_calc', 'mesh2obj_file', 'calc_rot_matrices', 'merge_someshes', 'find_meshes',
           ]


class MeshObject(object):
    def __init__(self, object_type, indices, vertices, normals=None,
                 color=None, bounding_box=None):
        self.object_type = object_type
        if vertices.ndim == 2 and vertices.shape[1] == 3:
            self.vertices = vertices.flatten()
        else:
            # assume flat array
            self.vertices = np.array(vertices, dtype=np.float32)
        if indices.ndim == 2 and indices.shape[1] == 3:
            self.indices = indices.flatten().astype(np.uint64)
        else:
            # assume flat array
            self.indices = np.array(indices, dtype=np.uint64)
        if len(self.vertices) == 0:
            self.center = 0
            self.max_dist = 1
            self._normals = np.zeros((0, 3))
            return
        if bounding_box is None:
            self.center, self.max_dist = get_bounding_box(self.vertices)
        else:
            self.center = bounding_box[0]
            self.max_dist = bounding_box[1]
        self.center = self.center.astype(np.float32)
        self.max_dist = self.max_dist.astype(np.float32)
        vert_resh = np.array(self.vertices).reshape((len(self.vertices) // 3, 3))
        vert_resh -= np.array(self.center, dtype=self.vertices.dtype)
        vert_resh = vert_resh / np.array(self.max_dist)
        self.vertices = vert_resh.reshape(len(self.vertices))
        if normals is not None and len(normals) == 0:
            normals = None
        if normals is not None and normals.ndim == 2:
            normals = normals.reshape(len(normals) * 3)
        self._normals = normals
        self._ext_color = color
        self._colors = None
        self.pca = None

    @property
    def colors(self):
        if self._ext_color is None:
            self._colors = np.ones(len(self.vertices) // 3 * 4) * 0.5
        elif np.isscalar(self._ext_color):
            self._colors = np.array(len(self.vertices) // 3 * [self._ext_color]).flatten()
        else:
            if np.ndim(self._ext_color) >= 2:
                self._ext_color = self._ext_color.squeeze()
                assert self._ext_color.shape[1] == 4, \
                    "'color' parameter has wrong shape"
                self._ext_color = self._ext_color.squeeze()
                assert self._ext_color.shape[1] == 4, \
                    "Rendering requires RGBA 'color' shape of (X, 4). Please" \
                    "add alpha channel."
                self._ext_color = self._ext_color.flatten()
            assert len(self._ext_color) / 4 == len(self.vertices) / 3, \
                "len(ext_color)/4 must be equal to len(vertices)/3."
            self._colors = self._ext_color
        return self._colors

    @property
    def vert_resh(self):
        vert_resh = np.array(self.vertices).reshape(-1, 3)
        return vert_resh

    @property
    def normals(self):
        if self._normals is None or len(self._normals) != len(self.vertices):
            log_proc.warning("Calculating normals")
            self._normals = unit_normal(self.vertices, self.indices)
        elif len(self._normals) != len(self.vertices):
            log_proc.debug("Calculating normals, because their shape differs from"
                           " vertices: %s (normals) vs. %s (vertices)" %
                           (str(self._normals.shape), str(self.vertices.shape)))
            self._normals = unit_normal(self.vertices, self.indices)
        return self._normals

    @property
    def normals_resh(self):
        return self.normals.reshape(-1, 3)

    def transform_external_coords(self, coords):
        """

        Args:
            coords: np.array

        Returns: np.array
            transformed coordinates

        """
        if len(coords) == 0:
            return coords
        coords = np.array(coords, dtype=np.float32)
        coords = coords - self.center
        coords /= self.max_dist
        return coords

    def retransform_external_coords(self, coords):
        coords = np.array(coords, dtype=np.float32)
        coords *= self.max_dist
        coords += self.center
        return coords.astype(np.int32)

    @property
    def bounding_box(self):
        return [self.center, self.max_dist]

    def perform_pca_rotation(self):
        """
        Rotates vertices into principal component coordinate system.
        """
        if self.pca is None:
            self.pca = PCA(n_components=3, whiten=False, random_state=0)
            self.pca.fit(self.vert_resh)
        self.vertices = self.pca.transform(
            self.vert_resh).reshape(len(self.vertices))

    def renormalize_vertices(self, bounding_box=None):
        """
        Renomralize, i.e. substract mean and divide by max. extent, vertices
        using either center and max. distance from self.vertices or given from
        keyword argument bounding_box.

        Args:
            bounding_box: tuple
                center, scale (applied as follows: self.vert_resh / scale)

        Returns:

        """
        if bounding_box is None:
            bounding_box = get_bounding_box(self.vertices)
        self.center, self.max_dist = bounding_box
        self.center = self.center.astype(np.float32)
        self.max_dist = self.max_dist.astype(np.float32)
        vert_resh = np.array(self.vertices).reshape(len(self.vertices) // 3, 3)
        vert_resh -= self.center
        vert_resh /= self.max_dist
        self.vertices = vert_resh.reshape(len(self.vertices))

    @property
    def vertices_scaled(self):
        return (self.vert_resh * self.max_dist + self.center).flatten()


def normalize_vertices(vertices: np.ndarray) -> np.ndarray:
    """
    Rotate, center and normalize vertices.

    Args:
        vertices: np.array
            [N, 1]

    Returns: array
        transformed vertices

    """
    vert_resh = vertices.reshape(len(vertices) // 3, 3)
    vert_resh = apply_pca(vert_resh)
    vert_resh -= np.median(vert_resh, axis=0)
    max_val = np.abs(vert_resh).max()
    vert_resh = vert_resh / max_val
    vertices = vert_resh.reshape(len(vertices)).astype(np.float32)
    return vertices


def calc_rot_matrices(coords: np.ndarray, vertices: np.ndarray, edge_length: Union[float, int],
                      nb_cpus: int = 1) -> np.ndarray:
    """
    # Optimization comment: bottleneck is now 'get_rotmatrix_from_points'

    Fits a PCA to local sub-volumes in order to rotate them according to
    its main process (e.g. x-axis will be parallel to the long axis of a tube)

    Args:
        coords: Center coordinates [M x 3]
        vertices: Vertices [N x 3]
        edge_length: Spatial extent of box used to querying vertices for the PCA fit (used for the view alignment).
        nb_cpus: Number of CPUs.

    Returns: Flattened OpenGL rotation matrix (Fortran ordering).

    """
    if not np.isscalar(edge_length):
        log_proc.warning('"calc_rot_matrices" now takes only scalar edgelengths'
                         '. Choosing np.min(edge_length) as query box edge'
                         ' length.')
        edge_length = np.min(edge_length)
    if len(vertices) > 1e5:
        vertices = vertices[::8]
    vertices = vertices.astype(np.float32)
    params = [(coords_ch, vertices, edge_length) for coords_ch in
              np.array_split(coords, nb_cpus, axis=0)]
    res = start_multiprocess_imap(calc_rot_matrices_helper, params,
                                  nb_cpus=nb_cpus, show_progress=False)
    rot_matrices = np.concatenate(res)
    return rot_matrices


def calc_rot_matrices_helper(args):
    """
    Fits a PCA to local sub-volumes in order to rotate them according to
    its main process (e.g. x-axis will be parallel to the long axis of a tube)

    Args:
        args: np.array [M x 3], np.array [N x 3], float/int
            coords, vertices, edge_length = spatial extent of box for querying vertices for pca fit

    Returns: np.array [M x 16]
        Fortran flattened OpenGL rotation matrix

    """
    coords, vertices, edge_length = args
    rot_matrices = np.zeros((len(coords), 16))
    edge_lengths = np.array([edge_length] * 3)
    vertices = vertices.astype(np.float32)
    for ii, c in enumerate(coords):
        bounding_box = np.array([c, edge_lengths], dtype=np.float32)
        inlier = np.array(vertices[in_bounding_box(vertices, bounding_box)])
        rot_matrices[ii] = get_rotmatrix_from_points(inlier)
    return rot_matrices


def get_rotmatrix_from_points(points: np.ndarray) -> np.ndarray:
    """
    Fits pca to input points and returns corresponding rotation matrix, usable
    in PyOpenGL.

    Args:
        points: Vertices/points used in PCA.

    Returns:
        Flat (Fortrain ordering) rotation matrix as returned by PCA with 3 components [4, 4].
    """
    if len(points) <= 2:
        return np.zeros(16)
    new_center = np.mean(points, axis=0)
    points -= new_center
    rot_mat = np.zeros((4, 4))
    rot_mat[:3, :3] = _calc_pca_components(points)
    rot_mat[3, 3] = 1
    rot_mat = rot_mat.flatten('F')
    return rot_mat


def _calc_pca_components(pts: np.ndarray) -> np.ndarray:
    """
    Retrieve Eigenvalue sorted Eigenvectors from input array.

    Args:
        pts: Input points.

    Returns:
        Eigenvalue sorted Eigenvectors.
    """
    cov = np.cov(pts, rowvar=False)
    evals, evecs = np.linalg.eig(cov)
    idx = np.argsort(evals)[::-1]
    evecs = evecs[:, idx].transpose()
    return evecs


def flag_empty_spaces(coords: np.ndarray, vertices: np.ndarray,
                      edge_length: Union[float, int, np.ndarray]) -> np.ndarray:
    """
    Flag empty locations.

    Args:
        coords: [M x 3]
        vertices: [N x 3]
        edge_length: Spatial extent of bounding box to look for vertex support.

    Returns:
        Bool array [M x 1]

    """
    if not np.isscalar(edge_length):
        log_proc.warning('"flag_empty_spaces" now takes only scalar edgelengths'
                         '. Choosing np.min(edge_length) as query box edge'
                         ' length.')
        edge_length = np.min(edge_length)
    if len(vertices) > 1e6:
        vertices = vertices[::8]
    empty_spaces = np.zeros((len(coords))).astype(np.bool)
    edge_lengths = np.array([edge_length] * 3)
    for ii, c in enumerate(coords):
        bounding_box = (c, edge_lengths)
        inlier = np.array(vertices[in_bounding_box(vertices, bounding_box)])
        if len(inlier) == 0:
            empty_spaces[ii] = True
    return empty_spaces


def get_bounding_box(coordinates: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Calculates center of coordinates and its maximum distance in any spatial
    dimension to the most distant point.

    Args:
        coordinates: Coordinates.

    Returns:
        Centers, maximum distance.

    """
    if coordinates.ndim == 2 and coordinates.shape[1] == 3:
        coord_resh = coordinates
    else:
        coord_resh = coordinates.reshape(len(coordinates) // 3, 3)
    mean = np.mean(coord_resh, axis=0)
    max_dist = np.max(np.abs(coord_resh - mean))
    return mean, max_dist


@jit
def get_avg_normal(normals, indices, nbvert):
    normals_avg = np.zeros((nbvert, 3), np.float32)
    for n in range(len(indices)):
        ix = indices[n]
        normals_avg[ix] += normals[n]
    return normals_avg


def unit_normal(vertices: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """
    Calculates normals per face (averaging corresponding vertex normals) and
    expands it to (averaged) normals per vertex.

    Args:
        vertices:
            Flattend vertices [N x 1].
        indices:
            Flattend indices [M x 1].

    Returns:
        Unit face normals per vertex [N x 1].

    """
    vertices = np.array(vertices, dtype=np.float32)
    nbvert = len(vertices) // 3
    # get coordinate list
    vert_lst = vertices.reshape(nbvert, 3)[indices]
    # get traingles from coordinates
    triangles = vert_lst.reshape(len(vert_lst) // 3, 3, 3)
    # calculate normals of triangles
    v = triangles[:, 1] - triangles[:, 0]
    w = triangles[:, 2] - triangles[:, 0]
    normals = np.cross(v, w)
    norm = np.linalg.norm(normals, axis=1)
    normals[norm != 0, :] = normals[norm != 0, :] / norm[norm != 0, None]
    # repeat normal, s.t. len(normals) == len(vertices), i.e. every vertex nows
    # its normal (multiple normals because one vertex is part of multiple triangles
    normals = np.array(list(itertools.chain.from_iterable(itertools.repeat(x, 3) for x in normals)))
    # average normal for every vertex
    normals_avg = get_avg_normal(normals, indices, nbvert)
    return -normals_avg.astype(np.float32).reshape(nbvert * 3)


def get_random_centered_coords(pts, nb, r):
    """

    Args:
        pts: np.array
            coordinates
        nb: int
            number of center of masses to be returned
        r: int
            radius of query_ball_point in order to get list of points for
            center of mass

    Returns: np.array
        coordinates of randomly located center of masses in pts

    """
    tree = spatial.cKDTree(pts)
    rand_ixs = np.random.randint(0, len(pts), nb)
    close_ixs = tree.query_ball_point(pts[rand_ixs], r)
    coms = np.zeros((nb, 3))
    for i, ixs in enumerate(close_ixs):
        coms[i] = np.mean(pts[ixs], axis=0)
    return coms


def merge_meshes(ind_lst, vert_lst, nb_simplices=3):
    """
    Combine several meshes into a single one.

    Args:
        ind_lst: list of np.array [N, 1]
        vert_lst: list of np.array [N, 1]
        nb_simplices: int
            Number of simplices, e.g. for triangles nb_simplices=3

    Returns: np.array, np.array

    """
    assert len(vert_lst) == len(ind_lst), "Length of indices list differs" \
                                          "from vertices list."
    if len(vert_lst) == 0:
        return [np.zeros((0,), dtype=np.uint64), np.zeros((0,)), np.zeros((0,))]
    else:
        all_vert = np.concatenate(vert_lst)
    # store index and vertex offset of every partial mesh
    vert_offset = np.cumsum([0, ] + [len(verts) // nb_simplices for verts in vert_lst]).astype(
        np.uint64)
    ind_ixs = np.cumsum([0, ] + [len(inds) for inds in ind_lst])
    all_ind = np.concatenate(ind_lst)
    for i in range(0, len(vert_lst)):
        start_ix, end_ix = ind_ixs[i], ind_ixs[i + 1]
        all_ind[start_ix:end_ix] += vert_offset[i]
    return all_ind, all_vert


def merge_meshes_incl_norm(ind_lst, vert_lst, norm_lst, nb_simplices=3):
    """
    Combine several meshes into a single one.

    Args:
        ind_lst: List[np.ndarray]
            array shapes [M, 1]
        vert_lst: List[np.ndarray]
            array shapes [N, 1]
        norm_lst: List[np.ndarray]
            array shapes [N, 1]
        nb_simplices: int
            Number of simplices, e.g. for triangles nb_simplices=3

    Returns: [np.array, np.array, np.array]

    """
    assert len(vert_lst) == len(ind_lst), "Length of indices list differs" \
                                          "from vertices list."
    if len(vert_lst) == 0:
        return [np.zeros((0,), dtype=np.uint64), np.zeros((0,)), np.zeros((0,))]
    else:
        all_vert = np.concatenate(vert_lst)

    if len(norm_lst) == 0:
        all_norm = np.zeros((0,))
    else:
        all_norm = np.concatenate(norm_lst)
    # store index and vertex offset of every partial mesh
    vert_offset = np.cumsum([0, ] + [len(verts) // nb_simplices for verts in vert_lst]).astype(
        np.uint64)
    ind_ixs = np.cumsum([0, ] + [len(inds) for inds in ind_lst])
    all_ind = np.concatenate(ind_lst)
    for i in range(0, len(vert_lst)):
        start_ix, end_ix = ind_ixs[i], ind_ixs[i + 1]
        all_ind[start_ix:end_ix] += vert_offset[i]
    return [all_ind, all_vert, all_norm]


def _mesh_loader(so):
    return so.mesh


def merge_someshes(sos: Iterable['segmentation.SegmentationObject'], nb_simplices: int = 3,
                   nb_cpus: int = 1, color_vals: Optional[Iterable[float]] = None,
                   cmap: Optional = None, alpha: float = 1.0, use_new_subfold: bool = True):
    """
    Merge meshes of SegmentationObjects. This will cache :py:class:`~syconn.reps.segmentation.SegmentationObject`.

    Args:
        sos: SegmentationObjects are used to get .mesh, N x 1
        nb_simplices: Number of simplices, e.g. for triangles nb_simplices=3
        nb_cpus: int
        color_vals: Color values for every mesh, N x 4 (rgba). No normalization!
        cmap: matplotlib colormap
        alpha: float
        use_new_subfold:

    Returns: np.array, np.array [, np.array]
        indices, vertices (scaled) [, colors]

    """
    all_ind = np.zeros((0,), dtype=np.uint64)
    all_norm = np.zeros((0,))
    all_vert = np.zeros((0,))
    colors = np.zeros((0,))
    if len(sos) == 0:
        return all_ind, all_vert, all_norm

    if nb_cpus > 1 or sos[0].version == 'tmp':  # assume all sos have the same type..
        meshes = start_multiprocess_imap(_mesh_loader, sos, nb_cpus=nb_cpus, show_progress=False)
    else:
        meshes = load_so_meshes_bulk(sos, use_new_subfold=use_new_subfold)
        for so in sos:
            so._mesh = meshes[so.id]
        meshes = [so.mesh for so in sos]
    if color_vals is not None and cmap is not None:
        color_vals = color_factory(color_vals, cmap, alpha=alpha)
    ind_lst = []
    vert_lst = []
    norm_lst = []
    color_lst = []
    for i, (ind, vert, norm) in enumerate(meshes):
        ind_lst.append(ind)
        vert_lst.append(vert)
        if norm is not None:
            norm_lst.append(norm)
        if color_vals is not None:
            color_lst.append(np.array([color_vals[i]] * len(vert)))
    # merge results
    if len(color_lst) != 0:
        colors = np.concatenate(color_lst)
        del color_lst
    if len(norm_lst) != 0:
        all_norm = np.concatenate(norm_lst)
        del norm_lst
    if len(vert_lst) != 0:
        all_vert = np.concatenate(vert_lst)
    if len(ind_lst) != 0:
        all_ind = np.concatenate(ind_lst)
        # store index and vertex offset of every partial mesh
        vert_offset = np.cumsum([0, ] + [len(verts) // nb_simplices for verts in vert_lst]).astype(np.uint64)
        ind_ixs = np.cumsum([0, ] + [len(inds) for inds in ind_lst])
        for i in range(0, len(vert_lst)):
            start_ix, end_ix = ind_ixs[i], ind_ixs[i + 1]
            all_ind[start_ix:end_ix] += vert_offset[i]

    assert len(all_vert) == len(all_norm) or len(all_norm) == 0, "Length of combined normals and vertices differ."
    if len(colors) > 0:
        return all_ind, all_vert, all_norm, colors
    return all_ind, all_vert, all_norm


def make_ply_string(dest_path, indices, vertices, rgba_color,
                    invert_vertex_order=False):
    """
    Creates a ply str that can be included into a .k.zip for rendering
    in KNOSSOS.
    # TODO: write out normals

    Args:
        dest_path:
        indices: np.array
        vertices: np.array
        rgba_color: Tuple[uint8] or np.array
        invert_vertex_order: bool
            Invert the vertex order.


    Returns: str

    """
    # create header
    vertices = vertices.astype(np.float32)
    indices = indices.astype(np.int32)
    if not rgba_color.ndim == 2:
        rgba_color = np.array(rgba_color, dtype=np.uint8).reshape((-1, 4))
    if not indices.ndim == 2:
        indices = np.array(indices, dtype=np.int64).reshape((-1, 3))
    if not vertices.ndim == 2:
        vertices = np.array(vertices, dtype=np.float32).reshape((-1, 3))
    if len(rgba_color) != len(vertices) and len(rgba_color) == 1 and rgba_color.shape[1] == 4:
        # TODO: create per tree color instead of per vertex color
        rgba_color = np.array([rgba_color[0] for i in range(len(vertices))],
                              dtype=np.uint8)
    else:
        if not (len(rgba_color) == len(vertices) and len(rgba_color[0]) == 4):
            msg = 'Color array has to be RGBA and to provide a color value f' \
                  'or every vertex!'
            log_proc.error(msg)
            raise ValueError(msg)
    if type(rgba_color) is list:
        rgba_color = np.array(rgba_color, dtype=np.uint8)
        log_proc.warn("Color input is list. It will now be converted "
                      "automatically, data will be unusable if not normalized"
                      " between 0 and 255. min/max of data:"
                      " {}, {}".format(rgba_color.min(), rgba_color.max()))
    elif not np.issubdtype(rgba_color.dtype, np.uint8):
        log_proc.warn("Color array is not of type integer or unsigned integer."
                      " It will now be converted automatically, data will be "
                      "unusable if not normalized between 0 and 255."
                      "min/max of data: {}, {}".format(rgba_color.min(),
                                                       rgba_color.max()))
        rgba_color = np.array(rgba_color, dtype=np.uint8)

    # ply file requires 1D object arrays
    ordering = -1 if invert_vertex_order else 1
    vertices = np.concatenate([vertices.astype(np.object),
                               rgba_color.astype(np.object)], axis=1)
    vertices = np.array([tuple(el) for el in vertices],
                        dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                               ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
                               ('alpha', 'u1')])
    # ply file requires 1D object arrays.
    indices = np.array([tuple([el[::ordering]], ) for el in indices],
                       dtype=[('vertex_indices', 'i4', (3,))])
    PlyData([PlyElement.describe(vertices, 'vertex'),
             PlyElement.describe(indices, 'face')]).write(dest_path)


def make_ply_string_wocolor(dest_path, indices, vertices,
                            invert_vertex_order=False):
    """
    Creates a ply str that can be included into a .k.zip for rendering
    in KNOSSOS.
    # TODO: write out normals

    Args:
        dest_path:
        indices: int
            iterable of indices
        vertices: int
            iterable of vertices
        invert_vertex_order: bool
            Invert the vertex order

    Returns: str

    """
    # create header
    vertices = vertices.astype(np.float32)
    indices = indices.astype(np.int32)
    if not indices.ndim == 2:
        indices = np.array(indices, dtype=np.int64).reshape((-1, 3))
    if not vertices.ndim == 2:
        vertices = np.array(vertices, dtype=np.float32).reshape((-1, 3))
    if invert_vertex_order:
        indices = indices[:, ::-1]
    ply_verts = np.empty(len(vertices), dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    ply_verts['x'], ply_verts['y'], ply_verts['z'] = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    ply_faces = np.empty(len(indices), dtype=[('vertex_indices', 'i4', (3,))])
    ply_faces['vertex_indices'] = indices
    PlyData([PlyElement.describe(ply_verts, 'vertex'),
             PlyElement.describe(ply_faces, 'face')]).write(dest_path)


def write_mesh2kzip(k_path, ind, vert, norm, color, ply_fname,
                    force_overwrite=False, invert_vertex_order=False):
    """
    Writes mesh as .ply's to k.zip file.

    Args:
        k_path: str
            path to zip
        ind: np.array
        vert: np.array
        norm: np.array
        color: tuple or np.array
            rgba between 0 and 255
        ply_fname: str
        force_overwrite: bool
        invert_vertex_order: bool
            Invert the vertex order.

    Returns:

    """
    if not k_path.endswith('.k.zip'):
        k_path += '.k.zip'
    if len(vert) == 0:
        log_proc.warn("'write_mesh2kzip' called with empty vertex array. Did not"
                      " write data to kzip. `ply_fname`. {}".format(ply_fname))
        return
    tmp_dest_p = '{}_{}'.format(k_path, ply_fname)
    if color is not None:
        make_ply_string(tmp_dest_p, ind, vert.astype(np.float32), color,
                        invert_vertex_order=invert_vertex_order)
    else:
        make_ply_string_wocolor(tmp_dest_p, ind, vert.astype(np.float32),
                                invert_vertex_order=invert_vertex_order)
    write_data2kzip(k_path, tmp_dest_p, ply_fname,
                    force_overwrite=force_overwrite)


def write_meshes2kzip(k_path, inds, verts, norms, colors, ply_fnames,
                      force_overwrite=True, verbose=True,
                      invert_vertex_order=False):
    """
    Writes meshes as .ply's to k.zip file.

    Args:
        k_path: str
            path to zip
        inds: list of np.array
        verts: list of np.array
        norms: list of np.array
        colors: list of tuple or np.array
            rgba between 0 and 255
        ply_fnames: list of str
        force_overwrite: bool
        verbose: bool
        invert_vertex_order: bool
            Invert the vertex order.

    Returns:

    """
    if not k_path.endswith('.k.zip'):
        k_path += '.k.zip'
    tmp_paths = []
    if verbose:
        log_proc.info('Generating ply files.')
        pbar = tqdm.tqdm(total=len(inds), leave=False)
    write_out_ply_fnames = []
    for i in range(len(inds)):
        vert = verts[i]
        ind = inds[i]
        norm = norms[i]
        color = colors[i]
        ply_fname = ply_fnames[i]
        tmp_dest_p = '{}_{}'.format(k_path, ply_fname)
        if len(vert) == 0:
            log_proc.warning("Mesh with zero-length vertex array. Skipping.")
            continue
        if color is not None:
            make_ply_string(tmp_dest_p, ind, vert.astype(np.float32), color,
                            invert_vertex_order=invert_vertex_order)
        else:
            make_ply_string_wocolor(tmp_dest_p, ind, vert.astype(np.float32),
                                    invert_vertex_order=invert_vertex_order)
        tmp_paths.append(tmp_dest_p)
        write_out_ply_fnames.append(ply_fname)
        if verbose:
            pbar.update(1)
    if verbose:
        pbar.close()
    data2kzip(k_path, tmp_paths, write_out_ply_fnames, force_overwrite=force_overwrite,
              verbose=verbose)


def get_bb_size(coords):
    bb_min, bb_max = np.min(coords, axis=0), np.max(coords, axis=0)
    return np.linalg.norm(bb_max - bb_min, ord=2)


def color_factory(c_values, mcmap, alpha=1.0):
    colors = []
    for c_val in c_values:
        curr_color = list(mcmap(c_val))
        curr_color[-1] = alpha
        colors.append(curr_color)
    return np.array(colors)


def compartmentalize_mesh(ssv: 'super_segmentation_object.SuperSegmentationObject', pred_key_appendix=""):
    """
    Splits SuperSegmentationObject mesh into axon, dendrite and soma. Based
    on axoness prediction of SV's contained in SuperSuperVoxel ssv.

    Args:
        ssv: SuperSegmentationObject
        pred_key_appendix: str
            Specific version of axoness prediction

    Returns: np.array
        Majority label of each face / triangle in mesh indices;
        Triangle faces are assumed. If majority class has n=1, majority label is
        set to -1.

    """
    # TODO: requires update to include the bouton labels as axon
    preds = np.array(start_multiprocess_obj("axoness_preds",
                                            [[sv, {"pred_key_appendix": pred_key_appendix}]
                                             for sv in ssv.svs],
                                            nb_cpus=ssv.nb_cpus))
    preds = np.concatenate(preds)
    locs = ssv.sample_locations()
    pred_coords = np.concatenate(locs)
    assert pred_coords.ndim == 2, "Sample locations of ssv have wrong shape."
    assert pred_coords.shape[1] == 3, "Sample locations of ssv have wrong shape."
    ind, vert, axoness = ssv._pred2mesh(pred_coords, preds, k=3,
                                        colors=(0, 1, 2))
    # get axoness of each vertex where indices are pointing to
    ind_comp = axoness[ind]
    ind = ind.reshape(-1, 3)
    vert = vert.reshape(-1, 3)
    norm = ssv.mesh[2].reshape(-1, 3)
    ind_comp = ind_comp.reshape(-1, 3)
    ind_comp_maj = np.zeros((len(ind)), dtype=np.uint8)
    for ii in range(len(ind)):
        triangle = ind_comp[ii]
        cnt = Counter(triangle)
        ax, n = cnt.most_common(1)[0]
        if n == 1:
            ax = -1
        ind_comp_maj[ii] = ax
    comp_meshes = {}
    for ii, comp_type in enumerate(["axon", "dendrite", "soma"]):
        comp_ind = ind[ind_comp_maj == ii].flatten()
        unique_comp_ind = np.unique(comp_ind)
        comp_vert = vert[unique_comp_ind].flatten()
        if len(ssv.mesh[2]) != 0:
            comp_norm = norm[unique_comp_ind].flatten()
        else:
            comp_norm = ssv.mesh[2]
        remap_dict = {}
        for i in range(len(unique_comp_ind)):
            remap_dict[unique_comp_ind[i]] = i
        comp_ind = np.array([remap_dict[i] for i in comp_ind], dtype=np.uint64)
        comp_meshes[comp_type] = [comp_ind, comp_vert, comp_norm]
    return comp_meshes


def mesh_creator_sso(ssv: 'super_segmentation_object.SuperSegmentationObject',
                     segobjs: Iterable[str] = ('sv', 'mi', 'sj', 'vc')):
    """
    Cache meshes of specified SegmentationObjects.

    Args:
        ssv: SuperSegmentationObject.
        segobjs: Types of SegmentationObjects.

    Returns:

    """
    ssv.enable_locking = False
    ssv.load_attr_dict()
    for obj_type in segobjs:
        _ = ssv.load_mesh(obj_type)
    ssv.clear_cache()


def find_meshes(chunk: np.ndarray, offset: np.ndarray, pad: int = 0,
                ds: Optional[Union[list, tuple, np.ndarray]] = None,
                scaling: Optional[Union[tuple, list, np.ndarray]] = None,
                meshing_props: Optional[dict] = None) -> Dict[int, List[np.ndarray]]:
    """
    Find meshes within a segmented cube. The offset is given in voxels. Mesh vertices are scaled according to
    ``global_params.config['scaling']``.

    Args:
        chunk: Cube which is processed.
        offset: Offset of the cube in voxels.
        pad: Pad chunk array with mode 'edge'.
        ds: Downsampling array in xyz. Default: No downsampling.
        scaling: Voxel size.
        meshing_props: Keyword arguments used in ``zmesh.Mesher.get_mesh``.

    Returns:
        The mesh of each segmentation ID in the input `chunk`. Vertices are in nm!
    """
    if scaling is None:
        scaling = np.array(global_params.config['scaling'], copy=True)
    else:
        scaling = np.array(scaling, copy=True)
    if meshing_props is None:
        meshing_props = global_params.config['meshes']['meshing_props']
    offset = offset * scaling
    # keep small segmentation objects
    seg_objs = set(np.unique(chunk))
    if 0 in seg_objs:
        seg_objs.remove(0)
    meshes = {ix: [np.zeros(0, dtype=np.uint32), np.zeros(0, dtype=np.float32),
                   np.zeros((0,), dtype=np.float32)] for ix in seg_objs}
    if ds is not None:
        ds = np.array(ds)
        chunk = zoom(chunk, 1 / ds, order=0)
        scaling *= ds
    if pad > 0:
        chunk = np.pad(chunk, 1, mode='edge')
        offset -= pad * scaling
    mesher = Mesher(scaling)
    mesher.mesh(chunk.swapaxes(0, 2))  # xyz -> zyx
    for obj_id in mesher.ids():
        # vertices are xyz in nm (after scaling)
        tmp = mesher.get_mesh(obj_id, **meshing_props)
        tmp.vertices[:] = (tmp.vertices + offset)
        # vertices can be below zero due to padding and down sampling.
        tmp.vertices[tmp.vertices[:] < 0] = 0
        meshes[obj_id] = [tmp.faces.flatten().astype(np.uint32),
                          tmp.vertices.flatten().astype(np.float32)]
        if tmp.normals is not None:
            meshes[obj_id].append(tmp.normals.flatten().astype(np.float32))
        else:
            meshes[obj_id].append(np.zeros((0,), dtype=np.float32))
        mesher.erase(obj_id)

    mesher.clear()

    return meshes


def mesh_chunk(args):
    attr_dir, obj_type = args
    scaling = global_params.config['scaling']
    meshing_props = global_params.config['meshes']['meshing_props']
    ad = AttributeDict(attr_dir + "/attr_dict.pkl", disable_locking=True)
    obj_ixs = list(ad.keys())
    if len(obj_ixs) == 0:
        return
    voxel_dc = VoxelStorage(attr_dir + "/voxel.pkl", disable_locking=True)
    md = MeshStorage(attr_dir + "/mesh.pkl", disable_locking=True, read_only=False)
    valid_obj_types = ["vc", "sj", "mi", "cs", 'syn', 'syn_ssv']
    if global_params.config.allow_mesh_gen_cells:
        valid_obj_types += ["sv"]
    if obj_type not in valid_obj_types:
        raise NotImplementedError("Object type '{}' must be one of the following:\n"
                                  "{}".format(obj_type, str(valid_obj_types)))
    ds = global_params.config['meshes']['downsampling'][obj_type]
    for ix in obj_ixs:
        min_obj_vx = global_params.config['meshes']['mesh_min_obj_vx']
        if ad[ix]['size'] < min_obj_vx:
            md[ix] = [np.zeros((0,), dtype=np.int32), np.zeros((0,), dtype=np.int32),
                      np.zeros((0,), dtype=np.float32)]
            continue
        # create binary mask as single 3D cube
        mask, off = voxel_dc.get_voxel_data_cubed(ix)
        # create mesh
        indices, vertices, normals = find_meshes(mask, off, pad=1, ds=ds, scaling=scaling, meshing_props=meshing_props)[
            ix]
        md[ix] = [indices.flatten(), vertices.flatten(), normals.flatten()]
    md.push()


def get_object_mesh(obj: 'segmentation.SegmentationObject', ds: Union[tuple, list, np.ndarray],
                    mesher_kwargs: Optional[dict] = None):
    """
    Get object mesh from object voxels using marching cubes. Boundary artifacts
    are minimized by using a single 3D mask array of the object.

    Notes:
        This method is not suited for large objects as it creates a single 3D binary mask of the object.

    Args:
        obj: SegmentationObject.
        ds: Magnitude of downsampling for each axis.
        mesher_kwargs: Keyword arguments parsed to 'find_meshes' method.

    Returns:
        vertices [N, 1], indices [M, 1], normals [M, 1]

    """
    if mesher_kwargs is None:
        mesher_kwargs = {}
    min_obj_vx = global_params.config['meshes']['mesh_min_obj_vx']
    zero_out = [np.zeros((0,), dtype=np.int32), np.zeros((0,), dtype=np.int32),
                np.zeros((0,), dtype=np.float32)]
    if obj.size < min_obj_vx:
        return zero_out

    # create binary mask as single 3D cube
    mask = obj.voxels
    off = obj.bounding_box[0]  # in voxel
    # create mesh; binary mask -> object always has ID 1
    indices, vertices, normals = find_meshes(mask, off, pad=1, ds=ds, scaling=obj.scaling, **mesher_kwargs)[1]
    if 0 < len(normals) != len(vertices):
        msg = f'Length of normals ({normals.shape}) does not correspond to length of vertices ({vertices.shape}).'
        log_proc.error(msg)
        raise ValueError(msg)
    return [indices.flatten(), vertices.flatten(), normals.flatten()]


def mesh2obj_file(dest_path: str, mesh: List[np.ndarray],
                  color: Optional[Union[int, np.ndarray]] = None,
                  center: Optional[np.ndarray] = None,
                  scale: Optional[float] = None):
    """
    Writes mesh to .obj file.

    Args:
        dest_path: Path to file.
        mesh: Flat arrays of indices (triangle faces), vertices and normals.
        color: Color as int or numpy array (rgba).
        center: Subtracts center from original vertex locations.
        scale: Multiplies vertex locations after centering.

    Returns:

    """
    mesh_obj = openmesh.TriMesh()
    ind, vert, norm = mesh
    if vert.ndim == 1:
        vert = vert.reshape(-1, 3)
    if ind.ndim == 1:
        ind = ind.reshape(-1, 3)
    if center is not None:
        vert -= center
    if scale is not None:
        vert *= scale
    vert_openmesh = []
    if color is not None:
        mesh_obj.request_vertex_colors()
        if color.ndim == 1:
            color = np.array([color] * len(vert))
        color = color.astype(np.float64)  # required by openmesh
    for ii, v in enumerate(vert):
        v = v.astype(np.float64)  # Point requires double
        v_openmesh = mesh_obj.add_vertex(v)
        if color is not None:
            mesh_obj.set_color(v_openmesh, color[ii])
        vert_openmesh.append(v_openmesh)
    for f in ind:
        f_openmesh = [vert_openmesh[f[0]], vert_openmesh[f[1]],
                      vert_openmesh[f[2]]]
        mesh_obj.add_face(f_openmesh)
    openmesh.write_mesh(dest_path, mesh_obj)


def mesh_area_calc(mesh):
    """

    Args:
        mesh: meshobject

    Returns: float
        Mesh area in um^2

    """
    return mesh_surface_area(mesh[1].reshape(-1, 3),
                             mesh[0].reshape(-1, 3)) / 1e6


def gen_mesh_voxelmask(voxel_iter: Iterator[Tuple[np.ndarray, np.ndarray]], scale: np.ndarray,
                       vertex_size: float = 10, boundary_struct: Optional[np.ndarray] = None,
                       depth: int = 10, compute_connected_components: bool = True,
                       voxel_size_simplify: Optional[float] = None,
                       min_vert_num: int = 200, overlap: int = 1, verbose: bool = False,
                       nb_neighbors: int = 20, std_ratio: float = 2.0) \
        -> Union[List[np.ndarray], List[List[np.ndarray]]]:
    """
    Args:
        voxel_iter: Iterator of binary voxel mask (3D cube) and cube offset (in voxels).
        scale: Size of voxels in `mask_list` in nm (x, y, z).
        vertex_size: In nm. Resolution used to simplify mesh.
        boundary_struct: Connectivity of kernel used to determine boundary
        depth: http://www.open3d.org/docs/latest/tutorial/Advanced/surface_reconstruction.html#Poi
            sson-surface-reconstruction :
            "An important parameter of the function is depth that defines the depth of the octree
            used for the surface reconstruction and hence implies the resolution of the resulting
            triangle mesh. A higher depth value means a mesh with more details."
        compute_connected_components: Compute connected components of mesh. Return list of meshes.
        voxel_size_simplify: Voxel size in nm when applying `simplify_vertex_clustering`. Defaults to `vertex_size`.
        min_vert_num: Minimum number of vertices of the connected component meshes (only applied if
            `compute_connected_components=True`).
        overlap: Overlap between adjacent masks in `mask_list`.
        verbose: Extra stdout output.
        nb_neighbors: Number of neighbors used to calculate distance mean and standard deviation. See
            http://www.open3d.org/docs/latest/tutorial/Advanced/pointcloud_outlier_removal.html#Statistical-outlier-removal
        std_ratio: Standard deviation of distance between points used as threshold for filtering. See
            http://www.open3d.org/docs/latest/tutorial/Advanced/pointcloud_outlier_removal.html#Statistical-outlier-removal

    Notes: Use `voxel_iter` with cubes that have 1-voxel-overlap to guarantee that segmentation instance boundaries
        that align with the 3D  array border are identified correctly.

    Returns:
        Flat Index/triangle, vertex and normals array of the mesh. List[ind, vert, norm] if
        ``compute_connected_components=True``.
    """
    if voxel_size_simplify is None:
        voxel_size_simplify = vertex_size
    if boundary_struct is None:
        # 26-connected
        boundary_struct = np.ones((3, 3, 3))
    pts, norm = [], []
    for m, off in tqdm.tqdm(voxel_iter, disable=not verbose, desc='VoxelLoad'):
        bndry = m.astype(np.float32) - binary_erosion(m, boundary_struct, iterations=1)
        if overlap > 0:
            m = m[overlap:-overlap, overlap:-overlap, overlap:-overlap]
            bndry = bndry[overlap:-overlap, overlap:-overlap, overlap:-overlap]
        try:
            grad = gaussianGradient(m.astype(np.float32), 3)  # sigma=3
        except RuntimeError:  # PreconditionViolation (current mask cube is smaller than kernel)
            m = np.pad(m, 10)
            grad = gaussianGradient(m.astype(np.float32), 3)[10:-10, 10:-10, 10:-10]
        # mult. by -1 to make normals point outwards
        mag = -np.linalg.norm(grad, axis=-1)
        grad[mag != 0] /= mag[mag != 0][..., None]
        nonzero_mask = np.nonzero(bndry)
        if np.abs(mag[nonzero_mask]).min() == 0:
            log_proc.warn('Found zero gradient during mesh generation.')
        pts_ = np.transpose(nonzero_mask) + off + overlap
        pts.append(pts_)
        norm_ = grad[nonzero_mask]
        norm.append(norm_)
    norm = np.concatenate(norm)
    pts = np.concatenate(pts) * scale
    assert norm.shape == pts.shape, 'Incorrect shapes for normals and points.'
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.normals = o3d.utility.Vector3dVector(norm)
    pcd = pcd.voxel_down_sample(voxel_size=vertex_size)  # reduce number of points
    # TODO: add to config
    pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=4*np.max(scale), max_nn=30))
    # # TODO: use orient_normals_consistent_tangent_plane as soon as open3d>0.10 is working
    # pcd.orient_normals_consistent_tangent_plane(100)
    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)

    # ball pivoting is slow
    # radii = np.array([1, 2, 3, 4], dtype=np.float32) * vertex_size
    # mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
    #     pcd, o3d.utility.DoubleVector(radii.tolist()))

    mesh = mesh.simplify_vertex_clustering(
        voxel_size=voxel_size_simplify,
        contraction=o3d.geometry.SimplificationContraction.Quadric)

    if compute_connected_components:
        triangle_clusters, cluster_n_triangles, cluster_area = (
            mesh.cluster_connected_triangles())
        # # TODO: remove explicit numpy conversion as soon as open3d>0.9 is working
        triangle_clusters = np.array(triangle_clusters)
        cluster_n_triangles = np.array(cluster_n_triangles)
        # # TODO: use select_by_index as soon as open3d>0.9 is working
        # triangles_to_remove = cluster_n_triangles[triangle_clusters] < min_vert_num
        # mesh.remove_triangles_by_mask(triangles_to_remove)
        # mesh = [mesh.select_by_index[np.transpose(np.nonzero(triangle_clusters == ix))] for ix in
        #         range(len(cluster_n_triangles))]
        mesh_ = []
        for ii in range(len(cluster_n_triangles)):
            if cluster_n_triangles[ii] < min_vert_num:
                continue
            m = copy.deepcopy(mesh)
            m.remove_triangles_by_mask(triangle_clusters != ii)
            mesh_.append(m)
        mesh = mesh_
    else:
        mesh = [mesh]
    for ii in range(len(mesh)):
        m = mesh[ii]
        verts = np.asarray(m.vertices).flatten()
        verts[verts < 0] = 0
        mesh[ii] = [np.asarray(m.triangles).flatten(), verts, np.asarray(m.vertex_normals).flatten()]
    return mesh


def calc_contact_syn_mesh(segobj: 'segmentation.SegmentationObject',
                          voxel_dc: Optional[Union[VoxelStorageLazyLoading, VoxelStorageDyn]] = None,
                          **gen_kwgs) -> Union[List[np.ndarray], List[List[np.ndarray]]]:
    """

    Args:
        segobj:
        voxel_dc:
        **gen_kwgs:

    Returns:

    """
    assert segobj.type in ['cs', 'syn', 'syn_ssv'], 'Object type not supported'
    if segobj._voxel_list is None:
        if segobj.type == 'cs':
            if voxel_dc is None:
                voxel_dc = VoxelStorageDyn(segobj.voxel_path, read_only=True, disable_locking=True, voxel_mode=False)
            voxels = np.array(voxel_dc.get_voxel_cache(segobj.id), dtype=np.uint32)
        else:
            if voxel_dc is None:
                voxel_dc = VoxelStorageLazyLoading(segobj.voxel_path)
            voxels = np.array(voxel_dc[segobj.id], dtype=np.uint32)
    else:
        voxels = np.array(segobj._voxel_list, dtype=np.uint32)
    abs_offset = np.min(voxels, axis=0)
    # reduce offset by one -> voxels in 3D cube will have an additional offset of 1 which creates a border of 0s.
    voxels -= abs_offset - 1
    id_mask = np.zeros(np.max(voxels, axis=0) + 2, dtype=np.bool)
    id_mask[voxels[:, 0], voxels[:, 1], voxels[:, 2]] = True
    return gen_mesh_voxelmask(zip([id_mask], [abs_offset - 1]), segobj.scaling, overlap=0, **gen_kwgs)


def calc_cell_mesh_from_points(segobj: 'segmentation.SegmentationObject', **gen_kwgs) \
        -> Union[List[np.ndarray], List[List[np.ndarray]]]:
    voxel_dc = VoxelStorage(segobj.voxel_path, read_only=True, disable_locking=True)
    voxel_iter = voxel_dc.iter_voxelmask_offset(segobj.id, overlap=1)
    return gen_mesh_voxelmask(voxel_iter, segobj.scaling, overlap=1, compute_connected_components=False, **gen_kwgs)
