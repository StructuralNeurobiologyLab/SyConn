# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Sven Dorkenwald, Philipp Schubert, Joergen Kornfeld

import itertools
import numpy as np
from collections import Counter
from numba import jit
from scipy import spatial, ndimage
from skimage import measure
from sklearn.decomposition import PCA
import openmesh
from plyfile import PlyData, PlyElement
from scipy.ndimage.morphology import binary_closing, binary_erosion,\
    binary_dilation
import tqdm
try:
    import vtkInterface
    __vtk_avail__ = True
except ImportError:
    __vtk_avail__ = False

from ..proc import log_proc
from ..handler.basics import write_data2kzip, data2kzip
from .image import apply_pca
from ..backend.storage import AttributeDict, MeshStorage, VoxelStorage
from ..config.global_params import MESH_DOWNSAMPLING, MESH_CLOSING, \
    get_dataset_scaling, MESH_MIN_OBJ_VX
from ..mp.mp_utils import start_multiprocess_obj, start_multiprocess_imap
try:
    # set matplotlib backend to offscreen
    import matplotlib
    matplotlib.use('agg')
    from vigra.filters import boundaryDistanceTransform, gaussianSmoothing
except ImportError:
    boundaryDistanceTransform, gaussianSmoothing = None, None
    log_proc.error('ModuleNotFoundError. Could not import VIGRA. '
                   'Mesh generation will not be possible.')

__all__ = ['MeshObject', 'get_object_mesh', 'merge_meshes', 'triangulation',
           'get_random_centered_coords', 'write_mesh2kzip', 'write_meshes2kzip',
           'compartmentalize_mesh', 'mesh_chunk', 'mesh_creator_sso']


class MeshObject(object):
    def __init__(self, object_type, indices, vertices, normals=None,
                 color=None, bounding_box=None):
        self.object_type = object_type
        if vertices.ndim == 2 and vertices.shape[1] == 3:
            self.vertices = vertices.flatten()
        else:
            # assume flat array
            self.vertices = np.array(vertices, dtype=np.float)
        if indices.ndim == 2 and indices.shape[1] == 3:
            self.indices = indices.flatten().astype(np.uint)
        else:
            # assume flat array
            self.indices = np.array(indices, dtype=np.uint)
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
        self.center = self.center.astype(np.float)
        self.max_dist = self.max_dist.astype(np.float)
        vert_resh = np.array(self.vertices).reshape((len(self.vertices) // 3, 3))
        vert_resh -= np.array(self.center, dtype=self.vertices.dtype)
        vert_resh = vert_resh / np.array(self.max_dist)
        self.vertices = vert_resh.reshape(len(self.vertices))
        if normals is not None and len(normals) == 0:
            normals = None
        if normals is not None and normals.ndim == 2:
            normals = normals.reshape(len(normals)*3)
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
                assert self._ext_color.shape[1] == 4,\
                    "'color' parameter has wrong shape"
                self._ext_color = self._ext_color.squeeze()
                assert self._ext_color.shape[1] == 4,\
                    "Rendering requires RGBA 'color' shape of (X, 4). Please" \
                    "add alpha channel."
                self._ext_color = self._ext_color.flatten()
            assert len(self._ext_color)/4 == len(self.vertices)/3, \
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
            log_proc.info("Calculating normals")
            self._normals = unit_normal(self.vertices, self.indices)
        elif len(self._normals) != len(self.vertices):
            log_proc.info("Calculating normals, because their shape differs from"
                  " vertices: %s (normals) vs. %s (vertices)" %
                  (str(self._normals.shape), str(self.vertices.shape)))
            self._normals = unit_normal(self.vertices, self.indices)
        return self._normals

    @property
    def normals_resh(self):
        return self.normals.reshape(-1, 3)

    def transform_external_coords(self, coords):
        """

        Parameters
        ----------
        coords : np.array

        Returns
        -------
        np.array
            transformed coordinates
        """
        if len(coords) == 0:
            return coords
        coords = np.array(coords, dtype=np.float)
        coords = coords - self.center
        coords /= self.max_dist
        return coords

    def retransform_external_coords(self, coords):
        coords = np.array(coords, dtype=np.float)
        coords *= self.max_dist
        coords += self.center
        return coords.astype(np.int)

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

        Parameters
        ----------
        bounding_box : tuple
            center, scale (applied as follows: self.vert_resh / scale)
        """
        if bounding_box is None:
            bounding_box = get_bounding_box(self.vertices)
        self.center, self.max_dist = bounding_box
        self.center = self.center.astype(np.float)
        self.max_dist = self.max_dist.astype(np.float)
        vert_resh = np.array(self.vertices).reshape(len(self.vertices) // 3, 3)
        vert_resh -= self.center
        vert_resh /= self.max_dist
        self.vertices = vert_resh.reshape(len(self.vertices))

    @property
    def vertices_scaled(self):
        return (self.vert_resh * self.max_dist + self.center).flatten()


def triangulation(pts, downsampling=(1, 1, 1), n_closings=0, single_cc=False,
                  decimate_mesh=0, gradient_direction='ascent',
                  force_single_cc=False):
    """
    Calculates triangulation of point cloud or dense volume using marching cubes
    by building dense matrix (in case of a point cloud) and applying marching
    cubes.

    Parameters
    ----------
    pts : np.array
        [N, 3] or [N, M, O] (dtype: uint8, bool)
    downsampling : Tuple[int]
        Magnitude of downsampling, e.g. 1, 2, (..) which is applied to pts
        for each axis
    n_closings : int
        Number of closings applied before mesh generation
    single_cc : bool
        Returns mesh of biggest connected component only
    decimate_mesh : float
        Percentage of mesh size reduction, i.e. 0.1 will leave 90% of the
        vertices
    gradient_direction : str
        defines orientation of triangle indices. 'ascent' is needed for KNOSSOS
         compatibility.
    force_single_cc : bool
        If True, performans dilations until only one foreground CC is present
        and then erodes with the same number to maintain size.

    Returns
    -------
    array, array, array
        indices [M, 3], vertices [N, 3], normals [N, 3]

    """
    assert type(downsampling) == tuple, "Downsampling has to be of type 'tuple'"
    assert (pts.ndim == 2 and pts.shape[1] == 3) or pts.ndim == 3, \
        "Point cloud used for mesh generation has wrong shape."
    if pts.ndim == 2:
        if np.max(pts) <= 1:
            msg = "Currently this function only supports point " \
                  "clouds with coordinates >> 1."
            log_proc.error(msg)
            raise ValueError(msg)
        offset = np.min(pts, axis=0)
        pts -= offset
        pts = (pts / downsampling).astype(np.uint32)
        # add zero boundary around object
        margin = n_closings + 5
        pts += margin
        bb = np.max(pts, axis=0) + margin
        volume = np.zeros(bb, dtype=np.float32)
        volume[pts[:, 0], pts[:, 1], pts[:, 2]] = 1
    else:
        volume = pts
        if np.any(np.array(downsampling) != 1):
            # volume = measure.block_reduce(volume, downsampling, np.max)
            ndimage.zoom(volume, downsampling, order=0)
        offset = np.array([0, 0, 0])
    # volume = multiBinaryErosion(volume, 1).astype(np.float32)
    if n_closings > 0:
        volume = binary_closing(volume, iterations=n_closings).astype(np.float32)
        if force_single_cc:
            n_dilations = 0
            while True:
                labeled, nb_cc = ndimage.label(volume)
                log_proc.debug('Forcing single CC, additional dilations {}, num'
                               'ber connected components: {}'
                               ''.format(n_dilations, nb_cc))
                if nb_cc == 1:  # does not count background
                    break
                # pad volume to maintain margin at boundary and correct offset
                volume = np.pad(volume, [(1, 1), (1, 1), (1, 1)],
                                mode='constant', constant_values=0)
                offset -= 1
                volume = binary_dilation(volume, iterations=1).astype(
                    np.float32)
                n_dilations += 1
    else:
        volume = volume.astype(np.float32)
    if single_cc:
        labeled, nb_cc = ndimage.label(volume)
        cnt = Counter(labeled[labeled != 0])
        l, occ = cnt.most_common(1)[0]
        volume = np.array(labeled == l, dtype=np.float32)
    # InterpixelBoundary, OuterBoundary, InnerBoundary
    dt = boundaryDistanceTransform(volume, boundary="InterpixelBoundary")
    dt[volume == 1] *= -1
    volume = gaussianSmoothing(dt, 1)
    if np.sum(volume < 0) == 0 or np.sum(volume > 0) == 0:  # less smoothing
        volume = gaussianSmoothing(dt, 0.5)
    try:
        verts, ind, norm, _ = measure.marching_cubes_lewiner(
            volume, 0, gradient_direction=gradient_direction)
    except Exception as e:
        log_proc.error(e)
        raise RuntimeError(e)
    if pts.ndim == 2:  # account for [5, 5, 5] offset
        verts -= margin
    verts = np.array(verts) * downsampling + offset
    if decimate_mesh > 0:
        if not __vtk_avail__:
            msg = "vtkInterface not installed. Please install vtkInterface.'" \
                  "git clone https://github.com/akaszynski/vtkInterface.git' " \
                  "and 'pip install -e vtkInterface'."
            log_proc.error(msg)
            raise ImportError(msg)
        log_proc.warning("'triangulation': Currently mesh-sparsification"
                         " may not preserve volume.")
        # add number of vertices in front of every face (required by vtkInterface)
        ind = np.concatenate([np.ones((len(ind), 1)).astype(np.int64) * 3, ind],
                             axis=1)
        mesh = vtkInterface.PolyData(verts, ind.flatten()).TriFilter()
        decimated_mesh = mesh.Decimate(decimate_mesh, volume_preservation=True)
        if decimated_mesh is None:  # maybe vtkInterface API changes and operates in-place -> TODO: check version differences and require one of them
            decimated_mesh = mesh
            if len(decimated_mesh.faces.reshape((-1, 4))[:, 1:]) == len(ind):
                log_proc.error(
                    "'triangulation': Mesh-sparsification could not sparsify"
                    " mesh.")
        # remove face sizes again
        ind = decimated_mesh.faces.reshape((-1, 4))[:, 1:]
        verts = decimated_mesh.points
        mo = MeshObject("", ind, verts)
        # compute normals
        norm = mo.normals.reshape((-1, 3))
    return np.array(ind, dtype=np.int), verts, norm


def get_object_mesh(obj, downsampling, n_closings, decimate_mesh=0,
                    triangulation_kwargs=None):
    """
    Get object mesh from object voxels using marching cubes.

    Parameters
    ----------
    obj : SegmentationObject
    downsampling : tuple of int
        Magnitude of downsampling for each axis
    n_closings : int
        Number of closings before mesh generation
    decimate_mesh : float
    triangulation_kwargs : dict
     Keyword arguments parsed to 'traingulation' call

    Returns
    -------
    array [N, 1], array [M, 1], array [M, 1]
        vertices, indices, normals
    """
    if triangulation_kwargs is None:
        triangulation_kwargs = {}
    if np.isscalar(obj.voxels):
        return np.zeros((0,), dtype=np.int32), np.zeros((0,), dtype=np.int32),\
               np.zeros((0,), dtype=np.float32)
    if len(obj.voxel_list) <= MESH_MIN_OBJ_VX:
        log_proc.warn('Did not create mesh for object of type "{}" '
                      ' with ID {} because it contained less than {} voxels.'
                      ''.format(obj.id, obj.type, len(obj.voxel_list)))
        return np.zeros((0,), dtype=np.int32), np.zeros((0,), dtype=np.int32),\
               np.zeros((0,), dtype=np.float32)
    try:
        indices, vertices, normals = triangulation(
            np.array(obj.voxel_list), downsampling=downsampling,
            n_closings=n_closings, decimate_mesh=decimate_mesh,
            **triangulation_kwargs)
    except RuntimeError as e:
        msg = 'Error during marching_cubes procedure of SegmentationObject {}' \
              ' of type "{}". It contained {} voxels'.format(
            obj.id, obj.type, len(obj.voxel_list))
        log_proc.error(msg)
        return np.zeros((0,)), np.zeros((0,)), np.zeros((0,))
    vertices += 1  # account for knossos 1-indexing
    vertices = np.round(vertices * obj.scaling)
    assert len(vertices) == len(normals) or len(normals) == 0, \
        "Length of normals (%s) does not correspond to length of" \
        " vertices (%s)." % (str(normals.shape), str(vertices.shape))
    return indices.flatten(), vertices.flatten(), normals.flatten()


def normalize_vertices(vertices):
    """
    Rotate, center and normalize vertices.

    Parameters
    ----------
    vertices : np.array
        [N, 1]

    Returns
    -------
    array
        transformed vertices
    """
    vert_resh = vertices.reshape(len(vertices) // 3, 3)
    vert_resh = apply_pca(vert_resh)
    vert_resh -= np.median(vert_resh, axis=0)
    max_val = np.abs(vert_resh).max()
    vert_resh = vert_resh / max_val
    vertices = vert_resh.reshape(len(vertices)).astype(np.float32)
    return vertices


def calc_rot_matrices(coords, vertices, edge_length):
    """
    # TODO: optimize with cython (bottleneck is probably 'in_bounding_box' -> create single for loop)
    Fits a PCA to local sub-volumes in order to rotate them according to
    its main process (e.g. x-axis will be parallel to the long axis of a tube)

    Parameters
    ----------
    coords : np.array [M x 3]
    vertices : np.array [N x 3]
    edge_length : float, int
        spatial extent of box for querying vertices for pca fit

    Returns
    -------
    np.array [M x 16]
        Fortran flattened OpenGL rotation matrix
    """
    if not np.isscalar(edge_length):
        log_proc.warning('"calc_rot_matrices" now takes only scalar edgelengths'
                         '. Choosing np.min(edge_length) as query box edge'
                         ' length.')
        edge_length = np.min(edge_length)
    if len(vertices) > 1e5:
        vertices = vertices[::8]
    rot_matrices = np.zeros((len(coords), 16))
    edge_lengths = np.array([edge_length] * 3)
    for ii, c in enumerate(coords):
        bounding_box = (c, edge_lengths)
        inlier = np.array(vertices[in_bounding_box(vertices, bounding_box)])
        rot_matrices[ii] = get_rotmatrix_from_points(inlier)
    return rot_matrices


def get_rotmatrix_from_points(points):
    """
    Fits pca to input points and returns corresponding rotation matrix, usable
    in PyOpenGL.

    Parameters
    ----------
    points : np.array

    Returns
    -------

    """
    if len(points) <= 2:
        return np.zeros((16))
    new_center = np.mean(points, axis=0)
    points -= new_center
    pca = PCA(n_components=3, random_state=0)
    pca.fit(points)
    rot_mat = np.zeros((4, 4))
    rot_mat[:3, :3] = pca.components_
    rot_mat[3, 3] = 1
    rot_mat = rot_mat.flatten('F')
    return rot_mat


def flag_empty_spaces(coords, vertices, edge_length):
    """
    Flag empty locations.

    Parameters
    ----------
    coords : np.array
        [M x 3]
    vertices : np.array
        [N x 3]
    edge_length : np.array
        spatial extent of bounding box to look for vertex support

    Returns
    -------
    np.array [M x 1]
        
    """
    if not np.isscalar(edge_length):
        log_proc.warning('"calc_rot_matrices" now takes only scalar edgelengths'
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


def get_bounding_box(coordinates):
    """
    Calculates center of coordinates and its maximum distance in any spatial
    dimension to the most distant point.

    Parameters
    ----------
    coordinates : np.array

    Returns
    -------
    np.array, float
        center, distance
    """
    if coordinates.ndim == 2 and coordinates.shape[1] == 3:
        coord_resh = coordinates
    else:
        coord_resh = coordinates.reshape(len(coordinates) // 3, 3)
    mean = np.mean(coord_resh, axis=0)
    max_dist = np.max(np.abs(coord_resh - mean))
    return mean, max_dist


@jit
def in_bounding_box(coords, bounding_box):
    """
    Loop version with numba

    Parameters
    ----------
    coords : np.array (N x 3)
    bounding_box : tuple (np.array, np.array)
        center coordinate and edge lengths of bounding box

    Returns
    -------
    np.array of bool
        inlying coordinates are indicated as true
    """
    edge_sizes = bounding_box[1] / 2
    coords = np.array(coords) - bounding_box[0]
    inlier = np.zeros((len(coords)), dtype=np.bool)
    for i in range(len(coords)):
        x_cond = (coords[i, 0] > -edge_sizes[0]) & (coords[i, 0] < edge_sizes[0])
        y_cond = (coords[i, 1] > -edge_sizes[1]) & (coords[i, 1] < edge_sizes[1])
        z_cond = (coords[i, 2] > -edge_sizes[2]) & (coords[i, 2] < edge_sizes[2])
        inlier[i] = x_cond & y_cond & z_cond
    return inlier


@jit
def get_avg_normal(normals, indices, nbvert):
    normals_avg = np.zeros((nbvert, 3), np.float)
    for n in range(len(indices)):
        ix = indices[n]
        normals_avg[ix] += normals[n]
    return normals_avg


def unit_normal(vertices, indices):
    """
    Calculates normals per face (averaging corresponding vertex normals) and
    expands it to (averaged) normals per vertex.

    Parameters
    ----------
    vertices : np.array [N x 1]
        Flattend vertices
    indices : np.array [M x 1]
        Flattend indices

    Returns
    -------
    np.array [N x 1]
        Unit face normals per vertex
    """
    vertices = np.array(vertices, dtype=np.float)
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
    return -normals_avg.astype(np.float32).reshape(nbvert*3)


def get_random_centered_coords(pts, nb, r):
    """

    Parameters
    ----------
    pts : np.array
        coordinates
    nb : int
        number of center of masses to be returned
    r : int
        radius of query_ball_point in order to get list of points for
        center of mass

    Returns
    -------
    np.array
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

    Parameters
    ----------
    ind_lst : list of np.array [N, 1]
    vert_lst : list of np.array [N, 1]
    nb_simplices : int
        Number of simplices, e.g. for triangles nb_simplices=3

    Returns
    -------
    np.array, np.array
    """
    assert len(vert_lst) == len(ind_lst), "Length of indices list differs" \
                                          "from vertices list."
    all_ind = np.zeros((0, ), dtype=np.uint)
    all_vert = np.zeros((0, ))
    for i in range(len(vert_lst)):
        all_ind = np.concatenate([all_ind, ind_lst[i] +
                                  len(all_vert)/nb_simplices])
        all_vert = np.concatenate([all_vert, vert_lst[i]])
    return all_ind, all_vert


def mesh_loader(so):
    return so.mesh


def merge_someshes(sos, nb_simplices=3, nb_cpus=1, color_vals=None,
                  cmap=None, alpha=1.0):
    """
    Merge meshes of SegmentationObjects.

    Parameters
    ----------
    sos : iterable of SegmentationObject
        SegmentationObjects are used to get .mesh, N x 1
    nb_simplices : int
        Number of simplices, e.g. for triangles nb_simplices=3
    color_vals : iterable of float
        color values for every mesh, N x 4 (rgba). No normalization!
    nb_cpus : int
    cmap : matplotlib colormap
    alpha : float

    Returns
    -------
    np.array, np.array [, np.array]
        indices, vertices (scaled) [,colors]
    """
    all_ind = np.zeros((0, ), dtype=np.uint)
    all_vert = np.zeros((0, ))
    all_norm = np.zeros((0, ))
    colors = np.zeros((0, ))
    meshes = start_multiprocess_imap(mesh_loader, sos, nb_cpus=nb_cpus,
                                     show_progress=False)
    if color_vals is not None and cmap is not None:
        color_vals = color_factory(color_vals, cmap, alpha=alpha)
    for i in range(len(meshes)):
        ind, vert, norm = meshes[i]
        assert len(vert) == len(norm) or len(norm) == 0, "Length of normals " \
                                                         "and vertices differ."
        all_ind = np.concatenate([all_ind, ind + len(all_vert)/nb_simplices])
        all_vert = np.concatenate([all_vert, vert])
        all_norm = np.concatenate([all_norm, norm])
        if color_vals is not None:
            curr_color = np.array([color_vals[i]]*len(vert))
            colors = np.concatenate([colors, curr_color])
    assert len(all_vert) == len(all_norm) or len(all_norm) == 0, \
        "Length of combined normals and vertices differ."
    if color_vals is not None:
        return all_ind, all_vert, all_norm, colors
    return all_ind, all_vert, all_norm


def make_ply_string(dest_path, indices, vertices, rgba_color):
    """
    Creates a ply str that can be included into a .k.zip for rendering
    in KNOSSOS.
    # TODO: write out normals

    Parameters
    ----------
    indices : np.array
    vertices : np.array
    rgba_color : Tuple[uint8] or np.array

    Returns
    -------
    str
    """
    # create header
    if not indices.ndim == 2:
        indices = np.array(indices, dtype=np.int).reshape((-1, 3))
    if not vertices.ndim == 2:
        vertices = np.array(vertices, dtype=np.float32).reshape((-1, 3))
    if len(rgba_color) != len(vertices) and len(rgba_color) == 4:
        # TODO: create per tree color instead of per vertex color
        rgba_color = np.array([rgba_color for i in range(len(vertices))],
                              dtype=np.uint8)
    else:
        if not (len(rgba_color) == len(vertices) and len(rgba_color[0]) == 4):
            msg = 'Color array has to be RGBA and to provide a color value f' \
                  'or every vertex!'
            log_proc.error(msg)
            raise ValueError(msg)
    if not rgba_color.ndim == 2:
        rgba_color = np.array(rgba_color, dtype=np.int).reshape((-1, 4))
    if type(rgba_color) is list:
        rgba_color = np.array(rgba_color, dtype=np.uint8)
        log_proc.warn("Color input is list. It will now be converted "
                      "automatically, data will be unusable if not normalized"
                      " between 0 and 255. min/max of data:"
                      " {}, {}".format(rgba_color.min(), rgba_color.max()))
    elif rgba_color.dtype.kind not in ("u", "i"):
        log_proc.warn("Color array is not of type integer or unsigned integer."
                      " It will now be converted automatically, data will be "
                      "unusable if not normalized between 0 and 255."
                      "min/max of data: {}, {}".format(rgba_color.min(),
                                                       rgba_color.max()))
        rgba_color = np.array(rgba_color, dtype=np.uint8)
    # ply file requires 1D object arrays,
    vertices = np.concatenate([vertices.astype(np.object),
                               rgba_color.astype(np.object)], axis=1)
    vertices = np.array([tuple(el) for el in vertices],
                        dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                               ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
                               ('alpha', 'u1')])
    # ply file requires 1D object arrays.
    indices = np.array([tuple([el], ) for el in indices],
                       dtype=[('vertex_indices', 'i4', (3,))])
    PlyData([PlyElement.describe(vertices, 'vertex'),
             PlyElement.describe(indices, 'face')]).write(dest_path)


def make_ply_string_wocolor(dest_path, indices, vertices):
    """
    Creates a ply str that can be included into a .k.zip for rendering
    in KNOSSOS.
    # TODO: write out normals

    Parameters
    ----------
    indices : iterable of indices (int)
    vertices : iterable of vertices (int)

    Returns
    -------
    str
    """
    # create header
    if not indices.ndim == 2:
        indices = np.array(indices, dtype=np.int).reshape((-1, 3))
    if not vertices.ndim == 2:
        vertices = np.array(vertices, dtype=np.float32).reshape((-1, 3))
    vertices = np.array([tuple(el) for el in vertices], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    indices = np.array([tuple([el], ) for el in indices],dtype=[('vertex_indices', 'i4', (3,))])
    PlyData([PlyElement.describe(vertices, 'vertex'),
             PlyElement.describe(indices, 'face')]).write(dest_path)


def write_mesh2kzip(k_path, ind, vert, norm, color, ply_fname,
                    force_overwrite=False):
    """
    Writes mesh as .ply's to k.zip file.

    Parameters
    ----------
    k_path : str
        path to zip
    ind : np.array
    vert : np.array
    norm : np.array
    color : tuple or np.array
        rgba between 0 and 255
    ply_fname : str
    """
    if len(vert) == 0:
        log_proc.warn("'write_mesh2kzip' call with empty vertex array. Did not"
                      " write data to kzip.")
        return
    tmp_dest_p = '{}_{}'.format(k_path, ply_fname)
    if color is not None:
        make_ply_string(tmp_dest_p, ind, vert.astype(np.float32), color)
    else:
        make_ply_string_wocolor(tmp_dest_p, ind, vert.astype(np.float32))
    write_data2kzip(k_path, tmp_dest_p, ply_fname,
                    force_overwrite=force_overwrite)


def write_meshes2kzip(k_path, inds, verts, norms, colors, ply_fnames,
                      force_overwrite=True, verbose=True):
    """
    Writes meshes as .ply's to k.zip file.

    Parameters
    ----------
    k_path : str
        path to zip
    inds : list of np.array
    verts : list of np.array
    norms : list of np.array
    colors : list of tuple or np.array
        rgba between 0 and 255
    ply_fnames : list of str
    force_overwrite : bool
    verbose : bool
    """
    if not force_overwrite:
        raise NotImplementedError('Currently modification of data in existing kzip is not implemented.')
    tmp_paths = []
    if verbose:
        log_proc.info('Generating ply files.')
        pbar = tqdm.tqdm(total=len(inds))
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
            make_ply_string(tmp_dest_p, ind, vert.astype(np.float32), color)
        else:
            make_ply_string_wocolor(tmp_dest_p, ind, vert.astype(np.float32))
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


def compartmentalize_mesh(ssv, pred_key_appendix=""):
    """
    Splits SuperSegmentationObject mesh into axon, dendrite and soma. Based
    on axoness prediction of SV's contained in SuperSuperVoxel ssv.

    Parameters
    ----------
    sso : SuperSegmentationObject
    pred_key_appendix : str
        Specific version of axoness prediction

    Returns
    -------
    np.array
        Majority label of each face / triangle in mesh indices;
        triangulation is assumed. If majority class has n=1, majority label is
        set to -1.
    """
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
        comp_ind = np.array([remap_dict[i] for i in comp_ind], dtype=np.uint)
        comp_meshes[comp_type] = [comp_ind, comp_vert, comp_norm]
    return comp_meshes


def mesh_creator_sso(ssv):
    ssv.enable_locking = False
    ssv.load_attr_dict()
    _ = ssv._load_obj_mesh(obj_type="mi", rewrite=False)
    _ = ssv._load_obj_mesh(obj_type="sj", rewrite=False)
    _ = ssv._load_obj_mesh(obj_type="vc", rewrite=False)
    _ = ssv._load_obj_mesh(obj_type="sv", rewrite=False)
    try:
        ssv.attr_dict["conn"] = ssv.attr_dict["conn_ids"]
        _ = ssv._load_obj_mesh(obj_type="conn", rewrite=False)
    except KeyError:
        log_proc.error("Loading 'conn' objects failed for SSV %s."
              % ssv.id)
    ssv.clear_cache()


def mesh_chunk(args):
    scaling = get_dataset_scaling()
    attr_dir, obj_type = args
    ad = AttributeDict(attr_dir + "/attr_dict.pkl", disable_locking=True)
    obj_ixs = list(ad.keys())
    if len(obj_ixs) == 0:
        log_proc.warning("EMPTY ATTRIBUTE DICT", attr_dir)
        return
    voxel_dc = VoxelStorage(attr_dir + "/voxel.pkl", disable_locking=True)
    md = MeshStorage(attr_dir + "/mesh.pkl", disable_locking=True, read_only=False)
    valid_obj_types = ["vc", "sj", "mi", "con"]
    if not obj_type in valid_obj_types:
        raise NotImplementedError("Object type must be one of the following:\n"
                                  "%s" % str(valid_obj_types))
    for ix in obj_ixs:
        # create voxel_list
        bin_arrs, block_offsets = voxel_dc[ix]
        voxel_list = np.array([], dtype=np.int32)
        for i_bin_arr in range(len(bin_arrs)):
            block_voxels = np.array(zip(*np.nonzero(bin_arrs[i_bin_arr])),
                                    dtype=np.int32)
            block_voxels += np.array(block_offsets[i_bin_arr])

            if len(voxel_list) == 0:
                voxel_list = block_voxels
            else:
                voxel_list = np.concatenate([voxel_list, block_voxels])
        # create mesh
        indices, vertices, normals = triangulation(np.array(voxel_list),
                                     downsampling=MESH_DOWNSAMPLING[obj_type],
                                     n_closings=MESH_CLOSING[obj_type])
        vertices *= scaling
        md[ix] = [indices.flatten(), vertices.flatten(), normals.flatten()]
    md.push()


def mesh2obj_file(dest_path, mesh, color=None, center=None):
    """
    Writes mesh to .obj file.

    Parameters
    ----------
    mesh : List[np.array]
     flattend arrays of indices (triangle faces), vertices and normals
    center : np.array


    Returns
    -------

    """
    options = openmesh.Options()
    options += openmesh.Options.Binary
    mesh_obj = openmesh.TriMesh()
    ind, vert, norm = mesh
    if vert.ndim == 1:
        vert = vert.reshape(-1 ,3)
    if ind.ndim == 1:
        ind = ind.reshape(-1 ,3)
    if center is not None:
        vert -= center
    vert_openmesh = []
    if color is not None:
        mesh_obj.request_vertex_colors()
        options += openmesh.Options.VertexColor
        if color.ndim == 1:
            color = np.array([color] * len(vert))
        color = color.astype(np.float64)  # required by openmesh
    for ii, v in enumerate(vert):
        v = v.astype(np.float64)  # Point requires double
        v_openmesh = mesh_obj.add_vertex(openmesh.TriMesh.Point(v[0], v[1], v[2]))
        if color is not None:
            mesh_obj.set_color(v_openmesh, openmesh.TriMesh.Color(*color[ii]))
        vert_openmesh.append(v_openmesh)
    for f in ind:
        f_openmesh = [vert_openmesh[f[0]], vert_openmesh[f[1]],
                      vert_openmesh[f[2]]]
        mesh_obj.add_face(f_openmesh)
    result = openmesh.write_mesh(mesh_obj, dest_path, options)
    if not result:
        log_proc.error("Error occured when writing mesh to .obj file.")
