# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Philipp Schubert, Sven Dorkenwald, Joergen Kornfeld

from typing import Union, List, Callable, Optional, Tuple, TYPE_CHECKING, Iterable

from . import log_proc
from .meshes import MeshObject, calc_rot_matrices
from .. import global_params
from ..backend.storage import CompressedStorage
from ..handler.basics import flatten_list
from ..handler.compression import arrtolz4string
from ..handler.multiviews import generate_palette, remap_rgb_labelviews, \
    rgb2id_array, rgba2id_array, id2rgba_array_contiguous
from ..mp.mp_utils import start_multiprocess_imap

if TYPE_CHECKING:
    from ..reps.super_segmentation import SuperSegmentationObject
    from ..reps.segmentation import SegmentationObject
import time
import os
import sys
import numpy as np


def load_rendering_func(func_name: str) -> Callable:
    # can't load more than one platform simultaneously
    os.environ['PYOPENGL_PLATFORM'] = global_params.config['pyopengl_platform']
    if global_params.config['pyopengl_platform'] == 'egl':
        try:
            try:
                import OpenGL.EGL
            except AttributeError:  # OSMesa has been enabled before
                # hacky, but successfully removes all OpenGL related imports
                for k in list(sys.modules.keys()):
                    if 'OpenGL' in k:
                        del sys.modules[k]
                import OpenGL.EGL
            from ..proc import rendering_egl as rendering_module
        except ImportError as e:
            os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
            global_params.config['pyopengl_platform'] = 'osmesa'
            try:
                import OpenGL.osmesa
            except AttributeError:  # OSMesa has been enabled before
                # hacky, but successfully removes all OpenGL related imports
                for k in list(sys.modules.keys()):
                    if 'OpenGL' in k:
                        del sys.modules[k]
                import OpenGL.osmesa
            from ..proc import rendering_osmesa as rendering_module
            log_proc.error('EGL requirements could not be imported ({}). '
                           'Switched to OSMESA platform.'.format(e))
    elif global_params.config['pyopengl_platform'] == 'osmesa':
        try:
            import OpenGL.osmesa
        except AttributeError:  # OSMesa has been enabled before
            # hacky, but successfully removes all OpenGL related imports
            for k in list(sys.modules.keys()):
                if 'OpenGL' in k:
                    del sys.modules[k]
            import OpenGL.osmesa
        from ..proc import rendering_osmesa as rendering_module
    else:
        msg = 'PYOpenGL environment has to be "egl" or "osmesa".'
        log_proc.error(msg)
        raise NotImplementedError(msg)
    return getattr(rendering_module, func_name)


def render_mesh(mo: MeshObject, **kwargs) -> np.ndarray:
    """
    Render super voxel raw views located at randomly chosen center of masses in
    vertex cloud.

    Args:
        mo: Mesh.
        **kwargs: Keyword arguments pass to :py:func:`~multi_view_mesh` call.

    Returns:
        View array.

    """
    multi_view_mesh = load_rendering_func('multi_view_mesh')
    if "physical_scale" in kwargs.keys():
        kwargs["physical_scale"] = kwargs["physical_scale"] / mo.max_dist
    mo_views = multi_view_mesh(mo.indices, mo.vertices, mo.normals,
                               colors=mo.colors, **kwargs)
    return mo_views


def render_mesh_coords(coords: np.ndarray, ind: np.ndarray, vert: np.ndarray,
                       **kwargs) -> np.ndarray:
    """
    Render raw views located at given coordinates in mesh
    Returns ViewContainer list if dest_dir is None, else writes
    views to dest_path.

    Args:
        coords: Rendering locations.
        ind: Mesh indices/faces [N, 1]
        vert: Mesh vertices [M, 1]
        **kwargs: Keyword arguments passed to :py:func:`_render_mesh_coords`.

    Returns: Views at each coordinate.

    """
    _render_mesh_coords = load_rendering_func('_render_mesh_coords')
    mesh = MeshObject("views", ind, vert)
    mesh._colors = None  # this enables backwards compatibility, check why this was used
    return _render_mesh_coords(coords, mesh, **kwargs)


# ------------------------------------------------------------------------------
# SSO rendering code
def render_sampled_sso(sso: 'SuperSegmentationObject', ws: Optional[Tuple[int, int]] = None, verbose: bool = False,
                       woglia: bool = True, return_rot_mat: bool = False, overwrite: bool = True,
                       add_cellobjects: Optional[Union[bool, Iterable[str]]] = None, index_views: bool = False,
                       return_views: bool = False, cellobjects_only: bool = False, rot_mat: Optional[np.ndarray] = None,
                       view_key: Optional[str] = None) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray], None]:
    """
    Renders for each SV views at sampled locations (number is dependent on
    SV mesh size with scaling fact) from combined mesh of all SV.

    Args:
        sso: SuperSegmentationObject
        ws: Window size in pixels (y, x). Default: See config.yml or custom configs in the working directory.
        verbose: Log additional information.
        woglia: Store views with "without glia" identifier, i.e. flag the views as being created after the glia
            separation.
        return_rot_mat: Return rotation matrices.
        add_cellobjects: Default: ('sj', 'vc', 'mi')
        overwrite: If True, do not skip existing views.
        index_views: Also render index views.
        return_views: Return view arrays.
        cellobjects_only: Only render cell objects.
        rot_mat: Rotation matrix array for every rendering location [N, 4, 4].
        view_key: String identifier for storing view arrays. Only needed if ``return_views=False``.

    Returns:
        Depending on `return_views` and `return_rot_mat`: None; View array;
        View array and rotation matrices; rotation matrices.
    """
    view_cfg = global_params.config['views']
    view_props_default = view_cfg['view_properties']
    if ws is None:
        ws = view_props_default['ws']
    if add_cellobjects is None or add_cellobjects is True:
        add_cellobjects = view_cfg['subcell_objects']
        if view_cfg['use_onthefly_views'] and 'sj' in add_cellobjects:
            add_cellobjects[add_cellobjects.index('sj')] = 'syn_ssv'

    # get coordinates for N SV's in SSO
    coords = sso.sample_locations(cache=False)
    if not overwrite:
        missing_sv_ixs = ~np.array(sso.view_existence(
            woglia=woglia, index_views=index_views, view_key=view_key), dtype=np.bool)
        missing_svs = np.array(sso.svs)[missing_sv_ixs]
        coords = np.array(coords)[missing_sv_ixs]
    else:
        missing_svs = np.array(sso.svs)
    if len(missing_svs) == 0:
        if return_views:
            return sso.load_views(woglia=woglia)
        return
    # len(part_views) == N + 1
    part_views = np.cumsum([0] + [len(c) for c in coords])
    flat_coords = np.array(flatten_list(coords))
    if verbose:
        start = time.time()
    if index_views:
        views, rot_mat = render_sso_coords_index_views(
            sso, flat_coords, ws=ws, return_rot_matrices=True, verbose=verbose, rot_mat=rot_mat)
    else:
        views, rot_mat = render_sso_coords(
            sso, flat_coords, ws=ws, verbose=verbose, add_cellobjects=add_cellobjects,
            return_rot_mat=True, cellobjects_only=cellobjects_only, rot_mat=rot_mat)
    if verbose:
        dur = time.time() - start
        log_proc.debug("Rendering of %d views took %0.2fs. "
                       "%0.4fs/SV" % (len(views), dur, float(dur) / len(sso.svs)))
    if not return_views:
        if cellobjects_only:
            log_proc.warning('`cellobjects_only=True` in `render_sampled_sso` call, views '
                             'will be written to file system in serial (this is slow).')
            for i, so in enumerate(missing_svs):
                sv_views = views[part_views[i]:part_views[i + 1]]
                so.save_views(sv_views, woglia=woglia, cellobjects_only=cellobjects_only,
                              index_views=index_views, enable_locking=True, view_key=view_key)
        else:
            write_sv_views_chunked(missing_svs, views, part_views,
                                   dict(woglia=woglia, index_views=index_views, view_key=view_key))
    if return_views:
        if return_rot_mat:
            return views, rot_mat
        return views
    if return_rot_mat:
        return rot_mat


def render_sso_coords(sso: 'SuperSegmentationObject', coords: np.ndarray,
                      add_cellobjects: Optional[Union[bool, Iterable[str]]] = None,
                      verbose: bool = False, clahe: bool = False, ws: Optional[Tuple[int]] = None,
                      cellobjects_only: bool = False, wire_frame: bool = False, nb_views: Optional[int] = None,
                      comp_window: Optional[float] = None, rot_mat: Optional[np.ndarray] = None,
                      return_rot_mat: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Render views of SuperSegmentationObject at given coordinates.

    Args:
        sso: SuperSegmentationObject
        coords: Rendering locations [N, 3].
        add_cellobjects: Default: ('sj', 'vc', 'mi'). This ordering determines the channel order of the view array.
        verbose: Log additional information.
        clahe: Use clahe to enahnce view contrast.
        ws: Window size in pixels (y, x). Default: See config.yml or custom configs in the working directory.
        cellobjects_only: Only render cell objects.
        wire_frame: Render the mesh as a wire frame.
        nb_views: Number of views. Default: See config.yml or custom configs in the working directory.
        comp_window: Window size in nm. the clipping box during rendering will have an extent
            of [comp_window, comp_window / 2, comp_window]. Default: 8 um.
        rot_mat: Rotation matrix array for every rendering location [N, 4, 4].
        return_rot_mat: Return rotation matrices, e.g. if not provided via `rot_mat`, this output can be provided
            for other rendering calls.

    Returns:
        Resulting views rendered at each location. Output shape: [len(coords),
        4 (default: cell outline + number of cell objects), nb_views, y, x].

    """
    view_cfg = global_params.config['views']
    view_props_default = view_cfg['view_properties']
    if comp_window is None:
        comp_window = view_props_default['comp_window']
    if ws is None:
        ws = view_props_default['ws']
    if add_cellobjects is None or add_cellobjects is True:
        add_cellobjects = view_cfg['subcell_objects']
        if view_cfg['use_onthefly_views'] and 'sj' in add_cellobjects:
            add_cellobjects[add_cellobjects.index('sj')] = 'syn_ssv'

    if verbose:
        log_proc.debug('Started "render_sso_coords" at {} locations with sub-cellular structures {} for SSO {} using '
                       'PyOpenGL platform "{}".'.format(len(coords), add_cellobjects, sso,
                                                        global_params.config['pyopengl_platform']))
        start = time.time()
    if nb_views is None:
        nb_views = view_props_default['nb_views']
    mesh = sso.mesh
    if verbose:
        log_proc.debug(f'Loaded cell mesh after {(time.time() - start):.2f} s.')
    if cellobjects_only:
        assert len(add_cellobjects) > 0, "Add cellobjects must contain at least one entry " \
                                         "when rendering cellobjects only."
        raw_views = np.ones((len(coords), nb_views, ws[0], ws[1]), dtype=np.uint8) * 255
        if rot_mat is None:
            mo = MeshObject("raw", mesh[0], mesh[1])
            mo._colors = None
            querybox_edgelength = comp_window / mo.max_dist
            rot_mat = calc_rot_matrices(mo.transform_external_coords(coords),
                                        mo.vert_resh, querybox_edgelength,
                                        nb_cpus=sso.nb_cpus)
    else:
        if len(mesh[1]) == 0 or len(coords) == 0:
            raw_views = np.ones((len(coords), nb_views, ws[0], ws[1]), dtype=np.uint8) * 255
            msg = "No mesh for SSO {} found with {} locations.".format(sso, len(coords))
            log_proc.warning(msg)
        else:
            raw_views, rot_mat = render_mesh_coords(
                coords, mesh[0], mesh[1], clahe=clahe, verbose=verbose,
                return_rot_matrices=True, ws=ws, wire_frame=wire_frame,
                rot_matrices=rot_mat, nb_views=nb_views, comp_window=comp_window)
    if add_cellobjects is not False and len(add_cellobjects) > 0:
        if cellobjects_only:
            res = []
        else:
            res = [raw_views[:, None]]
        for subcell_obj in add_cellobjects:
            mesh = sso.load_mesh(subcell_obj)
            if len(mesh[1]) != 0:
                views = render_mesh_coords(
                    coords, mesh[0], mesh[1], clahe=clahe, verbose=verbose,
                    rot_matrices=rot_mat, views_key=subcell_obj, ws=ws, nb_views=nb_views,
                    wire_frame=wire_frame, comp_window=comp_window)
            else:
                views = np.ones_like(raw_views) * 255
            res.append(views[:, None])
        res = np.concatenate(res, axis=1)
        if return_rot_mat:
            return res, rot_mat
        return res
    if return_rot_mat:
        return raw_views[:, None], rot_mat
    return raw_views[:, None]


def render_sso_coords_index_views(sso: 'SuperSegmentationObject', coords: np.ndarray, verbose: bool = False,
                                  ws: Optional[Tuple[int, int]] = None, rot_mat: Optional[np.ndarray] = None,
                                  nb_views: Optional[int] = None, comp_window: Optional[float] = None,
                                  return_rot_matrices: bool = False
                                  ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Uses per-face color via flattened vertices (i.e. vert[ind] -> slow!). This was added to be able
    to calculate the surface coverage captured by the views.

    Todo:
        * Add fast GL_POINT rendering to omit slow per-face coloring (redundant vertices) and
          expensive remapping from face IDs to vertex IDs.

    Args:
        sso: SuperSegmentationObject
        coords: Rendering locations [N, 3].
        verbose: Log additional information.
        ws: Window size in pixels (y, x). Default: See config.yml or custom configs in the working directory.
        rot_mat: np.array
        nb_views: Number of views. Default: See config.yml or custom configs in the working directory.
        comp_window: Window size in nm. the clipping box during rendering will have an extent
            of [comp_window, comp_window / 2, comp_window]. Default: 8 um.
        return_rot_matrices: Return rotation matrices, e.g. if not provided via `rot_mat`, this output can be provided
            for other rendering calls.

    Returns:
        Resulting index views rendered at each location. Output shape: [len(coords), 1, nb_views, y, x].

    """
    view_props_default = global_params.config['views']['view_properties']
    _render_mesh_coords = load_rendering_func('_render_mesh_coords')
    if comp_window is None:
        comp_window = view_props_default['comp_window']
    if ws is None:
        ws = view_props_default['ws']
    if verbose:
        log_proc.debug('Started "render_sso_coords_index_views" at {} locations for SSO {} using '
                       'PyOpenGL platform "{}".'.format(len(coords), sso.id,
                                                        global_params.config['pyopengl_platform']))
    if nb_views is None:
        nb_views = view_props_default['nb_views']
    ind, vert, norm = sso.mesh
    if len(vert) == 0 or len(coords) == 0:
        msg = "No mesh for SSO {} found with {} locations.".format(sso, len(coords))
        log_proc.critical(msg)
        res = np.ones((len(coords), nb_views, ws[1], ws[0], 3), dtype=np.uint8) * 255
        if not return_rot_matrices:
            return res
        else:
            return np.zeros((len(coords), 16), dtype=np.uint8), res
    try:
        color_array = id2rgba_array_contiguous(np.arange(len(vert) // 3))
    except ValueError as e:
        msg = "'render_sso_coords_index_views' failed with {} when " \
              "rendering SSV {}.".format(e, sso.id)
        log_proc.error(msg)
        raise ValueError(msg)
    if color_array.shape[1] == 3:  # add alpha channel
        color_array = np.concatenate([color_array, np.ones((len(color_array), 1),
                                                           dtype=np.uint8) * 255], axis=-1)
    color_array = color_array.astype(np.float32) / 255.
    # in init it seems color values have to be normalized, check problems with uniqueness if
    # they are normalized between 0 and 1.. OR check if it is possible to just switch color arrays to UINT8 -> Check
    # backwards compatibility with other color-dependent rendering methods
    # Create mesh object without redundant vertices to get same PCA rotation as for raw views
    if rot_mat is None:
        mo = MeshObject("raw", ind, vert, color=color_array, normals=norm)
        querybox_edgelength = comp_window / mo.max_dist
        rot_mat = calc_rot_matrices(mo.transform_external_coords(coords), mo.vert_resh, querybox_edgelength,
                                    nb_cpus=sso.nb_cpus)
    # create redundant vertices to enable per-face colors
    # vert = vert.reshape(-1, 3)[ind].flatten()
    # ind = np.arange(len(vert) // 3)
    # color_array = np.repeat(color_array, 3, axis=0)  # 3 <- triangles
    mo = MeshObject("raw", ind, vert, color=color_array, normals=norm)
    if return_rot_matrices:
        ix_views, rot_mat = _render_mesh_coords(
            coords, mo, verbose=verbose, ws=ws, depth_map=False, rot_matrices=rot_mat, smooth_shade=False,
            views_key="index", nb_views=nb_views, comp_window=comp_window, return_rot_matrices=return_rot_matrices)
        if ix_views.shape[-1] == 3:  # rgba rendering
            ix_views = rgb2id_array(ix_views)[:, None]
        else:  # rgba rendering
            ix_views = rgba2id_array(ix_views)[:, None]
        return ix_views, rot_mat
    ix_views = _render_mesh_coords(coords, mo, verbose=verbose, ws=ws, depth_map=False, rot_matrices=rot_mat,
                                   smooth_shade=False, views_key="index", nb_views=nb_views, comp_window=comp_window,
                                   return_rot_matrices=return_rot_matrices)
    if ix_views.shape[-1] == 3:
        ix_views = rgb2id_array(ix_views)[:, None]
    else:
        ix_views = rgba2id_array(ix_views)[:, None]
    scnd_largest = np.partition(np.unique(ix_views.flatten()), -2)[-2]  # largest value is background
    if scnd_largest > len(vert) // 3:
        log_proc.critical('Critical error during index-rendering: Maximum vertex'
                          ' ID which was rendered is bigger than vertex array.'
                          '{}, {}; SSV ID {}'.format(scnd_largest, len(vert) // 3, sso.id))
    return ix_views


def render_sso_coords_label_views(sso: 'SuperSegmentationObject', vertex_labels: np.ndarray, coords: np.ndarray,
                                  verbose: bool = False, ws: Optional[Tuple[int, int]] = None,
                                  rot_mat: Optional[np.ndarray] = None, nb_views: Optional[int] = None,
                                  comp_window: Optional[float] = None, return_rot_matrices: bool = False
                                  ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Render views with vertex colors corresponding to vertex labels.

    Args:
        sso:
        vertex_labels: np.array
            vertex labels [N, 1]. Ordering and length have to be the same as
            vertex array of SuperSegmentationObject (len(sso.mesh[1]) // 3).
        coords: Rendering locations [N, 3].
        verbose: Log additional information.
        ws: Window size in pixels (y, x). Default: See config.yml or custom configs in the working directory.
        rot_mat: np.array
        nb_views: Number of views. Default: See config.yml or custom configs in the working directory.
        comp_window: Window size in nm. the clipping box during rendering will have an extent
            of [comp_window, comp_window / 2, comp_window]. Default: 8 um.
        return_rot_matrices: Return rotation matrices, e.g. if not provided via `rot_mat`, this output can be provided
            for other rendering calls.

    Returns:
        Resulting label views rendered at each location. Output shape: [len(coords), 1, nb_views, y, x].
    """
    view_props_default = global_params.config['views']['view_properties']
    _render_mesh_coords = load_rendering_func('_render_mesh_coords')
    if comp_window is None:
        comp_window = view_props_default['comp_window']
    if ws is None:
        ws = view_props_default['ws']
    if nb_views is None:
        nb_views = view_props_default['nb_views']
    ind, vert, _ = sso.mesh
    if len(vertex_labels) != len(vert) // 3:
        raise ValueError("Length of vertex labels and vertices have to be equal.")
    palette = generate_palette(len(np.unique(vertex_labels)))
    color_array = palette[vertex_labels].astype(np.float32) / 255
    mo = MeshObject("neuron", ind, vert, color=color_array)
    label_views, rot_mat = _render_mesh_coords(coords, mo, depth_map=False, ws=ws,
                                               rot_matrices=rot_mat, nb_views=nb_views,
                                               smooth_shade=False, verbose=verbose,
                                               comp_window=comp_window,
                                               return_rot_matrices=True)
    label_views = remap_rgb_labelviews(label_views, palette)[:, None]
    if return_rot_matrices:
        return label_views, rot_mat
    return label_views


def get_sso_view_dc(sso: 'SuperSegmentationObject', verbose: bool = False) -> dict:
    """
    Extracts views from sampled positions in SSO for each SV.

    Args:
        sso: SuperSegmentationObject
        verbose: bool

    Returns:
        Dictionary with `sso` id as key and lz4 compressed view array
        (see `:py:func:~syconn.handler.compression.arrtolz4string`).

    """
    views = render_sampled_sso(sso, verbose=verbose, return_views=True)
    view_dc = {sso.id: arrtolz4string(views)}
    return view_dc


def render_sso_ortho_views(sso: 'SuperSegmentationObject') -> np.ndarray:
    """
    Renders three views of SSO mesh.

    Args:
        sso: SuperSegmentationObject

    Returns:
        View array.

    """
    multi_view_sso = load_rendering_func('multi_view_sso')
    views = np.zeros((3, 4, 1024, 1024))
    # init MeshObject to calculate rotation into PCA frame
    views[:, 0] = multi_view_sso(sso, ws=(1024, 1024), depth_map=True,
                                 obj_to_render=('sv',), )
    views[:, 1] = multi_view_sso(sso, ws=(1024, 1024), depth_map=True,
                                 obj_to_render=('mi',))
    views[:, 2] = multi_view_sso(sso, ws=(1024, 1024), depth_map=True,
                                 obj_to_render=('vc',))
    views[:, 3] = multi_view_sso(sso, ws=(1024, 1024), depth_map=True,
                                 obj_to_render=('sj',))
    return views


def render_sso_coords_multiprocessing(ssv: 'SuperSegmentationObject', n_jobs: int,
                                      rendering_locations: Optional[np.ndarray] = None,
                                      verbose: bool = False, render_kwargs: Optional[dict] = None,
                                      view_key: Optional[str] = None, render_indexviews: bool = True,
                                      return_views: bool = True) -> Union[None, np.ndarray]:
    """

    Args:
        ssv: SuperSegmentationObject
        n_jobs: int
            number of parallel jobs running on same node of cluster
        rendering_locations: array of locations to be rendered
            if not given, rendering locations are retrieved from the SSV's SVs.
            Results will be stored at SV locations.
        verbose: bool
            flag to show the progress of rendering.
        render_kwargs: dict
        view_key: str
        render_indexviews: bool
        return_views: If False and rendering_locations is None, views will be saved on supervoxel level.

    Returns:
        Array of views after rendering of locations or None.

    """
    if rendering_locations is not None and return_views is False:
        raise ValueError('"render_sso_coords_multiprocessing" received invalid '
                         'parameters (`rendering_locations!=None` and `return_v'
                         'iews=False`). When using specific rendering locations, '
                         'views have to be returned-')
    svs = None
    ssv.nb_cpus = n_jobs
    if rendering_locations is None:  # use SV rendering locations
        svs = list(ssv.svs)
        if ssv._sample_locations is None and not ssv.attr_exists("sample_locations"):
            ssv.load_attr_dict()
        rendering_locations = ssv.sample_locations(cache=False)
        # store number of rendering locations per SV -> Ordering of rendered
        # views must be preserved!
        part_views = np.cumsum([0] + [len(c) for c in rendering_locations])
        rendering_locations = np.concatenate(rendering_locations)
    if len(rendering_locations) == 0:
        log_proc.critical('No rendering locations found for SSV {}.'.format(ssv.id))
        return np.ones((0, global_params.config['views']['view_properties']['nb_views'], 256, 128),
                       dtype=np.uint8) * 255
    params = np.array_split(rendering_locations, n_jobs)

    # TODO: refactor kwargs!
    render_kwargs_def = {'add_cellobjects': True, 'verbose': verbose, 'clahe': False,
                         'ws': None, 'cellobjects_only': False, 'wire_frame': False,
                         'nb_views': None, 'comp_window': None, 'rot_mat': None, 'woglia': True,
                         'return_rot_mat': False, 'render_indexviews': render_indexviews}
    if render_kwargs is not None:
        render_kwargs_def.update(render_kwargs)

    _ = ssv.mesh  # cache mesh
    params = [[par, ssv, render_kwargs_def] for par in params]
    views = start_multiprocess_imap(_render_views_multiproc, params)
    views = np.concatenate(views)
    if svs is not None and return_views is False:
        start_writing = time.time()
        if render_kwargs_def['cellobjects_only']:
            log_proc.warning('`cellobjects_only=True` in `render_sampled_sso` call, views '
                             'will be written to file system in serial (this is slow).')
            for i, so in enumerate(svs):
                so.enable_locking = True
                sv_views = views[part_views[i]:part_views[i + 1]]
                so.save_views(sv_views, woglia=render_kwargs_def['woglia'],
                              cellobjects_only=render_kwargs_def['cellobjects_only'],
                              index_views=render_kwargs_def["render_indexviews"],
                              enable_locking=True)
        else:
            write_sv_views_chunked(svs, views, part_views,
                                   dict(woglia=render_kwargs_def['woglia'],
                                        index_views=render_kwargs_def["render_indexviews"],
                                        view_key=view_key))
        end_writing = time.time()
        log_proc.debug('Writing SV renderings took {:.2f}s'.format(
            end_writing - start_writing))
        return
    return views


def _render_views_multiproc(args: tuple) -> np.ndarray:
    coords, sso, kwargs = args

    render_indexviews = kwargs['render_indexviews']
    del kwargs['render_indexviews']

    # TODO: refactor kwargs
    if 'overwrite' in kwargs:
        del kwargs['overwrite']
    if render_indexviews:
        if 'add_cellobjects' in kwargs:
            del kwargs['add_cellobjects']
        if 'clahe' in kwargs:
            del kwargs['clahe']
        if 'wire_frame' in kwargs:
            del kwargs['wire_frame']
        if 'cellobjects_only' in kwargs:
            del kwargs['cellobjects_only']
        if 'return_rot_mat' in kwargs:
            del kwargs['return_rot_mat']
        if 'woglia' in kwargs:
            del kwargs['woglia']
        views = render_sso_coords_index_views(sso, coords, **kwargs)
    else:
        if 'woglia' in kwargs:
            del kwargs['woglia']
        views = render_sso_coords(sso, coords, **kwargs)
    return views


def write_sv_views_chunked(svs: List['SegmentationObject'], views: np.ndarray, part_views: np.ndarray,
                           view_kwargs: dict, disable_locking: bool = False):
    """

    Args:
        svs: SegmentationObjects
        views: View array.
        part_views: Cumulated number of views -> indices of start and end of SV views in `views` array.
        view_kwargs:
        disable_locking: Usually required as SVs are stored in chunks and rendered distributed on many
            processes.

    Returns:

    """
    view_dc = {}
    for sv_ix, sv in enumerate(svs):
        curr_view_dest = sv.view_path(**view_kwargs)
        view_ixs = part_views[sv_ix], part_views[sv_ix + 1]
        if curr_view_dest in view_dc:
            view_dc[curr_view_dest][sv.id] = view_ixs
        else:
            view_dc[curr_view_dest] = {sv.id: view_ixs}
    for k, v in view_dc.items():
        view_storage = CompressedStorage(k, read_only=False,
                                         disable_locking=disable_locking)
        for sv_id, sv_view_ixs in v.items():
            view_storage[sv_id] = views[sv_view_ixs[0]:sv_view_ixs[1]]
        view_storage.push()
        del view_storage
