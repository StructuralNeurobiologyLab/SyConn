# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Sven Dorkenwald, Philipp Schubert, Joergen Kornfeld

import time
import os
import shutil
import glob
import numpy as np
import pickle as pkl
from importlib import reload
import sys

from ..mp.batchjob_utils import QSUB_script
from . import log_proc
from .. import global_params
from ..handler.basics import flatten_list
from ..handler.compression import arrtolz4string
from ..backend.storage import CompressedStorage
from ..handler.multiviews import generate_palette, remap_rgb_labelviews,\
    rgb2id_array, rgba2id_array, id2rgba_array_contiguous
from .meshes import MeshObject, calc_rot_matrices

__all__ = ['load_rendering_func', 'render_mesh', 'render_mesh_coords',
           'render_sso_coords_multiprocessing', 'render_sso_coords',
           'render_sampled_sso', 'render_sso_coords_generic',
           'render_sso_ortho_views', 'render_sso_coords_index_views',
           'render_sso_coords_label_views']


def load_rendering_func(func_name):
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
            log_proc.warn('EGL requirements could not be imported ({}). '
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


def render_mesh(mo, **kwargs):
    """
    Render super voxel raw views located at randomly chosen center of masses in
    vertice cloud.

    Args:
        mo: MeshObject
            Mesh
        **kwargs:

    Returns:

    """
    multi_view_mesh = load_rendering_func('multi_view_mesh')
    if "physical_scale" in kwargs.keys():
        kwargs["physical_scale"] = kwargs["physical_scale"] / mo.max_dist
    mo_views = multi_view_mesh(mo.indices, mo.vertices, mo.normals,
                               colors=mo.colors, **kwargs)
    return mo_views


def render_mesh_coords(coords, ind, vert, **kwargs):
    """
    Render raw views located at given coordinates in mesh
    Returns ViewContainer list if dest_dir is None, else writes
    views to dest_path.

    Args:
        coords: np.array
        ind: np.array [N, 1]
        vert: np.array [N, 1]
        **kwargs:

    Returns: numpy.array
        views at each coordinate

    """
    _render_mesh_coords = load_rendering_func('_render_mesh_coords')
    mesh = MeshObject("views", ind, vert)
    mesh._colors = None  # this enables backwards compatibility, check why this was used
    return _render_mesh_coords(coords, mesh, **kwargs)


# ------------------------------------------------------------------------------
# SSO rendering code

def render_sampled_sso(sso, ws=(256, 128), verbose=False, woglia=True, return_rot_mat=False,
                       add_cellobjects=True, overwrite=True, index_views=False,
                       return_views=False, cellobjects_only=False, rot_mat=None,
                       view_key=None):
    """
    Renders for each SV views at sampled locations (number is dependent on
    SV mesh size with scaling fact) from combined mesh of all SV.

    Args:
        sso: SuperSegmentationObject
        ws: tuple
        verbose: bool
        woglia: bool
            without glia
        return_rot_mat:
        add_cellobjects: bool
        overwrite: bool
        index_views: bool
        return_views: bool
        cellobjects_only: bool
        rot_mat: np.ndarray
        view_key: str

    Returns:

    """
    # get coordinates for N SV's in SSO
    coords = sso.sample_locations(cache=False)
    if not overwrite:
        missing_sv_ixs = ~np.array(sso.view_existence(
            woglia=woglia, index_views=index_views, view_key=view_key), dtype=np.bool)
        missing_svs = np.array(sso.svs)[missing_sv_ixs]
        coords = np.array(coords)[missing_sv_ixs]
        log_proc.debug("Rendering {}/{} missing SVs of SSV {}. {}".format(
            len(missing_svs), len(sso.sv_ids), sso.id,
            "(index views)" if index_views else ""))
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
                       "%0.4fs/SV" % (len(views), dur, float(dur)/len(sso.svs)))
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


def render_sso_coords(sso, coords, add_cellobjects=True, verbose=False, clahe=False,
                      ws=None, cellobjects_only=False, wire_frame=False, nb_views=None,
                      comp_window=None, rot_mat=None, return_rot_mat=False):
    """
    Render views of SuperSegmentationObject at given coordinates.

    Args:
        sso: SuperSegmentationObject
        coords: np.array
            N, 3
        add_cellobjects: bool
        verbose: bool
        clahe: bool
        ws: Optional[Tuple[int]]
            Window size in pixels (y, x). Default: (256, 128)
        cellobjects_only: bool
        wire_frame: bool
        nb_views: int
        comp_window: Optional[float]
            window size in nm. the clipping box during rendering will have an extent
            of [comp_window, comp_window / 2, comp_window]. Default: 8 um
        rot_mat: np.array
        return_rot_mat: bool

    Returns: np.ndarray
        Resulting views rendered at each location.
        Output shape: len(coords), 4 [cell outline + number of cell objects], nb_views, y, x

    """
    if comp_window is None:
        comp_window = 8e3
    if ws is None:
        ws = (256, 128)
    if verbose:
        log_proc.debug('Started "render_sso_coords" at {} locations for SSO {} using PyOpenGL'
                       ' platform "{}".'.format(
            len(coords), sso.id, global_params.config['pyopengl_platform']))
        start = time.time()
    if nb_views is None:
        nb_views = global_params.config['views']['nb_views']
    mesh = sso.mesh
    if verbose:
        log_proc.debug(f'Loaded cell mesh after {time.time() - start} s.')
    if cellobjects_only:
        assert add_cellobjects, "Add cellobjects must be True when rendering" \
                                "cellobjects only."
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
    if add_cellobjects:
        mesh = sso.mi_mesh
        if len(mesh[1]) != 0:
            mi_views = render_mesh_coords(
                coords, mesh[0], mesh[1], clahe=clahe, verbose=verbose,
                rot_matrices=rot_mat, views_key="mi", ws=ws, nb_views=nb_views,
                wire_frame=wire_frame, comp_window=comp_window)
        else:
            mi_views = np.ones_like(raw_views) * 255
        mesh = sso.vc_mesh
        if len(mesh[1]) != 0:
            vc_views = render_mesh_coords(
                coords, mesh[0], mesh[1], clahe=clahe, verbose=verbose,
                rot_matrices=rot_mat, views_key="vc", ws=ws, nb_views=nb_views,
                wire_frame=wire_frame, comp_window=comp_window)
        else:
            vc_views = np.ones_like(raw_views) * 255
        mesh = sso.sj_mesh
        if len(mesh[1]) != 0:
            sj_views = render_mesh_coords(
                coords, mesh[0], mesh[1], clahe=clahe, verbose=verbose,
                rot_matrices=rot_mat, views_key="sj", ws=ws,nb_views=nb_views,
                wire_frame=wire_frame, comp_window=comp_window)
        else:
            sj_views = np.ones_like(raw_views) * 255
        if cellobjects_only:
            res = np.concatenate([mi_views[:, None], vc_views[:, None],
                                  sj_views[:, None]], axis=1)
            if return_rot_mat:
                return res, rot_mat
            return res
        res = np.concatenate([raw_views[:, None], mi_views[:, None],
                              vc_views[:, None], sj_views[:, None]], axis=1)
        if return_rot_mat:
            return res, rot_mat
        return res
    if return_rot_mat:
        return raw_views[:, None], rot_mat
    return raw_views[:, None]


def render_sso_coords_index_views(sso, coords, verbose=False, ws=None,
                                  rot_mat=None, nb_views=None,
                                  comp_window=None, return_rot_matrices=False):
    """
    Uses per-face color via flattened vertices (i.e. vert[ind] -> slow!). This was added to be able
    to calculate the surface coverage captured by the views.
    TODO: Add fast GL_POINT rendering to omit slow per-face coloring (redundant vertices) and
    expensive remapping from face IDs to vertex IDs.

    Args:
        sso: SuperSegmentationObject
        coords: np.array
            N, 3
        verbose: bool
        ws:  Optional[Tuple[int]]
            Window size in pixels (y, x). Default: (256, 128)
        rot_mat: np.array
        nb_views: int
        comp_window: float
            window size in nm. the clipping box during rendering will have an extent
            of [comp_window, comp_window / 2, comp_window]
        return_rot_matrices:

    Returns: np.ndarray
        array of views after rendering of locations.

    """
    _render_mesh_coords = load_rendering_func('_render_mesh_coords')
    if comp_window is None:
        comp_window = 8e3
    if ws is None:
        ws = (256, 128)
    if verbose:
        log_proc.debug('Started "render_sso_coords_index_views" at {} locations for SSO {} using '
                       'PyOpenGL platform "{}".'.format(len(coords), sso.id,
                                                        global_params.config['pyopengl_platform']))
    if nb_views is None:
        nb_views = global_params.config['views']['nb_views']
    # tim = time.time()
    ind, vert, norm = sso.mesh
    # tim1 = time.time()
    # if verbose:
    #     print("Time for initialising MESH {:.2f}s."
    #                        "".format(tim1 - tim))
    if len(vert) == 0 or len(coords) == 0:
        msg = "No mesh for SSO {} found with {} locations.".format(sso, len(coords))
        log_proc.critical(msg)
        res = np.ones((len(coords), nb_views, ws[1], ws[0], 3), dtype=np.uint8) * 255
        if not return_rot_matrices:
            return res
        else:
            return np.zeros((len(coords), 16), dtype=np.uint8), res
    try:
        # color_array = id2rgba_array_contiguous(np.arange(len(ind) // 3))
        color_array = id2rgba_array_contiguous(np.arange(len(vert) // 3))
    except ValueError as e:
        msg = "'render_sso_coords_index_views' failed with {} when " \
              "rendering SSV {}.".format(e, sso.id)
        log_proc.error(msg)
        raise ValueError(msg)
    if color_array.shape[1] == 3:  # add alpha channel
        color_array = np.concatenate([color_array, np.ones((len(color_array), 1),
                                                           dtype=np.uint8)*255], axis=-1)
    color_array = color_array.astype(np.float32) / 255.
    # in init it seems color values have to be normalized, check problems with uniqueness if
    # they are normalized between 0 and 1.. OR check if it is possible to just switch color arrays to UINT8 -> Check
    # backwards compatibility with other color-dependent rendering methods
    # Create mesh object without redundant vertices to get same PCA rotation as for raw views
    if rot_mat is None:
        mo = MeshObject("raw", ind, vert, color=color_array, normals=norm)
        querybox_edgelength = comp_window / mo.max_dist
        rot_mat = calc_rot_matrices(mo.transform_external_coords(coords),
                                    mo.vert_resh, querybox_edgelength,
                                    nb_cpus=sso.nb_cpus)
    # create redundant vertices to enable per-face colors
    # vert = vert.reshape(-1, 3)[ind].flatten()
    # ind = np.arange(len(vert) // 3)
    # color_array = np.repeat(color_array, 3, axis=0)  # 3 <- triangles
    mo = MeshObject("raw", ind, vert, color=color_array, normals=norm)
    if return_rot_matrices:
        ix_views, rot_mat = _render_mesh_coords(
            coords, mo, verbose=verbose, ws=ws, depth_map=False,
            rot_matrices=rot_mat, smooth_shade=False, views_key="index",
            nb_views=nb_views, comp_window=comp_window,
            return_rot_matrices=return_rot_matrices)
        if ix_views.shape[-1] == 3:  # rgba rendering
            ix_views = rgb2id_array(ix_views)[:, None]
        else:  # rgba rendering
            ix_views = rgba2id_array(ix_views)[:, None]
        return ix_views, rot_mat
    ix_views = _render_mesh_coords(coords, mo, verbose=verbose, ws=ws,
                                   depth_map=False, rot_matrices=rot_mat,
                                   smooth_shade=False, views_key="index",
                                   nb_views=nb_views, comp_window=comp_window,
                                   return_rot_matrices=return_rot_matrices)
    if ix_views.shape[-1] == 3:
        ix_views = rgb2id_array(ix_views)[:, None]
    else:
        ix_views = rgba2id_array(ix_views)[:, None]
    scnd_largest = np.partition(np.unique(ix_views.flatten()), -2)[-2]  # largest value is background
    if scnd_largest > len(vert) // 3:
        log_proc.critical('Critical error during index-rendering: Maximum vertex'
                          ' ID which was rendered is bigger than vertex array.'
                          '{}, {}; SSV ID {}'.format(
            scnd_largest, len(vert) // 3, sso.id))
    return ix_views


def render_sso_coords_label_views(sso, vertex_labels, coords, verbose=False,
                                  ws=None, rot_mat=None, nb_views=None,
                                  comp_window=None, return_rot_matrices=False):
    """
    Render views with vertex colors corresponding to vertex labels.

    Args:
        sso:
        vertex_labels: np.array
            vertex labels [N, 1]. Ordering and length have to be the same as
            vertex array of SuperSegmentationObject (len(sso.mesh[1]) // 3).
        coords:
        verbose:
        ws:
        rot_mat:
        nb_views:
        comp_window:
        return_rot_matrices:

    Returns:

    """
    _render_mesh_coords = load_rendering_func('_render_mesh_coords')
    if comp_window is None:
        comp_window = 8e3
    if ws is None:
        ws = (256, 128)
    if nb_views is None:
        nb_views = global_params.config['views']['nb_views']
    ind, vert, _ = sso.mesh
    if len(vertex_labels) != len(vert) // 3:
        raise ValueError("Length of vertex labels and vertices "
                         "have to be equal.")
    palette = generate_palette(len(np.unique(vertex_labels)))
    color_array = palette[vertex_labels].astype(np.float32)/255
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


def get_sso_view_dc(sso, verbose=False):
    """
    Extracts views from sampled positions in SSO for each SV.

    Args:
        sso: SuperSegmentationObject
        verbose: bool

    Returns: dict

    """
    views = render_sampled_sso(sso, verbose=verbose, return_views=True)
    view_dc = {sso.id: arrtolz4string(views)}
    return view_dc


def render_sso_ortho_views(sso):
    """
    Renders three views of SSO mesh.

    Args:
        sso: SuperSegmentationObject

    Returns: np.ndarray

    """
    multi_view_sso = load_rendering_func('multi_view_sso')
    views = np.zeros((3, 4, 1024, 1024))
    # init MeshObject to calculate rotation into PCA frame
    views[:, 0] = multi_view_sso(sso, ws=(1024, 1024), depth_map=True,
                                 obj_to_render=('sv', ), )
    views[:, 1] = multi_view_sso(sso, ws=(1024, 1024), depth_map=True,
                                 obj_to_render=('mi', ))
    views[:, 2] = multi_view_sso(sso, ws=(1024, 1024), depth_map=True,
                                 obj_to_render=('vc', ))
    views[:, 3] = multi_view_sso(sso, ws=(1024, 1024), depth_map=True,
                                 obj_to_render=('sj', ))
    return views


def render_sso_coords_multiprocessing(ssv, wd, n_jobs, n_cores=1, rendering_locations=None,
                                      verbose=False, render_kwargs=None, view_key=None,
                                      render_indexviews=True, return_views=True,
                                      disable_batchjob=True):
    """

    Args:
        ssv: SuperSegmentationObject
        wd: string
            working directory for accessing data
        n_jobs: int
            number of parallel jobs running on same node of cluster
        n_cores: int
            Cores per job
        rendering_locations: array of locations to be rendered
            if not given, rendering locations are retrieved from the SSV's SVs.
            Results will be stored at SV locations.
        verbose: bool
            flag to show the progress of rendering.
        render_kwargs: dict
        view_key: str
        render_indexviews: bool
        return_views: bool
            if False and rendering_locations is None, views will be saved at
            SSV SVs
        disable_batchjob: bool

    Returns: np.ndarray
        array of views after rendering of locations.

    """
    if rendering_locations is not None and return_views is False:
        raise ValueError('"render_sso_coords_multiprocessing" received invalid '
                         'parameters (`rendering_locations!=None` and `return_v'
                         'iews=False`). When using specific rendering locations, '
                         'views have to be returned-')
    svs = None
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
        # TODO: adapt hard-coded window size (256, 128) as soon as those are available in
        #  `global_params`
        return np.ones((0, global_params.config['views']['nb_views'], 256, 128), dtype=np.uint8) * 255
    params = np.array_split(rendering_locations, n_jobs)

    ssv_id = ssv.id
    working_dir = wd
    sso_kwargs = {'ssv_id': ssv_id,
                  'working_dir': working_dir,
                  "version": ssv.version,
                  'nb_cpus': n_cores,
                  'sv_ids': [sv.id for sv in ssv.svs]}
    # TODOO: refactor kwargs!
    render_kwargs_def = {'add_cellobjects': True, 'verbose': verbose, 'clahe': False,
                      'ws': None, 'cellobjects_only': False, 'wire_frame': False,
                      'nb_views': None, 'comp_window': None, 'rot_mat': None, 'woglia': True,
                     'return_rot_mat': False, 'render_indexviews': render_indexviews}
    if render_kwargs is not None:
        render_kwargs_def.update(render_kwargs)

    params = [[par, sso_kwargs, render_kwargs_def, ix] for ix, par in
              enumerate(params)]
    # This is single node multiprocessing -> `disable_batchjob=False`
    path_to_out = QSUB_script(
        params, "render_views_multiproc", suffix="_SSV{}".format(ssv_id),
        n_cores=n_cores, disable_batchjob=disable_batchjob,
        n_max_co_processes=n_jobs,
        additional_flags="--gres=gpu:1" if not disable_batchjob else "")
    out_files = glob.glob(path_to_out + "/*")
    views = []
    out_files2 = np.sort(out_files, axis=-1, kind='quicksort', order=None)
    for out_file in out_files2:
        with open(out_file, 'rb') as f:
            views.append(pkl.load(f))
    views = np.concatenate(views)
    shutil.rmtree(os.path.abspath(path_to_out + "/../"), ignore_errors=True)
    if svs is not None and return_views is False:
        start_writing = time.time()
        if render_kwargs_def['cellobjects_only']:
            log_proc.warning('`cellobjects_only=True` in `render_sampled_sso` call, views '
                             'will be written to file system in serial (this is slow).')
            for i, so in enumerate(svs):
                so.enable_locking = True
                sv_views = views[part_views[i]:part_views[i+1]]
                so.save_views(sv_views, woglia=render_kwargs_def['woglia'],
                              cellobjects_only=render_kwargs_def['cellobjects_only'],
                              index_views=render_kwargs_def["render_indexviews"])
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


def write_sv_views_chunked(svs, views, part_views, view_kwargs):
    """

    Args:
        svs: List[SegmentationObject]
        views: np.ndarray
        part_views: np.ndarray[int]
            Cumulated number of views -> indices of start and end of SV views in `views` array
        view_kwargs: dict

    Returns:

    """
    view_dc = {}
    for sv_ix, sv in enumerate(svs):
        curr_view_dest = sv.view_path(**view_kwargs)
        view_ixs = part_views[sv_ix], part_views[sv_ix+1]
        if curr_view_dest in view_dc:
            view_dc[curr_view_dest][sv.id] = view_ixs
        else:
            view_dc[curr_view_dest] = {sv.id: view_ixs}
    for k, v in view_dc.items():
        view_storage = CompressedStorage(k, read_only=False)  # locking is enabled by default
        for sv_id, sv_view_ixs in v.items():
            view_storage[sv_id] = views[sv_view_ixs[0]:sv_view_ixs[1]]
        view_storage.push()
        del view_storage


def render_sso_coords_generic(ssv, working_dir, rendering_locations, n_jobs=None,
                              verbose=False, render_indexviews=True):
    """

    Args:
        ssv: SuperSegmentationObject
        working_dir: string
            working directory for accessing data
        rendering_locations: array of locations to be rendered
            if not given, rendering locations are retrieved from the SSV's SVs. Results will be stored at SV locations.
        n_jobs: int
            number of parallel jobs running on same node of cluster
        verbose: bool
            flag to show th progress of rendering.
        render_indexviews: Bool
            Flag to choose between render_index_view and render_sso_coords

    Returns: np.ndarray
        array of views after rendering of locations.

    """
    if n_jobs is None:
        n_jobs = global_params.config['ncores_per_node'] // 10

    if render_indexviews is False:
        if len(rendering_locations) > 360:
            views = render_sso_coords_multiprocessing(
                ssv, working_dir, rendering_locations=rendering_locations,
                n_jobs=n_jobs, verbose=verbose, render_indexviews=render_indexviews)
        else:
            views = render_sso_coords(ssv, rendering_locations, verbose=verbose)
    else:
        if len(rendering_locations) > 140:
            views = render_sso_coords_multiprocessing(
                ssv, working_dir, rendering_locations=rendering_locations,
                render_indexviews=render_indexviews, n_jobs=n_jobs, verbose=verbose)
        else:
            views = render_sso_coords_index_views(ssv, rendering_locations, verbose=verbose)
    return views
