# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Sven Dorkenwald, Philipp Schubert, Joergen Kornfeld
import os
import time
from ctypes import sizeof, c_float, c_void_p, c_uint

import numpy as np
import tqdm
from PIL import Image
from scipy.ndimage.filters import gaussian_filter

from . import log_proc
from .image import rgb2gray, apply_clahe
from .meshes import merge_meshes, MeshObject, calc_rot_matrices
from .. import global_params

if os.environ['PYOPENGL_PLATFORM'] != 'osmesa':
    raise EnvironmentError(f'PyOpenGL backened should be "osmesa". '
                           f'Found "{os.environ["PYOPENGL_PLATFORM"]}".')
import OpenGL

OpenGL.USE_ACCELERATE = True
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GL.framebufferobjects import *
from OpenGL.arrays import *
from OpenGL.osmesa import *

log_proc.info('OSMesa rendering enabled.')

__all__ = ['init_object', 'init_ctx', 'init_opengl', '_render_mesh_coords',
           'multi_view_mesh_coords', 'multi_view_mesh', 'multi_view_sso']

# ------------------------------------ General rendering code ------------------------------------------
# structure definition of rendering data array
float_size = sizeof(c_float)
vertex_offset = c_void_p(0 * float_size)
normal_offset = c_void_p(3 * float_size)
color_offset = c_void_p(6 * float_size)
record_len = 10 * float_size


def init_object(indices, vertices, normals, colors, ws):
    """
    This function initializes objects for rendering from N triangles and M vertices. It takes in
    indices, vertices, normals, colors and window size as arguments and doesn't return anything.
    
    Args:
        indices (array_like): A 2D array of shape [3N, 1].
        vertices (array_like): A 2D array of shape [3M, 1].
        normals (array_like): A 2D array of shape [3M, 1].
        colors (array_like): A 2D array of shape [4M, 1].
        ws (tuple): A tuple representing the window size.
    """
    global ind_cnt, vertex_cnt
    indices = indices.astype(np.uint32)
    # create individual vertices for each triangle
    vertices = vertices.reshape(-1, 3)
    # adapt color array
    colors = colors.reshape(-1, 4)
    ind_cnt = len(indices)
    vertex_cnt = len(vertices)
    normals = normals.reshape(-1, 3)
    data = np.concatenate((vertices, normals, colors),
                          axis=1).astype(np.float32).reshape(-1)
    del vertices, normals, colors
    # enabling arrays
    glEnableClientState(GL_VERTEX_ARRAY)
    glEnableClientState(GL_NORMAL_ARRAY)
    glEnableClientState(GL_COLOR_ARRAY)
    # model data
    indices_buffer = indices.ctypes.data_as(ctypes.POINTER(c_uint))
    data_buffer = data.ctypes.data_as(ctypes.POINTER(c_uint))

    el_arr_buffer = glGenBuffers(1)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, el_arr_buffer)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices_buffer, GL_STATIC_DRAW)
    # del indices_buffer

    el_arr_buffer2 = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, el_arr_buffer2)
    glBufferData(GL_ARRAY_BUFFER, data.nbytes, data_buffer, GL_STATIC_DRAW)
    # del data_buffer

    # rbo fbo for storing projections
    fbo = glGenFramebuffers(1)
    glBindFramebuffer(GL_FRAMEBUFFER, fbo)
    del fbo

    rbo = glGenRenderbuffers(1)
    glBindRenderbuffer(GL_RENDERBUFFER, rbo)
    glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA, ws[0],
                          ws[1])
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                              GL_RENDERBUFFER, rbo)
    del rbo


def draw_object(triangulation=True):
    """
    This function draws elements in the current buffer. It takes in a boolean argument 
    'triangulation' which is True by default.
    
    Args:
        triangulation (bool, optional): A boolean value indicating whether to draw elements in 
        current buffer. Defaults to True.
    """
    glVertexPointer(3, GL_FLOAT, record_len, vertex_offset)
    glNormalPointer(GL_FLOAT, record_len, normal_offset)
    glColorPointer(4, GL_FLOAT, record_len, color_offset)
    if triangulation is True:
        glDrawElements(GL_TRIANGLES, ind_cnt, GL_UNSIGNED_INT, None)
        # glDrawArrays(GL_TRIANGLES, 0, vertex_cnt)
    elif triangulation == "points":
        glDrawElements(GL_POINTS, ind_cnt, GL_UNSIGNED_INT, None)
    else:
        glDrawElements(GL_QUADS, ind_cnt, GL_UNSIGNED_INT, None)


def screen_shot(ws, colored=False, depth_map=False, clahe=False,
                triangulation=True, egl_args=None):
    """
    This function creates a screenshot of the currently opened window and returns it as an array.
    
    Args:
        ws (tuple): A tuple representing the window size.
        colored (bool, optional): A boolean value indicating whether the screenshot should be 
        colored. Defaults to False.
        depth_map (bool, optional): A boolean value indicating whether to create a depth map. 
        Defaults to False.
        clahe (bool, optional): A boolean value indicating whether to apply Contrast Limited 
        Adaptive Histogram Equalization (CLAHE). Defaults to False.
        triangulation (bool, optional): A boolean value indicating whether to apply triangulation. 
        Defaults to True.
        egl_args (bool, optional): Optional arguments if EGL platform is used. Defaults to None.
    
    Returns:
        np.array: An array representing the screenshot.
    """
    glBindFramebuffer(GL_FRAMEBUFFER, 0)
    draw_object(triangulation)
    glReadBuffer(GL_FRONT)

    if depth_map:
        data = glReadPixels(0, 0, ws[0], ws[1],
                            GL_DEPTH_COMPONENT, GL_UNSIGNED_BYTE)
        data = Image.frombuffer("L", (ws[0], ws[1]), data, 'raw', 'L', 0, 1)
        data = np.asarray(data.transpose(Image.FLIP_TOP_BOTTOM))
        data = gaussian_filter(data, .7)
        if clahe:
            data = apply_clahe(data)
        if np.sum(data) == 0:
            data = np.ones_like(data) * 255
    elif colored:
        data = glReadPixels(0, 0, ws[0], ws[1],
                            GL_RGBA, GL_UNSIGNED_BYTE)
        data = Image.frombuffer("RGBA", (ws[0], ws[1]), data, 'raw', 'RGBA', 0, 1)
        data = np.asarray(data.transpose(Image.FLIP_TOP_BOTTOM))
    else:
        data = glReadPixels(0, 0, ws[0], ws[1],
                            GL_RGB, GL_UNSIGNED_BYTE)
        data = Image.frombuffer("RGB", (ws[0], ws[1]), data, 'raw', 'RGB', 0, 1)
        data = rgb2gray(np.asarray(data.transpose(Image.FLIP_TOP_BOTTOM))) * 255
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    return data


def init_ctx(ws, depth_map):
    """
    This function initializes the rendering context. It takes in a tuple representing the window 
    size and a boolean value indicating whether to create a depth map as arguments.
    
    Args:
        ws (tuple): A tuple representing the window size.
        depth_map (bool): A boolean value indicating whether to create a depth map.
    """
    # rendering
    ctx = OSMesaCreateContextExt(OSMESA_RGBA, 32, 0, 0, None)
    buf = arrays.GLubyteArray.zeros((ws[0], ws[1], 4)) + 1
    assert (OSMesaMakeCurrent(ctx, buf, GL_UNSIGNED_BYTE, ws[0], ws[1]))
    assert (OSMesaGetCurrentContext())
    OSMesaPixelStore(OSMESA_Y_UP, 0)
    return ctx


def init_opengl(ws, enable_lightning=False, clear_value=None, depth_map=False,
                smooth_shade=True, wire_frame=False):
    """
    This function initializes OpenGL settings. It takes in a tuple representing the window size, 
    boolean values indicating whether to enable lightning, create a depth map, apply smooth shading 
    and wire frame, and a float value representing the clear value as arguments.
    
    Args:
        ws (tuple): A tuple representing the window size.
        enable_lightning (bool, optional): A boolean value indicating whether to enable lightning. 
        Defaults to False.
        clear_value (float, optional): A float value representing the clear value. Defaults to None.
        depth_map (bool, optional): A boolean value indicating whether to create a depth map. 
        Defaults to False.
        smooth_shade (bool, optional): A boolean value indicating whether to apply smooth shading. 
        Defaults to True.
        wire_frame (bool, optional): A boolean value indicating whether to apply wire frame. 
        Defaults to False.
    """
    glEnable(GL_NORMALIZE)
    if enable_lightning:
        glEnable(GL_BLEND)
        glEnable(GL_POLYGON_SMOOTH)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glHint(GL_POLYGON_SMOOTH_HINT, GL_NICEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [.7, .7, .7, 1.0])
    glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE)
    if wire_frame:
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        depth_map = False
    if not depth_map:
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT, GL_AMBIENT)
    # objects in the foreground will be visible in the projection
    glEnable(GL_DEPTH_TEST)
    if smooth_shade:
        glShadeModel(GL_SMOOTH)
    else:
        glShadeModel(GL_FLAT)
    glViewport(0, 0, ws[0], ws[1])
    if clear_value is None:
        glClearColor(0., 0., 0., 0.)
    else:
        glClearColor(clear_value, clear_value, clear_value, clear_value)  # alpha changed from 0
        # to clear_value, PS 12Apr2019
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)


def multi_view_mesh(indices, vertices, normals, colors=None, alpha=None,
                    ws=(2048, 2048), physical_scale=None,
                    enable_lightning=False, depth_map=False,
                    nb_views=3, background=None):  # Mariana Sh added background function
    """
    Renders a mesh from multiple perspectives. The perspectives are equidistant from each other.
    
    Args:
        indices (array_like): Array of indices for the vertices of the mesh.
        vertices (array_like): Array of vertices that define the mesh.
        normals (array_like): Array of normals for the vertices of the mesh.
        colors (array_like, optional): Array of colors for the vertices of the mesh. Defaults to None.
        alpha (float, optional): Alpha value for the colors of the vertices. Defaults to None.
        ws (tuple, optional): Window size for the output images. Defaults to (2048, 2048).
        physical_scale (float, optional): Physical scale for the rendering. Defaults to None.
        enable_lightning (bool, optional): If True, enables lightning effects. Defaults to False.
        depth_map (bool, optional): If True, renders a depth map. Defaults to False.
        nb_views (int, optional): Number of views to render. Two views parallel to main component, 
            and N-2 views (evenly spaced in angle space) perpendicular to it. Defaults to 3.
        background (float, optional): Background color value between 0 and 1. Used as RGB values. 
            Defaults to None.
    
    Returns:
        np.array: Array of rendered views. Shape is (nb_views, ws[0], ws[1]).
    """
    ctx = init_ctx(ws, depth_map=depth_map)
    init_opengl(ws, enable_lightning, depth_map=depth_map, clear_value=background)
    vertices = np.array(vertices)
    indices = np.array(indices, dtype=np.uint64)
    if colors is not None:
        colored = True
        colors = np.array(colors)
    else:
        colored = False
        colors = np.ones(len(vertices) // 3 * 4) * 0.2
    if alpha is not None:
        colors[::4] = alpha
    c_views = []
    init_object(indices, vertices, normals, colors, ws)

    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glOrtho(-1, 1, 1, -1, -1, 1)
    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    if enable_lightning:
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [.7, .7, .7, 1.0])
        glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE)
    if physical_scale is not None:
        draw_scale(physical_scale)
    c_views.append(screen_shot(ws, colored, depth_map=depth_map)[None,])
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)
    for m in range(1, nb_views):
        if physical_scale is not None:
            draw_scale(physical_scale)
        glPushMatrix()
        glRotate(360. / nb_views * m, 1, 0, 0)
        if enable_lightning:
            glLightfv(GL_LIGHT0, GL_DIFFUSE, [.7, .7, .7, 1.0])
            glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE)
        c_views.append(screen_shot(ws, colored, depth_map=depth_map)[None,])
        glPopMatrix()
    OSMesaDestroyContext(ctx)
    return np.concatenate(c_views)


def multi_view_sso(sso, colors=None, obj_to_render=('sv',),
                   ws=(2048, 2048), physical_scale=None,
                   enable_lightning=True, depth_map=False,
                   nb_views=3, background=1, rot_mat=None,
                   triangulation=True):
    """
    Renders a SuperSegmentationObject from multiple perspectives. The perspectives are rotated around the
    first principle component. The angle between adjacent views is 360°/nb_views.
    
    Args:
        sso (SuperSegmentationObject): The SuperSegmentationObject to render.
        colors (dict, optional): Dictionary of colors for the objects to render. Defaults to None.
        obj_to_render (tuple, optional): Tuple of strings indicating the cell objects to render. Defaults to ('sv',).
        ws (tuple, optional): Window size for the output images. Defaults to (2048, 2048).
        physical_scale (float, optional): Physical scale for the rendering. Defaults to None.
        enable_lightning (bool, optional): If True, enables lightning effects. Defaults to True.
        depth_map (bool, optional): If True, renders a depth map. Defaults to False.
        nb_views (int, optional): Number of views to render. Two views are parallel to main component, and 
            N-2 views (evenly spaced in angle space) are perpendicular to it. Defaults to 3.
        background (int, optional): Background color value between 0 and 1. Used as RGB values. Defaults to 1.
        rot_mat (np.array, optional): 4x4 rotation matrix. Defaults to None.
        triangulation (bool, optional): If True, enables triangulation. Defaults to True.
    
    Returns:
        np.array: Array of rendered views.
    """
    if colors is not None:
        assert type(colors) == dict
    else:
        colors = {"sv": None, "mi": None, "vc": None, "sj": None}
    ctx = init_ctx(ws, depth_map=depth_map)
    init_opengl(ws, enable_lightning, depth_map=depth_map,
                clear_value=background)
    # initially loading mesh is still needed to get bounding box...
    sv_mesh = MeshObject("sv", sso.mesh[0], sso.mesh[1], sso.mesh[2],
                         colors["sv"])
    c_views = []
    norm, col = np.zeros(0, ), np.zeros(0, )
    ind, vert = [], []
    for object_type in ['vc', 'mi', 'sj', 'sv']:
        if not object_type in obj_to_render:
            continue
        curr_ind, curr_vert, curr_norm = sso.load_mesh(object_type)
        if len(curr_vert) == 0:
            continue
        m = MeshObject(object_type, curr_ind, curr_vert, curr_norm,
                       colors[object_type], sv_mesh.bounding_box)
        norm = np.concatenate([norm, m.normals])
        col = np.concatenate([col, m.colors])
        ind.append(m.indices)
        vert.append(m.vertices)
    ind, vert = merge_meshes(ind, vert)
    init_object(ind, vert, norm, col, ws)

    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glOrtho(-1, 1, 1, -1, -1, 1)
    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    if enable_lightning:
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [.7, .7, .7, 1.0])
        glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE)
    if physical_scale is not None:
        draw_scale(physical_scale)
    c_views.append(screen_shot(ws, True, depth_map=depth_map,
                               triangulation=triangulation)[None,])
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)
    for m in range(1, nb_views):
        if physical_scale is not None:
            draw_scale(physical_scale)
        glPushMatrix()
        glRotate(360. / nb_views * m, 1, 0, 0)
        if rot_mat is not None:
            glMultMatrixf(rot_mat)
        if enable_lightning:
            glLightfv(GL_LIGHT0, GL_DIFFUSE, [.7, .7, .7, 1.0])
            glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE)
        c_views.append(screen_shot(ws, True, depth_map=depth_map,
                                   triangulation=triangulation)[None,])
        glPopMatrix()
    OSMesaDestroyContext(ctx)
    return np.concatenate(c_views)


def multi_view_mesh_coords(mesh, coords, rot_matrices, edge_lengths, alpha=None,
                           ws=None, views_key="raw", nb_simplices=3,
                           depth_map=True, clahe=False, smooth_shade=True,
                           verbose=False, wire_frame=False, egl_args=None,
                           nb_views=None, triangulation=True):
    """
    Renders a mesh object from multiple viewpoints without creating a gl context. 
    
    Args:
        mesh (MeshObject): The mesh object to be rendered.
        coords (np.array): An array of shape [N, 3] that must correspond to rot_matrices.
        rot_matrices (np.array): Rotation matrices for each view in ViewContainer list vc_list.
        edge_lengths (np.array): Spatial extent for sub-volumes.
        alpha (float, optional): Alpha value for the mesh. Defaults to None.
        ws (tuple, optional): Window size used for rendering (resolution of array being 
            stored/saved). Defaults to None.
        views_key (str, optional): Key for the views. Defaults to "raw".
        nb_simplices (int, optional): Number of simplices used for meshes. Defaults to 3.
        depth_map (bool, optional): If True, render views as depth, else render without light 
            effects (binary). Defaults to True.
        clahe (bool, optional): If True, apply clahe to screenshot. Defaults to False.
        smooth_shade (bool, optional): If True, apply smooth shading. Defaults to True.
        verbose (bool, optional): If True, print verbose messages. Defaults to False.
        wire_frame (bool, optional): If True, render in wireframe mode. Defaults to False.
        egl_args (tuple, optional): Optional arguments if EGL platform is used. Defaults to None.
        nb_views (int, optional): Number of views. Defaults to None.
        triangulation (bool, optional): If True, apply triangulation. Defaults to True.
    
    Returns:
        np.array: Array of views, else None.
    """
    egl_args = None
    view_props_default = global_params.config['views']['view_properties']
    if nb_views is None:
        nb_views = view_props_default['nb_views']
    if ws is None:
        ws = view_props_default['ws']
    # center data
    assert isinstance(edge_lengths, np.ndarray)
    assert nb_simplices in [3, 4]
    vertices = mesh.vertices
    indices = mesh.indices
    colors = mesh.colors
    if mesh._normals is None:
        normals = np.zeros(len(vertices))
    else:
        normals = mesh.normals
    edge_lengths = edge_lengths / mesh.max_dist
    # default color
    if colors is not None and not depth_map:
        colored = True
        colors = np.array(colors)
    else:
        colored = False
        colors = np.ones(len(vertices) // 3 * 4) * 0.8
    if alpha is not None:
        colors[::4] = alpha
    if not colored:
        view_sh = (nb_views, ws[1], ws[0])
    else:
        view_sh = (nb_views, ws[1], ws[0], 4)
    res = np.ones([len(coords)] + list(view_sh), dtype=np.uint8) * 255
    init_opengl(ws, depth_map=depth_map, clear_value=1.0,
                smooth_shade=smooth_shade, wire_frame=wire_frame)
    init_object(indices, vertices, normals, colors, ws)
    n_empty_views = 0
    if verbose:
        pbar = tqdm.tqdm(total=len(res), mininterval=0.5, leave=False)
    for ii, c in enumerate(coords):
        c_views = np.ones(view_sh, dtype=np.float32)
        rot_mat = rot_matrices[ii]
        if np.sum(np.abs(rot_mat)) == 0 or np.sum(np.abs(mesh.vertices)) == 0:
            if views_key in ["raw", "index"]:
                log_proc.warning(
                    "Rotation matrix or vertices of '%s' with %d vertices is"
                    " zero during rendering at %s. Skipping."
                    % (views_key, len(mesh.vert_resh), str(c)))
            continue
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(-edge_lengths[0] / 2, edge_lengths[0] / 2, edge_lengths[1] / 2,
                -edge_lengths[1] / 2, -edge_lengths[2] / 2, edge_lengths[2] / 2)
        glMatrixMode(GL_MODELVIEW)

        transformed_c = mesh.transform_external_coords([c])[0]
        # dummy rendering, somehow first projection is always black
        _ = screen_shot(ws, colored=colored, depth_map=depth_map, clahe=clahe,
                        triangulation=triangulation, egl_args=egl_args)

        glMatrixMode(GL_MODELVIEW)
        for m in range(0, nb_views):
            if nb_views == 2:
                rot_angle = (-1) ** (m + 1) * 25  # views are orthogonal
            else:
                rot_angle = 360. / nb_views * m  # views are equi-angular
            glPushMatrix()
            glRotate(rot_angle, edge_lengths[0], 0, 0)
            glMultMatrixf(rot_mat)
            glTranslate(-transformed_c[0], -transformed_c[1], -transformed_c[2])
            light_position = [1., 1., 2., 0.]
            glLightfv(GL_LIGHT0, GL_POSITION, light_position)
            glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE)
            c_views[m] = screen_shot(ws, colored=colored, depth_map=depth_map,
                                     clahe=clahe, triangulation=triangulation,
                                     egl_args=egl_args)
            glPopMatrix()
        res[ii] = c_views
        if verbose:
            pbar.update(1)
        for cv in c_views:
            if views_key == "raw" or views_key == "index":
                if len(np.unique(cv)) == 1:
                    n_empty_views += 1
                    continue  # check at most one occurrence
    if n_empty_views / len(res) > 0.5:  # more than 10% locations contain at least one empty view
        log_proc.critical(
            "WARNING: Found {}/{} locations with empty views.\t'{}'-mesh with "
            "{} vertices. Example location: {}".format(n_empty_views, len(coords), views_key,
                                                       len(mesh.vertices), repr(c)))
    if verbose:
        pbar.close()
    return res


def draw_scale(size):
    """
    Draws a black bar of a given length with a fixed width.
    
    Args:
        size (float): The length of the bar to be drawn.
    
    Returns:
        None
    """
    glLineWidth(5)
    glBegin(GL_LINES)
    glColor(0, 0, 0, 1)
    glVertex2f(1 - 0.1 - size, 1 - 0.1)
    glVertex2f(1 - 0.1, 1 - 0.1)
    glEnd()


def _render_mesh_coords(coords, mesh, clahe=False, verbose=False, ws=None,
                        rot_matrices=None, views_key="raw",
                        return_rot_matrices=False, depth_map=True,
                        smooth_shade=True, wire_frame=False, nb_views=None,
                        triangulation=True, comp_window=8e3):
    """
    Renders raw views located at given coordinates in a mesh. Returns a ViewContainer list if 
    dest_dir is None, else writes views to dest_path.
    
    Args:
        coords (np.array): The coordinates at which to render the views.
        mesh (MeshObject): The mesh object to be rendered.
        clahe (bool, optional): If True, apply clahe to screenshot. Defaults to False.
        verbose (bool, optional): If True, print verbose messages. Defaults to False.
        ws (tuple, optional): Window size. Defaults to None.
        rot_matrices (np.array, optional): Rotation matrices. Defaults to None.
        views_key (str, optional): Key for the views. Defaults to "raw".
        return_rot_matrices (bool, optional): If True, return rotation matrices. Defaults to False.
        depth_map (bool, optional): If True, render views as depth. Defaults to True.
        smooth_shade (bool, optional): If True, apply smooth shading. Defaults to True.
        wire_frame (bool, optional): If True, render in wireframe mode. Defaults to False.
        nb_views (int, optional): Number of views. Defaults to None.
        triangulation (bool, optional): If True, apply triangulation. Defaults to True.
        comp_window (float, optional): Window length in NM along main p.c. for mesh view. Defaults to 8e3.
    
    Returns:
        np.array: Array of views at each coordinate.
    """
    view_props_default = global_params.config['views']['view_properties']
    if nb_views is None:
        nb_views = view_props_default['nb_views']
    if ws is None:
        ws = view_props_default['ws']
    if np.isscalar(comp_window):
        # TODO: consider using comp_window / 2 for z dimension again
        edge_lengths = np.array([comp_window, comp_window / 2, comp_window])
    else:
        edge_lengths = comp_window
    if verbose:
        start = time.time()
    querybox_edgelength = np.max(edge_lengths)
    if rot_matrices is None:
        rot_matrices = calc_rot_matrices(mesh.transform_external_coords(coords),
                                         mesh.vert_resh,
                                         querybox_edgelength / mesh.max_dist)
        if verbose:
            log_proc.debug("Calculation of rotation matrices took {:.2f}s."
                           "".format(time.time() - start))
    if verbose:
        log_proc.debug("Started local rendering at %d locations (%s)." %
                       (len(coords), views_key))
    ctx = init_ctx(ws, depth_map=depth_map)
    mviews = multi_view_mesh_coords(mesh, coords, rot_matrices, edge_lengths,
                                    clahe=clahe, views_key=views_key, ws=ws,
                                    depth_map=depth_map, verbose=verbose,
                                    smooth_shade=smooth_shade,
                                    triangulation=triangulation, egl_args=ctx,
                                    wire_frame=wire_frame, nb_views=nb_views)
    if verbose:
        end = time.time()
        log_proc.debug("Finished rendering mesh of type %s at %d locations after"
                       " %0.2fs" % (views_key, len(mviews), end - start))
    OSMesaDestroyContext(ctx)
    if return_rot_matrices:
        return mviews, rot_matrices
    return mviews
