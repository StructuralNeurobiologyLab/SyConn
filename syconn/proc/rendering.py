# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Sven Dorkenwald, Philipp Schubert, Joergen Kornfeld

from ctypes import sizeof, c_float, c_void_p, c_uint
from PIL import Image
import numpy as np
import shutil
import time
import os
import tqdm
import warnings
import glob
from scipy.ndimage.filters import gaussian_filter

from .image import rgb2gray, apply_clahe
from . import log_proc
from .. import global_params
from ..handler.basics import flatten_list, chunkify_successive, chunkify_successive_split
from ..handler.compression import arrtolz4string
from ..handler.multiviews import generate_palette, remap_rgb_labelviews,\
    rgb2id_array, id2rgb_array_contiguous, rgba2id_array, id2rgba_array_contiguous
from .meshes import merge_meshes, MeshObject, calc_rot_matrices
try:
    import os
    os.environ['PYOPENGL_PLATFORM'] = global_params.PYOPENGL_PLATFORM
    import OpenGL
    OpenGL.USE_ACCELERATE = True  # unclear behavior
    from OpenGL.GL import *
    from OpenGL.GLU import *
    from OpenGL.GL.framebufferobjects import *
    from OpenGL.arrays import *
except Exception as e:
    log_proc.error("Problem loading OpenGL: {}".format(e))
    pass

# can't load more than one platform simultaneously
if os.environ['PYOPENGL_PLATFORM'] == 'egl':
    try:
        from OpenGL.EGL import eglDestroyContext, eglSwapBuffers
        from OpenGL.EGL import EGL_SURFACE_TYPE, EGL_PBUFFER_BIT, EGL_BLUE_SIZE, \
            EGL_RED_SIZE, EGL_GREEN_SIZE, EGL_DEPTH_SIZE, \
            EGL_COLOR_BUFFER_TYPE, EGL_LUMINANCE_BUFFER, EGL_HEIGHT, \
            EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT, EGL_CONFORMANT, \
            EGL_OPENGL_BIT, EGL_CONFIG_CAVEAT, EGL_NONE, \
            EGL_DEFAULT_DISPLAY, EGL_NO_CONTEXT, EGL_WIDTH, \
            EGL_OPENGL_API, EGL_LUMINANCE_SIZE, EGL_NO_DISPLAY,\
            eglGetDisplay, eglInitialize, eglChooseConfig, \
            eglBindAPI, eglCreatePbufferSurface, EGL_ALPHA_SIZE,\
            eglCreateContext, eglMakeCurrent, EGLConfig, EGL_RGB_BUFFER
        log_proc.info('EGL rendering enabled.')
    except ImportError as e:
        os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
        from OpenGL.osmesa import *
        log_proc.warn('EGL requirements could not be imported ({}). '
                      'Switched to OSMESA platform.'.format(e))
elif os.environ['PYOPENGL_PLATFORM'] == 'osmesa':
    log_proc.info('OSMESA rendering enabled.')
    from OpenGL.osmesa import *
else:
    msg = 'PYOpenGL environment has to be "egl" or "osmesa".'
    log_proc.error(msg)
    raise NotImplementedError(msg)
import numpy as np
from ..mp.batchjob_utils import QSUB_script
try:
    import cPickle as pkl
except ImportError:
    import pickle as pkl


from .egl_ext import eglQueryDevicesEXT
MULTIGPU = True

# ------------------------------------ General rendering code ------------------------------------------
# structure definition of rendering data array
float_size = sizeof(c_float)
vertex_offset = c_void_p(0 * float_size)
normal_offset = c_void_p(3 * float_size)
color_offset = c_void_p(6 * float_size)
record_len = 10 * float_size


def init_object(indices, vertices, normals, colors, ws):
    """
    Initialize objects for rendering from N triangles and M vertices

    Parameters
    ----------
    indices : array_like
        [3N, 1]
    vertices : array_like
        [3M, 1]
    normals : array_like
        [3M, 1]
    colors : array_like
        [4M, 1]
    ws : tuple

    Returns
    -------

    """
    global ind_cnt, vertex_cnt
    indices = indices.astype(np.uint32)
    # create individual vertices for each triangle
    vertices = vertices.astype(np.float32).reshape(-1, 3)
    # adapt color array
    colors = colors.reshape(-1, 4)
    ind_cnt = len(indices)
    vertex_cnt = len(vertices)
    normals = normals.astype(np.float32).reshape(-1, 3)
    data = np.concatenate((vertices, normals, colors),
                          axis=1).reshape(-1)
    # enabling arrays
    glEnableClientState(GL_VERTEX_ARRAY)
    glEnableClientState(GL_NORMAL_ARRAY)
    glEnableClientState(GL_COLOR_ARRAY)
    # model data
    indices_buffer = (c_uint * len(indices))(*indices)
    data_buffer = (c_float * len(data))(*data)

    el_arr_buffer = glGenBuffers(1)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, el_arr_buffer)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices_buffer, GL_STATIC_DRAW)

    el_arr_buffer2 = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, el_arr_buffer2)
    glBufferData(GL_ARRAY_BUFFER, data_buffer, GL_STATIC_DRAW)

    # rbo fbo for screenshooting
    fbo = glGenFramebuffers(1)
    glBindFramebuffer(GL_FRAMEBUFFER, fbo)
    rbo = glGenRenderbuffers(1)
    glBindRenderbuffer(GL_RENDERBUFFER, rbo)
    glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA, ws[0],
                          ws[1])
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                              GL_RENDERBUFFER, rbo)
    del rbo
    del fbo
    del indices_buffer
    del data_buffer
    # return [el_arr_buffer, el_arr_buffer2]


def draw_object(triangulation=True):
    """
    Draw mesh.
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
    Create screenshot of currently opened window and return as array.

    Parameters
    ----------
    ws : tuple
    colored : bool
    depth_map : bool
    clahe : bool
    triangulation : bool

    Returns
    -------
    np.array
    """
    glBindFramebuffer(GL_FRAMEBUFFER, 0)
    draw_object(triangulation)
    if egl_args is None:
        glReadBuffer(GL_FRONT)
    else:
        eglSwapBuffers(egl_args[0], egl_args[2])

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
    # ctx = OSMesaCreateContext(OSMESA_RGBA, None)
    if os.environ['PYOPENGL_PLATFORM'] == 'egl':
        from OpenGL.EGL import EGL_SURFACE_TYPE, EGL_PBUFFER_BIT, EGL_BLUE_SIZE, \
            EGL_RED_SIZE, EGL_GREEN_SIZE, EGL_DEPTH_SIZE, \
            EGL_COLOR_BUFFER_TYPE, EGL_LUMINANCE_BUFFER, EGL_HEIGHT, \
            EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT, EGL_CONFORMANT, \
            EGL_OPENGL_BIT, EGL_CONFIG_CAVEAT, EGL_NONE, \
            EGL_DEFAULT_DISPLAY, EGL_NO_CONTEXT, EGL_WIDTH, \
            EGL_OPENGL_API, EGL_LUMINANCE_SIZE, EGL_NO_DISPLAY, EGL_TRUE, \
            eglGetDisplay, eglInitialize, eglChooseConfig, \
            eglBindAPI, eglCreatePbufferSurface, \
            eglCreateContext, eglMakeCurrent, EGLConfig, EGL_RGB_BUFFER

        major, minor = ctypes.c_long(), ctypes.c_long()
        num_configs = ctypes.c_long()
        configs = (EGLConfig * 1)()

        if (MULTIGPU):
            dev_on_node = eglQueryDevicesEXT()
            for i in range(0, len(dev_on_node)):
                dsp = eglGetDisplay(dev_on_node[i])
                try:
                    initialized = eglInitialize(dsp, major, minor)
                    if (initialized == EGL_TRUE):
                        break
                except:
                    pass
        else:
            dsp = eglGetDisplay(EGL_DEFAULT_DISPLAY)

            assert dsp != EGL_NO_DISPLAY, 'Invalid DISPLAY during egl init.'

        # Initialize EGL
            eglInitialize(dsp, major, minor)
        if depth_map:
            config_attr = arrays.GLintArray.asArray(
                [EGL_SURFACE_TYPE, EGL_PBUFFER_BIT, EGL_BLUE_SIZE, 8, EGL_RED_SIZE, 8,
                 EGL_GREEN_SIZE, 8,
                 EGL_DEPTH_SIZE, 8, EGL_COLOR_BUFFER_TYPE, #EGL_LUMINANCE_BUFFER,
                 EGL_RGB_BUFFER,
                 EGL_RENDERABLE_TYPE,
                 EGL_OPENGL_BIT,  EGL_NONE])   #EGL_CONFORMANT, EGL_OPENGL_BIT,
        else:
            config_attr = arrays.GLintArray.asArray(
                [EGL_SURFACE_TYPE, EGL_PBUFFER_BIT, EGL_BLUE_SIZE, 8, EGL_RED_SIZE, 8,
                 EGL_GREEN_SIZE, 8, EGL_ALPHA_SIZE, 8,
                 EGL_DEPTH_SIZE, 8, EGL_COLOR_BUFFER_TYPE,  # EGL_LUMINANCE_BUFFER,
                 EGL_RGB_BUFFER,
                 EGL_RENDERABLE_TYPE,
                 EGL_OPENGL_BIT, EGL_NONE])  # EGL_CONFORMANT, EGL_OPENGL_BIT,
        eglChooseConfig(dsp, config_attr, configs, 1, num_configs)

        # Bind EGL to the OpenGL API
        eglBindAPI(EGL_OPENGL_API)
        #attrbls = [major, minor,  EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT, EGL_TRUE, EGL_FALSE, EGL_FALSE,
         #          EGL_NO_RESET_NOTIFICATION]
        attrbls = None

        # Create an EGL context
        ctx = eglCreateContext(dsp, configs[0], EGL_NO_CONTEXT, attrbls)

        # Create an EGL pbuffer
        buf = eglCreatePbufferSurface(dsp, configs[0], [EGL_WIDTH, ws[0], EGL_HEIGHT, ws[1], EGL_NONE])
        # Make the EGL context current
        assert (eglMakeCurrent(dsp, buf, buf, ctx))
        ctx = [dsp, ctx, buf]
    elif os.environ['PYOPENGL_PLATFORM'] == 'osmesa':  # TODO: might be optimizable for depth map
        # rendering
        ctx = OSMesaCreateContextExt(OSMESA_RGBA, 32, 0, 0, None)
        buf = arrays.GLubyteArray.zeros((ws[0], ws[1], 4)) + 1
        assert (OSMesaMakeCurrent(ctx, buf, GL_UNSIGNED_BYTE, ws[0], ws[1]))
        assert (OSMesaGetCurrentContext())
        OSMesaPixelStore(OSMESA_Y_UP, 0)
    else:
        raise NotImplementedError('PYOpenGL environment has to be "egl" or "osmesa".')
    return ctx


def init_opengl(ws, enable_lightning=False, clear_value=None, depth_map=False,
                smooth_shade=True, wire_frame=False):
    """
    Initialize OpenGL settings.

    Parameters
    ----------
    ws : tuple
    enable_lightning : bool
    clear_value : float
    depth_map : bool

    Returns
    -------

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
    Render mesh from 3 (default) equidistant perspectives.

    Parameters
    ----------
    indices :
    vertices :
    normals :
    colors:
    alpha :
    nb_simplices :
    ws :
    physical_scale :
    enable_lightning :
    depth_map :
    nb_views : int
        two views parallel to main component, and N-2 views (evenly spaced in
        angle space) perpendicular to it.
    background
        float value for background (clear value) between 0 and 1 (used as RGB
        values)

    Returns
    -------
    np.array
        shape: (nb_views, ws[0], ws[1]
    """
    ctx = init_ctx(ws, depth_map=depth_map)
    init_opengl(ws, enable_lightning, depth_map=depth_map, clear_value=background)
    vertices = np.array(vertices)
    indices = np.array(indices, dtype=np.uint)
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
    c_views.append(screen_shot(ws, colored, depth_map=depth_map)[None, ])
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
        c_views.append(screen_shot(ws, colored, depth_map=depth_map)[None, ])
        glPopMatrix()
    # glFinish()
    if os.environ['PYOPENGL_PLATFORM'] == 'egl':
        eglDestroyContext(*ctx)
    else:
        OSMesaDestroyContext(ctx)
    return np.concatenate(c_views)


def multi_view_sso(sso, colors=None, obj_to_render=('sv',),
                   ws=(2048, 2048), physical_scale=None,
                   enable_lightning=True, depth_map=False,
                   nb_views=3, background=1, rot_mat=None,
                   triangulation=True):
    """
    Render mesh from nb_views (default: 3) perspectives rotated around the
     first principle component (angle between adjacent views is 360°/nb_views)

    Parameters
    ----------
    colors: dict
    save_skeleton : tuple of str
        cell objects to render (e.g. 'mi', 'sj', 'vc', ..)
    alpha :
    ws : tuple
        window size of output images (width, height)
    physical_scale :
    enable_lightning :
    depth_map :
    nb_views : int
        two views parallel to main component, and N-2 views (evenly spaced in
        angle space) perpendicular to it.
    background : int
        float value for background (clear value) between 0 and 1 (used as RGB
        values)
    rot_mat : np.array
        4 x 4 rotation matrix

    Returns
    -------

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
                               triangulation=triangulation)[None, ])
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
                                   triangulation=triangulation)[None, ])
        glPopMatrix()
    # glFinish()
    if os.environ['PYOPENGL_PLATFORM'] == 'egl':
        eglDestroyContext(*ctx)
    else:
        OSMesaDestroyContext(ctx)
    return np.concatenate(c_views)


def multi_view_mesh_coords(mesh, coords, rot_matrices, edge_lengths, alpha=None,
                           ws=(256, 128), views_key="raw", nb_simplices=3,
                           depth_map=True, clahe=False, smooth_shade=True,
                           verbose=False, wire_frame=False, egl_args=None,
                           nb_views=None, triangulation=True):
    """
    Same as multi_view_mesh_coords but without creating gl context.
    Parameters
    ----------
    mesh : MeshObject
    rot_matrices : np.array
        Rotation matrices for each view in ViewContainer list vc_list
    coords : np.array
        [N, 3], must correspond to rot_matrices
    edge_lengths : np.array
        Spatial extent for sub-volumes
    alpha : float
    views_key : str
    nb_simplices : int
        Number of simplices used for meshes
    ws : tuple of ints
        Window size used for rendering (resolution of array being stored/saved)
    depth_map : bool
        Render views as depth, else render without light effects (binary)
    clahe : bool
        apply clahe to screenshot
    wire_frame : bool

    Returns
    -------
    np.array
        Returns array of views, else None
    """
    if os.environ['PYOPENGL_PLATFORM'] != 'egl':
        egl_args = None
    if nb_views is None:
        nb_views = global_params.NB_VIEWS
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
    if verbose:
        pbar = tqdm.tqdm(total=len(res), mininterval=0.5)
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
        glOrtho(-edge_lengths[0]/2, edge_lengths[0]/2, edge_lengths[1]/2,
                -edge_lengths[1]/2, -edge_lengths[2]/2, edge_lengths[2]/2)
        glMatrixMode(GL_MODELVIEW)

        transformed_c = mesh.transform_external_coords([c])[0]
        # dummy rendering, somehow first screenshot is always black
        _ = screen_shot(ws, colored=colored, depth_map=depth_map, clahe=clahe,
                        triangulation=triangulation, egl_args=egl_args)
        # glPopMatrix()

        glMatrixMode(GL_MODELVIEW)
        for m in range(0, nb_views):
            if nb_views == 2:
                rot_angle = (-1)**(m+1)*25# views are orthogonal
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
        found_empty_view = False
        for cv in c_views:
            if len(np.unique(cv)) == 1:
                if views_key == "raw" or views_key == "index":
                    log_proc.critical("WARNING: Empty view of '{}'-mesh with {} vertices found. "
                                     "Existing color value: {}".format(views_key, len(mesh.vert_resh), np.unique(cv)))
                    found_empty_view = True
        if found_empty_view:
            log_proc.critical(
                "WARNING: View 1: %0.1f\t View 2: %0.1f\t#view in list: %d/%d\n"
                  "'%s'-mesh with %d vertices. Location: %s" %
                  (np.sum(c_views[0]), np.sum(c_views[1]), ii, len(coords),
                   views_key, len(mesh.vertices), repr(c)))
    if verbose:
        pbar.close()
    return res


def draw_scale(size):
    """
    Draws black bar of given length with fixed width.

    Parameters
    ----------
    size : float
    """
    glLineWidth(5)
    glBegin(GL_LINES)
    glColor(0, 0, 0, 1)
    glVertex2f(1 - 0.1 - size, 1 - 0.1)
    glVertex2f(1 - 0.1, 1 - 0.1)
    glEnd()


def render_mesh(mo, **kwargs):
    """
    Render super voxel raw views located at randomly chosen center of masses in
    vertice cloud.

    Parameters
    ----------
    mo : MeshObject
        Mesh
    """
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

    Parameters
    ----------
    coords : np.array
    ind : np.array [N, 1]
    vert : np.array [N, 1]

    Returns
    -------
    numpy.array
        views at each coordinate
    """
    mesh = MeshObject("views", ind, vert)
    mesh._colors = None  # this enables backwards compatibility, check why this was used
    return _render_mesh_coords(coords, mesh, **kwargs)


def _render_mesh_coords(coords, mesh, clahe=False, verbose=False, ws=(256, 128),
                        rot_matrices=None, views_key="raw",
                        return_rot_matrices=False, depth_map=True,
                        smooth_shade=True, wire_frame=False, nb_views=None,
                        triangulation=True, comp_window=8e3):
    """
    Render raw views located at given coordinates in mesh
     Returns ViewContainer list if dest_dir is None, else writes
    views to dest_path.

    Parameters
    ----------
    coords : np.array
    mesh : MeshObject
    clahe : bool
    verbose : bool
    ws : tuple
        Window size
    rot_matrices : np.array
    views_key : str
    return_rot_matrices : bool
    depth_map : bool
    wire_frame : bool
    comp_window : float
        window length in NM along main p.c. for mesh view

    Returns
    -------
    numpy.array
        views at each coordinate
    """
    if nb_views is None:
        nb_views = global_params.NB_VIEWS
    if np.isscalar(comp_window):
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
        log_proc.debug("Starting local rendering at %d locations (%s)." %
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
                       " %0.2fs" % (views_key,len(mviews), end - start))
    if os.environ['PYOPENGL_PLATFORM'] == 'egl':
        eglDestroyContext(*ctx)
    else:
        OSMesaDestroyContext(ctx)
    if return_rot_matrices:
        return mviews, rot_matrices
    return mviews


# ------------------------------------------------------------------------------
# SSO rendering code

def render_sampled_sso(sso, ws=(256, 128), verbose=False, woglia=True,
                       add_cellobjects=True, overwrite=True, index_views=False,
                       return_views=False, cellobjects_only=False):
    """

    Renders for each SV views at sampled locations (number is dependent on
    SV mesh size with scaling fact) from combined mesh of all SV.

    Parameters
    ----------
    sso : SuperSegmentationObject
    ws : tuple
    verbose : bool
    clahe : bool
    add_cellobjects : bool
    cellobjects_only : bool
    woglia : bool
        without glia
    index_views : bool
    overwrite : bool
    return_views : bool
    cellobjects_only : bool
    """
    # get coordinates for N SV's in SSO
    coords = sso.sample_locations(cache=False)
    if not overwrite:
        missing_sv_ixs = np.array([not so.views_exist(woglia=woglia)
                                   for so in sso.svs],
                                  dtype=np.bool)
        missing_svs = np.array(sso.svs)[missing_sv_ixs]
        coords = np.array(coords)[missing_sv_ixs]
        log_proc.info("Rendering %d/%d missing SVs of SSV %d."
                      % (len(missing_svs), len(sso.sv_ids), sso.id))
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
        views = render_sso_coords_index_views(sso, flat_coords, ws=ws,
                                              verbose=verbose)
    else:
        views = render_sso_coords(sso, flat_coords, ws=ws, verbose=verbose,
                                  add_cellobjects=add_cellobjects,
                                  cellobjects_only=cellobjects_only)
    if verbose:
        dur = time.time() - start
        log_proc.debug("Rendering of %d views took %0.2fs. "
                       "%0.4fs/SV" % (len(views), dur, float(dur)/len(sso.svs)))
    # if sso.version != 'tmp':
    if not return_views:
        for i, so in enumerate(missing_svs):
            sv_views = views[part_views[i]:part_views[i+1]]
            so.enable_locking = True
            so.save_views(sv_views, woglia=woglia, cellobjects_only=cellobjects_only,
                          index_views=index_views)
    # else:
    #     log_proc.warning('"render_sampled_sso" called but this SSV '
    #                      'has version "tmp", results will'
    #                      ' not be saved to disk.')
    if return_views:
        return views


def render_sso_coords(sso, coords, add_cellobjects=True, verbose=False, clahe=False,
                      ws=None, cellobjects_only=False, wire_frame=False,
                      nb_views=None, comp_window=None, rot_mat=None, return_rot_mat=False):
    """
    Render views of SuperSegmentationObject at given coordinates.

    Parameters
    ----------
    sso : SuperSegmentationObject
    coords : np.array
        N, 3
    add_cellobjects : bool
    verbose : bool
    clahe : bool
    ws : Optional[Tuple[int]]
        Window size in pixels (y, x). Default: (256, 128)
    cellobjects_only : bool
    wire_frame : bool
    nb_views : int
    comp_window : Optional[float]
        window size in nm. the clipping box during rendering will have an extent
         of [comp_window, comp_window / 2, comp_window]. Default: 8 um
    rot_mat : np.array
    return_rot_mat : bool

    Returns
    -------
    np.ndarray
        Resulting views rendered at each location.
        Output shape: len(coords), 4 [cell outline + number of cell objects], nb_views, y, x
    """
    if comp_window is None:
        comp_window = 8e3
    if ws is None:
        ws = (256, 128)
    if verbose:
        log_proc.debug('Started "render_sso_coords" at {} locations for SSO {} using PyOpenGL'
                       ' platform "{}".'.format(len(coords), sso.id, os.environ['PYOPENGL_PLATFORM']))
    if nb_views is None:
        nb_views = global_params.NB_VIEWS
    mesh = sso.mesh
    if cellobjects_only:
        assert add_cellobjects, "Add cellobjects must be True when rendering" \
                                "cellobjects only."
        raw_views = np.ones((len(coords), nb_views, ws[0], ws[1]), dtype=np.uint8) * 255
        if rot_mat is None:
            mo = MeshObject("raw", mesh[0], mesh[1])
            mo._colors = None
            querybox_edgelength = comp_window / mo.max_dist
            rot_mat = calc_rot_matrices(mo.transform_external_coords(coords),
                                        mo.vert_resh, querybox_edgelength)
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

    Parameters
    ----------
    sso : SuperSegmentationObject
    coords : np.array
        N, 3
    rot_mat :
    comp_window : float
        window size in nm. the clipping box during rendering will have an extent
         of [comp_window, comp_window / 2, comp_window]
    return_rot_matrices : bool
    add_cellobjects : bool
    verbose : bool
    ws : Optional[Tuple[int]]
        Window size in pixels (y, x). Default: (256, 128)
    rot_mat : np.array
    nb_views : int
    comp_window : Optional[float]
        window size in nm. the clipping box during rendering will have an extent
         of [comp_window, comp_window / 2, comp_window]. Default: 8 um
    return_rot_mat : bool

    Returns
    -------
    np.ndarray
        array of views after rendering of locations.
    -------
    """
    if comp_window is None:
        comp_window = 8e3
    if ws is None:
        ws = (256, 128)
    if verbose:
        log_proc.debug('Started "render_sso_coords_index_views" at {} locations for SSO {} using '
                  'PyOpenGL'
                       ' platform "{}".'.format(len(coords), sso.id, os.environ['PYOPENGL_PLATFORM']))
    if nb_views is None:
        nb_views = global_params.NB_VIEWS
    # tim = time.time()
    ind, vert, norm = sso.mesh
    # tim1 = time.time()
    # if verbose:
    #     print("Time for initialising MESH {:.2f}s."
    #                        "".format(tim1 - tim))
    if len(vert) == 0:
        msg = "No mesh for SSO {} found with {} locations.".format(sso, len(coords))
        log_proc.warning(msg)
        return np.ones((len(coords), nb_views, ws[1], ws[0], 3), dtype=np.uint8)
    try:
        color_array = id2rgba_array_contiguous(np.arange(len(ind) // 3))
    except ValueError as e:
        msg = "'render_sso_coords_index_views' failed with {} when " \
              "rendering SSV {}.".format(e, sso.id)
        log_proc.error(msg)
        raise ValueError(msg)
    if color_array.shape[1] == 3:  # add alpha channel
        color_array = np.concatenate([color_array, np.ones((len(color_array), 1),
                                                           dtype=np.uint8)*255],
                                     axis=-1)
    color_array = color_array.astype(np.float32) / 255.
    # in init it seems color values have to be normalized, check problems with uniqueness if
    # they are normalized between 0 and 1.. OR check if it is possible to just switch color arrays to UINT8 -> Check
    # backwards compatibility with other color-dependent rendering methods
    # Create mesh object without redundant vertices to get same PCA rotation as for raw views
    # tim = time.time()
    if rot_mat is None:
        mo = MeshObject("raw", ind, vert, color=color_array, normals=norm)
        querybox_edgelength = comp_window / mo.max_dist
        rot_mat = calc_rot_matrices(mo.transform_external_coords(coords),
                                    mo.vert_resh, querybox_edgelength)
    # create redundant vertices to enable per-face colors
    vert = vert.reshape(-1, 3)[ind].flatten()
    ind = np.arange(len(vert) // 3)
    color_array = np.repeat(color_array, 3, axis=0)  # 3 <- triangles
    mo = MeshObject("raw", ind, vert, color=color_array, normals=norm)
    # tim1 = time.time()
    # print("Time for initializing MESHOBJECT {:.2f}s."
    #                    "".format(tim1 - tim))
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
    # tim2 = time.time()
    # print("Time for _RENDER_MESH_COORDS {:.2f}s."
    #                    "".format(tim2 - tim1))
    if ix_views.shape[-1] == 3:
        ix_views = rgb2id_array(ix_views)[:, None]
    else:
        ix_views = rgba2id_array(ix_views)[:, None]
    return ix_views


def render_sso_coords_label_views(sso, vertex_labels, coords, verbose=False,
                                  ws=None, rot_mat=None, nb_views=None,
                                  comp_window=None, return_rot_matrices=False):
    """
    Render views with vertex colors corresponding to vertex labels.

    Parameters
    ----------
    sso :
    vertex_labels : np.array
        vertex labels [N, 1]. Ordering and length have to be the same as
        vertex array of SuperSegmentationObject (len(sso.mesh[1]) // 3).
    coords :
    verbose :
    ws :
    rot_mat :
    nb_views :
    comp_window :
    return_rot_matrices :

    Returns:
    -------

    """
    if comp_window is None:
        comp_window = 8e3
    if ws is None:
        ws = (256, 128)
    if nb_views is None:
        nb_views = global_params.NB_VIEWS
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
    Parameters
    ----------
    sso : SuperSegmentationObject
    verbose : bool

    Returns
    -------
    dict
    """
    views = render_sampled_sso(sso, verbose=verbose, return_views=True)
    view_dc = {sso.id: arrtolz4string(views)}
    return view_dc


def render_sso_ortho_views(sso):
    """
    Renders three views of SSO mesh.

    Parameters
    ----------
    sso : SuperSegmentationObject

    Returns
    -------
    np.ndarray
    """
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


def render_sso_coords_multiprocessing(ssv, wd, n_jobs, rendering_locations=None,
                                      verbose=False, render_kwargs=None,
                                      render_indexviews=True, return_views=True):
    """
    # TODO: currently the view sorting is not aligned with the other rendering methods

    Parameters
    ----------
    ssv : SuperSegmentationObject
    wd : string
        working directory for accessing data
    rendering_locations: array of locations to be rendered
        if not given, rendering locations are retrieved from the SSV's SVs. Results will be stored at SV locations.
    n_jobs : int
        number of parallel jobs running on same node of cluster
    verbose : bool
        flag to show th progress of rendering.
    return_views : bool
        if False and rendering_locations is None, views will be saved at
        SSV SVs

    Returns
    -------
    np.ndarray
        array of views after rendering of locations.
    -------

    """
    if rendering_locations is not None and return_views is False:
        raise ValueError('"render_sso_coords_multiprocessing" received invalid '
                         'parameters (`rendering_locations!=None` and `return_v'
                         'iews=False`). When using specific rendering locations, '
                         'views have to be returned-')
    tim = time.time()
    if rendering_locations is None:  # use SV rendering locations
        svs = list(ssv.svs)
        rendering_locations = [sv.sample_locations for sv in svs]
        # store number of rendering locations per SV -> Ordering of rendered
        # views must be preserved!
        part_views = np.cumsum([0] + [len(c) for c in rendering_locations])

    chunk_size = len(rendering_locations) // n_jobs + 1
    params = chunkify_successive(rendering_locations, chunk_size)
    ssv_id = ssv.id
    working_dir = wd
    sso_kwargs = {'ssv_id': ssv_id,
                  'working_dir': working_dir,
                  "version": ssv.version}
    render_kwargs_def = {'add_cellobjects': True, 'verbose': verbose, 'clahe': False,
                      'ws': None, 'cellobjects_only': False, 'wire_frame': False,
                      'nb_views': None, 'comp_window': None, 'rot_mat': None, 'wo_glia': True,
                     'return_rot_mat': False, 'render_indexviews': render_indexviews}
    if render_kwargs is not None:
        render_kwargs_def.update(render_kwargs)

    params = [[par, sso_kwargs, render_kwargs_def, ix] for ix, par in
              enumerate(params)]
    tim1 = time.time()
    if verbose:
        log_proc.debug("Time for OTHER COMPUTATION {:.2f}s."
                       "".format(tim1 - tim))
    # This is single node multiprocessing -> `disable_batchjob=False`
    path_to_out = QSUB_script(
        params, "render_views_multiproc", suffix="_SSV{}".format(ssv_id),
        queue=None, script_folder=None, n_cores=1, disable_batchjob=True,
        n_max_co_processes=n_jobs)
    out_files = glob.glob(path_to_out + "/*")
    views = []
    out_files2 = np.sort(out_files, axis=-1, kind='quicksort', order=None)
    for out_file in out_files2:
        with open(out_file, 'rb') as f:
            views.append(pkl.load(f))
    if(type(views[0]) is int):
        views = 0
    else:
        views = np.concatenate(views)
    shutil.rmtree(path_to_out + "/../", ignore_errors=True)
    if rendering_locations is None and return_views is False:
        for i, so in enumerate(svs):
            so.enable_locking = True
            sv_views = views[part_views[i]:part_views[i+1]]
            so.save_views(sv_views, woglia=render_kwargs_def['wo_glia'],
                          cellobjects_only=render_kwargs_def['cellobjects_only'],
                          index_views=render_kwargs_def["render_indexviews"])
        return
    return views


def render_sso_coords_generic(ssv, working_dir, rendering_locations, n_jobs=None,
                              verbose=False, render_indexviews=True):
    """

    Args:
        ssv: SuperSegmentationObject
        working_dir: string
            working directory for accessing data
        rendering_locations: array of locations to be rendered
            if not given, rendering locations are retrieved from the SSV's SVs. Results will be stored at SV locations.
        n_jobs : int
            number of parallel jobs running on same node of cluster
        verbose : bool
            flag to show th progress of rendering.
        render_indexviews: Bool
            Flag to choose between render_index_view and render_sso_coords

     Returns
    -------
    np.ndarray
        array of views after rendering of locations.
    -------

    """
    if n_jobs is None:
        n_jobs = global_params.NCORES_PER_NODE // 10

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
