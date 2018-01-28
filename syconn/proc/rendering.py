# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Sven Dorkenwald, Philipp Schubert, Joergen Kornfeld

from ctypes import sizeof, c_float, c_void_p, c_uint
from PIL import Image
from .image import rgb2gray, apply_clahe, normalize_img
from scipy.ndimage.filters import gaussian_filter
import numpy as np
import time
import sys
import warnings
from ..handler.basics import flatten_list
from ..handler.compression import arrtolz4string
from .meshes import merge_meshs, get_random_centered_coords, \
    MeshObject, calc_rot_matrices, flag_empty_spaces

try:
    import os
    if not os.environ.get('PYOPENGL_PLATFORM'):
        os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
    import OpenGL
    OpenGL.USE_ACCELERATE = False
    from OpenGL.GL import *
    from OpenGL.GLU import *
    from OpenGL.osmesa import *
    from OpenGL.GL.framebufferobjects import *
except Exception as e:
    print("Problem loading OpenGL:", e)
    pass

# ------------------------------------------------------------------------------
# General rendering code

comp_views = 2
comp_window = 8e3 # window length along main p.c. for mesh view
float_size = sizeof(c_float)
vertex_offset = c_void_p(0 * float_size)
normal_offset = c_void_p(3 * float_size)
color_offset  = c_void_p(6 * float_size)
record_len = 10 * float_size


def init_object(indices, vertices, normals, colors, ws):
    """
    Initialize objects for rendering.

    Parameters
    ----------
    indices : np.array [N, 1]
    vertices : np.array [N, 1]
    normals : np.array [N, 1]
    colors : np.array [N, 1]
    ws : tuple

    Returns
    -------

    """
    global ind_cnt
    vertices = vertices.astype(np.float32)
    indices = indices.astype(np.uint32)
    ind_cnt = len(indices)
    normals = normals.astype(np.float32)
    data = np.concatenate((vertices.reshape(len(vertices) / 3, 3),
                           normals.reshape(len(vertices) / 3, 3),
                           colors.reshape((len(vertices) / 3, 4))),
                           axis=1).reshape(len(vertices)*2 + len(colors))
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
    if triangulation:
        glDrawElements(GL_TRIANGLES, ind_cnt, GL_UNSIGNED_INT, None)
    else:
        glDrawElements(GL_QUADS, ind_cnt, GL_UNSIGNED_INT, None)


def screen_shot(ws, colored=False, depth_map=False, clahe=False):
    """
    Create screenshot of currently opened window and return as array.

    Parameters
    ----------
    ws : tuple
    colored : bool
    depth_map : bool
    clahe : bool

    Returns
    -------
    np.array
    """
    glBindFramebuffer(GL_FRAMEBUFFER, 0)
    draw_object()
    glReadBuffer(GL_FRONT)
    if depth_map:
        data = glReadPixels(0, 0, ws[0], ws[1],
                            GL_DEPTH_COMPONENT, GL_UNSIGNED_BYTE)
        data = Image.frombuffer(mode="L", size=(ws[0], ws[1]),
                               data=data)
        data = np.asarray(data.transpose(Image.FLIP_TOP_BOTTOM)).astype(np.float32)
        if clahe:
            data = apply_clahe(data)
        data = normalize_img(gaussian_filter(data, .7), max_val=1.)
        if np.sum(data) == 0:
            data = np.ones_like(data)
    elif colored:
        data = glReadPixels(0, 0, ws[0], ws[1],
                            GL_RGB, GL_UNSIGNED_BYTE)
        data = Image.frombuffer(mode="RGB", size=(ws[0], ws[1]),
                               data=data)
        data = np.asarray(data.buf(Image.FLIP_TOP_BOTTOM))
    else:
        data = glReadPixels(0, 0, ws[0], ws[1],
                            GL_RGB, GL_UNSIGNED_BYTE)
        data = Image.frombuffer(mode="RGB", size=(ws[0], ws[1]),
                               data=data)
        data = rgb2gray(np.asarray(data.transpose(Image.FLIP_TOP_BOTTOM)))
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    return data


# setup ######################################################################
def init_ctx(ws):
    # ctx = OSMesaCreateContext(OSMESA_RGBA, None)
    ctx = OSMesaCreateContextExt(OSMESA_RGBA, 32, 0, 0, None)
    buf = arrays.GLubyteArray.zeros((ws[0], ws[1], 4)) + 1
    assert(OSMesaMakeCurrent(ctx, buf, GL_UNSIGNED_BYTE, ws[0], ws[1]))
    assert(OSMesaGetCurrentContext())
    OSMesaPixelStore(OSMESA_Y_UP, 0)
    return ctx


def init_opengl(ws, enable_lightning=False, clear_value=None, depth_map=False):
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

    if not depth_map:
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT, GL_AMBIENT)
    else:
        glEnable(GL_DEPTH_TEST)

    glViewport(0, 0, ws[0], ws[1])
    if clear_value is None:
        glClearColor(0., 0., 0., 0.)
    else:
        glClearColor(clear_value, clear_value, clear_value, 0.)


def multi_view_mesh(indices, vertices, normals, colors=None, alpha=None,
                    ws=(2048, 2048), physical_scale=None,
                    enable_lightning=False, depth_map=False,
                    nb_views=3):
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

    Returns
    -------

    """
    ctx = init_ctx(ws)
    init_opengl(ws, enable_lightning, depth_map=depth_map)
    vertices = np.array(vertices)
    indices = np.array(indices, dtype=np.uint)
    if colors is not None:
        colored = True
        colors = np.array(colors)
    else:
        colored = False
        colors = np.ones(len(vertices) / 3 * 4) * 0.2
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
    OSMesaDestroyContext(ctx)
    return np.concatenate(c_views)


def multi_view_mesh_coords(mesh, coords, rot_matrices, edge_lengths, alpha=None,
                           ws=(256, 128), views_key="raw", nb_simplices=3,
                           depth_map=True, clahe=False):
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
        Number of simplices used for meshs
    ws : tuple of ints
        Window size used for rendering (resolution of array being stored/saved)
    depth_map : bool
        Render views as depth, else render without light effects (binary)
    clahe : bool
        apply clahe to screenshot

    Returns
    -------
    np.array
        Returns array of views, else None
    """
    # center data
    assert isinstance(edge_lengths, np.ndarray)
    assert nb_simplices in [3, 4]
    vertices = mesh.vertices
    indices = mesh.indices
    colors = mesh.colors
    if depth_map and mesh._normals is None:
        normals = np.zeros(len(vertices))
    else:
        normals = mesh.normals
    edge_lengths = edge_lengths / mesh.max_dist
    # default color
    if colors is not None:
        colors = np.array(colors)
    else:
        colors = np.ones(len(vertices) / 3 * 4) * 0.8
    if alpha is not None:
        colors[::4] = alpha
    res = np.ones((len(coords), comp_views, ws[1], ws[0]))
    init_opengl(ws, depth_map=depth_map, clear_value=1.0)
    init_object(indices, vertices, normals, colors, ws)
    for ii, c in enumerate(coords):
        c_views = np.ones((comp_views, ws[1], ws[0]),
                         dtype=np.float32)
        rot_mat = rot_matrices[ii]
        if np.sum(np.abs(rot_mat)) == 0 or np.sum(np.abs(mesh.vertices)) == 0:
            if views_key == "raw":
                warnings.warn("Rotation matrix or vertices of '%s' with %d "
                              "vertices is zero during rendering. Skipping."
                              % (views_key, len(mesh.vert_resh)), RuntimeWarning)
            continue
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(-edge_lengths[0]/2, edge_lengths[0]/2, edge_lengths[1]/2,
                -edge_lengths[1]/2, -edge_lengths[2]/2, edge_lengths[2]/2)
        glMatrixMode(GL_MODELVIEW)

        transformed_c = mesh.transform_external_coords([c])[0]
        # glPushMatrix()
        # glRotate(360. / 4 * 0, edge_lengths[0], 0, 0)
        # glMultMatrixf(rot_mat)
        # glTranslate(-transformed_c[0], -transformed_c[1], -transformed_c[2])
        # light_position = [1., 1., 2., 0.]
        # glLightfv(GL_LIGHT0, GL_POSITION, light_position)
        # glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE)
        # dummy rendering, somehow first screenshot is always black
        _ = screen_shot(ws, depth_map=depth_map, clahe=clahe)
        # glPopMatrix()

        glMatrixMode(GL_MODELVIEW)
        for m in range(0, comp_views):
            glPushMatrix()
            glRotate(360. / 4 * m, edge_lengths[0], 0, 0)
            glMultMatrixf(rot_mat)
            glTranslate(-transformed_c[0], -transformed_c[1], -transformed_c[2])
            light_position = [1., 1., 2., 0.]
            glLightfv(GL_LIGHT0, GL_POSITION, light_position)
            glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE)
            c_views[m] = screen_shot(ws, depth_map=depth_map, clahe=clahe)
            glPopMatrix()
        res[ii] = c_views
        found_empty_view = False
        for cv in c_views:
            if np.sum(cv) == 0 or np.sum(cv) == np.prod(cv.shape):
                if views_key == "raw":
                    warnings.warn("Empty view of '%s'-mesh with %d "
                                  "vertices found."
                                  % (views_key, len(mesh.vert_resh)),
                                  RuntimeWarning)
                    found_empty_view = True
            if found_empty_view:
                print "View 1: %0.1f\t View 2: %0.1f\t#view in list: %d/%d\n" \
                      "'%s'-mesh with %d vertices." %\
                      (np.sum(c_views[0]), np.sum(c_views[1]), ii, len(coords),
                       views_key, len(mesh.vertices))
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
    glColor(0,0,0, 1)
    glVertex2f(1 - 0.1 - size,1 - 0.1)
    glVertex2f(1 - 0.1,1 - 0.1)
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


def render_mesh_coords(coords, ind, vert, clahe=False, verbose=False, ws=(256, 128),
                       rot_matrices=None, views_key="raw", return_rot_matrices=False,
                       depth_map=True):
    """
    Render raw views located at given coordinates in mesh
     Returns ViewContainer list if dest_dir is None, else writes
    views to dest_path.

    Parameters
    ----------
    coords : np.array
    ind : np.array [N, 1]
    vert : np.array [N, 1]
    clahe : bool
    verbose : bool
    ws : tuple
        Window size
    rot_matrices : np.array
    views_key : str
    return_rot_matrices : bool
    depth_map : bool

    Returns
    -------
    numpy.array
        views at each coordinate
    """
    mesh = MeshObject(views_key, ind, vert)
    mesh.colors = None
    edge_lengths = np.array([comp_window, comp_window / 2, comp_window / 2])
    if verbose:
        start = time.time()
    if rot_matrices is None:
        rot_matrices = calc_rot_matrices(mesh.transform_external_coords(coords),
                                         mesh.vert_resh, edge_lengths / mesh.max_dist)
        local_rot_mat = rot_matrices
    else:
        empty_locs = flag_empty_spaces(coords, vert.reshape((-1, 3)),
                                       edge_lengths)
        local_rot_mat = np.array(rot_matrices)
        local_rot_mat[empty_locs] = 0
        # if views_key == "raw":
        # print "%d/%d spaces are empty while rendering '%s'." % \
        #       (np.sum(empty_locs), len(coords), views_key)
    if verbose:
        print "Calculation of rotation matrices took", time.time() - start
        print "Starting local rendering at %d locations (%s)." %\
              (len(coords), views_key)
    ctx = init_ctx(ws)
    mviews = multi_view_mesh_coords(mesh, coords, local_rot_mat, edge_lengths,
                                    clahe=clahe, views_key=views_key, ws=ws,
                                    depth_map=depth_map)
    if verbose:
        end = time.time()
        print "Finished rendering mesh of type %s at %d locations after" \
              " %0.1fs" % (views_key,len(mviews), end - start)
    OSMesaDestroyContext(ctx)
    if return_rot_matrices:
        return mviews, rot_matrices
    return mviews

# ------------------------------------------------------------------------------
# SSO rendering code


def render_sampled_sso(sso, ws=(256, 128), verbose=False, woglia=True,
                       add_cellobjects=True, overwrite=False,
                       return_views=False, cellobjects_only=False):
    """
    Renders for each SV views at sampled locations (number is dependent on
    SV mesh size with scaling fact) from combined mesh of all SV.
    ----------------------------------------------------------------------
    USED FOR AXONESS CLASSIFICATION
    ----------------------------------------------------------------------

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
    overwrite : bool
    return_views : bool
    cellobjects_only : bool
    """
    # get coordinates for N SV's in SSO
    if verbose:
        start = time.time()
    coords = sso.sample_locations()
    if not overwrite:
        missing_sv_ixs = np.array([not so.views_exist for so in sso.svs], dtype=np.bool)
        missing_svs = np.array(sso.svs)[missing_sv_ixs]
        coords = np.array(coords)[missing_sv_ixs]
    else:
        missing_svs = np.array(sso.svs)
    print "Rendering %d missing SV's." % len(missing_svs)
    if len(missing_svs) == 0:
        if return_views:
            return np.concatenate(sso.load_views(woglia=woglia))
        return
    # len(part_views) == N + 1
    part_views = np.cumsum([0] + [len(c) for c in coords])
    flat_coords = np.array(flatten_list(coords))
    views = render_sso_coords(sso, flat_coords, ws=ws, verbose=verbose,
                              add_cellobjects=add_cellobjects,
                              cellobjects_only=cellobjects_only)
    for i, so in enumerate(missing_svs):
        sv_views = views[part_views[i]:part_views[i+1]]
        so.save_views(sv_views, woglia=woglia, cellobjects_only=cellobjects_only)
    if verbose:
        dur = time.time() - start
        print "Rendering of %d views took %0.2fs (incl. read/write). " \
              "%0.4fs/SV" % (len(views), dur, float(dur)/len(sso.svs))
    if return_views:
        return np.concatenate(sso.load_views(woglia=woglia))


def render_sso_coords(sso, coords, add_cellobjects=True, verbose=False, clahe=False,
                      ws=(256, 128), cellobjects_only=False):
    """
    Render views of SuperSegmentationObject at given coordinates.
    
    Parameters
    ----------
    sso : 
    coords : 
    add_cellobjects : 
    verbose : 
    clahe : 
    ws : 

    Returns
    -------
    np.array
    """
    mesh = sso.mesh
    if len(mesh[1]) == 0:
        print "----------------------------------------------\n" \
              "No mesh for SSO %d found.\n"\
              "----------------------------------------------\n"
        return
    raw_views = np.ones((len(coords), 2, 128, 256))
    if cellobjects_only:
        assert add_cellobjects, "Add cellobjects must be True when rendering" \
                                "cellobjects only."
        edge_lengths = np.array([comp_window, comp_window / 2, comp_window / 2])
        mo = MeshObject("raw", mesh[0], mesh[1])
        mo.colors = None
        rot_mat = calc_rot_matrices(mo.transform_external_coords(coords),
                                    mo.vert_resh, edge_lengths / mo.max_dist)
    else:
        raw_views, rot_mat = render_mesh_coords(coords, mesh[0], mesh[1], clahe=clahe,
                                       verbose=verbose, return_rot_matrices=True, ws=ws)
    if add_cellobjects:
        mesh = sso.mi_mesh
        if len(mesh[1]) != 0:
            mi_views = render_mesh_coords(coords, mesh[0], mesh[1], clahe=clahe,
                                          verbose=verbose, rot_matrices=rot_mat,
                                          views_key="mi", ws=ws)
        else:
            mi_views = np.ones_like(raw_views)
        mesh = sso.vc_mesh
        if len(mesh[1]) != 0:
            vc_views = render_mesh_coords(coords, mesh[0], mesh[1], clahe=clahe,
                                          verbose=verbose, rot_matrices=rot_mat,
                                          views_key="vc", ws=ws)
        else:
            vc_views = np.ones_like(raw_views)
        mesh = sso.sj_mesh
        if len(mesh[1]) != 0:
            sj_views = render_mesh_coords(coords, mesh[0], mesh[1], clahe=clahe,
                                          verbose=verbose, rot_matrices=rot_mat,
                                          views_key="sj", ws=ws)
        else:
            sj_views = np.ones_like(raw_views)
        if cellobjects_only:
            return np.concatenate([mi_views[:, None], vc_views[:, None],
                                   sj_views[:, None]], axis=1)
        return np.concatenate([raw_views[:, None], mi_views[:, None],
                               vc_views[:, None], sj_views[:, None]], axis=1)
    return raw_views[:, None]


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


# ------------------------------------------------------------------------------
# Multiprocessing rendering code


def multi_render_sampled_svidlist(svixs):
    """
    Render SVs with ID's svixs using helper script in syconnfs.examples.
    OS rendering requires individual creation of OSMesaContext.
    Change kwargs for SOs in syconnfs.examples.render_helper_svidlist.

    Parameters
    ----------
    svixs : iterable
        SegmentationObject ID's
    """
    fpath = os.path.dirname(os.path.abspath(__file__))
    cmd = "python %s/../examples/render_helper_svidlist.py" % fpath
    for ix in svixs:
        cmd += " %d" % ix


def multi_render_sampled_sso(sso_ix):
    """
    Render SSO with ID sso_ix using helper script in syconnfs.examples.
    OS rendering requires individual creation of OSMesaContext.
    Change kwargs for SSO in syconnfs.examples.render_helper_svidlist.

    Parameters
    ----------
    sso_ix : int
    """
    fpath = os.path.dirname(os.path.abspath(__file__))
    cmd = "python %s/../examples/render_helper_sso.py" % fpath
    cmd += " %d" % sso_ix


