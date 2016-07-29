# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max-Planck-Institute for Medical Research, Heidelberg, Germany
# Authors: Sven Dorkenwald, Philipp Schubert, Joergen Kornfeld


import matplotlib.cm as cmx
import matplotlib.colors as colors
from syconn.processing.synapticity import syn_sign_prediction
from syconn.utils.datahandler import *
try:
    from mayavi import mlab
except (ImportError, ValueError), e:
    print e
    print "Could not load mayavi. Please install vtk and then mayavi."


def init_skel_vis(skel, min_pos, max_pos, hull_visualization=True, op=0.15,
                  save_vx_dir=None, cmap=None):
    """Initialize plot for skeleton

    Parameters
    ----------
    skel: Skeleton
    min_pos/max_pos : np.array
    hull_visualization: bool
    op: float
    save_vx_dir : str
    cmap : colormap
    """
    for node in list(skel.getNodes()):
        node_coord = arr(node.getCoordinate()) * (arr(skel.scaling)/10.)
        if np.any(node_coord < min_pos) or np.any(node_coord > max_pos):
            skel.removeNode(node)
    if hull_visualization is True:
        bools = get_box_coords(skel.hull_coords / 10., min_pos, max_pos,
                               ret_bool_array=True)
        hull_points = skel.hull_coords[bools] / 10.
        hull_normals = skel.hull_normals[bools]
        if save_vx_dir is not None:
            hull2text(hull_points, hull_normals, save_vx_dir +
                      skel.filename + 'hull.xyz')
            return
        # while len(hull_points) > 0.3e6:
            # hull_points = hull_points[::2]
            # print "Subsampling"
        if cmap is not None:
            skel.color = None
        mlab.points3d(hull_points[:, 0], hull_points[:, 1], hull_points[:, 2],
                      scale_factor=1.5, mode='sphere', color=skel.color,
                      opacity=op, colormap=cmap)
    else:
        add_anno_to_mayavi_window(skel, 1./10., 5, dataset_identifier='',
                                  show_outline=False)


def init_skel_vis_with_properties(skel, min_pos, max_pos,
                                  property='spiness_pred'):
    """Initializes mayavi environment for plotting skeleton with properties

    Parameters
    ----------
    skel: annotation object containing with skeleton nodes
    property : str
        property to visualize
    """
    node_axoness = {0: [], 1: [], 2:[]}
    node_coords = []
    for i, node in enumerate(list(skel.getNodes())):
        node_coord = arr(node.getCoordinate()) * (arr(skel.scaling)/10.)
        node_coords.append(node_coord)
        if np.any(node_coord < min_pos) or np.any(node_coord > max_pos):
            skel.removeNode(node)
        node_axoness[int(node.data[property])] += [i]
    hull_points = get_box_coords(skel.hull_coords / 10., min_pos, max_pos)
    while len(hull_points) > 0.3e6:
        hull_points = hull_points[::2]
    colors = [(0.75, 0.75, 0.75), (0.95, 0.05, 0.05), (0.15, 0.15, 0.15)]
    skel_tree = spatial.cKDTree(node_coords)
    dist, nn_ixs = skel_tree.query(hull_points, k=1)
    for i in range(3):
        ixs_with_axoness = node_axoness[i]
        curr_hull_points = []
        for ix in ixs_with_axoness:
            curr_hull_points += hull_points[nn_ixs == ix].tolist()
        if len(curr_hull_points) == 0:
            continue
        curr_hull_points = arr(curr_hull_points)
        mlab.points3d(curr_hull_points[:, 0], curr_hull_points[:, 1],
                      curr_hull_points[:, 2], scale_factor=4.5, mode='sphere',
                      color=colors[i], opacity=0.15)


def plot_skel(path, min_pos, max_pos, hull_visualization=False, cmap=None,
              color=None, vis_property=False, op=0.1, save_vx_dir=None):
    """Plots the pure skeleton at path using mayavi. Annotation file must
    contain tree 'skeleton'
    """
    if type(path) is str:
        [skel, mitos, vc, sj, soma] = load_anno_list([path])[0]
    else:
        skel = path
    if skel.color != None:
        skel.color = color
    if vis_property:
        init_skel_vis_with_properties(skel, min_pos, max_pos)
    else:
        init_skel_vis(skel, min_pos, max_pos, op=op, save_vx_dir=save_vx_dir,
                      hull_visualization=hull_visualization, cmap=cmap)


def coords_from_anno(skel):
    """Extracts the coordinates from skeleton

    Parameters
    ----------
    skel: Skeleton loaded with load_mapped_skeleton, i.e. object hull coords
    are appended

    Returns
    -------
    list of np.array
        cell object hull coordinates
    """
    mito_hulls = skel.mito_hull_coords / 10
    vc_hulls = skel.vc_hull_coords / 10
    sj_hulls = skel.sj_hull_coords / 10
    return mito_hulls, vc_hulls, sj_hulls


def plot_mapped_skel(path, min_pos, max_pos, color=(1./3, 1./3, 1./3), op=0.1,
                     hull_visualization=True, save_vx_dir=None,
                     plot_objects=(True, True, True), color_sj=False):
    """Plots the mapped skeleton at path using mayavi. Annotation file must
    contain trees 'vc', 'sj', 'mitos' and 'skeleton'

    Parameters
    ----------
    path : string
        Path to .k.zip or .nml of mapped annotation
    min_pos/max_pos : np.array
        coordinates of bounding box to plot
    color : tuple of float
        normalized rgb values
    op : float
        opacity
    hull_visualization : bool
        visualize hull
    save_vx_dir : str
        path to destination of plotted hull and object voxels
    plot_objects : tuple of bool
        which cell objects are to be plotted
    color_sj:
        colorize synapse type
    """
    [skel, mitos, vc, sj, soma] = load_anno_list([path])[0]
    print "Loaded mapped skeleton %s from %s and write it to %s" % (
        skel.filename, path, str(save_vx_dir))
    skel.color = color
    mitos, vc, sj = coords_from_anno(skel)
    if not (mitos == []) and mitos.ndim == 1:
        mitos = mitos[:, None]
    if not (vc == []) and vc.ndim == 1:
        vc = vc[:, None]
    if not (sj == []) and sj.ndim == 1:
        sj = sj[:, None]
    print "Found %d mitos, %d vc, %d sj in %s." % \
          (len(set(list(skel.mito_hull_ids))), len(set(list(skel.vc_hull_ids))),
           len(set(list(skel.sj_hull_ids))), skel.filename)
    mitos = get_box_coords(mitos, min_pos, max_pos)
    vc = get_box_coords(vc, min_pos, max_pos)
    sj = get_box_coords(sj, min_pos, max_pos)
    if save_vx_dir is None:
        while (len(mitos)+len(vc)+len(sj) > 0.4e6):
            # print "Downsmapling of objects!!"
            mitos = mitos[::4]
            vc = vc[::2]
            sj = sj[::2]
            skel.sj_hull_ids = skel.sj_hull_ids[::2]
            skel.sj_hull_coords = skel.sj_hull_coords[::2]
    else:
        while (len(mitos)+len(vc)+len(sj) > 5e6):
            # print "Downsmapling of objects!!"
            mitos = mitos[::4]
            vc = vc[::2]
            sj = sj[::2]
            skel.sj_hull_ids = skel.sj_hull_ids[::2]
            skel.sj_hull_coords = skel.sj_hull_coords[::2]
    # print "Plotting %d object hull voxels." % (len(mitos)+len(vc)+len(sj))
    # plot objects
    objects = [mitos, vc, sj]
    colors = [(0./255, 153./255, 255./255), (0.175,0.585,0.301), (0.849,0.138,0.133)]
    syn_type_coloers = [(222./255, 102./255, 255./255), (102./255, 255./255, 0./255)]
    for obj, color, plot_b, type in zip(objects, colors, plot_objects,
                                  ['mito', 'vc', 'sj']):
        if not plot_b:
            continue
        if len(obj) == 0:
            print "Left out object.", color
            continue
        if save_vx_dir is not None:
            hull2text(obj, get_normals(obj), save_vx_dir + '%s%s_%s.xyz'
            % (skel.filename, str(tuple((arr(color)*255).astype(np.int))), type))
            continue
        if type=='sj' and color_sj:
            for ix in set(skel.sj_hull_ids):
                ix_bool_arr = skel.sj_hull_ids == ix
                obj_hull = skel.sj_hull_coords[ix_bool_arr]
                syn_type_pred = syn_sign_prediction(obj_hull/arr([9., 9., 20.]))
                obj_hull /= 10.
                mlab.points3d(obj_hull[:, 0], obj_hull[:, 1], obj_hull[:, 2],
                              scale_factor=4.5, mode='sphere', opacity=0.7,
                              color=syn_type_coloers[syn_type_pred])
        else:
            mlab.points3d(obj[:, 0], obj[:, 1], obj[:, 2], scale_factor=4.5,
                          mode='sphere', color=color, opacity=0.7)
    init_skel_vis(skel, min_pos, max_pos, hull_visualization=hull_visualization,
                  op=op, save_vx_dir=save_vx_dir)


def write_img2png(nml_path, filename='mayavi_img', mag=5):
    """Writes plotted mayavi image to .png at head folder of path with
    supplement

    Parameters
    ----------
    nml_path : string
        path for output file
    filename : str
        filename
    mag : int
        resolution of image
    """
    head, tail = os.path.split(nml_path)
    img_path = head+"/"+filename+".png"
    mlab.savefig(img_path, magnification=mag)
    mlab.close()
    print "Writing image to %s." % img_path


def calc_obj_normals(object_vx):
    """Calculate normals approximated by vector to center voxel.

    Parameters
    ----------
    object_vx: np.array
        object voxel coordinates

    Returns
    -------
    np.array
        normalized normals for each voxel coordinate
    """
    center_coords = np.mean(object_vx, axis=0)
    normals = object_vx - center_coords
    normals /= np.linalg.norm(normals)
    return normals


def get_cmap(N, cmap):
    """Returns a function that maps each index in 0, 1, ... N-1 to a distinct
    RGB color.

    Parameters
    ----------
    N : int
    cmap : colormap

    Returns
    -------
    function
    """
    color_norm = colors.Normalize(vmin=0, vmax=N-1)
    scalar_map = cmx.ScalarMappable(norm=color_norm, cmap=cmap)
    def map_index_to_rgb_color(index):
        return scalar_map.to_rgba(index)[:3]
    return map_index_to_rgb_color


def norm_rgb_color(color):
    """Normalize color

    Parameters
    ----------
    color : tuple of int/float

    Returns
    -------
    tuple of float
        normalized rgb
    """
    normed_color = (color[0]/255., color[1]/255., color[2]/255.)
    return normed_color


def add_anno_to_mayavi_window(anno, node_scaling=1.0, override_node_radius=500.,
                              edge_radius=250., show_outline=False,
                              dataset_identifier='', opacity=1):
    """Adds an annotation to a currently open mayavi plotting window

    Parameters
    ----------
    anno : annotation object
    node_scaling: float
        scaling factor for node radius
    override_node_radius : bool
    edge_radius: float
        radius of tubes for each edge
    show_outline : bool
    dataset_identifier : str
    opacity : float
    """
    if type(anno) == list:
        nodes = []
        for this_anno in anno:
            nodes.extend(this_anno.getNodes())
        color = anno[0].color
    else:
        nodes = list(anno.getNodes())
        color = anno.color
    coords = np.array(
        [node.getCoordinate_scaled() for node in nodes]) * node_scaling
    sc = np.hsplit(coords, 3)
    x = [el[0] for el in sc[0].tolist()]
    y = [el[0] for el in sc[1].tolist()]
    z = [el[0] for el in sc[2].tolist()]
    if override_node_radius > 0.:
        s = [override_node_radius] * len(nodes)
    else:
        s = [node.getDataElem('radius') for node in nodes]
    s = np.array(s)
    s = s * node_scaling
    pts = mlab.points3d(x, y, z, s, color=color, scale_factor=1.0,
                        opacity=opacity)
    # dict for faster lookup, nodes.index(node) adds O(n^2)
    nodeIndexMapping = {}
    for nodeIndex, node in enumerate(nodes):
        nodeIndexMapping[node] = nodeIndex
    edges = []
    for node in nodes:
        for child in node.getChildren():
            try:
                edges.append(
                    (nodeIndexMapping[node], nodeIndexMapping[child]))
            except:
                print 'Phantom child node, annotation object inconsistent'
    # plot the edges
    pts.mlab_source.dataset.lines = np.array(edges)
    pts.mlab_source.update()
    tube = mlab.pipeline.tube(pts, tube_radius=edge_radius)
    mlab.pipeline.surface(tube, color=anno.color)

    if show_outline:
        if dataset_identifier == 'j0126':
            mlab.outline(extent=(0, 108810, 0, 106250, 0, 115220),
                         opacity=0.5,
                         line_width=5.)
        elif dataset_identifier == 'j0251':
            mlab.outline(extent=(0, 270000, 0, 270000, 0, 387350),
                         opacity=0.5,
                         line_width=5.)
        elif dataset_identifier == 'j0256':
            mlab.outline(extent=(0, 166155, 0, 166155, 0, 77198),
                         opacity=0.5,
                         line_width=1., color=(0.5, 0.5, 0.5))
        else:
            print('Please add a dataset identifier string')
