# -*- coding: utf-8 -*-
__author__ = 'pschuber'
import matplotlib.cm as cmx
import matplotlib.colors as colors
import matplotlib.pyplot as plt
try:
    from mayavi import mlab
except (ImportError, ValueError), e:
    print "Could not load mayavi. Please install vtk and then mayavi."
from multiprocessing import Pool
from scipy import stats
from sklearn.decomposition import PCA

from syconn.contactsites import convert_to_standard_cs_name
from syconn.brainqueries import enrich_tracings, remap_tracings
from syconn.processing.cell_types import load_celltype_feats
from syconn.processing.features import get_obj_density
from syconn.processing.synapticity import syn_sign_prediction
from syconn.utils.datahandler import *
from syconn.new_skeleton.newskeleton import NewSkeleton, SkeletonAnnotation,\
    SkeletonNode


def init_skel_vis(skel, min_pos, max_pos, hull_visualization=True, op=0.15,
                  save_vx_dir=None, cmap=None):
    for node in list(skel.getNodes()):
        node_coord = arr(node.getCoordinate()) * (arr(skel.scaling)/10.)
        if np.any(node_coord < min_pos) or np.any(node_coord > max_pos):
            skel.removeNode(node)
    if hull_visualization is True:
        bools = get_box_coords(skel._hull_coords / 10., min_pos, max_pos,
                               ret_bool_array=True)
        hull_points = skel._hull_coords[bools] / 10.
        hull_normals = skel._hull_normals[bools]
        if save_vx_dir is not None:
            hull2text(hull_points, hull_normals, save_vx_dir +
                      skel.filename + 'hull.xyz')
            return
        # while len(hull_points) > 0.3e6:
            # hull_points = hull_points[::2]
            # print "Subsampling"
        if cmap is not None:
            skel.color = None
        print "Plotting %d hull voxels." % len(hull_points)
        mlab.points3d(hull_points[:, 0], hull_points[:, 1], hull_points[:, 2],
                      scale_factor=1.5, mode='sphere', color=skel.color,
                      opacity=op, colormap=cmap)
    else:
        add_anno_to_mayavi_window(skel, 1./10., 5, dataset_identifier='',
                                  show_outline=False)


def init_skel_vis_with_properties(skel, min_pos, max_pos, property='spiness_pred'):
    """
    Initializes mayavi environment for plotting skeleton.
    :param skel: annotation object containing with skeleton nodes
    :param hull_visualization: bool use hull points for skeleton visualization
    """
    node_axoness = {0: [], 1: [], 2:[]}
    node_coords = []
    for i, node in enumerate(list(skel.getNodes())):
        node_coord = arr(node.getCoordinate()) * (arr(skel.scaling)/10.)
        node_coords.append(node_coord)
        if np.any(node_coord < min_pos) or np.any(node_coord > max_pos):
            skel.removeNode(node)
        node_axoness[int(node.data[property])] += [i]
    hull_points = get_box_coords(skel._hull_coords / 10., min_pos, max_pos)
    while len(hull_points) > 0.3e6:
        hull_points = hull_points[::2]
        print "Subsampling"
    print "Plotting %d hull voxels." % len(hull_points)
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
    """
    Plots the pure skeleton at path using mayavi. Annotation file must contain
    tree 'skeleton'.
    :param path: string to .k.zip or .nml of annotation
    """
    if type(path) is str:
        [skel, mitos, p4, az, soma] = load_anno_list([path])[0]
    else:
        skel = path
    print "Using skeleton scaling %s divided by 10." % str(skel.scaling)
    if skel.color != None:
        skel.color = color
    if vis_property:
        init_skel_vis_with_properties(skel, min_pos, max_pos)
    else:
        init_skel_vis(skel, min_pos, max_pos, op=op, save_vx_dir=save_vx_dir,
                      hull_visualization=hull_visualization, cmap=cmap)


def coords_from_anno(skel):
    """
    Extracts the coordinates from annotation at path given DataHandler. If data-
    handler is not given, it is assumed that the object hull voxels are written
    in the corresponding object tree.
    :param obj_trees: list of annotation objects containing mapped cell objects
    :return: list of array node coordinates
    """
    mito_hulls = skel.mito_hull_coords / 10
    mito_hull_ids = skel.mito_hull_ids
    p4_hulls = skel.p4_hull_coords / 10
    az_hulls = skel.az_hull_coords / 10
    return mito_hulls, p4_hulls, az_hulls


def plot_mapped_skel(path, min_pos, max_pos, color=(1./3, 1./3, 1./3), op=0.1,
                     only_syn=False, cs_path=None, hull_visualization=True,
                     save_vx_dir=None, plot_objects=[True, True, True],
                     color_az=False):
    """
    Plots the mapped skeleton at path using mayavi. Annotation file must
    contain trees 'p4', 'az', 'mitos' and 'skeleton'
    :param path: string to .k.zip or .nml of mapped annotation
    :param only_syn: bool Only plot synapse
    :param cs_path: str Hpath to contact site nml of these skeletons.
    :return:
    """
    [skel, mitos, p4, az, soma] = load_anno_list([path])[0]
    print "Loaded mapped skeleton %s from %s and write it to %s" % (
        skel.filename, path, str(save_vx_dir))
    skel.color = color
    if only_syn:
        if cs_path is None:
            return
        mitos = []
        p4_id, az_id = get_syn_from_cs_nml(cs_path)
        if p4_id is None and az_id is None:
            mitos, p4, az = coords_from_anno(skel)
        else:
            p4 = []
            az = []
            for id in p4_id:
                p4 += list(skel.p4_hull_coords[skel.p4_hull_ids == id] / 10)
            for id in az_id:
                az += list(skel.az_hull_coords[skel.az_hull_ids == id] / 10)
            p4 = arr(p4)
            az = arr(az)
    else:
        mitos, p4, az = coords_from_anno(skel)
    if not (mitos == []) and mitos.ndim == 1:
        mitos = mitos[:, None]
    if not (p4 == []) and p4.ndim == 1:
        p4 = p4[:, None]
    if not (az == []) and az.ndim == 1:
        az = az[:, None]
    print "Found %d mitos, %d p4, %d az in %s." % (len(set(list(skel.mito_hull_ids))), len(set(list(skel.p4_hull_ids))),
                                                   len(set(list(skel.az_hull_ids))), skel.filename)
    mitos = get_box_coords(mitos, min_pos, max_pos)
    p4 = get_box_coords(p4, min_pos, max_pos)
    az = get_box_coords(az, min_pos, max_pos)
    if save_vx_dir is None:
        while (len(mitos)+len(p4)+len(az) > 0.4e6):
            print "Downsmapling of objects!!"
            mitos = mitos[::4]
            p4 = p4[::2]
            az = az[::2]
            skel.az_hull_ids = skel.az_hull_ids[::2]
            skel.az_hull_coords = skel.az_hull_coords[::2]
    else:
        while (len(mitos)+len(p4)+len(az) > 5e6):
            print "Downsmapling of objects!!"
            mitos = mitos[::4]
            p4 = p4[::2]
            az = az[::2]
            skel.az_hull_ids = skel.az_hull_ids[::2]
            skel.az_hull_coords = skel.az_hull_coords[::2]
    print "Plotting %d object hull voxels." % (len(mitos)+len(p4)+len(az))
    # plot objects
    objects = [mitos, p4, az]
    colors = [(0./255, 153./255, 255./255), (0.175,0.585,0.301), (0.849,0.138,0.133)]
    syn_type_coloers = [(222./255, 102./255, 255./255), (102./255, 255./255, 0./255)]
    for obj, color, plot_b, type in zip(objects, colors, plot_objects,
                                  ['mito', 'p4', 'az']):
        if not plot_b:
            continue
        if len(obj) == 0:
            print "Left out object.", color
            continue
        if save_vx_dir is not None:
            hull2text(obj, get_normals(obj), save_vx_dir + '%s%s_%s.xyz'
            % (skel.filename, str(tuple((arr(color)*255).astype(np.int))), type))
            continue
        print "Plotted %d %s voxel." % (len(obj), type)
        if type=='az' and color_az:
            for ix in set(skel.az_hull_ids):
                ix_bool_arr = skel.az_hull_ids == ix
                obj_hull = skel.az_hull_coords[ix_bool_arr]
                syn_type_pred = syn_sign_prediction(obj_hull/arr([9., 9., 20.]))
                obj_hull = obj_hull / 10.
                mlab.points3d(obj_hull[:, 0], obj_hull[:, 1], obj_hull[:, 2],
                              scale_factor=4.5, mode='sphere', opacity=0.7,
                              color=syn_type_coloers[syn_type_pred])
                print "Plotted synapse of type", syn_type_pred,\
                    "with %d vx" % len(obj_hull)
        else:
            mlab.points3d(obj[:, 0], obj[:, 1], obj[:, 2], scale_factor=4.5,
                          mode='sphere', color=color, opacity=0.7)
    init_skel_vis(skel, min_pos, max_pos, hull_visualization=hull_visualization,
                  op=op, save_vx_dir=save_vx_dir)


def write_img2png(nml_path, filename='mayavi_img', mag=5):
    """
    Writes plotted mayavi image to .png at head folder of path with supplement.
    :param nml_path: string path for output file
    """
    head, tail = os.path.split(nml_path)
    img_path = head+"/"+filename+".png"
    mlab.savefig(img_path, magnification=mag)
    mlab.close()
    print "Writing image to %s." % img_path


def get_cs_path(paths):
    """
    Combines the given paths to skeleton nmls to find path to pariwise
    contact site nml.
    :param paths: str tuple of paths to skeleton nmls
    :return: path to pairwise contact site nml
    """
    head, tail = os.path.split(paths[0])
    head = '/lustre/pschuber/consensi_fuer_joergen/nml_obj/'
    try:
        post_id = re.findall('iter_\d+_(\d+)-', paths[0])[0]
        pre_id = re.findall('iter_\d+_(\d+)-', paths[1])[0]
    except IndexError:
        pre_id = os.path.split(paths[0])[1]
        post_id = os.path.split(paths[1])[1]

    cs_file_path = head+"/contact_sites/pairwise/skel_%s_%s.nml" % (post_id, pre_id)
    if os.path.isfile(cs_file_path):
        pass
    else:
        cs_file_path = head+"/contact_sites/pairwise/skel_%s_%s.nml" % (pre_id, post_id)
        if not os.path.isfile(cs_file_path):
            print cs_file_path
            print "ERROR. Could not find pairwise nml file."
    return cs_file_path


def plot_post_pre(paths=['/home/pschuber/data/skel2skel/nml_obj/pre_04.k.zip',
                '/home/pschuber/data/skel2skel/nml_obj/post_04.k.zip']):
    """
    Creates mayavi plot of post and pre synaptic neuron.
    :param dh: object DataHandler
    :param paths: list of strings path to pre and post neuron
    """
    post = paths[0]
    pre = paths[1]
    cs_file_path = get_cs_path(paths)
    min_pos, max_pos, center_coord = get_box_from_cs_nml(cs_file_path)
    mlab.figure(2, bgcolor=(1, 1, 1), fgcolor=(0.5, 0.5, 0.5))
    mlab.clf()
    print "Plotting mapped post-synaptic skeleton"
    plot_mapped_skel(post, min_pos, max_pos, color=(1./2, 1./5, 1./5),
                     only_syn=True, cs_path=cs_file_path)
    print "Plotting mapped pre-synaptic skeleton"
    plot_mapped_skel(pre, min_pos, max_pos, color=(1./5, 1./5, 1./2),
                     only_syn=True, cs_path=cs_file_path)
    return


def get_box_from_cs_nml(path, offset=arr([250, 250, 250]), cs_nb=1):
    """
    Parses center coordinate from contact_site.nml and calculates bounding box.
    :param path: str to contact_site.nml
    :param offset: array defining bounding box
    :return:
    """
    cs_anno = au.loadj0126NML(path)[0]
    center_coord = None
    cs_str = 'cs%d' % cs_nb
    for node in cs_anno.getNodes():
        n_comment = node.getComment()
        if 'center_p4_az' in n_comment and cs_str in n_comment:
            center_coord = arr(node.getCoordinate()) * arr(cs_anno.scaling)/10.
            break
    if center_coord is None:
        print "Could not find center with p4 and az! Using contact_site."
        for node in cs_anno.getNodes():
            if 'center' in node.getComment() and 'cs1_' in node.getComment():
                center_coord = arr(node.getCoordinate()) * arr(cs_anno.scaling)\
                               / 10.
                break
    min_pos, max_pos = [center_coord - offset, center_coord + offset]
    return min_pos, max_pos, center_coord


def get_syn_from_cs_nml(path):
    """
    Parses p4 and az ids of synapse from pairwise contact site nml.
    :param path: str to contact_site.nml
    :return: int tuple p4 and az ID
    """
    cs_anno = au.loadj0126NML(path)[0]
    cs_nb = None
    for node in cs_anno.getNodes():
        if 'center_p4_az' in node.getComment():
            cs_nb = re.findall('cs(\d+)_', node.getComment())[0]
            break
    if cs_nb is None:
        print "Did not find center_p4_az, looking for first contact site."
        cs_nb = 'cs1_'
    p4_id = []
    az_id = []
    for node in cs_anno.getNodes():
        comment = node.getComment()
        if 'cs%s_' % cs_nb in comment:
            if '_p4-' in comment:
                p4_id.append(int(re.findall('_p4-(\d+)', node.getComment())[0]))
            if '_az-' in comment:
                az_id.append(int(re.findall('_az-(\d+)', node.getComment())[0]))
    return p4_id, az_id


def plot_obj(dh, paths=['/home/pschuber/Desktop/skel2skel/nml_obj/pre_04.k.zip',
                '/home/pschuber/Desktop/skel2skel/nml_obj/post_04.k.zip']):
    """
    Plots all segmentationDataObjects in certain bounding box found by center coords of
    contact site in contact_site.nml
    :param dh: DataHandler object
    :param paths: paths to skeleton nmls being plotted, same as in plot_post_pre()
    """
    cs_file_path = get_cs_path(paths)
    min_pos, max_pos, center = get_box_from_cs_nml(cs_file_path,
                                                   offset=arr([350, 350, 350]))
    real_min_pos, real_max_pos, real_center = get_box_from_cs_nml(cs_file_path,
                                                offset=arr([400, 400, 400]))
    nb_cpus = max(cpu_count()-1, 1)

    # get synapse objects
    [skel, mitos, p4, az] = load_anno_list([paths[1]])[0]
    skel1_mitos, skel1_p4, skel1_az = coords_from_anno(skel)

    [skel, mitos, p4, az] = load_anno_list([paths[0]])[0]
    skel2_mitos, skel2_p4, skel2_az = coords_from_anno(skel)
    filter_size=[4, 1594, 498]
    coords = arr(dh.p4.rep_coords) * dh.scaling / 10.
    bool_1 = np.all(coords >= min_pos, axis=1) & \
             np.all(coords <= max_pos, axis=1)
    ids = arr(dh.p4.ids)[bool_1][::12]
    print "Found %d p4." %len(ids)
    curr_objects = [dh.p4.object_dict[key] for key in ids if
                    dh.p4.object_dict[key].size >= filter_size[1]]
    pool = Pool(processes=nb_cpus)
    voxel_list = pool.map(helper_get_hull_voxels, curr_objects)
    pool.close()
    pool.join()
    p4 = arr([voxel for voxels in voxel_list for voxel in voxels])*skel.scaling
    p4 = np.concatenate((p4, skel1_p4, skel2_p4), axis=0)
    p4 = get_box_coords(p4, real_min_pos, real_max_pos)
    print "Plotting %d object hull voxels." % len(p4)
    coords = arr(dh.az.rep_coords) * dh.scaling / 10.
    bool_1 = np.all(coords >= min_pos, axis=1) & \
             np.all(coords <= max_pos, axis=1)
    ids = arr(dh.az.ids)[bool_1][::12]
    print "Found %d az." % len(ids)
    curr_objects = [dh.az.object_dict[key] for key in ids if
                    dh.az.object_dict[key].size >= filter_size[2]]
    pool = Pool(processes=nb_cpus)
    voxel_list = pool.map(helper_get_hull_voxels, curr_objects)
    pool.close()
    pool.join()
    az = arr([voxel for voxels in voxel_list for voxel in voxels])*skel.scaling
    az = np.concatenate((az, skel1_az, skel2_az), axis=0)
    az = get_box_coords(az, real_min_pos, real_max_pos)
    print "Plotting %d object hull voxels." % len(az)
    coords = arr(dh.mitos.rep_coords) * dh.scaling / 10.
    bool_1 = np.all(coords >= min_pos, axis=1) & \
             np.all(coords <= max_pos, axis=1)
    ids = arr(dh.mitos.ids)[bool_1][::12]
    print "Found %d mitos." %len(ids)
    curr_objects = [dh.mitos.object_dict[key] for key in ids if
                    dh.mitos.object_dict[key].size >= filter_size[0]]
    pool = Pool(processes=nb_cpus)
    voxel_list = pool.map(helper_get_hull_voxels, curr_objects)
    pool.close()
    pool.join()
    mitos = arr([voxel for voxels in voxel_list for voxel in voxels])*skel.scaling
    mitos = np.concatenate((mitos, skel1_mitos, skel2_mitos), axis=0)
    mitos = get_box_coords(mitos, real_min_pos, real_max_pos)
    print "Plotting %d object hull voxels." % len(mitos)

    # plot objects
    mlab.figure(1, bgcolor=(1, 1, 1), fgcolor=(0.5, 0.5, 0.5))
    mlab.clf()
    objects = [mitos, p4, az]
    colors = [(68./(68+85+255.), 85./(68+85+255.), 255./(68+85+255.)),
             (129./(129.+255+29), 255./(129.+255+29), 29/(129.+255+29)),
             (255./(255+44+93), 44./(255+44+93), 93./(255+44+93))]
    for obj, color in zip(objects, colors):
        if len(obj) == 0:
            print "Left out object.", color
            continue
        mlab.points3d(obj[:, 0], obj[:, 1], obj[:, 2], scale_factor=3.5,
                      mode='sphere', color=color, opacity=1.0)


def write_cs_p4_syn_skels(wd, dh=None, renew=False):
    """
    Plot workflow figure and save it!
    :param dest_path: str path to destination directory
    :param dh: DataHandler
    :param renew: Mapped skeletons
    """
    dest_path = wd + '/figures/',
    source = get_paths_of_skelID(['227', '83', '216'])
    paths = get_paths_of_skelID(['227', '83', '216'],
                                traced_skel_dir=wd + '/neurons/')
    if dh is None:
        dh = DataHandler(wd)
    if renew:
        enrich_tracings(source, dh=dh, overwrite=True, write_obj_voxel=True)
    print "Plotting skeletons", paths
    center_coord = arr([5130.,  4901.4,  6092. ])
    offset=arr([300, 300, 300])
    min_pos, max_pos = [center_coord - offset, center_coord + offset]

    mlab.figure(1, bgcolor=(1, 1, 1), fgcolor=(0.5, 0.5, 0.5))
    mlab.clf()
    plot_skel(paths[0], min_pos, max_pos, color=(1, 140./255, 0),
              hull_visualization=True, op=0.15)
    plot_skel(paths[1], min_pos, max_pos, color=(0.171, 0.485, 0.731),
              hull_visualization=True, op=0.15)
    plot_skel(paths[2], min_pos, max_pos, color=(1, 20/255., 0),
              hull_visualization=True, op=0.15)
    v_zoom = (125.40980215746366,
     31.992829913950441,
     1217.3956294932052,
     arr([ 5130.        ,  4997.59985352,  6092.        ]))
    mlab.view(*v_zoom)
    write_img2png(dest_path, 'cs_p4_syn_plot.png', mag=2)


def write_new_syn_cs(dest_path='/lustre/pschuber/figures/shawns_figure/',
                       dh=None, renew=False, save_vx_arr=False):
    """
    Plot workflow figure and save it!
    :param dest_path: str path to destination directory
    :param dh: DataHandler
    :param renew: Mapped skeletons
    """
    source = get_paths_of_skelID(['548', '88'])
    paths = get_filepaths_from_dir('/lustre/pschuber/figures/syns/nml_obj/')
    if dh is None:
        dh = DataHandler()
    if renew:
        enrich_tracings(source, dh=dh, overwrite=True, write_obj_voxel=True)
    print "Plotting skeletons", paths
    center_coord = np.array([2475, 2215, 540])*np.array([9., 9., 20.]) / 10.
    offset = np.array([200, 200, 200])
    min_pos, max_pos = [center_coord - offset, center_coord + offset]
    mlab.figure(1, bgcolor=(1, 1, 1), fgcolor=(0.5, 0.5, 0.5))
    mlab.clf()
    view = (13.139370011851103,
 72.617572162230971,
 890.03663369265792,
 arr([ 2279.98120117,  2035.88122559,  1080.        ]))

    plot_mapped_skel(paths[0], min_pos, max_pos, color=(1., 3/4., 0), op=0.8,
                     hull_visualization=True, save_vx_arr=save_vx_arr)
    plot_mapped_skel(paths[1], min_pos, max_pos, color=(3./4, 3./4, 3./4),
                     op=0.8, hull_visualization=True, save_vx_arr=save_vx_arr)
    plot_mapped_skel(paths[2], min_pos, max_pos, color=(1./4, 1./4, 1./4),
                     op=0.8, hull_visualization=True, save_vx_arr=save_vx_arr)
    if not save_vx_arr:
        mlab.view(*view)
        write_img2png(dest_path, 'test', mag=1)


def write_workflow_fig(dest_path='/lustre/pschuber/figures/spines/',
                       dh=None, renew=False):
    """
    Plot workflow figure and save it!
    :param dest_path: str path to destination directory
    :param dh: DataHandler
    :param renew: Mapped skeletons
    """
    source = get_paths_of_skelID(['120', '518'])
    paths = get_paths_of_skelID(['548', '88'], traced_skel_dir='/lustre/pschuber/'
                                                'mapped_soma_tracings/nml_obj/')
    if renew:
        if dh is None:
            dh = DataHandler()
            enrich_tracings(source, dh=dh, overwrite=True, write_obj_voxel=True)
    print "Plotting skeletons", paths
    cs_file_path = get_cs_path(paths)

    min_pos, max_pos, center_coord = get_box_from_cs_nml(cs_file_path,
                                                         offset=(3000, 3000, 3000))
    v_zoom = (24.724344563964589,
     37.420854959196014,
     7086.2440000000124,
     arr([ 5862.11725067,  7059.92156165,  4126.19249013]))

    mlab.figure(1, bgcolor=(1, 1, 1), fgcolor=(0.5, 0.5, 0.5))
    mlab.clf()
    plot_skel(paths[1], min_pos, max_pos, color=(0.7, 0.7, 0.7),
              hull_visualization=True, vis_property=True)
    mlab.view(*v_zoom)
    write_img2png(dest_path, 'big_skeleton_hull_colored_new', mag=4)


def write_cell_types(dest_path='/home/pschuber/figures/cell_types/',
                     offset=2100, rebuild=False):
    """
    Plot cell type figures with mapped objexts, axoness and skelet.
    :param dest_path: str path to destination directory
    :param rebuild: Mapped skeletons
    """
    paths = ['/home/pschuber/data/cell_type_examples/excitatory axons_3.k.zip',
             '/home/pschuber/data/cell_type_examples/medium spiny neuron_4.k.zip']
    center_coords = arr([[5788, 5238, 3085], [6058, 3822, 1064]]) * \
                        [9, 9, 20.] / 10
    type_names = ['excitory_axon', 'medium_spiny_neuron']
    if rebuild:
        remap_tracings(paths, mito_min_votes=35)
    for ii, path in enumerate(paths):
        assert os.path.isfile(path), "AnotationObject does not exist."
        print "Plotting skeleton", path
        min_pos, max_pos = [center_coords[ii] - offset,
                            center_coords[ii] + offset]
        mlab.figure(1, bgcolor=(1, 1, 1), fgcolor=(0.5, 0.5, 0.5))
        mlab.clf()
        plot_skel(path, min_pos, max_pos, color=(0.7, 0.7, 0.7),
                  hull_visualization=True, vis_property=True)
        plot_mapped_skel(path, min_pos, max_pos, color=(0.7, 0.7, 0.7),
                         hull_visualization=False)
        # mlab.view(*v_zoom)
        write_img2png(dest_path, type_names[ii], mag=5)


def write_synapse_cells(dest_path='/home/pschuber/figures/cell_types/',
                        offset=700):
    """
    Plot synapse types of cell pair.
    :param dest_path: str path to destination directory
    """
    syn_paths = ['/lustre/sdorkenw/synapse_matrices/'
                 'type_skel_31_496_nbsyn_3.nml',
                 '/lustre/sdorkenw/synapse_matrices/'
                 'type_skel_241_190_nbsyn_26.nml']
    center_coords = arr([[5404, 6825, 4390],  [7212, 5988, 2342]]) * \
                        [9, 9, 20.] / 10
    cs_dict = load_pkl2obj('/lustre/pschuber/st250_pt3_minvotes18/nml_obj/'
                           'contact_sites_new/cs_dict.pkl')
    for ii, path in enumerate(syn_paths):
        skel_ids = re.findall('(\d+)_(\d+)', path)[0]
        skel_paths = get_paths_of_skelID(skel_ids, traced_skel_dir='/lustre/'
                                         'pschuber/m_consensi_rr/nml_obj/')
        print "Plotting synapses", path
        mlab.figure(1, bgcolor=(1, 1, 1), fgcolor=(0.5, 0.5, 0.5))
        mlab.clf()
        synapse_tree = au.loadj0126NML(path)[0]
        cnt_symm = 0
        cnt_asym = 0
        all_ov_vxs = np.zeros((0, 3))
        for syn in synapse_tree.getNodes():
            comment = syn.getComment()
            syn_type, syn_key = re.findall('(\w+); (\w+)', comment)[0]
            if syn_type == 'asym' or syn_type == 'a':
                cnt_asym += 1
                syn_color = tuple(np.array([15, 105, 93]) / 255.)
            elif syn_type == 'symm' or syn_type == 'y':
                cnt_symm += 1
                syn_color = tuple(np.array([0, 0, 139]) / 255.)
            else:
                print comment
                continue
            if ii == 1:
                min_pos, max_pos = [center_coords[ii] - offset,
                    center_coords[ii] + offset]
                overlap_vx = get_box_coords(cs_dict[syn_key]['overlap_vx'] / 10.,
                                            min_pos, max_pos)
            else:
                overlap_vx = cs_dict[syn_key]['overlap_vx'] / 10.
                all_ov_vxs = np.concatenate((overlap_vx, all_ov_vxs), axis=0)
                min_pos, max_pos = [np.min(all_ov_vxs, axis=0) - 100,
                                    np.max(all_ov_vxs, axis=0) + 100]
            #print "Found %d synapse voxels" % len(overlap_vx)
            overlap_vx = overlap_vx[::2]
            mlab.points3d(overlap_vx[:, 0], overlap_vx[:, 1], overlap_vx[:, 2],
                          opacity=0.15, scale_factor=4.5, mode='sphere',
                          color=syn_color)
        print "Found %d symmetric and %d asymmetric synapses." % (cnt_symm,
                                                                  cnt_asym)
        # plot_skel(skel_paths[0], min_pos, max_pos,
        #                  color=tuple(arr([122, 2, 17])/255.),
        #                  hull_visualization=True, op=0.4)
        # plot_skel(skel_paths[1], min_pos, max_pos, color=(0.3, 0.3, 0.3),
        #                  hull_visualization=True, op=0.4)
        # write_img2png(dest_path, 'syn_vis'+skel_ids[0]+'_'+skel_ids[1], mag=1)


def write_synapse_cells_voxel(dest_path='/lustre/pschuber/figures/syns/'):
    """
    Plot synapse types of cell pair.
    :param dest_path: str path to destination directory
    """
    syn_paths = ['/lustre/sdorkenw/synapse_matrices/'
                 'type_skel_31_496_nbsyn_3.nml',
                 '/lustre/sdorkenw/synapse_matrices/'
                 'type_skel_241_190_nbsyn_26.nml', '371_472', '1_578']
    cs_dict = load_pkl2obj('/lustre/pschuber/st250_pt3_minvotes18/nml_obj/'
                           'contact_sites_new/cs_dict.pkl')
    min_pos = np.zeros(3)
    max_pos = np.array([99999, 99999, 99999])
    for ii, path in enumerate(syn_paths):
        if ii < 2:
            continue
        skel_ids = re.findall('(\d+)_(\d+)', path)[0]
        print "Writing hull and synapse voxel of skeleton pair %s and %s." \
        % (skel_ids[0], skel_ids[1])
        curr_dest = dest_path + skel_ids[0] + '_' + skel_ids[1] + '/'
        if not os.path.isdir(curr_dest):
            os.makedirs(curr_dest)
        skel_paths = get_paths_of_skelID(skel_ids, traced_skel_dir='/lustre/'
                                         'pschuber/mapped_soma_tracings/nml_obj/')
        # print "Writing synapses", path
        # print "Loaded skels", skel_ids, "from", skel_paths, "writing to", curr_dest
        # synapse_tree = au.loadj0126NML(path)[0]
        # cnt_symm = 0
        # cnt_asym = 0
        # syn_sym = np.zeros((0, 3))
        # syn_sym_normals = np.zeros((0, 3))
        # syn_asym = np.zeros((0, 3))
        # syn_asym_normals = np.zeros((0, 3))
        # for syn in synapse_tree.getNodes():
        #     comment = syn.getComment()
        #     syn_type, syn_key = re.findall('(\w+); \d+; (\w+)', comment)[0]
        #     syn_vx = cs_dict[syn_key]['overlap_vx'] / 10.
        #     syn_vx_normal = calc_obj_normals(syn_vx)
        #     if syn_type == 'asym' or syn_type == 'a':
        #         cnt_asym += 1
        #         syn_asym = np.concatenate((syn_asym, syn_vx), axis=0)
        #         syn_asym_normals = np.concatenate((syn_asym_normals,
        #                                            syn_vx_normal), axis=0)
        #     elif syn_type == 'symm' or syn_type == 'y':
        #         cnt_symm += 1
        #         syn_sym = np.concatenate((syn_sym, syn_vx), axis=0)
        #         syn_sym_normals = np.concatenate((syn_sym_normals,
        #                                           syn_vx_normal), axis=0)
        #     else:
        #         print comment
        #         continue
        write_syns_as_types(skel_paths[0], curr_dest)
        write_syns_as_types(skel_paths[1], curr_dest)
        # print "Normal shape:", syn_sym_normals.shape
        # print "Normal shape:", syn_asym_normals.shape
        # print "%d sym synapse voxel with shape %s" % (len(syn_sym),
        #                                              str(syn_sym.shape))
        # print "%d asym synapse voxel with shape %s" % (len(syn_asym),
        #                                               str(syn_asym.shape))
        # hull2text(syn_sym, syn_sym_normals, curr_dest + 'syn_sym.xyz')
        # hull2text(syn_asym, syn_asym_normals, curr_dest + 'syn_asym.xyz')
        # print "Found %d symmetric and %d asymmetric synapses." % (cnt_symm,
        #                                                           cnt_asym)
        plot_mapped_skel(skel_paths[0], min_pos, max_pos, save_vx_dir=curr_dest,
                  hull_visualization=True, plot_objects=[True, True, True])
        plot_mapped_skel(skel_paths[1], min_pos, max_pos, save_vx_dir=curr_dest,
                  hull_visualization=True, plot_objects=[True, True, True])


def write_syns_as_types(path, dest_dir):
    skel = load_mapped_skeleton(path, True, False)[0]
    az_dict = load_objpkl_from_kzip(path)[2]
    voxel_lists = [[], []]
    normal_list = [[], []]
    dummy_skel = NewSkeleton()
    dummy_anno = SkeletonAnnotation()
    dummy_anno.scaling = [9, 9, 20]
    for val in az_dict.object_dict.values():
        type = syn_sign_prediction(val.voxels)
        # voxel_lists[type] += list(val.hull_voxels)
        # syn_vx_normal = calc_obj_normals(val.hull_voxels)
        # normal_list[type] += list(syn_vx_normal)
        node = SkeletonNode().from_scratch(
            dummy_anno, val.rep_coord[0], val.rep_coord[1],
            val.rep_coord[2], radius=(val.size/4./np.pi*3)**(1/3.))
        node.setPureComment('%d' % type)
        dummy_anno.addNode(node)
    dummy_skel.add_annotation(dummy_anno)
    dummy_skel.to_kzip(dest_dir + skel.filename + 'syn_anno.k.zip')
    # hull2text(np.array(voxel_lists[0]), np.array(normal_list[0]),
    #           dest_dir + skel.filename + 'syn_asym.xyz')
    # hull2text(np.array(voxel_lists[1]), np.array(normal_list[1]),
    #           dest_dir + skel.filename + 'syn_sym.xyz')


def calc_obj_normals(object_vx):
    """
    Calculate normals approximated by vector to center voxel.
    :param object_vx: array of object voxel coords
    :return: array normalized normals
    """
    center_coords = np.mean(object_vx, axis=0)
    normals = object_vx - center_coords
    normals /= np.linalg.norm(normals)
    return normals


def write_cell_type_cells():
    """
    Plot synapse types of cell pair.
    :param dest_path: str path to destination directory
    """
    paths = ['/home/pschuber/data/cell_type_examples/excitatory axons_3.k.zip',
             '/home/pschuber/data/cell_type_examples/medium spiny neuron_4.k.zip']
    min_pos = np.zeros(3)
    max_pos = np.array([99999, 99999, 99999])
    for path in paths:
        plot_mapped_skel(path, min_pos, max_pos,
                         save_vx_dir='/lustre/pschuber/gt_cell_types/cells_vx/',
                         color=tuple(arr([122, 2, 17])/255.),
                         hull_visualization=True, op=0.4)


def write_axoness_cell(skel_path=None):
    if skel_path is None:
        skel_path = get_paths_of_skelID(['182'], '/lustre/pschuber/mapped_soma_tracings/nml_obj/')[0]
    skel, _, _, _, _ = load_mapped_skeleton(skel_path, True, False)
    hull = skel._hull_coords
    normals = skel._hull_normals
    node_coords = []
    node_ids = []
    axon_ids = []
    dend_ids = []
    soma_ids = []
    for ii, n in enumerate(skel.getNodes()):
        node_coords.append(n.getCoordinate_scaled())
        if n.data['axoness_pred'] == '1':
            axon_ids.append(ii)
        elif n.data['axoness_pred'] == '0':
            dend_ids.append(ii)
        elif int(float(n.data["axoness_pred"])) == 2:
            soma_ids.append(ii)
        else:
            print "Unknown prediction vun de axoness."
    print "found %d soma nodes." % len(soma_ids)
    print "Assigning hull points."
    skel_tree = spatial.cKDTree(node_coords)
    dist, nn_ixs = skel_tree.query(hull, k=1)
    soma_hull = []
    soma_hull_normals = []
    axon_hull = []
    axon_hull_normals = []
    dend_hull = []
    dend_hull_normals = []
    for ii, ix in enumerate(nn_ixs):
        if ix in soma_ids:
            soma_hull.append(hull[ii])
            soma_hull_normals.append(normals[ii])
        elif ix in axon_ids:
            axon_hull.append(hull[ii])
            axon_hull_normals.append(normals[ii])
        else:
            dend_hull.append(hull[ii])
            dend_hull_normals.append(normals[ii])
    dest = '/lustre/pschuber/figures/hulls/axoness/'
    hull2text(arr(soma_hull), arr(soma_hull_normals), dest + 'soma.xyz')
    hull2text(arr(axon_hull), arr(axon_hull_normals), dest + 'axon.xyz')
    hull2text(arr(dend_hull), arr(dend_hull_normals), dest + 'dend.xyz')


def write_myelin_cell(skel_path=None):
    max_myelin_nbs = 0
    myelin_nb_l = []
    all_paths = get_filepaths_from_dir('/lustre/pschuber/mapped_soma_tracings/nml_obj/')
    for skel_path in all_paths:
        # if skel_path is None:
        #     skel_path = get_paths_of_skelID(['190'], '/lustre/pschuber/mapped_soma_tracings/nml_obj/')[0]
        skel, _, _, _, _ = load_mapped_skeleton(skel_path, True, False)
        hull = skel._hull_coords
        # print len(hull)
        normals = skel._hull_normals
        normals = normals[hull[:, 0] < 110000]
        hull = hull[hull[:, 0] < 110000]
        normals = normals[hull[:, 1] < 110000]
        hull = hull[hull[:, 1] < 110000]
        normals = normals[hull[:, 2] < 110000]
        hull = hull[hull[:, 2] < 110000]
        # print len(hull)
        node_coords = []
        myelin_ids = []
        for ii, n in enumerate(skel.getNodes()):
            node_coords.append(n.getCoordinate_scaled())
            if n.data['myelin_pred'] == '1':
                myelin_ids.append(ii)
        print "Found %d myelin nodes." % len(myelin_ids)
        print "Assigning hull points."
        myelin_nb_l.append(len(myelin_ids))
        if len(myelin_ids) > max_myelin_nbs:
            max_myelin_nbs = len(myelin_ids)
            max_path = skel_path
    skel, _, _, _, _ = load_mapped_skeleton(max_path, True, False)
    hull = skel._hull_coords
    # print len(hull)
    normals = skel._hull_normals
    normals = normals[hull[:, 0] < 110000]
    hull = hull[hull[:, 0] < 110000]
    normals = normals[hull[:, 1] < 110000]
    hull = hull[hull[:, 1] < 110000]
    normals = normals[hull[:, 2] < 110000]
    hull = hull[hull[:, 2] < 110000]
    # print len(hull)
    node_coords = []
    myelin_ids = []
    for ii, n in enumerate(skel.getNodes()):
        node_coords.append(n.getCoordinate_scaled())
        if n.data['myelin_pred'] == '1':
            myelin_ids.append(ii)
    skel_tree = spatial.cKDTree(node_coords)
    dist, nn_ixs = skel_tree.query(hull, k=1)
    myelin_hull = []
    myelin_hull_normals = []
    normal_hull = []
    normal_hull_normals = []
    for ii, ix in enumerate(nn_ixs):
        if ix in myelin_ids:
            myelin_hull.append(hull[ii])
            myelin_hull_normals.append(normals[ii])
        else:
            normal_hull.append(hull[ii])
            normal_hull_normals.append(normals[ii])
    print "Found:\n", arr(all_paths)[np.argsort(myelin_nb_l)[-10:]], "with nodes:\n",\
    np.sort(myelin_nb_l)[-10:]
    dest = '/lustre/pschuber/figures/hulls/myelin/'
    hull2text(arr(myelin_hull)/10., arr(myelin_hull_normals), dest + 'myelin_hull.xyz')
    hull2text(arr(normal_hull, dtype=np.int)/10., arr(normal_hull_normals), dest + 'normal_hull.xyz')


def get_celltype_samples():
    gt_path='/lustre/pschuber/gt_cell_types/'
    dest = '/lustre/pschuber/figures/hulls/cell_type_samples/'
    cell_type_pred_dict = load_pkl2obj(gt_path + 'cell_pred_dict.pkl')
    type_dict = {0: [], 1: [], 2: [], 3: []}
    for k, v in cell_type_pred_dict.iteritems():
        type_dict[v].append(k)
    sample_list = []
    for k, v in type_dict.iteritems():
        sample_ids = np.random.randint(0, len(v), 3)
        paths = get_paths_of_skelID(arr(v)[sample_ids],
                    traced_skel_dir='/lustre/pschuber/mapped_soma_tracings/nml_obj/')
        for p in paths:
            print "Copy file from", p
            dir, fname = os.path.split(p)
            shutil.copyfile(p, dest+'type'+str(k)+'_'+fname)


def neck_head_az_size_hist(cs_dir, recompute=False):
    """

    :param cs_dir:
    :param recompute:
    :return:
    """
    prop_dict = load_pkl2obj(cs_dir + '/property_dict.pkl')
    cs_dict = load_pkl2obj(cs_dir + '/cs_dict.pkl')
    consensi_celltype_label = load_pkl2obj('/lustre/pschuber/gt_cell_types/'
                                        'consensi_celltype_labels_reviewed3.pkl')
    spine_types = [[], [], []]
    syn_types = [[], [], []]
    az_sizes = [[], [], []]
    if recompute:
        for old_key, val in prop_dict.iteritems():
            key = None
            new_names = convert_to_standard_cs_name(old_key)
            for var in new_names:
                try:
                    center_coord = cs_dict[var]['center_coord']
                    key = var
                    break
                except KeyError:
                    continue
            if key is None:
                print "Didnt find CS with key %s" % old_key
            skel_ids = val.keys()
            skel1, skel2 = val.values()
            ax1 = skel1["axoness"]
            ax2 = skel2["axoness"]
            if (ax1 == 2) or (ax2 == 2):
                print "Found soma syn in ", key
                continue
            if ax1 == ax2:
                continue
            if cs_dict[key]["syn_pred"] == 0:
                continue
            spiness = [skel1["spiness"], skel2["spiness"]]
            syn_spiness = spiness[[ax1, ax2].index(0)]
            dend_id = int(skel_ids[[ax1, ax2].index(0)])
            try:
                class_nb = consensi_celltype_label[dend_id]
            except KeyError:
                continue
            if class_nb != 1:
                continue
            spine_types[syn_spiness].append(key)
            syn_sign = syn_sign_prediction(cs_dict[var]['overlap_vx']/np.array([9., 9., 20.]))
            syn_types[syn_spiness].append(syn_sign)
            az_sizes[syn_spiness].append(cs_dict[key]["overlap_area"])
        print "Found shaft/head/neck synapses:\t%d/%d/%d" % (len(spine_types[0]),
                                                             len(spine_types[1]),
                                                             len(spine_types[2]))
        print "Mean sizes:\t\t\t%0.3f+-%0.3f, %0.3f+-%0.3f, %0.3f+-%0.3f" % \
              (np.mean(az_sizes[0]), np.std(az_sizes[0]), np.mean(az_sizes[1]),
               np.std(az_sizes[1]), np.mean(az_sizes[2]), np.std(az_sizes[2]))
        az_sizes[0] = np.array(az_sizes[0])
        az_sizes[1] = np.array(az_sizes[1])
        az_sizes[2] = np.array(az_sizes[2])
        syn_types[0] = np.array(syn_types[0])
        syn_types[1] = np.array(syn_types[1])
        syn_types[2] = np.array(syn_types[2])
        np.save("/lustre/pschuber/figures/syns/spines/shaft_sizes.npy", az_sizes[0])
        np.save("/lustre/pschuber/figures/syns/spines/head_sizes.npy", az_sizes[1])
        np.save("/lustre/pschuber/figures/syns/spines/neck_sizes.npy", az_sizes[2])
        np.save("/lustre/pschuber/figures/syns/spines/shaft_synsign.npy", syn_types[0])
        np.save("/lustre/pschuber/figures/syns/spines/head_synsign.npy", syn_types[1])
        np.save("/lustre/pschuber/figures/syns/spines/neck_synsign.npy", syn_types[2])
    else:
        az_sizes[0] = np.load("/lustre/pschuber/figures/syns/spines/shaft_sizes.npy")
        az_sizes[1] = np.load("/lustre/pschuber/figures/syns/spines/head_sizes.npy")
        az_sizes[2] = np.load("/lustre/pschuber/figures/syns/spines/neck_sizes.npy")
        syn_types[0] = np.load("/lustre/pschuber/figures/syns/spines/shaft_synsign.npy")
        syn_types[1] = np.load("/lustre/pschuber/figures/syns/spines/head_synsign.npy")
        syn_types[2] = np.load("/lustre/pschuber/figures/syns/spines/neck_synsign.npy")
    # shaft_ratio = np.sum(az_sizes[0][syn_types[0] == 1]) / np.sum(az_sizes[0])
    # head_ratio = np.sum(az_sizes[1][syn_types[1] == 1]) / np.sum(az_sizes[1])
    # neck_ratio = np.sum(az_sizes[2][syn_types[2] == 1]) / np.sum(az_sizes[2])
    shaft_ratio = np.sum(syn_types[0] == 1) / float(len(syn_types[0]))
    head_ratio = np.sum(syn_types[1] == 1) / float(len(syn_types[1]))
    neck_ratio = np.sum(syn_types[2] == 1) / float(len(syn_types[2]))
    loc_syn_type_plot([head_ratio, neck_ratio, shaft_ratio], save_path=\
    "/lustre/pschuber/figures/syns/spines/loc_syn_type.png")
    return


def loc_syn_type_plot(data, save_path="/lustre/sdorkenw/figures/" \
                                                "loc_syn_type.png"):
    fig, ax = plt.subplots()#aspect=1)
    fig.patch.set_facecolor('white')

    names = ["spine head", "spine neck", "shaft"]

    ax.tick_params(axis='x', which='major', labelsize=22, direction='out',
                   length=4, width=3,  right="off", top="off", pad=10)
    ax.tick_params(axis='y', which='major', labelsize=24, direction='out',
                   length=4, width=3,  right="off", top="off", pad=10)

    ax.tick_params(axis='x', which='minor', labelsize=12, direction='out',
                   length=0, width=1, right="off", top="off", pad=10)
    ax.tick_params(axis='y', which='minor', labelsize=12, direction='out',
                   length=0, width=1, right="off", top="off", pad=10)

    ax.spines['left'].set_linewidth(3)
    ax.spines['bottom'].set_linewidth(3)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.set_ylabel("Ratio of symmetric Synapses", fontsize=26)

    ind = np.arange(3)                # the x locations for the groups

    width = .6

    ax.bar(ind-width/2., data, width,
           color=".3",
           edgecolor='k',
           linewidth=2.,
           align='center',
           label='contact site evaluation')

    # plt.legend(loc="upper left", frameon=False, prop={'size': 22})

    # plt.xlim([-0.5, len(data)])
    plt.xticks(ind, names, rotation=0, fontsize=22)
    # ax.yaxis.set_major_locator(mticker.FixedLocator(np.logspace(0, 12, num=4, base=10.)))
    ax.set_ylim(0., 1)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    # plt.show(block=False)


def get_cmap(N, cmap):
    '''Returns a function that maps each index in 0, 1, ... N-1 to a distinct
    RGB color.'''
    color_norm  = colors.Normalize(vmin=0, vmax=N-1)
    scalar_map = cmx.ScalarMappable(norm=color_norm, cmap=cmap)
    def map_index_to_rgb_color(index):
        return scalar_map.to_rgba(index)[:3]
    return map_index_to_rgb_color


def norm_rgb_color(color):
    print color
    normed_color = (color[0]/255., color[1]/255., color[2]/255.)
    print normed_color
    return normed_color


def plot_csarea_nbsyns(pre=0, post=1):
    """
    Plots two keys of phildict as scatter plot.
    :return:
    """
    syn_props = load_pkl2obj('/lustre/sdorkenw/synapse_matrices/phil_dict_all.pkl')
    consensi_celltype_label = load_pkl2obj('/lustre/pschuber/gt_cell_types/'
                                        'consensi_celltype_labels_reviewed3.pkl')
    cell_type_pred_dict = load_pkl2obj('/lustre/pschuber/gt_cell_types/loo_cell_pred_dict_novel.pkl')
    syn_area_l = []
    cum_syn_area_l = []
    cs_area_l = []
    nb_syn_l = []
    nb_cs_l = []
    for k, v in syn_props.iteritems():
        if v['total_size_area'] == 0:
            continue
        id1, id2 = re.findall('(\d+)_(\d+)', k)[0]
        try:
            type1 = consensi_celltype_label[int(id1)]
            type2 = consensi_celltype_label[int(id2)]
        except:
            continue
        if (type1 != pre) or (type2 != post):
            continue
        nb_syns = len(v['sizes_area'])
        nb_syn_l.append(nb_syns)
        # nb_cs_l.append(nb_cs)
        syn_area_l += list(np.array(v['sizes_area'])[np.array(v['sizes_area'])!=0])
        cum_syn_area_l.append(v['total_size_area'])
        cs_area_l.append(v['total_cs_area'])
    print "Plotting %d points." % len(syn_area_l)
    # plt.figure()
    # ax = plt.subplot(111)
    # ax.scatter(np.array(cs_area_l), np.array(nb_cs_l))
    # plt.xlabel('contact size area in square microns')
    # plt.ylabel('Number of cs')
    # plt.figure()
    # ax = plt.subplot(111)
    # ax.scatter(np.array(syn_area_l), np.array(nb_syn_l))
    # plt.xlabel('synapse area in square microns')
    # plt.ylabel('Number of syns')
    # plt.figure()
    # ax = plt.subplot(111)
    # ax.scatter(np.array(cs_area_l), np.array(nb_syn_l))
    # plt.xlabel('contact size area in square microns')
    # plt.ylabel('Number of syns')
# Use the histogram function to bin the data
    counts, bin_edges = np.histogram(np.array(syn_area_l), bins=50)
    counts_nb_syns, bin_edges_nb_syns = np.histogram(np.array(nb_syn_l), bins=30)
    counts_cum, bin_edges_cum = np.histogram(np.array(cum_syn_area_l), bins=30)

    # Now find the cdf
    cdf = np.cumsum(counts)
    cdf_nb_syns = np.cumsum(counts_nb_syns)
    cdf_cum= np.cumsum(counts_nb_syns)

    plt.figure()
    ax = plt.subplot(111)
    ax.plot(bin_edges_nb_syns[1:], cdf_nb_syns.astype(np.float)/np.max(cdf_nb_syns))
    plt.xlim(0, 32)
    plt.show()

    save = np.concatenate((bin_edges[1:][None, ], (cdf.astype(np.float)/np.max(cdf))[None, ]), axis=0)
    save_cum = np.concatenate((bin_edges_cum[1:][None, ], (cdf_cum.astype(np.float)/np.max(cdf_cum))[None, ]), axis=0)
    save_nb_syns = np.concatenate((bin_edges_nb_syns[1:][None, ],
            (cdf_nb_syns.astype(np.float)/np.max(cdf_nb_syns))[None, ]), axis=0)
    plt.xlabel('Synapse area in microns')
    plt.ylabel('Counts')
    # calculate the proportional values of samples
    np.save('/lustre/pschuber/figures/synapse_cum/%d_%d.npy' % (pre, post), save)
    np.save('/lustre/pschuber/figures/synapse_cum/nb_syns_%d_%d.npy' % (pre, post), save_nb_syns)
    np.save('/lustre/pschuber/figures/synapse_cum/cum_%d_%d.npy' % (pre, post), save_cum)
    # plt.show(block=False)
    # plt.savefig('/lustre/pschuber/figures/synapse_cum/%d_%d.png' % (pre, post))


def plot_msn_spinedensities():
    gpi_msn = [21,  62,  73, 136, 148, 167, 170, 227, 261, 270, 307, 321, 354,
               371, 377, 385, 397, 402, 404, 416, 496, 508, 511]
    gpe_msn = [ 1,   5,  21,  23,  29,  29,  43,  58,  66,  73, 115, 116, 182,
                187, 223, 364, 397, 423, 429, 455, 480, 582, 588, 606, 611, 675]
    # gpi_skel_paths = get_paths_of_skelID(gpi_msn)
    # gpe_skel_paths = get_paths_of_skelID(gpe_msn)
    gpi_densities = []
    gpe_densities = []
    feat_names = np.load('/lustre/pschuber/gt_cell_types/feat_names.npy')
    feat_ix = np.where(feat_names == 'sh density (dend.)')[0][0]
    skeleton_ids, skeleton_feats = load_celltype_feats('/lustre/pschuber/gt_cell_types/')
    assert skeleton_feats.shape[1] == len(feat_names), "Features corrupted."
    for id in gpi_msn:
        cell_ix = np.where(skeleton_ids == id)[0][0]
        gpi_densities.append(skeleton_feats[cell_ix][feat_ix])
    for id in gpe_msn:
        cell_ix = np.where(skeleton_ids == id)[0][0]
        gpe_densities.append(skeleton_feats[cell_ix][feat_ix])
    fig, ax = plt.subplots()
    ax.hist(arr(gpi_densities), normed=True, range=[0, 0.7], alpha=0.8,
            bins=20, label='GPi')
    ax.hist(arr(gpe_densities), normed=True, range=[0, 0.7], alpha=0.8,
            bins=20, label='GPe')
    plt.xlabel('sh density [um^-1]')
    plt.legend()
    plt.show(block=False)
    print "Mean of GPe: %0.4f and GPi: %0.4f" % (np.mean(gpe_densities),
                                                 np.mean(gpi_densities))
    print "T-test:", stats.ttest_ind(arr(gpi_densities), arr(gpe_densities), False)
    msn_feats = []
    all_densities = []
    cell_type_pred_dict = load_pkl2obj('/lustre/pschuber/gt_cell_types/cell_pred_dict.pkl')
    for k, v in cell_type_pred_dict.iteritems():
        if v == 1:
            cell_ix = np.where(skeleton_ids == k)[0][0]
            gpi_densities.append(skeleton_feats[cell_ix][feat_ix])
            msn_feats.append(skeleton_feats[cell_ix])
    fig, ax = plt.subplots()
    ax.hist(arr(gpi_densities), normed=True, range=[0, 0.7], alpha=0.8,
            bins=40)
    plt.show(block=False)
    pca = PCA(n_components=2)
    msn_feats = pca.fit_transform(msn_feats)
    plt.scatter(msn_feats[:, 0], msn_feats[:, 1])
    plt.show()
    raise()


def plot_ray_proba(nb):
    probas = np.load("/home/pschuber/membrane_probas_skel1.npy")
    radii = np.load("//home/pschuber/membrane_radii_skel1.npy") * 10
    fig, ax = plt.subplots()
    ax.tick_params(axis='x', which='major', labelsize=22, direction='out',
                   length=4, width=3,  right="off", top="off", pad=10)
    ax.tick_params(axis='y', which='major', labelsize=22, direction='out',
                   length=4, width=3,  right="off", top="off", pad=10)

    ax.tick_params(axis='x', which='minor', labelsize=12, direction='out',
                   length=0, width=1, right="off", top="off", pad=10)
    ax.tick_params(axis='y', which='minor', labelsize=12, direction='out',
                   length=0, width=1, right="off", top="off", pad=10)
    for ii, label in enumerate(ax.yaxis.get_ticklabels()):
        if (ii==len(ax.yaxis.get_ticklabels())-1):
            label.set_visible(False)
        if ii % 2 == 0:
            continue
        label.set_visible(False)
    plt.xticks(np.arange(0, radii[nb]+50, 25))
    for ii, label in enumerate(ax.xaxis.get_ticklabels()):
        if ii==len(ax.xaxis.get_ticklabels())-1:
            label.set_visible(False)
        if ii % 2 == 0:
            continue
        label.set_visible(False)
    ax.spines['left'].set_linewidth(3)
    ax.spines['bottom'].set_linewidth(3)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.plot(np.linspace(0, radii[nb], len(probas[nb][0])),
             np.cumsum(probas[nb][0])/255., '0.3', lw=7)
    plt.hlines(2.2, 0, radii[nb], 'r', lw=7)
    ax.set_xlabel('Ray travel distance [nm]', fontsize=26)
    ax.set_ylabel('Cumulated membrane prob.', fontsize=26)
    plt.xlim(0, radii[nb])
    plt.tight_layout()
    plt.savefig("/lustre/pschuber/figures/ray_casting.png", dpi=600)
    plt.show()


def plot_obj_density(recompute=False, obj='mito', property='axoness_pred', value=1,
                     return_abs_density=True):
    """
    Plots two keys of phildict as scatter plot.
    :return:
    """
    consensi_celltype_label = load_pkl2obj('/lustre/pschuber/gt_cell_types/'
                                        'consensi_celltype_labels_reviewed3.pkl')
    cell_type_pred_dict = load_pkl2obj('/lustre/pschuber/gt_cell_types/'
                                       'loo_cell_pred_dict_novel.pkl')
    if return_abs_density:
        dest_path = "/lustre/pschuber/figures/%s_densities_axoness%d.npy" %\
                        (obj, value)
    else:
        dest_path = "/lustre/pschuber/figures/%s_densities_axoness%d_vol.npy" %\
                        (obj, value)
    if not os.path.isfile(dest_path) or recompute:
        obj_densities = [[], [], [], []]
        for k, v in cell_type_pred_dict.iteritems():
            cell_path = get_paths_of_skelID([k], '/lustre/pschuber/'
                        'mapped_soma_tracings/nml_obj/')[0]
            obj_density = get_obj_density(cell_path, property=property,
                                           value=value, obj=obj,
                                          return_abs_density=return_abs_density)
            obj_densities[int(v)].append(obj_density)

            np.save(dest_path, obj_densities)
    else:
        obj_densities = np.load(dest_path)


def plot_meansyn_celltypes(pre=0, post=1):
    """
    Plots two keys of phildict as scatter plot.
    :return:
    """
    syn_props = load_pkl2obj('/lustre/sdorkenw/synapse_matrices/phil_dict.pkl')
    cell_type_pred_dict = load_pkl2obj('/lustre/pschuber/gt_cell_types/loo_cell_pred_dict_novel.pkl')
    nb_syn_l = []
    syns_from_axon_to_post = []
    for k, v in syn_props.iteritems():
        if v['total_size_area'] == 0:
            continue
        id1, id2 = re.findall('(\d+)_(\d+)', k)[0]
        try:
            type1 = cell_type_pred_dict[int(id1)]
            type2 = cell_type_pred_dict[int(id2)]
        except:
            continue
        if (type1 != pre) or (type2 != post):
            continue
        if np.any(np.array(v['partner_axoness']) == 1):
            indiv_syn_sizes = np.array(v['sizes_area'])
            indiv_syn_axoness = (indiv_syn_sizes == 1)[indiv_syn_sizes != 0]
            nb_syns = len(indiv_syn_sizes[indiv_syn_sizes != 0][~indiv_syn_axoness])
        else:
            nb_syns = len(np.array(v['sizes_area'])[np.array(v['sizes_area']) != 0])
        if nb_syns == 0:
            continue
        syns_from_axon_to_post += [nb_syns]
    np.save('/lustre/pschuber/figures/synapse_cum/meansyns_from%d_to%d.npy' %
            (pre, post), arr(syns_from_axon_to_post))


def calc_meansyns_wrapper():
    for i in range(4):
        for j in range(1,4):
            plot_meansyn_celltypes(i, j)


def add_anno_to_mayavi_window(anno, node_scaling=1.0, override_node_radius=500.,
                              edge_radius=250., show_outline=False,
                              dataset_identifier='', opacity=1):
        '''

        Adds an annotation to a currently open mayavi plotting window

        Parameters: anno: annotation object
        node_scaling: float, scaling factor for node radius
        edge_radius: float, radius of tubes for each edge

        '''

        # plot the nodes
        # x, y, z are numpy arrays, s as well

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

        # separate x, y and z; mlab needs that
        # datasetDims = np.array(anno.datasetDims)


        x = [el[0] for el in sc[0].tolist()]
        y = [el[0] for el in sc[1].tolist()]
        z = [el[0] for el in sc[2].tolist()]
        if override_node_radius > 0.:
            s = [override_node_radius] * len(nodes)
        else:
            s = [node.getDataElem('radius') for node in nodes]

        s = np.array(s)
        s = s * node_scaling
        # s[0] = 5000
        # extent=[1, 108810, 1, 106250, 1, 115220]
        # raise
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

        return