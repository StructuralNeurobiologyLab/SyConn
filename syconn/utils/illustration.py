# -*- coding: utf-8 -*-
import matplotlib.cm as cmx
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from multiprocessing import Pool
from syconn.contactsites import convert_to_standard_cs_name
from syconn.brainqueries import enrich_tracings
from syconn.processing.features import get_obj_density
from syconn.processing.synapticity import syn_sign_prediction
from syconn.utils.datahandler import *
try:
    from mayavi import mlab
except (ImportError, ValueError), e:
    print "Could not load mayavi. Please install vtk and then mayavi."
__author__ = 'pschuber'


def init_skel_vis(skel, min_pos, max_pos, hull_visualization=True, op=0.15,
                  save_vx_dir=None, cmap=None):
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
    hull_points = get_box_coords(skel.hull_coords / 10., min_pos, max_pos)
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
        [skel, mitos, vc, sj, soma] = load_anno_list([path])[0]
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
    vc_hulls = skel.vc_hull_coords / 10
    sj_hulls = skel.sj_hull_coords / 10
    return mito_hulls, vc_hulls, sj_hulls


def plot_mapped_skel(path, min_pos, max_pos, color=(1./3, 1./3, 1./3), op=0.1,
                     only_syn=False, cs_path=None, hull_visualization=True,
                     save_vx_dir=None, plot_objects=(True, True, True),
                     color_sj=False):
    """
    Plots the mapped skeleton at path using mayavi. Annotation file must
    contain trees 'vc', 'sj', 'mitos' and 'skeleton'
    :param path: string to .k.zip or .nml of mapped annotation
    :param only_syn: bool Only plot synapse
    :param cs_path: str Hpath to contact site nml of these skeletons.
    :return:
    """
    [skel, mitos, vc, sj, soma] = load_anno_list([path])[0]
    print "Loaded mapped skeleton %s from %s and write it to %s" % (
        skel.filename, path, str(save_vx_dir))
    skel.color = color
    if only_syn:
        if cs_path is None:
            return
        mitos = []
        vc_id, sj_id = get_syn_from_cs_nml(cs_path)
        if vc_id is None and sj_id is None:
            mitos, vc, sj = coords_from_anno(skel)
        else:
            vc = []
            sj = []
            for id in vc_id:
                vc += list(skel.vc_hull_coords[skel.vc_hull_ids == id] / 10)
            for id in sj_id:
                sj += list(skel.sj_hull_coords[skel.sj_hull_ids == id] / 10)
            vc = arr(vc)
            sj = arr(sj)
    else:
        mitos, vc, sj = coords_from_anno(skel)
    if not (mitos == []) and mitos.ndim == 1:
        mitos = mitos[:, None]
    if not (vc == []) and vc.ndim == 1:
        vc = vc[:, None]
    if not (sj == []) and sj.ndim == 1:
        sj = sj[:, None]
    print "Found %d mitos, %d vc, %d sj in %s." % (len(set(list(skel.mito_hull_ids))), len(set(list(skel.vc_hull_ids))),
                                                   len(set(list(skel.sj_hull_ids))), skel.filename)
    mitos = get_box_coords(mitos, min_pos, max_pos)
    vc = get_box_coords(vc, min_pos, max_pos)
    sj = get_box_coords(sj, min_pos, max_pos)
    if save_vx_dir is None:
        while (len(mitos)+len(vc)+len(sj) > 0.4e6):
            print "Downsmapling of objects!!"
            mitos = mitos[::4]
            vc = vc[::2]
            sj = sj[::2]
            skel.sj_hull_ids = skel.sj_hull_ids[::2]
            skel.sj_hull_coords = skel.sj_hull_coords[::2]
    else:
        while (len(mitos)+len(vc)+len(sj) > 5e6):
            print "Downsmapling of objects!!"
            mitos = mitos[::4]
            vc = vc[::2]
            sj = sj[::2]
            skel.sj_hull_ids = skel.sj_hull_ids[::2]
            skel.sj_hull_coords = skel.sj_hull_coords[::2]
    print "Plotting %d object hull voxels." % (len(mitos)+len(vc)+len(sj))
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
        print "Plotted %d %s voxel." % (len(obj), type)
        if type=='sj' and color_sj:
            for ix in set(skel.sj_hull_ids):
                ix_bool_arr = skel.sj_hull_ids == ix
                obj_hull = skel.sj_hull_coords[ix_bool_arr]
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


def plot_post_pre(paths):
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
        if 'center_vc_sj' in n_comment and cs_str in n_comment:
            center_coord = arr(node.getCoordinate()) * arr(cs_anno.scaling)/10.
            break
    if center_coord is None:
        print "Could not find center with vc and sj! Using contact_site."
        for node in cs_anno.getNodes():
            if 'center' in node.getComment() and 'cs1_' in node.getComment():
                center_coord = arr(node.getCoordinate()) * arr(cs_anno.scaling)\
                               / 10.
                break
    min_pos, max_pos = [center_coord - offset, center_coord + offset]
    return min_pos, max_pos, center_coord


def get_syn_from_cs_nml(path):
    """
    Parses vc and sj ids of synapse from pairwise contact site nml.
    :param path: str to contact_site.nml
    :return: int tuple vc and sj ID
    """
    cs_anno = au.loadj0126NML(path)[0]
    cs_nb = None
    for node in cs_anno.getNodes():
        if 'center_vc_sj' in node.getComment():
            cs_nb = re.findall('cs(\d+)_', node.getComment())[0]
            break
    if cs_nb is None:
        print "Did not find center_vc_sj, looking for first contact site."
        cs_nb = 'cs1_'
    vc_id = []
    sj_id = []
    for node in cs_anno.getNodes():
        comment = node.getComment()
        if 'cs%s_' % cs_nb in comment:
            if '_vc-' in comment:
                vc_id.append(int(re.findall('_vc-(\d+)', node.getComment())[0]))
            if '_sj-' in comment:
                sj_id.append(int(re.findall('_sj-(\d+)', node.getComment())[0]))
    return vc_id, sj_id


def plot_obj(dh, paths):
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
    [skel, mitos, vc, sj] = load_anno_list([paths[1]])[0]
    skel1_mitos, skel1_vc, skel1_sj = coords_from_anno(skel)

    [skel, mitos, vc, sj] = load_anno_list([paths[0]])[0]
    skel2_mitos, skel2_vc, skel2_sj = coords_from_anno(skel)
    filter_size=[4, 1594, 498]
    coords = arr(dh.vc.rep_coords) * dh.scaling / 10.
    bool_1 = np.all(coords >= min_pos, axis=1) & \
             np.all(coords <= max_pos, axis=1)
    ids = arr(dh.vc.ids)[bool_1][::12]
    print "Found %d vc." %len(ids)
    curr_objects = [dh.vc.object_dict[key] for key in ids if
                    dh.vc.object_dict[key].size >= filter_size[1]]
    pool = Pool(processes=nb_cpus)
    voxel_list = pool.map(helper_get_hull_voxels, curr_objects)
    pool.close()
    pool.join()
    vc = arr([voxel for voxels in voxel_list for voxel in voxels])*skel.scaling
    vc = np.concatenate((vc, skel1_vc, skel2_vc), axis=0)
    vc = get_box_coords(vc, real_min_pos, real_max_pos)
    print "Plotting %d object hull voxels." % len(vc)
    coords = arr(dh.sj.rep_coords) * dh.scaling / 10.
    bool_1 = np.all(coords >= min_pos, axis=1) & \
             np.all(coords <= max_pos, axis=1)
    ids = arr(dh.sj.ids)[bool_1][::12]
    print "Found %d sj." % len(ids)
    curr_objects = [dh.sj.object_dict[key] for key in ids if
                    dh.sj.object_dict[key].size >= filter_size[2]]
    pool = Pool(processes=nb_cpus)
    voxel_list = pool.map(helper_get_hull_voxels, curr_objects)
    pool.close()
    pool.join()
    sj = arr([voxel for voxels in voxel_list for voxel in voxels])*skel.scaling
    sj = np.concatenate((sj, skel1_sj, skel2_sj), axis=0)
    sj = get_box_coords(sj, real_min_pos, real_max_pos)
    print "Plotting %d object hull voxels." % len(sj)
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
    objects = [mitos, vc, sj]
    colors = [(68./(68+85+255.), 85./(68+85+255.), 255./(68+85+255.)),
             (129./(129.+255+29), 255./(129.+255+29), 29/(129.+255+29)),
             (255./(255+44+93), 44./(255+44+93), 93./(255+44+93))]
    for obj, color in zip(objects, colors):
        if len(obj) == 0:
            print "Left out object.", color
            continue
        mlab.points3d(obj[:, 0], obj[:, 1], obj[:, 2], scale_factor=3.5,
                      mode='sphere', color=color, opacity=1.0)


def write_cs_vc_syn_skels(wd, dh=None, renew=False):
    """
    Plot workflow figure and save it!
    :param dest_path: str path to destination directory
    :param dh: DataHandler
    :param renew: Mapped skeletons
    """
    dest_path = wd + '/figures/'
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
    write_img2png(dest_path, 'cs_vc_syn_plot.png', mag=2)


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
    plot_mapped_skel(paths[0], min_pos, max_pos, color=(1., 3/4., 0), op=0.8,
                     hull_visualization=True, save_vx_arr=save_vx_arr)
    plot_mapped_skel(paths[1], min_pos, max_pos, color=(3./4, 3./4, 3./4),
                     op=0.8, hull_visualization=True, save_vx_arr=save_vx_arr)
    plot_mapped_skel(paths[2], min_pos, max_pos, color=(1./4, 1./4, 1./4),
                     op=0.8, hull_visualization=True, save_vx_arr=save_vx_arr)
    if not save_vx_arr:
        write_img2png(dest_path, 'test', mag=1)


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


def write_axoness_cell(skel_path=None):
    if skel_path is None:
        skel_path = get_paths_of_skelID(['182'], '/lustre/pschuber/mapped_soma_tracings/nml_obj/')[0]
    skel, _, _, _, _ = load_mapped_skeleton(skel_path, True, False)
    hull = skel.hull_coords
    normals = skel.hull_normals
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


def neck_head_sj_size_hist(cs_dir, recompute=False):
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
    sj_sizes = [[], [], []]
    if recompute:
        for old_key, val in prop_dict.iteritems():
            key = None
            new_names = convert_to_standard_cs_name(old_key)
            for var in new_names:
                try:
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
            sj_sizes[syn_spiness].append(cs_dict[key]["overlap_area"])
        print "Found shaft/head/neck synapses:\t%d/%d/%d" % (len(spine_types[0]),
                                                             len(spine_types[1]),
                                                             len(spine_types[2]))
        print "Mean sizes:\t\t\t%0.3f+-%0.3f, %0.3f+-%0.3f, %0.3f+-%0.3f" % \
              (np.mean(sj_sizes[0]), np.std(sj_sizes[0]), np.mean(sj_sizes[1]),
               np.std(sj_sizes[1]), np.mean(sj_sizes[2]), np.std(sj_sizes[2]))
        sj_sizes[0] = np.array(sj_sizes[0])
        sj_sizes[1] = np.array(sj_sizes[1])
        sj_sizes[2] = np.array(sj_sizes[2])
        syn_types[0] = np.array(syn_types[0])
        syn_types[1] = np.array(syn_types[1])
        syn_types[2] = np.array(syn_types[2])
        np.save("/lustre/pschuber/figures/syns/spines/shaft_sizes.npy", sj_sizes[0])
        np.save("/lustre/pschuber/figures/syns/spines/head_sizes.npy", sj_sizes[1])
        np.save("/lustre/pschuber/figures/syns/spines/neck_sizes.npy", sj_sizes[2])
        np.save("/lustre/pschuber/figures/syns/spines/shaft_synsign.npy", syn_types[0])
        np.save("/lustre/pschuber/figures/syns/spines/head_synsign.npy", syn_types[1])
        np.save("/lustre/pschuber/figures/syns/spines/neck_synsign.npy", syn_types[2])
    else:
        sj_sizes[0] = np.load("/lustre/pschuber/figures/syns/spines/shaft_sizes.npy")
        sj_sizes[1] = np.load("/lustre/pschuber/figures/syns/spines/head_sizes.npy")
        sj_sizes[2] = np.load("/lustre/pschuber/figures/syns/spines/neck_sizes.npy")
        syn_types[0] = np.load("/lustre/pschuber/figures/syns/spines/shaft_synsign.npy")
        syn_types[1] = np.load("/lustre/pschuber/figures/syns/spines/head_synsign.npy")
        syn_types[2] = np.load("/lustre/pschuber/figures/syns/spines/neck_synsign.npy")
    # shaft_ratio = np.sum(sj_sizes[0][syn_types[0] == 1]) / np.sum(sj_sizes[0])
    # head_ratio = np.sum(sj_sizes[1][syn_types[1] == 1]) / np.sum(sj_sizes[1])
    # neck_ratio = np.sum(sj_sizes[2][syn_types[2] == 1]) / np.sum(sj_sizes[2])
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

    ind = np.arange(3)

    width = .6

    ax.bar(ind-width/2., data, width,
           color=".3",
           edgecolor='k',
           linewidth=2.,
           align='center',
           label='contact site evaluation')

    plt.xticks(ind, names, rotation=0, fontsize=22)
    ax.set_ylim(0., 1)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)


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


def get_obj_density(recompute=False, obj='mito', property='axoness_pred', value=1,
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
    return obj_densities


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
        """Adds an annotation to a currently open mayavi plotting window

        Parameters
        ----------
        anno: annotation object
        node_scaling: float
            scaling factor for node radius
        edge_radius: float
            radius of tubes for each edge
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
