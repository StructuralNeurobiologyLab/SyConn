# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max-Planck-Institute of Neurobiology, Munich, Germany
# Authors: Philipp Schubert, Joergen Kornfeld

import numpy as np
from knossos_utils.skeleton_utils import load_skeleton
from sklearn.neighbors import KDTree
from syconn.proc.meshes import MeshObject, write_mesh2kzip, merge_meshes
from syconn.handler.multiviews import generate_rendering_locs
from syconn.proc.graphs import bfs_smoothing
from syconn.proc.rendering import render_sso_coords, _render_mesh_coords,\
    render_sso_coords_index_views
from syconn.proc.ssd_assembly import init_sso_from_kzip
from syconn.reps.super_segmentation import SuperSegmentationObject, SuperSegmentationDataset
from syconn.reps.views import ViewContainer
from syconn import global_params
from syconn.handler.compression import save_to_h5py
from syconn.handler.multiviews import generate_palette, remap_rgb_labelviews, str2intconverter
from syconn.mp.mp_utils import start_multiprocess_imap
import re
from multiprocessing import cpu_count
import os, glob
import zipfile
# from scipy.misc import imsave
from imageio import imwrite
from sklearn.model_selection import train_test_split
import glob
global initial_run


def check_kzip_completeness(data_path: str, fnames: list):
    """
    Check the completeness of the kzip file, each kzip file must contain:
    meta.pkl, mi.pkl, sj.pkl, sv.pkl, sv.pkl, vc.pkl and annotation.xml
    """
    filtered_fnames = []
    for file_name in fnames:
        kzip_path = data_path + '/' + file_name
        zip = zipfile.ZipFile(kzip_path)
        files = zip.namelist()

        if zip is not None and set(files) == {'meta.pkl', 'annotation.xml', 'sj.ply', 'vc.ply', 'mi.ply', 'sv.ply'}:
            filtered_fnames.append(file_name)

    return filtered_fnames

def get_all_fname(data_path):
    """
    Get file names in the given path
    """
    fnames = []
    cell_combinations = set()
    os.chdir(data_path)

    # count = 0
    for file in glob.glob("*.k.zip"):
        # filter out duplicate merged_cells
        cell_ids = re.findall(r"(\d+)", data_path + file)[1:]
        if tuple(cell_ids) in cell_combinations or tuple(cell_ids[::-1]) in cell_combinations:
            continue
        cell_combinations.add(tuple(cell_ids))

        fnames.append('/' + file)
        # count += 1
    return fnames


def generate_label_views(kzip_path, ssd_version, gt_type, n_voting=40, nb_views=2,
                         ws=(256, 128), comp_window=8e3,
                         out_path=None, verbose=False):
    """

    Parameters
    ----------
    kzip_path : str
    gt_type :  str
    ssd_version : str
    n_voting : int
        Number of collected nodes during BFS for majority vote (label smoothing)
    nb_views : int
    ws: Tuple[int]
    comp_window : float
    initial_run : bool
        if True, will copy SSV from default SSD to SSD with version=gt_type
    out_path : str
        If given, export mesh colored accoring to GT labels
    verbose : bool
        Print additional information

    Returns
    -------
    Tuple[np.array]
        raw, label and index views
    """
    assert gt_type in ["merger_gt"], "Currently only merger GT is supported"
    n_labels = 3
    palette = generate_palette(n_labels)

    # @debug
    # sso_id = int(re.findall(r"/(\d+).", kzip_path)[0])
    # sso_id1 = int(re.findall(r"(\d+)", kzip_path)[1])
    # sso_id2 = int(re.findall(r"(\d+)", kzip_path)[2])
    merged_sso_id = re.findall(r"(\d+)", kzip_path)[1:]

    sso = init_sso_from_kzip(kzip_path, sso_id=1)

    # if initial_run:  # use default SSD version
    #     # orig_sso = SuperSegmentationObject(sso_id)
    #     orig_sso1 = SuperSegmentationObject(sso_id1)
    #     orig_sso2 = SuperSegmentationObject(sso_id2)
    #     # orig_sso.copy2dir(dest_dir=sso.ssv_dir, safe=False)
    #     orig_sso1.copy2dir(dest_dir=sso.ssv_dir, safe=False)
    #     orig_sso2.copy2dir(dest_dir=sso.ssv_dir, safe=False)
    # if not sso.attr_dict_exists:
    #     msg = 'Attribute dict of original SSV was not copied successfully ' \
    #           'to target SSD.'
    #     raise ValueError(msg)
    sso.load_attr_dict()
    indices, vertices, normals = sso.mesh

    # # Load mesh
    vertices = vertices.reshape((-1, 3))

    # load skeleton
    skel = load_skeleton(kzip_path)
    if len(skel) == 1:
        skel = list(skel.values())[0]
    else:
        skel = skel["skeleton"]
    skel_nodes = list(skel.getNodes())

    node_coords = np.array([n.getCoordinate() * sso.scaling for n in skel_nodes])
    # node_labels = np.array([str2intconverter(n.getComment(), gt_type) for n in skel_nodes], dtype=np.int)
    # node_labels = np.array([int(n.data['merger_gt']) for n in skel_nodes], dtype=np.int)
    node_labels = np.array([int(n.data['merger_gt'][0]) for n in skel_nodes], dtype=np.int)
    node_coords = node_coords[(node_labels != -1)]
    node_labels = node_labels[(node_labels != -1)]

    # create KD tree from skeleton node coordinates
    tree = KDTree(node_coords)
    # transfer labels from skeleton to mesh
    dist, ind = tree.query(vertices, k=1)
    vertex_labels = node_labels[ind]  # retrieving labels of vertices
    if n_voting > 0:
        vertex_labels = bfs_smoothing(vertices, vertex_labels, n_voting=n_voting)
    color_array = palette[vertex_labels].astype(np.float32) / 255.

    if out_path is not None:
        if gt_type == 'spgt':  # RBG-A value
            colors = [[0.6, 0.6, 0.6, 1], [0.9, 0.2, 0.2, 1], [0.1, 0.1, 0.1, 1], [0.05, 0.6, 0.6, 1], [0.9, 0.9, 0.9, 1]]
        elif gt_type == 'merger_gt':
                     # ignore (green)     , true-merger(grey) , false-merger(red)
            colors = [[0.5, 0.5, 0.5, 0.4], [0.5, 0.5, 0.5, 0.4], [0.96, 0.14, 0.347, 1]]
        else:# dendrite, axon, soma, bouton, terminal, background
            colors = [[0.6, 0.6, 0.6, 1], [0.9, 0.2, 0.2, 1], [0.1, 0.1, 0.1, 1],
                      [0.05, 0.6, 0.6, 1], [0.6, 0.05, 0.05, 1], [0.9, 0.9, 0.9, 1]]
        colors = (np.array(colors) * 255).astype(np.uint8)
        color_array_mesh = colors[vertex_labels][:, 0]  # TODO: check why only first element, maybe colors introduces an additional axis
        write_mesh2kzip("{}/sso_{}_gtlabels.k.zip".format(out_path, '_'.join(merged_sso_id)),
                        sso.mesh[0], sso.mesh[1], sso.mesh[2], color_array_mesh,
                        ply_fname="gtlabels.ply")

    # Initializing mesh object with ground truth coloring
    mo = MeshObject("neuron", indices, vertices, color=color_array)

    # use downsampled locations for view locations, only if they are close to a
    # labeled skeleton node
    locs = generate_rendering_locs(vertices, comp_window / 6)  # 6 rendering locations per comp.
    # window
    dist, ind = tree.query(locs)
    locs = locs[dist[:, 0] < 2000]#[::3][:5]  # TODO add as parameter

    # # # To get view locations
    # dest_folder = os.path.expanduser("~") + \
    #               "/spiness_skels/{}/view_imgs_{}/".format(sso_id, n_voting)
    # if not os.path.isdir(dest_folder):
    #     os.makedirs(dest_folder)
    # loc_text = ''
    # for i, c in enumerate(locs):
    #     loc_text += str(i) + "\t" + str((c / np.array([10, 10, 20])).astype(np.int)) +'\n' #rescalling to the voxel grid
    # with open("{}/viewcoords.txt".format(dest_folder), "w") as f:
    #     f.write(loc_text)
    # # # DEBUG PART END
    label_views, rot_mat = _render_mesh_coords(locs, mo, depth_map=False,
                                               return_rot_matrices=True, ws=ws,
                                               smooth_shade=False, nb_views=nb_views,
                                               comp_window=comp_window, verbose=verbose)
    label_views = remap_rgb_labelviews(label_views[..., :3], palette)[:, None]
    # TODO: the 3 neglects the alpha channel, i.e. remapping labels bigger than 256**3 becomes
    #  invalid
    index_views = render_sso_coords_index_views(sso, locs, rot_mat=rot_mat, verbose=verbose,
                                                nb_views=nb_views, ws=ws, comp_window=comp_window)
    raw_views = render_sso_coords(sso, locs, nb_views=nb_views, ws=ws,
                                  comp_window=comp_window, verbose=verbose,
                                  rot_mat=rot_mat)
    return raw_views, label_views, index_views


def GT_generation_from_kzip(kzip_paths, ssd_version, gt_type, nb_views, dest_dir=None,
                  n_voting=0, ws=(256, 128), comp_window=8e3, h5_suffix=""):
    """
    Generates a .npy GT file from all kzip paths.

    Parameters
    ----------
    kzip_paths : List[str]
    gt_type : str
    n_voting : int
        Number of collected nodes during BFS for majority vote (label smoothing)
    Returns
    -------

    """
    # sso_ids = [re.findall(r"(\d+)", kzip_path)[1:] for kzip_path in kzip_paths]
    # ssd = SuperSegmentationDataset()
    # load mesh from kzip

    # if not np.all([ssv.lookup_in_attribute_dict("size") is not None for ssv in
    #                ssd.get_super_segmentation_object(sso_ids)]):
    #     print("Not all SSV IDs are part of " \
    #           "the current SSD. IDs: {}".format([sso_id for sso_id in sso_ids if sso_id not in
    #                                              ssd.ssv_ids]))
    #     kzip_paths = np.array(kzip_paths)[np.array([ssv.lookup_in_attribute_dict("size") is not None for ssv in
    #                                                 ssd.get_super_segmentation_object(sso_ids)])]
    #     print("Ignoring missing IDs. Using {} k.zip files for GT "
    #           "generation,".format(len(kzip_paths)))

    if dest_dir is None:
        dest_dir = os.path.expanduser("~/{}_semseg/".format(gt_type))
    if not os.path.isdir(dest_dir):
        os.makedirs(dest_dir)
    dest_p_cache = "{}/cache_{}votes/".format(dest_dir, n_voting)
    # @debug
    # dest_p_cache: /u/yliu/merger_gt_semseg//cache_40votes/
    params = [(p, ssd_version, gt_type, n_voting, nb_views, ws, comp_window, dest_p_cache) for p in kzip_paths]
    if not os.path.isdir(dest_p_cache):
        os.makedirs(dest_p_cache)
    start_multiprocess_imap(gt_generation_helper, params, nb_cpus=cpu_count(),
                            debug=False)

    # TODO: in case GT is too big to hold all views in memory
    # if gt_type == 'axgt':
    #     return
    # Create Dataset splits for training, validation and test
    all_raw_views = []
    all_label_views = []
    # all_index_views = []  # Removed index views
    print("Writing views.")
    for ii in range(len(kzip_paths)):
        # sso_id = int(re.findall(r"/(\d+).", kzip_paths[ii])[0])
        sso_id = re.findall(r"(\d+)", kzip_paths[ii])[1:]
        
        # dest_p = "{}/{}/".format(dest_p_cache, sso_id)
        dest_p = "{}/{}/".format(dest_p_cache, '_'.join(sso_id))
        raw_v = np.load(dest_p + "raw.npy")
        label_v = np.load(dest_p + "label.npy")
        # index_v = np.load(dest_p + "index.npy")  # Removed index views
        all_raw_views.append(raw_v)
        all_label_views.append(label_v)
        # @debug
        # print("all_raw_views: {}".format(type(all_raw_views)))
        # print("all_label_views: {}".format(type(all_label_views)))
        # all_index_views.append(index_v)  # Removed index views

    all_raw_views = np.concatenate(all_raw_views)
    all_label_views = np.concatenate(all_label_views)
    # all_index_views = np.concatenate(all_index_views)  # Removed index views
    print("{} view locations collected. Shuffling views.".format(len(all_label_views)))
    np.random.seed(0)
    ixs = np.arange(len(all_raw_views))
    np.random.shuffle(ixs)
    all_raw_views = all_raw_views[ixs]
    all_label_views = all_label_views[ixs]
    # all_index_views = all_index_views[ixs]  # Removed index views
    print("Swapping axes.")
    all_raw_views = all_raw_views.swapaxes(2, 1)
    all_label_views = all_label_views.swapaxes(2, 1)
    # all_index_views = all_index_views.swapaxes(2, 1)  # Removed index views
    print("Reshaping arrays.")
    all_raw_views = all_raw_views.reshape((-1, 4, ws[1], ws[0]))
    all_label_views = all_label_views.reshape((-1, 1, ws[1], ws[0]))
    # # all_index_views = all_index_views.reshape((-1, 1, 128, 256))  # Removed index views
    # # all_raw_views = np.concatenate([all_raw_views, all_index_views], axis=1)  # Removed index views
    raw_train, raw_valid, label_train, label_valid = train_test_split(all_raw_views,
                                                                      all_label_views, train_size=0.9,
                                                                      shuffle=False)
    raw_train, raw_valid = raw_train.astype(np.uint8), raw_valid.astype(np.uint8)
    label_train, label_valid = label_train.astype(np.uint8), label_valid.astype(np.uint8)
    # # raw_valid, raw_test, label_valid, label_test = train_test_split(raw_other, label_other, train_size=0.5, shuffle=False)  # Removed index views
    print("Writing h5 files.")
    os.makedirs(dest_dir, exist_ok=True)
    # chunk output data
    for ii in range(5):
        save_to_h5py([raw_train[ii::5]], dest_dir + "/raw_train_{}_{}.h5".format(ii, h5_suffix),
                     ["raw"])
        save_to_h5py([raw_valid[ii::5]], dest_dir + "/raw_valid_{}_{}.h5".format(ii, h5_suffix),
                     ["raw"])
        # save_to_h5py([raw_test], dest_dir + "/raw_test.h5",
        # ["raw"])  # Removed index views
        save_to_h5py([label_train[ii::5]], dest_dir + "/label_train_{}_{}.h5".format(ii, h5_suffix),
                     ["label"])
        save_to_h5py([label_valid[ii::5]], dest_dir + "/label_valid_{}_{}.h5".format(ii, h5_suffix),
                     ["label"])
    # save_to_h5py([label_test], dest_dir + "/label_test.h5",
    # ["label"])  # Removed index views

def GT_generation(kzip_paths, ssd_version, gt_type, nb_views, dest_dir=None,
                  n_voting=40, ws=(256, 128), comp_window=8e3, h5_suffix=""):
    """
    Generates a .npy GT file from all kzip paths.

    Parameters
    ----------
    kzip_paths : List[str]
    gt_type : str
    n_voting : int
        Number of collected nodes during BFS for majority vote (label smoothing)
    Returns
    -------

    """
    sso_ids = [int(re.findall(r"/(\d+).", kzip_path)[0]) for kzip_path in kzip_paths]
    ssd = SuperSegmentationDataset()
    if not np.all([ssv.lookup_in_attribute_dict("size") is not None for ssv in
                   ssd.get_super_segmentation_object(sso_ids)]):
        print("Not all SSV IDs are part of " \
            "the current SSD. IDs: {}".format([sso_id for sso_id in sso_ids if sso_id not in
                                               ssd.ssv_ids]))
        kzip_paths = np.array(kzip_paths)[np.array([ssv.lookup_in_attribute_dict("size") is not None for ssv in
                   ssd.get_super_segmentation_object(sso_ids)])]
        print("Ignoring missing IDs. Using {} k.zip files for GT "
              "generation,".format(len(kzip_paths)))
    if dest_dir is None:
        dest_dir = os.path.expanduser("~/{}_semseg/".format(gt_type))
    if not os.path.isdir(dest_dir):
        os.makedirs(dest_dir)
    dest_p_cache = "{}/cache_{}votes/".format(dest_dir, n_voting)
    params = [(p, ssd_version, gt_type, n_voting, nb_views, ws, comp_window, dest_p_cache) for p in kzip_paths]
    if not os.path.isdir(dest_p_cache):
        os.makedirs(dest_p_cache)
    start_multiprocess_imap(gt_generation_helper, params, nb_cpus=cpu_count(),
                            debug=False)
    # TODO: in case GT is too big to hold all views in memory
    # if gt_type == 'axgt':
    #     return
    # Create Dataset splits for training, validation and test
    all_raw_views = []
    all_label_views = []
    # all_index_views = []  # Removed index views
    print("Writing views.")
    for ii in range(len(kzip_paths)):
        sso_id = int(re.findall(r"/(\d+).", kzip_paths[ii])[0])
        dest_p = "{}/{}/".format(dest_p_cache, sso_id)
        raw_v = np.load(dest_p + "raw.npy")
        label_v = np.load(dest_p + "label.npy")
        # index_v = np.load(dest_p + "index.npy")  # Removed index views
        all_raw_views.append(raw_v)
        all_label_views.append(label_v)
        # all_index_views.append(index_v)  # Removed index views
    all_raw_views = np.concatenate(all_raw_views)
    all_label_views = np.concatenate(all_label_views)
    # all_index_views = np.concatenate(all_index_views)  # Removed index views
    print("{} view locations collected. Shuffling views.".format(len(all_label_views)))
    np.random.seed(0)
    ixs = np.arange(len(all_raw_views))
    np.random.shuffle(ixs)
    all_raw_views = all_raw_views[ixs]
    all_label_views = all_label_views[ixs]
    # all_index_views = all_index_views[ixs]  # Removed index views
    print("Swapping axes.")
    all_raw_views = all_raw_views.swapaxes(2, 1)
    all_label_views = all_label_views.swapaxes(2, 1)
    # all_index_views = all_index_views.swapaxes(2, 1)  # Removed index views
    print("Reshaping arrays.")
    all_raw_views = all_raw_views.reshape((-1, 4, ws[1], ws[0]))
    all_label_views = all_label_views.reshape((-1, 1, ws[1], ws[0]))
    # # all_index_views = all_index_views.reshape((-1, 1, 128, 256))  # Removed index views
    # # all_raw_views = np.concatenate([all_raw_views, all_index_views], axis=1)  # Removed index views
    raw_train, raw_valid, label_train, label_valid = train_test_split(all_raw_views,
                                                                      all_label_views, train_size=0.9,
                                                                      shuffle=False)
    # # raw_valid, raw_test, label_valid, label_test = train_test_split(raw_other, label_other, train_size=0.5, shuffle=False)  # Removed index views
    print("Writing h5 files.")
    os.makedirs(dest_dir, exist_ok=True)
    # chunk output data
    for ii in range(5):
        save_to_h5py([raw_train[ii::5]], dest_dir + "/raw_train_{}_{}.h5".format(ii, h5_suffix),
                     ["raw"])
        save_to_h5py([raw_valid[ii::5]], dest_dir + "/raw_valid_{}_{}.h5".format(ii, h5_suffix),
                     ["raw"])
        # save_to_h5py([raw_test], dest_dir + "/raw_test.h5",
        # ["raw"])  # Removed index views
        save_to_h5py([label_train[ii::5]], dest_dir + "/label_train_{}_{}.h5".format(ii, h5_suffix),
                     ["label"])
        save_to_h5py([label_valid[ii::5]], dest_dir + "/label_valid_{}_{}.h5".format(ii, h5_suffix),
                     ["label"])
        print("dest_dir_{}: {}".format(ii, dest_dir))
    # save_to_h5py([label_test], dest_dir + "/label_test.h5",
    # ["label"])  # Removed index views


def gt_generation_helper(args):
    kzip_path, ssd_version, gt_type, n_voting, nb_views, ws, comp_window, dest_dir = args
    raw_views, label_views, index_views = generate_label_views(kzip_path, ssd_version, gt_type,
                                                               n_voting, nb_views, ws, comp_window,
                                                               out_path=dest_dir) # out_path set, colored meshes will be written out

    # sso_id = int(re.findall(r"/(\d+).", kzip_path)[0])
    sso_id = re.findall(r"(\d+)", kzip_path)[1:]
    dest_p = "{}/{}/".format(dest_dir, '_'.join(sso_id))
    if not os.path.isdir(dest_p):
        os.makedirs(dest_p)
    np.save(dest_p + "raw.npy", raw_views)
    np.save(dest_p + "label.npy", label_views)
    save_to_h5py([raw_views, label_views.astype(np.uint8)], dest_dir + "/data_{}.h5".format('_'.join(sso_id)),
                 ["raw", "label"])
    # np.save(dest_p + "index.npy", index_views)

    # DEBUG PART START, write out images for manual inspection in Fiji
    # from syconn.reps.super_segmentation_object import merge_axis02
    # # raw_views_wire = merge_axis02(raw_views_wire)[:, :, None]
    # raw_views = merge_axis02(raw_views)[:, :, None][:10]
    # label_views = merge_axis02(label_views)[:, :, None][:10]
    # index_views = merge_axis02(index_views)[:, :, None][:10]
    # sso_id = int(re.findall(r"/(\d+).", kzip_path)[0])
    # h5py_path = os.path.expanduser("~") + "/spiness_skels_biggercontext/{}/view_imgs_{}/".format(sso_id, n_voting)
    # if not os.path.isdir(h5py_path):
    #     os.makedirs(h5py_path)
    # # save_to_h5py([raw_views[:, 0, 0], label_views[:, 0, 0],
    # #               index_views[:, 0, 0]], h5py_path + "/views.h5",
    # #              ["raw", "label", "index"])
    # vc = ViewContainer("", views=raw_views)
    # # vc_wire = ViewContainer("", views=raw_views_wire)
    # # randomize color map of index views
    # colored_indices = np.zeros(list(index_views.shape) + [3], dtype=np.uint8)
    # for ix in np.unique(index_views):
    #     rand_col = np.random.randint(0, 256, 3)
    #     colored_indices[index_views == ix] = rand_col
    # for ii in range(len(raw_views)):
    #     # vc_wire.write_single_plot("{}/{}_raw_wire.png".format(h5py_path, ii), ii)
    #     vc.write_single_plot("{}/{}_raw.png".format(h5py_path, ii), ii)
    #     imsave(h5py_path + "{}_label.png".format(ii), label_views[:, 0, 0][ii])
    #     imsave(h5py_path + "{}_index.png".format(ii), colored_indices[:, 0, 0][ii])
    # DEBUG PART END


if __name__ == "__main__":

    #############################
    # Unit test
    ############################
    if 1:
        comp_window = 10240 * 3
        ws = (256, 128)
        dest_gt_dir = "/wholebrain/scratch/yliu/false_merger/{}".format(ws[0]) #output directory
        # dest_gt_dir = "/home/kloping/wholebrain/scratch/yliu/false_merger/{}".format(ws[0])  # local test: output directory
        os.makedirs(dest_gt_dir, exist_ok=True)
        # global_params.wd = "/wholebrain/scratch/areaxfs3/"
        # assert global_params.wd == "/wholebrain/scratch/areaxfs3/"
        initial_run = False
        n_views = 3
        label_file_folder = "/wholebrain/u/yliu/develop/SyConn/scripts/false_merger/generated_cells"
        # label_file_folder = "/home/kloping/mpi_develop/develop/SyConn/scripts/false_merger" # local test
        # file_names = ["/syn78739_cells33286658_30666759.k.zip",
        #               "/syn669316_cells31272448_26034194.k.zip",
        #               "syn373853_cells8636931_3062786.k.zip"]  # Test
        all_file_names = get_all_fname(label_file_folder)
        file_names = check_kzip_completeness(label_file_folder, all_file_names)
        file_paths = [label_file_folder + "/" + fname for fname in file_names][::-1]
        GT_generation_from_kzip(file_paths, 'merger_gt', 'merger_gt', n_views, ws=ws, comp_window=comp_window)

    # spiness
    if 0:
        comp_window = 10240  # 40 nm pixel size
        ws = (256, 128)
        dest_gt_dir = "/wholebrain/scratch/areaxfs3/ssv_semsegspiness/gt_h5_files_20nm_{" \
                      "}/".format(ws[0])
        os.makedirs(dest_gt_dir, exist_ok=True)
        global_params.wd = "/wholebrain/scratch/areaxfs3/"
        assert global_params.wd == "/wholebrain/scratch/areaxfs3/"
        initial_run = False
        n_views = 3
        label_file_folder = "/wholebrain/scratch/areaxfs3/ssv_spgt/" \
                            "spiness_skels_annotated/"
        file_names = ["/27965455.039.k.zip",
                      "/23044610.037.k.zip", "/4741011.074.k.zip",
                      "/18279774.089.k.zip", "/26331138.046.k.zip"]
        file_paths = [label_file_folder + "/" + fname for fname in file_names][::-1]
        GT_generation(file_paths, 'spgt', 'spgt', n_views, ws=ws, comp_window=comp_window)

    # axoness
    if 0:
        initial_run = True
        ws = (1024, 512)
        comp_window = 40.96e3 * 1.0  # ~40nm pixel size
        n_views = 6  # increase views per location due to higher pixel size, also increases GT
        # diversity


        # # Process original data
        # dest_gt_dir = "/wholebrain/scratch/areaxfs3/ssv_semsegaxoness/gt_h5_files_80nm_{" \
        #               "}/".format(ws[0])
        # global_params.wd = "/wholebrain/scratch/areaxfs3/"
        # assert global_params.wd == "/wholebrain/scratch/areaxfs3/"
        # label_file_folder = "/wholebrain/scratch/areaxfs3/ssv_semsegaxoness" \
        #                     "/gt_axoness_semseg_skeletons/batch1/"
        # file_paths = glob.glob(label_file_folder + '*.k.zip', recursive=False)
        # GT_generation(file_paths, 'semsegaxoness', 'axgt', n_views, dest_dir=dest_gt_dir,
        #               ws=ws, comp_window=40.96e3*2, n_voting=0)  # disable BFS smoothing on ve
        #
        # # Process new batches
        # global_params.wd = "/wholebrain/songbird/j0126/areaxfs_v6/"
        # assert global_params.wd == "/wholebrain/songbird/j0126/areaxfs_v6/"
        #
        # label_file_folder = "/wholebrain/scratch/areaxfs3/ssv_semsegaxoness" \
        #                     "/gt_axoness_semseg_skeletons//batch3/"  # BATCH2_Feb2019/Annotations/"
        # file_paths = glob.glob(label_file_folder + '*.k.zip', recursive=False)
        # GT_generation(file_paths, 'semsegaxoness', 'axgt', n_views, dest_dir=dest_gt_dir,
        #               ws=ws, comp_window=40.96e3*2, n_voting=0)
        #
        # label_file_folder = "/wholebrain/scratch/areaxfs3/ssv_semsegaxoness" \
        #                     "/gt_axoness_semseg_skeletons/BATCH2_Feb2019/Annotations/"
        # file_paths = glob.glob(label_file_folder + '*.k.zip', recursive=False)
        # GT_generation(file_paths, 'semsegaxoness', 'axgt', n_views, dest_dir=dest_gt_dir,
        #               ws=ws, comp_window=40.96e3*2, n_voting=0)  # disable BFS smoothing on vertices (probalby not needed on cell compartment level)

        # # # bouton GT
        # # BATCH 1
        # dest_gt_dir = "/wholebrain/scratch/areaxfs3/ssv_semsegaxoness/gt_h5_files_40nm_{" \
        #               "}_with_BOUTONS_batch1_v2/".format(ws[0])
        # global_params.wd = "/wholebrain/scratch/areaxfs3/"
        # assert global_params.wd == "/wholebrain/scratch/areaxfs3/"
        # label_file_folder = "/wholebrain/scratch/areaxfs3/ssv_semsegaxoness" \
        #                     "/gt_axoness_semseg_skeletons/NEW_including_boutons/batch1_results/"
        # file_paths = glob.glob(label_file_folder + '*.k.zip', recursive=False)
        # GT_generation(file_paths, 'semsegaxoness', 'axgt', n_views, dest_dir=dest_gt_dir,
        #               ws=ws, comp_window=comp_window, n_voting=0, h5_suffix="batch1")  # disable
        # # BFS smoothing on
        # # vertices (probalby not needed on cell compartment level)
        # # Batch 2
        dest_gt_dir = "/wholebrain/scratch/areaxfs3/ssv_semsegaxoness/gt_h5_files_40nm_{" \
                      "}_with_BOUTONS_batch2_v5/".format(ws[0])
        global_params.wd = "/wholebrain/songbird/j0126/areaxfs_v6/"
        assert global_params.wd == "/wholebrain/songbird/j0126/areaxfs_v6/"
        label_file_folder = "/wholebrain/scratch/areaxfs3/ssv_semsegaxoness" \
                            "/gt_axoness_semseg_skeletons/NEW_including_boutons/batch2_results_v2/"
        file_paths = glob.glob(label_file_folder + '*.k.zip', recursive=False)
        GT_generation(file_paths, 'semsegaxoness', 'axgt', n_views, dest_dir=dest_gt_dir,
                      ws=ws, comp_window=comp_window, n_voting=0, h5_suffix="batch2")  # disable
        # BFS smoothing on
