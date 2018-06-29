# SyConn
# Copyright (c) 2016 Philipp J. Schubert
# All rights reserved

# download/import all necessary work packages
import numpy as np
from knossos_utils.skeleton_utils import load_skeleton
from sklearn.neighbors import KDTree
from syconn.proc.meshes import MeshObject, rgb2id_array, id2rgb_array_contiguous
from syconn.proc.graphs import bfs_smoothing
from syconn.handler.basics import majority_element_1d
from syconn.proc.rendering import render_sso_coords, _render_mesh_coords,\
    render_sso_coords_index_views
from syconn.reps.super_segmentation import SuperSegmentationObject
from syconn.reps.views import ViewContainer
from syconn.handler.compression import save_to_h5py
from syconn.mp.shared_mem import start_multiprocess_imap
# from scripts.rendering.inversed_mapping import id2rgb_array
# import matplotlib.pylab as plt
# from imageio import imwrite
from numba import jit
import re
import tqdm
import os
import time
from scipy.misc import imsave
from sklearn.model_selection import train_test_split


def generate_palette(nr_classes, return_rgba=True):
    """
    Creates a RGB(A) palette for N classes.

    Parameters
    ----------
    nr_classes : int
    return_rgba : bool
        If True returned array has shape (N, 4) instead of (N, 3)

    Returns
    -------
    np.array
        Unique color array for N input classes
    """
    classes_ids = np.arange(nr_classes) #reserve additional class id for background
    classes_rgb = id2rgb_array_contiguous(classes_ids)  # convention: do not use 1, 1, 1; will be background value
    if return_rgba:
        classes_rgb = np.concatenate([classes_rgb, np.ones(classes_rgb.shape[:-1])[..., None] * 255], axis=1)
    return classes_rgb.astype(np.uint8)


@jit
def remap_rgb_labelviews(rgb_view, palette):
    """

    Parameters
    ----------
    rgb_view :
    palette :

    Returns
    -------

    """
    label_view_flat = rgb_view.flatten().reshape((-1, 3))
    background_label = len(palette) + 1
    # convention: Use highest ID as background
    remapped_label_views = np.ones((len(label_view_flat), ), dtype=np.uint16) * background_label
    for kk in range(len(label_view_flat)):
        if np.all(label_view_flat[kk] == 255):  # background
            continue
        for i in range(len(palette)):
            if (label_view_flat[kk, 0] == palette[i, 0]) and \
               (label_view_flat[kk, 1] == palette[i, 1]) and \
               (label_view_flat[kk, 2] == palette[i, 2]):
                remapped_label_views[kk] = i
                break
    return remapped_label_views.reshape(rgb_view.shape[:-1])


# create function that converts information in string type to the information in integer type
def str2intconverter(comment, gt_type):
    if gt_type == "axgt":
        if comment == "gt_axon":
            return 1
        elif comment == "gt_dendrite":
            return 0
        elif comment == "gt_soma":
            return 2
        else:
            return -1
    elif gt_type == "spgt":
        if "head" in comment:
            return 1
        elif "neck" in comment:
            return 0
        elif "shaft" in comment:
            return 2
        elif "other" in comment:
            return 3
        else:
            return -1
    else: raise ValueError("Given groundtruth type is not valid.")


def generate_label_views(kzip_path, gt_type="spgt", n_voting=40):
    """

    Parameters
    ----------
    kzip_path : str
    gt_type :  str
    n_voting : int
        Number of collected nodes during BFS for majority vote (label smoothing)

    Returns
    -------

    """
    assert gt_type in ["axgt", "spgt"], "Currently only spine and axon GT is supported"
    n_labels = 3 if gt_type == "axgt" else 4
    palette = generate_palette(n_labels)
    sso_id = int(re.findall("/(\d+).", kzip_path)[0])
    sso = SuperSegmentationObject(sso_id, version=gt_type)
    indices, vertices, normals = sso.mesh

    # # Load mesh
    vertices = vertices.reshape((-1, 3))

    # load skeleton
    skel = load_skeleton(kzip_path)["skeleton"]
    skel_nodes = list(skel.getNodes())

    node_coords = np.array([n.getCoordinate() * sso.scaling for n in skel_nodes])
    node_labels = np.array([str2intconverter(n.getComment(), gt_type) for n in skel_nodes], dtype=np.int)
    node_coords = node_coords[(node_labels != -1)]
    node_labels = node_labels[(node_labels != -1)]

    # create KD tree from skeleton node coordinates
    tree = KDTree(node_coords)
    # transfer labels from skeleton to mesh
    dist, ind = tree.query(vertices, k=1)

    vertex_labels = node_labels[ind]  # retrieving labels of vertices

    vertex_labels = bfs_smoothing(vertices, vertex_labels, n_voting=n_voting)

    color_array = palette[vertex_labels].astype(np.float32)/255

    # Initializing mesh object with ground truth coloring
    mo = MeshObject("neuron", indices, vertices, color=color_array)

    # use downsampled locations for view locations, only if they are close to a
    # labeled skeleton node
    locs = np.concatenate(sso.sample_locations())
    dist, ind = tree.query(locs)
    locs = locs[dist[:, 0] < 2000][::3][:100]

    # # DEBUG PART START
    dest_folder = os.path.expanduser("~") + \
                  "/spiness_skels/{}/view_imgs_{}/".format(sso_id, n_voting)
    if not os.path.isdir(dest_folder):
        os.makedirs(dest_folder)
    loc_text = ''
    for i, c in enumerate(locs):
        loc_text += str(i+1) + "\t" + str((c / np.array([10, 10, 20])).astype(np.int)) +'\n' #rescalling to the voxel grid
    with open("{}/viewcoords.txt".format(dest_folder), "w") as f:
        f.write(loc_text)
    # # DEBUG PART END

    label_views, rot_mat = _render_mesh_coords(locs, mo, depth_map=False,
                                               return_rot_matrices=True,
                                               smooth_shade=False)
    index_views = render_sso_coords_index_views(sso, locs, rot_matrices=rot_mat)
    raw_views = render_sso_coords(sso, locs)
    raw_views_wire = render_sso_coords(sso, locs, wire_frame=True, ws=(2048, 1024))   # TODO: HACK
    return raw_views_wire, raw_views, remap_rgb_labelviews(label_views, palette)[:, None], rgb2id_array(index_views)[:, None]


def GT_generation(kzip_paths, dest_dir=None, gt_type="spgt", n_voting=40):
    """
    Generates a .npy GT file from all kzip paths.

    Parameters
    ----------
    kzip_paths : list of str
    gt_type : str
    n_voting : int
        Number of collected nodes during BFS for majority vote (label smoothing)
    Returns
    -------

    """
    if dest_dir is None:
        dest_dir = os.path.expanduser("~") + "/spine_gt_multiview/"
    params = [(p, gt_type, n_voting) for p in kzip_paths]
    res = start_multiprocess_imap(gt_generation_helper, params, nb_cpus=5,
                                  debug=False)
    #
    # # Create Dataset splits for training, validation and test
    # all_raw_views = []
    # all_label_views = []
    # all_index_views = []
    # for ii in range(len(kzip_paths)):
    #     all_raw_views.append(res[ii][0])
    #     all_label_views.append(res[ii][1])
    #     all_index_views.append(res[ii][2])
    # all_raw_views = np.concatenate(all_raw_views)
    # all_label_views = np.concatenate(all_label_views)
    # all_index_views = np.concatenate(all_index_views)
    # print("Shuffling views.")
    # ixs = np.arange(len(all_raw_views))
    # np.random.shuffle(ixs)
    # all_raw_views = all_raw_views[ixs]
    # all_label_views = all_label_views[ixs]
    # all_index_views = all_index_views[ixs]
    # print("Swapping axes.")
    # all_raw_views = all_raw_views.swapaxes(2, 1)
    # all_label_views = all_label_views.swapaxes(2, 1)
    # all_index_views = all_index_views.swapaxes(2, 1)
    # print("Reshaping arrays.")
    # all_raw_views = all_raw_views.reshape((-1, 4, 128, 256))
    # all_label_views = all_label_views.reshape((-1, 1, 128, 256))
    # all_index_views = all_index_views.reshape((-1, 1, 128, 256))
    # all_raw_views = np.concatenate([all_raw_views, all_index_views], axis=1)
    # raw_train, raw_other, label_train, label_other = \
    #     train_test_split(all_raw_views, all_label_views, train_size=0.8, shuffle=True)
    # raw_valid, raw_test, label_valid, label_test = \
    #     train_test_split(raw_other, label_other, train_size=0.5, shuffle=True)
    # print("Writing h5 files.")
    # save_to_h5py([raw_train], dest_dir + "/raw_train.h5",
    #              ["raw"])
    # save_to_h5py([raw_valid], dest_dir + "/raw_valid.h5",
    #              ["raw"])
    # save_to_h5py([raw_test], dest_dir + "/raw_test.h5",
    #              ["raw"])
    # save_to_h5py([label_train], dest_dir + "/label_train.h5",
    #              ["label"])
    # save_to_h5py([label_valid], dest_dir + "/label_valid.h5",
    #              ["label"])
    # save_to_h5py([label_test], dest_dir + "/label_test.h5",
    #              ["label"])


def gt_generation_helper(args):
    kzip_path, gt_type, n_voting = args
    sso_id = int(re.findall("/(\d+).", kzip_path)[0])
    raw_views_wire, raw_views, label_views, index_views = generate_label_views(kzip_path, gt_type, n_voting)
    #
    # DEBUG PART START
    h5py_path = os.path.expanduser("~") + "/spiness_skels/{}/view_imgs_{}/".format(sso_id, n_voting)
    if not os.path.isdir(h5py_path):
        os.makedirs(h5py_path)
    # save_to_h5py([raw_views[:, 0, 0], label_views[:, 0, 0],
    #               index_views[:, 0, 0]], h5py_path + "/views.h5",
    #              ["raw", "label", "index"])
    vc = ViewContainer("", views=raw_views)
    vc_wire = ViewContainer("", views=raw_views_wire)
    # randomize color map of index views
    colored_indices = np.zeros(list(index_views.shape) + [3], dtype=np.uint8)
    for ix in np.unique(index_views):
        rand_col = np.random.randint(0, 256, 3)
        colored_indices[index_views == ix] = rand_col
    for ii in range(len(raw_views)):
        vc_wire.write_single_plot("{}/{}_raw_wire.tif".format(h5py_path, ii), ii)
        vc.write_single_plot("{}/{}_raw.tif".format(h5py_path, ii), ii)
        imsave(h5py_path + "{}_label.tif".format(ii), label_views[:, 0, 0][ii])
        imsave(h5py_path + "{}_index.tif".format(ii), colored_indices[:, 0, 0][ii])
    # DEBUG PART END

    return raw_views, label_views, index_views


if __name__ == "__main__":
    label_file_folder = "/wholebrain/scratch/areaxfs3/ssv_spgt/" \
                        "spiness_skels_annotated/"
    file_names = ["/23044610.035.k.zip", "/4741011.073.k.zip",
                  "/18279774.078.k.zip", "/26331138.043.k.zip",
                  "/27965455.032.k.zip"]
    file_paths = [label_file_folder + "/" + fname for fname in file_names][::-1]
    GT_generation(file_paths)
