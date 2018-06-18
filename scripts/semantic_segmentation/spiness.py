# SyConn
# Copyright (c) 2016 Philipp J. Schubert
# All rights reserved

# download/import all necessary work packages
import numpy as np
from knossos_utils.skeleton_utils import load_skeleton
from sklearn.neighbors import KDTree
from syconn.proc.meshes import MeshObject, rgb2id_array, id2rgb_array_contiguous
from syconn.handler.basics import majority_element_1d
from syconn.proc.rendering import render_sso_coords, _render_mesh_coords,\
    render_sso_coords_index_views
from syconn.reps.super_segmentation import SuperSegmentationObject
from syconn.handler.compression import save_to_h5py
from syconn.mp.shared_mem import start_multiprocess_imap
# from scripts.rendering.inversed_mapping import id2rgb_array
# import matplotlib.pylab as plt
# from imageio import imwrite
import re
import tqdm
import os
import time
from sklearn.model_selection import train_test_split

# define palette, but also take care of inverse mapping 'remap_rgb_labelviews'
# due to speed issues labels have to be given axis wise:
#  e.g. (1, 0, 0), (2, 0, 0), ..., (255, 0, 0) and (0, 1, 0), ... (0, 255, 0)
# this defines rgb values for labels 0, 1 and 2


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
    classes_ids = np.arange(nr_classes+1) #reserve additional class id for background
    classes_rgb = id2rgb_array_contiguous(classes_ids)[1:]  # convention: background is (0,0,0)
    if return_rgba:
        classes_rgb = np.concatenate([classes_rgb, np.ones(classes_rgb.shape[:-1])[..., None] * 255], axis=1)
    return classes_rgb.astype(np.uint8)

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
    background_label = len(palette)
    # convention: Use highest ID as background, if we have 3 foreground labels (e.g. 0, 1, 2) then len(palette) will be three which is already one bigger then the highest foreground label
    remapped_label_views = np.ones((len(label_view_flat), ), dtype=np.uint16) * background_label  # use (0,0,0) as background color
    for i in range(len(palette)):
        mask = (label_view_flat[:, 0] == palette[i, 0]) &\
               (label_view_flat[:, 1] == palette[i, 1]) &\
               (label_view_flat[:, 2] == palette[i, 2])
        remapped_label_views[mask] = i
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
        else:
            return -1
    else: raise ValueError("Given groundtruth type is not valid.")


# def map_rgb2label(rgb):
#     if np.all(rgb == palette[0, :-1]):
#         return 0
#     elif np.all(rgb == palette[0, :-1]):
#         return 1
#     elif np.all(rgb == palette[0, :-1]):
#         return 2  #
#     else:
#         return BACKGROUND_LABEL  # background


def generate_label_views(kzip_path, gt_type="spgt"):
    assert gt_type in ["axgt", "spgt"], "Currently only spine and axon GT is supported"
    palette = generate_palette(3) # currently in all GT types we only need 3 foreground labels
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
    node_coords = node_coords[node_labels != -1]
    node_labels = node_labels[node_labels != -1]

    # create KD tree from skeleton node coordinates
    tree = KDTree(node_coords)
    # transfer labels from skeleton to mesh
    dist, ind = tree.query(vertices, k=1)

    vertex_labels = node_labels[ind]  # retrieving labels of vertices

    # if no skeleton nodes closer than 2um were found set their label
    # to 2 (shaft; basically this is our background class)
    vertex_labels[dist > 2000] = 2
    # smooth vertex labels
    tree = KDTree(vertices)
    _, ind = tree.query(vertices, k=50)
    # now extract k-closest labels for every vertex
    vertex_labels = vertex_labels[ind]
    # apply majority voting; remove auxiliary axis
    vertex_labels = np.apply_along_axis(majority_element_1d, 1, vertex_labels)[:, 0]
    color_array = palette[vertex_labels].astype(np.float32)/255
    # Initializing mesh object with ground truth coloring
    mo = MeshObject("neuron", indices, vertices, color=color_array)

    # use downsampled locations for view locations, only if they are close to a
    # labeled skeleton node
    locs = np.concatenate(sso.sample_locations())
    dist, ind = tree.query(locs)
    locs = locs[dist[:, 0] < 2000]
    print("Rendering label views.")

    # # DEBUG PART
    # loc_text = ''
    # for i, c in enumerate(locs):
    #     loc_text += str(i+1) + "\t" + str((c / np.array([10, 10, 20])).astype(np.int)) +'\n' #rescalling to the voxel grid
    # with open("/u/shum/view_coordinates_files/view_coords_{}.txt".format(sso_id), "w") as f:
    #     f.write(loc_text)
    # # DEBUG PART
    label_views, rot_mat = _render_mesh_coords(locs, mo, depth_map=False,
                                      return_rot_matrices=True, smooth_shade=False)
    # sso._pred2mesh(node_coords, node_labels, dest_path="/wholebrain/u/shum/sso_%d_skeletonlabels.k.zip" %
    #                                                    sso.id, ply_fname="0.ply")
    print("Rendering index views.")
    index_views = render_sso_coords_index_views(sso, locs,
                                                rot_matrices=rot_mat)
    print("Rendering raw views.")
    raw_views = render_sso_coords(sso, locs)
    print("Remapping views.")
    return raw_views, remap_rgb_labelviews(label_views, palette)[:, None], rgb2id_array(index_views)[:, None]


def GT_generation(kzip_paths, gt_type="spgt"):
    """
    Generates a .npy GT file from all kzip paths.

    Parameters
    ----------
    kzip_paths :
    gt_type :

    Returns
    -------

    """
    # dc = {}
    gt_path = os.path.expanduser("~") + "/spgt_npy_files/"
    params = [(p, "spgt") for p in kzip_paths]
    res = start_multiprocess_imap(gt_generation_helper, params, nb_cpus=20, debug=False)
    all_raw_views = []
    all_label_views = []
    for ii in range(len(kzip_paths)):
        all_raw_views.append(res[ii][0])
        all_label_views.append(res[ii][1])
    all_raw_views = np.concatenate(all_raw_views)
    all_label_views = np.concatenate(all_label_views)
    #TODO label views!!!
    print("Shuffling views.")
    ixs = np.arange(len(all_raw_views))
    np.random.shuffle(ixs)
    all_raw_views = all_raw_views[ixs]
    all_label_views = all_label_views[ixs]
    print("Swapping axes.")
    all_raw_views = all_raw_views.swapaxes(2, 1)
    all_label_views = all_label_views.swapaxes(2, 1)
    print("Reshaping arrays.")
    all_raw_views = all_raw_views.reshape((-1, 4, 256, 128))
    all_label_views = all_label_views.reshape((-1, 1, 256, 128))
    raw_train, raw_other, label_train, label_other = \
        train_test_split(all_raw_views, all_label_views, train_size=0.8, shuffle=True)
    raw_valid, raw_test, label_valid, label_test = \
        train_test_split(raw_other, label_other, train_size=0.5, shuffle=True)
    h5py_path = os.path.expanduser("~") + "/spine_gt_multiview/"
    print("Writing h5 files.")
    save_to_h5py([raw_train], h5py_path + "/raw_train.h5",
                 ["raw"])
    save_to_h5py([raw_valid], h5py_path + "/raw_valid.h5",
                 ["raw"])
    save_to_h5py([raw_test], h5py_path + "/raw_test.h5",
                 ["raw"])
    save_to_h5py([label_train], h5py_path + "/label_train.h5",
                 ["label"])
    save_to_h5py([label_valid], h5py_path + "/label_valid.h5",
                 ["label"])
    save_to_h5py([label_test], h5py_path + "/label_test.h5",
                 ["label"])
        # sso_id = int(re.findall("/(\d+).", kzip_paths[ii])[0])
        # all_views = res[ii]
        # dc[sso_id] = all_views
    # np.save("{}/spgt_semantic_segmentation.npy".format(gt_path), dc)


def gt_generation_helper(args):
    kzip_path, gt_type = args
    sso_id = int(re.findall("/(\d+).", kzip_path)[0])
    raw_views, label_views, index_views = generate_label_views(kzip_path, gt_type)
    # raw shape: (locations, channels, nb_views, 256, 128), label_views: (N,auxiliary axis, nb_views, 256, 128)
    # remap index values to uint16 for fiji compatibility...
    # print("Writing h5 file.")
    # # TODO: check types of views again...
    # h5py_path = os.path.expanduser("~") + "/h5py_files/"
    # save_to_h5py([raw_views[:, 0, 0], label_views[:, 0, 0]], h5py_path + "/sample_views_{}.h5".format(sso_id),
    #              ["raw", "label"])
    # # DEBUG PART
    # subset_ix_views = index_views[0:50, 0, 0]
    # for ii, uv in enumerate(np.unique(subset_ix_views)):
    #     subset_ix_views[subset_ix_views == uv] = ii
    # # # TODO: check types of views again...
    # h5py_path = os.path.expanduser("~") + "/h5py_files/"
    # save_to_h5py([raw_views[0:50, 0, 0], label_views[0:50, 0, 0],
    #               subset_ix_views.astype(np.uint16)], h5py_path + "/sample_views_DEBUG_k50_sm_{}.h5".format(sso_id),
    #              ["raw", "label", "index"])
    # # DEBUG PART
    return raw_views, label_views, index_views


if __name__ == "__main__":
    label_file_folder = "/wholebrain/scratch/areaxfs3/ssv_spgt/" \
                        "spiness_skels_annotated/"
    file_names = ["/23044610.027.k.zip", "/4741011.045.k.zip", "/18279774.052.k.zip", "/26331138.037.k.zip", "/27965455.029.k.zip"]
    file_paths = [label_file_folder + "/" + fname for fname in file_names]
    GT_generation(file_paths)
