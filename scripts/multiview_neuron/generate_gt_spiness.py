# SyConn
# Copyright (c) 2016 Philipp J. Schubert
# All rights reserved

# download/import all necessary work packages
import numpy as np
from knossos_utils.skeleton_utils import load_skeleton
from sklearn.neighbors import KDTree
from syconn.proc.meshes import MeshObject
from syconn.proc.graphs import bfs_smoothing
from syconn.proc.rendering import render_sso_coords, _render_mesh_coords,\
    render_sso_coords_index_views
from syconn.reps.super_segmentation import SuperSegmentationObject
from syconn.reps.views import ViewContainer
from syconn.handler.compression import save_to_h5py
from syconn.handler.multiviews import generate_palette, remap_rgb_labelviews, str2intconverter
from syconn.mp.mp_utils import start_multiprocess_imap
import re
import os
from scipy.misc import imsave
from sklearn.model_selection import train_test_split


def generate_label_views(kzip_path, gt_type="spgt", n_voting=40, nb_views=2,
                         ws=(256, 128), comp_window=8e3, initial_run=False):
    """

    Parameters
    ----------
    kzip_path : str
    gt_type :  str
    n_voting : int
        Number of collected nodes during BFS for majority vote (label smoothing)
    nb_views : int
    ws: Tuple[int]
    comp_window : float
    initial_run : bool
        if True, will copy SSV from default SSD to SSD with version=gt_type

    Returns
    -------

    """
    assert gt_type in ["axgt", "spgt"], "Currently only spine and axon GT is supported"
    n_labels = 3 if gt_type == "axgt" else 4
    palette = generate_palette(n_labels)
    sso_id = int(re.findall("/(\d+).", kzip_path)[0])
    sso = SuperSegmentationObject(sso_id, version=gt_type)
    if initial_run:  # use default SSD version
        orig_sso = SuperSegmentationObject(sso_id)
        orig_sso.copy2dir(dest_dir=sso.ssv_dir)
    if not sso.attr_dict_exists:
        msg = 'Attribute dict of original SSV was not copied successfully ' \
              'to target SSD.'
        raise ValueError(msg)
    sso.load_attr_dict()
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

    # for getting vertex GT
    # np.save("/wholebrain/u/pschuber/spiness_skels/sso_%d_vertlabels.k.zip" % sso.id, vertex_labels)
    # np.save("/wholebrain/u/pschuber/spiness_skels/sso_%d_verts.k.zip" % sso.id, vertices)

    # for getting colored meshes
    # colors = [[0.6, 0.6, 0.6, 1], [0.9, 0.2, 0.2, 1], [0.1, 0.1, 0.1, 1], [0.05, 0.6, 0.6, 1], [0.9, 0.9, 0.9, 1]]
    # colors = np.array(colors) * 255
    # color_array = (colors[vertex_labels].astype(np.float32))[:, 0]
    # write_mesh2kzip("/wholebrain/u/pschuber/spiness_skels/sso_%d_skeletonlabels.k.zip" % sso.id,
    #                 sso.mesh[0], sso.mesh[1], sso.mesh[2], color_array,
    #                 ply_fname="spiness.ply")

    # Initializing mesh object with ground truth coloring
    mo = MeshObject("neuron", indices, vertices, color=color_array)

    # use downsampled locations for view locations, only if they are close to a
    # labeled skeleton node
    locs = np.concatenate(sso.sample_locations(cache=False))
    dist, ind = tree.query(locs)
    locs = locs[dist[:, 0] < 2000]#[::3][:5]


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
                                               comp_window=comp_window, verbose=True)
    label_views = remap_rgb_labelviews(label_views, palette)[:, None]
    index_views = render_sso_coords_index_views(sso, locs, rot_mat=rot_mat, verbose=True,
                                                nb_views=nb_views, ws=ws, comp_window=comp_window)
    raw_views = render_sso_coords(sso, locs, nb_views=nb_views, ws=ws,
                                  comp_window=comp_window, verbose=True,
                                  rot_mat=rot_mat)
    return raw_views, label_views, index_views


def GT_generation(kzip_paths, nb_views, dest_dir=None, gt_type="spgt",
                  n_voting=40, ws=(256, 128), comp_window=8e3):
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
        dest_dir = os.path.expanduser("~") + "/spine_gt_multiview_biggercontext/"
    if not os.path.isdir(dest_dir):
        os.makedirs(dest_dir)
    params = [(p, gt_type, n_voting, nb_views, ws, comp_window) for p in kzip_paths]
    dest_p = os.path.expanduser("~") + "/spiness_skels_biggercontext/cache_{}votes".format(n_voting)
    if not os.path.isdir(dest_p):
        os.makedirs(dest_p)
    start_multiprocess_imap(gt_generation_helper, params, nb_cpus=5,
                            debug=False)
    # Create Dataset splits for training, validation and test
    all_raw_views = []
    all_label_views = []
    # all_index_views = []  # Removed index views
    for ii in range(len(kzip_paths)):
        sso_id = int(re.findall("/(\d+).", kzip_paths[ii])[0])
        dest_p = os.path.expanduser("~") + "/spiness_skels_biggercontext/cache_{}votes/{}/".format(n_voting, sso_id)
        raw_v = np.load(dest_p + "raw.npy")
        label_v = np.load(dest_p + "label.npy")
        # index_v = np.load(dest_p + "index.npy")  # Removed index views
        all_raw_views.append(raw_v)
        all_label_views.append(label_v)
        # all_index_views.append(index_v)  # Removed index views
    all_raw_views = np.concatenate(all_raw_views)
    all_label_views = np.concatenate(all_label_views)
    # all_index_views = np.concatenate(all_index_views)  # Removed index views
    print("Shuffling views.")
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
    all_raw_views = all_raw_views.reshape((-1, 4, 128, 256))
    all_label_views = all_label_views.reshape((-1, 1, 128, 256))
    # all_index_views = all_index_views.reshape((-1, 1, 128, 256))  # Removed index views
    # all_raw_views = np.concatenate([all_raw_views, all_index_views], axis=1)  # Removed index views
    raw_train, raw_valid, label_train, label_valid = train_test_split(all_raw_views, all_label_views, train_size=0.8, shuffle=False)
    # raw_valid, raw_test, label_valid, label_test = train_test_split(raw_other, label_other, train_size=0.5, shuffle=False)  # Removed index views
    print("Writing h5 files.")
    save_to_h5py([raw_train], dest_dir + "/raw_train_v2.h5",
                 ["raw"])
    save_to_h5py([raw_valid], dest_dir + "/raw_valid_v2.h5",
                 ["raw"])
    # save_to_h5py([raw_test], dest_dir + "/raw_test.h5",
    # ["raw"])  # Removed index views
    save_to_h5py([label_train], dest_dir + "/label_train_v2.h5",
                 ["label"])
    save_to_h5py([label_valid], dest_dir + "/label_valid_v2.h5",
                 ["label"])
    # save_to_h5py([label_test], dest_dir + "/label_test.h5",
    # ["label"])  # Removed index views


def gt_generation_helper(args):
    kzip_path, gt_type, n_voting, nb_views, ws, comp_window = args
    raw_views, label_views, index_views = generate_label_views(kzip_path, gt_type,
                                                               n_voting, nb_views, ws, comp_window)

    sso_id = int(re.findall("/(\d+).", kzip_path)[0])
    dest_p = os.path.expanduser("~") + "/spiness_skels_biggercontext/cache_{}votes/{}/".format(n_voting, sso_id)
    if not os.path.isdir(dest_p):
        os.makedirs(dest_p)
    np.save(dest_p + "raw.npy", raw_views)
    np.save(dest_p + "label.npy", label_views)
    # np.save(dest_p + "index.npy", index_views)

    # DEBUG PART START, write out images for manual inspection in Fiji
    from syconn.reps.super_segmentation_object import merge_axis02
    # raw_views_wire = merge_axis02(raw_views_wire)[:, :, None]
    raw_views = merge_axis02(raw_views)[:, :, None][:10]
    label_views = merge_axis02(label_views)[:, :, None][:10]
    index_views = merge_axis02(index_views)[:, :, None][:10]
    sso_id = int(re.findall("/(\d+).", kzip_path)[0])
    h5py_path = os.path.expanduser("~") + "/spiness_skels_biggercontext/{}/view_imgs_{}/".format(sso_id, n_voting)
    if not os.path.isdir(h5py_path):
        os.makedirs(h5py_path)
    # save_to_h5py([raw_views[:, 0, 0], label_views[:, 0, 0],
    #               index_views[:, 0, 0]], h5py_path + "/views.h5",
    #              ["raw", "label", "index"])
    vc = ViewContainer("", views=raw_views)
    # vc_wire = ViewContainer("", views=raw_views_wire)
    # randomize color map of index views
    colored_indices = np.zeros(list(index_views.shape) + [3], dtype=np.uint8)
    for ix in np.unique(index_views):
        rand_col = np.random.randint(0, 256, 3)
        colored_indices[index_views == ix] = rand_col
    for ii in range(len(raw_views)):
        # vc_wire.write_single_plot("{}/{}_raw_wire.png".format(h5py_path, ii), ii)
        vc.write_single_plot("{}/{}_raw.png".format(h5py_path, ii), ii)
        imsave(h5py_path + "{}_label.png".format(ii), label_views[:, 0, 0][ii])
        imsave(h5py_path + "{}_index.png".format(ii), colored_indices[:, 0, 0][ii])
    # DEBUG PART END


if __name__ == "__main__":
    n_views = 5
    label_file_folder = "/wholebrain/scratch/areaxfs3/ssv_spgt/" \
                        "spiness_skels_annotated/"
    file_names = ["/27965455.039.k.zip",
                  "/23044610.037.k.zip", "/4741011.074.k.zip",
                  "/18279774.089.k.zip", "/26331138.046.k.zip",
                  ]
    # stored both valid and test data as validation data for network: (2280, 1, 128, 256)
    # train data: (9120, 6, 128, 256), accidently concatenated raw views twice
    file_paths = [label_file_folder + "/" + fname for fname in file_names][::-1]
    GT_generation(file_paths, n_views, ws=(512, 256), comp_window=16e3)

    # start_multiprocess_imap(generate_label_views, file_paths)