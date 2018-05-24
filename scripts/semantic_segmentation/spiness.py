# SyConn
# Copyright (c) 2016 Philipp J. Schubert
# All rights reserved

# download/import all necessary work packages
import numpy as np
from knossos_utils.skeleton_utils import load_skeleton
from sklearn.neighbors import KDTree
from syconn.proc.meshes import MeshObject, rgb2id_array
from syconn.handler.basics import majority_element_1d
from syconn.proc.rendering import render_sso_coords, _render_mesh_coords,\
    render_sso_coords_index_views
from syconn.reps.super_segmentation import SuperSegmentationObject
from syconn.handler.compression import save_to_h5py
#import matplotlib.pylab as plt
from imageio import imwrite
import re
import os
import time
BACKGROUND_LABEL = 3

# define palette, but also take care of inverse mapping 'remap_rgb_labelviews'
# due to speed issues labels have to be given axis wise:
#  e.g. (1, 0, 0), (2, 0, 0), ..., (255, 0, 0) and (0, 1, 0), ... (0, 255, 0)
# this defines rgb values for labels 0, 1 and 2
palette = np.array([[1, 0, 0, 1],  # red, e.g. dendrite
                    [0, 1, 0, 1],  # green, e.g. axon
                    [0, 0, 1, 1]],  # blue, e.g. soma
                   dtype=np.uint8)
# create function that converts information in string type to the information in integer type


def remap_rgb_labelviews(rgb_view):
    label_view_flat = rgb_view.flatten().reshape((-1, 3))
    remapped_label_views = np.ones((len(label_view_flat), ), dtype=np.uint16) * BACKGROUND_LABEL
    remapped_label_views[label_view_flat[:, 0] == 255] = 0
    remapped_label_views[label_view_flat[:, 1] == 255] = 1
    remapped_label_views[label_view_flat[:, 2] == 255] = 2
    # MAKE SURE BACKGROUND_LABEL IS BIGGER THAN BIGGEST FOREGROUND LABEL # TODO: make this automated..
    # print("Finisehd remapping rgb-> label IDs after [min]:", (time.time()-start)/60.)
    return remapped_label_views.reshape(rgb_view.shape[:-1])


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


def generate_label_views(kzip_path, gt_type="axgt"):
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
    _, ind = tree.query(vertices, k=100)
    # now extract k-closest labels for every vertex
    vertex_labels = vertex_labels[ind]
    # apply majority voting; remove auxiliary axis
    vertex_labels = np.apply_along_axis(majority_element_1d, 1, vertex_labels)[:, 0]
    color_array = palette[vertex_labels]
    # Initializing mesh object with ground truth coloring
    mo = MeshObject("neuron", indices, vertices, color=color_array)

    # use downsampled locations for view locations, only if they are close to a
    # labeled skeleton node
    locs = np.concatenate(sso.sample_locations())
    dist, ind = tree.query(locs)
    locs = locs[dist[:, 0] < 2000]
    print("rendering labels")
    label_views, rot_mat = _render_mesh_coords(locs, mo, depth_map=False,
                                      return_rot_matrices=True)
    sso._pred2mesh(node_coords, node_labels, dest_path="/wholebrain/scratch/pschuber/sso_%d_skeletonlabels.k.zip" % sso.id, ply_fname="0.ply")
    print("rendering index")
    index_views = render_sso_coords_index_views(sso, locs, rot_matrices=rot_mat)
    print("rendering raw")
    raw_views = render_sso_coords(sso, locs)

    # DEBUG PRINTS
    print(locs[:10] / np.array([10, 10, 20]))
    # print coordinates of vertices rendered in index views 0..9
    vertices_unscaled = sso.mesh[1].reshape((-1, 3)) / np.array([10, 10, 20])
    for ii in range(10):
        print(vertices_unscaled[index_views[ii, 0].flatten()])

    return raw_views, remap_rgb_labelviews(label_views), rgb2id_array(index_views)


if __name__ == "__main__":
    #print(generate_label_views)
    label_file_folder = "/wholebrain/scratch/areaxfs3/ssv_spgt/" \
                        "spiness_skels_annotated/"
    # kzip_path = label_file_folder + "/28985344.001.k.zip"
    kzip_path = label_file_folder + "/23044610.008.k.zip"
    raw_views, label_views, index_views = generate_label_views(kzip_path, gt_type="spgt")
    # raw shape: (locations, channels, nb_views, 256, 128), label_views: (N, nb_views, 256, 128)
    # remap index values to uint16 for fiji compatibility...
    subset_ix_views = index_views[:10, 0]
    for ii, uv in enumerate(np.unique(subset_ix_views)):
        subset_ix_views[subset_ix_views == uv] = ii
    home = os.path.expanduser("~")
    # TODO: check types of views again...
    save_to_h5py([raw_views[:10, 0, 0], label_views[:10, 0],
                  subset_ix_views.astype(np.uint16)], home + "/sample_views.h5",
                 ["raw", "label", "index"])
    # label_views = remap_rgb_labelviews(label_views)