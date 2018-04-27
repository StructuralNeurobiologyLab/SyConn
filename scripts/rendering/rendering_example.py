# download/import all necessary work packages

import numpy as np
from knossos_utils.skeleton_utils import load_skeleton
from sklearn.neighbors import KDTree
from syconn.proc.meshes import MeshObject
from syconn.proc.rendering import render_mesh, _render_mesh_coords, render_sso_coords
from syconn.reps.super_segmentation import SuperSegmentationObject
#import matplotlib.pylab as plt
from scipy.misc import imsave
import re
import os

#Plan:
#1)


#create color palette for rgb to labels labelling

palette = np.array([[1, 0, 0, 1],  # red
                    [0, 1, 0, 1],  # green
                    [0, 0, 1, 1]])  # blue


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


def map_rgb2label(rgb):
    if np.all(rgb == palette[0, :-1]):
        return 1  # 1 refers to dendrite
    elif np.all(rgb == palette[0, :-1]):
        return 2  # 2 refers to axon
    elif np.all(rgb == palette[0, :-1]):
        return 3  # 2 refers to soma
    else:
        return 0  # background


def generate_GT_views(kzip_path, gt_type="axgt"):
    """

    :param kzip_path:
    :param gt_type: str
        Either "axgt" for axoness groundtruth or "spgt" for spiness groundtruth
    :return:
    """
    # define pathes to skeleton(kzip file) and mesh(only vertices, npy file)
    # TODO: retrieve vert_fname and ind_fname from kzip_path, use package 're" (regular expression) for that
    # retrieve the indentefication number of  sso
    #base_folder = os.path.split(kzip_path)[0] + "/"
    sso_id = int(re.findall("/(\d+).", kzip_path)[0])
    # vert_fname = base_folder + str(sso_id) + "_vert.npy"
    # ind_fname = base_folder + str(sso_id) + "_ind.npy"
    sso = SuperSegmentationObject(sso_id, version=gt_type)
    indices, vertices, normals = sso.mesh
    # # Load mesh
    # vertices = np.load(vert_fname)  # vertices(coordinates) are usually stored in 1D array!
    vertices = vertices.reshape((-1, 3))
    #
    # # in order to proceed, we have to create 2D array so, that number of columns would
    # # be const=3(number of coordinates in space) and the number of lines would be variable so far, so we have
    # # always 3 coordinates (to that refers -1)
    #
    # indices = np.load(ind_fname)

    # load skeleton
    skel = load_skeleton(kzip_path)["skeleton"]
    skel_nodes = list(skel.getNodes())

    # can be done also in a single for loop over "for n in skel_nodes:"
    node_coords = np.array(
        [n.getCoordinate_scaled() for n in skel_nodes])  # extracting node coordinates, rescalling from
    # voxel frame to physical frame with _scaled())
    node_labels = np.array([str2intconverter(n.getComment(), gt_type) for n in skel_nodes], dtype=np.int)
    node_coords = node_coords[node_labels != -1][:10]
    node_labels = node_labels[node_labels != -1][:10]

    # create KD tree from skeleton node coordinates
    tree = KDTree(node_coords)

    # transfer labels from skeleton to mesh
    dist, ind = tree.query(vertices, k=1)  # always two functions dist, ind; KD tree is built from node coordinates;
    # we are querying in mesh coordinates(vertices) to find out, which node in mesh is closes (1, 0, 0, 1)t to the node in skeleton; the
    # result is indices of sceleton nodes
    nearest_skeleton_node_labels = node_labels[ind]  # retrieving labels of vertices

    labels = nearest_skeleton_node_labels
    # print(labels.shape)
    # labels = labels.squeeze()
    # print(labels.shape)

    # transfering labels to RGBA values using dictionary
    #
    # rgba_dict = {0: (1, 0, 0, 1), 1:(0, 1, 0, 1), 2:(0, 0, 1, 1)}
    #
    # #creating color array
    # color_array = np.zeros((len(labels), 4))
    # for i in range(len(labels)):
    #     color_array[i] = rgba_dict[labels[i]]

    color_array = palette[labels]
    # Initializing mesh object
    mo = MeshObject("neuron", indices, vertices, color=color_array)

    label_views = _render_mesh_coords(node_coords, mo, depth_map=False)
    mo._colors = None
    raw_views = _render_mesh_coords(node_coords, mo, depth_map=True)

    # label_views = render_mesh(mo)
    return label_views, raw_views



def remap_rgb_view(rgb_view):  # (3, 2048, 2048, 3)
    label_view_flat = rgb_view.flatten()
    label_view_resh = label_view_flat.reshape((-1, 3))
    remappend_label_view = np.zeros((len(label_view_resh), ))
    for ix in range(len(label_view_resh)):
        l = map_rgb2label(label_view_resh[ix])
        remappend_label_view[ix] = l
    return remappend_label_view.reshape(rgb_view.shape[:-1])  # (3, 2048, 2048, )

if __name__ == "__main__":
    #print(generate_label_views)
    label_file_folder = "/u/shum/spine_label_files/"
    # kzip_path = label_file_folder + "/28985344.001.k.zip"
    kzip_path = label_file_folder + "/4741011.k.zip"
    label_views, raw_views = generate_GT_views(kzip_path, gt_type="spgt")
    # label view shapes: (N, 2, 256, 128, 1), raw views:(N, 2, 256, 128, 4)
    imsave("/u/shum/labels0.png", label_views[0, 0])
    raise()
    imsave("/u/shum/raw0.png", raw_views[0, 0, ..., ])
    remapped_label_views = remap_rgb_view(label_views)
    imsave ("/u/shum/rmlabels0.png", remapped_label_views[0, 0])

#print(render.shape)

# (3, 2048, 2048, 3), (number views, x, y, number channels RGB)
#from scipy.misc import imsave
#imsave("/u/shum/test4.png", render[2])

#plt.imshow(render[0])
#plt.figure()
#plt.imshow(render[1])
#plt.figure()
#plt.imshow(render[2])


