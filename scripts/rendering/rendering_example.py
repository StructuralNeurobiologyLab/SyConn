# download/import all necessary work packages

import numpy as np
from knossos_utils.skeleton_utils import load_skeleton
from sklearn.neighbors import KDTree
from syconn.proc.meshes import MeshObject
from syconn.proc.rendering import render_mesh
import matplotlib.pylab as plt


# create function that converts information in string type to the information in integer type
def str2intconverter(comment):
    if comment == "gt_axon":
        return 1
    elif comment == "gt_dendrite":
        return 0
    elif comment == "gt_soma":
        return 2
    else:
        return -1

# define pathes to skeleton(kzip file) and mesh(only vertices, npy file)

kzip_path = "/home/mshumlia/devel/helloworld/28_03_2018/28985344.001.k.zip"
ply_file_vert = "/home/mshumlia/devel/helloworld/3_04_2018/28985344_vert.npy"
ply_file_ind = "/home/mshumlia/devel/helloworld/3_04_2018/28985344_ind.npy"

# Load mesh
vertices = np.load(ply_file_vert) #vertices(coordinates) are usually stored in 1D array!
vertices = vertices.reshape((-1, 3))


# in order to proceed, we have to create 2D array so, that number of columns would
# be const=3(number of coordinates in space) and the number of lines would be variable so far, so we have
# always 3 coordinates (to that refers -1)


indices = np.load(ply_file_ind)

# load skeleton
valid_comments = ['gt_axon', "gt_dendrite", "gt_soma"]
skel = load_skeleton(kzip_path)["skeleton"]
skel_nodes = list(skel.getNodes())


# can be done also in a single for loop over "for n in skel_nodes:"
node_coords = np.array([n.getCoordinate_scaled() for n in skel_nodes]) #extracting node coordinates, rescalling from
# voxel frame to physical frame with _scaled())
node_labels = np.array([str2intconverter(n.getComment()) for n in skel_nodes], dtype=np.int)
node_coords = node_coords[node_labels != -1]
node_labels = node_labels[node_labels != -1]

# create KD tree from skeleton node coordinates
tree = KDTree(node_coords)

# transfer labels from skeleton to mesh
dist, ind = tree.query(vertices, k=1) #always two functions dist, ind; KD tree is built from node coordinates;
# we are querying in mesh coordinates(vertices) to find out, which node in mesh is closes (1, 0, 0, 1)t to the node in skeleton; the
# result is indices of sceleton nodes
nearest_skeleton_node_labels = node_labels[ind] #retrieving labels from indices

# labels = nearest_skeleton_node_labels
# print(labels.shape)
# labels = labels.squeeze()
# print(labels.shape)

#transfering labels to RGBA values using dictionary
#
# rgba_dict = {0: (1, 0, 0, 1), 1:(0, 1, 0, 1), 2:(0, 0, 1, 1)}
#
# #creating color array
# color_array = np.zeros((len(labels), 4))
# for i in range(len(labels)):
#     color_array[i] = rgba_dict[labels[i]]

palette = np.array( [ [1,0,0,1],              # red
                      [0,1,0,1],              # green
                      [0,0,1,1] ])            # blue

color_array = palette[labels]

#Initializing mesh object
mo = MeshObject("neuron", indices, vertices, color = color_array)
render = render_mesh(mo)
print(render.shape)
plt.imshow(render[0])
plt.figure()
plt.imshow(render[1])
plt.figure()
plt.imshow(render[2])


