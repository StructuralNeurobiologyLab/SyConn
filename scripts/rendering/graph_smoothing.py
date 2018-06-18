# SyConn
# Copyright (c) 2018
# All rights reserved

import numpy as np
from knossos_utils.skeleton_utils import load_skeleton
from sklearn.neighbors import KDTree
from syconn.proc.meshes import MeshObject, rgb2id_array, id2rgb_array_contiguous
from syconn.handler.basics import majority_element_1d
from syconn.proc.rendering import render_sso_coords, _render_mesh_coords,\
    render_sso_coords_index_views
from syconn.reps.super_segmentation import SuperSegmentationObject
import networkx as nx


def graph_creator(indices, vertices):
    G = nx.Graph()
    G.add_nodes_from(indices)
    triangles = indices.reshape((-1,3))
    for i in range(len(triangles)):
        for a, b, c in range(triangles[i]):
        G.add_edge(a, b)
        G.add_edge(b, c)
        G.add_edge(a, c)






def new_label_views():


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

    # DEBUG PART
    loc_text = ''
    for i, c in enumerate(locs):
        loc_text += str(i+1) + "\t" + str((c / np.array([10, 10, 20])).astype(np.int)) +'\n' #rescalling to the voxel grid
    with open("/u/shum/view_coordinates_files/view_coords_{}.txt".format(sso_id), "w") as f:
        f.write(loc_text)
    # DEBUG PART
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