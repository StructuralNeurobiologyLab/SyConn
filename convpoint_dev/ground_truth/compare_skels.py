# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max-Planck-Institute of Neurobiology, Munich, Germany
# Authors: Jonathan Klimesch

import os
import glob
import re
from syconn import global_params
from syconn.handler.multiviews import generate_rendering_locs
from syconn.proc.graphs import create_graph_from_coords
from syconn.reps.super_segmentation import SuperSegmentationObject
from syconn.reps.super_segmentation_helper import create_sso_skeleton_fast


if __name__ == "__main__":
    # set paths
    dest_dir = "/wholebrain/u/jklimesch/gt/skeleton_comparison/"
    global_params.wd = "/wholebrain/songbird/j0126/areaxfs_v6/"
    label_file_folder = "/wholebrain/u/jklimesch/gt/gt_julian/"

    files = glob.glob(label_file_folder + '*.k.zip')

    if not os.path.isdir(dest_dir):
        os.makedirs(dest_dir)

    for file in files:
        # load sso and respective mesh
        sso_id = int(re.findall(r"/(\d+).", file)[0])
        sso = SuperSegmentationObject(sso_id)
        indices, vertices, normals = sso.mesh
        vertices = vertices.reshape((-1, 3))

        # Option 1
        sso = create_sso_skeleton_fast(sso, max_dist_thresh_iter2=10000)
        sso.save_skeleton_to_kzip(dest_dir+str(sso_id)+'_test.k.zip')

        # Option 2
        # for i in (500, 1000, 2000, 3000, 5000):
        #     locs = generate_rendering_locs(vertices, i)
        #     graph = create_graph_from_coords(locs, mst=True)
        #     skel = {'edges': np.array(graph.edges), 'nodes': locs/sso.scaling, 'diameters': np.ones(len(locs))*15}
        #     sso.skeleton = skel
        #     sso.save_skeleton_to_kzip(dest_dir+str(sso_id)+'_rl_{}.k.zip'.format(i))
