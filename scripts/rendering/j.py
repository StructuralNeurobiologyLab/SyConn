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
# from syconn.handler.compression import save_to_h5py
# from scripts.rendering.inversed_mapping import id2rgb_array
# #import matplotlib.pylab as plt
# from imageio import imwrite
import re
import os
import time

# define palette, but also take care of inverse mapping 'remap_rgb_labelviews'
# due to speed issues labels have to be given axis wise:
#  e.g. (1, 0, 0), (2, 0, 0), ..., (255, 0, 0) and (0, 1, 0), ... (0, 255, 0)
# this defines rgb values for labels 0, 1 and 2
def generate_palette(nr_classes):
    classes_ids = np.arange(nr_classes+1) #reserve additional class id for background
    classes_rgb = id2rgb_array_contiguous(classes_ids)[1:].astype(np.float32) / 255  # convention: background is (0,0,0)
    return classes_rgb
print(generate_palette())