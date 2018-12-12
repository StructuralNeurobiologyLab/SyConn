# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Sven Dorkenwald, Philipp Schubert, Joergen Kornfeld

# --------- Define global working directory
# wd = "/wholebrain/songbird/j0126/areaxfs_v5"
# wd = "/wholebrain/scratch/areaxfs3/"
wd = "/wholebrain/songbird/j0126/areaxfs_v6/"
# wd = '/mnt/j0126/areaxfs_v10/'

# --------- Define backend
backend = "FS"

# ---------- Rendering backend
PYOPENGL_PLATFORM = 'osmesa'

# --------------------------------------------------------------- GLIA PARAMETER
# min. connected component size of glia nodes/SV after thresholding glia proba
min_cc_size_glia = 8e3  # in nm; L1-norm on vertex bounding box
# min. connected component size of neuron nodes/SV after thresholding glia proba
min_cc_size_neuron = 8e3  # in nm; L1-norm on vertex bounding box
min_single_sv_size = 30000  # in number of voxels
glia_thresh = 0.161489   # Threshold for glia classification
SUBCC_SIZE_BIG_SSV = 40  # number of sv used during local rendering. The total number of SV used are SUBCC_SIZE_BIG_SSV + 2*(SUBCC_CHUNKE_SIZE_BIG_SSV-1)
RENDERING_MAX_NB_SV = 5e3
SUBCC_CHUNK_SIZE_BIG_SSV = 31  # number of SV for which views are rendered in one pass

# --------------------------------------------------------------- MESH PARAMETER
MESH_DOWNSAMPLING = {"sv": (8, 8, 4), "sj": (2, 2, 1), "vc": (4, 4, 2),
                     "mi": (8, 8, 4), "cs": (2, 2, 1), "conn": (2, 2, 1)}
MESH_CLOSING = {"sv": 0, "sj": 0, "vc": 0, "mi": 0, "cs": 0, "conn": 4}
SKEL_FEATURE_CONTEXT = {"axoness": 8000, "spiness": 1000}  # in nm
DISABLE_FILE_LOGGING = True

# -------------------------------------------------------- CLASSIFICATION MODELS
model_dir = wd + '/models/'
mpath_tnet = '{}/TN-10-Neighbors/'.format(model_dir)
mpath_spiness = '{}/FCN-VGG13--Lovasz--NewGT/'.format(model_dir)
mpath_celltype = '{}/celltype_g1_20views_v3/g1_20views_v3-FINAL.mdl'.format(model_dir)
mpath_axoness = '{}/axoness_g1_v3/g1_v3-FINAL.mdl'.format(model_dir)
mpath_glia = '{}/glia_g0_v0/g0_v0-FINAL.mdl'.format(model_dir)

# --------------------------------------------------------------- VIEW PARAMETER
NB_VIEWS = 2

# -------------------------------------------------------------- SPINE PARAMETER
min_spine_cc_size = 10
min_edge_dist_spine_graph = 110

# -------------------------------------------------------- COMPARTMENT PARAMETER
DIST_AXONESS_AVERAGING = 10000

# ----------------------------------------------------------- CELLTYPE PARAMETER


def get_dataset_scaling():
    """
    Helper method to get dataset scaling.

    Returns
    -------
    tuple of float
        (X, Y, Z)
    """
    from .parser import Config
    cfg = Config(wd)
    return cfg.entries["Dataset"]["scaling"]
