# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Sven Dorkenwald, Philipp Schubert, Joergen Kornfeld
import os

# ---------------------- STATIC AND GLOBAL PARAMETERS # -----------------------

# --------- GLOBAL WORKING DIRECTORY
# wd = "/wholebrain/scratch/areaxfs3/"
wd = "/wholebrain/songbird/j0126/areaxfs_v6/"
# wd = '/mnt/j0126/areaxfs_v10/'

# --------- BACKEND DEFINITIONS
# TODO: Use batchjob_script_folder everywhere or remove because this is the default now.
BATCH_PROC_SYSTEM = 'SLURM'  # If None, fall-back is single node multiprocessing
batchjob_script_folder = os.path.dirname(os.path.abspath(__file__)) + \
                         "/../QSUB_scripts/"
MEM_PER_NODE = 249.5e3  # in MB
NCORES_PER_NODE = 20
# TOOD: Use NCORE_TOTAL everywhere
NCORE_TOTAL = 340
# TODO: Use NGPU_TOTAL everywhere
NGPU_TOTAL = 34

# TODO: add generic parser method for initial RAG and handle case without glia-splitting
path_initrag = '/wholebrain/songbird/j0126/RAGs/v4b_20180407_v4b_20180407_'\
               'merges_newcb_ids_cbsplits.txt'
# currently a mergelist/RAG of the following form is expected:
# ID, ID
#    .
#    .
# ID, ID
rag_suffix = ""  # identifier in case there will be more than one RAG

# --------- LOGGING
# 'None' disables logging of SyConn modules (e.g. proc, handler, ...) to files.
# Logs of executed scripts (syconn/scripts) will be stored at the
# working directory + '/logs/' nonetheless.
default_log_dir = None

# --------- BACKEND DEFINITIONS
backend = "FS"  # File system
PYOPENGL_PLATFORM = 'osmesa'  # Rendering

py36path = '/u/pschuber/anaconda3/envs/py36/bin/python'  # TODO: make this more elegant, e.g. bash script with 'source activate py36'

# file logging for individual modules, and per job. Only use in case of
# debugging with single core processing. Logs for scripts in 'SyConn/scripts/'
# will be stored at wd + '/logs/'.
DISABLE_FILE_LOGGING = True

# --------- CONTACT SITE PARAMETERS
cs_gap_nm = 300

# --------- MESH PARAMETERS
existing_cell_organelles = ['mi', 'sj', 'vc', 'syn_ssv']
MESH_DOWNSAMPLING = {"sv": (8, 8, 4), "sj": (2, 2, 1), "vc": (4, 4, 2),
                     "mi": (8, 8, 4), "cs": (2, 2, 1), "conn": (2, 2, 1),
                     'syn_ssv': (2, 2, 1)}
MESH_CLOSING = {"sv": 0, "sj": 0, "vc": 0, "mi": 0, "cs": 0,
                "conn": 4, 'syn_ssv': 20}

MESH_MIN_OBJ_VX = 10

# --------- VIEW PARAMETERS
NB_VIEWS = 2

# --------- GLIA PARAMETERS
# min. connected component size of glia nodes/SV after thresholding glia proba
min_cc_size_glia = 8e3  # in nm; L1-norm on vertex bounding box
# min. connected component size of neuron nodes/SV after thresholding glia proba
min_cc_size_neuron = 8e3  # in nm; L1-norm on vertex bounding box

glia_thresh = 0.161489   # Threshold for glia classification
SUBCC_SIZE_BIG_SSV = 35  # number of sv used during local rendering. The total number of SV used are SUBCC_SIZE_BIG_SSV + 2*(SUBCC_CHUNKE_SIZE_BIG_SSV-1)
RENDERING_MAX_NB_SV = 5e3
SUBCC_CHUNK_SIZE_BIG_SSV = 9  # number of SV for which views are rendered in one pass

# --------- CLASSIFICATION MODELS
model_dir = wd + '/models/'
mpath_tnet = '{}/TN-10-Neighbors/'.format(model_dir)
mpath_spiness = '{}/FCN-VGG13--Lovasz--NewGT/'.format(model_dir)
mpath_celltype = '{}/celltype_g1_20views_v3/g1_20views_v3-FINAL.mdl'.format(model_dir)
mpath_axoness = '{}/axoness_g1_v2/g1_v2-FINAL.mdl'.format(model_dir)
mpath_glia = '{}/glia_g0_v0/g0_v0-FINAL.mdl'.format(model_dir)

# --------- RFC PARAMETERS
SKEL_FEATURE_CONTEXT = {"axoness": 8000, "spiness": 1000}  # in nm

# --------- SPINE PARAMETERS
min_spine_cc_size = 10
min_edge_dist_spine_graph = 110

# --------- COMPARTMENT PARAMETERS
DIST_AXONESS_AVERAGING = 10000

# --------- CELLTYPE PARAMETERS


# TODO: Put all static parameters from config.ini into this file.
# TODO: All versioning can stay in config.ini because it this is dynamic
def get_dataset_scaling():
    """
    Helper method to get dataset scaling.

    Returns
    -------
    tuple of float
        (X, Y, Z)
    """
    from .parser import Config
    import numpy as np
    cfg = Config(wd)
    return np.array(cfg.entries["Dataset"]["scaling"])
