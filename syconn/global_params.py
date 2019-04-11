# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Sven Dorkenwald, Philipp Schubert, Joergen Kornfeld

import warnings
import os
import logging
from .handler.config import DynConfig

warnings.filterwarnings("ignore", message=".*You are using implicit channel selection.*")
warnings.filterwarnings("ignore", message=".*You are initializing a KnossosDataset from a path.*")
warnings.filterwarnings("ignore", message=".*dataset.value has been deprecated.*")  # h5py deprecation warning

# ---------------------- STATIC AND GLOBAL PARAMETERS # -----------------------
# --------- GLOBAL WORKING DIRECTORY
# wd = "/wholebrain/scratch/areaxfs3/"
# wd = "/wholebrain/songbird/j0126/areaxfs_v6/"
wd = None
# wd = "/wholebrain/scratch/areaxfs3/"
# wd = '/mnt/j0126/areaxfs_v10/'

# TODO: Put all dataset-specific parameters into config.ini / configspec.ini

rag_suffix = ""  # identifier in case there will be more than one RAG, TODO: Remove

# All subsquent parameter are dataset independent and do not have to be stored at
# config.ini in the working directory

# --------- BACKEND DEFINITIONS
BATCH_PROC_SYSTEM = 'SLURM'  # If None, fall-back is single node multiprocessing
batchjob_script_folder = os.path.dirname(os.path.abspath(__file__)) + \
                         "/batchjob_scripts/"
# TODO refactor syconn and get rid of all qsub_pe and qsub_queue kwargs and only use batch_job_enabled(),
#  the default in QSUB_script should then be BATCH_PE and BATCH_QUEUE
BATCH_PE = 'default'
BATCH_QUEUE = 'all.q'
# TODO: Use computing settings everywhere
MEM_PER_NODE = 249.5e3  # in MB
NCORES_PER_NODE = 20
NGPUS_PER_NODE = 2
NNODES_TOTAL = 17
NCORE_TOTAL = NNODES_TOTAL * NCORES_PER_NODE
NGPU_TOTAL = NNODES_TOTAL * NGPUS_PER_NODE

backend = "FS"  # File system
PYOPENGL_PLATFORM = 'egl'  # Rendering
DISABLE_LOCKING = False

# --------- LOGGING
# 'None' disables logging of SyConn modules (e.g. proc, handler, ...) to files.
# Logs of executed scripts (syconn/scripts) will be stored at the
# working directory + '/logs/' nonetheless.
default_log_dir = None
# TODO: remove all verbose kwargs and verbose log.info and execute log.debug() instead
log_level = logging.DEBUG  # INFO, DEBUG

# file logging for individual modules, and per job. Only use in case of
# debugging with single core processing. Logs for scripts are located in 'SyConn/scripts/'
# will be stored at wd + '/logs/'.
DISABLE_FILE_LOGGING = True

# --------- CELL ORGANELLE PARAMETERS
thresh_mi_bbd_mapping = 25e3

# --------- CONTACT SITE AND SYNAPSE PARAMETERS
# Synaptic junction bounding box diagonal threshold in nm; objects above will not be used during `syn_gen_via_cset`
thresh_sj_bbd_syngen = 25e3
thresh_syn_proba = 0.5  # RFC probability used for classifying whether syn or not
cs_gap_nm = 250
# mapping parameters in 'map_objects_to_synssv'; assignment of cellular organelles to syn_ssv
max_vx_dist_nm = 2000
max_rep_coord_dist_nm = 4000
sym_thresh = 0.225  # above will be assigned synaptic sign (-1, inhibitory) and <= will be (1, excitatory)


# --------- MESH PARAMETERS
existing_cell_organelles = ['mi', 'sj', 'vc']
MESH_DOWNSAMPLING = {"sv": (4, 4, 2), "sj": (2, 2, 1), "vc": (4, 4, 2),
                     "mi": (8, 8, 4), "cs": (2, 2, 1), "conn": (2, 2, 1),
                     'syn_ssv': (2, 2, 1)}
MESH_CLOSING = {"sv": 0, "sj": 0, "vc": 0, "mi": 0, "cs": 0,
                "conn": 4, 'syn_ssv': 20}
MESH_MIN_OBJ_VX = 100  # adapt to size threshold


# --------- VIEW PARAMETERS
NB_VIEWS = 2

# --------- GLIA PARAMETERS
# min. connected component size of glia nodes/SV after thresholding glia proba
min_cc_size_ssv = 8e3  # in nm; L1-norm on vertex bounding box

# Threshold for glia classification
glia_thresh = 0.161489
# number of sv used during local rendering. The total number of SV used are SUBCC_SIZE_BIG_SSV + 2*(SUBCC_CHUNKE_SIZE_BIG_SSV-1)
SUBCC_SIZE_BIG_SSV = 35
RENDERING_MAX_NB_SV = 5e3
# number of SV for which views are rendered in one pass
SUBCC_CHUNK_SIZE_BIG_SSV = 9


# --------- RFC PARAMETERS
SKEL_FEATURE_CONTEXT = {"axoness": 8000, "spiness": 1000}  # in nm


# --------- SPINE PARAMETERS
min_spine_cc_size = 10
min_edge_dist_spine_graph = 110
gt_path_spineseg = '/wholebrain/scratch/areaxfs3/ssv_spgt/spgt_semseg/'  # TODO: add to config


# --------- COMPARTMENT PARAMETERS
DIST_AXONESS_AVERAGING = 10000
gt_path_axonseg = '/wholebrain/scratch/areaxfs3/ssv_semsegaxoness/gt_h5_files_80nm/'  # TODO: add to config


# --------- CELLTYPE PARAMETERS
view_properties_large = dict(verbose=False, ws=(512, 512), nb_views_render=6,
                             comp_window=40960, nb_views_model=4)

# --------- MORPHOLOGY EMBEDDING
ndim_embedding = 10

# general config object
config = DynConfig()
