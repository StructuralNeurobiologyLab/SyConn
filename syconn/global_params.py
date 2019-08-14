# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Philipp Schubert, Joergen Kornfeld

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
batchjob_script_folder = os.path.dirname(os.path.abspath(__file__)) + "/batchjob_scripts/"

# TODO refactor syconn and get rid of all qsub_pe and qsub_queue kwargs and only
#  use batch_job_enabled(), the default in QSUB_script should then be
#  BATCH_PE and BATCH_QUEUE
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
PYOPENGL_PLATFORM = 'egl'  # Rendering: 'egl' or 'osmesa'
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
# Synaptic junction bounding box diagonal threshold in nm; objects above will
# not be used during `syn_gen_via_cset`
thresh_sj_bbd_syngen = 25e3
thresh_syn_proba = 0.5  # RFC probability used for classifying whether syn or not
thresh_syn_size = 10  # minimum number of voxel for synapses in SSVs  # TODO: tweak, increase
# used for agglomerating 'syn' objects (cell supervoxel-based synapse fragments)
# into 'syn_ssv'
cs_gap_nm = 250
CS_FILTERSIZE = [13, 13, 7]
CS_NCLOSING = max(CS_FILTERSIZE)
# mapping parameters in 'map_objects_to_synssv'; assignment of cellular
# organelles to syn_ssv
max_vx_dist_nm = 2000
max_rep_coord_dist_nm = 4000
# above will be assigned synaptic sign (-1, inhibitory) and <= will be
# (1, excitatory)
sym_thresh = 0.225

# --------- VIEW PARAMETERS
# TODO: move all default view parameters here
NB_VIEWS = 2

# --------- GLIA PARAMETERS
# min. connected component size of glia nodes/SV after thresholding glia proba
min_cc_size_ssv = 8e3  # in nm; L1-norm on vertex bounding box

# Threshold for glia classification
glia_thresh = 0.161489
# number of sv used during local rendering. The total number of SV used are
# SUBCC_SIZE_BIG_SSV + 2*(SUBCC_CHUNKE_SIZE_BIG_SSV-1)
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
DIST_AXONESS_AVERAGING = 10000  # also used for myelin averaging
gt_path_axonseg = '/wholebrain/scratch/areaxfs3/ssv_semsegaxoness' \
                  '/all_bouton_data/'  # TODO: add to config

# `k=0` will not map predictions to unpredicted vertices -> faster
# `k` is the parameter used in `semseg2mesh`
view_properties_semsegax = dict(verbose=False, ws=(1024, 512), nb_views=3,
                                comp_window=40.96e3 * 1., semseg_key='axoness',
                                k=0)
# mapping of vertex labels to skeleton nodes; ignore labels 5 (background)
# and unpredicted (6), use labels of the k-nearest vertices
map_properties_semsegax = dict(k=50, ds_vertices=1, ignore_labels=[5, 6])

# TODO: add view properties for spine prediction
# mapping parameters of the semantic segmentation prediction to the cell mesh
# Note: ``k>0`` means that the predictions are propagated to unpredicted and backround labels
# via nearest neighbors.
semseg2mesh_spines = dict(semseg_key="spiness", force_recompute=True, k=0)
# no ignore labels used for the spine predictions because the predictions were propagated to
# unpredicted and background vertices via `semseg2mesh`
semseg2coords_spines = dict(k=50, ds_vertices=1, ignore_labels=[4, 5])

# --------- CELLTYPE PARAMETERS
view_properties_large = dict(verbose=False, ws=(512, 512), nb_views_render=6,
                             comp_window=40960, nb_views_model=4)

# --------- MORPHOLOGY EMBEDDING
ndim_embedding = 10

# general config object
config = DynConfig()

# --------- MESH PARAMETERS
existing_cell_organelles = ['mi', 'sj', 'vc']
MESH_MIN_OBJ_VX = 100  # adapt to size threshold

meshing_props = dict(normals=True, simplification_factor=300,
                     max_simplification_error=40)
