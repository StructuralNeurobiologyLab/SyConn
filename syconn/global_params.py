# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Sven Dorkenwald, Philipp Schubert, Joergen Kornfeld
import os
from .handler.config import Config
import sys
# ---------------------- STATIC AND GLOBAL PARAMETERS # -----------------------
# --------- GLOBAL WORKING DIRECTORY
# wd = "/wholebrain/scratch/areaxfs3/"
wd = "/wholebrain/songbird/j0126/areaxfs_v6/"
# wd = '/mnt/j0126/areaxfs_v10/'

# TODO: Put all dataset-specific parameters into config.ini / configspec.ini


# --------- Required data paths
# TODO: add generic parser method for initial RAG and handle case without glia-splitting, refactor RAG path handling
# TODO:(cover case if glia removal was not performed, change resulting rag paths after glia removal from 'glia' to 'rag'
class _PathHandler(Config):
    """
    Enables dynamic and SyConn-wide update of working directory 'wd'.
    """
    def __init__(self):
        super().__init__(wd)

    def _check_actuality(self):
        """
        Crucial check, which triggers the update everytime wd is not the same as
         self.working dir
        """
        if super().working_dir != wd:
            super().__init__(wd)

    @property
    def entries(self):
        self._check_actuality()
        return super().entries

    @property
    def working_dir(self):
        self._check_actuality()
        return super().working_dir

    @property
    def kd_seg_path(self):
        return self.entries['Paths']['kd_seg']

    @property
    def kd_sym_path(self):
        return self.entries['Paths']['kd_sym']

    @property
    def kd_asym_path(self):
        return self.entries['Paths']['kd_asym']

    @property
    def kd_sj_path(self):
        return self.entries['Paths']['kd_sj']

    @property
    def kd_vc_path(self):
        return self.entries['Paths']['kd_vc']

    @property
    def kd_mi_path(self):
        return self.entries['Paths']['kd_mi']

    @property
    # TODO: make this more elegant, e.g. bash script with 'source activate py36'
    def py36path(self):
        return self.entries['Paths']['py36path']

    # TODO: Work-in usage of init_rag_path
    @property
    def init_rag_path(self):
        """
        # currently a mergelist/RAG of the following form is expected:
        # ID, ID
        #    .
        #    .
        # ID, ID

        Returns
        -------
        str
        """
        # self._check_actuality()
        return self.entries['Paths']['init_rag']

    # TODO: make model names more generic, this is not adjustable in config, see also TODOs above
    # --------- CLASSIFICATION MODELS
    @property
    def model_dir(self):
        return self.working_dir + '/models/'

    @property
    def mpath_tnet(self):
        return self.working_dir + '/TN-10-Neighbors/'

    @property
    def mpath_spiness(self):
        return self.working_dir + '/FCN-VGG13--Lovasz--NewGT/'

    @property
    def mpath_celltype(self):
        return self.working_dir + '/celltype_g1_20views_v2/celltype_g1_20views_v2-LAST.mdl'

    @property
    def mpath_axoness(self):
        return self.working_dir + '/axoness_g1_v2/g1_v2-FINAL.mdl'

    @property
    def mpath_glia(self):
        return self.working_dir + '/glia_g0_v0/g0_v0-FINAL.mdl'

    @property
    def mpath_syn_rfc(self):
        return self.working_dir + '/conn_syn_rfc//rfc'

    @property
    def allow_mesh_gen_cells(self):
        return self.entries['Mesh']['allow_mesh_gen_cells']

    @property
    def allow_skel_gen(self):
        return self.entries['Skeleton']['allow_skel_gen']


rag_suffix = ""  # identifier in case there will be more than one RAG, TODO: Remove
paths = _PathHandler()  # TODO: rename paths to config or similar

# All subsquent parameter are dataset independent and do not have to be stored at
# config.ini in the working directory

# --------- BACKEND DEFINITIONS
BATCH_PROC_SYSTEM = 'SLURM'  # If None, fall-back is single node multiprocessing
batchjob_script_folder = os.path.dirname(os.path.abspath(__file__)) + \
                         "/QSUB_scripts/"
MEM_PER_NODE = 249.5e3  # in MB
NCORES_PER_NODE = 20
# TOOD: Use NCORE_TOTAL everywhere
NCORE_TOTAL = 340
# TODO: Use NGPU_TOTAL everywhere
NGPU_TOTAL = 34


# --------- LOGGING
# 'None' disables logging of SyConn modules (e.g. proc, handler, ...) to files.
# Logs of executed scripts (syconn/scripts) will be stored at the
# working directory + '/logs/' nonetheless.
default_log_dir = None
# TODO: remove all verbose kwargs and verbose log.info and execute log.debug() instead
log_level = 'DEBUG'  # INFO, DEBUG

# file logging for individual modules, and per job. Only use in case of
# debugging with single core processing. Logs for scripts are located in 'SyConn/scripts/'
# will be stored at wd + '/logs/'.
DISABLE_FILE_LOGGING = True


# --------- BACKEND DEFINITIONS
backend = "FS"  # File system
PYOPENGL_PLATFORM = 'egl'  # Rendering

# --------- CONTACT SITE PARAMETERS
# Synaptic junction bounding box diagonal threshold in nm; objects above will not be used during `syn_gen_via_cset`
thresh_sj_bbd_syngen = 25e3
cs_gap_nm = 250
# mapping parameters in 'map_objects_to_synssv'; assignment of cellular organelles to syn_ssv
max_vx_dist_nm = 2000
max_rep_coord_dist_nm = 4000
sym_thresh = 0.225  # above will be assigned synaptic sign (-1, inhibitory) and <= will be (1, excitatory)


# --------- MESH PARAMETERS
existing_cell_organelles = ['mi', 'sj', 'vc']
MESH_DOWNSAMPLING = {"sv": (8, 8, 4), "sj": (2, 2, 1), "vc": (4, 4, 2),
                     "mi": (8, 8, 4), "cs": (2, 2, 1), "conn": (2, 2, 1),
                     'syn_ssv': (2, 2, 1)}
MESH_CLOSING = {"sv": 0, "sj": 0, "vc": 0, "mi": 0, "cs": 0,
                "conn": 4, 'syn_ssv': 20}
MESH_MIN_OBJ_VX = 100  # adapt to size threshold


# --------- VIEW PARAMETERS
NB_VIEWS = 2

# --------- GLIA PARAMETERS
# CC size threshold needs to be the same  # TODO: remove coexistence
# min. connected component size of glia nodes/SV after thresholding glia proba
min_cc_size_glia = 8e3  # in nm; L1-norm on vertex bounding box
# min. connected component size of neuron nodes/SV after thresholding glia proba
min_cc_size_neuron = min_cc_size_glia  # in nm; L1-norm on vertex bounding box

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


# --------- COMPARTMENT PARAMETERS
DIST_AXONESS_AVERAGING = 10000


# --------- CELLTYPE PARAMETERS

def get_dataset_scaling():
    """
    Helper method to get dataset scaling.

    Returns
    -------
    tuple of float
        (X, Y, Z)
    """
    from syconn.handler.config import Config
    import numpy as np
    cfg = Config(wd)
    return np.array(cfg.entries["Dataset"]["scaling"])
