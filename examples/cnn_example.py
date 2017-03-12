# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max-Planck-Institute for Medical Research, Heidelberg, Germany
# Authors: Sven Dorkenwald, Philipp Schubert, Joergen Kornfeld

from syconn.processing import initialization, objectextraction
from knossos_utils import knossosdataset
from knossos_utils import chunky
from syconn.multi_proc import multi_proc_main as mpm
import syconn

import glob
import os
import shutil

home_dir = os.environ['HOME'] + "/"
syconn_dir = syconn.__path__[0] + "/"

path_ultrustructure_cnn_1 = [home_dir + "/syconn_paper_models/BIRD_MIGA_config.py",
                             home_dir + "/syconn_paper_models/BIRD_MIGA.param"]
path_ultrustructure_cnn_2 = [home_dir + "/syconn_paper_models/BIRD_ARGUS_config.py",
                             home_dir + "/syconn_paper_models/BIRD_ARGUS.param"]
path_syn_type_cnn = [home_dir + "/syconn_paper_models/BIRD_TYPE_config.py",
                     home_dir + "/syconn_paper_models/BIRD_TYPE.param"]
path_barrier_cnn_1 = [home_dir + "/syconn_paper_models/BIRD_barrier_config.py",
                      home_dir + "/syconn_paper_models/BIRD_barrier.param"]
path_barrier_cnn_2 = [home_dir + "/syconn_paper_models/BIRD_rbarrier_config.py",
                      home_dir + "/syconn_paper_models/BIRD_rbarrier.param"]

path_to_knossosdataset = "/lustre/sdorkenw/j0126_cubed_realligned/"
path_to_chunkdataset = "/lustre/sdorkenw/j0126_dense_syconn_v2/chunkdataset_u/"

n_jobs = 100

qsub_pe_gpu = "synapse0102"  # gpu pe
qsub_pe_cpu = "openmp"  # gpu pe


# ------------------------------------------------------------------------ Setup

if not os.path.exists(path_to_chunkdataset):
    os.makedirs(path_to_chunkdataset)

kd_raw = knossosdataset.KnossosDataset()
kd_raw.initialize_from_knossos_path(path_to_knossosdataset)

if os.path.exists(path_to_chunkdataset + "/chunk_dataset.pkl"):
    cset = chunky.load_dataset(path_to_chunkdataset)
else:
    cset = initialization.initialize_cset(kd_raw, path_to_chunkdataset,
                                          [1850, 1850, 120])


if not os.path.exists(home_dir + ".theanorc"):
    print "Creating .theanorc in your home"
    shutil.copy(syconn_dir + "/utils/default_theanorc",
                home_dir + "/.theanorc")

# -------------------------------------------------------------- CNN Predictions

# batch_size1 = [22, 270, 270]
# batch_size2 = [18, 220, 220]
batch_size1 = [30, 340, 340]
batch_size2 = [22, 270, 270]
# batch_size1 = [36, 440, 440]
# batch_size2 = [30, 340, 340]

mutex_paths = glob.glob(cset.path_head_folder + "chunky_*/mutex_*")
for path in mutex_paths:
    os.removedirs(path)

offset = [120, 120, 30]

# Synaptic junctions, vesicle clouds, mitochondria - stage 1 -------------------
params = []
for _ in range(n_jobs):
    params.append([cset,
                   path_ultrustructure_cnn_1[0],
                   path_ultrustructure_cnn_1[1],
                   ["MIGA"], ["none", "mi", "vc", "sj"], offset,
                   batch_size1, kd_raw.knossos_path])

# mpm.SUBP_script(params, "join_chunky_inference")
mpm.QSUB_script(params, "join_chunky_inference", pe=qsub_pe_gpu, delay=10,
                delay_one=600)

objectextraction.validate_chunks(cset, "MIGA", ["mi", "vc", "sj"],
                                 qsub_pe=qsub_pe_cpu)
# Synaptic junctions, vesicle clouds, mitochondria - stage 2 -------------------
params = []
for i_job in range(n_jobs):
    params.append([cset,
                   path_ultrustructure_cnn_2[0],
                   path_ultrustructure_cnn_2[1],
                   ["ARGUS", "MIGA"], ["none", "mi", "vc", "sj"], offset,
                   batch_size1, kd_raw.knossos_path])

# mpm.SUBP_script(params, "join_chunky_inference")
mpm.QSUB_script(params, "join_chunky_inference", pe=qsub_pe_gpu, delay=10,
                delay_one=600)

objectextraction.validate_chunks(cset, "ARGUS", ["mi", "vc", "sj"],
                                 qsub_pe=qsub_pe_cpu)

# Type of synaptic junctions ---------------------------------------------------
params = []
for i_job in range(n_jobs):
    params.append([cset,
                   path_syn_type_cnn[0],
                   path_syn_type_cnn[1],
                   ["TYPE"], ["none", "asym", "sym"], offset,
                   batch_size1, kd_raw.knossos_path])

# mpm.SUBP_script(params, "join_chunky_inference")
mpm.QSUB_script(params, "join_chunky_inference", pe=qsub_pe_gpu, delay=10,
                delay_one=600)

objectextraction.validate_chunks(cset, "TYPE", ["asym", "sym"],
                                 qsub_pe=qsub_pe_cpu)
# Barrier - stage 1 ------------------------------------------------------------
params = []
for i_job in range(n_jobs):
    params.append([cset,
                   path_barrier_cnn_1[0],
                   path_barrier_cnn_1[1],
                   ["BARRIER"], ["none", "bar"], offset,
                   batch_size2, kd_raw.knossos_path])

# mpm.SUBP_script(params, "join_chunky_inference")
mpm.QSUB_script(params, "join_chunky_inference", pe=qsub_pe_gpu, delay=10,
                delay_one=600)

objectextraction.validate_chunks(cset, "BARRIER", ["bar"],
                                 qsub_pe=qsub_pe_cpu)
# Barrier - stage 2 ------------------------------------------------------------
params = []
for i_job in range(n_jobs):
    params.append([cset,
                   path_barrier_cnn_2[0],
                   path_barrier_cnn_2[1],
                   ["RBARRIER", "BARRIER"], ["none", "bar"], offset,
                   batch_size2, kd_raw.knossos_path])

# mpm.SUBP_script(params, "join_chunky_inference")
mpm.QSUB_script(params, "join_chunky_inference", pe=qsub_pe_gpu, delay=10,
                delay_one=600)

objectextraction.validate_chunks(cset, "RBARRIER", ["bar"],
                                 qsub_pe=qsub_pe_cpu)

