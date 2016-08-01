# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max-Planck-Institute for Medical Research, Heidelberg, Germany
# Authors: Sven Dorkenwald, Philipp Schubert, Joergen Kornfeld

import argparse

from processing import initialization, objectextraction as oe
from knossos_utils import knossosdataset
from knossos_utils import chunky
from multi_proc import multi_proc_main as mpm
import syconn

import glob
import numpy as np
import os
import shutil


def parseargs():
    parser = argparse.ArgumentParser(
    usage="Evaluate </path/to_work_dir> [--gpus <int>, <int>]] "
          "[--CNNsize <int> (0-4)]")
    parser.add_argument("main_path", type=str)
    parser.add_argument("--gpus", nargs='+', type=int, default=None)
    parser.add_argument("--CNNsize", type=int, default=2)
    return parser.parse_args()

commandline_args = parseargs()

use_qsub = False
home_dir = os.environ['HOME'] + "/"
syconn_dir = syconn.__path__[0] + "/"

main_path = os.path.abspath(commandline_args.main_path)
gpus = commandline_args.gpus
CNN_size = commandline_args.CNNsize

if CNN_size > 4:
    print "CNNsize too big; set to 4"
    CNN_size = 4

if CNN_size < 0:
    print "CNNsize too small; set to 0"
    CNN_size = 0

if gpus is None:
    gpus = [None]

if not "/" == main_path[-1]:
    main_path += "/"

# ------------------------------------------------------------------------ Setup

assert os.path.exists(main_path + "/models/BIRD_MIGA_config.py")
assert os.path.exists(main_path + "/models/BIRD_MIGA.param")
assert os.path.exists(main_path + "/models/BIRD_ARGUS_config.py")
assert os.path.exists(main_path + "/models/BIRD_ARGUS.param")
assert os.path.exists(main_path + "/models/BIRD_barrier_config.py")
assert os.path.exists(main_path + "/models/BIRD_barrier.param")
assert os.path.exists(main_path + "/models/BIRD_rbarrier_config.py")
assert os.path.exists(main_path + "/models/BIRD_rbarrier.param")
assert os.path.exists(main_path + "/models/BIRD_TYPE_config.py")
assert os.path.exists(main_path + "/models/BIRD_TYPE.param")
assert os.path.exists(main_path + "/models/rf_synapses/rfc_syn.pkl")
assert os.path.exists(main_path + "/models/rf_axoness/rf.pkl")
assert os.path.exists(main_path + "/models/rf_spiness/rf.pkl")
assert os.path.exists(main_path + "/models/rf_celltypes/rf.pkl")
tracing_paths = syconn.get_filepaths_from_dir(main_path + "/tracings/")
assert len(tracing_paths) > 1
assert os.path.exists(main_path + "/knossosdatasets/raw/")

kd_raw = knossosdataset.KnossosDataset()
kd_raw.initialize_from_knossos_path(main_path + "/knossosdatasets/raw/")

if os.path.exists(main_path + "chunkdataset.chunk_dataset.pkl"):
    cset = chunky.load_dataset(main_path + "chunkdataset.chunk_dataset.pkl")
else:
    cset = initialization.initialize_cset(kd_raw, main_path, [500, 500, 250])


if not os.path.exists(home_dir + ".theanorc"):
    print "Creating .theanorc in your home"
    shutil.copy(syconn_dir + "/utils/default_theanorc",
                home_dir + "/.theanorc")
else:
    print ".theanorc detected, checking"
    device_error = False
    with open(home_dir + "/.theanorc", "r") as f:
        lines = f.readlines()
        for line in lines:
            if not "#" in line and "device" in line and "gpu" in line:
                if len(gpus) > 1:
                    device_error = True
                elif gpus[0] is None:
                    device_error = True
                elif line.strip().endswith("u") and not gpus[0] == 0:
                    device_error = True
                elif not line.strip().endswith("u"):
                    if not int(line.strip()[-1]) == gpus[0]:
                        device_error = True
            elif not "#" in line and "linker" in line and "cvm_nogc" in line:
                print "\nYou turned garbage collection off in your ~/.theanorc"\
                      " - try removing this if you encounter memory problems.\n"

    if device_error:
        raise Exception("Please change your device setting in ~/.theanorc"
                        "to cpu, because current settings do not allow the "
                        "allocation of your specified gpu(s)")

# -------------------------------------------------------------- CNN Predictions

if gpus[0] is None:
    batch_size1 = [40, 500, 500]
    batch_size2 = [40, 500, 500]
else:
    if CNN_size == 0:
        batch_size1 = [18, 220, 220]
        batch_size2 = [18, 160, 160]
    elif CNN_size == 1:
        batch_size1 = [22, 270, 270]
        batch_size2 = [18, 220, 220]
    elif CNN_size == 2:
        batch_size1 = [30, 340, 340]
        batch_size2 = [22, 270, 270]
    elif CNN_size == 3:
        batch_size1 = [36, 440, 440]
        batch_size2 = [30, 340, 340]
    elif CNN_size == 4:
        batch_size1 = [40, 500, 500]
        batch_size2 = [36, 440, 440]
    else:
        raise Exception("CNNsize not supported")

mutex_paths = glob.glob(cset.path_head_folder + "chunky_*/mutex_*")
for path in mutex_paths:
    os.removedirs(path)

offset = [120, 120, 30]

# Synaptic junctions, vesicle clouds, mitochondria - stage 1
params = []
for gpu in gpus:
    params.append([cset,
                   main_path + "/models/BIRD_MIGA_config.py",
                   main_path + "/models/BIRD_MIGA.param",
                   ["MIGA"], ["none", "mi", "vc", "sj"], offset,
                   batch_size1, kd_raw.knossos_path, gpu])

mpm.SUBP_script(params, "join_chunky_inference")

# Synaptic junctions, vesicle clouds, mitochondria - stage 2
params = []
for gpu in gpus:
    params.append([cset,
                   main_path + "/models/BIRD_ARGUS_config.py",
                   main_path + "/models/BIRD_ARGUS.param",
                   ["ARGUS", "MIGA"], ["none", "mi", "vc", "sj"],
                   offset, batch_size1, kd_raw.knossos_path, gpu])

mpm.SUBP_script(params, "join_chunky_inference")

# Type of synaptic junctions
params = []
for gpu in gpus:
    params.append([cset,
                   main_path + "/models/BIRD_TYPE_config.py",
                   main_path + "/models/BIRD_TYPE.param",
                   ["TYPE"], ["none", "asym", "sym"], offset,
                   batch_size1, kd_raw.knossos_path, gpu])

mpm.SUBP_script(params, "join_chunky_inference")

# Barrier - stage 1
params = []
for gpu in gpus:
    params.append([cset,
                   main_path + "/models/BIRD_barrier_config.py",
                   main_path + "/models/BIRD_barrier.param",
                   ["BARRIER"], ["none", "bar"], offset,
                   batch_size1, kd_raw.knossos_path, gpu])

mpm.SUBP_script(params, "join_chunky_inference")

# Barrier - stage 2
params = []
for gpu in gpus:
    params.append([cset,
                   main_path + "/models/BIRD_rbarrier_config.py",
                   main_path + "/models/BIRD_rbarrier.param",
                   ["RBARRIER", "BARRIER"], ["none", "bar"],
                   offset, batch_size2, kd_raw.knossos_path, gpu])

mpm.SUBP_script(params, "join_chunky_inference")

# ------------------------------------------------ Conversion to knossosdatasets

kd_bar = knossosdataset.KnossosDataset()
if os.path.exists(main_path + "knossosdatasets/rrbarrier/"):
    kd_bar.initialize_from_knossos_path(main_path + "/knossosdatasets/rrbarrier/")
else:
    bar = cset.from_chunky_to_matrix(kd_raw.boundary, [0, 0, 0], "RBARRIER",
                                     ["bar"], dtype=np.uint8,
                                     show_progress=True)["bar"]
    kd_bar.initialize_from_matrix(main_path + "knossosdatasets/rrbarrier/",
                                  scale=[9, 9, 20],
                                  experiment_name="j0126_dense",
                                  data=bar,
                                  mags=[1, 2, 4, 8])
    bar = None

kd_asym = knossosdataset.KnossosDataset()
if os.path.exists(main_path + "knossosdatasets/asymmetric/"):
    kd_asym.initialize_from_knossos_path(main_path + "/knossosdatasets/asymmetric/")
kd_sym = knossosdataset.KnossosDataset()
if os.path.exists(main_path + "knossosdatasets/symmetric/"):
    kd_sym.initialize_from_knossos_path(main_path + "/knossosdatasets/symmetric/")

if not kd_asym.initialized or not kd_sym.initialized:
    types = cset.from_chunky_to_matrix(kd_raw.boundary, [0, 0, 0], "TYPE",
                                       ["asym", "sym"], dtype=np.uint8,
                                       show_progress=True)

    if not kd_asym.initialized:
        kd_asym.initialize_from_matrix(main_path + "knossosdatasets/asymmetric/",
                                       scale=[9, 9, 20],
                                       experiment_name="j0126_dense",
                                       data=types["asym"],
                                       mags=[1, 2, 4, 8])

    if not kd_sym.initialized:
        kd_sym.initialize_from_matrix(main_path + "knossosdatasets/symmetric/",
                                      scale=[9, 9, 20],
                                      experiment_name="j0126_dense",
                                      data=types["sym"],
                                      mags=[1, 2, 4, 8])

        types = None

# ------------------------------------------------------------ Object Extraction

oe.from_probabilities_to_objects(cset, "ARGUS",
                                 ["sj"],
                                 thresholds=[int(4*255/21.)],
                                 debug=False,
                                 suffix="3",
                                 use_qsub=use_qsub)

oe.from_probabilities_to_objects(cset, "ARGUS",
                                 ["vc"],
                                 thresholds=[int(6*255/21.)],
                                 debug=False,
                                 suffix="5",
                                 membrane_kd_path=kd_bar.knossos_path,
                                 use_qsub=use_qsub)

oe.from_probabilities_to_objects(cset, "ARGUS",
                                 ["mi"],
                                 thresholds=[int(9*255/21.)],
                                 debug=False,
                                 suffix="8",
                                 use_qsub=use_qsub)

# ------------ Create hull and map objects to tracings and classify compartments

syconn.enrich_tracings_all(main_path, use_qsub=use_qsub)

# ---------------------------------- Classify contact sites as synaptic or touch

syconn.detect_synapses(main_path, use_qsub=use_qsub)

# --------------------------------------------------- Create connectivity matrix

syconn.type_sorted_wiring(main_path)

