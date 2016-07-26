# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max-Planck-Institute for Medical Research, Heidelberg, Germany
# Authors: Sven Dorkenwald, Philipp Schubert, Joergen Kornfeld

import argparse
def parseargs():
    parser = argparse.ArgumentParser(
    usage="Evaluate </path/to_work_dir> [--gpus <int>, <int>]]")
    parser.add_argument("main_path", type=str)
    parser.add_argument("--gpus", nargs='+', type=int)
    return parser.parse_args()

commandline_args = parseargs()

from processing import initialization, objectextraction as oe, \
    predictor_cnn as pc
from knossos_utils import knossosdataset
from knossos_utils import chunky
from multi_proc import multi_proc_main as mpm
import syconn

import glob
import numpy as np
import os
import shutil

home_dir = os.environ['HOME'] + "/"
syconn_dir = syconn.__path__[0] + "/"

main_path = commandline_args.main_path
gpus = commandline_args.gpus

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
    print ".theanorc detected"


# -------------------------------------------------------------- CNN Predictions

if gpus[0] is None:
    batch_size1 = [40, 500, 500]
    batch_size2 = [40, 500, 500]
else:
    batch_size1 = [22, 270, 270]
    batch_size2 = [18, 220, 220]

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

raise()

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
                                 suffix="3")

oe.from_probabilities_to_objects(cset, "ARGUS",
                                 ["vc"],
                                 thresholds=[int(6*255/21.)],
                                 debug=False,
                                 suffix="5",
                                 membrane_kd_path=kd_bar.knossos_path)

oe.from_probabilities_to_objects(cset, "ARGUS",
                                 ["mi"],
                                 thresholds=[int(9*255/21.)],
                                 debug=False,
                                 suffix="8")

# ------------ Create hull and map objects to tracings and classify compartments

syconn.enrich_tracings_all(main_path)

# ----------------------------------------------------------- Classify cell type

syconn.predict_celltype_label(main_path)

# ---------------------------------- Classify contact sites as synaptic or touch

syconn.detect_synapses(main_path)

# --------------------------------------------------- Create connectivity matrix

syconn.type_sorted_wiring(main_path)

