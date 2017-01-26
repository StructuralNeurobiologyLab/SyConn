# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max-Planck-Institute for Medical Research, Heidelberg, Germany
# Authors: Sven Dorkenwald, Philipp Schubert, Joergen Kornfeld

# This script assumes a dense reconstruction in form of Knossos overlaycubes
# and extracted objects gathered as demonstrated in <full_run_example.py>

import argparse

from syconn.utils import densedataset
from syconn.processing import initialization, objectextraction as oe
from knossos_utils import knossosdataset
from knossos_utils import chunky
from syconn.multi_proc import multi_proc_main as mpm
import syconn

import glob
import numpy as np
import os
import shutil


def parseargs():
    parser = argparse.ArgumentParser(
    usage="Evaluate </path/to_work_dir>"
          "[--qsub_pe <str>] [--qsub_queue <str>]")
    parser.add_argument("main_path", type=str)
    parser.add_argument("--qsub_pe", type=str, default=None)
    parser.add_argument("--qsub_queue", type=str, default=None)
    return parser.parse_args()

commandline_args = parseargs()

home_dir = os.environ['HOME'] + "/"
syconn_dir = syconn.__path__[0] + "/"

main_path = os.path.abspath(commandline_args.main_path)
qsub_pe = commandline_args.qsub_pe
qsub_queue = commandline_args.qsub_queue

knossos_raw_path = "/run/media/sdorkenw/K338/j0126_realigned_v4b_min1k_cbs_ext0/"
# knossos_raw_path = main_path + "/knossosdatasets/raw/"

if not "/" == main_path[-1]:
    main_path += "/"

# ------------------------------------------------------------------------ Setup


# assert os.path.exists(main_path + "/models/rf_synapses/rfc_syn.pkl")
# assert os.path.exists(main_path + "/models/rf_axoness/rf.pkl")
# assert os.path.exists(main_path + "/models/rf_spiness/rf.pkl")
# assert os.path.exists(main_path + "/models/rf_celltypes/rf.pkl")
# assert os.path.exists(main_path + "/knossosdatasets/raw/")
# assert os.path.exists(main_path + "/knossosdatasets/symmetric/")
# assert os.path.exists(main_path + "/knossosdatasets/asymmetric/")
# assert os.path.exists()

kd_raw = knossosdataset.KnossosDataset()
kd_raw.initialize_from_knossos_path(knossos_raw_path)

if os.path.exists(main_path + "chunkdataset.chunk_dataset.pkl"):
    cset = chunky.load_dataset(main_path + "chunkdataset.chunk_dataset.pkl")
else:
    cset = initialization.initialize_cset(kd_raw, main_path, [512, 512, 256])


# ------------------------------------------------------------ SuperVoxel Extraction

# Write segmentation to chunky first
densedataset.export_dense_segmentation_to_cset(cset, kd_raw, nb_cpus=4)

oe.from_ids_to_objects(cset, "dense_segmentation", ["sv"],
                       debug=False, qsub_pe=qsub_pe, qsub_queue=qsub_queue)

# oe.from_probabilities_to_objects(cset, "ARGUS",
#                                  ["vc"],
#                                  thresholds=[int(6*255/21.)],
#                                  debug=False,
#                                  suffix="5",
#                                  membrane_kd_path=kd_bar.knossos_path,
#                                  qsub_pe=qsub_pe,
#                                  qsub_queue=qsub_queue)
#
# oe.from_probabilities_to_objects(cset, "ARGUS",
#                                  ["mi"],
#                                  thresholds=[int(9*255/21.)],
#                                  debug=False,
#                                  suffix="8",
#                                  qsub_pe=qsub_pe,
#                                  qsub_queue=qsub_queue)
#
# # ------------ Create hull and map objects to tracings and classify compartments
#
# syconn.enrich_tracings_all(main_path, qsub_pe=qsub_pe, qsub_queue=qsub_queue)
#
# # ---------------------------------- Classify contact sites as synaptic or touch
#
# syconn.detect_synapses(main_path, qsub_pe=qsub_pe, qsub_queue=qsub_queue)
#
# # --------------------------------------------------- Create connectivity matrix
#
# syconn.type_sorted_wiring(main_path)
#
