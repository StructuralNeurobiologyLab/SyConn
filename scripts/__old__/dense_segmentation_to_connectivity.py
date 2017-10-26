# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max-Planck-Institute for Medical Research, Heidelberg, Germany
# Authors: Sven Dorkenwald, Philipp Schubert, Joergen Kornfeld

# This script assumes a dense reconstruction in form of Knossos overlaycubes
# and extracted objects gathered as demonstrated in <full_run_example.py>

import argparse

from syconn.utils import densedataset_helper as ddh
# from syconn.processing import initialization, objectextraction as oe, contact_sites as cs
from knossos_utils import knossosdataset
from knossos_utils import chunky
from syconn.multi_proc import multi_proc_main as mpm
# import syconn
from syconn.extraction import object_extraction_wrapper as oew

import glob
import numpy as np
import os
import shutil


def parseargs():
    parser = argparse.ArgumentParser(usage="</path/to_work_dir>"
                                           "[--qsub_pe <str>] "
                                           "[--qsub_queue <str>]")
    parser.add_argument("main_path", type=str)
    parser.add_argument("--qsub_pe", type=str, default=None)
    parser.add_argument("--qsub_queue", type=str, default=None)
    return parser.parse_args()

commandline_args = parseargs()

home_dir = os.environ['HOME'] + "/"
# syconn_dir = syconn.__path__[0] + "/"

main_path = os.path.abspath(commandline_args.main_path)
qsub_pe = commandline_args.qsub_pe
qsub_queue = commandline_args.qsub_queue

# knossos_raw_path = main_path + "/j0126_realigned_v4b_cbs_ext0_fix.conf"
knossos_raw_path = main_path + "/knossosdatasets/j0126_realigned_v4b_cbs_ext0_fix/"

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

# if os.path.exists(main_path + "/chunkdataset_sv_v1/chunk_dataset.pkl"):
#     cset_sv = chunky.load_dataset(main_path + "/chunkdataset_sv_v1/",
#                                   update_paths=True)
#     chunky.save_dataset(cset_sv)
# else:
# cset_sv = initialization.initialize_cset(kd_raw, main_path + "/chunkdataset_sv_v2/",
#                                          [1024, 1024, 256])
# chunky.save_dataset(cset_sv)

if os.path.exists(main_path + "/chunkdataset_u_v2/chunk_dataset.pkl"):
    cset_u = chunky.load_dataset(main_path + "/chunkdataset_u_v2/",
                                 update_paths=True)
    chunky.save_dataset(cset_u)
# else:
# cset_u = initialization.initialize_cset(kd_raw, main_path + "/chunkdataset_u_v2/",
#                                         [1850, 1850, 120])
# chunky.save_dataset(cset_u)


# if os.path.exists(main_path + "/chunkdataset_cs_v2/chunk_dataset.pkl"):
#     cset_cs = chunky.load_dataset(main_path + "/chunkdataset_cs_v2/",
#                                   update_paths=True)
#     chunky.save_dataset(cset_cs)
# else:
#     cset_cs = initialization.initialize_cset(kd_raw, main_path + "/chunkdataset_cs_v2/",
#                                              [512, 512, 256])
#     chunky.save_dataset(cset_cs)


# -------------------------------------------------------- Supervoxel Extraction

# if qsub_queue or qsub_pe:
#     import_batch_size = int(len(cset_sv.chunk_dict.keys()) / 1000)
# else:
#     import_batch_size = None

# # Write segmentation to chunky first
# ddh.export_dense_segmentation_to_cset(cset_sv, kd_raw, datatype=np.uint32,
#                                       nb_cpus=10, pe=qsub_pe,
#                                       queue=qsub_queue,
#                                       batch_size=import_batch_size)
#
# oe.validate_chunks(cset_sv, "dense_segmentation", ["sv"], qsub_pe=qsub_pe,
#                    qsub_queue=qsub_queue)
#
# oe.extract_ids(cset_sv, "dense_segmentation", ["sv"], qsub_pe=qsub_pe,
#                qsub_queue=qsub_queue)

# Extract supervoxels as objects
# oe.from_ids_to_objects(cset_sv, "dense_segmentation", ["sv"],
#                        overlaydataset_path=knossos_raw_path, qsub_pe=qsub_pe,
#                        qsub_queue=qsub_queue, n_max_processes=240)

# ------------------------------------------------------- Contact Site detection

# cs.find_contact_sites(cset_cs, knossos_raw_path, "cs_sv", qsub_pe=qsub_pe,
#                       qsub_queue=qsub_queue, n_max_co_processes=100)


# cs.extract_contact_sites(cset_sv, "cs_sv", main_path,
#                          n_max_co_processes=100, qsub_pe=qsub_pe,
#                          qsub_queue=qsub_queue)


# oe.from_ids_to_objects(cset_cs, "cs_sv", ["cs_agg"], qsub_pe=qsub_pe,
#                        qsub_queue=qsub_queue, n_max_processes=150)

# ------------------------------------------------------- Export knossosdatasets

# if not os.path.exists(main_path + "/knossosdatasets/"):
#     os.makedirs(main_path + "/knossosdatasets/")
#
# kd_bar = knossosdataset.KnossosDataset()
# if os.path.exists(main_path + "knossosdatasets/rbarrier/"):
#     kd_bar.initialize_from_knossos_path(main_path + "/knossosdatasets/rbarrier/")
# else:
#     kd_bar.initialize_without_conf(main_path + "knossosdatasets/rbarrier/",
#                                    boundary=kd_raw.boundary,
#                                    scale=kd_raw.scale,
#                                    experiment_name="j0126_rbarrier",
#                                    mags=[1, 2, 4, 8])
#     cset_u.export_cset_to_kd(kd_bar, "RBARRIER", ["bar"], nb_threads=[8, 1],
#                              coordinate=None, size=None,
#                              stride=[4 * 128, 4 * 128, 4 * 128],
#                              as_raw=True,
#                              unified_labels=False)

# kd_asym = knossosdataset.KnossosDataset()
# if os.path.exists(main_path + "knossosdatasets/asymmetric/"):
#     kd_asym.initialize_from_knossos_path(main_path +
#                                          "/knossosdatasets/asymmetric/")
# else:
#     kd_asym.initialize_without_conf(main_path + "knossosdatasets/asymmetric/",
#                                     boundary=kd_raw.boundary,
#                                     scale=kd_raw.scale,
#                                     experiment_name="j0126_asymmetric",
#                                     mags=[1, 2, 4, 8])
#     cset_u.export_cset_to_kd(kd_asym, "TYPE", ["asym"], nb_threads=[1, 1],
#                              coordinate=None, size=None,
#                              stride=[4 * 128, 4 * 128, 4 * 128],
#                              as_raw=True,
#                              unified_labels=False)
#
# kd_sym = knossosdataset.KnossosDataset()
# if os.path.exists(main_path + "knossosdatasets/symmetric/"):
#     kd_sym.initialize_from_knossos_path(main_path +
#                                         "/knossosdatasets/symmetric/")
# else:
#     kd_sym.initialize_without_conf(main_path + "knossosdatasets/symmetric/",
#                                    boundary=kd_raw.boundary,
#                                    scale=kd_raw.scale,
#                                    experiment_name="j0126_symmetric",
#                                    mags=[1, 2, 4, 8])
#     cset_u.export_cset_to_kd(kd_sym, "TYPE", ["sym"], nb_threads=[1, 1],
#                              coordinate=None, size=None,
#                              stride=[4 * 128, 4 * 128, 4 * 128],
#                              as_raw=True,
#                              unified_labels=False)

# -------------------------------------------- Ultrastructural object extraction

# print "Extracting SJ"
#
# oe.from_ids_to_objects(cset_u, "ARGUS_stitched_components3", ["sj"],
#                        qsub_pe=qsub_pe, qsub_queue=qsub_queue,
#                        n_max_processes=240)


# print "Extracting MI"
#
# oe.from_ids_to_objects(cset_u, "ARGUS_stitched_components8", ["mi"],
#                        qsub_pe=qsub_pe, qsub_queue=qsub_queue,
#                        n_max_processes=200)


# print "Extracting VC"
#
# oe.from_ids_to_objects(cset_u, "ARGUS_stitched_components5", ["vc"],
#                        qsub_pe=qsub_pe, qsub_queue=qsub_queue,
#                        n_max_processes=200)

oew.from_probabilities_to_objects(cset_u, "ARGUS",
                                  ["sj"],
                                  thresholds=[int(4*255/21.)],
                                  debug=False,
                                  suffix="rf_3",
                                  qsub_pe=qsub_pe,
                                  qsub_queue=qsub_queue,
                                  n_max_processes=100,
                                  n_folders_fs=10000)


# oe.from_probabilities_to_objects(cset_u, "ARGUS",
#                                  ["mi"],
#                                  thresholds=[int(9*255/21.)],
#                                  debug=False,
#                                  suffix="8",
#                                  qsub_pe=qsub_pe,
#                                  qsub_queue=qsub_queue)

# oe.from_probabilities_to_objects(cset_u, "ARGUS",
#                                  ["vc"],
#                                  thresholds=[int(6*255/21.)],
#                                  debug=False,
#                                  suffix="5",
#                                  membrane_filename="RBARRIER",
#                                  # hdf5_name_membrane="bar",
#                                  qsub_pe=qsub_pe,
#                                  qsub_queue=qsub_queue)

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
