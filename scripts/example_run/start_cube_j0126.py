# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max-Planck-Institute of Neurobiology, Munich, Germany
# Authors: Philipp Schubert, Joergen Kornfeld

from knossos_utils import knossosdataset
knossosdataset._set_noprint(True)
import numpy as np
import os
import subprocess
import glob
import shutil
import re
import sys
import time
import argparse
import networkx as nx

from syconn.handler.prediction import parse_movement_area_from_zip
from syconn.handler.config import generate_default_conf, initialize_logging
from syconn.handler.compression import load_from_h5py
from syconn import global_params
from syconn.exec import exec_init, exec_syns, exec_multiview, exec_dense_prediction, exec_skeleton


# TODO add materialize button and store current process in config.ini
#  -> allows to resume interrupted processes
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SyConn example run')
    parser.add_argument('--working_dir', type=str, default='',
                        help='Working directory of SyConn')
    args = parser.parse_args()

    # ----------------- DEFAULT WORKING DIRECTORY ---------------------
    # TODO: if this is changed, ALSO ADAPT THE RAG SOURCE FILE SOME LINES BELOW
    # working_dir = "/wholebrain/songbird/j0126/areaxfs_v10_v4b_base_20180214_full_agglo_cbsplit/"
    working_dir = "/wholebrain/scratch/mariakaw/example_cube_start_cube_j0126/"

    # copy rag to its locations inside the WD
    # ----------------- DEFAULT WORKING DIRECTORY ---------------------

    # ----------------- DATA DIRECTORY ---------------------
    orig_data_dir = '/wholebrain/songbird/j0126/areaxfs_v5/'
    raw_kd_path = orig_data_dir + '/knossosdatasets/'
    sj_kd_path = "/wholebrain/scratch/areaxfs/knossosdatasets/j0126_3d_az_realigned/"
    vc_kd_path = "/wholebrain/scratch/areaxfs/knossosdatasets/j0126_3d_p4_realigned/"
    mi_kd_path = "/wholebrain/scratch/areaxfs/knossosdatasets/j0126_3d_mito_realigned/"
    kd_asym_path = "/wholebrain/scratch/areaxfs/knossosdatasets/j0126_3d_asymmetric_realigned/"
    kd_sym_path = "/wholebrain/scratch/areaxfs/knossosdatasets/j0126_3d_symmetric_realigned/"
    syntype_avail = (kd_asym_path is not None) and (kd_sym_path is not None)
    # ----------------- DATA DIRECTORY ---------------------

    # More parameters
    prior_glia_removal = True
    use_new_meshing = True
    scale = np.array([10, 10, 20])
    chunk_size = (512, 512, 512)
    # bb = [np.array((500, 500, 500)), np.array((2500, 2500, 2500))]
    bb = None

    n_folders_fs = 10000
    n_folders_fs_sc = 10000
    experiment_name = 'j0126'

    # The transform functions will be applied when loading the segmentation data of cell organelles
    # in order to convert them into binary fore- and background
    # currently using `dill` package to support lambda expressions, a weak feature. Make
    #  sure all dependencies within the lambda expressions are imported in
    #  `QSUB_gauss_threshold_connected_components.py` (here: numpy)

    # Preparing data
    # --------------------------------------------------------------------------
    # Setup working directory and logging
    example_wd = os.path.expanduser(working_dir)

    # copy models to working directory
    curr_dir = os.path.dirname(os.path.realpath(__file__)) + '/'
    if os.path.isdir(curr_dir + '/models/') and not os.path.isdir(example_wd + '/models/'):
        shutil.copytree(curr_dir + '/models', example_wd + '/models/')

    log = initialize_logging(experiment_name, log_dir=example_wd + '/logs/')
    time_stamps = [time.time()]
    step_idents = ['t-0']
    log.info('Step 0/8 - Preparation')

    bd = None
    bb = None

    # Preparing config
    # currently this is were SyConn looks for the neuron rag
    if global_params.wd is not None:
        log.critical('Example run started. Original working directory defined'
                     ' in `global_params.py` '
                     'is overwritten and set to "{}".'.format(example_wd))

    key_val_pairs_conf = [
        ('prior_glia_removal', prior_glia_removal),
        ('pyopengl_platform', 'egl'),
        ('batch_proc_system', "SLURM"),
        ('ncores_per_node', 20),
        ('ngpus_per_node', 2),
        ('nnodes_total', 17),
    ]

    generate_default_conf(example_wd, scale,
                          key_value_pairs=key_val_pairs_conf,
                          force_overwrite=True,
                          kd_seg=raw_kd_path, kd_asym=kd_asym_path, kd_mi=mi_kd_path,
                          kd_sj=sj_kd_path, kd_sym=kd_sym_path, kd_vc=vc_kd_path)

    global_params.wd = example_wd
    os.makedirs(global_params.config.temp_path, exist_ok=True)

    # get original RAG
    # rag_txtfname = working_dir + "/v4b_base_20180214.full_agglo.cbsplit.csv"
    rag_txtfname = working_dir + "/v4b_base_20180214.full_agglo.cbsplit.csv"

    G = nx.Graph()  # TODO: Make this more general
    with open(rag_txtfname, 'r') as f:
        for l in f.readlines():
            edges = [int(v) for v in re.findall('(\d+)', l)]
            G.add_edge(edges[0], edges[1])
    # G = nx.read_edgelist(rag_txtfname)  # in case a networkx edgelist is given

    nx.write_edgelist(G, global_params.config.init_rag_path)
    start = time.time()
    # Checking models
    for mpath_key in ['mpath_spiness', 'mpath_syn_rfc', 'mpath_celltype',
                      'mpath_axoness', 'mpath_glia']:
        mpath = getattr(global_params.config, mpath_key)
        if not (os.path.isfile(mpath) or os.path.isdir(mpath)):
            raise ValueError('Could not find model "{}". Make sure to copy the'
                             ' "models" folder into the current working '
                             'directory "{}".'.format(mpath, example_wd))

    # Start SyConn
    # --------------------------------------------------------------------------
    log.info('Step 0/8 - Predicting sub-cellular structures')
    # # TODO: uncomment
    # exec_dense_prediction.predict_myelin()  # myelin is not needed before `run_create_neuron_ssd`
    time_stamps.append(time.time())
    step_idents.append('Dense predictions')

    log.info('Data will be processed in "{}".'.format(example_wd))
    time_stamps.append(time.time())
    step_idents.append('Preparation')

    log.info('Step 1/8 - Creating SegmentationDatasets (incl. SV meshes) and the initial RAG.')

    # TODO: uncomment
    exec_init.init_cell_subcell_sds(chunk_size=chunk_size, n_folders_fs_sc=n_folders_fs_sc,
                                    n_folders_fs=n_folders_fs, cube_of_interest_bb=bb,
                                    load_cellorganelles_from_kd_overlaycubes=False)

    # TODO: uncomment
    exec_init.run_create_rag()
    time_stamps.append(time.time())
    step_idents.append('SD generation')
    print("\n\n\n HAPPY END \n\n\n")
    # if global_params.config.prior_glia_removal:
    #     log.info('Step 1.5/8 - Glia separation')
    #     # TODO: uncomment
    # #     exec_multiview.run_glia_rendering()
    # #     exec_multiview.run_glia_prediction(e3=True)
    #     # TODO: uncomment
    # #     exec_multiview.run_glia_splitting()
    #     time_stamps.append(time.time())
    #     step_idents.append('Glia separation')
    #
    # log.info('Step 2/8 - Creating SuperSegmentationDataset')
    # # TODO: uncomment
    # # exec_multiview.run_create_neuron_ssd()
    # # TODO: remove! only used for partial runs, otherwise this mapping is performed inside `run_create_neuron_ssd`
    # # # TODO: uncomment
    # # exec_skeleton.map_myelin_global()
    # time_stamps.append(time.time())
    # step_idents.append('SSD generation')
    #
    # log.info('Step 3/8 - Synapse detection')
    # # TODO: uncomment
    # exec_syns.run_syn_generation(chunk_size=chunk_size, n_folders_fs=n_folders_fs_sc,
    #                              cube_of_interest_bb=bb)
    # time_stamps.append(time.time())
    # step_idents.append('Synapse detection')
    #
    # log.info('Step 4/8 - Neuron rendering')
    # # TODO: uncomment
    # # exec_multiview.run_neuron_rendering()
    # time_stamps.append(time.time())
    # step_idents.append('Neuron rendering')
    #
    # log.info('Step 5/8 - Axon prediction')
    # # # OLD
    # # exec_multiview.run_axoness_prediction(e3=True)
    # # exec_multiview.run_axoness_mapping()
    # # # TODO: uncomment
    # # exec_multiview.run_semsegaxoness_prediction()
    # # exec_multiview.run_semsegaxoness_mapping()
    # time_stamps.append(time.time())
    # step_idents.append('Axon prediction')
    #
    # log.info('Step 6/8 - Spine prediction')
    # # TODO: check if errors in batchjob submission failed to to memory error
    # #  only - then allow resubmission of jobs
    # # TODO: uncomment
    # # exec_multiview.run_spiness_prediction()
    # time_stamps.append(time.time())
    # step_idents.append('Spine prediction')
    #
    # log.info('Step 7/9 - Morphology extraction')
    # # TODO: uncomment
    # # exec_multiview.run_morphology_embedding()
    # time_stamps.append(time.time())
    # step_idents.append('Morphology extraction')
    #
    # log.info('Step 8/9 - Celltype analysis')
    # # TODO: uncomment
    # # exec_multiview.run_celltype_prediction()
    # time_stamps.append(time.time())
    # step_idents.append('Celltype analysis')
    #
    # log.info('Step 9/9 - Matrix export')
    # exec_syns.run_matrix_export()
    # time_stamps.append(time.time())
    # step_idents.append('Matrix export')
    #
    # time_stamps = np.array(time_stamps)
    # dts = time_stamps[1:] - time_stamps[:-1]
    # dt_tot = time_stamps[-1] - time_stamps[0]
    # dt_tot_str = time.strftime("%Hh:%Mmin:%Ss", time.gmtime(dt_tot))
    # time_summary_str = "\nEM data analysis of experiment '{}' finished after" \
    #                    " {}.\n".format(experiment_name, dt_tot_str)
    # n_steps = len(step_idents[1:]) - 1
    # for i in range(len(step_idents[1:])):
    #     step_dt = time.strftime("%Hh:%Mmin:%Ss", time.gmtime(dts[i]))
    #     step_dt_perc = int(dts[i] / dt_tot * 100)
    #     step_str = "[{}/{}] {}\t\t\t{}\t\t\t{}%\n".format(
    #         i, n_steps, step_idents[i+1], step_dt, step_dt_perc)
    #     time_summary_str += step_str
    # log.info(time_summary_str)
    # log.info('Setting up flask server for inspection. Annotated cell reconst'
    #          'ructions and wiring can be analyzed via the KNOSSOS-SyConn plugin'
    #          ' at `SyConn/scripts/kplugin/syconn_knossos_viewer.py`.')
    # fname_server = os.path.dirname(os.path.abspath(__file__)) + \
    #                '/../kplugin/server.py'
    # os.system('python {} --working_dir={} --port=10002'.format(
    #     fname_server, example_wd))