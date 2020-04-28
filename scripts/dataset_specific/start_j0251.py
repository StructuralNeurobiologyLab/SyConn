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
import sys
import time
import argparse
import networkx as nx

from syconn.handler.prediction import parse_movement_area_from_zip
from syconn.handler.config import generate_default_conf, initialize_logging
from syconn.handler.compression import load_from_h5py
from syconn import global_params
from syconn.exec import exec_init, exec_syns, exec_multiview, exec_dense_prediction


if __name__ == '__main__':
    # ----------------- DEFAULT WORKING DIRECTORY ---------------------
    working_dir = "/ssdscratch/pschuber/songbird/j0251/rag_flat_Jan2019/"
    experiment_name = 'j0251'
    scale = np.array([10, 10, 25])
    prior_glia_removal = True
    key_val_pairs_conf = [
        ('glia', {'prior_glia_removal': prior_glia_removal}),
        ('pyopengl_platform', 'egl'),
        ('batch_proc_system', 'SLURM'),
        ('ncores_per_node', 20),
        ('ngpus_per_node', 2),
        ('nnodes_total', 17),
        ('meshes', {'use_new_meshing': True}),
        ('views', {'use_large_fov_views_ct': False,
                   'use_new_renderings_locs': True,
                   'nb_views': 3}),
        ('cell_objects', {'sym_label': 1, 'asym_label': 2,
                          'min_obj_vx': {'sv': 100}})  # flattened RAG contains only on SV per cell
    ]
    chunk_size = None
    n_folders_fs = 10000
    n_folders_fs_sc = 10000

    # ----------------- DATA DIRECTORY ---------------------
    raw_kd_path = '/wholebrain/songbird/j0251/j0251_72_clahe2/'
    root_dir = '/ssdscratch/pschuber/songbird/j0251/'
    kd_asym_path = root_dir + 'j0251_asym_sym/'
    kd_sym_path = root_dir + 'j0251_asym_sym/'
    syntype_avail = (kd_asym_path is not None) and (kd_sym_path is not None)
    seg_kd_path = root_dir + 'latest_seg/'
    mi_kd_path = root_dir + 'latest_sj_vc_mito/'
    vc_kd_path = root_dir + 'latest_sj_vc_mito/'
    sj_kd_path = root_dir + 'latest_sj_vc_mito/'

    # The transform functions will be applied when loading the segmentation data of cell organelles
    # in order to convert them into binary fore- and background
    # currently using `dill` package to support lambda expressions, a weak feature. Make
    #  sure all dependencies within the lambda expressions are imported in
    #  `QSUB_gauss_threshold_connected_components.py` (here: numpy)
    cellorganelle_transf_funcs = dict(mi=lambda x: ((x == 1)).astype(np.uint8),
                                      vc=lambda x: ((x == 3)).astype(np.uint8),
                                      sj=lambda x: ((x == 2)).astype(np.uint8))

    # Preparing data
    # --------------------------------------------------------------------------
    # Setup working directory and logging
    log = initialize_logging(experiment_name, log_dir=working_dir + '/logs/')
    time_stamps = [time.time()]
    step_idents = ['t-0']
    log.info('Step 0/8 - Preparation')

    bb = None
    bd = None

    # Preparing config
    # currently this is were SyConn looks for the neuron rag
    if global_params.wd is not None:
        log.critical('Example run started. Original working directory defined'
                     ' in `global_params.py` '
                     'is overwritten and set to "{}".'.format(working_dir))

    generate_default_conf(
        working_dir, scale, syntype_avail=syntype_avail,
        kd_seg=seg_kd_path, kd_mi=mi_kd_path,
        kd_vc=vc_kd_path, kd_sj=sj_kd_path,
        kd_sym=kd_sym_path, kd_asym=kd_asym_path,
        key_value_pairs=key_val_pairs_conf,
        force_overwrite=True)

    global_params.wd = working_dir
    os.makedirs(global_params.config.temp_path, exist_ok=True)
    start = time.time()

    # Checking models
    for mpath_key in ['mpath_spiness', 'mpath_syn_rfc', 'mpath_celltype',
                      'mpath_axoness', 'mpath_glia']:
        mpath = getattr(global_params.config, mpath_key)
        if not (os.path.isfile(mpath) or os.path.isdir(mpath)):
            raise ValueError('Could not find model "{}". Make sure to copy the'
                             ' "models" folder into the current working '
                             'directory "{}".'.format(mpath, working_dir))

    # Start SyConn
    # --------------------------------------------------------------------------
    log.info('Finished example cube initialization (shape: {}). Starting'
             ' SyConn pipeline.'.format(bd))
    log.info('Example data will be processed in "{}".'.format(working_dir))
    time_stamps.append(time.time())
    step_idents.append('Preparation')

    log.info('Step 0/8 - Predicting sub-cellular structures')
    # myelin is not needed before `run_create_neuron_ssd`
    # exec_dense_prediction.predict_myelin(raw_kd_path)
    time_stamps.append(time.time())
    step_idents.append('Dense predictions')

    # log.info('Step 1/8 - Creating SegmentationDatasets (incl. SV meshes)')
    # exec_init.init_cell_subcell_sds(chunk_size=chunk_size, n_folders_fs_sc=n_folders_fs_sc,
    #                                 n_folders_fs=n_folders_fs,
    #                                 load_cellorganelles_from_kd_overlaycubes=True,
    #                                 transf_func_kd_overlay=cellorganelle_transf_funcs,
    #                                 n_cores=1,
    #                                 max_n_jobs=global_params.config.ncore_total * 4)
    #
    # # generate flattened RAG
    # from syconn.reps.segmentation import SegmentationDataset
    # sd = SegmentationDataset(obj_type="sv", working_dir=global_params.config.working_dir)
    # rag_sub_g = nx.Graph()
    # # add SV IDs to graph via self-edges
    # # mesh_bb = sd.load_cached_data('mesh_bb')  # N, 2, 3
    # # mesh_bb = np.linalg.norm(mesh_bb[:, 1] - mesh_bb[:, 0], axis=1)
    # # filtered_ids = sd.ids[mesh_bb > global_params.config['glia']['min_cc_size_ssv']]
    # rag_sub_g.add_edges_from([[el, el] for el in sd.ids])
    # # log.info('{} SVs were added to the RAG after application of the size '
    # #          'filter.'.format(len(filtered_ids)))
    # nx.write_edgelist(rag_sub_g, global_params.config.init_rag_path)
    #
    # exec_init.run_create_rag()
    #
    # time_stamps.append(time.time())
    # step_idents.append('SD generation')

    if global_params.config.prior_glia_removal:
        log.info('Step 1.5/8 - Glia separation')
        # exec_multiview.run_glia_rendering()
        # exec_multiview.run_glia_prediction()
        # exec_multiview.run_glia_splitting()
        time_stamps.append(time.time())
        step_idents.append('Glia separation')

    log.info('Step 2/8 - Creating SuperSegmentationDataset')
    # exec_multiview.run_create_neuron_ssd()
    time_stamps.append(time.time())
    step_idents.append('SSD generation')

    log.info('Step 3/8 - Neuron rendering')
    # exec_multiview.run_neuron_rendering()
    time_stamps.append(time.time())
    step_idents.append('Neuron rendering')

    log.info('Step 4/8 - Synapse detection')
    exec_syns.run_syn_generation(chunk_size=chunk_size,
                                 n_folders_fs=n_folders_fs_sc)
    time_stamps.append(time.time())
    step_idents.append('Synapse detection')

    log.info('Step 5/8 - Axon prediction')
    exec_multiview.run_semsegaxoness_prediction()
    exec_multiview.run_semsegaxoness_mapping()
    time_stamps.append(time.time())
    step_idents.append('Axon prediction')

    log.info('Step 6/8 - Spine prediction')
    exec_multiview.run_spiness_prediction()
    time_stamps.append(time.time())
    step_idents.append('Spine prediction')

    log.info('Step 7/9 - Morphology extraction')
    exec_multiview.run_morphology_embedding()
    time_stamps.append(time.time())
    step_idents.append('Morphology extraction')

    log.info('Step 8/9 - Celltype analysis')
    exec_multiview.run_celltype_prediction()
    time_stamps.append(time.time())
    step_idents.append('Celltype analysis')

    log.info('Step 9/9 - Matrix export')
    exec_syns.run_matrix_export()
    time_stamps.append(time.time())
    step_idents.append('Matrix export')

    time_stamps = np.array(time_stamps)
    dts = time_stamps[1:] - time_stamps[:-1]
    dt_tot = time_stamps[-1] - time_stamps[0]
    dt_tot_str = time.strftime("%Hh:%Mmin:%Ss", time.gmtime(dt_tot))
    time_summary_str = "\nEM data analysis of experiment '{}' finished after" \
                       " {}.\n".format(experiment_name, dt_tot_str)
    n_steps = len(step_idents[1:]) - 1
    for i in range(len(step_idents[1:])):
        step_dt = time.strftime("%Hh:%Mmin:%Ss", time.gmtime(dts[i]))
        step_dt_perc = int(dts[i] / dt_tot * 100)
        step_str = "[{}/{}] {}\t\t\t{}\t\t\t{}%\n".format(
            i, n_steps, step_idents[i+1], step_dt, step_dt_perc)
        time_summary_str += step_str
    log.info(time_summary_str)
    log.info('Setting up flask server for inspection. Annotated cell reconst'
             'ructions and wiring can be analyzed via the KNOSSOS-SyConn plugin'
             ' at `SyConn/scripts/kplugin/syconn_knossos_viewer.py`.')
    fname_server = os.path.dirname(os.path.abspath(__file__)) + \
                   '/../kplugin/server.py'
    os.system('python {} --working_dir={} --port=10001'.format(
        fname_server, working_dir))
