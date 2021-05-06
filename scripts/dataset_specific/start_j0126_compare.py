# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max-Planck-Institute of Neurobiology, Munich, Germany
# Authors: Philipp Schubert, Joergen Kornfeld

import os
import time
import argparse
import re
import networkx as nx
import numpy as np
from syconn.reps.segmentation import SegmentationDataset
from syconn.proc.sd_proc import dataset_analysis
from syconn.proc.ssd_proc import map_synssv_objects
from syconn.extraction import cs_processing_steps as cps
from syconn.handler.config import initialize_logging
from syconn import global_params
from syconn.exec import exec_syns, exec_render, exec_skeleton, exec_init, exec_inference


if __name__ == '__main__':
    raise DeprecationWarning('This script is outdated. See earlier commits for functional versions.')
    parser = argparse.ArgumentParser(description='SyConn example run')
    parser.add_argument('--working_dir', type=str, default='',
                        help='Working directory of SyConn')
    args = parser.parse_args()

    n_folders_fs = 10000
    n_folders_fs_sc = 10000

    example_wd = '/ssdscratch/pschuber/songbird/j0126/areaxfs_v10_v4b_base_' \
                 '20180214_full_agglo_cbsplit/'

    experiment_name = 'j0126'
    log = initialize_logging(experiment_name, log_dir=example_wd + '/logs/')
    time_stamps = [time.time()]
    step_idents = ['t-0']

    global_params.wd = example_wd

    # get original RAG
    rag_txtfname = f"{example_wd}/v4b_base_20180214.full_agglo.cbsplit.csv"
    G = nx.Graph()
    with open(rag_txtfname, 'r') as f:
        for l in f.readlines():
            edges = [int(v) for v in re.findall(r'(\d+)', l)]
            G.add_edge(edges[0], edges[1])

    nx.write_edgelist(G, global_params.config.init_svgraph_path)
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
    log.info('Data will be processed in "{}".'.format(example_wd))
    time_stamps.append(time.time())
    step_idents.append('Preparation')

    log.info('Step 1/8 - Creating SegmentationDatasets (incl. SV meshes) and'
             ' the initial RAG.')
    # exec_init.run_create_rag()
    time_stamps.append(time.time())
    step_idents.append('SD generation')

    if global_params.config.prior_astrocyte_removal:
        log.info('Step 1.5/8 - Astrocyte separation')
        exec_render.run_astrocyte_rendering()
        exec_inference.run_astrocyte_prediction()
        exec_inference.run_astrocyte_splitting()
        time_stamps.append(time.time())
        step_idents.append('Astrocyte separation')

    log.info('Step 2/8 - Creating SuperSegmentationDataset')
    exec_init.run_create_neuron_ssd()
    exec_skeleton.map_myelin_global()
    time_stamps.append(time.time())
    step_idents.append('SSD generation')

    log.info('Step 3/8 - Synapse detection')
    sd_syn_ssv = SegmentationDataset(working_dir=global_params.config.working_dir,
                                     obj_type='syn_ssv')
    # # This creates an SD of type 'syn_ssv'
    cps.combine_and_split_syn_old(global_params.config.working_dir,
                              cs_gap_nm=global_params.config['cell_objects']['cs_gap_nm'],
                              log=log, n_folders_fs=n_folders_fs)
    cps.extract_synapse_type(sd_syn_ssv,
                             kd_sym_path=global_params.config.kd_sym_path,
                             kd_asym_path=global_params.config.kd_asym_path,
                             sym_label=2, asym_label=1)

    log.info('Synapse objects were created.')
    dataset_analysis(sd_syn_ssv, compute_meshprops=True)
    log.info(f'SegmentationDataset of type "syn_ssv" was generated with {len(sd_syn_ssv.ids)} '
             f'objects.')
    cps.map_objects_to_synssv(global_params.config.working_dir, log=log)
    log.info('Cellular organelles were mapped to "syn_ssv".')
    cps.classify_synssv_objects(global_params.config.working_dir, log=log)
    log.info('Synapse prediction finished.')
    log.info('Collecting and writing syn-ssv objects to SSV attribute '
             'dictionary.')
    # This needs to be run after `classify_synssv_objects` and before
    # `map_synssv_objects` if the latter uses thresholding for synaptic objects
    # just collect new data: ``recompute=False``
    dataset_analysis(sd_syn_ssv, compute_meshprops=False, recompute=False)
    map_synssv_objects(log=log)
    log.info('Finished.')
    time_stamps.append(time.time())
    step_idents.append('Synapse detection')

    log.info('Step 4/8 - Neuron rendering')
    exec_render.run_neuron_rendering()
    time_stamps.append(time.time())
    step_idents.append('Neuron rendering')

    log.info('Step 5/8 - Axon prediction')
    exec_inference.run_semsegaxoness_prediction()
    exec_inference.run_semsegaxoness_mapping()
    time_stamps.append(time.time())
    step_idents.append('Axon prediction')

    log.info('Step 6/8 - Spine prediction')
    exec_inference.run_spiness_prediction()
    exec_syns.run_spinehead_volume_calc()
    time_stamps.append(time.time())
    step_idents.append('Spine prediction')

    log.info('Step 7/9 - Morphology extraction')
    exec_inference.run_morphology_embedding()
    time_stamps.append(time.time())
    step_idents.append('Morphology extraction')

    log.info('Step 8/9 - Celltype analysis')
    exec_inference.run_celltype_prediction()
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
    os.system('python {} --working_dir={} --port=10002'.format(
        fname_server, example_wd))
