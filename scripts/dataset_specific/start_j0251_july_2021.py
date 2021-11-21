# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max-Planck-Institute of Neurobiology, Munich, Germany
# Authors: Philipp Schubert, Joergen Kornfeld

import os
import time
import numpy as np
import re
import glob
import networkx as nx
import pandas
import tqdm

from syconn.handler.config import generate_default_conf, initialize_logging
from syconn import global_params
from syconn.proc.stats import FileTimer
from syconn.exec import exec_init, exec_syns, exec_render, exec_dense_prediction, exec_inference, exec_skeleton


if __name__ == '__main__':
    # ----------------- DEFAULT WORKING DIRECTORY ---------------------
    working_dir = "/ssdscratch/songbird/j0251/j0251_72_seg_20210127_agglo2/"
    experiment_name = 'j0251'
    scale = np.array([10, 10, 25])
    key_val_pairs_conf = [
        ('min_cc_size_ssv', 2000),  # minimum bounding box diagonal of cell (fragments) in nm
        ('glia', {'prior_astrocyte_removal': False}),
        ('pyopengl_platform', 'egl'),
        ('batch_proc_system', 'SLURM'),
        ('ncores_per_node', 20),
        ('ngpus_per_node', 2),
        ('nnodes_total', 17),
        ('use_point_models', True),
        ('meshes', {'use_new_meshing': True}),
        ('cell_contacts',
         {'generate_cs_ssv': False,  # cs_ssv: contact site objects between cells
          'min_path_length_partners': 50,
          }),
        ('views', {'use_onthefly_views': True,
                   'use_new_renderings_locs': True,
                   'view_properties': {'nb_views': 3}
                   }),
        ('slurm', {'exclude_nodes': ['wb08', 'wb09', 'wb10']}),
        ('cell_objects',
         {'sym_label': 1, 'asym_label': 2,
          'min_obj_vx': {'sv': 1},
          # first remove small fragments, close existing holes, then erode to trigger watershed segmentation
          'extract_morph_op': {'mi': ['binary_opening', 'binary_closing', 'binary_erosion', 'binary_erosion',
                                      'binary_erosion', 'binary_erosion'],
                               'sj': ['binary_opening', 'binary_closing'],
                               'vc': ['binary_opening', 'binary_closing', 'binary_erosion', 'binary_erosion']}
          }
         )
    ]
    chunk_size = None
    n_folders_fs = 10000
    n_folders_fs_sc = 10000

    # ----------------- DATA DIRECTORY ---------------------
    raw_kd_path = '/wholebrain/songbird/j0251/j0251_72_clahe2/'
    root_dir = '/ssdscratch/songbird/j0251/segmentation/'
    seg_kd_path = root_dir + 'j0251_72_seg_20210127_agglo2/'
    kd_asym_path = root_dir + 'j0251_asym_sym/'
    kd_sym_path = root_dir + 'j0251_asym_sym/'
    syntype_avail = (kd_asym_path is not None) and (kd_sym_path is not None)
    mi_kd_path = root_dir + 'latest_sj_vc_mito/'
    vc_kd_path = root_dir + 'latest_sj_vc_mito/'
    sj_kd_path = root_dir + 'latest_sj_vc_mito/'

    # The transform functions will be applied when loading the segmentation data of cell organelles
    # in order to convert them into binary fore- and background
    # currently using `dill` package to support lambda expressions, a weak feature. Make
    #  sure all dependencies within the lambda expressions are imported in
    #  `batchjob_object_segmentation.py` (here: numpy)
    cellorganelle_transf_funcs = dict(mi=lambda x: (x == 1).astype('u1'),
                                      vc=lambda x: (x == 3).astype('u1'),
                                      sj=lambda x: (x == 2).astype('u1'))

    # Preparing data
    # --------------------------------------------------------------------------
    # Setup working directory and logging
    log = initialize_logging(experiment_name, log_dir=working_dir + '/logs/')
    ftimer = FileTimer(working_dir + '/.timing.pkl')
    ftimer.start('Preparation')

    # # Preparing config
    # # currently this is were SyConn looks for the neuron rag
    # if global_params.wd is not None:
    #     log.critical('Example run started. Original working directory defined'
    #                  ' in `global_params.py` '
    #                  'is overwritten and set to "{}".'.format(working_dir))
    #
    # generate_default_conf(working_dir, scale, syntype_avail=syntype_avail, kd_seg=seg_kd_path, kd_mi=mi_kd_path,
    #                       kd_vc=vc_kd_path, kd_sj=sj_kd_path, kd_sym=kd_sym_path, kd_asym=kd_asym_path,
    #                       key_value_pairs=key_val_pairs_conf, force_overwrite=True)
    #
    # global_params.wd = working_dir
    # os.makedirs(global_params.config.temp_path, exist_ok=True)
    # start = time.time()
    #
    # # check model existence
    # for mpath_key in ['mpath_spiness', 'mpath_syn_rfc', 'mpath_celltype_e3',
    #                   'mpath_axonsem', 'mpath_glia_e3', 'mpath_myelin',
    #                   'mpath_tnet']:
    #     mpath = getattr(global_params.config, mpath_key)
    #     if not (os.path.isfile(mpath) or os.path.isdir(mpath)):
    #         raise ValueError('Could not find model "{}". Make sure to copy the'
    #                          ' "models" folder into the current working '
    #                          'directory "{}".'.format(mpath, working_dir))
    # ftimer.stop()

    # # Start SyConn
    # # --------------------------------------------------------------------------
    # log.info('Starting SyConn pipeline for data cube (shape: {}).'.format(ftimer.dataset_shape))
    # log.critical('Working directory is set to "{}".'.format(working_dir))
    #
    # log.info('Step 2/9 - Creating SegmentationDatasets (incl. SV meshes)')
    # ftimer.start('SD generation')
    # exec_init.init_cell_subcell_sds(chunk_size=chunk_size, n_folders_fs_sc=n_folders_fs_sc,
    #                                 n_folders_fs=n_folders_fs,
    #                                 load_cellorganeclles_from_kd_overlaycubes=True,
    #                                 transf_func_kd_overlay=cellorganelle_transf_funcs,
    #                                 max_n_jobs=global_params.config.ncore_total * 4)
    #
    # # generate flattened RAG
    # from syconn.reps.segmentation import SegmentationDataset
    # sd = SegmentationDataset(obj_type="sv", working_dir=global_params.config.working_dir)
    # rag_sub_g = nx.Graph()
    # # add SV IDs to graph via self-edges
    # mesh_bb = sd.load_numpy_data('mesh_bb')  # N, 2, 3
    # mesh_bb = np.linalg.norm(mesh_bb[:, 1] - mesh_bb[:, 0], axis=1)
    # filtered_ids = sd.ids[mesh_bb > global_params.config['min_cc_size_ssv']]
    # rag_sub_g.add_edges_from([[el, el] for el in filtered_ids)
    # log.info('{} SVs were added to the RAG after applying BBD size '
    #          'filter (min. BBD = {global_params.config['min_cc_size_ssv']} nm).'.format(len(filtered_ids)))
    # nx.write_edgelist(rag_sub_g, global_params.config.init_svgraph_path)
    #
    # exec_init.run_create_rag(graph_node_dtype=np.uint32)
    # ftimer.stop()
    #
    # log.info('Step 3/9 - Creating SuperSegmentationDataset')
    # ftimer.start('SSD generation')
    # exec_init.run_create_neuron_ssd(ncores_per_job=4)
    # ftimer.stop()
    #
    # log.info('Step 4/10 - Skeleton generation')
    # ftimer.start('Skeleton generation')
    # exec_skeleton.run_skeleton_generation()
    # ftimer.stop()

    # log.info('Step 5/9 - Synapse detection')
    # ftimer.start('Synapse detection')
    # exec_syns.run_syn_generation(chunk_size=chunk_size, n_folders_fs=n_folders_fs_sc,
    #                              transf_func_sj_seg=cellorganelle_transf_funcs['sj'])
    # ftimer.stop()
    #
    # log.info('Step 7/9 - Morphology extraction')
    # ftimer.start('Morphology extraction')
    # exec_inference.run_cell_embedding()
    # ftimer.stop()

    # Process contact sites only with some nodes
    global_params.config['slurm']['exclude_nodes'] = ['wb07', 'wb08', 'wb09',
                                                      'wb10', 'wb11']
    global_params.config.write_config()
    time.sleep(10)  # wait for changes to apply
    log.info('Step 5.5/9 - Contact detection')
    ftimer.start('Contact detection')
    if global_params.config['cell_contacts']['generate_cs_ssv']:
        exec_syns.run_cs_ssv_generation(n_folders_fs=n_folders_fs_sc)
    else:
        log.info('Cell-cell contact detection ("cs_ssv" objects) disabled. Skipping.')
    ftimer.stop()

    # global_params.config['slurm']['exclude_nodes'] = []
    # global_params.config.write_config()
    # time.sleep(10)  # wait for changes to apply
    #
    # log.info('Step 6/9 - Compartment prediction')
    # ftimer.start('Compartment predictions')
    # exec_inference.run_semsegaxoness_prediction()
    # if not global_params.config.use_point_models:
    #     exec_inference.run_semsegspiness_prediction()
    # ftimer.stop()
    #
    # ftimer.start('Spine head volume estimation')
    # exec_syns.run_spinehead_volume_calc()
    # ftimer.stop()

    # log.info('Step 8/9 - Celltype analysis')
    # ftimer.start('Celltype analysis')
    # exec_inference.run_celltype_prediction()
    # ftimer.stop()
    #
    # log.info('Step 9/9 - Matrix export')
    # ftimer.start('Matrix export')
    # exec_syns.run_matrix_export()
    # ftimer.stop()
    #
    # time_summary_str = ftimer.prepare_report()
    # log.info(time_summary_str)
    # # log.info('Setting up flask server for inspection. Annotated cell reconstructions and wiring '
    # #          'can be analyzed via the KNOSSOS-SyConn plugin at '
    # #          '`SyConn/scripts/kplugin/syconn_knossos_viewer.py`.')
    # # os.system(f'syconn.server --working_dir={example_wd} --port=10001')
