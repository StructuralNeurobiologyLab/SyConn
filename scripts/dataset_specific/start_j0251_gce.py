# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max-Planck-Institute of Neurobiology, Munich, Germany
# Authors: Philipp Schubert, Joergen Kornfeld

import os
import time
import numpy as np
import networkx as nx
import shutil

from syconn.handler.basics import FileTimer
from syconn.handler.config import generate_default_conf, initialize_logging
from syconn import global_params
from syconn.exec import exec_init, exec_syns, exec_render, exec_dense_prediction, exec_inference

# TODO: set myelin knossosdataset - currently mapping of myelin to skeletons does not allow partial cubes

if __name__ == '__main__':
    # ----------------- DEFAULT WORKING DIRECTORY ---------------------
    experiment_name = 'j0251'
    scale = np.array([10, 10, 25])
    prior_glia_removal = False
    key_val_pairs_conf = [
        ('glia', {'prior_glia_removal': prior_glia_removal, 'min_cc_size_ssv': 5000}),  # in nm
        ('pyopengl_platform', 'egl'),
        ('batch_proc_system', 'SLURM'),
        ('ncores_per_node', 32),
        ('ngpus_per_node', 2),
        ('nnodes_total', 5),
        ('mem_per_node', 208990),
        ('use_point_models', False),
        ('skeleton', {'use_kimimaro': True}),
        ('meshes', {'use_new_meshing': True}),
        ('views', {'use_onthefly_views': True,
                   'use_new_renderings_locs': True,
                   'view_properties': {'nb_views': 3}
                   }),
        ('cell_objects',
         {'sym_label': 1, 'asym_label': 2,
          'min_obj_vx': {'sv': 100},  # flattened RAG contains only on SV per cell
          # first remove small fragments, close existing holes, then erode to trigger watershed segmentation
          'extract_morph_op': {'mi': ['binary_opening', 'binary_closing', 'binary_erosion', 'binary_erosion',
                                      'binary_erosion'],
                               'sj': ['binary_opening', 'binary_closing', 'binary_erosion'],
                               'vc': ['binary_opening', 'binary_closing', 'binary_erosion']}
          }
         )
    ]

    chunk_size = (512, 512, 256)
    n_folders_fs = 10000
    n_folders_fs_sc = 10000

    # ----------------- DATA DIRECTORY ---------------------
    raw_kd_path = None
    root_dir = '/mnt/j0251_data/'
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
    #  `batchjob_object_segmentation.py` (here: numpy)
    cellorganelle_transf_funcs = dict(mi=lambda x: (x == 1).astype(np.uint8),
                                      vc=lambda x: (x == 3).astype(np.uint8),
                                      sj=lambda x: (x == 2).astype(np.uint8))

    # Prepare data
    # --------------------------------------------------------------------------
    # Setup working directory and logging
    shape_j0251 = np.array([27119, 27350, 15494])
    cube_size = np.array([2048, 2048, 1024]) // 2
    cube_offset = (shape_j0251 - cube_size) // 2
    cube_of_interest_bb = (cube_offset, cube_offset + cube_size)
    # cube_of_interest_bb = None  # process the entire cube!
    working_dir = f"/mnt/example_runs/j0251_off{'_'.join(map(str, cube_offset))}_size{'_'.join(map(str, cube_size))}"
    log = initialize_logging(experiment_name, log_dir=working_dir + '/logs/')
    ftimer = FileTimer(working_dir + '/.timing.pkl')

    log.info('Step 0/9 - Preparation')
    ftimer.start('Preparation')
    # figure out SyConn data path and copy models to working directory
    for curr_dir in [os.path.dirname(os.path.realpath(__file__)) + '/',
                     os.path.abspath(os.path.curdir) + '/',
                     os.path.expanduser('~/SyConn/')]:
        if os.path.isdir(curr_dir + '/models'):
            break
    if os.path.isdir(curr_dir + '/models/') and not os.path.isdir(working_dir + '/models/'):
        shutil.copytree(curr_dir + '/models', working_dir + '/models/')

    # Prepare config
    if global_params.wd is not None:
        log.critical('Example run started. Original working directory defined in `global_params.py` '
                     'is overwritten and set to "{}".'.format(working_dir))

    generate_default_conf(working_dir, scale, syntype_avail=syntype_avail, kd_seg=seg_kd_path, kd_mi=mi_kd_path,
                          kd_vc=vc_kd_path, kd_sj=sj_kd_path, kd_sym=kd_sym_path, kd_asym=kd_asym_path,
                          key_value_pairs=key_val_pairs_conf, force_overwrite=True)

    global_params.wd = working_dir
    os.makedirs(global_params.config.temp_path, exist_ok=True)

    # create symlink to myelin predictions
    if not os.path.exists(f'/mnt/j0251_data/myelin {working_dir}/knossosdatasets/'):
        assert os.path.exists('/mnt/j0251_data/myelin')
        os.system(f'ln -s /mnt/j0251_data/myelin {working_dir}/knossosdatasets/')

    # check model existence
    for mpath_key in ['mpath_spiness', 'mpath_syn_rfc', 'mpath_celltype_e3', 'mpath_axonsem', 'mpath_glia_e3',
                      'mpath_myelin', 'mpath_tnet']:
        mpath = getattr(global_params.config, mpath_key)
        if not (os.path.isfile(mpath) or os.path.isdir(mpath)):
            raise ValueError('Could not find model "{}". Make sure to copy the "models" folder into the current '
                             'working directory "{}".'.format(mpath, working_dir))
    ftimer.stop()

    # Start SyConn
    # --------------------------------------------------------------------------
    log.info('Finished example cube initialization (shape: {}). Starting'
             ' SyConn pipeline.'.format(cube_size))
    log.info('Example data will be processed in "{}".'.format(working_dir))

    log.info('Step 1/9 - Predicting sub-cellular structures')
    # ftimer.start('Myelin prediction')
    # # myelin is not needed before `run_create_neuron_ssd`
    # exec_dense_prediction.predict_myelin(raw_kd_path, cube_of_interest=cube_of_interest_bb)
    # ftimer.stop()

    log.info('Step 2/9 - Creating SegmentationDatasets (incl. SV meshes)')
    ftimer.start('SD generation')
    exec_init.init_cell_subcell_sds(chunk_size=chunk_size, n_folders_fs_sc=n_folders_fs_sc,
                                    n_folders_fs=n_folders_fs, cube_of_interest_bb=cube_of_interest_bb,
                                    load_cellorganelles_from_kd_overlaycubes=True,
                                    transf_func_kd_overlay=cellorganelle_transf_funcs,
                                    max_n_jobs=global_params.config.ncore_total * 4)

    # generate flattened RAG
    from syconn.reps.segmentation import SegmentationDataset
    sd = SegmentationDataset(obj_type="sv", working_dir=global_params.config.working_dir)
    rag_sub_g = nx.Graph()
    # add SV IDs to graph via self-edges
    mesh_bb = sd.load_cached_data('mesh_bb')  # N, 2, 3
    mesh_bb = np.linalg.norm(mesh_bb[:, 1] - mesh_bb[:, 0], axis=1)
    filtered_ids = sd.ids[mesh_bb > global_params.config['glia']['min_cc_size_ssv']]
    rag_sub_g.add_edges_from([[el, el] for el in sd.ids])
    log.info('{} SVs were added to the RAG after application of the size '
             'filter.'.format(len(filtered_ids)))
    nx.write_edgelist(rag_sub_g, global_params.config.init_rag_path)

    exec_init.run_create_rag()
    ftimer.stop()

    if global_params.config.prior_glia_removal:
        log.info('Step 2.5/9 - Glia separation')
        ftimer.start('Glia separation')
        if not global_params.config.use_point_models:
            exec_render.run_glia_rendering()
            exec_inference.run_glia_prediction()
        else:
            exec_inference.run_glia_prediction_pts()
        exec_inference.run_glia_splitting()
        ftimer.stop()

    log.info('Step 3/9 - Creating SuperSegmentationDataset')
    ftimer.start('SSD generation')
    exec_init.run_create_neuron_ssd(cube_of_interest_bb=cube_of_interest_bb)
    ftimer.stop()

    if not (global_params.config.use_onthefly_views or global_params.config.use_point_models):
        log.info('Step 3.5/9 - Neuron rendering')
        ftimer.start('Neuron rendering')
        exec_render.run_neuron_rendering()
        ftimer.stop()

    log.info('Step 4/9 - Synapse detection')
    ftimer.start('Synapse detection')
    exec_syns.run_syn_generation(chunk_size=chunk_size, n_folders_fs=n_folders_fs_sc,
                                 cube_of_interest_bb=cube_of_interest_bb)
    ftimer.stop()

    log.info('Step 5/9 - Axon prediction')
    ftimer.start('Axon prediction')
    exec_inference.run_semsegaxoness_prediction()
    ftimer.stop()

    log.info('Step 6/9 - Spine prediction')
    ftimer.start('Spine prediction')
    exec_inference.run_semsegspiness_prediction()
    exec_syns.run_spinehead_volume_calc()
    ftimer.stop()

    log.info('Step 7/9 - Morphology extraction')
    ftimer.start('Morphology extraction')
    exec_inference.run_morphology_embedding()
    ftimer.stop()

    log.info('Step 8/9 - Celltype analysis')
    ftimer.start('Celltype analysis')
    exec_inference.run_celltype_prediction()
    ftimer.stop()

    log.info('Step 9/9 - Matrix export')
    ftimer.start('Matrix export')
    exec_syns.run_matrix_export()
    ftimer.stop()

    time_summary_str = ftimer.prepare_report(experiment_name)
    log.info(time_summary_str)
    log.info('Setting up flask server for inspection. Annotated cell reconstructions and wiring can be analyzed via '
             'the KNOSSOS-SyConn plugin at `SyConn/scripts/kplugin/syconn_knossos_viewer.py`.')
    os.system(f'syconn.server --working_dir={working_dir} --port=10001')

