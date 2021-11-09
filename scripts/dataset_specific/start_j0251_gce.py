# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max-Planck-Institute of Neurobiology, Munich, Germany
# Authors: Philipp Schubert, Joergen Kornfeld

import os
import time
import re
import numpy as np
import networkx as nx
import shutil
from multiprocessing import Process

from syconn.proc.stats import FileTimer
from syconn.handler.config import generate_default_conf, initialize_logging
from syconn import global_params
from syconn.mp.batchjob_utils import batchjob_enabled, nodestates_slurm
from syconn.exec import exec_init, exec_syns, exec_render, exec_dense_prediction, exec_inference, exec_skeleton


if __name__ == '__main__':
    # ----------------- DEFAULT WORKING DIRECTORY ---------------------
    test_point_models = True
    test_view_models = True
    assert test_point_models or test_view_models
    experiment_name = 'j0251'
    scale = np.array([10, 10, 25])
    number_of_nodes = 24
    node_states = nodestates_slurm()
    node_state = next(
        iter(node_states.values()))
    exclude_nodes = []
    for nk in list(node_states.keys())[number_of_nodes:]:
        exclude_nodes.append(nk)
        del node_states[nk]
    # check cluster state
    assert number_of_nodes == np.sum([v['state'] == 'idle' for v in node_states.values()])
    ncores_per_node = node_state['cpus']
    mem_per_node = node_state['memory']
    ngpus_per_node = node_state['gres']
    shape_j0251 = np.array([27119, 27350, 15494])
    # 10.5* for 4.9, *9 for 3.13, *7.5 for 1.81, *6 for 0.927, *4.5 for 0.391, *3 for 0.115 TVx
    cube_size = (np.array([2048, 2048, 1024]) * 9).astype(np.int32)
    # all for 10 TVx
    cube_offset = ((shape_j0251 - cube_size) // 2).astype(np.int32)
    cube_of_interest_bb = np.array([cube_offset, cube_offset + cube_size], dtype=np.int32)
    # cube_of_interest_bb = None  # process the entire cube!
    prior_astrocyte_removal = True
    use_point_models = True
    key_val_pairs_conf = [
        ('min_cc_size_ssv', 5000),  # minimum bounding box diagonal of cell (fragments) in nm
        ('glia', {'prior_astrocyte_removal': prior_astrocyte_removal}),
        ('pyopengl_platform', 'egl'),
        ('batch_proc_system', 'SLURM'),
        ('ncores_per_node', ncores_per_node),
        ('ngpus_per_node', 2),
        ('nnodes_total', number_of_nodes),
        ('mem_per_node', mem_per_node),
        ('use_point_models', use_point_models),
        ('skeleton', {'use_kimimaro': True}),
        ('meshes', {'use_new_meshing': True}),
        ('views', {'use_new_renderings_locs': True,
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
         ),
        ('cube_of_interest_bb', cube_of_interest_bb.tolist()),
        ('slurm', {'exclude_nodes': exclude_nodes})
    ]
    chunk_size = (512, 512, 256)
    if cube_size[0] <= 2048:
        chunk_size = (256, 256, 256)
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
    working_dir = f"/glusterfs/example_runs/j0251_off{'_'.join(map(str, cube_offset))}_size" \
                  f"{'_'.join(map(str, cube_size))}_{number_of_nodes}nodes"
    log = initialize_logging(experiment_name, log_dir=working_dir + '/logs/')
    ftimer = FileTimer(working_dir + '/.timing.pkl')
    shutil.copy(os.path.abspath(__file__), f'{working_dir}/logs/')

    log.info('Step 0/10 - Preparation')
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
    if not os.path.exists(f'{working_dir}/knossosdatasets/myelin'):
        assert os.path.exists('/mnt/j0251_data/myelin')
        os.makedirs(f'{working_dir}/knossosdatasets/', exist_ok=True)
        os.symlink('/mnt/j0251_data/myelin', f'{working_dir}/knossosdatasets/myelin')
        assert os.path.exists(f'{working_dir}/knossosdatasets/myelin')
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
    #
    # log.info('Step 1/10 - Predicting sub-cellular structures')
    # ftimer.start('Myelin prediction')
    # # myelin is not needed before `run_create_neuron_ssd`
    # exec_dense_prediction.predict_myelin(raw_kd_path, cube_of_interest=cube_of_interest_bb)
    # ftimer.stop()

    log.info('Step 2/10 - Creating SegmentationDatasets (incl. SV meshes)')
    ftimer.start('SD generation')
    exec_init.init_cell_subcell_sds(chunk_size=chunk_size, n_folders_fs_sc=n_folders_fs_sc,
                                    n_folders_fs=n_folders_fs, cube_of_interest_bb=cube_of_interest_bb,
                                    load_cellorganelles_from_kd_overlaycubes=True,
                                    transf_func_kd_overlay=cellorganelle_transf_funcs)

    # generate flattened RAG
    from syconn.reps.segmentation import SegmentationDataset
    sd = SegmentationDataset(obj_type="sv", working_dir=global_params.config.working_dir)
    rag_sub_g = nx.Graph()
    # add SV IDs to graph via self-edges
    mesh_bb = sd.load_numpy_data('mesh_bb')  # N, 2, 3
    mesh_bb = np.linalg.norm(mesh_bb[:, 1] - mesh_bb[:, 0], axis=1)
    filtered_ids = sd.ids[mesh_bb > global_params.config['min_cc_size_ssv']]
    rag_sub_g.add_edges_from([[el, el] for el in sd.ids])
    log.info('{} SVs were added to the RAG after applying the size '
             'filter.'.format(len(filtered_ids)))
    nx.write_edgelist(rag_sub_g, global_params.config.init_svgraph_path)

    exec_init.run_create_rag()
    ftimer.stop()
    #
    log.info('Step 3/10 - Astrocyte separation')
    if global_params.config.prior_astrocyte_removal:
        if test_view_models:
            global_params.config['use_point_models'] = False
            global_params.config.write_config()
            time.sleep(10)  # wait for changes to apply
            ftimer.start('Glia prediction (multi-views)')
            # if not global_params.config.use_point_models:
            exec_render.run_astrocyte_rendering()
            exec_inference.run_astrocyte_prediction()
            ftimer.stop()

        # else:
        if test_point_models:
            global_params.config['use_point_models'] = True
            global_params.config.write_config()
            time.sleep(10)  # wait for changes to apply
            ftimer.start('Glia prediction (points)')
            exec_inference.run_astrocyte_prediction_pts()
            ftimer.stop()

        ftimer.start('Glia splitting')
        exec_inference.run_astrocyte_splitting()
        ftimer.stop()

    log.info('Step 4/10 - Creating SuperSegmentationDataset')
    ftimer.start('SSD generation')
    exec_init.run_create_neuron_ssd()
    ftimer.stop()

    def start_skel_gen():
        log.info('Step 6/10 - Skeleton generation')
        ftimer.start('Skeleton generation')
        exec_skeleton.run_skeleton_generation(cube_of_interest_bb=cube_of_interest_bb)
        ftimer.stop()

    def start_neuron_rendering():
        if not (global_params.config.use_onthefly_views or global_params.config.use_point_models):
            log.info('Step 6.5/10 - Neuron rendering')
            ftimer.start('Neuron rendering')
            exec_render.run_neuron_rendering()
            ftimer.stop()

    def start_syn_gen():
        log.info('Step 5/10 - Synapse detection')
        ftimer.start('Synapse detection')
        exec_syns.run_syn_generation(chunk_size=chunk_size, n_folders_fs=n_folders_fs_sc,
                                     cube_of_interest_bb=cube_of_interest_bb)
        ftimer.stop()

    # skeleton and synapse generation and rendering have independent dependencies and target storage
    assert global_params.config.use_onthefly_views, ('"use_onthefly_views" must be True to enable parallel '
                                                     'execution of skel and syn generation')
    procs = []
    for func in [start_syn_gen, start_skel_gen, start_neuron_rendering]:
        if 1:  # not batchjob_enabled():  # do not use parallel processing for timings
            func()
            continue
        p = Process(target=func)
        p.start()
        procs.append(p)
        time.sleep(10)
    for p in procs:  # procs is empty list, if batch jobs are disabled.
        p.join()
        if p.exitcode != 0:
            raise Exception(f'Worker {p.name} stopped unexpectedly with exit '
                            f'code {p.exitcode}.')
        p.close()

    log.info('Step 7/10 - Compartment prediction')
    if test_view_models:
        global_params.config['use_point_models'] = False
        global_params.config.write_config()
        time.sleep(10)  # wait for changes to apply
        ftimer.start('Compartment predictions (multi-views)')
        exec_inference.run_semsegaxoness_prediction()
        exec_inference.run_semsegspiness_prediction()
        ftimer.stop()
    # if not global_params.config.use_point_models:
    if test_point_models:
        ftimer.start('Compartment predictions (points)')
        global_params.config['use_point_models'] = True
        global_params.config.write_config()
        exec_inference.run_semsegaxoness_prediction()
        ftimer.stop()

    # TODO: this step can be launched in parallel with the morphology extraction!
    ftimer.start('Spine head calculation')
    exec_syns.run_spinehead_volume_calc()
    ftimer.stop()

    log.info('Step 8/10 - Morphology extraction')
    if test_view_models:
        global_params.config['use_point_models'] = False
        global_params.config.write_config()
        time.sleep(10)  # wait for changes to apply
        ftimer.start('Morphology extraction (multi-views)')
        exec_inference.run_morphology_embedding()
        ftimer.stop()
    if test_point_models:
        global_params.config['use_point_models'] = True
        global_params.config.write_config()
        time.sleep(10)  # wait for changes to apply
        ftimer.start('Morphology extraction (points)')
        exec_inference.run_morphology_embedding()
        ftimer.stop()

    log.info('Step 9/10 - Celltype analysis')
    if test_view_models:
        global_params.config['use_point_models'] = False
        global_params.config.write_config()
        time.sleep(10)  # wait for changes to apply
        ftimer.start('Celltype analysis (multi-views)')
        exec_inference.run_celltype_prediction()
        ftimer.stop()
    if test_point_models:
        global_params.config['use_point_models'] = True
        global_params.config.write_config()
        time.sleep(10)  # wait for changes to apply
        ftimer.start('Celltype analysis (points)')
        exec_inference.run_celltype_prediction()
        ftimer.stop()

    log.info('Step 10/10 - Matrix export')
    ftimer.start('Matrix export')
    exec_syns.run_matrix_export()
    ftimer.stop()

    time_summary_str = ftimer.prepare_report()
    log.info(time_summary_str)

    # remove unimportant stuff for timings
    print('Deleting data that is not required anymore.')
    import glob, tqdm
    if test_view_models:
        for fname in tqdm.tqdm(glob.glob(working_dir + '/sv_0/so_storage*/*'), desc='SVs'):
            shutil.rmtree(fname)
    tmp_del_dir = f'{working_dir}/DEL_cube_size{"_".join(map(str, cube_size))}_{number_of_nodes}nodes/'
    os.makedirs(tmp_del_dir)
    for d in tqdm.tqdm(['models', 'vc_0', 'sj_0', 'syn_ssv_0', 'syn_0', 'ssv_0', 'mi_0', 'cs_0',
                        'knossosdatasets', 'SLURM', 'tmp', 'chunkdatasets', 'ssv_gliaremoval'], desc='Folders'):
        shutil.move(f'{working_dir}/{d}', tmp_del_dir)
    shutil.move(tmp_del_dir, f'{working_dir}/../')
