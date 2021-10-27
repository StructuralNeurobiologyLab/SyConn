# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max-Planck-Institute of Neurobiology, Munich, Germany
# Authors: Philipp Schubert, Joergen Kornfeld

import os
import time
import argparse
import numpy as np
import yaml

from syconn import global_params
from syconn.handler.config import generate_default_conf, initialize_logging
from syconn.proc.stats import FileTimer


def main():
    # parse arguments
    parser = argparse.ArgumentParser(description='SyConn example run')
    parser.add_argument('--working_dir', type=str, required=True,
                        help='Working directory of SyConn')
    parser.add_argument('--log_level', type=str, default='INFO',
                        help='Level of logging (INFO, DEBUG).')
    parser.add_argument('--steps', type=str, default='all',
                        help='Which steps to run: comma-separated list of ints (1-13) or "all"')
    parser.add_argument('--skip_steps', type=str, default='none',
                        help='Which steps to skip: comma-separated list of ints (1-13) or "none"')
    parser.add_argument('--overwrite', dest='overwrite', action='store_true',
                        help='Overwrite generated data.')
    parser.add_argument('--params', type=str, required=True,
                        help='YAML file with config for this run.')
    parser.set_defaults(overwrite=False)
    args = parser.parse_args()
    log_level = args.log_level

    if args.skip_steps == 'none':
        skip_steps = set()
    else:
        skip_steps = set(int(xx) for xx in args.skip_steps.split(','))
    if args.steps == 'all':
        steps = set(range(1, 14))
    else:
        steps = set(int(xx) for xx in args.steps.split(','))

    steps_todo = sorted(steps-skip_steps)

    steps_names = {
        1: 'dense_predictions',
        2: 'sd_generation',
        3: 'astrocyte_separation',
        4: 'ssd_generation',
        5: 'skeleton_generation',
        6: 'synapse_detection',
        7: 'neuron_rendering',
        8: 'contact_detection',
        9: 'compartment_predictions',
        10: 'morphology_extraction',
        11: 'celltype_analysis',
        12: 'matrix_export',
        13: 'start_server', }

    todo = [steps_names[xx] for xx in steps_todo]
    todo_str = '\n\t'.join(todo)
    print(f'Will run the following steps (specified: do: "{args.steps}", skip: "{args.skip_steps}"):\n\n\t{todo_str}')

    # ----------------- DEFAULT WORKING DIRECTORY ---------------------
    working_dir = args.working_dir
    experiment_name = 'fw363'
    scale = np.array([14, 14, 25])

    chunk_size = (512, 512, 512)
    n_folders_fs = 10000
    n_folders_fs_sc = 10000
    ncores_skelgen = 2
    cube_size_skelgen = np.array([500, 500, 280])

    with open(args.params, 'r') as fp:
        print(f'Loading yml from {args.params}')
        y = yaml.safe_load(fp)
        key_val_pairs_conf = [(xx, yy) for xx, yy in y.items()]

    # ----------------- DATA DIRECTORY ---------------------
    seg_kd_path = '/axon/scratch/fsvara2/fw363_seg/fw363_seg_neurites/'
    vc_kd_path = '/axon/scratch/fsvara2/fw363_seg/fw363_seg_syns/'
    sj_kd_path = '/axon/scratch/fsvara2/fw363_seg/fw363_seg_syns/'
    mi_kd_path = '/axon/scratch/fsvara2/fw363_seg/fw363_seg_syns/'
    svgraph = '/axon/scratch/fsvara2/fw363_seg/vertices_edges_strfragments_undirected-with-action_db-2021-08-06-prefiltered-2021-09-08_edgelist.csv'

    kd_asym_path = None
    kd_sym_path = None
    syntype_avail = False

    # The transform functions will be applied when loading the segmentation data of cell organelles
    # in order to convert them into binary fore- and background
    # currently using `dill` package to support lambda expressions, a weak feature. Make
    #  sure all dependencies within the lambda expressions are imported in
    #  `batchjob_object_segmentation.py` (here: numpy)
    cellorganelle_transf_funcs = dict(vc=lambda x: (x == 2).astype(np.uint8),   # vesicle cloud
                                      sj=lambda x: (x == 1).astype(np.uint8),   # synaptic cleft
                                      mi=lambda x: (x == 2).astype(np.uint8))
    load_cellorganelles_from_kd_overlaycubes = True

    # Preparing data
    # --------------------------------------------------------------------------
    # Setup working directory and logging
    log = initialize_logging(experiment_name, log_dir=working_dir + '/logs/')
    ftimer = FileTimer(working_dir + '/.timing.pkl')
    ftimer.start('Preparation')

    # Preparing config
    # currently this is were SyConn looks for the neuron rag

    generate_default_conf(working_dir, scale, syntype_avail=syntype_avail, kd_seg=seg_kd_path, kd_mi=mi_kd_path,
                          kd_vc=vc_kd_path, kd_sj=sj_kd_path, kd_sym=kd_sym_path, kd_asym=kd_asym_path,
                          key_value_pairs=key_val_pairs_conf, init_svgraph_path=svgraph, force_overwrite=True)

    global_params.wd = working_dir
    os.makedirs(global_params.config.temp_path, exist_ok=True)
    start = time.time()

    # check model existence
    for mpath_key in ['mpath_spiness', 'mpath_syn_rfc', 'mpath_celltype_e3',
                      'mpath_axonsem', 'mpath_glia_e3', 'mpath_myelin',
                      'mpath_tnet']:
        mpath = getattr(global_params.config, mpath_key)
        if not (os.path.isfile(mpath) or os.path.isdir(mpath)):
            raise ValueError('Could not find model "{}". Make sure to copy the'
                             ' "models" folder into the current working '
                             'directory "{}".'.format(mpath, working_dir))
    ftimer.stop()

    # keep imports here to guarantee the correct usage of pyopengl platform if batch processing
    # system is None
    from syconn.exec import exec_init, exec_syns, exec_render, exec_dense_prediction, exec_inference, exec_skeleton

    # Start SyConn
    # --------------------------------------------------------------------------
    log.info('Starting SyConn pipeline for data cube (shape: {}).'.format(ftimer.dataset_shape))
    log.critical('Working directory is set to "{}".'.format(working_dir))

    if 'sd_generation' in todo:
        log.info('Step 2/12 - Creating SegmentationDatasets (incl. SV meshes)')
        ftimer.start('SD generation')
        exec_init.init_cell_subcell_sds(chunk_size=chunk_size, n_folders_fs=n_folders_fs,
                                        n_folders_fs_sc=n_folders_fs_sc, overwrite=args.overwrite,
                                        load_cellorganelles_from_kd_overlaycubes=load_cellorganelles_from_kd_overlaycubes,
                                        transf_func_kd_overlay=cellorganelle_transf_funcs,
                                        max_n_jobs=global_params.config.ncore_total * 4)
        exec_init.run_create_rag(is_prefiltered=True)
        ftimer.stop()

    if 'astrocyte_separation' in todo:
        log.info('Step 3/12 - Astrocyte separation')
        if global_params.config.prior_astrocyte_removal:
            ftimer.start('Astrocyte separation')
            if not global_params.config.use_point_models:
                exec_render.run_astrocyte_rendering()
                exec_inference.run_astrocyte_prediction()
            else:
                exec_inference.run_astrocyte_prediction_pts()
            exec_inference.run_astrocyte_splitting()
            ftimer.stop()
        else:
            log.info('Astrocyte separation disabled. Skipping.')

    if 'ssd_generation' in todo:
        log.info('Step 4/12 - Creating SuperSegmentationDataset')
        ftimer.start('SSD generation')
        exec_init.run_create_neuron_ssd(overwrite=args.overwrite, ncores_per_job=4)
        ftimer.stop()

    if 'skeleton_generation' in todo:
        log.info('Step 5/12 - Skeleton generation')
        ftimer.start('Skeleton generation')
        exec_skeleton.run_skeleton_generation(ncores_skelgen=ncores_skelgen, cube_size=cube_size_skelgen)
        ftimer.stop()

    if 'synapse_detection' in todo:
        log.info('Step 6/12 - Synapse detection')
        ftimer.start('Synapse detection')
        if cellorganelle_transf_funcs is not None:
            transf = cellorganelle_transf_funcs['sj']
        else:
            transf = None
        exec_syns.run_syn_generation(
            chunk_size=chunk_size, n_folders_fs=n_folders_fs_sc, overwrite=args.overwrite,transf_func_sj_seg=transf)
        ftimer.stop()

    if 'neuron_rendering' in todo:
        if not (global_params.config.use_onthefly_views or global_params.config.use_point_models):
            log.info('Step 7/12 - Neuron rendering')
            ftimer.start('Neuron rendering')
            exec_render.run_neuron_rendering()
            ftimer.stop()

    if 'contact_detection' in todo:
        log.info('Step 8/12 - Contact detection')
        ftimer.start('Contact detection')
        if global_params.config['generate_cs_ssv']:
            exec_syns.run_cs_ssv_generation(n_folders_fs=n_folders_fs_sc, overwrite=args.overwrite)
        else:
            log.info('Cell-cell contact detection ("cs_ssv" objects) disabled. Skipping.')
        ftimer.stop()

    if 'compartment_predictions' in todo:
        log.info('Step 9/12 - Compartment prediction')
        ftimer.start('Compartment predictions')
        exec_inference.run_semsegaxoness_prediction()
        if not global_params.config.use_point_models:
            exec_inference.run_semsegspiness_prediction()
        exec_syns.run_spinehead_volume_calc()
        ftimer.stop()

    if 'morphology_extraction' in todo:
        log.info('Step 10/12 - Morphology extraction')
        ftimer.start('Morphology extraction')
        exec_inference.run_morphology_embedding()
        ftimer.stop()

    if 'celltype_analysis' in todo:
        log.info('Step 11/12 - Celltype analysis')
        ftimer.start('Celltype analysis')
        exec_inference.run_celltype_prediction()
        ftimer.stop()

    if 'matrix_export' in todo:
        log.info('Step 12/12 - Matrix export')
        ftimer.start('Matrix export')
        exec_syns.run_matrix_export()
        ftimer.stop()

    time_summary_str = ftimer.prepare_report()
    log.info(time_summary_str)

    if 'start_server' in todo:
        log.info('Setting up flask server for inspection. Annotated cell reconstructions and wiring '
                 'can be analyzed via the KNOSSOS-SyConn plugin at '
                 '`SyConn/scripts/kplugin/syconn_knossos_viewer.py`.')
        os.system(f'syconn.server --working_dir={args.working_dir} --port=10001')


if __name__ == '__main__':
    main()
