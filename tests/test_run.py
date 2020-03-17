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
import glob
import shutil
import sys
import time

from syconn import global_params
from syconn.handler.config import generate_default_conf, initialize_logging


def test_full_run():
    example_cube_id = 1
    working_dir = "~/SyConn/tests/example_cube{}/".format(example_cube_id)
    example_wd = os.path.expanduser(working_dir) + "/"

    # set up basic parameter, log, working directory and config file
    log = initialize_logging('example_run', log_dir=example_wd + '/logs/')
    experiment_name = 'j0126_example'
    scale = np.array([10, 10, 20])
    prior_glia_removal = True
    key_val_pairs_conf = [
        ('glia', {'prior_glia_removal': prior_glia_removal}),
        ('pyopengl_platform', 'egl'),  # 'osmesa' or 'egl'
        ('batch_proc_system', None),  # None, 'SLURM' or 'QSUB'
        ('ncores_per_node', 20),
        ('ngpus_per_node', 2),
        ('nnodes_total', 1),
    ]
    chunk_size = (256, 256, 256)
    n_folders_fs = 1000
    n_folders_fs_sc = 1000
    curr_dir = os.path.expanduser('~/SyConn//')
    h5_dir = curr_dir + '/data{}/'.format(example_cube_id)
    kzip_p = curr_dir + '/example_cube{}.k.zip'.format(example_cube_id)

    if not (sys.version_info[0] == 3 and sys.version_info[1] >= 6):
        log.critical('Python version <3.6. This is untested!')

    generate_default_conf(example_wd, scale,
                          key_value_pairs=key_val_pairs_conf,
                          force_overwrite=True)

    if global_params.wd is not None:
        log.critical('Example run started. Working directory was overwritten and set'
                     ' to "{}".'.format(example_wd))
    global_params.wd = example_wd
    os.makedirs(global_params.config.temp_path, exist_ok=True)

    # keep imports here to guarantee the correct usage of pyopengl platform if batch processing
    # system is None
    from syconn.exec import exec_init, exec_syns, exec_multiview, exec_dense_prediction
    from syconn.handler.prediction import parse_movement_area_from_zip
    from syconn.handler.compression import load_from_h5py

    # PREPARE TOY DATA
    log.info('Step 0/8 - Preparation')

    time_stamps = [time.time()]
    step_idents = ['t-0']

    # copy models to working directory
    if os.path.isdir(curr_dir + '/models/') and not os.path.isdir(example_wd + '/models/'):
        shutil.copytree(curr_dir + '/models', example_wd + '/models/')

    if not os.path.isfile(kzip_p) or not os.path.isdir(h5_dir):
        raise FileNotFoundError('Example data could not be found at "{}".'.format(curr_dir))
    if not os.path.isfile(h5_dir + 'seg.h5') or len(glob.glob(h5_dir + '*.h5')) != 7\
            or not os.path.isfile(h5_dir + 'neuron_rag.bz2'):
        raise FileNotFoundError('Example data could not be found at "{}".'.format(h5_dir))

    os.makedirs(example_wd + '/glia/', exist_ok=True)

    # check model existence
    for mpath_key in ['mpath_spiness', 'mpath_syn_rfc', 'mpath_celltype_e3',
                      'mpath_axonsem', 'mpath_glia_e3', 'mpath_myelin',
                      'mpath_tnet']:
        mpath = getattr(global_params.config, mpath_key)
        if not (os.path.isfile(mpath) or os.path.isdir(mpath)):
            raise ValueError('Could not find model "{}". Make sure to copy the'
                             ' "models" folder into the current working '
                             'directory "{}".'.format(mpath, example_wd))

    if not prior_glia_removal:
        shutil.copy(h5_dir + "/neuron_rag.bz2", global_params.config.init_rag_path)
    else:
        shutil.copy(h5_dir + "/rag.bz2", global_params.config.init_rag_path)

    # TODO: get from h5 file
    bb = parse_movement_area_from_zip(kzip_p)
    offset = np.array([0, 0, 0])
    bd = bb[1] - bb[0]

    # INITIALIZE DATA
    # TODO: switch to streaming confs instead of h5 files
    kd = knossosdataset.KnossosDataset()
    kd.initialize_from_matrix(global_params.config.kd_seg_path, scale, experiment_name,
                              offset=offset, boundary=bd, fast_downsampling=True,
                              data_path=h5_dir + 'raw.h5', mags=[1, 2, 4], hdf5_names=['raw'],
                              force_overwrite=True)

    seg_d = load_from_h5py(h5_dir + 'seg.h5', hdf5_names=['seg'])[0]
    kd.from_matrix_to_cubes(offset, mags=[1, 2, 4], data=seg_d,
                            fast_downsampling=True, as_raw=False)

    kd_mi = knossosdataset.KnossosDataset()
    kd_mi.initialize_from_matrix(global_params.config.kd_mi_path, scale, experiment_name,
                                 offset=offset, boundary=bd, fast_downsampling=True,
                                 data_path=h5_dir + 'mi.h5', mags=[1, 2], hdf5_names=['mi'],
                                 force_overwrite=True)

    kd_vc = knossosdataset.KnossosDataset()
    kd_vc.initialize_from_matrix(global_params.config.kd_vc_path, scale, experiment_name,
                                 offset=offset, boundary=bd, fast_downsampling=True,
                                 data_path=h5_dir + 'vc.h5', mags=[1, 2], hdf5_names=['vc'],
                                 force_overwrite=True)

    kd_sj = knossosdataset.KnossosDataset()
    kd_sj.initialize_from_matrix(global_params.config.kd_sj_path, scale, experiment_name,
                                 offset=offset, boundary=bd, fast_downsampling=True,
                                 data_path=h5_dir + 'sj.h5', mags=[1, 2], hdf5_names=['sj'],
                                 force_overwrite=True)

    kd_sym = knossosdataset.KnossosDataset()
    kd_sym.initialize_from_matrix(global_params.config.kd_sym_path, scale, experiment_name,
                                  offset=offset, boundary=bd, fast_downsampling=True,
                                  data_path=h5_dir + 'sym.h5', mags=[1, 2], hdf5_names=['sym'],
                                  force_overwrite=True)

    kd_asym = knossosdataset.KnossosDataset()
    kd_asym.initialize_from_matrix(global_params.config.kd_asym_path, scale,
                                   experiment_name, offset=offset, boundary=bd,
                                   fast_downsampling=True, data_path=h5_dir + 'asym.h5',
                                   mags=[1, 2], hdf5_names=['asym'],
                                   force_overwrite=True)
    time_stamps.append(time.time())
    step_idents.append('Preparation')

    log.info('Finished example cube initialization (shape: {}). Starting'
             ' SyConn pipeline.'.format(bd))
    log.info('Example data will be processed in "{}".'.format(example_wd))

    # # START SyConn
    log.info('Step 0/8 - Predicting sub-cellular structures')
    # TODO: launch all inferences in parallel
    # exec_dense_prediction.predict_myelin()  # myelin is not needed before `run_create_neuron_ssd`
    # TODO: if performed, work-in paths of the resulting KDs to the config
    #  TODO: might require also require adaptions in init_cell_subcell_sds
    # exec_dense_prediction.predict_cellorganelles()
    # TODO: if performed, work-in paths of the resulting KDs to the config
    # exec_dense_prediction.predict_synapsetype()
    # time_stamps.append(time.time())
    # step_idents.append('Dense predictions')

    log.info('Step 1/8 - Creating SegmentationDatasets (incl. SV meshes)')
    exec_init.init_cell_subcell_sds(chunk_size=chunk_size, n_folders_fs=n_folders_fs,
                                    n_folders_fs_sc=n_folders_fs_sc)
    exec_init.run_create_rag()

    time_stamps.append(time.time())
    step_idents.append('SD generation')

    if global_params.config.prior_glia_removal:
        log.info('Step 1.5/8 - Glia separation')
        exec_multiview.run_glia_rendering()
        exec_multiview.run_glia_prediction(e3=True)
        exec_multiview.run_glia_splitting()
        time_stamps.append(time.time())
        step_idents.append('Glia separation')

    log.info('Step 2/8 - Creating SuperSegmentationDataset')
    exec_multiview.run_create_neuron_ssd()
    time_stamps.append(time.time())
    step_idents.append('SSD generation')

    # TODO: launch steps 3 and 4 in parallel
    log.info('Step 3/8 - Neuron rendering')
    exec_multiview.run_neuron_rendering()
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
    shutil.rmtree(example_wd)
