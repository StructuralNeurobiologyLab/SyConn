# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max-Planck-Institute of Neurobiology, Munich, Germany
# Authors: Philipp Schubert

import os
import glob
import shutil
import sys
import numpy as np

from syconn import global_params
from syconn.handler.config import generate_default_conf, initialize_logging
from syconn.proc.stats import FileTimer

from knossos_utils import knossosdataset


def test_full_run():
    example_cube_id = 1
    working_dir = "~/SyConn/tests/example_cube{}/".format(example_cube_id)
    example_wd = os.path.expanduser(working_dir) + "/"
    shutil.rmtree(example_wd, ignore_errors=True)
    # set up basic parameter, log, working directory and config file
    log = initialize_logging('example_run', log_dir=example_wd + '/logs/')
    experiment_name = 'j0126_example'
    scale = np.array([10, 10, 20])
    prior_glia_removal = True
    use_myelin = True
    key_val_pairs_conf = [
        ('glia', {'prior_glia_removal': prior_glia_removal}),
        ('pyopengl_platform', 'egl'),  # 'osmesa' or 'egl'
        ('batch_proc_system', None),  # None, 'SLURM' or 'QSUB'
        ('ncores_per_node', 20),
        ('ngpus_per_node', 2),
        ('nnodes_total', 1),
        ('log_level', 'INFO'),
        ('cell_objects', {
          })
    ]
    if example_cube_id == 1:
        chunk_size = (256, 256, 128)
    elif example_cube_id == 2:
        chunk_size = (256, 256, 256)
    else:
        chunk_size = (512, 512, 256)
    n_folders_fs = 100
    n_folders_fs_sc = 100
    for curr_dir in [os.path.dirname(os.path.realpath(__file__)) + '/',
                     os.path.abspath(os.path.curdir) + '/',
                     os.path.expanduser('~/SyConn/')]:
        h5_dir = curr_dir + '/data{}/'.format(example_cube_id)
        if os.path.isdir(h5_dir):
            break
    if not os.path.isdir(h5_dir):
        raise FileNotFoundError(f'Example data folder could not be found'
                                f' at "{curr_dir}".')
    if not os.path.isfile(h5_dir + 'seg.h5') or len(glob.glob(h5_dir + '*.h5')) != 7\
            or not os.path.isfile(h5_dir + 'neuron_rag.bz2'):
        raise FileNotFoundError(f'Incomplete example data in folder "{h5_dir}".')
    if not (sys.version_info[0] == 3 and sys.version_info[1] >= 6):
        log.critical('Python version <3.6. This is untested!')

    generate_default_conf(example_wd, scale, key_value_pairs=key_val_pairs_conf, force_overwrite=False)

    if global_params.config.working_dir is not None and \
            os.path.normpath(global_params.config.working_dir) != os.path.normpath(example_wd):
        os.environ['syconn_wd'] = example_wd

    global_params.wd = example_wd
    from syconn.reps.super_segmentation import SuperSegmentationDataset
    assert os.path.normpath(SuperSegmentationDataset().config.working_dir) == os.path.normpath(example_wd)
    os.makedirs(example_wd, exist_ok=True)

    # keep imports here to guarantee the correct usage of pyopengl platform if batch processing
    # system is None
    from syconn.exec import exec_init, exec_syns, exec_render, exec_dense_prediction, exec_inference, exec_skeleton
    from syconn.handler.compression import load_from_h5py

    # PREPARE TOY DATA
    log.info(f'Step 0/9 - Preparation')
    ftimer = FileTimer(example_wd + '/.timing.pkl')
    ftimer.start('Preparation')

    # copy models to working directory
    if os.path.isdir(curr_dir + '/models/') and not os.path.isdir(example_wd + '/models/'):
        shutil.copytree(curr_dir + '/models', example_wd + '/models/')
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

    tmp = load_from_h5py(h5_dir + 'sj.h5', hdf5_names=['sj'])[0]
    offset = np.array([0, 0, 0])
    bd = np.array(tmp.shape)
    del tmp

    # INITIALIZE DATA
    if not os.path.isdir(global_params.config.kd_sj_path):
        kd = knossosdataset.KnossosDataset()
        kd.initialize_from_matrix(global_params.config.kd_seg_path, scale, experiment_name,
                                  offset=offset, boundary=bd, fast_downsampling=True,
                                  data_path=h5_dir + 'raw.h5', mags=[1, 2, 4], hdf5_names=['raw'])

        seg_d = load_from_h5py(h5_dir + 'seg.h5', hdf5_names=['seg'])[0].swapaxes(0, 2)  # xyz -> zyx
        kd.save_seg(offset=offset, mags=[1, 2, 4], data=seg_d, data_mag=1)
        del kd, seg_d
        kd_sym = knossosdataset.KnossosDataset()
        kd_sym.initialize_from_matrix(global_params.config.kd_sym_path, scale, experiment_name,
                                      offset=offset, boundary=bd, fast_downsampling=True,
                                      data_path=h5_dir + 'sym.h5', mags=[1, 2], hdf5_names=['sym'])
        del kd_sym
        kd_asym = knossosdataset.KnossosDataset()
        kd_asym.initialize_from_matrix(global_params.config.kd_asym_path, scale,
                                       experiment_name, offset=offset, boundary=bd,
                                       fast_downsampling=True, data_path=h5_dir + 'asym.h5',
                                       mags=[1, 2], hdf5_names=['asym'])
        del kd_asym
        kd_mi = knossosdataset.KnossosDataset()
        kd_mi.initialize_from_matrix(global_params.config.kd_mi_path, scale, experiment_name,
                                     offset=offset, boundary=bd, fast_downsampling=True,
                                     data_path=h5_dir + 'mi.h5', mags=[1, 2], hdf5_names=['mi'])
        del kd_mi
        kd_vc = knossosdataset.KnossosDataset()
        kd_vc.initialize_from_matrix(global_params.config.kd_vc_path, scale, experiment_name,
                                     offset=offset, boundary=bd, fast_downsampling=True,
                                     data_path=h5_dir + 'vc.h5', mags=[1, 2], hdf5_names=['vc'])
        del kd_vc
        kd_sj = knossosdataset.KnossosDataset()
        kd_sj.initialize_from_matrix(global_params.config.kd_sj_path, scale, experiment_name,
                                     offset=offset, boundary=bd, fast_downsampling=True,
                                     data_path=h5_dir + 'sj.h5', mags=[1, 2], hdf5_names=['sj'])
        del kd_sj
    ftimer.stop()

    log.info(f'Finished example cube initialization (shape: {bd}). Starting SyConn pipeline.')
    log.info('Example data will be processed in "{}".'.format(example_wd))

    # START SyConn
    log.info('Step 1/9 - Predicting sub-cellular structures')
    ftimer.start('Dense predictions')
    if use_myelin:
        exec_dense_prediction.predict_myelin()
    ftimer.stop()

    log.info('Step 2/9 - Creating SegmentationDatasets (incl. SV meshes)')
    ftimer.start('SD generation')
    exec_init.init_cell_subcell_sds(chunk_size=chunk_size, n_folders_fs=n_folders_fs,
                                    n_folders_fs_sc=n_folders_fs_sc)
    exec_init.run_create_rag()
    ftimer.stop()

    log.info('Step 3/9 - Glia separation')
    if global_params.config.prior_glia_removal:
        ftimer.start('Glia separation')
        if not global_params.config.use_point_models:
            exec_render.run_glia_rendering()
            exec_inference.run_glia_prediction()
        else:
            exec_inference.run_glia_prediction_pts()
        exec_inference.run_glia_splitting()
        ftimer.stop()
    else:
        log.info('Glia separation disabled. Skipping.')

    log.info('Step 4/9 - Creating SuperSegmentationDataset')
    ftimer.start('SSD generation')
    exec_init.run_create_neuron_ssd()
    ftimer.stop()

    log.info('Step 5/10 - Creating SuperSegmentationDataset')
    ftimer.start('Skeleton generation')
    exec_skeleton.run_skeleton_generation(map_myelin=use_myelin)
    ftimer.stop()

    if not (global_params.config.use_onthefly_views or global_params.config.use_point_models):
        log.info('Step 5.5/9 - Neuron rendering')
        ftimer.start('Neuron rendering')
        exec_render.run_neuron_rendering()
        ftimer.stop()

    log.info('Step 6/9 - Synapse detection')
    ftimer.start('Synapse detection')
    exec_syns.run_syn_generation(chunk_size=chunk_size, n_folders_fs=n_folders_fs_sc)
    ftimer.stop()

    log.info('Step 7/9 - Compartment prediction')
    ftimer.start('Compartment predictions')
    exec_inference.run_semsegaxoness_prediction()
    if not global_params.config.use_point_models:
        exec_inference.run_semsegspiness_prediction()
    exec_syns.run_spinehead_volume_calc()
    ftimer.stop()

    log.info('Step 8/9 - Morphology extraction')
    ftimer.start('Morphology extraction')
    exec_inference.run_morphology_embedding()
    ftimer.stop()

    log.info('Step 9/9 - Celltype analysis')
    ftimer.start('Celltype analysis')
    exec_inference.run_celltype_prediction()
    ftimer.stop()

    log.info('Matrix export')
    ftimer.start('Matrix export')
    exec_syns.run_matrix_export()
    ftimer.stop()

    time_summary_str = ftimer.prepare_report()
    log.info(time_summary_str)

    if os.environ.get('syconn_wd') is not None:
        del os.environ['syconn_wd']


if __name__ == '__main__':
    test_full_run()
