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
import argparse

from syconn.handler.prediction import parse_movement_area_from_zip
from syconn.handler.logger import initialize_logging
from syconn.handler.config import get_default_conf_str
from syconn import global_params
from syconn.exec import exec_init, exec_syns, exec_multiview


# TODO add materialize button and store current process in config.ini -> allows to resume interrupted processes
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SyConn example run')
    parser.add_argument('--working_dir', type=str, default='~/SyConn/example_cube/',
                        help='Working directory of SyConn')
    parser.add_argument('--example_cube', type=str, default='1',
                        help='Used toy data. Either "1" (400 x 400 x 600) or "2" ().')
    args = parser.parse_args()
    example_wd = os.path.expanduser(args.working_dir)
    example_cube_id = os.path.expanduser(args.example_cube)

    # PREPARE TOY DATA
    curr_dir = os.path.dirname(os.path.realpath(__file__)) + '/'
    if example_cube_id == "1":
        h5_dir = curr_dir + 'data1/'
        kzip_p = curr_dir + '/example_cube1.k.zip'
    elif example_cube_id == "2":
        h5_dir = curr_dir + 'data2/'
        kzip_p = curr_dir + '/example_cube2.k.zip'
    else:
        raise NotImplementedError('Please use valid example cube identifier ("1" or "2")!')
    if not os.path.isfile(h5_dir + 'seg.h5') or len(glob.glob(h5_dir + '*.h5')) != 7\
            or not os.path.isfile(h5_dir + 'neuron_rag.bz2'):
        raise ValueError('Example data could not be found at "{}".'.format(h5_dir))

    log = initialize_logging('example_run', log_dir=example_wd + '/logs/')
    bb = parse_movement_area_from_zip(kzip_p)
    offset = np.array([0, 0, 0])
    bd = bb[1] - bb[0]
    scale = np.array([10, 10, 20])
    experiment_name = 'j0126_example'

    # PREPARE CONFIG
    os.makedirs(example_wd + '/glia/', exist_ok=True)  # currently this is were SyConn looks for the neuron rag # TODO refactor
    shutil.copy(h5_dir + "/neuron_rag.bz2", example_wd + '/glia/neuron_rag.bz2')
    global_params.wd = example_wd
    log.critical('Example run started. Working directory is overwritten and set to "{}".'.format(example_wd))
    if not (sys.version_info[0] == 3 and sys.version_info[1] == 6):
        py36path = subprocess.check_output('source deactivate; source activate py36;'
                                           ' which python', shell=True).decode().replace('\n', '')
    else:
        py36path = ""
    config_str, configspec_str = get_default_conf_str(example_wd, py36path)
    with open(example_wd + 'config.ini', 'w') as f:
        f.write(config_str)
    with open(example_wd + 'configspec.ini', 'w') as f:
        f.write(configspec_str)

    for mpath_key in ['mpath_spiness', 'mpath_syn_rfc', 'mpath_celltype', 'mpath_axoness', 'mpath_glia']:
        mpath = getattr(global_params.config, mpath_key)
        if not (os.path.isfile(mpath) or os.path.isdir(mpath)):
            raise ValueError('Could not find model "{}". Make sure to copy the "models" folder into the'
                             ' current working directory "{}".'.format(mpath, example_wd))

    log.info('Finished example cube preparation {}. Starting SyConn pipeline.'.format(bd))
    log.info('Example data will be processed in "{}".'.format(example_wd))

    # INITIALIZE DATA
    # TODO: data too big to put into github repository, add alternative to pull data into h5_dir
    log.info('Step 0/8 - Preparation')
    kd = knossosdataset.KnossosDataset()
    # kd.set_channel('jpg')
    kd.initialize_from_matrix(global_params.config.kd_seg_path, scale, experiment_name,
                              offset=offset, boundary=bd, fast_downsampling=True,
                              data_path=h5_dir + 'raw.h5', mags=[1, 2], hdf5_names=['raw'])
    kd.from_matrix_to_cubes(offset, mags=[1, 2], data_path=h5_dir + 'seg.h5',
                            datatype=np.uint64, fast_downsampling=True,
                            as_raw=False, hdf5_names=['seg'])

    kd_mi = knossosdataset.KnossosDataset()
    # kd_mi.set_channel('jpg')
    kd_mi.initialize_from_matrix(global_params.config.kd_mi_path, scale, experiment_name,
                                 offset=offset, boundary=bd, fast_downsampling=True,
                                 data_path=h5_dir + 'mi.h5', mags=[1, 2], hdf5_names=['mi'])

    kd_vc = knossosdataset.KnossosDataset()
    # kd_vc.set_channel('jpg')
    kd_vc.initialize_from_matrix(global_params.config.kd_vc_path, scale, experiment_name,
                                 offset=offset, boundary=bd, fast_downsampling=True,
                                 data_path=h5_dir + 'vc.h5', mags=[1, 2], hdf5_names=['vc'])

    kd_sj = knossosdataset.KnossosDataset()
    # kd_sj.set_channel('jpg')
    kd_sj.initialize_from_matrix(global_params.config.kd_sj_path, scale, experiment_name,
                                 offset=offset, boundary=bd, fast_downsampling=True,
                                 data_path=h5_dir + 'sj.h5', mags=[1, 2], hdf5_names=['sj'])

    kd_sym = knossosdataset.KnossosDataset()
    # kd_sym.set_channel('jpg')
    kd_sym.initialize_from_matrix(global_params.config.kd_sym_path, scale, experiment_name,
                                  offset=offset, boundary=bd, fast_downsampling=True,
                                  data_path=h5_dir + 'sym.h5', mags=[1, 2], hdf5_names=['sym'])

    kd_asym = knossosdataset.KnossosDataset()
    # kd_asym.set_channel('jpg')
    kd_asym.initialize_from_matrix(global_params.config.kd_asym_path, scale, experiment_name,
                                   offset=offset, boundary=bd, fast_downsampling=True,
                                   data_path=h5_dir + 'asym.h5', mags=[1, 2], hdf5_names=['asym'])

    # run SyConn - without glia removal
    log.info('Step 1/8 - Creating SegmentationDatasets (incl. SV meshes)')
    exec_init.run_create_sds(chunk_size=(128, 128, 128), n_folders_fs=100)

    log.info('Step 2/8 - Creating SuperSegmentationDataset')
    exec_multiview.run_create_neuron_ssd(prior_glia_removal=False)

    # run_syn_generation can run in parallel to steps 4, 5, 6 and 7. Steps 3-7
    # are finally required for step 8, especially steps 4-7 could run in
    # parallel, because they require only medium MEM and CPU resources
    log.info('Step 3/8 - Synapse identification')
    exec_syns.run_syn_generation(chunk_size=(128, 128, 128), n_folders_fs=100)

    log.info('Step 4/8 - Neuron rendering')
    exec_multiview.run_neuron_rendering()

    log.info('Step 5/8 - Axon prediction')
    exec_multiview.run_axoness_prediction(n_jobs=4, e3=True)
    exec_multiview.run_axoness_mapping()

    log.info('Step 6/8 - Celltype prediction')
    exec_multiview.run_celltype_prediction(n_jobs=4)

    log.info('Step 7/8 - Spine prediction')
    exec_multiview.run_spiness_prediction(n_jobs=4)

    log.info('Step 8/8 - Synapse analysis')
    exec_syns.run_syn_analysis()

    log.info('SyConn analysis of "{}" has finished. Setting up flask server for'
             ' inspection. Annotated cell reconstructions and wiring can be analyzed via '
             'the KNOSSOS-SyConn plugin at `SyConn/scripts/kplugin/syconn_knossos_viewer.py`.'
             ''.format(experiment_name))
    fname_server = os.path.dirname(os.path.abspath(__file__)) + '/../kplugin/server.py'
    os.system('python {} --working_dir={} --port=10002'.format(fname_server, example_wd))
