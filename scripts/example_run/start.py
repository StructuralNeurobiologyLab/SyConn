# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max-Planck-Institute of Neurobiology, Munich, Germany
# Authors: Philipp Schubert, Joergen Kornfeld

from knossos_utils import knossosdataset
import numpy as np
import os
import subprocess
import glob
import shutil

from syconn.handler.prediction import parse_movement_area_from_zip
from syconn.handler.logger import initialize_logging
from syconn import global_params
from syconn.exec import exec_init, exec_syns, exec_multiview


if __name__ == '__main__':
    # PREPARE TOY DATA
    curr_dir = os.path.dirname(os.path.realpath(__file__)) + '/'
    h5_dir = curr_dir + 'data/'
    if not os.path.isfile(h5_dir + 'seg.h5') or len(glob.glob(h5_dir + '*.h5')) != 7\
            or not os.path.isfile(h5_dir + 'neuron_rag.bz2'):
        raise ValueError('Example data could not be found at "{}".'.format(h5_dir))

    example_wd = os.path.expanduser('~/SyConn/example_cube/')
    log = initialize_logging('example_run', log_dir=example_wd + '/logs/')
    log.info('Step 0/8 - Preparation\nExample can be found at "{}".'.format(example_wd))

    kzip_p = curr_dir + '/example_cube_small.k.zip'
    bb = parse_movement_area_from_zip(kzip_p)
    offset = np.array([0, 0, 0])
    bd = bb[1] - bb[0]
    scale = np.array([10, 10, 20])
    experiment_name = 'j0126_example'

    # create KDs
    # TODO: data too big to put into github repository, add alternative to pull data into h5_dir

    kd = knossosdataset.KnossosDataset()
    kd.initialize_from_matrix(example_wd + 'knossosdatasets/seg/', scale, experiment_name,
                              offset=offset, boundary=bd, fast_downsampling=True,
                              data_path=h5_dir + 'raw.h5', mags=[1, 2], hdf5_names=['raw'])
    kd.from_matrix_to_cubes(offset, mags=[1, 2], data_path=h5_dir + 'seg.h5',
                            datatype=np.uint64, fast_downsampling=True,
                            as_raw=False, hdf5_names=['seg'])

    kd_mi = knossosdataset.KnossosDataset()
    kd_mi.initialize_from_matrix(example_wd + 'knossosdatasets/mi/', scale, experiment_name,
                                 offset=offset, boundary=bd, fast_downsampling=True,
                                 data_path=h5_dir + 'mi.h5', mags=[1, 2], hdf5_names=['mi'])

    kd_vc = knossosdataset.KnossosDataset()
    kd_vc.initialize_from_matrix(example_wd + 'knossosdatasets/vc/', scale, experiment_name,
                                 offset=offset, boundary=bd, fast_downsampling=True,
                                 data_path=h5_dir + 'vc.h5', mags=[1, 2], hdf5_names=['vc'])

    kd_sj = knossosdataset.KnossosDataset()
    kd_sj.initialize_from_matrix(example_wd + 'knossosdatasets/sj/', scale, experiment_name,
                                 offset=offset, boundary=bd, fast_downsampling=True,
                                 data_path=h5_dir + 'sj.h5', mags=[1, 2], hdf5_names=['sj'])

    kd_sym = knossosdataset.KnossosDataset()
    kd_sym.initialize_from_matrix(example_wd + 'knossosdatasets/sym/', scale, experiment_name,
                                  offset=offset, boundary=bd, fast_downsampling=True,
                                  data_path=h5_dir + 'sym.h5', mags=[1, 2], hdf5_names=['sym'])

    kd_asym = knossosdataset.KnossosDataset()
    kd_asym.initialize_from_matrix(example_wd + 'knossosdatasets/asym/', scale, experiment_name,
                                   offset=offset, boundary=bd, fast_downsampling=True,
                                   data_path=h5_dir + 'asym.h5', mags=[1, 2], hdf5_names=['asym'])

    # PREPARE CONFIG
    os.makedirs(example_wd + '/glia/', exist_ok=True)
    shutil.copy(curr_dir + "/data/neuron_rag.bz2", example_wd + '/glia/neuron_rag.bz2')
    global_params.wd = example_wd
    py36path = subprocess.check_output('source deactivate; source activate py36;'
                                       ' which python', shell=True).decode().replace('\n', '')
    config_str = """[Versions]
sv = 0
vc = 0
sj = 0
syn = 0
syn_ssv = 0
mi = 0
ssv = 0
cs_agg = 0
ax_gt = 0

[Paths]
kd_seg = {}
kd_sym = {}
kd_asym = {}
kd_sj = {}
kd_vc = {}
kd_mi = {}
init_rag = {}
py36path = {}

[LowerMappingRatios]
mi = 0.5
sj = 0.1
vc = 0.5

[UpperMappingRatios]
mi = 1.
sj = 0.9
vc = 1.

[Sizethresholds]
mi = 2786
sj = 498
vc = 1584
    """.format(example_wd + 'knossosdatasets/seg/', example_wd + 'knossosdatasets/sym/',
               example_wd + 'knossosdatasets/asym/', example_wd + 'knossosdatasets/sj/',
               example_wd + 'knossosdatasets/vc/', example_wd + 'knossosdatasets/mi/', '', py36path)
    with open(example_wd + 'config.ini', 'w') as f:
        f.write(config_str)

    configspec_str = """
[Versions]
__many__ = string

[Paths]
__many__ = string()

[Dataset]
scaling = float_list(min=3, max=3)

[LowerMappingRatios]
__many__ = float

[UpperMappingRatios]
__many__ = float

[Sizethresholds]
__many__ = integer
"""
    with open(example_wd + 'configspec.ini', 'w') as f:
        f.write(configspec_str)

    log.info('Finished example cube preparation {}. Starting SyConn pipeline.'.format(bd))

    # RUN SYCONN
    log.info('Step 1/8 - Creating SegmentationDatasets')
    # TODO: currently example run does not support fallback for SLURM entirely -> adapt and test
    exec_init.run_create_sds()

    log.info('Step 2/8 - Creating SuperSegmentationDataset')
    exec_multiview.run_create_neuron_ssd()

    log.info('Step 3/8 - Neuron rendering')
    # TODO: create fallback if SV meshes are not available (e.g. use mesh_from_scratch)
    exec_multiview.run_neuron_rendering()

    log.info('Step 4/8 - Axon prediction')
    exec_multiview.run_axoness_prediction()
    # TODO: create fallback if SV skeletons are not available (e.g. use rendering locations?)
    exec_multiview.run_axoness_mapping()

    log.info('Step 5/8 - Celltype prediction')
    exec_multiview.run_celltype_prediction()

    log.info('Step 6/8 - Spine prediction')
    exec_multiview.run_spiness_prediction()

    log.info('Step 7/8 - Synapse identification')
    exec_syns.run_syn_generation()

    log.info('Step 8/8 - Synapse analysis')
    exec_syns.run_syn_analysis()



