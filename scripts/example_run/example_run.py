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
import shutil
from syconn.handler.prediction import parse_movement_area_from_zip
from syconn.handler.logger import initialize_logging
from syconn import global_params
from syconn.exec import exec_init, exec_syns, exec_multiview


if __name__ == '__main__':
    # PREPARE TOY DATA
    # load h5 data
    curr_dir = os.path.dirname(os.path.realpath(__file__)) + '/'
    example_wd = os.path.expanduser('~/SyConn/example_cube/')
    log = initialize_logging('example_run', log_dir=example_wd + '/logs/')
    kzip_p = curr_dir + '/example_cube.k.zip'
    bb = parse_movement_area_from_zip(kzip_p)
    offset = np.array([0, 0, 0])
    bd = bb[1] - bb[0]
    scale = np.array([10, 10, 20])
    experiment_name = 'j0126_example'

    log.info('Preparing SyConn example at "{}".'.format(example_wd))
    # create KDs
    # TODO: data too big to put into github repository, add alternative to pull data into h5_dir
    h5_dir = curr_dir + 'data/'

    # kd = knossosdataset.KnossosDataset()
    # kd.initialize_from_matrix(example_wd + 'knossosdatasets/seg/', scale, experiment_name,
    #                           offset=offset, boundary=bd, fast_downsampling=True,
    #                           data_path=h5_dir + 'raw.h5', mags=[1, 2], hdf5_names=['raw'])
    # kd.from_matrix_to_cubes(offset, mags=[1, 2], data_path=h5_dir + 'seg.h5',
    #                         datatype=np.uint64, fast_downsampling=True,
    #                         as_raw=False, hdf5_names=['seg'])
    #
    # kd_mi = knossosdataset.KnossosDataset()
    # kd_mi.initialize_from_matrix(example_wd + 'knossosdatasets/mi/', scale, experiment_name,
    #                              offset=offset, boundary=bd, fast_downsampling=True,
    #                              data_path=h5_dir + 'mi.h5', mags=[1, 2], hdf5_names=['mi'])
    #
    # kd_vc = knossosdataset.KnossosDataset()
    # kd_vc.initialize_from_matrix(example_wd + 'knossosdatasets/vc/', scale, experiment_name,
    #                              offset=offset, boundary=bd, fast_downsampling=True,
    #                              data_path=h5_dir + 'vc.h5', mags=[1, 2], hdf5_names=['vc'])
    #
    # kd_sj = knossosdataset.KnossosDataset()
    # kd_sj.initialize_from_matrix(example_wd + 'knossosdatasets/sj/', scale, experiment_name,
    #                              offset=offset, boundary=bd, fast_downsampling=True,
    #                              data_path=h5_dir + 'sj.h5', mags=[1, 2], hdf5_names=['sj'])
    #
    # kd_sym = knossosdataset.KnossosDataset()
    # kd_sym.initialize_from_matrix(example_wd + 'knossosdatasets/sym/', scale, experiment_name,
    #                               offset=offset, boundary=bd, fast_downsampling=True,
    #                               data_path=h5_dir + 'sym.h5', mags=[1, 2], hdf5_names=['sym'])
    #
    # kd_asym = knossosdataset.KnossosDataset()
    # kd_asym.initialize_from_matrix(example_wd + 'knossosdatasets/asym/', scale, experiment_name,
    #                                offset=offset, boundary=bd, fast_downsampling=True,
    #                                data_path=h5_dir + 'asym.h5', mags=[1, 2], hdf5_names=['asym'])

    # PREPARE CONFIG
    os.makedirs(example_wd + '/glia/', exist_ok=True)
    shutil.copy(curr_dir + "/data/neuron_rag.bz2", example_wd + '/glia/neuron_rag.bz2')
    global_params.wd = example_wd
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
kd_seg_path = {}
kd_sym_path = {}
kd_asym_path = {}
kd_sj = {}
kd_vc = {}
kd_mi = {}

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
    """.format(example_wd + 'seg/', example_wd + 'sym/', example_wd + 'asym/', example_wd + 'sj/',
               example_wd + 'vc/', example_wd + 'mi/')
    with open(example_wd + 'config.ini', 'w') as f:
        f.write(config_str)

    log.info('Finished example cube preparation. Starting SyConn pipeline.')
    # RUN SYCONN
    # TODO: currently example run does not support QSUB/SLURM, because global_params.wd is not changed in the file, only in memory. Alternative could be to use bash variables or similar
    exec_init.run_create_sds()

    exec_multiview.run_create_neuron_ssd()

    raise()
    # TODO: create fallback if SV meshes are not available (e.g. use mesh_from_scratch)
    exec_multiview.run_neuron_rendering()
    exec_multiview.run_axoness_prediction()
    # TODO: create fallback if SV skeletons are not available (e.g. use rendering locations?)
    exec_multiview.run_axoness_mapping()
    exec_multiview.run_celltype_prediction()
    exec_multiview.run_spiness_prediction()

    exec_syns.run_syn_generation()
    exec_syns.run_syn_analysis()



