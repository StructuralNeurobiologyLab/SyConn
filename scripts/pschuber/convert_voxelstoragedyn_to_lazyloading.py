# SyConn-dev
# Copyright (c) 2016 Philipp J. Schubert
# All rights reserved
import os

from syconn import global_params
from syconn.backend.storage import VoxelStorageDyn, VoxelStorageLazyLoading
from syconn.reps.segmentation import SegmentationDataset
from syconn.mp.mp_utils import start_multiprocess_imap


def conversion_helper(path: str):
    vx_path = path + '/voxel'
    if not os.path.isfile(vx_path + '.pkl'):
        return
    vx_dc = VoxelStorageDyn(vx_path, voxel_mode='syn_ssv' not in vx_path)
    vx_dc_lazy = VoxelStorageLazyLoading(vx_path, overwrite=True)
    for k in vx_dc.keys():
        vx_dc_lazy[k] = vx_dc._dc_intern['voxel_cache'][k]
    vx_dc_lazy.push()
    del vx_dc_lazy
    vx_dc_lazy = VoxelStorageLazyLoading(vx_path)
    for k in vx_dc.keys():
        if k not in vx_dc_lazy:
            print('OLAJNHEGOIUENGOAIENDG')
    vx_dc_lazy.close()
    return


def del_helper(path):
    # do not delete entire voxel.pkl files from syn, as they also contain rep coord
    path += 'voxel.pkl'
    assert 'syn_ssv' not in path
    if not os.path.isfile(path):
        return
    assert os.path.isfile(path.replace('.pkl', '.npz'))
    vx_dc = VoxelStorageDyn(path, voxel_mode=True)
    if 'voxel_cache' in vx_dc._dc_intern:
        del vx_dc._dc_intern['voxel_cache']
        vx_dc._cache_dc = {}  # this will prevent pushing cache_dc again to disk
        vx_dc.push()


if __name__ == '__main__':
    global_params.wd = '/ssdscratch/songbird/j0251/j0251_72_seg_20210127_agglo2/'
    sd_syn = SegmentationDataset('syn')
    sd_syn_ssv = SegmentationDataset('syn_ssv')
    params = list(sd_syn.iter_so_dir_paths())  # list(sd_syn_ssv.iter_so_dir_paths()) +

    # start_multiprocess_imap(conversion_helper, params, nb_cpus=40, show_progress=True, debug=False)
    start_multiprocess_imap(del_helper, params, nb_cpus=10, show_progress=True, debug=False)
