# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max-Planck-Institute of Neurobiology, Munich, Germany
# Authors: Philipp Schubert, Joergen Kornfeld

from syconn.reps.super_segmentation import *
from syconn.handler.prediction import get_celltype_model, get_axoness_model, \
    get_semseg_spiness_model, get_glia_model
from syconn.backend.storage import AttributeDict, CompressedStorage
from syconn.reps.segmentation import SegmentationDataset
import time
import sys


def helper_func(paths):
    num_locs = []
    for ad_p in paths:
        ad = AttributeDict(ad_p + 'attr_dict.pkl', read_only=True,
                           disable_locking=True)
        sample_locs = np.concatenate(ad['sample_locations'])
        num_locs.append(len(sample_locs))
    return num_locs


def helper_func_sd(paths):
    num_locs = []
    for p in paths:
        loc_dc = CompressedStorage(p + '/locations.pkl', read_only=True,
                           disable_locking=True)
        sample_locs = [np.concatenate(sl) for sl in loc_dc.values()]
        num_locs += [len(sl) for sl in sample_locs]
    return num_locs

# TODO: make this a test on toy data (which has to be created and added to the repo)
if __name__ == '__main__':
    # performed on SSD at '/wholebrain/songbird/j0126/areaxfs_v6//ssv_0/', 17Jan02019
    ssd = SuperSegmentationDataset(working_dir='/wholebrain/songbird/j0126/areaxfs_v6/')
    sd = SegmentationDataset(obj_type='sv', working_dir='/wholebrain/songbird/j0126/areaxfs_v6/')

    # # Statistics of SSVs in datatset
    # all_paths = chunkify(glob.glob(ssd.path + "/so_storage/*/*/*/"), 500)
    # num_samplelocs = start_multiprocess_imap(helper_func, all_paths, nb_cpus=20)
    # num_samplelocs = np.concatenate(num_samplelocs)  # transform list of lists into 1D array
    # print('#SSVs: {}\nMean #sample_locs: {}\nTotal #sample_locs: {}'.format(len(ssd.ssv_ids),
    #                                                 np.mean(num_samplelocs), np.sum(num_samplelocs)))
    # # Statistics of SVs in the original datatset
    # all_paths = chunkify(sd.so_dir_paths, 500)
    # num_samplelocs = start_multiprocess_imap(helper_func_sd, all_paths, nb_cpus=20)
    # num_samplelocs = np.concatenate(num_samplelocs)  # transform list of lists into 1D array
    # print('#SVs: {}\nMean #sample_locs: {}\nTotal #sample_locs: {}'.format(len(sd.ids),
    #                                                 np.mean(num_samplelocs), np.sum(num_samplelocs)))

    ssvs = ssd.get_super_segmentation_object([26607617, 27525127])
    [ssv.load_attr_dict() for ssv in ssvs]
    ssvs_tmp = [SuperSegmentationObject(ssv.id, create=False, version='tmp',
                                        sv_ids=ssv.sv_ids) for ssv in ssvs]
    # perform python dependent processing
    if not (sys.version_info[0] == 3 and sys.version_info[1] > 5):

        # sample locs
        print('Sampling rendering locations [s]:')
        for ii in range(3):
            start = time.time()
            for ssv_tmp in ssvs_tmp:
                ssv_tmp.nb_cpus = 1  # default when using GPU
                print('Sampling SSV {}'.format(ssv_tmp.id))
                ssv_tmp.sample_locations(cache=False, force=True)
            print('Run {}: {:.4f}'.format(ii, time.time() - start))

        # glia
        m = get_glia_model()
        print('Predicting glia [s]:')
        for ii in range(3):
            start = time.time()
            for ssv_tmp in ssvs_tmp:
                ssv_tmp.nb_cpus = 10  # default when using GPU
                print('Predicting SSV {} with {} sample locations.'.format(
                    ssv_tmp.id, len(np.concatenate(ssv_tmp.sample_locations()))))
                ssv_tmp.predict_views_gliaSV(m)
            print('Run {}: {:.4f}'.format(ii, time.time() - start))
        del m

        # celltyes
        m = get_celltype_model()
        print('Predicting celltype [s]:')
        for ii in range(3):
            start = time.time()
            for ssv_tmp in ssvs_tmp:
                ssv_tmp.nb_cpus = 10  # default when using GPU
                print('Predicting SSV {} with {} sample locations.'.format(
                    ssv_tmp.id, len(np.concatenate(ssv_tmp.sample_locations()))))
                ssv_tmp.predict_celltype_cnn(m, overwrite=True)
            print('Run {}: {:.4f}'.format(ii, time.time() - start))
        del m

        # axoness
        m = get_axoness_model()
        print('Predicting axoness [s]:')
        for ii in range(3):
            start = time.time()
            for ssv_tmp in ssvs_tmp:
                ssv_tmp.nb_cpus = 10  # default when using GPU
                print('Predicting SSV {} with {} sample locations.'.format(
                    ssv_tmp.id, len(np.concatenate(ssv_tmp.sample_locations()))))
                ssv_tmp.predict_views_axoness(m)
            print('Run {}: {:.4f}'.format(ii, time.time() - start))
        del m

        # NEURON RENDERING
        render_kwargs = dict(add_cellobjects=True, overwrite=True,
                             skip_indexviews=False, verbose=True)
        for ii in range(len(ssvs)):
            ssvs_tmp[ii].attr_dict = ssvs[
                ii].attr_dict  # copy mapped object info

        # render views with EGL
        print('{} neuron rendering'.format(global_params.PYOPENGL_PLATFORM))
        for ii in range(3):
            start = time.time()
            for ssv_tmp in ssvs_tmp:
                ssv_tmp.render_views(**render_kwargs)
            print('Run {}: {:.4f}s'.format(ii, time.time() - start))
    else:  # using PY36!
        # spiness
        m = get_semseg_spiness_model()
        print('Predicting spiness [s]:')
        for ii in range(3):
            start = time.time()
            for ssv_tmp in ssvs_tmp:
                ssv_tmp.nb_cpus = 10  # default when using GPU
                print('Predicting SSV {} with {} sample locations.'.format(
                    ssv_tmp.id, len(np.concatenate(ssv_tmp.sample_locations()))))
                ssv_tmp.predict_semseg(m, 'spiness', verbose=True)
            print('Run {}: {:.4f}'.format(ii, time.time() - start))
        del m

        print('Mapping spiness [s]:')
        for ii in range(3):
            start = time.time()
            for ssv_tmp in ssvs_tmp:
                ssv_tmp.nb_cpus = 2
                ssv_tmp.semseg2mesh('spiness', force_overwrite=True)
            print('Run {}: {:.4f}'.format(ii, time.time() - start))


