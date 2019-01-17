# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Philipp Schubert, Sven Dorkenwald, Joergen Kornfeld
import os

from syconn.config import global_params
from syconn.handler.prediction import get_axoness_model
from syconn.handler.basics import chunkify
from syconn.reps.super_segmentation import SuperSegmentationDataset
from syconn.reps.super_segmentation_helper import find_missing_sv_attributes_in_ssv
from syconn.handler.logger import initialize_logging
from syconn.mp import qsub_utils as qu


def axoness_pred_exists(sv):
    sv.load_attr_dict()
    return 'axoness_probas' in sv.attr_dict


if __name__ == "__main__":
    log = initialize_logging('axon_prediction', global_params.wd + '/logs/',
                             overwrite=False)
    # TODO: currently working directory has to be set globally in global_params and is not adjustable
    # here because all qsub jobs will start a script referring to 'global_params.wd'
    ssd = SuperSegmentationDataset(working_dir=global_params.wd)
    sd = ssd.get_segmentationdataset("sv")
    # chunk them
    multi_params = chunkify(sd.so_dir_paths, 100)
    pred_key = "axoness_probas"  # leave this fixed because it is used all over
    # get model properties
    log.info('Performing axon prediction of neuron views. Labels will be stored '
             'on SV level in the attribute dict with key "{}"'.format(pred_key))
    m = get_axoness_model()
    model_kwargs = dict(model_path=m._path, normalize_data=m.normalize_data,
                        imposed_batch_size=m.imposed_batch_size, nb_labels=m.nb_labels,
                        channels_to_load=m.channels_to_load)
    # all other kwargs like obj_type='sv' and version are the current SV SegmentationDataset by default
    so_kwargs = dict(working_dir=global_params.wd)
    # for axoness views set woglia to True (because glia were removed beforehand),
    #  raw_only to False
    pred_kwargs = dict(woglia=True, pred_key=pred_key, verbose=False,
                       raw_only=False)
    multi_params = [[par, model_kwargs, so_kwargs, pred_kwargs] for
                    par in multi_params]

    for par in multi_params:
        mk = par[1]
        # Single GPUs are made available for every job via slurm, no need for random assignments.
        mk["init_gpu"] = 0  # np.random.rand(0, 2)
    path_to_out = qu.QSUB_script(multi_params, "predict_sv_views_chunked",
                                 n_max_co_processes=15, pe="openmp", queue=None,
                                 script_folder=None, n_cores=10,
                                 suffix="_axoness", additional_flags="--gres=gpu:1")  # removed -V
    log.info('Finished axon prediction. Now checking for missing predictions.')
    res = find_missing_sv_attributes_in_ssv(ssd, pred_key, n_cores=10)
    if len(res) > 0:
        log.error("Attribute '{}' missing for follwing"
                  " SVs:\n{}".format(pred_key, res))
    else:
        log.info('Success.')
