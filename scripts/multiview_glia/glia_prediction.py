# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max-Planck-Institute of Neurobiology, Munich, Germany
# Authors: Philipp Schubert, Joergen Kornfeld
import numpy as np
import os

from syconn.config.global_params import wd
from syconn.handler.logger import initialize_logging
from syconn.reps.segmentation import SegmentationDataset
from syconn.reps.segmentation_helper import find_missing_sv_attributes
from syconn.handler.prediction import get_glia_model
from syconn.handler.basics import chunkify, parse_cc_dict_from_kml
from syconn.mp import qsub_utils as qu


if __name__ == "__main__":
    log = initialize_logging('glia_prediction', wd + '/logs/',
                             overwrite=False)
    # only append to this key if needed (for e.g. different versions, change accordingly in 'axoness_mapping.py')
    pred_key = "glia_probas"
    # Load initial RAG from  Knossos mergelist text file.
    init_rag_p = wd + "initial_rag.txt"
    assert os.path.isfile(init_rag_p), "Initial RAG could not be found at %s."\
                                       % init_rag_p
    init_rag = parse_cc_dict_from_kml(init_rag_p)
    log.info('Found {} CCs with a total of {} SVs in inital RAG.'
          ''.format(len(init_rag), np.sum([len(v) for v in init_rag.values()])))
    # chunk them
    sd = SegmentationDataset("sv", working_dir=wd)
    multi_params = chunkify(sd.so_dir_paths, 100)
    # get model properties
    m = get_glia_model()
    model_kwargs = dict(model_path=m._path,
                        normalize_data=m.normalize_data,
                        imposed_batch_size=m.imposed_batch_size,
                        nb_labels=m.nb_labels,
                        channels_to_load=m.channels_to_load)
    # all other kwargs like obj_type='sv' and version are the current SV SegmentationDataset by default
    so_kwargs = dict(working_dir=wd)
    # for glia views set woglia to False (because glia are included),
    #  raw_only to True
    pred_kwargs = dict(woglia=False, pred_key=pred_key, verbose=False,
                       raw_only=True)

    multi_params = [[par, model_kwargs, so_kwargs, pred_kwargs] for par in
                    multi_params]
    # randomly assign to gpu 0 or 1
    for par in multi_params:
        mk = par[1]
        mk["init_gpu"] = 0  # GPUs are made available for every job via slurm, no need for random assignments: np.random.rand(0, 2)
    path_to_out = qu.QSUB_script(multi_params, "predict_sv_views_chunked",
                                 n_max_co_processes=25, pe="openmp",
                                 queue=None, n_cores=10, suffix="_glia",
                                 script_folder=None,
                                 additional_flags="--gres=gpu:1")  # removed -V
    log.info('Finished glia prediction. Checking completeness.')
    res = find_missing_sv_attributes(sd, pred_key, n_cores=10)
    if len(res) > 0:
        log.error("Attribute '{}' missing for follwing"
                  " SVs:\n{}".format(pred_key, res))
    else:
        log.info('Success.')