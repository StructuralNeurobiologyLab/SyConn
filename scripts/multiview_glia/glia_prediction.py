# SyConn
# Copyright (c) 2018 Philipp J. Schubert, J. Kornfeld
# All rights reserved
from syconn.reps.rep_helper import parse_cc_dict_from_kml
from syconn.config.global_params import wd
from syconn.reps.super_segmentation import SuperSegmentationObject
from syconn.handler.prediction import get_glia_model
from syconn.handler.basics import chunkify
from syconn.config.global_params import get_dataset_scaling
from syconn.mp import qsub_utils as qu
import tqdm
import numpy as np
import os

#
# def new_glia_preds(init_rag):
#     pbar = tqdm.tqdm(total=len(init_rag))
#     m = get_glia_model()
#     for cc_ix, svixs in init_rag.iteritems():
#         # only create this SSV virtuall (-> 'version'='tmp').
#         # Views and predictions are stored in SVs with flags 'wo_glia' and 'raw_only'
#         #  and 'glia_probas'
#         ssv = SuperSegmentationObject(cc_ix, version="tmp", nb_cpus=20,
#                                       working_dir=wd, create=False, sv_ids=svixs,
#                                       scaling=get_dataset_scaling())
#         ssv.predict_views_gliaSV(m, pred_key_appendix="", verbose=False)
#         pbar.update(1)


if __name__ == "__main__":
    # Load initial RAG from  Knossos mergelist text file.
    init_rag_p = wd + "initial_rag.txt"
    assert os.path.isfile(init_rag_p), "Initial RAG could not be found at %s."\
                                       % init_rag_p
    init_rag = parse_cc_dict_from_kml(init_rag_p)
    # new_glia_preds(init_rag)

    # NEW AND UNTESTED
    sv_ids = np.concatenate(init_rag.values())
    np.random.shuffle(sv_ids)
    # chunk them
    multi_params = chunkify(sv_ids, 2000)
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
    pred_kwargs = dict(woglia=False, pred_key="", nb_cpus=1, verbose=False,
                       raw_only=True)

    multi_params = [[par, model_kwargs, so_kwargs, pred_kwargs] for par in
                    multi_params]
    script_folder = os.path.dirname(
        os.path.abspath(__file__)) + "/../../syconn/QSUB_scripts/"
    path_to_out = qu.QSUB_script(multi_params, "predict_sv_views",
                                 n_max_co_processes=200, pe="openmp",
                                 queue=None, n_cores=10, suffix="_glia",
                                 script_folder=script_folder)