# SyConn
# Copyright (c) 2018 Philipp J. Schubert, J. Kornfeld
# All rights reserved
from syconn.reps.rep_helper import parse_cc_dict_from_kml
from syconn.config.global_params import wd
from syconn.reps.super_segmentation import SuperSegmentationObject
from syconn.reps.segmentation import SegmentationDataset
from syconn.reps.segmentation_helper import find_missing_sv_attributes
from syconn.handler.prediction import get_glia_model
from syconn.handler.basics import chunkify
from syconn.config.global_params import get_dataset_scaling
from syconn.mp import qsub_utils as qu
import tqdm
import numpy as np
import os


if __name__ == "__main__":
    # only append to this key if needed (for e.g. different versions, change accordingly in 'axoness_mapping.py')
    pred_key = "glia_probas"
    # Load initial RAG from  Knossos mergelist text file.
    init_rag_p = wd + "initial_rag.txt"
    assert os.path.isfile(init_rag_p), "Initial RAG could not be found at %s."\
                                       % init_rag_p
    init_rag = parse_cc_dict_from_kml(init_rag_p)

    # chunk them
    sd = SegmentationDataset("sv", working_dir=wd)
    multi_params = chunkify(sd.so_dir_paths, 75)

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
        mk["init_gpu"] = np.random.rand(0, 2)
    script_folder = os.path.dirname(
        os.path.abspath(__file__)) + "/../../syconn/QSUB_scripts_chunked/"
    path_to_out = qu.QSUB_script(multi_params, "predict_sv_views",
                                 n_max_co_processes=25, pe="openmp",
                                 queue=None, n_cores=10, suffix="_glia",
                                 script_folder=script_folder,
                                 sge_additional_flags="-V")
    res = find_missing_sv_attributes(sd, pred_key, n_cores=10)
    if len(res) > 0:
        print("Attribute '{}' missing for follwing"
              " SVs:\n{}".format(pred_key, res))