# SyConn
# Copyright (c) 2018 Philipp J. Schubert, J. Kornfeld
# All rights reserved
from syconn.reps.rep_helper import parse_cc_dict_from_kml
from syconn.config.global_params import wd
from syconn.reps.super_segmentation import SuperSegmentationObject
from syconn.handler.prediction import get_glia_model
from syconn.config.global_params import get_dataset_scaling
import tqdm
import os


def new_glia_preds(init_rag):
    pbar = tqdm.tqdm(total=len(init_rag))
    m = get_glia_model()
    for cc_ix, svixs in init_rag.iteritems():
        # only create this SSV virtuall (-> 'version'='tmp').
        # Views and predictions are stored in SVs with flags 'wo_glia' and 'raw_only'
        #  and 'glia_probas'
        ssv = SuperSegmentationObject(cc_ix, version="tmp", nb_cpus=20,
                                      working_dir=wd, create=False, sv_ids=svixs,
                                      scaling=get_dataset_scaling())
        ssv.predict_views_axoness(m, pred_key_appendix="_v2", verbose=False)
        pbar.update(1)


if __name__ == "__main__":
    # Load initial RAG from  Knossos mergelist text file.
    init_rag_p = wd + "initial_rag.txt"
    assert os.path.isfile(init_rag_p), "Initial RAG could not be found at %s."\
                                       % init_rag_p
    init_rag = parse_cc_dict_from_kml(init_rag_p)
    new_glia_preds(init_rag)