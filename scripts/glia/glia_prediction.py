# SyConn
# Copyright (c) 2016 Philipp J. Schubert
# All rights reserved
from syconn.config.global_params import wd
from syconn.reps.super_segmentation import SuperSegmentationObject
from syconn.handler.prediction import get_glia_model
from syconn.config.global_params import get_dataset_scaling


def new_glia_preds(ssv_ixs):
    m = get_glia_model()
    for cc_ix, svixs in ssv_ixs:
        sso = SuperSegmentationObject(cc_ix, version="tmp", nb_cpus=20,
                                      working_dir=wd, create=False, sv_ids=svixs,
                                      scaling=get_dataset_scaling())
        sso.predict_views_gliaSV(m)
