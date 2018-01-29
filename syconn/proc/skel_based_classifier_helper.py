# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Sven Dorkenwald, Philipp Schubert, Joergen Kornfeld

import numpy as np
from syconn.reps import super_segmentation as ss


def generate_clf_data_thread(args):
    this_id = args[0]
    version = args[1]
    working_dir = args[2]
    feature_contexts_nm = args[3]
    save_path = args[4]

    ssd = ss.SuperSegmentationDataset(working_dir, version)
    sso = ssd.get_super_segmentation_object(this_id)

    for feature_context_nm in feature_contexts_nm:
        print "---", this_id, feature_context_nm
        feats = sso.skel_features(feature_context_nm=feature_context_nm)
        np.save(save_path % (feature_context_nm, this_id), feats)
