# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max-Planck-Institute of Neurobiology, Munich, Germany
# Authors: Philipp Schubert, Joergen Kornfeld

import numpy as np
import os

from ..reps import super_segmentation as ss
from ..reps import super_segmentation_helper as ssh
from ..proc import log_proc


def generate_clf_data_thread(args):
    this_id = args[0]
    ssd_version = args[1]
    working_dir = args[2]
    feature_contexts_nm = args[3]
    save_path = args[4]
    comment_converter = args[5]
    overwrite = args[6]

    ssd = ss.SuperSegmentationDataset(working_dir, version=ssd_version)
    sso = ssd.get_super_segmentation_object(this_id)
    sso.enable_locking = False

    for feature_context_nm in feature_contexts_nm:
        print("---", this_id, feature_context_nm)
        if os.path.isfile(sso.skeleton_kzip_path):
            label_array = ssh.label_array_for_sso_skel(sso, comment_converter)
            if not np.all(label_array == -1):
                print("Found node-wise annotations in {}.".format(sso.skeleton_kzip_path))
                label_dir, label_fname = os.path.split(save_path)
                label_save_path = label_dir + "/" + \
                                  label_fname.replace("features", "labels")
                np.save(label_save_path % (feature_context_nm, this_id), label_array)
        feats = sso.skel_features(feature_context_nm=feature_context_nm, overwrite=overwrite)
        log_proc.debug('feature array shape {}; feature context {} nm;'
                       ' SSV {}'.format(feats.shape, feature_context_nm, sso.id))
        np.save(save_path % (feature_context_nm, this_id), feats)
