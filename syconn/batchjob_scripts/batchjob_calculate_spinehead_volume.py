# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max-Planck-Institute for Medical Research, Heidelberg, Germany
# Authors: Sven Dorkenwald, Philipp Schubert, Jörgen Kornfeld

import sys
import pickle as pkl
from syconn.reps.super_segmentation_helper import extract_spinehead_volume_mesh
from syconn.reps.super_segmentation import *

path_storage_file = sys.argv[1]
path_out_file = sys.argv[2]

with open(path_storage_file, 'rb') as f:
    args = []
    while True:
        try:
            args.append(pkl.load(f))
        except EOFError:
            break

sso_ids = args[0]

sd_syn_ssv = SegmentationDataset('syn_ssv', cache_properties=['rep_coord'])

ssd = SuperSegmentationDataset(sd_lookup={'syn_ssv': sd_syn_ssv})
for sso in ssd.get_super_segmentation_object(sso_ids):
    assert sso.load_attr_dict() == 0, f"Attribute of SSO {sso.id} does not exist."
    extract_spinehead_volume_mesh(sso)
    sso.save_attr_dict()
    del sso

with open(path_out_file, "wb") as f:
    pkl.dump("", f)
