# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max-Planck-Institute of Neurobiology, Munich, Germany
# Authors: Philipp Schubert, Joergen Kornfeld
import numpy as np
try:
    from knossos_utils import mergelist_tools
except ImportError:
    from knossos_utils import mergelist_tools_fallback as mergelist_tool


def assemble_from_mergelist(ssd, mergelist):
    if mergelist is not None:
        assert "sv" in ssd.version_dict
        if isinstance(mergelist, dict):
            pass
        elif isinstance(mergelist, str):
            with open(mergelist, "r") as f:
                mergelist = mergelist_tools. \
                    subobject_map_from_mergelist(f.read())
        else:
            raise Exception("sv_mapping has unknown type")

    ssd.reversed_mapping_dict = mergelist

    for sv_id in mergelist.values():
        ssd.mapping_dict[sv_id] = []

    # Changed -1 defaults to 0
    # ssd._id_changer = np.zeros(np.max(list(mergelist.keys())) + 1,
    #                           dtype=np.uint)
    # TODO: check if np.int might be a problem for big datasets
    ssd._id_changer = np.ones(int(np.max(list(mergelist.keys())) + 1),
                              dtype=np.int) * (-1)

    for sv_id in mergelist.keys():
        ssd.mapping_dict[mergelist[sv_id]].append(sv_id)
        ssd._id_changer[sv_id] = mergelist[sv_id]

    ssd.save_dataset_shallow()


def split_ssv(ssd, ssv_id, sv_ids_1, sv_ids_2):
    pass


def merge_ssvs(ssd, ssv_ids):
    pass