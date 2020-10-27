# -*- coding: utf-8 -*-
# MorphX - Toolkit for morphology exploration and segmentation
#
# Copyright (c) 2020 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Jonathan Klimesch

from typing import List, Tuple
from morphx.classes.pointcloud import PointCloud
from syconn.reps.super_segmentation_object import SuperSegmentationObject


def get_preds(sso: SuperSegmentationObject, key: str):
    labels = sso.label_dict()[key]
    pred_mask = (labels != -1).reshape(-1)
    mesh = sso.mesh[1].reshape((-1, 3))
    verts = mesh[pred_mask]
    preds = labels[pred_mask]
    return PointCloud(vertices=verts, labels=preds)


def replace_preds(sso: SuperSegmentationObject, base: str, replace: List[Tuple[int, str, List[Tuple[int, int]]]]):
    """
    Args:
        replace: (int, str, mappings) - In base, replace labels with int by labels from str, mapped by mappings.
    """
    labels = sso.label_dict()[base]
    for item in replace:
        replace_labels = sso.label_dict()[item[1]]
        for mapping in item[2]:
            replace_labels[replace_labels == mapping[0]] = mapping[1]
        replace_mask = (labels == item[0]).reshape(-1)
        labels[replace_mask] = replace_labels[replace_mask]
    pred_mask = (labels != -1).reshape(-1)
    mesh = sso.mesh[1].reshape((-1, 3))
    verts = mesh[pred_mask]
    preds = labels[pred_mask]
    return PointCloud(vertices=verts, labels=preds)
