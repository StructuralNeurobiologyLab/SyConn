# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max-Planck-Institute of Neurobiology, Munich, Germany
# Authors: Philipp Schubert, Joergen Kornfeld
import numpy as np
import zipfile
import os
import shutil
import re
import pickle as pkl
from ..handler.basics import read_mesh_from_zip
from ..reps.super_segmentation import SuperSegmentationObject
from .. import global_params
import networkx as nx


def init_sso_from_kzip(path, load_as_tmp=True, sso_id=None):
    """
    Initializes cell reconstruction from k.zip file.
    The k.zip needs the following content:
        - Mesh files: 'sv.ply', 'mi.ply', 'sj.ply', 'vc.ply'
        - meta dict: 'meta.pkl'
        - [Optional] Rendering locations: 'sample_locations.pkl'
        (currently broekn to use .npy, fixed in python 3.7)
        - [Optional] Supervoxel graph: 'rag.bz2'
        - [Optional] Skeleton representation: 'skeleton.pkl'
        - [Optional] attribute dict: 'attr_dict.pkl'

    Parameters
    ----------
    path : str
        Path to kzip which contains SSV data
    sso_id : int
        ID of SSV, if not given looks for the first scalar occurrence in `path`
    load_as_tmp : bool
        If True then `working_dir` and `version_dict` in meta.pkl dictionary is
         not passed to SSO constructor, instead all version will be set to 'tmp'
         and working directory will be None. Used to process SSO independent on working directory.

    Returns
    -------
    SuperSegmentationObject
    """
    if sso_id is None:
        sso_id = int(re.findall("/(\d+).", path)[0])
    zip = zipfile.ZipFile(path)
    files = zip.namelist()

    # attribute dictionary
    with zipfile.ZipFile(path, allowZip64=True) as z:
        f = z.open("meta.pkl")
        meta_dc = pkl.load(f)

    if load_as_tmp:
        for k in meta_dc['version_dict']:
            meta_dc['version_dict'][k] = 'tmp'
        meta_dc['working_dir'] = None

    sso = SuperSegmentationObject(sso_id, version="tmp", **meta_dc)
    # Required to enable prediction in 'tmp' SSVs # TODO: change those properties in SSO constructor
    sso._mesh_caching = True
    sso._view_caching = True

    # meshes
    for obj_type in global_params.existing_cell_organelles + ["sv"]:
        ply_name = "{}.ply".format(obj_type)
        if ply_name in files:
            sso._meshes[obj_type] = read_mesh_from_zip(path, ply_name)

    # skeleton
    if "skeleton.pkl" in files:
        with zipfile.ZipFile(path, allowZip64=True) as z:
            f = z.open("skeleton.pkl")
            sso._skeleton = pkl.load(f)  # or loads?  returns a dict

    # attribute dictionary
    if "attr_dict.pkl" in files:
        with zipfile.ZipFile(path, allowZip64=True) as z:
            f = z.open("attr_dict.pkl")
            sso.attr_dict = pkl.load(f)

    # Sample locations
    if "sample_locations.pkl" in files:
        with zipfile.ZipFile(path, allowZip64=True, mode='r') as z:
            f = z.open("sample_locations.pkl")
            sso._sample_locations = pkl.load(f)
            # # currently broken, fixed in python 3.7:
            # https://stackoverflow.com/questions/33742544/zip-file-not-seekable
            # f = z.open("sample_locations.npy", mode='r')
            # sso._sample_locations = np.load(f)

    # RAG
    if "rag.bz2" in files:
        with zipfile.ZipFile(path, allowZip64=True) as z:
            tmp_dir = os.path.dirname(path)
            tmp_p = "{}/rag.bz2".format(tmp_dir)
            z.extract('rag.bz2', tmp_dir)
            sso._sv_graph = nx.read_edgelist(tmp_p, nodetype=np.uint)
            os.remove(tmp_p)
            _ = sso.rag  # invoke node conversion into SegmentationObjects
    return sso


def init_ssd_from_kzips(dir_path):
    pass
