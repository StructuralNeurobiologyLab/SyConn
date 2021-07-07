# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max-Planck-Institute of Neurobiology, Munich, Germany
# Authors: Philipp Schubert, Joergen Kornfeld
import os
import pickle as pkl
import re
import zipfile

import networkx as nx
import numpy as np

from .. import global_params
from ..handler.basics import read_mesh_from_zip, read_meshes_from_zip
from ..reps.super_segmentation import SuperSegmentationObject


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

    Args:
        path: str
            Path to kzip which contains SSV data
        load_as_tmp: bool
            If True then `working_dir` and `version_dict` in meta.pkl dictionary is
            not passed to SSO constructor, instead all version will be set to 'tmp'
            and working directory will be None. Used to process SSO independent on working directory.
        sso_id: int
            ID of SSV, if not given looks for the first scalar occurrence in `path`

    Returns: SuperSegmentationObject

    """
    if sso_id is None:
        sso_id = int(re.findall(r"/(\d+).", path)[0])
    files = list(zipfile.ZipFile(path).namelist())

    # attribute dictionary
    with zipfile.ZipFile(path, allowZip64=True) as z:
        f = z.open("meta.pkl")
        meta_dc = pkl.load(f)

    if load_as_tmp:
        for k in meta_dc['version_dict']:
            meta_dc['version_dict'][k] = 'tmp'
        meta_dc['working_dir'] = None
        meta_dc['version'] = 'tmp'
        if 'sso_id' in meta_dc:
            del meta_dc['sso_id']
    else:
        if 'sso_id' not in meta_dc or meta_dc['sso_id'] is None:
            raise ValueError('Loading cell reconstruction with load_as_tmp=False '
                             'which requires the SuperSegmentationObject ID. None '
                             'found in meta dictionary.')
        sso_id = meta_dc['sso_id']
        del meta_dc['sso_id']
    sso = SuperSegmentationObject(sso_id, **meta_dc)
    # TODO: change those properties in SSO constructor
    # Required to enable prediction in 'tmp' SSVs
    sso._mesh_caching = True
    sso._object_caching = True
    sso._view_caching = True

    # meshes
    for obj_type in global_params.config['process_cell_organelles'] + ["sv", "syn_ssv"]:
        ply_name = "{}.ply".format(obj_type)
        if ply_name in files:
            sso._meshes[obj_type] = read_mesh_from_zip(path, ply_name)
            files.remove(ply_name)

    # skeleton
    if "skeleton.pkl" in files:
        with zipfile.ZipFile(path, allowZip64=True) as z:
            f = z.open("skeleton.pkl")
            sso.skeleton = pkl.load(f)  # or loads?  returns a dict
        files.remove("skeleton.pkl")
    # attribute dictionary
    if "attr_dict.pkl" in files:
        with zipfile.ZipFile(path, allowZip64=True) as z:
            f = z.open("attr_dict.pkl")
            sso.attr_dict = pkl.load(f)
        files.remove("attr_dict.pkl")

    # Sample locations
    if "sample_locations.pkl" in files:
        with zipfile.ZipFile(path, allowZip64=True, mode='r') as z:
            f = z.open("sample_locations.pkl")
            sso._sample_locations = pkl.load(f)
            # # currently broken, fixed in python 3.7:
            # https://stackoverflow.com/questions/33742544/zip-file-not-seekable
            # f = z.open("sample_locations.npy", mode='r')
            # sso._sample_locations = np.load(f)
        files.remove("sample_locations.pkl")

    # RAG
    if "rag.bz2" in files:
        with zipfile.ZipFile(path, allowZip64=True) as z:
            tmp_dir = os.path.dirname(path)
            tmp_p = "{}/rag.bz2".format(tmp_dir)
            z.extract('rag.bz2', tmp_dir)
            sso._sv_graph = nx.read_edgelist(tmp_p, nodetype=np.uint64)
            os.remove(tmp_p)
            _ = sso.rag  # invoke node conversion into SegmentationObjects
        files.remove("rag.bz2")
    ply_files = []
    sv_ids = []
    for fname in files:
        match = re.match(r'sv_(\d+).ply', fname)
        if match is not None:
            ply_files.append(fname)
            sv_ids.append(int(match[1]))
    if len(ply_files):
        if 'sv' in sso.attr_dict:
            if len(np.setdiff1d(sv_ids, sso.attr_dict['sv'])):
                raise ValueError(f'Inconsistency in cell supervoxel IDs (attr_dict vs meshes).')
        else:
            sso.attr_dict['sv'] = sv_ids
        sv_meshes = read_meshes_from_zip(path, ply_files)
        for m, sv in zip(sv_meshes, sso.svs):
            sv._mesh = m
    return sso


def init_ssd_from_kzips(dir_path):
    pass
