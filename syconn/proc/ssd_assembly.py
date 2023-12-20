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
    Initializes a SuperSegmentationObject (SSO) from a k.zip file. The k.zip file should contain
    mesh files ('sv.ply', 'mi.ply', 'sj.ply', 'vc.ply'), a meta dictionary ('meta.pkl'), and
    optionally rendering locations ('sample_locations.pkl'), a supervoxel graph ('rag.bz2'), a
    skeleton representation ('skeleton.pkl'), and an attribute dictionary ('attr_dict.pkl').
    Note: Currently broken to use .npy for rendering locations, fixed in python 3.7.
    
    Args:
        path (str): Path to the k.zip file containing SSO data.
        load_as_tmp (bool, optional): If True, the 'working_dir' and 'version_dict' in the
            meta.pkl dictionary are not passed to the SSO constructor. Instead, all versions will
            be set to 'tmp' and the working directory will be None. This is used to process the SSO
            independent of the working directory. Defaults to True.
        sso_id (int, optional): ID of the SSO. If not provided, the function looks for the first
            scalar occurrence in `path`. Defaults to None.
    
    Returns:
        SuperSegmentationObject: Initialized SSO.
    
    Raises:
        ValueError: If 'load_as_tmp' is False and no SSO ID is found in the meta dictionary.
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

    # meshes; added 'sj' because test-kzips do not contain 'syn_ssv' objects
    obj_types = set(global_params.config['process_cell_organelles']).union({'sv', 'sj'})
    for obj_type in obj_types:
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
    """
    Placeholder function for initializing a SuperSegmentationDataset (SSD) from multiple k.zip files.
    
    Args:
        dir_path (str): Directory path containing the k.zip files.
    
    Returns:
        None
    """
    pass
