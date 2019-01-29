# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max-Planck-Institute for Medical Research, Heidelberg, Germany
# Authors: Sven Dorkenwald, Philipp Schubert, Jörgen Kornfeld

import sys
import numpy as np
try:
    import cPickle as pkl
except ImportError:
    import pickle as pkl
from syconn.proc.sd_proc import sos_dict_fact, init_sos
from syconn.reps.rep_helper import surface_samples
from syconn.backend.storage import AttributeDict, CompressedStorage, MeshStorage
from syconn import global_params


path_storage_file = sys.argv[1]
path_out_file = sys.argv[2]

with open(path_storage_file, 'rb') as f:
    args = []
    while True:
        try:
            args.append(pkl.load(f))
        except EOFError:
            break

so_chunk_paths = args[0]
so_kwargs = args[1]

working_dir = so_kwargs['working_dir']
global_params.wd = working_dir
# TODO: preprocess meshes in case they dont exist and then load mesh dict next to the attribute dict
for p in so_chunk_paths:
    # get SV IDs stored in this storage
    attr_dc_p = p + "/attr_dict.pkl"
    mesh_dc_p = p + "/mesh.pkl"
    ad = AttributeDict(attr_dc_p, disable_locking=True)
    md = MeshStorage(mesh_dc_p, disable_locking=True)
    svixs = list(ad.keys())
    sd = sos_dict_fact(svixs, **so_kwargs)
    sos = init_sos(sd)
    # compute locations and use already processed meshes
    loc_dc_p = p + "/locations.pkl"
    loc_dc = CompressedStorage(loc_dc_p, disable_locking=True)
    for so in sos:
        try:
            ix = so.id
            if not ix in md.keys():
                verts = so.mesh[1].reshape(-1, 3)
            else:
                verts = md[ix][1].reshape(-1, 3)
            if len(verts) == 0:
                coords = np.array([so.rep_coord, ], dtype=np.float32)
            else:
                coords = surface_samples(verts).astype(np.float32)
            loc_dc[ix] = coords
        except Exception as e:
            print('ERROR during sample location generation of SV {}: {}'.format(so.id, str(e)))
    loc_dc.push()

with open(path_out_file, "wb") as f:
    pkl.dump("0", f)
