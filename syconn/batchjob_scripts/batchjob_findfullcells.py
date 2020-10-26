# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Alexandra Rother
import sys
import os
import pickle as pkl
from collections import defaultdict
import numpy as np

from syconn.proc.analysis_prep_func import find_full_cells_sso

path_storage_file = sys.argv[1]
path_out_file = sys.argv[2]

with open(path_storage_file, 'rb') as f:
    args = []
    while True:
        try:
            args.append(pkl.load(f))
        except EOFError:
            break
cellids, celltype = args

nb_cpus = os.environ.get('SLURM_CPUS_PER_TASK')
if nb_cpus is not None:
    nb_cpus = int(nb_cpus)

res = defaultdict(np.zeros(3))
resid = np.zeros(len(cellids))

for ix, cellid in enumerate(cellids):
    cellid, soma_centre = find_full_cells_sso(cellid, celltype)
    if cellid == 0:
        continue
    if type(soma_centre) != np.ndarray:
        res[cellid] = soma_centre
        resid[ix] = cellid
    else:
        resid[ix] = cellid

resid = resid[resid != 0]
if len(resid) != 0:
    with open(path_out_file, "wb") as f:
        pkl.dump(resid, f)

    if len(res) != 0:
        with open(path_out_file, "wb") as f:
            pkl.dump([resid, res], f)

with open(path_out_file, "wb") as f:
    pkl.dump("0", f)
