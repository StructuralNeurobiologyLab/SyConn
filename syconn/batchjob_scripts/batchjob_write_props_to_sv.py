# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Philipp Schubert, Jörgen Kornfeld

import sys
import pickle as pkl
from syconn.proc import sd_proc

path_storage_file = sys.argv[1]
path_out_file = sys.argv[2]

with open(path_storage_file, 'rb') as f:
    args = []
    while True:
        try:
            args.append(pkl.load(f))
        except EOFError:
            break

out = sd_proc._write_props_to_sv_thread(args)

with open(path_out_file, "wb") as f:
    pkl.dump(out, f)
