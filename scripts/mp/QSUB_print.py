# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Philipp Schubert, Joergen Kornfeld

import sys
try:
    import cPickle as pkl
except ImportError:
    import pickle as pkl
import time
# get paths to job handling directories
path_storage_file = sys.argv[1]
path_out_file = sys.argv[2]

# retrieve all arguments
with open(path_storage_file, 'rb') as f:
    args = []
    while True:
        try:
            args.append(pkl.load(f))
        except EOFError:
            break

numbers = args[0]
for n in numbers:
    print(n)
    time.sleep(1)

with open(path_out_file, "wb") as f:
    pkl.dump(None, f)
