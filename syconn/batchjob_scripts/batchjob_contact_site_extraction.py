# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Philipp Schubert, Jörgen Kornfeld

import sys
import dill
# leave import here - needed by "shipping" lambda expression with numpy usage in arguments via "dill" package
import numpy as np
import pickle as pkl

from syconn.extraction import cs_extraction_steps

path_storage_file = sys.argv[1]
path_out_file = sys.argv[2]

with open(path_storage_file, 'rb') as f:
    args = []
    while True:
        try:
            args.append(dill.load(f))
        except EOFError:
            break

out = cs_extraction_steps._contact_site_extraction_thread(args)

with open(path_out_file, "wb") as f:
    pkl.dump(out, f)
