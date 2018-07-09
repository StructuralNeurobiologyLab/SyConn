# SyConnMP
# All rights reserved

import sys
import numpy as np
try:
    import cPickle as pkl
except ImportError:
    import pickle as pkl
import time
# get paths to job handling directories
path_storage_file = sys.argv[1]
path_out_file = sys.argv[2]

# retrieve all arguments
with open(path_storage_file) as f:
    args = []
    while True:
        try:
            args.append(pkl.load(f))
        except:
            break

numbers = args[0]
for n in numbers:
    print(n)
    time.sleep(0.2)
