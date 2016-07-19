import sys

import cPickle as pickle
from syconn.processing import objectextraction_helper as oeh

path_storage_file = sys.argv[1]
path_out_file = sys.argv[2]

with open(path_storage_file) as f:
    args = []
    while True:
        try:
            args.append(pickle.load(f))
        except:
            break

oeh.create_objects_from_voxels_thread(args)

