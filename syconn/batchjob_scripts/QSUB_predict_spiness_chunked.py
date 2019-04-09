import sys
try:
    import cPickle as pkl
except ImportError:
    import pickle as pkl
from syconn.reps.super_segmentation_helper import pred_sv_chunk_semseg

path_storage_file = sys.argv[1]
path_out_file = sys.argv[2]

with open(path_storage_file, 'rb') as f:
    args = []
    while True:
        try:
            args.append(pkl.load(f))
        except EOFError:
            break

pred_sv_chunk_semseg(args)

with open(path_out_file, "wb") as f:
    pkl.dump("0", f)
