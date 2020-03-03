import sys

try:
    import cPickle as pkl
except ImportError:
    import pickle as pkl
from syconn.proc.skeleton import kimimaro_skelgen

path_storage_file = sys.argv[1]
path_out_file = sys.argv[2]

with open(path_storage_file, 'rb') as f:
    args = []
    while True:
        try:
            args.append(pkl.load(f))
        except EOFError:
            break
cube_size, cube_offset, overlap = args
skels = kimimaro_skelgen(cube_size, cube_offset, overlap)

with open(path_out_file, "wb") as f:
    pkl.dump(skels, f)

