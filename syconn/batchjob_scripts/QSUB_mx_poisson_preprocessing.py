import sys
try:
    import cPickle as pkl
except ImportError:
    import pickle as pkl
from morphx.preprocessing import mesh2poisson as m2p

path_storage_file = sys.argv[1]
path_out_file = sys.argv[2]

with open(path_storage_file, 'rb') as f:
    args = []
    while True:
        try:
            args.append(pkl.load(f))
        except EOFError:
            break

out = m2p.process_single_thread(args)

with open(path_out_file, "wb") as f:
    pkl.dump(None, f)
