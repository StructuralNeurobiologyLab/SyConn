import sys
from syconn.handler.prediction import dense_predictor #stored at right location?

try:
    import cPickle as pkl
except ImportError:
    import pickle as pkl

path_storage_file = sys.argv[1]
path_out_file = sys.argv[2]

with open(path_storage_file, 'rb') as f:
    args = []
    while True:
        try:
            args.append(pkl.load(f))
        except EOFError:
            break

dense_predictor(args)

with open(path_out_file, "wb") as f:
    pkl.dump(None, f)
