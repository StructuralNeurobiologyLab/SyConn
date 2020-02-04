import sys
try:
    import cPickle as pkl
except ImportError:
    import pickle as pkl
from neuronx.convpoint import e3_trainings as train

path_storage_file = sys.argv[1]
path_out_file = sys.argv[2]

with open(path_storage_file, 'rb') as f:
    args = []
    while True:
        try:
            args.append(pkl.load(f))
        except EOFError:
            break

out = train.training_thread(args)

with open(path_out_file, "wb") as f:
    pkl.dump(None, f)
