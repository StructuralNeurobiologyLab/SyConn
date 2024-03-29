import sys
try:
    import cPickle as pkl
except ImportError:
    import pickle as pkl

from syconn.analysis.compartment_pts.evaluation.evaluate_on_synapses import evaluate_syn_thread

path_storage_file = sys.argv[1]
path_out_file = sys.argv[2]

with open(path_storage_file, 'rb') as f:
    args = []
    while True:
        try:
            args.append(pkl.load(f))
        except EOFError:
            break

evaluate_syn_thread(args[0])

with open(path_out_file, "wb") as f:
    pkl.dump(None, f)
