import sys
try:
    import cPickle as pkl
except ImportError:
    import pickle as pkl
from syconn.reps.super_segmentation_helper import pred_sv_chunk_semseg
from syconn import global_params
from syconn.handler import basics
from syconn.mp.mp_utils import start_multiprocess_imap

path_storage_file = sys.argv[1]
path_out_file = sys.argv[2]

with open(path_storage_file, 'rb') as f:
    args = []
    while True:
        try:
            args.append(pkl.load(f))
        except EOFError:
            break

so_chunk_paths = args[0]
so_kwargs = args[1]
pred_kwargs = args[2]
ncpus = global_params.config['ncores_per_node'] // global_params.config['ngpus_per_node']
# TODO: increase
n_worker = 1
params = [(paths, so_kwargs, pred_kwargs) for paths in basics.chunkify(so_chunk_paths, n_worker * 4)]

start_multiprocess_imap(pred_sv_chunk_semseg, params, nb_cpus=n_worker, debug=False)

with open(path_out_file, "wb") as f:
    pkl.dump("0", f)
