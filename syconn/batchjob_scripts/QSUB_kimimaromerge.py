import sys
from syconn.handler.basics import load_pkl2obj
try:
    import cPickle as pkl
except ImportError:
    import pickle as pkl
from syconn.proc.skeleton import kimimaro_mergeskels, kimimaro_skels_tokzip

path_storage_file = sys.argv[1]
path_out_file = sys.argv[2]

with open(path_storage_file, 'rb') as f:
    args = []
    while True:
        try:
            args.append(pkl.load(f))
        except EOFError:
            break

path2results_dc, cell_ids, zipname = args
restuls_dc = load_pkl2obj(path2results_dc)
full_skels = dict()

for cell_id in cell_ids:
    combined_skel = kimimaro_mergeskels(path2results_dc[cell_id], cell_id)
    kimimaro_skels_tokzip(combined_skel,cell_id, zipname)
    full_skels[cell_id] = combined_skel


with open(path_out_file, "wb") as f:
    pkl.dump(full_skels, f)

