import sys
import os
import pickle as pkl
from multiprocessing import cpu_count

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
cube_size, cube_offset, overlap, cube_of_interest_bb = args

nb_cpus = os.environ.get('SLURM_CPUS_PER_TASK')
if nb_cpus is not None:
    nb_cpus = int(nb_cpus)
skels = kimimaro_skelgen(cube_size, cube_offset, overlap, cube_of_interest_bb, nb_cpus=nb_cpus)

with open(path_out_file[:-4] + '_ids.pkl', "wb") as f:
    pkl.dump(list(skels.keys()), f)

with open(path_out_file, "wb") as f:
    pkl.dump(skels, f)

