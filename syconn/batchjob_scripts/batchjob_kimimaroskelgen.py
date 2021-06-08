import sys
import os
import pickle as pkl
from collections import defaultdict
import time
import tqdm

from syconn import global_params
from syconn.proc.skeleton import kimimaro_skelgen
from syconn.reps.super_segmentation import SuperSegmentationDataset


path_storage_file = sys.argv[1]
path_out_file = sys.argv[2]

with open(path_storage_file, 'rb') as f:
    args = []
    while True:
        try:
            args.append(pkl.load(f))
        except EOFError:
            break
cube_size, cube_offsets, ds = args
skel_params = global_params.config["skeleton"]['kimimaro_skelgen']
nb_cpus = os.environ.get('SLURM_CPUS_PER_TASK')
if nb_cpus is not None:
    nb_cpus = int(nb_cpus)

res = defaultdict(list)
res_ids = []
ssd = SuperSegmentationDataset(working_dir=global_params.config.working_dir)
for cube_offset in tqdm.tqdm(cube_offsets, total=len(cube_offsets)):
    skels = kimimaro_skelgen(cube_size, cube_offset, ds=ds, nb_cpus=nb_cpus, ssd=ssd,
                             **skel_params)
    for k, v in skels.items():
        res[k].append(v)

with open(path_out_file[:-4] + '_ids.pkl', "wb") as f:
    pkl.dump(list(res.keys()), f)

with open(path_out_file, "wb") as f:
    pkl.dump(res, f)

