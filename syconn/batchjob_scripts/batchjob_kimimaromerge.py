import sys
import os
import pickle as pkl
import numpy as np
from syconn.handler.basics import load_pkl2obj
from syconn.proc.skeleton import kimimaro_mergeskels
from syconn import global_params
from syconn.reps.super_segmentation_object import SuperSegmentationObject

path_storage_file = sys.argv[1]
path_out_file = sys.argv[2]

with open(path_storage_file, 'rb') as f:
    args = []
    while True:
        try:
            args.append(pkl.load(f))
        except EOFError:
            break

working_dir = global_params.config.working_dir
scaling = global_params.config["scaling"]
path2results_dc, ssv_ids = args
results_dc = load_pkl2obj(path2results_dc)

nb_cpus = os.environ.get('SLURM_CPUS_PER_TASK')
if nb_cpus is not None:
    nb_cpus = int(nb_cpus)

for ssv_id in ssv_ids:
    combined_skel = kimimaro_mergeskels(results_dc[ssv_id], ssv_id, nb_cpus=nb_cpus)
    sso = SuperSegmentationObject(ssv_id, working_dir=working_dir)

    sso.skeleton = dict()
    if combined_skel.vertices.size > 0:
        sso.skeleton["nodes"] = combined_skel.vertices / scaling  # to fit voxel coordinates
        # get radius in pseudo-voxel units (used by Knossos)
        sso.skeleton["diameters"] = (combined_skel.radii / scaling[0]) * 2  # divide by x scale
        sso.skeleton["edges"] = combined_skel.edges
    else:
        sso.skeleton["nodes"] = np.array([sso.rep_coord], dtype=np.float32)
        sso.skeleton["diameters"] = np.zeros((1, ), dtype=np.float32)
        sso.skeleton["edges"] = np.array([[0, 0], ], dtype=np.int64)
    sso.save_skeleton()

with open(path_out_file, "wb") as f:
    pkl.dump("0", f)

