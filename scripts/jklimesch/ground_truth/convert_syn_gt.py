import os
import re
import glob
import numpy as np
import pickle as pkl
from syconn import global_params
from knossos_utils.skeleton_utils import load_skeleton
from syconn.reps.super_segmentation import SuperSegmentationObject


def comment2int(comment: str):
    if comment == "shaft":
        return 0
    elif comment == "axon":
        return 1
    elif comment == "head":
        return 2
    elif comment == "soma":
        return 3
    else:
        return -1


def v32v6(v3_id: int):
    "Convert from areaxfs_v3 to areaxfs_v6"
    if v3_id == 141995:
        return 6201349
    elif v3_id == 11833344 or v3_id == 28479489:
        return v3_id
    elif v3_id == 28410880:
        return 14522373


if __name__ == '__main__':
    # load annotation object and corresponding skeleton
    gt_path = os.path.expanduser('~/thesis/current_work/paper/data/syn_gt/v6/')
    global_params.wd = "/wholebrain/songbird/j0126/areaxfs_v6/"
    kzips = glob.glob(gt_path + '*.k.zip')
    total = {}
    for kzip in kzips:
        sso_id = v32v6(int(re.findall(r"_(\d+).", kzip)[0]))
        sso = SuperSegmentationObject(sso_id)
        a_obj = load_skeleton(kzip)
        if len(a_obj) != 1:
            raise ValueError("File contains more or less than one skeleton!")
        a_obj = list(a_obj.values())[0]
        a_nodes = list(a_obj.getNodes())
        a_node_coords = np.array([n.getCoordinate() * sso.scaling for n in a_nodes])
        a_node_labels = np.array([comment2int(n.getComment()) for n in a_nodes], dtype=np.int)
        total[str(sso_id) + '_c'] = a_node_coords[a_node_labels != -1]
        total[str(sso_id) + '_l'] = a_node_labels[a_node_labels != -1]
    f = open(os.path.expanduser('~/thesis/current_work/paper/data/syn_gt/converted_v6_no_outlier.pkl'), 'wb')
    pkl.dump(total, f)
    f.close()
