import os
import numpy as np
import pickle as pkl
import sklearn.metrics as sm
from syconn import global_params
from scipy.spatial import cKDTree
from syconn.reps.super_segmentation import SuperSegmentationObject

if __name__ == "__main__":
    gt_path = os.path.expanduser('~/thesis/current_work/paper/data/syn_gt/')
    global_params.wd = "/wholebrain/songbird/j0126/areaxfs_v6/"
    with open(os.path.expanduser('~/thesis/current_work/paper/data/syn_gt/converted.pkl'), 'rb') as f:
        data = pkl.load(f)
    total_gt = np.array((0, 1))
    total_preds = np.array((0, 1))
    for key in data:
        if '_l' in key:
            pass
        sso_id = int(key[:-2])
        sso = SuperSegmentationObject(sso_id)
        mesh = sso.mesh[1].reshape((-1, 3))
        tree = cKDTree(mesh)
        coords = data[key]
        result = np.zeros((len(coords), 1))
        dist, ind = tree.query(coords, k=20)
        label_dc = sso.label_dict()
        # 0: dendrite, 1: axon, 2: soma, 3: neck, 4: head
        for ix in range(len(coords)):
            ads_preds = label_dc['ads'][ind[ix]]
            maj_vot = np.argmax(np.bincount(ads_preds))
            if maj_vot == 2 or maj_vot == 1:
                result[ix] = maj_vot
            elif maj_vot == 0:
                dnh_preds = label_dc['dnh'][ind[ix]]
                maj_vot = np.argmax(np.bincount(dnh_preds))
                if maj_vot == 1:
                    maj_vot = 3
                if maj_vot == 2:
                    maj_vot = 4
                result[ix] = maj_vot
        total_gt = np.concatenate((total_gt, data[str(sso_id)+'_l']))
        total_preds = np.concatenate((total_preds, result))
    targets = ['dendrite', 'axon', 'soma', 'neck', 'head']
    report = sm.classification_report(total_gt, total_preds, target_names=targets)
    print(report)
