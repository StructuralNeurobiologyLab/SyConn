import os
import numpy as np
import pickle as pkl
import sklearn.metrics as sm
from syconn import global_params
from scipy.spatial import cKDTree
from .syconn_eval import replace_preds
from morphx.classes.pointcloud import PointCloud
from syconn.reps.super_segmentation import SuperSegmentationObject

""" works on ads and ds only """
if __name__ == "__main__":
    gt_path = os.path.expanduser('~/thesis/current_work/paper/data/syn_gt/')
    global_params.wd = "/wholebrain/scratch/areaxfs3/"
    with open(os.path.expanduser('~/thesis/current_work/paper/data/syn_gt/converted_v3.pkl'), 'rb') as f:
        data = pkl.load(f)
    save_path = os.path.expanduser(f'~/thesis/current_work/paper/syn_tests/2020_10_14_8000_8192_cp_cp_q/dh_syns/')
    save_path_examples = save_path + 'examples/'
    if not os.path.exists(save_path_examples):
        os.makedirs(save_path_examples)
    total_gt = None
    total_preds = None
    nn = 20
    for key in data:
        if '_l' not in key:
            sso_id = int(key[:-2])
            sso = SuperSegmentationObject(sso_id)

            # 0: dendrite, 1: axon, 2: soma
            pc_ads = replace_preds(sso, 'ads', [])
            # 0: dendrite, 1: neck, 2: head
            pc_dnh = replace_preds(sso, 'dnh_20_10_14_cp_cp_q', [])

            tree_ads = cKDTree(pc_ads.vertices)
            tree_dnh = cKDTree(pc_dnh.vertices)
            coords = data[key]
            result = np.zeros(len(coords))
            dist_ads, ind_ads = tree_ads.query(coords, k=nn)
            dist_dnh, ind_dnh = tree_dnh.query(coords, k=nn)
            gt = data[str(sso_id)+'_l']

            pc_ads.save2pkl(os.path.expanduser(save_path + f'{sso_id}_ads.pkl'))
            pc_dnh.save2pkl(os.path.expanduser(save_path + f'{sso_id}_dnh.pkl'))

            mask = np.ones((len(coords), 1), dtype=bool)

            # 0: dendrite, 1: axon, 2: head, 3: soma
            for ix in range(len(gt)):
                if gt[ix] == 1 or gt[ix] == 3:
                    mask[ix] = False
                    continue
                preds = pc_ads.labels[ind_ads[ix]].reshape(-1).astype(int)
                mv = np.argmax(np.bincount(preds))
                tree = tree_ads
                pc = pc_ads
                if mv != 0:
                    result[ix] = 3
                else:
                    preds = pc_dnh.labels[ind_dnh[ix]].reshape(-1).astype(int)
                    mv = np.argmax(np.bincount(preds))
                    tree = tree_dnh
                    pc = pc_dnh
                    if mv == 1:
                        result[ix] = 1
                    else:
                        result[ix] = mv
                if result[ix] != gt[ix]:
                    idcs_ball = tree.query_ball_point(coords[ix], 5000)
                    verts = np.concatenate((pc.vertices[idcs_ball], coords[ix].reshape((-1, 3))))
                    labels = np.concatenate((pc.labels[idcs_ball], np.array([4]).reshape((-1, 1))))
                    pc_local = PointCloud(vertices=verts, labels=labels)
                    pc_local.move(-coords[ix])
                    pc_local.save2pkl(save_path_examples + f'{sso_id}_{ix}_p{int(result[ix])}_gt{int(gt[ix])}.pkl')
            mask = mask.reshape(-1)
            if total_gt is None:
                total_gt = gt.reshape((-1, 1))[mask]
                total_preds = result.reshape((-1, 1))[mask]
            else:
                total_gt = np.concatenate((total_gt, gt.reshape((-1, 1))[mask]))
                total_preds = np.concatenate((total_preds, result.reshape((-1, 1))[mask]))

    targets = ['dendrite', 'neck', 'head']
    # targets = ['dendrite', 'head']
    # total_preds[total_preds == 2] = 1
    # total_gt[total_gt == 2] = 1
    report = sm.classification_report(total_gt, total_preds, target_names=targets)
    print(report)
