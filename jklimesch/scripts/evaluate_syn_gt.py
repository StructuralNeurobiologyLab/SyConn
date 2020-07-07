import os
import numpy as np
import pickle as pkl
import sklearn.metrics as sm
from syconn import global_params
from scipy.spatial import cKDTree
from neuronx.utils.syconn_eval import *
from morphx.classes.pointcloud import PointCloud
from syconn.reps.super_segmentation import SuperSegmentationObject

if __name__ == "__main__":
    gt_path = os.path.expanduser('~/thesis/current_work/paper/data/syn_gt/')
    global_params.wd = "/wholebrain/songbird/j0126/areaxfs_v6/"
    with open(os.path.expanduser('~/thesis/current_work/paper/data/syn_gt/converted.pkl'), 'rb') as f:
        data = pkl.load(f)
    total_gt = np.empty((1, 1))
    total_preds = np.empty((1, 1))
    for key in data:
        if '_l' not in key:
            sso_id = int(key[:-2])
            sso = SuperSegmentationObject(sso_id)

            pc = replace_preds(sso, 'ads', [(0, 'dnh', [(1, 5), (2, 6)]), (1, 'abt', [(0, 3), (2, 4)])])
            pc_syn = replace_preds(sso, 'ads', [(2, 'ads', [(2, 3)]), (0, 'dnh', [(1, 0)])])

            # pc.save2pkl(os.path.expanduser(f'~/thesis/tmp/{sso_id}_pure.pkl'))
            # pc_syn.save2pkl(os.path.expanduser(f'~/thesis/tmp/{sso_id}_syn.pkl'))

            tree = cKDTree(pc_syn.vertices)
            coords = data[key]
            result = np.zeros(len(coords))
            dist, ind = tree.query(coords, k=20)
            gt = data[str(sso_id)+'_l']

            print(f"{sso_id}: {len(pc.vertices)}")
            total_verts = np.concatenate((pc_syn.vertices, coords))
            total_labels = np.concatenate((pc_syn.labels, gt.reshape((-1, 1))))
            pc_total = PointCloud(vertices=total_verts, labels=total_labels)
            pc_total.save2pkl(os.path.expanduser(f'~/thesis/tmp/{sso_id}_total.pkl'))

            # 0: dendrite, 1: axon, 2: head, 3: soma
            for ix in range(len(coords)):
                preds = pc_syn.labels[ind[ix]].reshape(-1).astype(int)
                result[ix] = np.argmax(np.bincount(preds))
                # if result[ix] != gt[ix]:
                #     idcs_ball = tree.query_ball_point(coords[ix], 500)
                #     pc_local = PointCloud(vertices=pc_syn.vertices[idcs_ball], labels=pc_syn.labels[idcs_ball])
                #     pc_local.save2pkl(
                #         os.path.expanduser(f'~/thesis/tmp/{sso_id}_{ix}_p{int(result[ix])}_gt{int(gt[ix])}.pkl'))
            total_gt = np.concatenate((total_gt, gt.reshape((-1, 1))))
            total_preds = np.concatenate((total_preds, result.reshape((-1, 1))))

    targets = ['dendrite', 'axon', 'head', 'soma']
    report = sm.classification_report(total_gt, total_preds, target_names=targets)
    print(report)
