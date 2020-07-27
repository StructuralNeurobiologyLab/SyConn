import os
import numpy as np
import pickle as pkl
import sklearn.metrics as sm
from syconn import global_params
from scipy.spatial import cKDTree
from neuronx.utils.syconn_eval import *
from morphx.classes.pointcloud import PointCloud
from morphx.processing import clouds
from syconn.reps.super_segmentation import SuperSegmentationObject


""" Checks ads prediction first and falls back to dnh if ads prediciton results in 0 (dendrite). """
# if __name__ == "__main__":
#     gt_path = os.path.expanduser('~/thesis/current_work/paper/data/syn_gt/')
#     global_params.wd = "/wholebrain/scratch/areaxfs3/"
#     with open(os.path.expanduser('~/thesis/current_work/paper/data/syn_gt/converted_v3.pkl'), 'rb') as f:
#         data = pkl.load(f)
#     total_gt = np.empty((1, 1))
#     total_preds = np.empty((1, 1))
#     nn = 20
#     print(nn)
#     for key in data:
#         if '_l' not in key:
#             sso_id = int(key[:-2])
#             sso = SuperSegmentationObject(sso_id)
#
#             # 0: dendrite, 1: axon, 2: soma
#             pc_ads = replace_preds(sso, 'ads', [])
#             # 0: dendrite, 1: neck, 2: head
#             pc_dnh = replace_preds(sso, 'dnh', [(1, 'dnh', [(1, 0)])])
#
#             tree_ads = cKDTree(pc_ads.vertices)
#             tree_dnh = cKDTree(pc_dnh.vertices)
#             coords = data[key]
#             result = np.zeros(len(coords))
#             dist_ads, ind_ads = tree_ads.query(coords, k=nn)
#             dist_dnh, ind_dnh = tree_dnh.query(coords, k=nn)
#             gt = data[str(sso_id)+'_l']
#
#             # print(f"{sso_id}: {len(pc.vertices)}")
#             # total_verts = np.concatenate((pc_syn.vertices, coords))
#             # total_labels = np.concatenate((pc_syn.labels, gt.reshape((-1, 1))))
#             # pc_total = PointCloud(vertices=total_verts, labels=total_labels)
#             # pc_total.save2pkl(os.path.expanduser(f'~/thesis/tmp/{sso_id}_total.pkl'))
#
#             # 0: dendrite, 1: axon, 2: head, 3: soma
#             for ix in range(len(coords)):
#                 preds = pc_ads.labels[ind_ads[ix]].reshape(-1).astype(int)
#                 mv = np.argmax(np.bincount(preds))
#                 tree = tree_ads
#                 pc = pc_ads
#                 if mv != 0:
#                     if mv == 2:
#                         result[ix] = 3
#                     else:
#                         result[ix] = mv
#                 else:
#                     preds = pc_dnh.labels[ind_dnh[ix]].reshape(-1).astype(int)
#                     result[ix] = np.argmax(np.bincount(preds))
#                     tree = tree_dnh
#                     pc = pc_dnh
#                 if result[ix] != gt[ix]:
#                     idcs_ball = tree.query_ball_point(coords[ix], 2000)
#                     verts = np.concatenate((pc.vertices[idcs_ball], coords[ix].reshape((-1, 3))))
#                     labels = np.concatenate((pc.labels[idcs_ball], np.array([4]).reshape((-1, 1))))
#                     pc_local = PointCloud(vertices=verts, labels=labels)
#                     pc_local.move(-coords[ix])
#                     pc_local.save2pkl(
#                         os.path.expanduser(f'~/thesis/tmp/{sso_id}_{ix}_p{int(result[ix])}_gt{int(gt[ix])}.pkl'))
#             total_gt = np.concatenate((total_gt, gt.reshape((-1, 1))))
#             total_preds = np.concatenate((total_preds, result.reshape((-1, 1))))
#
#     targets = ['dendrite', 'axon', 'head', 'soma']
#     report = sm.classification_report(total_gt, total_preds, target_names=targets)
#     print(report)


""" Merges ads and dnh prediction. """
if __name__ == "__main__":
    gt_path = os.path.expanduser('~/thesis/current_work/paper/data/syn_gt/')
    global_params.wd = "/wholebrain/songbird/j0126/areaxfs_v6/"
    with open(os.path.expanduser('~/thesis/current_work/paper/data/syn_gt/converted_v6_no_outlier.pkl'), 'rb') as f:
        data = pkl.load(f)
    total_gt = np.empty((1, 1))
    total_preds = np.empty((1, 1))
    nn = 50
    print(nn)
    for key in data:
        if '_l' not in key:
            sso_id = int(key[:-2])
            sso = SuperSegmentationObject(sso_id)

            pc = replace_preds(sso, 'ads', [(0, 'dnh', [(1, 5), (2, 6)]), (1, 'abt', [(0, 3), (2, 4)])])
            pc_syn = replace_preds(sso, 'ads', [(2, 'ads', [(2, 3)]), (0, 'dnh', [(1, 2)])])

            # pc.save2pkl(os.path.expanduser(f'~/thesis/tmp/{sso_id}_pure.pkl'))
            # pc_syn.save2pkl(os.path.expanduser(f'~/thesis/tmp/{sso_id}_syn.pkl'))

            tree = cKDTree(pc_syn.vertices)
            coords = data[key]
            result = np.zeros(len(coords))
            dist, ind = tree.query(coords, k=nn)
            gt = data[str(sso_id)+'_l']

            # print(f"{sso_id}: {len(pc.vertices)}")
            # total_verts = np.concatenate((pc_syn.vertices, coords))
            # total_labels = np.concatenate((pc_syn.labels, gt.reshape((-1, 1))))
            # pc_total = PointCloud(vertices=total_verts, labels=total_labels)
            # pc_total.save2pkl(os.path.expanduser(f'~/thesis/tmp/{sso_id}_total.pkl'))

            # 0: dendrite, 1: axon, 2: head, 3: soma
            for ix in range(len(coords)):
                preds = pc_syn.labels[ind[ix]].reshape(-1).astype(int)
                result[ix] = np.argmax(np.bincount(preds))
                if result[ix] != gt[ix]:
                    idcs_ball = tree.query_ball_point(coords[ix], 2000)
                    verts = np.concatenate((pc_syn.vertices[idcs_ball], coords[ix].reshape((-1, 3))))
                    labels = np.concatenate((pc_syn.labels[idcs_ball], np.array([4]).reshape((-1, 1))))
                    pc_local = PointCloud(vertices=verts, labels=labels)
                    pc_local.move(-coords[ix])
                    pc_local.save2pkl(
                        os.path.expanduser(f'~/thesis/tmp/{sso_id}_{ix}_p{int(result[ix])}_gt{int(gt[ix])}.pkl'))
            total_gt = np.concatenate((total_gt, gt.reshape((-1, 1))))
            total_preds = np.concatenate((total_preds, result.reshape((-1, 1))))

    targets = ['dendrite', 'axon', 'head', 'soma']
    report = sm.classification_report(total_gt, total_preds, target_names=targets)
    print(report)


""" Removes axon and soma. """
# if __name__ == "__main__":
#     gt_path = os.path.expanduser('~/thesis/current_work/paper/data/syn_gt/')
#     global_params.wd = "/wholebrain/scratch/areaxfs3/"
#     with open(os.path.expanduser('~/thesis/current_work/paper/data/syn_gt/converted_v3.pkl'), 'rb') as f:
#         data = pkl.load(f)
#     total_gt = np.empty((1, 1))
#     total_preds = np.empty((1, 1))
#     for key in data:
#         if '_l' not in key:
#             sso_id = int(key[:-2])
#             sso = SuperSegmentationObject(sso_id)
#
#             pc = replace_preds(sso, 'ads', [(0, 'dnh', [(1, 5), (2, 6)]), (1, 'abt', [(0, 3), (2, 4)])])
#             pc_syn = replace_preds(sso, 'ads', [(2, 'ads', [(2, -1), (1, -1)]), (0, 'dnh', [(1, 0)])])
#
#             # pc.save2pkl(os.path.expanduser(f'~/thesis/tmp/{sso_id}_pure.pkl'))
#             # pc_syn.save2pkl(os.path.expanduser(f'~/thesis/tmp/{sso_id}_syn.pkl'))
#
#             tree = cKDTree(pc_syn.vertices)
#             coords = data[key]
#             gt = data[str(sso_id)+'_l']
#             d_mask = gt == 0
#             h_mask = gt == 2
#             dh_mask = np.logical_or(d_mask, h_mask).reshape(-1)
#             gt = gt[dh_mask]
#             coords = coords[dh_mask]
#             result = np.zeros(len(coords))
#             dist, ind = tree.query(coords, k=20)
#
#             print(f"{sso_id}: {len(pc.vertices)}")
#             total_verts = np.concatenate((pc_syn.vertices, coords))
#             total_labels = np.concatenate((pc_syn.labels, gt.reshape((-1, 1))))
#             pc_total = PointCloud(vertices=total_verts, labels=total_labels)
#             pc_total.save2pkl(os.path.expanduser(f'~/thesis/tmp/{sso_id}_total.pkl'))
#
#             # 0: dendrite, 1: axon, 2: head, 3: soma
#             for ix in range(len(coords)):
#                 preds = pc_syn.labels[ind[ix]].reshape(-1).astype(int)
#                 result[ix] = np.argmax(np.bincount(preds))
#                 # if result[ix] != gt[ix]:
#                 #     idcs_ball = tree.query_ball_point(coords[ix], 500)
#                 #     pc_local = PointCloud(vertices=pc_syn.vertices[idcs_ball], labels=pc_syn.labels[idcs_ball])
#                 #     pc_local.save2pkl(
#                 #         os.path.expanduser(f'~/thesis/tmp/{sso_id}_{ix}_p{int(result[ix])}_gt{int(gt[ix])}.pkl'))
#             total_gt = np.concatenate((total_gt, gt.reshape((-1, 1))))
#             total_preds = np.concatenate((total_preds, result.reshape((-1, 1))))
#
#     targets = ['dendrite', 'head']
#     total_gt[total_gt == 2] = 1
#     total_preds[total_preds == 2] = 1
#     report = sm.classification_report(total_gt, total_preds, target_names=targets)
#     print(report)