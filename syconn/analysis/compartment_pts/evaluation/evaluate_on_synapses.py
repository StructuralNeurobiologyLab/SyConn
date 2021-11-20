import os
import numpy as np
import pickle as pkl
import sklearn.metrics as sm
from syconn import global_params
from scipy.spatial import cKDTree
from .utils import replace_preds
from morphx.classes.pointcloud import PointCloud
from syconn.reps.super_segmentation import SuperSegmentationObject


def evaluate_syn_thread(args):
    base = args[0]
    path = args[1]
    do_pred_key = args[2]
    global_params.wd = "/wholebrain/scratch/areaxfs3/"
    with open(os.path.expanduser('/wholebrain/scratch/jklimesch/gt/syn_gt/converted_v3.pkl'), 'rb') as f:
        data = pkl.load(f)
    files = os.listdir(base + path)
    save_path = ''
    for file in files:
        if 'syn_e_final' in file:
            save_path = os.path.expanduser(base + path + '/' + file + '/')
            break
    save_path_examples = save_path + 'examples/'
    save_path_log = save_path + 'log/'
    if not os.path.exists(save_path_examples):
        os.makedirs(save_path_examples)
    if not os.path.exists(save_path_log):
        os.makedirs(save_path_log)
    total_gt = None
    total_preds = None
    nn = 20
    for key in data:
        if '_l' not in key:
            sso_id = int(key[:-2])
            sso = SuperSegmentationObject(sso_id)

            with open(save_path + str(sso_id) + '.pkl', 'rb') as f:
                preds = pkl.load(f)

            # 0: dendrite, 1: axon, 2: soma
            pc_ads = replace_preds(sso, do_pred_key, [])
            # 0: dendrite, 1: neck, 2: head
            pc_dnh = replace_preds(sso, preds, [])

            tree_ads = cKDTree(pc_ads.vertices)
            tree_dnh = cKDTree(pc_dnh.vertices)
            coords = data[key]
            result = np.zeros(len(coords))
            dist_ads, ind_ads = tree_ads.query(coords, k=nn)
            dist_dnh, ind_dnh = tree_dnh.query(coords, k=nn)
            gt = data[str(sso_id)+'_l']

            pc_ads.save2pkl(os.path.expanduser(save_path_log + f'{sso_id}_do.pkl'))
            pc_dnh.save2pkl(os.path.expanduser(save_path_log + f'{sso_id}_dnh.pkl'))

            mask = np.ones((len(coords), 1), dtype=bool)

            # 0: dendrite, 1: neck, 2: head, 3: other
            for ix in range(len(gt)):
                if gt[ix] == 1 or gt[ix] == 3:
                    mask[ix] = False
                    continue
                preds = pc_ads.labels[ind_ads[ix]].reshape(-1).astype(int)
                mv = np.argmax(np.bincount(preds))
                tree = tree_ads
                pc = pc_ads
                if mv == 1:
                    result[ix] = 3
                elif mv == 2:
                    result[ix] = 4
                else:
                    preds = pc_dnh.labels[ind_dnh[ix]].reshape(-1).astype(int)
                    mv = np.argmax(np.bincount(preds))
                    tree = tree_dnh
                    pc = pc_dnh
                    result[ix] = mv
                if result[ix] != gt[ix] and mask[ix]:
                    idcs_ball = tree.query_ball_point(coords[ix], 5000)
                    verts = np.concatenate((pc.vertices[idcs_ball], coords[ix].reshape((-1, 3))))
                    labels = np.concatenate((pc.labels[idcs_ball], np.array([4]).reshape((-1, 1))))
                    pc_local = PointCloud(vertices=verts, labels=labels)
                    pc_local.move(-coords[ix])
                    pc_local.save2pkl(save_path_examples + f'{sso_id}_{ix}_p{int(result[ix])}'
                                                           f'_gt{int(gt[ix])}_{do_pred_key}.pkl')
            mask = mask.reshape(-1)
            if total_gt is None:
                total_gt = gt.reshape((-1, 1))[mask]
                total_preds = result.reshape((-1, 1))[mask]
            else:
                total_gt = np.concatenate((total_gt, gt.reshape((-1, 1))[mask]))
                total_preds = np.concatenate((total_preds, result.reshape((-1, 1))[mask]))

    unique = np.unique(total_preds)
    # axon and soma are not present in gt (get filtered out), therefore neck can take place of axon
    targets = ['dendrite', 'neck', 'head', 'axon', 'soma']
    targets = [targets[int(i)] for i in unique]
    report = sm.classification_report(total_gt, total_preds, target_names=targets)
    report_dict = sm.classification_report(total_gt, total_preds, target_names=targets, output_dict=True)
    with open(f'{save_path_log}report_{do_pred_key}.txt', 'w') as f:
        f.write(report)
    with open(f'{save_path_log}report_{do_pred_key}.pkl', 'wb') as f:
        pkl.dump(report_dict, f)
    print(report)


def evaluate_syn_thread_dnho(base_dir, pred_key):
    global_params.wd = "/wholebrain/scratch/areaxfs3/"
    with open(os.path.expanduser('/wholebrain/scratch/jklimesch/gt/syn_gt/converted_v3.pkl'), 'rb') as f:
        data = pkl.load(f)
    files = os.listdir(base_dir)
    save_path = ''
    for file in files:
        if pred_key in file:
            save_path = os.path.expanduser(base_dir + '/' + file + '/')
            break
    save_path_examples = save_path + 'examples/'
    save_path_log = save_path + 'log/'
    if not os.path.exists(save_path_examples):
        os.makedirs(save_path_examples)
    if not os.path.exists(save_path_log):
        os.makedirs(save_path_log)
    total_gt = None
    total_preds = None
    nn = 20
    report_str = ""
    for key in data:
        if '_l' not in key:
            sso_id = int(key[:-2])
            sso = SuperSegmentationObject(sso_id)

            with open(save_path + str(sso_id) + '.pkl', 'rb') as f:
                preds = pkl.load(f)

            # 0: dendrite, 1: neck, 2: head, 3: other
            pc_dnho = replace_preds(sso, preds, [])

            tree_dnho = cKDTree(pc_dnho.vertices)
            coords = data[key]
            result = np.zeros(len(coords))
            _, ind_dnho = tree_dnho.query(coords, k=nn)
            gt = data[str(sso_id)+'_l']

            pc_dnho.save2pkl(os.path.expanduser(save_path_log + f'{sso_id}_dnho.pkl'))

            mask = np.ones((len(coords), 1), dtype=bool)

            # 0: dendrite, 1: neck, 2: head, 3: other
            for ix in range(len(gt)):
                if gt[ix] == 1 or gt[ix] == 3:
                    mask[ix] = False
                    continue
                preds = pc_dnho.labels[ind_dnho[ix]].reshape(-1).astype(int)
                mv = np.argmax(np.bincount(preds))

                if mv == 3:
                    result[ix] = 3
                else:
                    result[ix] = mv
                if result[ix] != gt[ix] and mask[ix]:
                    idcs_ball = tree_dnho.query_ball_point(coords[ix], 5000)
                    verts = np.concatenate((pc_dnho.vertices[idcs_ball], coords[ix].reshape((-1, 3))))
                    labels = np.concatenate((pc_dnho.labels[idcs_ball], np.array([4]).reshape((-1, 1))))
                    pc_local = PointCloud(vertices=verts, labels=labels)
                    pc_local.move(-coords[ix])
                    pc_local.save2pkl(save_path_examples + f'{sso_id}_{ix}_p{int(result[ix])}_gt{int(gt[ix])}.pkl')
                    report_str += f'\nPred: {result[ix]} Gt: {gt[ix]} loc [voxels]: {coords[ix] / sso.scaling}'
            mask = mask.reshape(-1)
            if total_gt is None:
                total_gt = gt.reshape((-1, 1))[mask]
                total_preds = result.reshape((-1, 1))[mask]
            else:
                total_gt = np.concatenate((total_gt, gt.reshape((-1, 1))[mask]))
                total_preds = np.concatenate((total_preds, result.reshape((-1, 1))[mask]))

    unique = np.unique(total_preds)
    # axon and soma are not present in gt (get filtered out), therefore neck can take place of axon
    targets = ['dendrite', 'neck', 'head', 'other']
    targets = [targets[int(i)] for i in unique]
    report = sm.classification_report(total_gt, total_preds, target_names=targets)
    report_dict = sm.classification_report(total_gt, total_preds, target_names=targets, output_dict=True)
    with open(save_path_log + 'report.txt', 'w') as f:
        f.write(report)
    with open(save_path_log + 'report.pkl', 'wb') as f:
        pkl.dump(report_dict, f)
    return report_str + '\n' + report
