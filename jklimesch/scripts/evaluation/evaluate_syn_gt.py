import os
import numpy as np
import pickle as pkl
import sklearn.metrics as sm
from syconn import global_params
from scipy.spatial import cKDTree
from syconn_eval import replace_preds
from morphx.classes.pointcloud import PointCloud
from syconn.reps.super_segmentation import SuperSegmentationObject

base = '~/working_dir/paper/dnh_model_comparison/'
path_list = [('2020_10_14_8000_8192_cp_cp_q', 570),
             ('2020_11_08_2000_2048_cp_cp_q', 90),
             ('2020_11_08_2000_2048_cp_cp_q_2', 90),
             ('2020_11_08_8000_2048_cp_cp_q', 480),
             ('2020_11_08_8000_2048_cp_cp_q_2', 480),
             ('2020_11_08_8000_8192_cp_cp_q', 300),
             ('2020_11_08_8000_8192_cp_cp_q_co', 510),
             ('2020_11_08_8000_8192_cp_cp_q_co_2', 510),
             ('2020_11_08_8000_8192_cp_cp_q_nn', 450),
             ('2020_11_08_8000_8192_cp_cp_q_nn_2', 420),
             ('2020_11_08_8000_32768_cp_cp_q', 120),
             ('2020_11_08_8000_32768_cp_cp_q_2', 270),
             ('2020_11_08_24000_32768_cp_cp_q', 570),
             ('2020_11_08_24000_32768_cp_cp_q_2', 540),
             ('2020_11_09_2000_2048_cp_cp_q_3', 120),
             ('2020_11_09_8000_1024_cp_cp_q', 600),
             ('2020_11_09_8000_1024_cp_cp_q_2', 600),
             ('2020_11_09_8000_8192_cp_cp_q_2', 360),
             ('2020_11_09_8000_8192_cp_cp_q_bn', 540),
             ('2020_11_09_8000_8192_cp_cp_q_bn_2', 570),
             ('2020_11_11_2000_2048_cp_cp_q_3', 360),
             ('2020_11_11_8000_1024_cp_cp_q_3', 450),
             ('2020_11_11_8000_2048_cp_cp_q_3', 450),
             ('2020_11_11_8000_8192_cp_cp_q_bn_3', 480),
             ('2020_11_11_8000_8192_cp_cp_q_co_3', 450),
             ('2020_11_11_8000_32768_cp_cp_q_3', 390),
             ('2020_11_11_24000_32768_cp_cp_q_3', 480),
             ('2020_11_16_8000_8192_cp_cp_q_nn_3', 690),
             ('2020_11_16_8000_8192_cp_cp_q_nn_4', 690)]


for red in range(5, 6):
    for path in path_list:
        global_params.wd = "/wholebrain/scratch/areaxfs3/"
        with open(os.path.expanduser('~/working_dir/gt/syn_gt/converted_v3.pkl'), 'rb') as f:
            data = pkl.load(f)
        save_path = os.path.expanduser(base + path[0] + f'/syn_eval_dnh_red{red}_border/')
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
                pc_dnh = replace_preds(sso, f'{path[0]}_e{path[1]}_red{red}_border', [])

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
                        pc_local.save2pkl(save_path_examples + f'{sso_id}_{ix}_p{int(result[ix])}_gt{int(gt[ix])}.pkl')
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
        with open(save_path + 'report.txt', 'w') as f:
            f.write(report)
        with open(save_path + 'report.pkl', 'wb') as f:
            pkl.dump(report_dict, f)
        print(report)
