import os
import numpy as np
import pickle as pkl
import sklearn.metrics as sm
from syconn import global_params
from scipy.spatial import cKDTree
from utils import merge
from morphx.classes.pointcloud import PointCloud
from syconn.reps.super_segmentation import SuperSegmentationDataset


def write_confusion_matrix(cm: np.array, names: list) -> str:
    txt = f"{'':<15}"
    for name in names:
        txt += f"{name:<15}"
    txt += '\n'
    for ix, name in enumerate(names):
        txt += f"{name:<15}"
        for num in cm[ix]:
            txt += f"{num:<15}"
        txt += '\n'
    return txt


# synapse GT labels: 0: dendrite, 1: axon, 2: head, 3: soma
mapping = {0: 0, 1: 1, 2: 6, 3: 2}
exclude = []
red = 5

if __name__ == "__main__":
    with open(os.path.expanduser('~/working_dir/gt/syn_gt/converted_v3.pkl'), 'rb') as f:
        data = pkl.load(f)
    ssd = SuperSegmentationDataset(working_dir="/wholebrain/scratch/areaxfs3/")
    save_path = os.path.expanduser(f'~/working_dir/paper/hierarchy/')
    save_path_examples = save_path + 'examples/'
    if not os.path.exists(save_path_examples):
        os.makedirs(save_path_examples)
    total_gt = np.empty((0, 1))
    total_preds = np.empty((0, 1))
    nn = 20
    error_count = 0
    for key in data:
        if '_l' not in key:
            sso_id = int(key[:-2])
            sso = ssd.get_super_segmentation_object(sso_id)

            # 0: axon, 1: bouton, 2: terminal
            with open(save_path + f'abt/syn_eval_red{red}/' + str(sso_id) + '.pkl', 'rb') as f:
                abt = pkl.load(f)
            # 0: dendrite, 1: neck, 2: head
            with open(save_path + f'dnh/syn_eval_red{red}/' + str(sso_id) + '.pkl', 'rb') as f:
                dnh = pkl.load(f)
            # 0: dendrite, 1: axon, 2: soma,
            with open(save_path + f'ads/syn_eval_red{red}/' + str(sso_id) + '.pkl', 'rb') as f:
                ads = pkl.load(f)

            pc = merge(sso, ads, {1: (abt, [(1, 3), (2, 4), (0, 1)]), 0: (dnh, [(1, 5), (2, 6)])})
            pc.save2pkl(save_path + str(sso_id) + '.pkl')

            # query synapse coordinates in KDTree of vertices
            tree = cKDTree(pc.vertices)
            coords = data[key]
            result = np.zeros(len(coords))
            dist, ind = tree.query(coords, k=nn)
            gt = data[str(sso_id)+'_l']

            mask = np.ones((len(coords), 1), dtype=bool)
            for ix in range(len(gt)):
                gt[ix] = mapping[gt[ix]]
                if gt[ix] in exclude:
                    mask[ix] = False
                preds = pc.labels[ind[ix]].reshape(-1).astype(int)
                mv = np.argmax(np.bincount(preds))
                if mv in [3, 4]:
                    mv = 1
                result[ix] = mv
                if result[ix] != gt[ix] and mask[ix]:
                    error_count += 1
                    idcs_ball = tree.query_ball_point(coords[ix], 5000)
                    verts = np.concatenate((pc.vertices[idcs_ball], coords[ix].reshape((-1, 3))))
                    labels = np.concatenate((pc.labels[idcs_ball], np.array([4]).reshape((-1, 1))))
                    pc_local = PointCloud(vertices=verts, labels=labels)
                    pc_local.move(-coords[ix])
                    pc_local.save2pkl(save_path_examples + f'{sso_id}_{ix}_p{int(result[ix])}_gt{int(gt[ix])}.pkl')
            mask = mask.reshape(-1)
            total_gt = np.concatenate((total_gt, gt.reshape((-1, 1))[mask]))
            total_preds = np.concatenate((total_preds, result.reshape((-1, 1))[mask]))

    labels = ['dendrite', 'axon', 'soma', 'bouton', 'terminal', 'neck', 'head']
    targets = ['dendrite', 'axon', 'soma', 'neck', 'head']
    exclude = [labels[i] for i in exclude]
    report = f'Excluded synapse labels: {exclude}\n\n'
    report += sm.classification_report(total_gt, total_preds, target_names=targets)
    cm = sm.confusion_matrix(total_gt, total_preds)
    report += '\n\n'
    report += write_confusion_matrix(cm, targets)
    report += f'\n\nNumber of errors: {error_count}'
    with open(save_path + 'report.txt', 'w') as f:
        f.write(report)
    print(report)
