import os
import numpy as np
from typing import List, Optional
from morphx.classes.hybridcloud import HybridCloud
from morphx.processing.clouds import Center
from syconn.reps.super_segmentation import SuperSegmentationDataset


def sso2hc(sso_ids: List[int], data: SuperSegmentationDataset, save_path: str, label_keys: Optional[List[str]] = None):
    """
    Saves pointclouds of sso's with given ids from data with all
    available label predictions as pkl files to path.
    """
    save_path = os.path.expanduser(save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for sso_id in sso_ids:
        if sso_id not in data.ssv_ids:
            continue
        sso = data.get_super_segmentation_object(sso_id)
        verts = sso.load_mesh('sv')[1].reshape(-1, 3)
        if not sso.load_skeleton():
            raise ValueError(f'Couldnt find skeleton of {sso}')
        nodes, edges = sso.skeleton['nodes'] * sso.scaling, sso.skeleton['edges']
        if label_keys is None:
            hc = HybridCloud(vertices=verts, labels=np.ones((len(verts), 1)), nodes=nodes, edges=edges)
            hc.save2pkl(f'{save_path}{sso_id}_raw.pkl')
        else:
            for key in label_keys:
                labels = sso.label_dict()[key]
                hc = HybridCloud(vertices=verts, labels=labels, nodes=nodes, edges=edges)
                transform = Center()
                transform(hc)
                hc.save2pkl(f'{save_path}{sso_id}_{key}.pkl')


if __name__ == '__main__':
    ssd = SuperSegmentationDataset(
        working_dir="/wholebrain/scratch/arother/assembled_core_relabeled_base_merges_relabeled_to_v4b_base_20180214_full_agglo_cbsplit_with_reconnects_no_soma_merger_manual_edges_removed/")
    path = '~/thesis/current_work/explore_syconn/kimimaro/'
    ids = [25362218,   107283,  9221702,  1619851, 12231815, 34217175, 29327350, 25393993,  6993099, 22802417,
           2584916, 17920291, 16902380,  2169440, 32327585,  2218026, 24556061, 26148645, 29840046,  1226094]
    sso2hc(ids, ssd, path)
