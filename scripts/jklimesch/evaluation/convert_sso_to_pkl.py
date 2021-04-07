import os
import numpy as np
import open3d as o3d
from typing import List, Optional, Dict
from morphx.classes.hybridcloud import HybridCloud
from morphx.classes.cloudensemble import CloudEnsemble
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


def sso2pickle(sso_ids: List[int], data: SuperSegmentationDataset, save_path: str, cos: List[str],
               voxel_dc: Dict[str, int]):
    save_path = os.path.expanduser(save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for sso_id in sso_ids:
        if sso_id not in data.ssv_ids:
            continue
        sso = data.get_super_segmentation_object(sso_id)
        clouds = {}
        ce_hc = None
        no_pred = []
        for co in cos:
            pcd = o3d.geometry.PointCloud()
            verts = sso.load_mesh(co)[1].reshape(-1, 3)
            pcd.points = o3d.utility.Vector3dVector(verts)
            pcd, _ = pcd.voxel_down_sample_and_trace(voxel_dc[co], pcd.get_min_bound(), pcd.get_max_bound())
            verts = np.asarray(pcd.points)
            if co == 'sv':
                sso.load_skeleton()
                nodes, edges = sso.skeleton['nodes'] * sso.scaling, sso.skeleton['edges']
                hc = HybridCloud(vertices=verts, nodes=nodes, edges=edges)
                ce_hc = hc
            else:
                hc = HybridCloud(vertices=verts)
                clouds[co] = hc
                no_pred.append(co)
        ce = CloudEnsemble(clouds, ce_hc, no_pred=no_pred)
        ce.save2pkl(f'{save_path}/sso_{sso.id}.pkl')


if __name__ == '__main__':
    # ssd = SuperSegmentationDataset(working_dir="/wholebrain/scratch/areaxfs3/")
    # path = '~/thesis/gt/syn_gt/'
    # ids = [141995, 11833344, 28410880, 28479489]
    # sso2pickle(ids, ssd, path, ['sv', 'mi', 'vc', 'sj'], voxel_dc={'sv': 80, 'mi': 100, 'vc': 100, 'sj': 100})

    ssd = SuperSegmentationDataset(working_dir="/wholebrain/scratch/areaxfs3/")
    path = '~/thesis/current_work/explore_syconn/inference_test/'
    ids = [141995]
    sso2hc(ids, ssd, path, label_keys=['ds', 'ads'])
