import os
from typing import List
from morphx.classes.pointcloud import PointCloud
from syconn.reps.super_segmentation import SuperSegmentationDataset


def sso2hc(sso_ids: List[int], data: SuperSegmentationDataset, save_path: str):
    """
    Saves pointclouds of sso's with given ids from data with all
    available label predictions as pkl files to path.
    """
    save_path = os.path.expanduser(save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for sso_id in sso_ids:
        sso = data.get_super_segmentation_object(sso_id)
        verts = sso.load_mesh('sv')[1].reshape(-1, 3)
        for key in sso.label_dict():
            labels = sso.label_dict()[key]
            pc = PointCloud(vertices=verts, labels=labels)
            pc.save2pkl(f'{save_path}{sso_id}_{key}.pkl')


if __name__ == '__main__':
    ssd = SuperSegmentationDataset(working_dir="/wholebrain/songbird/j0126/areaxfs_v6/")
    path = '~/thesis/current_work/explore_syconn/'
    ids = [1090051, 2091009, 2734465, 2854913, 3447296, 8003584, 8339462, 10919937]
    sso2hc(ids, ssd, path)
