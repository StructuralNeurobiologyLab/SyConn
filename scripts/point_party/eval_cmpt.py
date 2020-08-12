import os
from tqdm import tqdm
import numpy as np
from syconn.reps.super_segmentation import SuperSegmentationObject, SuperSegmentationDataset
from morphx.classes.pointcloud import PointCloud

if __name__ == '__main__':
    wd = os.path.expanduser('~/SyConn/example_cube3/')
    save_path = os.path.expanduser('~/thesis/current_work/paper/pipeline_tests/example_cube3/')
    ssd = SuperSegmentationDataset(wd)
    total_verts = None
    total_labels = None
    for ix in tqdm(range(len(ssd.ssv_ids))):
        id = ssd.ssv_ids[ix]
        sso = SuperSegmentationObject(id)
        verts = sso.mesh[1].reshape((-1, 3))
        ads = sso.label_dict()['ads']
        abt = sso.label_dict()['abt']
        dnh = sso.label_dict()['dnh']
        a_mask = (ads == 1).reshape(-1)
        d_mask = (ads == 0).reshape(-1)
        abt[abt == 0] = 3
        abt[abt == 2] = 4
        dnh[dnh == 1] = 5
        dnh[dnh == 2] = 6
        ads[a_mask] = abt[a_mask]
        ads[d_mask] = dnh[d_mask]
        if ix % 10 == 0:
            pc = PointCloud(vertices=verts, labels=ads)
            pc.save2pkl(save_path + f"{id}.pkl")
        if total_verts is None:
            total_verts = verts
            total_labels = ads
        else:
            total_verts = np.concatenate((total_verts, verts))
            total_labels = np.concatenate((total_labels, ads))
    total_pc = PointCloud(vertices=total_verts, labels=total_labels)
    total_pc.save2pkl(save_path + "total.pkl")
