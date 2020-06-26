import os
import numpy as np
import open3d as o3d
from syconn import global_params
from morphx.classes.hybridcloud import HybridCloud
from typing import Union, Tuple, List
from knossos_utils.skeleton import Skeleton
from knossos_utils.skeleton_utils import load_skeleton, write_skeleton, write_anno
from morphx.processing import clouds
from morphx.classes.pointcloud import PointCloud
from morphx.processing.hybrids import extract_subset
from morphx.processing.objects import context_splitting_kdt_many
from syconn.reps.super_segmentation import SuperSegmentationObject


pts_feat_dict = dict(sv=0, mi=1, syn_ssv=2, vc=3)
pts_feat_ds_dict = dict(celltype=dict(sv=70, mi=100, syn_ssv=70, syn_ssv_sym=70, syn_ssv_asym=70, vc=100),
                        glia=dict(sv=50, mi=100, syn_ssv=100, syn_ssv_sym=100, syn_ssv_asym=100, vc=100),
                        compartment=dict(sv=80, mi=100, syn_ssv=100, syn_ssv_sym=100, syn_ssv_asym=100, vc=100))


def sso2hc(sso: SuperSegmentationObject, feats: Union[Tuple, str], feat_labels: Union[Tuple, int], pt_type: str):
    if type(feats) == str:
        feats = [feats]
    if type(feat_labels) == int:
        feat_labels = [feat_labels]
    vert_dc = dict()
    label_dc = dict()
    obj_bounds = {}
    offset = 0
    idcs_dict = {}
    for k in feats:
        pcd = o3d.geometry.PointCloud()
        verts = sso.load_mesh(k)[1].reshape(-1, 3)
        pcd.points = o3d.utility.Vector3dVector(verts)
        pcd, idcs = pcd.voxel_down_sample_and_trace(pts_feat_ds_dict[pt_type][k], pcd.get_min_bound(),
                                                    pcd.get_max_bound())
        idcs = np.max(idcs, axis=1)
        idcs_dict[k] = idcs
        vert_dc[k] = np.asarray(pcd.points)
        obj_bounds[k] = [offset, offset + len(pcd.points)]
        offset += len(pcd.points)
        label_dc[k] = np.ones((len(pcd.points), 1))*pts_feat_dict[k]
    sample_pts = np.concatenate([vert_dc[k] for k in feats])
    sample_labels = np.concatenate([label_dc[k] for k in feats])
    if not sso.load_skeleton():
        raise ValueError(f'Couldnt find skeleton of {sso}')
    nodes, edges = sso.skeleton['nodes'] * sso.scaling, sso.skeleton['edges']
    hc = HybridCloud(nodes, edges, vertices=sample_pts, labels=sample_labels, obj_bounds=obj_bounds)
    # cache verts2node
    _ = hc.verts2node
    return hc, idcs_dict


if __name__ == '__main__':
    global_params.wd = "/wholebrain/songbird/j0126/areaxfs_v6/"
    skel_path = os.path.expanduser('~/thesis/tmp/a_skel.k.zip')
    target_path = os.path.expanduser('~/thesis/tmp/contexts/')
    ctx_size = 10000
    sso = SuperSegmentationObject(22335491)
    feat_dc = dict(pts_feat_dict)
    hc, _ = sso2hc(sso, tuple(feat_dc.keys()), tuple(feat_dc.values()), 'compartment')
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(hc.nodes)
    pcd, idcs = pcd.voxel_down_sample_and_trace(ctx_size, pcd.get_min_bound(), pcd.get_max_bound())
    source_nodes = np.max(idcs, axis=1)
    ctxs = context_splitting_kdt_many(hc, source_nodes, ctx_size)

    transform = clouds.Compose([clouds.Normalization(10000), clouds.Center()])

    for ix, ctx in enumerate(ctxs):
        hc_sub, idcs_sub = extract_subset(hc, ctx)
        hc_sample, idcs_sample = clouds.sample_cloud(hc_sub, 5000)
        transform(hc_sample)
        hc_sample.save2pkl(target_path + f'{ix}.pkl')


    # a_obj = load_skeleton(skel_path)
    # skel = a_obj['skeleton']
    # a_nodes = list(skel.getNodes())
    # for ix, ctx in enumerate(ctxs):
    #     for node_ix in ctx:
    #         a_nodes[node_ix].setComment(f'{ix%3}')
    # s = Skeleton()
    # s.set_scaling(sso.scaling)
    # s.add_annotation(skel)
    # s.to_kzip(target_path)

