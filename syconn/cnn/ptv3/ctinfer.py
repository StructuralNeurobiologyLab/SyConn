"""Lean cell type inference loader for SyConn SSVs (bypasses SuperSegmentationObject; reproduces
`_load_ssv_hc` + the training-time context sampling of `pts_loader_scalar` with graph-distance contexts).

Per SSV: skeleton.pkl + mesh_dc.pkl -> voxel-downsampled point cloud with object-type features (sv 0, mi 1, vc 2,
syn_sym 3, syn_asym 4) -> `redundancy` source nodes (2.5 um voxel-downsampled skeleton, as in the existing cell type inference) ->
graph-distance contexts (dijkstra, cutoff ctx_size) -> unique contexts (identical node sets are computed once).
"""
import os, pickle
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
import open3d as o3d

WD = os.environ.get('syconn_wd', "/cajal/nvmescratch/projects/data/songbird/j0251/j0251_72_seg_20210127_agglo2_syn_20220811_celltypes_20230822")
SCALING = np.array([10., 10., 25.], dtype=np.float32)
FEATS = (('sv', 0, 70), ('mi', 1, 100), ('syn_ssv_sym', 3, 70), ('syn_ssv_asym', 4, 70), ('vc', 2, 100))  # (key, label, voxel nm)


def ssv_dir(sid: int) -> str:
    return f"{WD}/ssv_0/so_storage/{sid % 100}/{sid % 10000}/{sid}/"


def read_raw(sid: int):
    d = ssv_dir(sid)
    with open(d + 'skeleton.pkl', 'rb') as f:
        sk = pickle.load(f)
    with open(d + 'mesh_dc.pkl', 'rb') as f:
        mesh = pickle.load(f)
    return sk, mesh


def build_cloud(sk, mesh):
    """-> nodes (N,3) nm float32, edges (E,2), verts (V,3) float32, labels (V,) int8 (label order: sv, mi, sym, asym, vc)."""
    nodes = np.asarray(sk['nodes'], dtype=np.float32) * SCALING
    edges = np.asarray(sk['edges'], dtype=np.int64).reshape(-1, 2)
    vs, ls = [], []
    for key, lab, vox in FEATS:
        v = np.asarray(mesh[key][1], dtype=np.float64).reshape(-1, 3)
        if len(v) == 0:
            continue
        pcd = o3d.geometry.PointCloud(); pcd.points = o3d.utility.Vector3dVector(v)
        v = np.asarray(pcd.voxel_down_sample(voxel_size=vox).points, dtype=np.float32)
        vs.append(v); ls.append(np.full(len(v), lab, dtype=np.int8))
    if vs:
        verts = np.concatenate(vs); labels = np.concatenate(ls)
    else:
        verts = np.zeros((0, 3), np.float32); labels = np.zeros(0, np.int8)
    return nodes, edges, verts, labels


def source_nodes(nodes, redundancy, rng):
    """As in prediction_pts: voxel_down_sample_and_trace(2500) on skeleton nodes, one representative (max index) per voxel, then
    `redundancy` random picks (without replacement if enough)."""
    key = np.floor((nodes - nodes.min(0)) / 2500.0).astype(np.int64)
    key = (key[:, 0] * 1_000_003 + key[:, 1]) * 1_000_003 + key[:, 2]
    order = np.lexsort((np.arange(len(nodes)), key))
    ks = key[order]
    last = np.r_[ks[1:] != ks[:-1], True]
    reps = order[last]
    return rng.choice(reps, redundancy, replace=len(reps) < redundancy)


def contexts(nodes, edges, verts, labels, redundancy=20, ctx_size=20000.0, npoints=50000, scale=2000.0, rng=None):
    """-> list of unique contexts [(coords (M,3) float32 normalised+centred, labels (M,) int8)], inverse (redundancy,)
    mapping each of the `redundancy` draws to its unique context, sources (redundancy,)."""
    rng = np.random.default_rng() if rng is None else rng
    N = len(nodes)
    if N == 0 or len(verts) == 0:
        return [], np.zeros(redundancy, np.int64), np.zeros(redundancy, np.int64)
    src = source_nodes(nodes, redundancy, rng)
    if len(edges):
        w = np.linalg.norm(nodes[edges[:, 0]] - nodes[edges[:, 1]], axis=1).astype(np.float64)
        g = csr_matrix((w, (edges[:, 0], edges[:, 1])), shape=(N, N))
        usrc, inv_src = np.unique(src, return_inverse=True)
        dist = dijkstra(g, directed=False, indices=usrc, limit=ctx_size)
        masks = (dist <= ctx_size)[inv_src]  # (R, N) bool
    else:
        masks = np.zeros((redundancy, N), bool); masks[np.arange(redundancy), src] = True
    nn = cKDTree(nodes).query(verts, k=1)[1]  # nearest skeleton node per vertex
    # unique node sets
    _, uix, inv = np.unique(np.packbits(masks, axis=1), axis=0, return_index=True, return_inverse=True)
    inv = inv.reshape(-1)
    out = []
    for u in uix:
        vix = np.flatnonzero(masks[u][nn])
        if len(vix) > npoints:
            vix = rng.choice(vix, npoints, replace=False)
        c = verts[vix]
        if len(c):
            c = (c / scale); c = c - c.mean(0)  # Normalization(2000) then Center (order irrelevant)
        out.append((c.astype(np.float32), labels[vix]))
    return out, inv, src
