# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max-Planck-Institute of Neurobiology, Munich, Germany
# Authors: Philipp Schubert, Joergen Kornfeld
import glob
import os
from collections import defaultdict
from typing import TYPE_CHECKING, Dict, Optional, Tuple, List, Union, Iterable, Any

import numpy as np

from . import log_reps
from . import rep_helper as rh
from .rep_helper import surface_samples
from .. import global_params
from ..backend.storage import AttributeDict, CompressedStorage, MeshStorage, \
    VoxelStorage, SkeletonStorage, VoxelStorageDyn
from ..handler.basics import chunkify
from ..handler.multiviews import generate_rendering_locs
from ..mp.mp_utils import start_multiprocess_imap
from ..proc.graphs import create_graph_from_coords

if TYPE_CHECKING:
    from ..reps.segmentation import SegmentationObject, SegmentationDataset
MeshType = Union[Tuple[np.ndarray, np.ndarray, np.ndarray], List[np.ndarray],
                 Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]


def glia_pred_so(so: 'SegmentationObject', thresh: float,
                 pred_key_appendix: str) -> int:
    """
    Perform the glia classification of a cell supervoxel (0: neuron, 1: glia).

    Args:
        so: The cell supervoxel object.
        thresh: Threshold used for the classification.
        pred_key_appendix: Additional prediction key.

    Returns:

    """
    assert so.type == "sv"
    pred_key = "glia_probas" + pred_key_appendix
    if pred_key not in so.attr_dict:
        so.load_attr_dict()
    try:
        preds = np.array(so.attr_dict[pred_key][:, 1] > thresh, dtype=np.int32)
        pred = np.mean(so.attr_dict[pred_key][:, 1]) > thresh
    except KeyError:
        raise KeyError('Could not find glia proba key `{}` in so,attr_dict (keys: {})'.format(
            pred_key, so.attr_dict.keys()))
    if pred == 0:
        return 0
    glia_votes = np.sum(preds)
    if glia_votes > int(len(preds) * 0.7):
        return 1
    return 0


def glia_proba_so(so: 'SegmentationObject', pred_key_appendix: str) -> float:
    """
    Get mean glia probability of a cell supervoxel (0: neuron, 1: glia).

    Args:
        so: The cell supervoxel object.
        pred_key_appendix: Additional prediction key.

    Returns:

    """
    assert so.type == "sv"
    pred_key = "glia_probas" + pred_key_appendix
    if pred_key not in so.attr_dict:
        so.load_attr_dict()
    return np.mean(so.attr_dict[pred_key][:, 1])


def acquire_obj_ids(sd: 'SegmentationDataset'):
    """
    Acquires all obj ids present in the dataset. Loads id array if available.
    Assembles id list by iterating over all voxel / attribute dicts,
    otherwise (very slow).
    """
    sd._ids = sd.load_numpy_data('id')
    if sd._ids is None:
        paths = glob.glob(sd.so_storage_path + "/*/*/*/") + \
                glob.glob(sd.so_storage_path + "/*/*/") + \
                glob.glob(sd.so_storage_path + "/*/")
        sd._ids = []
        for path in paths:
            if os.path.exists(path + "attr_dict.pkl"):
                this_ids = list(AttributeDict(path + "attr_dict.pkl", read_only=True).keys())
            else:
                this_ids = []

            sd._ids += this_ids

        sd._ids = np.array(sd._ids)
        np.save(sd.path_ids, sd._ids)


def save_voxels(so: 'SegmentationObject', bin_arr: np.ndarray,
                offset: np.ndarray, overwrite: bool = False):
    """
    Helper function to save SegmentationObject voxels.

    Args:
        so: SegmentationObject
        bin_arr: np.array
            Binary mask array, 0: background, 1: supervoxel locations.
        offset: np.array
        overwrite: bool

    Returns:

    """
    assert bin_arr.dtype == bool

    voxel_dc = VoxelStorage(so.voxel_path, read_only=False,
                            disable_locking=True)

    if so.id in voxel_dc and not overwrite:
        voxel_dc.append(so.id, bin_arr, offset)
    else:
        voxel_dc[so.id] = [bin_arr], [offset]

    voxel_dc.push(so.voxel_path)


def load_voxels_depr(so: 'SegmentationObject',
                     voxel_dc: Optional[VoxelStorage] = None) -> np.ndarray:
    """
    Helper function to load voxels of a SegmentationObject as 3D array.
    Also calculates size and bounding box and assigns it to `so._size` and `so._bounding_box` respectively.

    Args:
        so: SegmentationObject
        voxel_dc: VoxelStorage

    Returns: np.array
        3D binary mask array, 0: background, 1: supervoxel locations.

    """
    if voxel_dc is None:
        voxel_dc = VoxelStorage(so.voxel_path, read_only=True,
                                disable_locking=True)

    so._size = 0
    if so.id not in voxel_dc:
        msg = f"Voxels of {so} do not exist!"
        log_reps.error(msg)
        raise KeyError(msg)

    bin_arrs, block_offsets = voxel_dc[so.id]
    block_extents = []
    for i_bin_arr in range(len(bin_arrs)):
        block_extents.append(np.array(bin_arrs[i_bin_arr].shape) +
                             block_offsets[i_bin_arr])

    block_offsets = np.array(block_offsets, dtype=np.int32)
    block_extents = np.array(block_extents, dtype=np.int32)

    so._bounding_box = np.array([block_offsets.min(axis=0),
                                 block_extents.max(axis=0)])
    voxels = np.zeros(so.bounding_box[1] - so.bounding_box[0],
                      dtype=np.bool)

    for i_bin_arr in range(len(bin_arrs)):
        box = [block_offsets[i_bin_arr] - so.bounding_box[0],
               block_extents[i_bin_arr] - so.bounding_box[0]]

        so._size += np.sum(bin_arrs[i_bin_arr])

        voxels[box[0][0]: box[1][0],
        box[0][1]: box[1][1],
        box[0][2]: box[1][2]][bin_arrs[i_bin_arr]] = True

    return voxels


def load_voxels_downsampled(so: 'SegmentationObject',
                            ds: Tuple[int, int, int] = (2, 2, 1)) -> Union[np.ndarray, List]:
    if isinstance(so.voxels, int):
        return []

    return so.voxels[::ds[0], ::ds[1], ::ds[2]]


def load_voxel_list(so: 'SegmentationObject') -> np.ndarray:
    """
    Helper function to load voxels of a SegmentationObject.

    Args:
        so: SegmentationObject.

    Returns: np.array
        2D array of coordinates to all voxels in SegmentationObject.

    """
    if so._voxels is not None:
        voxel_list = np.transpose(np.nonzero(so.voxels)) + so.bounding_box[0]
    elif so.type in ['syn_ssv', 'syn', 'cs', 'cs_ssv']:
        voxel_dc = VoxelStorageDyn(so.voxel_path, read_only=True, disable_locking=True, voxel_mode=False)
        voxel_list = voxel_dc.get_voxel_cache(so.id)
    else:
        voxel_dc = VoxelStorageDyn(so.voxel_path, read_only=True, disable_locking=True)
        bin_arrs, block_offsets = voxel_dc[so.id]

        voxel_list = []
        for i_bin_arr in range(len(bin_arrs)):
            block_voxels = np.transpose(np.nonzero(bin_arrs[i_bin_arr]))
            block_voxels += block_offsets[i_bin_arr]
            voxel_list.append(block_voxels)
        voxel_list = np.concatenate(voxel_list)
    return voxel_list


def load_voxel_list_downsampled(so, downsampling=(2, 2, 1)):
    """
    TODO: refactor, probably more efficient implementation possible.

    Args:
        so: SegmentationObject
        downsampling: Tuple[int]

    Returns:

    """
    downsampling = np.array(downsampling)
    dvoxels = so.load_voxels_downsampled(downsampling)
    voxel_list = np.array(np.transpose(np.nonzero(dvoxels)), dtype=np.int32)
    voxel_list = voxel_list * downsampling + np.array(so.bounding_box[0])
    return voxel_list


def load_voxel_list_downsampled_adapt(so, downsampling=(2, 2, 1)):
    downsampling = np.array(downsampling, dtype=np.int32)
    dvoxels = so.load_voxels_downsampled(downsampling)

    if len(dvoxels) == 0:
        return []

    while True:
        if True in dvoxels:
            break

        downsampling = downsampling // 2
        downsampling[downsampling < 1] = 1
        dvoxels = so.load_voxels_downsampled(downsampling)

    voxel_list = np.array(np.transpose(np.nonzero(dvoxels)), dtype=np.int32)
    voxel_list = voxel_list * downsampling + np.array(so.bounding_box[0])

    return voxel_list


def load_mesh(so: 'SegmentationObject', recompute: bool = False) -> MeshType:
    """
    Load mesh of SegmentationObject.
    TODO: Currently ignores potential color/label array.

    Args:
        so: SegmentationObject
        recompute: bool

    Returns:
        indices, vertices, normals; all flattened

    """
    if not recompute and so.mesh_exists:
        try:
            mesh = MeshStorage(so.mesh_path,
                               disable_locking=True)[so.id]
            if len(mesh) == 2:
                indices, vertices = mesh
                normals = np.zeros((0,), dtype=np.float32)
            elif len(mesh) == 3:
                indices, vertices, normals = mesh
                col = np.zeros(0, dtype=np.uint8)
            elif len(mesh) == 4:
                indices, vertices, normals, col = mesh
        except Exception as e:
            msg = "\n%s\nException occured when loading mesh.pkl of SO (%s)" \
                  "with id %d.".format(e, so.type, so.id)
            log_reps.error(msg)
            return [np.zeros((0,)).astype(np.int32), np.zeros((0,)), np.zeros((0,))]
    else:
        if so.type == "sv" and not global_params.config.allow_mesh_gen_cells:
            log_reps.error("Mesh of SV %d not found.\n" % so.id)
            return [np.zeros((0,)).astype(np.int), np.zeros((0,)), np.zeros((0,))]
        indices, vertices, normals = so.mesh_from_scratch()
        col = np.zeros(0, dtype=np.uint8)
        try:
            so._save_mesh(indices, vertices, normals)
        except Exception as e:
            log_reps.error("Mesh of %s %d could not be saved:\n%s\n".format(
                so.type, so.id, e))
    vertices = np.array(vertices, dtype=np.int32)
    indices = np.array(indices, dtype=np.int64)
    normals = np.array(normals, dtype=np.float32)
    col = np.array(col, dtype=np.uint8)
    return [indices, vertices, normals]


def load_skeleton(so: 'SegmentationObject', recompute: bool = False) -> dict:
    """

    Args:
        so: SegmentationObject
        recompute: Compute skeleton, will not store it in ``SkeletonStorage``.

    Returns:
        Dictionary with "nodes", "diameters" and "edges".

    """
    empty_skel = dict(nodes=np.zeros((0, 3)).astype(np.int54), edges=np.zeros((0, 2)),
                      diameters=np.zeros((0,)).astype(np.int32))
    if not recompute and so.skeleton_exists:
        try:
            skeleton_dc = SkeletonStorage(so.skeleton_path, disable_locking=not so.enable_locking)
            skel = skeleton_dc[so.id]
            if np.ndim(skel['nodes']) == 1:
                skel['nodes'] = skel['nodes'].reshape((-1, 3))
            if np.ndim(skel['edges']) == 1:
                skel['edges'] = skel['edges'].reshape((-1, 2))
        except Exception as e:
            log_reps.error("\n{}\nException occured when loading skeletons.pkl"
                           " of SO ({}) with id {}.".format(e, so.type, so.id))
            return empty_skel
    elif recompute:
        skel = generate_skeleton_sv(so)
    else:
        msg = f"Skeleton of {so} (size: {so.size}) not found.\n"
        if so.type == "sv":
            if so.size == 1:  # small SVs don't have a skeleton
                log_reps.debug(msg)
            else:
                log_reps.error(msg)
                raise ValueError(msg)
        return empty_skel
    return skel


def save_skeleton(so: 'SegmentationObject', overwrite: bool = False):
    """

    Args:
        so:
        overwrite:

    Returns:

    """
    skeleton_dc = SkeletonStorage(so.skeleton_path, read_only=False, disable_locking=not so.enable_locking)
    if not overwrite and so.id in skeleton_dc:
        raise ValueError(f"Skeleton of {so} already exists.")
    skeleton_dc[so.id] = so.skeleton
    skeleton_dc.push()


def sv_view_exists(args):
    ps, woglia = args
    missing_ids = []
    for p in ps:
        ad = AttributeDict(p + "/attr_dict.pkl", disable_locking=True)
        obj_ixs = ad.keys()
        view_dc_p = p + "/views_woglia.pkl" if woglia else p + "/views.pkl"
        view_dc = CompressedStorage(view_dc_p, disable_locking=True)
        missing_ids = np.setdiff1d(list(obj_ixs), list(view_dc.keys()))
    return missing_ids


def find_missing_sv_views(sd, woglia, n_cores=20):
    folders = sd.so_dir_paths
    np.random.shuffle(folders)
    multi_params = chunkify(folders, 1000)
    params = [(ps, woglia) for ps in multi_params]
    res = start_multiprocess_imap(sv_view_exists, params, nb_cpus=n_cores,
                                  debug=False)
    return np.concatenate(res)


def sv_skeleton_missing(sv):
    if sv.skeleton is None:
        sv.load_skeleton()
    return (sv.skeleton is None) or (len(sv.skeleton['nodes']) == 0)


def find_missing_sv_skeletons(svs, n_cores=20):
    res = start_multiprocess_imap(sv_skeleton_missing, svs, nb_cpus=n_cores,
                                  debug=False)
    return [svs[kk].id for kk in range(len(svs)) if res[kk]]


def sv_attr_exists(args):
    ps, attr_key = args
    missing_ids = []
    for p in ps:
        ad = AttributeDict(p + "/attr_dict.pkl", disable_locking=True)
        for k, v in ad.items():
            if attr_key not in v:
                missing_ids.append(k)
    return missing_ids


def find_missing_sv_attributes(sd: 'SegmentationDataset', attr_key: str, n_cores: int = 20):
    """

    Args:
        sd:
        attr_key: str
        n_cores: int

    Returns:

    """
    multi_params = chunkify(sd.so_dir_paths, 100)
    params = [(ps, attr_key) for ps in multi_params]
    res = start_multiprocess_imap(sv_attr_exists, params, nb_cpus=n_cores, debug=False)
    return np.concatenate(res)


def load_so_meshes_bulk(sos: Union[List['SegmentationObject'], Iterable['SegmentationObject']],
                        use_new_subfold: bool = True, cache_decomp=True) -> MeshStorage:
    """
    Bulk loader for SegmentationObject (SO) meshes. Minimizes IO by loading IDs from the same storage at the same time.
    This will not assign the ``_mesh`` attribute!

    Args:
        sos: SegmentationObjects
        use_new_subfold: Use new sub-folder structure
        cache_decomp: Cache decompressed meshes.

    Returns:
        Dictionary, key: ID, value: mesh
    """
    md_out = MeshStorage(None)  # in-memory dict with compression
    if len(sos) == 0:
        return md_out
    base_path = sos[0].so_storage_path
    nf = sos[0].n_folders_fs
    subf_from_ix = rh.subfold_from_ix_new if use_new_subfold else \
        rh.subfold_from_ix_OLD
    sub2ids = defaultdict(list)
    for so in sos:
        if so._mesh is None:
            subf = subf_from_ix(so.id, nf)
            sub2ids[subf].append(so.id)
        else:
            md_out[so.id] = so._mesh
    for subfold, ids in sub2ids.items():
        mesh_path = f'{base_path}/{subfold}/mesh.pkl'
        md = MeshStorage(mesh_path, disable_locking=True,
                         cache_decomp=cache_decomp)
        for so_id in ids:
            md_out._dc_intern[so_id] = md._dc_intern[so_id]
    assert len(md_out) == len(sos)
    return md_out


def load_so_attr_bulk(sos: List['SegmentationObject'],
                      attr_keys: Union[str, Iterable[str]],
                      use_new_subfold: bool = True,
                      allow_missing: bool = False) -> Union[Dict[str, Dict[int, Any]], Dict[int, Any]]:
    """
    Bulk loader for SegmentationObject (SO) meshes. Minimizes IO by loading IDs from the same storage at the same time.
    Returns a single dict if only one attr_key is provided or a dict of dicts if many.
    This method will also check if the requested attribute(s) already exist in the object's ``attr_dict``. This means
    using ``cache_properties`` when initializing ``SegmentationDataset`` might be beneficial to avoid exhaustive file
    reads in case `sos` is large.

    Args:
        sos: SegmentationObjects
        attr_keys: Attribute key(s).
        use_new_subfold: Use new sub-folder structure
        allow_missing: If True, sets attribute value to None if missing. If False and missing, raise KeyError.

    Returns:
        (Dict. with key: attr_key of) dict. with key: ID, value: attribute value
    """
    if type(attr_keys) is str:
        attr_keys = [attr_keys]
    out = {attr_k: dict() for attr_k in attr_keys}
    if len(sos) == 0:
        if len(attr_keys) == 1:
            out = out[attr_keys[0]]
        return out
    base_path = sos[0].so_storage_path
    nf = sos[0].n_folders_fs
    subf_from_ix = rh.subfold_from_ix_new if use_new_subfold else rh.subfold_from_ix_OLD
    sub2ids = defaultdict(list)
    for so in sos:
        keys_missing = len(attr_keys)
        # use cached/loaded attributes
        for k in attr_keys:
            if k in so.attr_dict:
                out[k][so.id] = so.attr_dict[k]
                keys_missing -= 1
        if keys_missing == 0:
            continue
        subf = subf_from_ix(so.id, nf)
        sub2ids[subf].append(so.id)
    for subfold, ids in sub2ids.items():
        attr_p = f'{base_path}/{subfold}/attr_dict.pkl'
        ad = AttributeDict(attr_p, disable_locking=True)
        for so_id in ids:
            so_dict = ad[so_id]
            for attr_key in attr_keys:
                try:
                    out[attr_key][so_id] = so_dict[attr_key]
                except KeyError as e:
                    if allow_missing:
                        out[attr_key][so_id] = None
                    else:
                        raise KeyError(e)
    if len(attr_keys) == 1:
        out = out[attr_keys[0]]
    return out


def prepare_so_attr_cache(sd: 'SegmentationDataset', so_ids: np.ndarray, attr_keys: List[str]) -> Dict[str, dict]:
    """

    Args:
        sd: SegmentationDataset.
        so_ids: SegmentationObject IDs for which to collect the attributes.
        attr_keys: Attribute keys to collect. Corresponding numyp arrays must exist.

    Returns:
        Dictionary with `attr_keys` as keys and an attribute dictionary as values for the IDs `so_ids`, e.g.
        ``attr_cache[attr_keys[0]][so_ids[0]]`` will return the attribute value of type ``attr_keys[0]`` for the first
        SegmentatonObect in `so_ids`.
    """
    attr_cache = {k: dict() for k in attr_keys}
    # TODO: Use BinarySearchStore
    soid2ix = {so_id: sd.soid2ix[so_id] for so_id in so_ids}
    sd._soid2ix = None  # free memory
    for attr in attr_keys:
        np_cache = sd.load_numpy_data(attr, allow_nonexisting=False)
        for so_id in so_ids:
            attr_cache[attr][so_id] = np_cache[soid2ix[so_id]]
        del np_cache
    return attr_cache


def load_so_voxels_bulk(sos: List['SegmentationObject'],
                        use_new_subfold: bool = True, cache_decomp=True):
    """

    Args:
        sos:
        use_new_subfold:
        cache_decomp:

    Returns:

    """
    raise NotImplementedError('WIP')
    vd_out = VoxelStorage(None, cache_decomp=cache_decomp)  # in-memory dict with compression
    if len(sos) == 0:
        return vd_out
    base_path = sos[0].so_storage_path
    nf = sos[0].n_folders_fs
    subf_from_ix = rh.subfold_from_ix_new if use_new_subfold else \
        rh.subfold_from_ix_OLD
    sub2ids = defaultdict(list)
    for so in sos:
        subf = subf_from_ix(so.id, nf)
        sub2ids[subf].append(so.id)
    cnt = 0
    for subfold, ids in sub2ids.items():
        voxel_path = f'{base_path}/{subfold}/voxel.pkl'
        vd = VoxelStorage(voxel_path, disable_locking=True)
        for so_id in ids:
            cnt += 1
            vd_out._dc_intern[so_id] = vd._dc_intern[so_id]
    assert cnt == len(sos)
    return vd_out


def _helper_func(args):
    ps, use_vxsize = args
    out = []
    for p in ps:
        if not use_vxsize:
            w = len(AttributeDict(p + '/attr_dict.pkl', disable_locking=True))
        else:
            w = np.sum([v['size'] for v in AttributeDict(p + '/attr_dict.pkl', disable_locking=True).values()])
        out.append(w)
    return out


def get_sd_load_distribution(sd: 'SegmentationDataset', use_vxsize: bool = True) -> np.ndarray:
    """
    Get the load distribution (number of objects per storage) of the SegmentationDataset's AttributeDicts.

    Args:
        sd: SegmentationDataset
        use_vxsize:

    Returns:
        Load array.
    """
    n_objects = start_multiprocess_imap(_helper_func, [(ch, use_vxsize) for ch in chunkify(sd.so_dir_paths, 1000)],
                                        nb_cpus=None)
    return np.concatenate(n_objects).astype(np.int64)


def generate_skeleton_sv(so: 'SegmentationObject') -> Dict[str, np.ndarray]:
    """
    Poor man's solution to generate a SV "skeleton". Used for glia predictions.

    Args:
        so:

    Returns:
        Dictionary with keys "nodes", "edges" and "diameters" (all 1).
    """
    verts = so.mesh[1].reshape(-1, 3)
    # choose random subset of surface vertices
    np.random.seed(0)
    ixs = np.arange(len(verts))
    np.random.shuffle(ixs)
    ixs = ixs[:int(0.5 * len(ixs))]
    if global_params.config.use_new_renderings_locs:
        locs = generate_rendering_locs(verts[ixs], 1000)
    else:
        locs = surface_samples(verts[ixs], bin_sizes=(1000, 1000, 1000),
                               max_nb_samples=10000, r=500)
    g = create_graph_from_coords(locs, mst=True)
    if g.number_of_edges() == 1:
        edge_list = np.array(list(g.edges()))
    else:
        edge_list = np.array(g.edges())
    del g
    if edge_list.ndim != 2:
        raise ValueError("Edge list ist not a 2D array: {}\n{}".format(
            edge_list.shape, edge_list))
    skeleton = dict()
    skeleton["nodes"] = (locs / np.array(so.scaling)).astype(np.int32)
    skeleton["edges"] = edge_list
    skeleton["diameters"] = np.ones(len(locs))
    return skeleton
