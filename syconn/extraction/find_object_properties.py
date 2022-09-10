import numba
import scipy.ndimage
from numba import typed
from numba import types
import numpy as np
from typing import Tuple, List

from syconn import global_params
from syconn.extraction.block_processing_C import process_block_nonzero
# needed because all other module import these two methods from here
from syconn.extraction.find_object_properties_C import find_object_properties, map_subcell_extract_props


int64_arr2d = types.int64[:, :]
int64_arr1d = types.int64[:]
uint64_tuple = types.UniTuple(numba.uint64, 2)
uint64_arr1d_dict = types.DictType(types.uint64, int64_arr1d)
uint64_arr2d_dict = types.DictType(types.uint64, int64_arr2d)
uint64_int64_dict = types.DictType(types.uint64, types.int64)


@numba.jit(nopython=True)
def extract_cs_syntype_64bit(cs_seg: np.ndarray, syn_mask: np.ndarray, asym_mask: np.ndarray, sym_mask: np.ndarray)\
        -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray],
                 np.ndarray, np.ndarray]:
    """
    Extract synaptic properties for every contact site ID tuple inside `cs_seg`.

    Count of synaptic foreground voxel and if voxel is foreground also sums the number of symmetric
    and asymmetric voxels. The type ratio can be computed
    by using the total syn foreground voxels assigned to the CS object.

    Notes:
        * `rep_coord` currently first voxel (could be more representative).
        * Assumes that the partner IDs in `cs_seg` are sorted, e.g. (1, 2) and (2, 1) must not exist.
        *  cs_seg, syn_mask, sym_mask and asym_mask  must all have the same spatial shape.

    Args:
        cs_seg: Contact site segmentation (XYZC, with C=2).
        syn_mask: Synapse prediction (binary foreground mask, synapse=1, background=0).
        asym_mask: Asymmetric type prediction (binary foreground mask, asym=1, background=0).
        sym_mask: Symmetric type prediction (binary foreground mask, sym=1, background=0).

    Returns:
        Representative coordinate, bounding box and size for contact sites, representative coordinate, bounding box
        and size for synapses (for voxels with ``syn_mask=1``), counts for asymmetric and symmetric voxels. All objects
        are nested dictionaries with contact site/synapse partner IDs as keys.
    """

    rep_coords = typed.Dict.empty(
        key_type=types.uint64,
        value_type=uint64_arr1d_dict,
    )
    bounding_box = typed.Dict.empty(
        key_type=types.uint64,
        value_type=uint64_arr2d_dict,
    )
    sizes = typed.Dict.empty(
        key_type=types.uint64,
        value_type=uint64_int64_dict,
    )

    rep_coords_syn = typed.Dict.empty(
        key_type=types.uint64,
        value_type=uint64_arr1d_dict,
    )
    bounding_box_syn = typed.Dict.empty(
        key_type=types.uint64,
        value_type=uint64_arr2d_dict,
    )
    sizes_syn = typed.Dict.empty(
        key_type=types.uint64,
        value_type=uint64_int64_dict,
    )

    cs_asym = typed.Dict.empty(
        key_type=types.uint64,
        value_type=uint64_int64_dict,
    )
    cs_sym = typed.Dict.empty(
        key_type=types.uint64,
        value_type=uint64_int64_dict,
    )

    sh = cs_seg.shape
    for x in range(sh[0]):
        for y in range(sh[1]):
            for z in range(sh[2]):
                key = cs_seg[x, y, z]
                # update CS properties
                # if one element is zero, both have to be - skip as they are background
                if key[0] == 0:
                    continue
                dc_local_bb = bounding_box.get(key[0])
                if dc_local_bb is None:
                    rep_coords_local = typed.Dict.empty(
                        key_type=types.uint64,
                        value_type=int64_arr1d,
                    )
                    bounding_box_local = typed.Dict.empty(
                        key_type=types.uint64,
                        value_type=int64_arr2d,
                    )
                    sizes_local = typed.Dict.empty(
                        key_type=types.uint64,
                        value_type=types.int64,
                    )
                    bounding_box_local[key[1]] = [(x, y, z), (x + 1, y + 1, z + 1)]
                    sizes_local[key[1]] = 1
                    rep_coords_local[key[1]] = [x, y, z]
                    bounding_box[key[0]] = bounding_box_local
                    sizes[key[0]] = sizes_local
                    rep_coords[key[0]] = rep_coords_local
                elif dc_local_bb.get(key[1]) is None:
                    bounding_box[key[0]][key[1]] = [(x, y, z), (x+1, y+1, z+1)]
                    sizes[key[0]][key[1]] = 1
                    rep_coords[key[0]][key[1]] = [x, y, z]
                else:
                    local_bb = dc_local_bb.get(key[1])
                    local_bb[0][0] = min(local_bb[0][0], x)
                    local_bb[0][1] = min(local_bb[0][1], y)
                    local_bb[0][2] = min(local_bb[0][2], z)
                    local_bb[1][0] = max(local_bb[1][0], x + 1)
                    local_bb[1][1] = max(local_bb[1][1], y + 1)
                    local_bb[1][2] = max(local_bb[1][2], z + 1)
                    sizes[key[0]][key[1]] += 1
                # extract synapse properties and syntype info
                syn_vx = syn_mask[x, y, z]
                # IMPORTANT! ONLY COUNT SYN TYPES IF FOREGROUND IS TRUE
                if syn_vx == 0:
                    continue
                dc_local_bb = bounding_box_syn.get(key[0])
                if dc_local_bb is None:
                    rep_coords_local = typed.Dict.empty(
                        key_type=types.uint64,
                        value_type=int64_arr1d,
                    )
                    bounding_box_local = typed.Dict.empty(
                        key_type=types.uint64,
                        value_type=int64_arr2d,
                    )
                    sizes_local = typed.Dict.empty(
                        key_type=types.uint64,
                        value_type=types.int64,
                    )
                    bounding_box_local[key[1]] = [(x, y, z), (x + 1, y + 1, z + 1)]
                    sizes_local[key[1]] = 1
                    rep_coords_local[key[1]] = [x, y, z]
                    bounding_box_syn[key[0]] = bounding_box_local
                    sizes_syn[key[0]] = sizes_local
                    rep_coords_syn[key[0]] = rep_coords_local
                elif dc_local_bb.get(key[1]) is None:
                    bounding_box_syn[key[0]][key[1]] = [(x, y, z), (x+1, y+1, z+1)]
                    sizes_syn[key[0]][key[1]] = 1
                    rep_coords_syn[key[0]][key[1]] = [x, y, z]
                else:
                    local_bb = dc_local_bb.get(key[1])
                    local_bb[0][0] = min(local_bb[0][0], x)
                    local_bb[0][1] = min(local_bb[0][1], y)
                    local_bb[0][2] = min(local_bb[0][2], z)
                    local_bb[1][0] = max(local_bb[1][0], x + 1)
                    local_bb[1][1] = max(local_bb[1][1], y + 1)
                    local_bb[1][2] = max(local_bb[1][2], z + 1)
                    sizes_syn[key[0]][key[1]] += 1

                # store sym. and asym. voxels counts
                if asym_mask[x, y, z] == 1:
                    dc_cs_asym_local = cs_asym.get(key[0])
                    if dc_cs_asym_local is None:
                        cs_asym_local = typed.Dict.empty(
                            key_type=types.uint64,
                            value_type=types.int64,
                        )
                        cs_asym_local[key[1]] = 1
                        cs_asym[key[0]] = cs_asym_local
                    elif dc_cs_asym_local.get(key[1]) is None:
                        cs_asym[key[0]][key[1]] = 1
                    else:
                        cs_asym[key[0]][key[1]] += 1
                if sym_mask[x, y, z] == 1:
                    dc_cs_sym_local = cs_sym.get(key[0])
                    if dc_cs_sym_local is None:
                        cs_sym_local = typed.Dict.empty(
                            key_type=types.uint64,
                            value_type=types.int64,
                        )
                        cs_sym_local[key[1]] = 1
                        cs_sym[key[0]] = cs_sym_local
                    elif dc_cs_sym_local.get(key[1]) is None:
                        cs_sym[key[0]][key[1]] = 1
                    else:
                        cs_sym[key[0]][key[1]] += 1
    return (rep_coords, bounding_box, sizes), \
           (rep_coords_syn, bounding_box_syn, sizes_syn), cs_asym, cs_sym


@numba.jit(nopython=True)
def find_object_properties_cs_64bit(cs_seg: np.ndarray):
    """
    Extract contact properties for every contact site ID tuple inside `cs_seg`.

    Notes:
        * `rep_coord` currently first voxel (could be more representative).
        * Assumes that the partner IDs in `cs_seg` are sorted, e.g. (1, 2) and (2, 1) must not exist.
        *  cs_seg, syn_mask, sym_mask and asym_mask  must all have the same spatial shape.

    Args:
        cs_seg: Contact site segmentation (XYZC, with C=2).

    Returns:
        Representative coordinate, bounding box and size for contact sites. All objects
        are nested dictionaries with contact site/synapse partner IDs as keys.
    """
    rep_coords = typed.Dict.empty(
        key_type=types.uint64,
        value_type=uint64_arr1d_dict,
    )
    bounding_box = typed.Dict.empty(
        key_type=types.uint64,
        value_type=uint64_arr2d_dict,
    )
    sizes = typed.Dict.empty(
        key_type=types.uint64,
        value_type=uint64_int64_dict,
    )

    for x in range(cs_seg.shape[0]):
        for y in range(cs_seg.shape[1]):
            for z in range(cs_seg.shape[2]):
                key = cs_seg[x, y, z]
                # if one element is zero, both have to be - skip as they are background
                if key[0] == 0:
                    continue
                dc_local_bb = bounding_box.get(key[0])
                if dc_local_bb is None:
                    rep_coords_local = typed.Dict.empty(
                        key_type=types.uint64,
                        value_type=int64_arr1d,
                    )
                    bounding_box_local = typed.Dict.empty(
                        key_type=types.uint64,
                        value_type=int64_arr2d,
                    )
                    sizes_local = typed.Dict.empty(
                        key_type=types.uint64,
                        value_type=types.int64,
                    )
                    bounding_box_local[key[1]] = np.array([(x, y, z), (x + 1, y + 1, z + 1)], dtype=np.int64)
                    sizes_local[key[1]] = 1
                    rep_coords_local[key[1]] = np.array([x, y, z], dtype=np.int64)
                    bounding_box[key[0]] = bounding_box_local
                    sizes[key[0]] = sizes_local
                    rep_coords[key[0]] = rep_coords_local
                elif dc_local_bb.get(key[1]) is None:
                    print(key)
                    bounding_box[key[0]][key[1]] = np.array([(x, y, z), (x+1, y+1, z+1)], dtype=np.int64)
                    sizes[key[0]][key[1]] = 1
                    rep_coords[key[0]][key[1]] = np.array([x, y, z], dtype=np.int64)  # TODO: could be more representative
                    print(rep_coords)
                else:
                    local_bb = dc_local_bb.get(key[1])
                    local_bb[0][0] = min(local_bb[0][0], x)
                    local_bb[0][1] = min(local_bb[0][1], y)
                    local_bb[0][2] = min(local_bb[0][2], z)
                    local_bb[1][0] = max(local_bb[1][0], x + 1)
                    local_bb[1][1] = max(local_bb[1][1], y + 1)
                    local_bb[1][2] = max(local_bb[1][2], z + 1)
                    sizes[key[0]][key[1]] += 1
    return rep_coords, bounding_box, sizes


def convert_nvox2ratio_syntype(syn_cnts, sym_cnts, asym_cnts):
    """
    Get ratio of sym. and asym. voxels to the synaptic foreground voxels of each contact site
    object.
    Sym. and asym. ratios do not necessarily sum to 1 if types are predicted independently.

    Args:
        syn_cnts:
        sym_cnts:
        asym_cnts:

    Returns:

    """
    # TODO implement in numba

    sym_ratio = {}
    asym_ratio = {}
    for cs_id, cnt in syn_cnts.items():
        if cs_id in sym_cnts:
            sym_ratio[cs_id] = sym_cnts[cs_id] / cnt
        else:
            sym_ratio[cs_id] = 0
        if cs_id in asym_cnts:
            asym_ratio[cs_id] = asym_cnts[cs_id] / cnt
        else:
            asym_ratio[cs_id] = 0
    return asym_ratio, sym_ratio


def merge_type_dicts(type_dicts: List[dict]):
    """
    Merge map dictionaries in-place. Values will be stored in first dictionary

    Args:
        type_dicts:

    Returns:

    """
    tot_map = type_dicts[0]
    for el in type_dicts[1:]:
        # iterate over subcell. ids with dictionaries as values which store
        # the number of overlap voxels to cell SVs
        for cs_id, cnt in el.items():
            if cs_id in tot_map:
                tot_map[cs_id] += cnt
            else:
                tot_map[cs_id] = cnt


def merge_voxel_dicts(voxel_dicts: List[dict], key_to_str=False):
    """
    Merge map dictionaries values will be stored in first dictionary, this method converts numpy arrays into lists.

    Args:
        voxel_dicts:
        key_to_str: If False, keep keys as they are, which is needed when loading data from npz files (default).
            If True, converts keys to strings to enable `np.savez` (requires str).
    """
    tot_map = voxel_dicts[0]
    for el in voxel_dicts[1:]:
        # iterate over subcell. ids with dictionaries as values which store
        # voxel coordinates
        for cs_id, vxs in el.items():
            if key_to_str:
                cs_id = str(cs_id)
            if cs_id in tot_map:
                tot_map[cs_id].extend(vxs)
            else:
                if isinstance(vxs, np.ndarray):
                    vxs = vxs.tolist()
                tot_map[cs_id] = vxs


def detect_cs_64bit(arr: np.ndarray) -> np.ndarray:
    """
    Uses :func:`detect_seg_boundaries` to generate initial contact mask.

    Args:
        arr: 3D segmentation array

    Returns:
        4D contact site segmentation array (XYZC; with C=2).
    """
    # first identify boundary voxels
    bdry = detect_seg_boundaries(arr)

    stencil = np.array(global_params.config['cell_objects']['cs_filtersize'])
    assert np.sum(stencil % 2) == 3
    offset = stencil // 2

    # extract adjacent majority ID on sparse boundary voxels
    offset = np.array([(-offset[0], offset[0]), (-offset[1], offset[1]), (-offset[2], offset[2])])
    cs_seg = detect_contact_partners(arr, bdry, offset)
    return cs_seg


@numba.jit(nopython=True)
def detect_contact_partners(seg_arr: np.ndarray, edge_arr: np.ndarray, offset: np.ndarray) -> np.ndarray:
    """
    Identify whether IDs differ within `offset` and return boundary mask. Resulting array will be ``2*offset`` smaller
    than input `seg_arr` ("valid convolution").

    Args:
        seg_arr: Segmentation volume (XYZ).
        edge_arr: Boundary/edge mask array (XYZ). Inspects location if != 0, skips if 0.
        offset: Offset for all spatial axes. Must have shape (3, 2). E.g. [(-1, 1), (-1, 1), (-1, 1)]
            will check a 3x3x3 cube around every voxel.

    Returns:
        Boundary mask. Axes will be ``2*offset`` smaller.
    """
    nx, ny, nz = seg_arr.shape[:3]
    contact_partners = np.zeros((nx+offset[0, 0]-offset[0, 1],
                                 ny+offset[1, 0]-offset[1, 1],
                                 nz+offset[2, 0]-offset[2, 1], 2
                                 ), dtype=np.uint64)
    for xx in range(-offset[0, 0], nx-offset[0, 1]):
        for yy in range(-offset[1, 0], ny-offset[1, 1]):
            for zz in range(-offset[2, 0], nz-offset[2, 1]):
                center_id = seg_arr[xx, yy, zz]
                if edge_arr[xx, yy, zz] == 0:
                    continue
                d = typed.Dict.empty(
                    key_type=numba.uint64,
                    value_type=numba.uint64,
                )
                # inspect cube around center voxel
                for neigh_x in range(offset[0, 0], offset[0, 1] + 1):
                    for neigh_y in range(offset[1, 0], offset[1, 1] + 1):
                        for neigh_z in range(offset[2, 0], offset[2, 1] + 1):
                            neigh_id = seg_arr[xx + neigh_x, yy + neigh_y, zz + neigh_z]
                            if (neigh_id == 0) or (neigh_id == center_id):
                                continue
                            if neigh_id in d:
                                d[neigh_id] += numba.uint64(1)
                            else:
                                d[neigh_id] = numba.uint64(1)
                if len(d) != 0:
                    # get most common ID
                    most_comm = 0
                    most_comm_cnt = 0
                    for k, v in d.items():
                        if most_comm_cnt < v:
                            most_comm = k
                            most_comm_cnt = v
                    partners = [most_comm, center_id] if center_id > most_comm else [center_id, most_comm]
                    contact_partners[xx+offset[0, 0], yy+offset[1, 0], zz+offset[2, 0]] = partners
    return contact_partners


@numba.jit(nopython=True)
def detect_seg_boundaries(arr: np.ndarray) -> np.ndarray:
    """
    Identify whether IDs differ within 6-connectivity and return boolean mask.
    0 IDs are skipped.

    Args:
        arr: Segmentation volume (XYZ).

    Returns:
        Binary boundary mask (1: segmentation boundary, 0: inside segmentation or background).
    """
    nx, ny, nz = arr.shape[:3]
    boundary = np.zeros((nx, ny, nz), dtype=np.bool_)
    for xx in range(nx):
        for yy in range(ny):
            for zz in range(nz):
                center_id = arr[xx, yy, zz]
                # no need to flag background
                if center_id == 0:
                    continue
                for neigh_x, neigh_y, neigh_z in [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0),
                                                  (0, 0, -1), (0, 0, 1)]:
                    if (xx + neigh_x < 0) or (xx + neigh_x >= nx):
                        continue
                    if (yy + neigh_y < 0) or (yy + neigh_y >= ny):
                        continue
                    if (zz + neigh_z < 0) or (zz + neigh_z >= nz):
                        continue
                    neigh_id = arr[xx + neigh_x, yy + neigh_y, zz + neigh_z]
                    boundary[xx, yy, zz] = (neigh_id != center_id) or boundary[xx, yy, zz]
    return boundary


def detect_cs(arr: np.ndarray) -> np.ndarray:
    """
    Only works if ``arr.dtype`` is uint32. Use detect_cs_64bit for uin64 segmentation.

    Args:
        arr: 3D segmentation array (only np.uint32).

    Returns:
        3D contact site instance segmentation array (np.uint64).
    """
    edges = detect_seg_boundaries(arr).astype(np.uint32, copy=False)
    arr = arr.astype(np.uint32, copy=False)
    cs_seg = process_block_nonzero(
        edges, arr, global_params.config['cell_objects']['cs_filtersize'])
    return cs_seg
