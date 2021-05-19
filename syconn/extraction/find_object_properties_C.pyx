# distutils: language = c++
from libcpp.unordered_map cimport unordered_map
cimport cython
from libcpp cimport bool
from libc.stdint cimport int, float
from libc.stdint cimport uint64_t, uint32_t
from libcpp.vector cimport vector
import numpy as np

ctypedef fused n_type:
    uint64_t
    uint32_t

ctypedef vector[int] int_vec
ctypedef vector[int_vec] int_vec_vec
ctypedef vector[n_type[:, :, :]] uintarr_vec
ctypedef vector[unordered_map[uint64_t, int_vec]] umvec_rc
ctypedef vector[unordered_map[uint64_t, int_vec_vec]] umvec_bb
ctypedef vector[unordered_map[uint64_t, int]] umvec_size
ctypedef unordered_map[uint64_t, uint64_t] um_uint2uint
ctypedef vector[unordered_map[uint64_t, um_uint2uint]] umvec_map


def find_object_properties(n_type[:, :, :] chunk):
    cdef unordered_map[uint64_t, int_vec] rep_coords
    cdef unordered_map[uint64_t, int_vec_vec] bounding_box
    cdef unordered_map[uint64_t, int] sizes
    cdef int_vec_vec *local_bb

    for x in range(chunk.shape[0]):
        for y in range(chunk.shape[1]):
            for z in range(chunk.shape[2]):
                key = chunk[x, y, z]
                if key == 0:
                    continue
                if sizes.count(key):
                    local_bb = & (bounding_box[key])
                    local_bb[0][0][0] = min(local_bb[0][0][0], x)
                    local_bb[0][0][1] = min(local_bb[0][0][1], y)
                    local_bb[0][0][2] = min(local_bb[0][0][2], z)
                    local_bb[0][1][0] = max(local_bb[0][1][0], x + 1)
                    local_bb[0][1][1] = max(local_bb[0][1][1], y + 1)
                    local_bb[0][1][2] = max(local_bb[0][1][2], z + 1)
                    sizes[key] += 1
                else:
                    bounding_box[key] = ((x, y, z), (x+1, y+1, z+1))
                    sizes[key] = 1
                    rep_coords[key] = [x, y, z]  # TODO: could be more representative
    return rep_coords, bounding_box, sizes


def um_intvec_fact():
    cdef unordered_map[uint64_t, int_vec] um
    return um


def um_intvecvec_fact():
    cdef unordered_map[uint64_t, int_vec_vec] um
    return um


def um_int_fact():
    cdef unordered_map[uint64_t, int] um
    return um


def um_umint_fact():
    cdef um_uint2uint um
    return um


def map_subcell_C(n_type[:, :, :] ch, n_type[:, :, :, :] subcell_chs):
    """ch and subcell_chs must all have the same shape!
    TODO: check if uint32 and uint64 works with current unordered map definition,
    using n_type instead of uint64 results in an error.
    Returns rep coord, bounding box and size dict for cell segmentation objects,
    list of those property dicts for each subcellular segmentation type and number
    of overlapping voxels between subcellular and cell segmentation
    (Dict[Dict[uint]: subcell ID -> cell ID -> count)"""

    # property dicts for cell SV
    sh = ch.shape

    n_subcell = subcell_chs.shape[0]
    cdef umvec_map subcell_mapping_dicts


    for ii in range(n_subcell):
        assert (subcell_chs[ii].shape[0] == sh[0]) &\
               (subcell_chs[ii].shape[1] == sh[1]) &\
               (subcell_chs[ii].shape[2] == sh[2]), "Segmentation of cells and subcellular structures must have same shape. {} {} {} {} {} {}".format(
               subcell_chs[ii].shape[0], subcell_chs[ii].shape[1], subcell_chs[ii].shape[2], sh[0], sh[1], sh[2])
        subcell_mapping_dicts.push_back(um_umint_fact())

    for x in range(sh[0]):
        for y in range(sh[1]):
            for z in range(sh[2]):
                key = ch[x, y, z]
                if key == 0:
                    continue
                # extract subcell object properties and add mapping info
                for ii in range(n_subcell):
                    subcell_key = subcell_chs[ii, x, y, z]
                    if subcell_key == 0:
                        continue
                    # subcell_mapping = subcell_mapping_dicts[ii]
                    if subcell_mapping_dicts[ii][subcell_key].count(key):
                        subcell_mapping_dicts[ii][subcell_key][key] += 1
                    else:
                        subcell_mapping_dicts[ii][subcell_key][key] = 1
    return subcell_mapping_dicts


def map_subcell_extract_props(n_type[:, :, :] ch, n_type[:, :, :, :] subcell_chs):
    """ch and subcell_chs must all have the same shape!
    TODO: check if uint32 and uint64 works with current unordered map definition,
    using n_type instead of uint64 results in an error.
    Returns rep coord, bounding box and size dict for cell segmentation objects,
    list of those property dicts for each subcellular segmentation type and number
    of overlapping voxels between subcellular and cell segmentation
    (Dict[Dict[uint]: subcell ID -> cell ID -> count)"""

    # property dicts for cell SV
    cdef unordered_map[uint64_t, int_vec] rep_coords
    cdef unordered_map[uint64_t, int_vec_vec] bounding_box
    cdef unordered_map[uint64_t, int] sizes
    cdef int_vec_vec *local_bb
    sh = ch.shape

    # property dicts for subcellular structures (e.g. mi, sj etc)
    cdef umvec_rc subcell_rc_dicts
    cdef umvec_bb subcell_bb_dicts
    cdef umvec_size subcell_size_dicts
    cdef umvec_map subcell_mapping_dicts

    n_subcell = subcell_chs.shape[0]
    for ii in range(n_subcell):
        assert (subcell_chs[ii].shape[0] == sh[0]) &\
               (subcell_chs[ii].shape[1] == sh[1]) &\
               (subcell_chs[ii].shape[2] == sh[2]), "Segmentation of cells and subcellular structures must have same shape. {} {} {} {} {} {}".format(
               subcell_chs[ii].shape[0], subcell_chs[ii].shape[1], subcell_chs[ii].shape[2], sh[0], sh[1], sh[2])
        subcell_rc_dicts.push_back(um_intvec_fact())
        subcell_bb_dicts.push_back(um_intvecvec_fact())
        subcell_size_dicts.push_back(um_umint_fact())
        subcell_mapping_dicts.push_back(um_umint_fact())

    for x in range(sh[0]):
        for y in range(sh[1]):
            for z in range(sh[2]):
                key = ch[x, y, z]
                # extract subcell object properties and add mapping info
                for ii in range(n_subcell):
                    #subcell_rc, subcell_bb, subcell_sizes = subcell_rc_dicts[ii], subcell_bb_dicts[ii], subcell_size_dicts[ii]
                    #subcell_mapping = subcell_mapping_dicts[ii]
                    subcell_key = subcell_chs[ii, x, y, z]
                    if subcell_key == 0:
                        continue
                    if subcell_bb_dicts[ii].count(subcell_key):
                        local_bb = & (subcell_bb_dicts[ii][subcell_key])
                        local_bb[0][0][0] = min(local_bb[0][0][0], x)
                        local_bb[0][0][1] = min(local_bb[0][0][1], y)
                        local_bb[0][0][2] = min(local_bb[0][0][2], z)
                        local_bb[0][1][0] = max(local_bb[0][1][0], x + 1)
                        local_bb[0][1][1] = max(local_bb[0][1][1], y + 1)
                        local_bb[0][1][2] = max(local_bb[0][1][2], z + 1)
                        subcell_size_dicts[ii][subcell_key] += 1
                        if key != 0:
                            if subcell_mapping_dicts[ii][subcell_key].count(key):
                                subcell_mapping_dicts[ii][subcell_key][key] += 1
                            else:
                                subcell_mapping_dicts[ii][subcell_key][key] = 1
                    else:
                        subcell_bb_dicts[ii][subcell_key] = ((x, y, z), (x+1, y+1, z+1))
                        subcell_size_dicts[ii][subcell_key] = 1
                        subcell_rc_dicts[ii][subcell_key] = [x, y, z]  # TODO: could be more representative
                        if key != 0:
                            # definitely first occurrence
                            subcell_mapping_dicts[ii][subcell_key][key] = 1

                # extract cell object properties
                if key == 0:
                    continue
                if bounding_box.count(key):
                    local_bb = & (bounding_box[key])
                    local_bb[0][0][0] = min(local_bb[0][0][0], x)
                    local_bb[0][0][1] = min(local_bb[0][0][1], y)
                    local_bb[0][0][2] = min(local_bb[0][0][2], z)
                    local_bb[0][1][0] = max(local_bb[0][1][0], x + 1)
                    local_bb[0][1][1] = max(local_bb[0][1][1], y + 1)
                    local_bb[0][1][2] = max(local_bb[0][1][2], z + 1)
                    sizes[key] += 1
                else:
                    bounding_box[key] = ((x, y, z), (x+1, y+1, z+1))
                    sizes[key] = 1
                    rep_coords[key] = [x, y, z]  # TODO: could be more representative
    return [rep_coords, bounding_box, sizes], [subcell_rc_dicts, subcell_bb_dicts, subcell_size_dicts], subcell_mapping_dicts
