# distutils: language = c++
from cython.view cimport array as cvarray
from libc.stdint cimport uint64_t, uint32_t, uint8_t
cimport cython
from libc.stdlib cimport rand
from libcpp.map cimport map
from cython.operator import dereference, postincrement
from libcpp.unordered_map cimport unordered_map
from libcpp.vector cimport vector
import numpy as np
import timeit

ctypedef unordered_map[uint64_t, uint64_t] um_uint2uint
ctypedef vector[int] int_vec
ctypedef vector[int_vec] int_vec_vec

ctypedef fused n_type:
    uint64_t
    uint32_t


# TODO: consider changing def to cdef
def kernel(uint32_t[:, :, :] chunk, uint64_t center_id):

    cdef map[uint32_t, int] unique_ids

    for i in range(chunk.shape[0]):
        for j in range(chunk.shape[1]):
            for k in range(chunk.shape[2]):
                unique_ids[chunk[i][j][k]] = unique_ids[chunk[i][j][k]] + 1
    unique_ids[0] = 0
    unique_ids[center_id] = 0
    cdef int theBiggest  = 0
    cdef uint64_t key = 0

    cdef map[uint32_t,int].iterator it = unique_ids.begin()
    while it != unique_ids.end():
        if dereference(it).second > theBiggest:
            theBiggest =  dereference(it).second
            key = dereference(it).first
        postincrement(it)

    if theBiggest > 0:
        if center_id > key:
            return (key << 32 ) + center_id
        else:
            return (center_id << 32) + key

    else:
        return key


def process_block(uint32_t[:, :, :] edges, uint32_t[:, :, :] arr, stencil1=(7,7,3)):

    cdef int stencil[3]
    stencil[:] = [stencil1[0], stencil1[1], stencil1[2]]
    assert (stencil[0]%2 + stencil[1]%2 + stencil[2]%2 ) == 3
    cdef uint64_t[:, :, :] out = cvarray(shape = (arr.shape[0], arr.shape[1],
                                                  arr.shape[2]), itemsize = sizeof(uint64_t), format = 'Q')
    out [:, :, :] = 0
    cdef int offset[3]
    offset[:] = [stencil[0]//2, stencil[1]//2, stencil[2]//2] ### check what type do you need
    cdef int center_id
    cdef uint32_t[:, :, :] chunk = cvarray(shape=(2*offset[0]+2, 2*offset[2]+2, 2*offset[2]+2),
                                           itemsize=sizeof(int), format='i')

    for x in range(offset[0], arr.shape[0] - offset[0]):
        for y in range(offset[1], arr.shape[1] - offset[1]):
            for z in range(offset[2], arr.shape[2] - offset[2]):
                if edges[x, y, z] == 0:
                    continue

                center_id = arr[x, y, z] #be sure that it's 32 or 64 bit intiger
                chunk = arr[x - offset[0]: x + offset[0] + 1, y - offset[1]: y + offset[1], z - offset[2]: z + offset[2]]
                out[x, y, z] = kernel(chunk, center_id)

    return out


def process_block_nonzero(uint32_t[:, :, :] edges, uint32_t[:, :, :] arr, stencil1=(7,7,3)):

    cdef int stencil[3]
    stencil[:] = [stencil1[0], stencil1[1], stencil1[2]]
    assert (stencil[0]%2 + stencil[1]%2 + stencil[2]%2 ) == 3

    cdef uint64_t[:, :, :] out = cvarray(shape = (1 + arr.shape[0] - stencil[0],
                                                arr.shape[1] - stencil[1] + 1, arr.shape[2] - stencil[2] + 1),
                                                itemsize = sizeof(uint64_t), format = 'Q')
    out[:, :, :] = 0
    cdef int center_id
    cdef int offset[3]
    offset [:] = [stencil[0]//2, stencil[1]//2, stencil[2]//2]
    cdef uint32_t[:, :, :] chunk = cvarray(shape=(stencil[0]+1, stencil[1]+1, stencil[2]+1),
                                           itemsize=sizeof(uint32_t), format='I')

    for x in range(0, edges.shape[0]-2*offset[0]):
        for y in range(0, edges.shape[1]-2*offset[1]):
            for z in range(0, edges.shape[2]-2*offset[2]):
                if edges[x+offset[0], y+offset[1], z+offset[2]] == 0:
                    continue
                center_id = arr[x + offset[0], y + offset[1], z + offset[2]]
                chunk = arr[x: x + stencil[0], y: y + stencil[1], z: z + stencil[2]]
                out[x, y, z] =  kernel(chunk, center_id)
    return out


def extract_cs_syntype(n_type[:, :, :] cs_seg, uint8_t[:, :, :] syn_mask,
    uint8_t[:, :, :] asym_mask, uint8_t[:, :, :] sym_mask):
    """cs_seg, syn_mask, sym_mask and asym_mask  must all have the same shape!
    TODO: check if uint32 and uint64 works with current unordered map definition, using n_type instead of uint64 results in an error.
    Returns synaptic properties for every contact site ID inside `cs_seg`.
    Dict, Dict
    Count of synaptic foreground voxel and if voxel is foreground also sums the number of symmetric
    and asymmetric voxels. The type ratio can be computed
    by using the total syn foreground voxels assigned to the CS object.
    """
    cdef unordered_map[uint64_t, int_vec] rep_coords
    cdef unordered_map[uint64_t, int_vec_vec] bounding_box
    cdef unordered_map[uint64_t, int] sizes
    cdef int_vec_vec *local_bb

    sh = cs_seg.shape

    cdef um_uint2uint cs_syn
    cdef um_uint2uint cs_asym
    cdef um_uint2uint cs_sym
    cdef int syntype_vx
    cdef n_type syn_vx

    for x in range(sh[0]):
        for y in range(sh[1]):
            for z in range(sh[2]):
                key = cs_seg[x, y, z]

                # update CS properties
                # IMPORTANT! ONLY COUNT SYN TYPES IF FOREGROUND IS TRUE
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
                    rep_coords[key] = [x, y, z]
                # extract syntype info

                # store sym. and asym. voxels counts
                if asym_mask[x, y, z]:
                    if cs_asym.count(key):
                        cs_asym[key] += 1
                    else:
                        cs_asym[key] = 1
                if sym_mask[x, y, z]:
                    if cs_sym.count(key):
                        cs_sym[key] += 1
                    else:
                        cs_sym[key] = 1
    return [rep_coords, bounding_box, sizes], cs_asym, cs_sym