# distutils: language = c++
from libcpp.unordered_map cimport unordered_map
cimport cython
from libcpp cimport bool
from libc.stdint cimport int, float
from libc.stdint cimport uint64_t, uint32_t
from libcpp.vector cimport vector

ctypedef vector[int] int_vec
ctypedef vector[int_vec] int_vec_vec


ctypedef fused n_type:
    uint64_t
    uint32_t


def _find_object_propertiesC(n_type[:, :, :] chunk):
    cdef unordered_map[uint64_t, int_vec] rep_coords
    cdef unordered_map[uint64_t, int_vec_vec] bounding_box
    cdef unordered_map[uint64_t, int] sizes

    for x in range(chunk.shape[0]):
        for y in range(chunk.shape[1]):
            for z in range(chunk.shape[2]):
                key = chunk[x, y, z]
                if bounding_box.count(key):
                    old_bb = bounding_box[key]
                    bounding_box[key] = [[min(old_bb[0][0], x),
                    min(old_bb[0][1], y), min(old_bb[0][2], z)],
                    [max(old_bb[1][0], x+1),
                    max(old_bb[1][1], y+1),
                    max(old_bb[1][2], z+1)]]
                    sizes[key] += 1
                else:
                    bounding_box[key] = [[x, y, z], [x+1, y+1, z+1]]
                    sizes[key] = 1
                    rep_coords[key] = [x, y, z]  # TODO: could be more representative
    return rep_coords, bounding_box, sizes


#def extract_bounding_box(n_type[:, :, :] chunk):
#    box = dict()
#    for x in range(chunk.shape[0]):
#        for y in range(chunk.shape[1]):
#            for z in range(chunk.shape[2]):
#                key = chunk[x, y, z]
#                if key in box:
#                    box[key] = {np.array([min(box[key][0][0], x), min(box[key][0][1], y), min(box[key][0][2], z)]),
#                                np.array([max(box[key][1][0], x), max(box[key][1][1], y), max(box[key][1][2], z)])}
#
#                else:
#                    box[key] = {np.array([x, y, z]), np.array([x, y, z])}
#    return box
