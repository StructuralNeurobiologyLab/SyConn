#!python
# import numpy as np
import operator

import numpy as np
cimport numpy as np

DTYPE = np.uint32
ctypedef np.uint32_t DTYPE_t

def bincount(np.ndarray arr):
    bindict = {}

    cdef np.ndarray[DTYPE_t, ndim=1] unique_elements
    unique_elements, counts = np.unique(arr, return_counts=True)
    if unique_elements[0] == 0:
        return unique_elements[1:], counts[1:]
    else:
        return unique_elements, counts


def kernel(np.ndarray[np.uint32_t, ndim=1] chunk, center_id):
    unique_ids, counts = np.unique(chunk, return_counts=True)

    counts[unique_ids == 0] = -1
    counts[unique_ids == center_id] = -1

    if np.max(counts) > 0:
        partner_id = unique_ids[np.argmax(counts)]

        if center_id > partner_id:
            return (partner_id << 32) + center_id
        else:
            return (center_id << 32) + partner_id
    else:
        return 0

def process_block(np.ndarray[np.uint32_t, ndim = 3] edges,
                  np.ndarray[np.uint32_t, ndim = 3] arr,
                  np.ndarray[np.int, ndim=1] kernel_size):
    cdef arr_shape = np.shape(arr)
    cdef  out = np.zeros((arr_shape[0] - kernel_size[0] + 1,
                          arr_shape[1] - kernel_size[1] + 1,
                          arr_shape[2] - kernel_size[2] + 1), dtype=np.uint64)

    cdef np.npy_intp x, y, z

    cdef int x_offset = np.floor(kernel_size[0] / 2)
    cdef int y_offset = np.floor(kernel_size[1] / 2)
    cdef int z_offset = np.floor(kernel_size[2] / 2)

    cdef nze = np.nonzero(edges[x_offset: -x_offset,
                                y_offset: -y_offset,
                                z_offset: -z_offset])

    for x, y, z in zip(nze[0], nze[1], nze[2]):
        center_id = arr[x + x_offset, y + y_offset, z + z_offset]
        chunk = arr[x: x + kernel_size[0], y: y + kernel_size[1], z: z + kernel_size[2]]
        out[x, y, z] = kernel(chunk, center_id)
    return out