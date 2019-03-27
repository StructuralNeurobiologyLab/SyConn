# distutils: language = c++
from libc.stdint cimport uint64_t, uint32_t
import numpy as np
cimport cython

ctypedef fused n_type:
    uint64_t
    uint32_t

def extract_bounding_box(n_type[:, :, :] chunk):
    box = dict()

    for x in range(chunk.shape[0]):
        for y in range(chunk.shape[1]):
            for z in range(chunk.shape[2]):
                key = chunk[x, y, z]
                if key in box:
                    box[key] = { np.array([min(box[key][0][0], x), min(box[key][0][1], y), min(box[key][0][2], z)]),
                                 np.array([max(box[key][1][0], x), max(box[key][1][1], y), max(box[key][1][2], z)])}

                else:
                    box[key] = {np.array([x, y, z]), np.array([x, y, z])}
    return box
