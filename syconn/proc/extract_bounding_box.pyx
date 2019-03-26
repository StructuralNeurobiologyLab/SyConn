# distutils: language = c++
from libc.stdint cimport uint64_t, uint32_t
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
                    box[key] = {'xMin': min(box[key]['xMin'], x), 'xMax': max(box[key]['xMax'], x),
                                'yMin': min(box[key]['yMin'], y), 'yMax': max(box[key]['yMax'], y),
                                'zMin': min(box[key]['zMin'], z), 'zMax': max(box[key]['zMax'], z)}
                else:
                    box[key] = {'xMin': x, 'xMax': x, 'yMin': y, 'yMax': y,
                                'zMin': z, 'zMax': z}
    return box
