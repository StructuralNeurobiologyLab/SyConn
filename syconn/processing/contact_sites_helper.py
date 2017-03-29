import numpy as np
import scipy.ndimage

from ..utils import datahandler, basics

from knossos_utils import chunky, knossosdataset


def contact_site_detection_thread(args):
    chunk = args[0]
    knossos_path = args[1]
    filename = args[2]

    kd = knossosdataset.KnossosDataset()
    kd.initialize_from_knossos_path(knossos_path)

    overlap = np.array([3, 3, 1], dtype=np.int)
    offset = np.array(chunk.coordinates - overlap)
    size = 2 * overlap + np.array(chunk.size)
    data = kd.from_overlaycubes_to_matrix(size, offset, datatype=np.uint32)

    contacts = detect_cs(data)

    datahandler.save_to_h5py([contacts],
                             chunk.folder + filename +
                             ".h5", ["cs"])


def detect_cs(arr):
    jac = np.zeros([3, 3, 3], dtype=np.int)
    jac[1, 1, 1] = -6
    jac[1, 1, 0] = 1
    jac[1, 1, 2] = 1
    jac[1, 0, 1] = 1
    jac[1, 2, 1] = 1
    jac[2, 1, 1] = 1
    jac[0, 1, 1] = 1

    edges = scipy.ndimage.convolve(arr.astype(np.int), jac) < 0

    edges = edges.astype(np.uint32)
    arr = arr.astype(np.uint32)

    # cs_seg = cse.process_chunk(edges, arr, [7, 7, 3])
    cs_seg = process_block_nonzero(edges, arr, [7, 7, 3])

    return cs_seg


def kernel(chunk, center_id):
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


def process_block(edges, arr, stencil=(7, 7, 3)):
    stencil = np.array(stencil, dtype=np.int)
    assert np.sum(stencil % 2) == 3

    out = np.zeros_like(arr, dtype=np.uint64)
    offset = stencil / 2
    for x in range(offset[0], arr.shape[0] - offset[0]):
        for y in range(offset[1], arr.shape[1] - offset[1]):
            for z in range(offset[2], arr.shape[2] - offset[2]):
                if edges[x, y, z] == 0:
                    continue

                center_id = arr[x, y, z]
                chunk = arr[x - offset[0]: x + offset[0] + 1, y - offset[1]: y + offset[1], z - offset[2]: z + offset[2]]
                out[x, y, z] = kernel(chunk, center_id)
    return out


def process_block_nonzero(edges, arr, stencil=(7, 7, 3)):
    stencil = np.array(stencil, dtype=np.int)
    assert np.sum(stencil % 2) == 3

    arr_shape = np.array(arr.shape)
    out = np.zeros(arr_shape - stencil + 1, dtype=np.uint64)
    offset = stencil / 2
    nze = np.nonzero(edges[offset[0]: -offset[0], offset[1]: -offset[1], offset[2]: -offset[2]])
    for x, y, z in zip(nze[0], nze[1], nze[2]):
        center_id = arr[x + offset[0], y + offset[1], z + offset[2]]
        chunk = arr[x: x + stencil[0], y: y + stencil[1], z: z + stencil[2]]
        out[x, y, z] = kernel(chunk, center_id)
    return out
