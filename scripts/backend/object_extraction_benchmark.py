import numpy as np
import time
from collections import defaultdict as ddict
from functools import partial

from knossos_utils import knossosdataset
from syconn.handler import basics


def new_extraction(this_segmentation):

    start = time.time()
    alist = this_segmentation.tolist()
    uniqueID_coords_dict = ddict(list)  # {sv_id: [(x1,y1,z1),(x2,y2,z2),...]}
    extracted_voxels_dict = ddict(list)
    len_0 = len(this_segmentation)
    len_1 = len(this_segmentation[0])
    len_2 = len(this_segmentation[0][0])
    for x in range(len_0):         # faster than np.enumerate() and np.ndindex()
        for y in range(len_1):
            for z in range(len_2):
                sv_id = alist[x][y][z]
                if sv_id == 0:
                    continue
                uniqueID_coords_dict[sv_id].append((x, y, z))
    print("first block: ", time.time() - start)

    start2 = time.time()

    for sv_id in uniqueID_coords_dict.keys():
        sv_coords = uniqueID_coords_dict[sv_id]
        mask_offset = np.min(sv_coords, axis=0)
        mask_coords = sv_coords - mask_offset
        size = np.max(sv_coords, axis=0) - mask_offset + (1, 1, 1)
        mask_coords = np.transpose(mask_coords)
        mask = np.zeros(tuple(size), dtype=bool)
        mask[mask_coords[0, :], mask_coords[1, :], mask_coords[2, :]] = True
        extracted_voxels_dict[sv_id] = mask, mask_offset

    end = time.time()
    print("second block: ", end - start2)
    print("New extraction: Processing %d objects took %ds" % (len(uniqueID_coords_dict.keys()), end - start))

    return extracted_voxels_dict


def old_extraction_modified(this_segmentation):

    start = time.time()
    unique_ids = np.unique(this_segmentation)
    extracted_voxels_dict = {}
    for i_unique_id in range(len(unique_ids)):
        unique_id = unique_ids[i_unique_id]

        if unique_id == 0:
            continue

        id_mask = this_segmentation == unique_id
        id_mask, in_chunk_offset = basics.crop_bool_array(id_mask)
        extracted_voxels_dict[unique_id] = id_mask, in_chunk_offset

    end = time.time()
    print("Old extraction: Processing %d objects took %ds" % (len(extracted_voxels_dict.keys()), end - start))
    # import pdb; pdb.set_trace()
    return extracted_voxels_dict


if __name__ == '__main__':
    # set parameters
    # subvol_size = [512, 512, 512]
    # subvol_size = [300, 300, 300]
    # subvol_size = [256, 256, 256]
    # subvol_size = [126, 126, 126]
    subvol_size = [16, 16, 16]
    # subvol_offset = [500, 500, 100]
    # subvol_offset = [300, 300, 300]
    subvol_offset = [0, 0, 0]

    kd_seg_path = "/wholebrain/scratch/areaxfs_example/1k_cube/"
    sd_path = "/wholebrain/scratch/jmark/oe_test/"

    kd = knossosdataset.KnossosDataset()
    kd.initialize_from_knossos_path(kd_seg_path)

    try:
        this_segmentation = kd.from_overlaycubes_to_matrix(subvol_size,     # return np.ndarray
                                                           subvol_offset)
    except:
        this_segmentation = kd.from_overlaycubes_to_matrix(subvol_size,
                                                           subvol_offset,
                                                           datatype=np.uint32)

    new = new_extraction(this_segmentation)
    old = old_extraction_modified(this_segmentation)
