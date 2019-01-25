import numpy as np
import time
from collections import defaultdict as ddict
import itertools
import csv
from knossos_utils import knossosdataset
knossosdataset._set_noprint(True)
from syconn.handler import basics
import multiprocessing
from syconn import global_params


def new_extraction(result_queue, params):
    this_segmentation = params[0]
    subvol_size = params[1]
    offset = params[2]

    try:
        start = time.time()
        uniqueID_coords_dict = ddict(list)  # {sv_id: [(x0,y0,z0),(x1,y1,z1),...]}
        extracted_voxels_dict = ddict(list)

        dims = this_segmentation.shape
        indices = itertools.product(range(dims[0]), range(dims[1]), range(dims[2]))
        for idx in indices:
            sv_id = this_segmentation[idx]
            uniqueID_coords_dict[sv_id].append(idx)

        for sv_id in uniqueID_coords_dict:
            if sv_id == 0:
                continue

            sv_coords = uniqueID_coords_dict[sv_id]
            sv_mask_offset = np.min(sv_coords, axis=0)

            sv_mask_coords = sv_coords - sv_mask_offset
            size = np.max(sv_coords, axis=0) - sv_mask_offset + (1, 1, 1)
            sv_mask_coords = np.transpose(sv_mask_coords)
            sv_mask = np.zeros(tuple(size), dtype=bool)
            sv_mask[sv_mask_coords[0, :], sv_mask_coords[1, :], sv_mask_coords[2, :]] = True
            extracted_voxels_dict[sv_id] = sv_mask, sv_mask_offset

        total_time = time.time() - start
        print(subvol_size, offset, 'new returned')
        result_queue.put(['new', subvol_size, offset, len(list(extracted_voxels_dict.keys())), total_time])
    except:
        print(subvol_size, offset, 'new failed')
        result_queue.put(['new', subvol_size, offset, 0, -1])


def old_extraction_modified(result_queue, params):
    this_segmentation = params[0]
    subvol_size = params[1]
    offset = params[2]

    try:
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

        total_time = time.time() - start
        print(subvol_size, offset, 'old returned')
        result_queue.put(['old', subvol_size, offset, len(list(extracted_voxels_dict.keys())), total_time])
    except:
        print(subvol_size, offset, 'old failed')
        result_queue.put(['old', subvol_size, offset, 0, -1])


if __name__ == '__main__':

    kd_seg_path = global_params.config.kd_seg_path

    kd = knossosdataset.KnossosDataset()
    kd.initialize_from_knossos_path(kd_seg_path)

    subvol_sizes = [[16, 16, 16], [32, 32, 32], [64, 64, 64], [128, 128, 128], [192, 192, 192], [256, 256, 256],
                          [384, 384, 384], [512, 512, 512]]
    subvol_offsets = [[4516, 9397, 173], [5556, 5216, 1423], [3650, 7744, 1220], [8895, 1951, 987], [6856, 7058, 3227],
                    [4354, 7764, 3268], [4167, 8257, 2793]]  # , [6397, 6916, 293], [203, 7004, 2585], [714, 7767, 2538]]

    result_queue = multiprocessing.Queue()

    params = []
    for size in subvol_sizes:
        for offset in subvol_offsets:
            this_segmentation = kd.from_overlaycubes_to_matrix(size, offset)
            params.append([this_segmentation, size, offset])


    # size = [512, 512, 512]
    # for offset in subvol_offsets:
    #     this_segmentation = kd.from_overlaycubes_to_matrix(size, offset)
    #     params.append([this_segmentation, size, offset])

    processes = []
    for param in params:
        p_new = multiprocessing.Process(target=new_extraction, args=(result_queue, param))
        p_old = multiprocessing.Process(target=old_extraction_modified, args=(result_queue, param))
        processes.append(p_new)
        processes.append(p_old)
        p_new.start()
        p_old.start()

    for p in processes:
        p.join()

    results = [['extraction_type', 'subvolume_size', 'offset', 'no_of_objects', 'extraction_time']]
    while result_queue.empty() == False:
        results.append(result_queue.get())

    with open('/wholebrain/u/jmark/extraction_benchmark_all_runs_07_offsets.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows(results)
