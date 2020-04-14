from syconn.reps.rep_helper import ix_from_subfold_new, subfold_from_ix_new, get_unique_subfold_ixs
from syconn.reps.segmentation import SegmentationDataset
import numpy as np
import tempfile
import shutil
from collections import defaultdict
from syconn import global_params

lst_n_folder_fs = [10 ** i for i in range(1, 4)]


def test_subfold_from_ix():
    obj_ids = np.arange(1e5)
    for n_folder_fs in lst_n_folder_fs:
        dest_dc = defaultdict(list)
        for obj_id in obj_ids:
            storage_ident = subfold_from_ix_new(obj_id, n_folder_fs)
            dest_dc[storage_ident].append(obj_id)
        stored_ids = np.concatenate(list(dest_dc.values()))
        u_ids = np.unique(stored_ids)
        assert len(u_ids) == len(stored_ids)


def test_subfold2ix_inverse():
    with tempfile.TemporaryDirectory() as working_dir:
        global_params.wd = working_dir
        global_params.config['paths']['use_new_subfold'] = True
        for n_folder_fs in lst_n_folder_fs:
            storage_rep_ids = get_unique_subfold_ixs(n_folder_fs)
            for rep_id in storage_rep_ids:
                storage_ident = subfold_from_ix_new(rep_id, n_folder_fs)
                assert rep_id == ix_from_subfold_new(storage_ident, n_folder_fs)


def test_uint64():
    return
