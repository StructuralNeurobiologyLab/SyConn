from syconn import global_params
from syconn.reps.super_segmentation import SuperSegmentationDataset
from syconn.reps.segmentation import SegmentationDataset
import numpy as np
import time
import os
from syconn.handler.basics import load_pkl2obj, write_obj2pkl
from syconn.handler.config import initialize_logging
from syconn.mp import batchjob_utils as qu
from syconn.handler.basics import chunkify_weighted
from syconn.handler.config import initialize_logging
from syconn.mp import batchjob_utils as qu
from syconn import global_params
import shutil
from syconn.proc.analysis_prep_func import synapse_amount_percell

def find_full_cells(ct_list, filename, syn_amount=True):
    '''
    # celltypes: j0256: STN = 0, DA = 1, MSN = 2, LMAN = 3, HVC = 4, TAN = 5, GPe = 6, GPi = 7,
#                      FS=8, LTS=9, NGF=10
    Args:
        ct_list: list of celltypes in int
        filename: name of directory dictionaries are saved in
        syn_amount: if True, also create a dictionary with amount of synapses per cell

    Returns:

    '''

    ssd = SuperSegmentationDataset(working_dir=global_params.config.working_dir)
    sd_synssv = SegmentationDataset("syn_ssv", working_dir=global_params.config.working_dir)
    if not os.path.exists(filename):
        os.mkdir(filename)
    log = initialize_logging('analysis prep', log_dir=filename + '/logs/')
    ct_dict = {0: "STN", 1: "DA", 2: "MSN", 3: "LMAN", 4: "HVC", 5: "TAN", 6: "GPe", 7: "GPi", 8: "FS", 9: "LTS", 10: "NGF"}

    max_n_jobs = min(global_params.config.ncore_total * 4, 1000)
    for ix, ct in enumerate(ct_list):
        log.info('Step %.1i/%.1i find full cells of celltype %.3s' % (ix + 1, len(ct_list), ct_dict[ct]))
        cell_ids = ssd.ssv_ids[ssd.load_cached_data("celltype_cnn_e3") == ct]
        multi_params = chunkify_weighted(cell_ids, max_n_jobs,
                                         ssd.load_cached_data('size')[ssd.load_cached_data("celltype_cnn_e3") == ct])
        out_dir = qu.batchjob_script(multi_params, "findfullcells", log=log, remove_jobfolder=False, n_cores=1,
                                     max_iterations=10)
        cell_array = np.zeros(len(cell_ids))
        somas = np.zeros((len(cell_ids), 3))
        for i, file in enumerate(out_dir):
            part_list = load_pkl2obj(file)
            if len(part_list) != 2:
                continue
            part_cell_array = part_list[0]
            part_soma_centers = part_list[1]
            start = i * len(part_cell_array)
            cell_array[start:start + len(part_cell_array) - 1] = part_cell_array.astype(int)
            somas[start:start + len(part_cell_array) - 1] = part_soma_centers

        inds = np.array(cell_array != 0)
        cell_array = cell_array[inds]
        somas = somas[inds]
        cell_dict = {int(cell_array[i]): somas[i] for i in range(0, len(cell_array))}
        dict_path = ("%s/full_%.3s_dict.pkl" % (filename, ct_dict[ct]))
        arr_path = ("%s/full_%.3s_arr.pkl" % (filename, ct_dict[ct]))
        write_obj2pkl(dict_path, cell_dict)
        write_obj2pkl(arr_path, cell_array)
        shutil.rmtree(os.path.abspath(out_dir + '/../'))
