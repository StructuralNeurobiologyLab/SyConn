# SyConn
# Copyright (c) 2018 Philipp J. Schubert, J. Kornfeld
# All rights reserved
import numpy as np
import os
import glob
try:
    import cPickle as pkl
except ImportError:
    import pickle as pkl

from syconn.config import global_params
from syconn.handler.logger import initialize_logging
from syconn.handler.basics import chunkify
from syconn.reps.super_segmentation import SuperSegmentationDataset
from syconn.mp import qsub_utils as qu


# ~2.5h with 24 gpus
if __name__ == "__main__":
    log = initialize_logging('celltype_prediction', global_params.wd + '/logs/',
                             overwrite=False)
    ssd = SuperSegmentationDataset(working_dir=global_params.wd)
    # shuffle SV IDs
    np.random.seed(0)
    ssv_ids = ssd.ssv_ids

    log.info('Starting cell type prediction.')
    nb_svs_per_ssv = np.array([len(ssd.mapping_dict[ssv_id])
                               for ssv_id in ssd.ssv_ids])
    multi_params = ssd.ssv_ids
    ordering = np.argsort(nb_svs_per_ssv)
    multi_params = multi_params[ordering[::-1]]
    multi_params = chunkify(multi_params, 100)
    # job parameter will be read sequentially, i.e. in order to provide only
    # one list as parameter one needs an additonal axis
    multi_params = [(ixs, ) for ixs in multi_params]

    path_to_out = qu.QSUB_script(multi_params, "predict_cell_type", pe="openmp",
                                 n_max_co_processes=34, queue=None,
                                 script_folder=None, suffix="",
                                 n_cores=10, additional_flags="--gres=gpu:1")
    log.info('Finished prediction of {} SSVs. Checking completeness.'
             ''.format(len(ordering)))
    out_files = glob.glob(path_to_out + "*.pkl")
    err = []
    for fp in out_files:
        with open(fp, "rb") as f:
            local_err = pkl.load(f)
        err += list(local_err)
    if len(err) > 0:
        log.error("{} errors occurred for SSVs with ID: "
                  "{}".format(len(err), [el[0] for el in err]))
    else:
        log.info('Success.')
