# SyConn
# Copyright (c) 2018 Philipp J. Schubert, J. Kornfeld
# All rights reserved
import numpy as np
import tqdm

from syconn.config import global_params
from syconn.handler.logger import initialize_logging
from syconn.handler.prediction import get_celltype_model
from syconn.handler.basics import chunkify
from syconn.reps.super_segmentation import SuperSegmentationDataset, SuperSegmentationObject
from syconn.reps.super_segmentation_helper import predict_sso_celltype
from syconn.mp.mp_utils import start_multiprocess_imap


def celltype_predictor(args):
    ssv_ids = args
    # randomly initialize gpu
    m = get_celltype_model(init_gpu=np.random.randint(0, 2))
    pbar = tqdm.tqdm(total=len(ssv_ids))
    missing_ssvs = []
    for ix in ssv_ids:
        ssv = SuperSegmentationObject(ix, working_dir=global_params.wd)
        ssv.nb_cpus = 1
        try:
            predict_sso_celltype(ssv, m, overwrite=True)
        except Exception as e:
            missing_ssvs.append((ssv.id, e))
            print(repr(e))
        pbar.update(1)
    pbar.close()
    return missing_ssvs


if __name__ == "__main__":
    log = initialize_logging('celltype_prediction', global_params.wd + '/logs/')
    ssd = SuperSegmentationDataset(working_dir=global_params.wd)
    # shuffle SV IDs
    np.random.seed(0)
    ssv_ids = ssd.ssv_ids
    np.random.shuffle(ssv_ids)
    # TODO: use BATCHJOB Script
    log.info('Starting cell type prediction.')
    err = start_multiprocess_imap(celltype_predictor, chunkify(ssd.ssv_ids, 15),
                                  nb_cpus=6)
    err = np.concatenate(err)
    log.info('Finished cell type prediction. Checking completeness.')
    if len(err) > 0:
        log.error("{} errors occurred for SSVs with ID: "
                  "{}".format(len(err), [el[0] for el in err]))
    else:
        log.info('Success.')


# TODO: perform async. data loading and model predictions, see
# https://stackoverflow.com/questions/12474182/asynchronously-read-and-process-an-image-in-python