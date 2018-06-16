# SyConn
# Copyright (c) 2018 Philipp J. Schubert, J. Kornfeld
# All rights reserved
from syconn.config.global_params import wd
from syconn.handler.prediction import get_celltype_model
from syconn.handler.basics import chunkify
from syconn.reps.super_segmentation import SuperSegmentationDataset, SuperSegmentationObject
from syconn.reps.super_segmentation_helper import predict_sso_celltype
from syconn.mp import qsub_utils as qu
from syconn.mp.shared_mem import start_multiprocess_imap, start_multiprocess
import numpy as np
import tqdm
import time
import os


def predictor(args):
    ssv_ids = args
    # randomly initialize gpu
    m = get_celltype_model(init_gpu=np.random.randint(0, 2))
    pbar = tqdm.tqdm(total=len(ssv_ids))
    missing_ssvs = []
    for ix in ssv_ids:
        ssv = SuperSegmentationObject(ix, working_dir=wd)
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
    ssd = SuperSegmentationDataset(working_dir=wd)
    # shuffle SV IDs
    np.random.seed(0)
    ssv_ids = ssd.ssv_ids
    np.random.shuffle(ssv_ids)
    err = start_multiprocess_imap(predictor, chunkify(ssd.ssv_ids, 15),
                                  nb_cpus=6)
    err = np.concatenate(err)
    if len(err) > 0:
        print("{} errors occurred for SSVs with ID: "
              "{}".format(len(err), [el[0] for el in err]))


# TODO: perform async. data loading and model predictions, see
# https://stackoverflow.com/questions/12474182/asynchronously-read-and-process-an-image-in-python