# SyConn
# Copyright (c) 2018 Philipp J. Schubert, J. Kornfeld
# All rights reserved
from syconn.config.global_params import wd
from syconn.handler.prediction import get_axoness_model_V2
from syconn.reps.super_segmentation import SuperSegmentationDataset
import numpy as np
import tqdm


if __name__ == "__main__":
    ssd = SuperSegmentationDataset(working_dir=wd)
    m = get_axoness_model_V2()
    # shuffle SV IDs
    np.random.seed(0)
    ssv_ids = ssd.ssv_ids
    np.random.shuffle(ssv_ids)
    pbar = tqdm.tqdm(total=len(ssv_ids))
    # TODO perform SSV chunk processing
    for ix in ssv_ids:
        ssv = ssd.get_super_segmentation_object(ix)
        ssv.nb_cpus = 20
        ssv.predict_views_axoness(m, pred_key_appendix="_v2", verbose=False)
        pbar.update(1)