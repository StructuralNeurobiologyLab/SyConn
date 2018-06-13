# SyConn
# Copyright (c) 2018 Philipp J. Schubert, J. Kornfeld
# All rights reserved
from syconn.config.global_params import wd
from syconn.handler.prediction import get_axoness_model_V2
from syconn.handler.basics import chunkify
from syconn.reps.super_segmentation import SuperSegmentationDataset
from syconn.mp import qsub_utils as qu
import numpy as np
import os


def axoness_pred_exists(sv):
    sv.load_attr_dict()
    return 'axoness_probas_v2' in sv.attr_dict

if __name__ == "__main__":
    # ssd = SuperSegmentationDataset(working_dir=wd)
    # m = get_axoness_model_V2()
    # # shuffle SV IDs
    # np.random.seed(0)
    # ssv_ids = ssd.ssv_ids
    # np.random.shuffle(ssv_ids)
    # pbar = tqdm.tqdm(total=len(ssv_ids))
    # # TODO perform SSV chunk processing
    # for ix in ssv_ids:  # good test sample: [21383168]
    #     ssv = ssd.get_super_segmentation_object(ix)
    #     ssv.nb_cpus = 25
    #     res = start_multiprocess_imap(axoness_pred_exists, ssv.svs,
    #                                   nb_cpus=ssv.nb_cpus)
    #     if not np.all(res):
    #         ssv.predict_views_axoness(m, pred_key_appendix="_v2", verbose=False)
    #     pbar.update(1)


# NEW AND UNTESTED
    ssd = SuperSegmentationDataset(working_dir=wd)
    sv_ids = ssd.sv_ids
    np.random.shuffle(sv_ids)
    # chunk them
    multi_params = chunkify(sv_ids, 100000)
    # get model properties
    m = get_axoness_model_V2()
    model_kwargs = dict(model_path=m._path, normalize_data=m.normalize_data,
                        imposed_batch_size=m.imposed_batch_size, nb_labels=m.nb_labels,
                        channels_to_load=m.channels_to_load)
    # all other kwargs like obj_type='sv' and version are the current SV SegmentationDataset by default
    so_kwargs = dict(working_dir=wd)
    # for axoness views set woglia to True (because glia were removed beforehand),
    #  raw_only to False
    pred_kwargs = dict(woglia=True, pred_key="_v2", nb_cpus=1, verbose=False,
                       raw_only=False)

    multi_params = [[par, model_kwargs, so_kwargs, pred_kwargs] for par in multi_params[:2]]
    script_folder = os.path.dirname(
        os.path.abspath(__file__)) + "/../../syconn/QSUB_scripts/"
    path_to_out = qu.QSUB_script(multi_params, "predict_sv_views",
                                 n_max_co_processes=30, pe="openmp", queue=None,
                                 script_folder=script_folder, n_cores=10,
                                 suffix="_axoness")
