import numpy as np
from syconn.handler.prediction_pts import predict_cpmt_ssv
from syconn.reps.super_segmentation import SuperSegmentationDataset


if __name__ == '__main__':
    wd = "/wholebrain/songbird/j0126/areaxfs_v6/"
    mpath = ''

    ssd_kwargs = dict(working_dir=wd)
    ssd = SuperSegmentationDataset(**ssd_kwargs)

    # ixs = np.random.choice(len(ssd.ssv_ids), 10, replace=False)
    # ssv_ids = ssd.ssv_ids[ixs]
    # sizes = ssd.load_cached_data('size')[ixs]
    # ssv_ids = ssv_ids[np.argsort(sizes)[::-1]]

    ssv_ids = np.array([2734465, 2854913, 8003584, 8339462, 10919937, 26169344, 26501121])
    ssv_params = [dict(ssv_id=ssv.id, sv_ids=ssv.sv_ids, sv_graph=ssv.sv_graph_uint,
                       **ssd_kwargs) for ssv in ssd.get_super_segmentation_object(ssv_ids)]

    res = predict_cpmt_ssv(ssv_params, mpath=mpath, nloader=1, npredictor=1,
                           postproc_kwargs=dict(pred_key='cpmt_probas_test'))
