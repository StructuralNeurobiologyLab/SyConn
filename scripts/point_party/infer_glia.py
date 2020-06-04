import re
from syconn.handler import config
import numpy as np
from syconn.handler.prediction_pts import predict_glia_ssv
from syconn.reps.super_segmentation import SuperSegmentationDataset
import os
import networkx as nx

if __name__ == '__main__':
    wd = "/wholebrain/songbird/j0126/areaxfs_v6/"
    base_dir = '/wholebrain/scratch/pschuber/e3trainings_BAK/ptconv_2020_06_03/'
    ssd_kwargs = dict(working_dir=wd)
    mdir = base_dir + '/glia_pts_scale1500_nb30000_ctx15000_swish_gn_eval0/'

    mpath = f'{mdir}/state_dict_final.pth'
    assert mpath

    ssd = SuperSegmentationDataset(**ssd_kwargs)
    ixs = np.random.choice(len(ssd.ssv_ids), 10, replace=False)
    ssv_ids = ssd.ssv_ids[ixs]
    sizes = ssd.load_cached_data('size')[ixs]
    ssv_ids = ssv_ids[np.argsort(sizes)[::-1]]
    ssv_params = [dict(ssv_id=ssv.id, sv_ids=ssv.sv_ids, sv_graph=ssv.sv_graph_uint,
                       **ssd_kwargs) for ssv in ssd.get_super_segmentation_object(ssv_ids)]
    nodes = [10861339, 10705496, 16300957, 16393831, 16300965, 16394199]
    g = nx.generators.classic.complete_graph(nodes)
    ssv_params.append(dict(ssv_id=10861339, sv_ids=nodes, sv_graph=g, **ssd_kwargs))
    res = predict_glia_ssv(ssv_params, mpath=mpath, nloader=1, npredictor=1,
                           postproc_kwargs=dict(pred_key='glia_probas_test'))
    raise()
