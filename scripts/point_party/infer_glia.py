import re
from syconn.handler import config
import numpy as np
from syconn.handler.prediction_pts import predict_glia_ssv
from syconn.reps.super_segmentation import SuperSegmentationDataset, SuperSegmentationObject
import os
import networkx as nx

if __name__ == '__main__':
    wd = "/wholebrain/songbird/j0126/areaxfs_v6/"
    base_dir = '/wholebrain/scratch/pschuber/e3_trainings_convpoint/'
    ssd_kwargs = dict(working_dir=wd)
    mdir = base_dir + '/glia_pts_scale500_nb10000_ctx5000_swish_gn_eval0/'

    mpath = f'{mdir}/state_dict.pth'
    assert mpath

    ssd = SuperSegmentationDataset(**ssd_kwargs)
    np.random.seed(0)
    # ixs = np.random.choice(len(ssd.ssv_ids), 2, replace=False)
    # ssv_ids = ssd.ssv_ids[ixs]
    # sizes = ssd.load_numpy_data('size')[ixs]
    # ssv_ids = ssv_ids[np.argsort(sizes)[::-1]]

    ssv_ids = []
    ssv_params = [dict(ssv_id=ssv.id, sv_ids=ssv.sv_ids, sv_graph=ssv.sv_graph_uint, version='tmp',
                       **ssd_kwargs) for ssv in ssd.get_super_segmentation_object(ssv_ids)]

    nodes = [10861339, 10705496, 16300957, 16393831, 16300965, 16394199, 16394279, 16300952, 10864571,
             10961702, 10868575]
    g = nx.generators.classic.complete_graph(nodes)
    ssv_params.append(dict(ssv_id=nodes[0], sv_ids=nodes, sv_graph=g, version='tmp', **ssd_kwargs))
    # nodes = [10864918, 10958215, 10958277, 10958221, 10864571, 10951524, 10875873]
    # g = nx.generators.classic.complete_graph(nodes)
    # ssv_params.append(dict(ssv_id=nodes[0], sv_ids=nodes, sv_graph=g, version='tmp', **ssd_kwargs))

    res = predict_glia_ssv(ssv_params, mpath=mpath, nloader=4, npredictor=2,
                           postproc_kwargs=dict(pred_key='glia_probas_test'))

    for ssv in map(lambda x: SuperSegmentationObject(**x), ssv_params):
        kname = f'/wholebrain/scratch/pschuber/tmp/gliapred_{ssv.id}.k.zip'
        ssv.gliapred2mesh(kname, pred_key_appendix='_test')
        ssv.save_skeleton_to_kzip(kname)
        locs = np.concatenate(ssv.sample_locations())

        print(np.array([sv.glia_pred(0.161, '_test') for sv in ssv.svs]))
        print(np.array([len(sv.sample_locations()) for sv in ssv.svs]))

        print(np.array([sv.glia_proba('_test') for sv in ssv.svs]))
        print([np.mean(so.attr_dict['glia_probas_test'], axis=0) for so in ssv.svs])
        raise()
