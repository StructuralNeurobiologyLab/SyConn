import numpy as np
import morphx.processing.clouds as clouds
from elektronn3.models.convpoint import SegBig
from syconn.handler.prediction_pts import predict_cpmt_ssv, pts_loader_cpmt, pts_pred_cmpt
from syconn.reps.super_segmentation import SuperSegmentationDataset


def test_cmpt_pipeline():
    wd = "/wholebrain/songbird/j0126/areaxfs_v6/"
    ssd_kwargs = dict(working_dir=wd)
    ssd = SuperSegmentationDataset(**ssd_kwargs)
    ssv_ids = np.array([2734465, 2854913, 8003584, 8339462, 10919937, 26169344, 26501121])
    ssv_params = [dict(ssv_id=ssv.id, sv_ids=ssv.sv_ids, sv_graph=ssv.sv_graph_uint,
                       **ssd_kwargs) for ssv in ssd.get_super_segmentation_object(ssv_ids)]
    m = SegBig(4, 3)
    transform = [clouds.Normalization(5000), clouds.Center()]
    transform = clouds.Compose(transform)

    # pipeline simulation
    res = pts_loader_cpmt([ssv_params[0]], 8, 2000, 5000, use_myelin=False, transform=transform, base_node_dst=10000)


if __name__ == '__main__':
    import ipdb
    ipdb.set_trace()
    test_cmpt_pipeline()
