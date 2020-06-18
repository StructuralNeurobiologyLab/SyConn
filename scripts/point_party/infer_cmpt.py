from syconn.handler.prediction_pts import predict_cmpt_ssd, get_cpmt_model_pts

if __name__ == '__main__':
    wd = "/wholebrain/songbird/j0126/areaxfs_v6/"
    ssd_kwargs = dict(working_dir=wd)
    ssv_ids = [491527, 12179464]
    mpath = '~/thesis/current_work/paper/test_models/'
    predict_cmpt_ssd(ssd_kwargs=ssd_kwargs, ssv_ids=ssv_ids, mpath=mpath)
