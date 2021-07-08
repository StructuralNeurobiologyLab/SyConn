from syconn.handler import basics, training
from syconn.mp.batchjob_utils import batchjob_script

if __name__ == '__main__':
    params = []
    cnn_script = '/wholebrain/u/amancu/Projects/SyConn/scripts/amancu/train_merge_error.py'
    radii = [1000, 2000]
    # radii = [1000]
    ctx = 20000
    npoints = 15000
    scale = 5000
    model = 'lcp'
    # optimizers = ['Adam', 'SGD']
    # optimizers = ['Adam']
    optimizers = ['SGD']
    conv = 'ConvPoint'
    # conv = 'FKAConv'
    save_root = f'/wholebrain/scratch/amancu/mergeError/trainings/{model}/'

    for radius in radii:
        for optimizer in optimizers:
            params.append(
                [cnn_script, dict(sr=save_root, r=radius, model=model, opt=optimizer, conv=conv, sp=npoints, ctx=ctx,
                                  scale_norm=scale, use_bias=True)])
    params = list(basics.chunkify_successive(params, 1))
    batchjob_script(params, 'launch_trainer', n_cores=5,
                    additional_flags='--time=7-0 --qos=720h --mem=125000 --gres=gpu:1',
                    disable_batchjob=False,
                    # batchjob_folder=f'/wholebrain/scratch/amancu/batchjobs/mergeError/{model}/{conv}/',
                    batchjob_folder=f'/wholebrain/scratch/amancu/batchjobs/mergeError/{model}/CyclicLR/',
                    # batchjob_folder=f'/wholebrain/scratch/amancu/batchjobs/mergeError/{model}/ExponentialLR/',
                    remove_jobfolder=False, overwrite=True)
