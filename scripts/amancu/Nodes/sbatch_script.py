from syconn.handler import basics, training
from syconn.mp.batchjob_utils import batchjob_script

if __name__ == '__main__':
    params = []
    cnn_script = '/wholebrain/u/amancu/Projects/SyConn/scripts/amancu/Nodes/train_merge_error.py'
    radii = [3000]
    # cs_merge_radii = [1000]
    ctx = 20000
    npoints = 10000
    scale = 5000
    model = 'SegSmall'
    arch='archLrg'
    optimizers = [('Adam', 'StepLR')]
    # optimizers = ['SGD']
    conv = 'ConvPoint'
    save_root = f'/wholebrain/scratch/amancu/mergeError/Nodes/Trainings/'
    # resume_root = f'/wholebrain/scratch/amancu/mergeError/trainings/lcp/see/lcp_r2000_ConvPoint_SearchQuantized_archLrg_run3_SGD_CyclicLR_weights1,2_FocalLoss/state_dict.pth'

    for radius in radii:
        for (optimizer, scheduler) in optimizers:
            params.append(
                [cnn_script,
                 dict(sr=save_root, r=radius, model=model, arch=arch, opt=optimizer, lr=scheduler, conv=conv, sp=npoints, ctx=ctx,
                      scale_norm=scale, use_bias=True)])
    params = list(basics.chunkify_successive(params, 1))
    batchjob_script(params, 'launch_trainer', n_cores=5,
                    additional_flags='--time=7-0 --qos=720h --mem=125000 --gres=gpu:1',
                    disable_batchjob=False,
                    batchjob_folder=f'/wholebrain/scratch/amancu/batchjobs/mergeError/{model}/',
                    remove_jobfolder=False, overwrite=True)
