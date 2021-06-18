from syconn.handler import basics, training
from syconn.mp.batchjob_utils import batchjob_script

if __name__ == '__main__':
    nfold = 10
    params = []
    cnn_script = '/wholebrain/u/pschuber/devel/SyConn-dev/syconn/cnn/cnn_celltype_randla_j0251.py'

    for npoints, ctx in ([50000, 20000], [25000, 20000], ):
        scale = int(ctx / 10)
        for run in range(3):
            base_dir = f'/wholebrain/scratch/pschuber/e3_trainings_randla_celltypes_j0251/' \
                       f'celltype_pts{npoints}_ctx{ctx}/'
            for cval in range(nfold):
                save_root = f'{base_dir}/celltype_CV{cval}/'
                params.append([cnn_script, dict(sr=save_root, sp=npoints, cval=cval, seed=run, ctx=ctx, scale_norm=scale,
                                                use_bias=True)])
    params = list(basics.chunkify_successive(params, 2))
    batchjob_script(params, 'launch_trainer', n_cores=20, additional_flags='--time=7-0 --qos=720h --gres=gpu:2',
                    disable_batchjob=True,
                    batchjob_folder=f'/wholebrain/scratch/pschuber/batchjobs/launch_trainer_celltypes_j0251_randla',
                    remove_jobfolder=False, overwrite=True, exclude_nodes=['wb02', 'wb03', 'wb04', 'wb05', 'wb06', 'wb07'])

    # for npoints, ctx in ([25000, 15000], ):
    #     scale = int(ctx / 10)
    #     save_root = f'/wholebrain/scratch/pschuber/e3_trainings_randla_celltypes_j0251/' \
    #                 f'celltype_pts{npoints}_ctx{ctx}_allGT/'
    #     params.append([cnn_script, dict(sr=save_root, sp=npoints, cval=-1, seed=0, ctx=ctx,
    #                                     scale_norm=scale)])
    # params = list(basics.chunkify_successive(params, 1))
    # batchjob_script(params, 'launch_trainer', n_cores=10, additional_flags='--time=7-0 --qos=720h --gres=gpu:1',
    #                 disable_batchjob=False,
    #                 batchjob_folder=f'/wholebrain/scratch/pschuber/batchjobs'
    #                                 f'/launch_trainer_celltypes_j0251_allGT_randla/',
    #                 remove_jobfolder=False, overwrite=True, exclude_nodes=[])
