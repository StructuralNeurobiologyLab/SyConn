from syconn.handler import basics, training
from syconn.mp.batchjob_utils import batchjob_script

if __name__ == '__main__':
    nfold = 10
    params = []
    cnn_script = '/wholebrain/u/pschuber/devel/SyConn/syconn/cnn/cnn_celltype_ptcnv_j0251.py'

    # for npoints, ctx in ([50000, 20000], [25000, 20000], [25000, 4000], [5000, 20000], [75000, 20000],):
    #     scale = int(ctx / 10)
    #     for run in range(3):
    #         if npoints == 50000 and ctx == 20000:
    #             for use_syntype, cellshape_only in zip([1, 0, 0], [0, 0, 1]):
    #                 base_dir = f'/wholebrain/scratch/pschuber/e3_trainings_convpoint_celltypes_j0251_rerunFeb21/' \
    #                            f'celltype_pts{npoints}_ctx{ctx}'
    #                 if cellshape_only:
    #                     base_dir += '_cellshape_only/'
    #                 elif not use_syntype:
    #                     base_dir += '_no_syntype/'
    #                 for cval in range(nfold):
    #                     save_root = f'{base_dir}/celltype_CV{cval}/'
    #                     params.append(
    #                         [cnn_script, dict(sr=save_root, sp=npoints, cval=cval, seed=run, ctx=ctx, scale_norm=scale,
    #                                           use_bias=True, cellshape_only=bool(cellshape_only), use_syntype=bool(use_syntype))])
    #         else:
    #             base_dir = f'/wholebrain/scratch/pschuber/e3_trainings_convpoint_celltypes_j0251_rerunFeb21/' \
    #                        f'celltype_pts{npoints}_ctx{ctx}'
    #             for cval in range(nfold):
    #                 save_root = f'{base_dir}/celltype_CV{cval}/'
    #                 params.append([cnn_script, dict(sr=save_root, sp=npoints, cval=cval, seed=run, ctx=ctx, scale_norm=scale,
    #                                                 use_bias=True)])
    # params = list(basics.chunkify_successive(params, 2))
    # batchjob_script(params, 'launch_trainer', n_cores=20, additional_flags='--time=7-0 --qos=720h --gres=gpu:2',
    #                 disable_batchjob=False,
    #                 batchjob_folder=f'/wholebrain/scratch/pschuber/batchjobs/launch_trainer_celltypes_j0251_ptconv',
    #                 remove_jobfolder=False, overwrite=True, exclude_nodes=['wb02', 'wb03', 'wb04', 'wb05', 'wb06', 'wb07'])

    for npoints, ctx in ([25000, 20000], [50000, 20000]):
        scale = int(ctx / 10)
        for run in range(3):
            base_dir = f'/wholebrain/scratch/pschuber/e3_trainings_convpoint_celltypes_j0251_rerunFeb21/myelin_ablation/' \
                       f'celltype_pts{npoints}_ctx{ctx}'
            for cval in range(nfold):
                save_root = f'{base_dir}/celltype_CV{cval}/'
                params.append([cnn_script, dict(sr=save_root, sp=npoints, cval=cval, seed=run, ctx=ctx, scale_norm=scale,
                               use_bias=True)])
    params = list(basics.chunkify_successive(params, 5))
    batchjob_script(params, 'launch_trainer', n_cores=20, additional_flags='--time=7-0 --qos=720h --gres=gpu:2',
                    disable_batchjob=False,
                    batchjob_folder=f'/wholebrain/scratch/pschuber/batchjobs/launch_trainer_c'
                                    f'elltypes_j0251_myelin_ablation/',
                    remove_jobfolder=False, overwrite=True, exclude_nodes=['wb02', 'wb03', 'wb04', 'wb05', 'wb06'])
