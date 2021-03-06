from syconn.handler import basics, training
from syconn.mp.batchjob_utils import batchjob_script


if __name__ == '__main__':
    cnn_script = '/wholebrain/u/pschuber/devel/SyConn/syconn/cnn/cnn_celltype_ptcnv_j0251.py'
    params = []
    # for npoints in [5000, 25000, 50000, 75000, 100000]:
    for npoints, ctx in zip([25000, 10000], [2500, 10000], [50000, 10000]):
        scale = int(ctx / 10)
        for run in range(3):
            base_dir = f'/wholebrain/scratch/pschuber/e3_trainings_convpoint_celltypes/' \
                       f'celltype_pts{npoints}_ctx{ctx}/'
            for cval in range(10):
                save_root = f'{base_dir}/celltype_eval{cval}/'
                params.append([cnn_script, dict(sr=save_root, sp=npoints, cval=cval, seed=run, ctx=ctx,
                                                scale_norm=scale, use_bias=True)])
        params = list(basics.chunkify_successive(params, 5))
        batchjob_script(params, 'launch_trainer', n_cores=20, additional_flags='--time=7-0 --qos=720h --gres=gpu:2',
                        disable_batchjob=False, batchjob_folder='/wholebrain/scratch/pschuber/batchjobs/launch_trainer/',
                        remove_jobfolder=True, overwrite=True)
