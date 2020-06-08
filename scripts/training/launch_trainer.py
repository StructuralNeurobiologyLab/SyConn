from syconn.handler import basics, training
from syconn.mp.batchjob_utils import batchjob_script


if __name__ == '__main__':
    cnn_script = '/wholebrain/u/pschuber/devel/SyConn/syconn/cnn/cnn_celltype_ptcnv.py'
    base_dir = '/wholebrain/scratch/pschuber/e3_trainings_convpoint/'
    params = []
    # for npoints in [5000, 25000, 50000, 75000, 100000]:
    for npoints in [25000]:
        for run in range(3):
            for cval in range(10):
                save_root = f'{base_dir}/celltype_eval{run}_sp{npoints//1000}k/'
                params.append([cnn_script, dict(sr=save_root, sp=npoints, cval=cval, seed=run, ctx=5000,
                                                scale_norm=500, use_bias=True)])
    params = list(basics.chunkify_successive(params, 5))
    batchjob_script(params, 'launch_trainer', n_cores=20, additional_flags='--gres=gpu:2 --time=7-0 --qos=720h',
                    disable_batchjob=False, batchjob_folder='/wholebrain/scratch/pschuber/batchjobs/launch_trainer/',
                    remove_jobfolder=True, overwrite=True)
