from syconn.handler import basics, training
from syconn.mp.batchjob_utils import batchjob_script


if __name__ == '__main__':
    cnn_script = '/wholebrain/u/pschuber/devel/SyConn-dev/syconn/cnn/cnn_celltype_ptcnv.py'
    base_dir = '/wholebrain/scratch/pschuber/e3_trainings_convpoint/'
    params = []
    for run in range(3):
        for cval in range(10):
            for npoints in [10000, 20000, 40000, 60000, 80000]:
                save_root = f'{base_dir}/celltype_eval{run}_sp{npoints//1000}k/'
                params.append([cnn_script, dict(sr=save_root, sp=npoints, cval=cval)])
    params = list(basics.chunkify_successive(params, 12))
    print(params[0])
    batchjob_script(params, 'launch_trainer', n_cores=10, additional_flags='--gres=gpu:1', disable_batchjob=False,
                    batchjob_folder='/wholebrain/scratch/pschuber/batchjobs/launch_trainer/')

