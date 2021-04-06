from syconn.handler import basics, training
from syconn.mp.batchjob_utils import batchjob_script


if __name__ == '__main__':
    nfold = 10
    params = []
    cnn_script = '/wholebrain/u/pschuber/devel/SyConn/syconn/cnn/cnn_celltype_cmn_j0251.py'
    for run in range(3):
        base_dir = f'/wholebrain/scratch/pschuber/e3_trainings_cmn_celltypes_j0251/'
        for cval in range(nfold):
            params.append([cnn_script, dict(sr=f'{base_dir}/celltype_CV{cval}/', cval=cval, seed=run)])
    params = list(basics.chunkify_successive(params, 1))
    batchjob_script(params, 'launch_trainer', n_cores=20, additional_flags='--time=7-0 --qos=720h --gres=gpu:2 --mem=0',
                    disable_batchjob=False,
                    batchjob_folder=f'/wholebrain/scratch/pschuber/batchjobs/launch_trainer_celltypes_cmn_j0251/',
                    remove_jobfolder=False, overwrite=True,
                    exclude_nodes=['wb06', 'wb07'])
