from syconn.handler import basics, training
from syconn.mp.batchjob_utils import batchjob_script

if __name__ == '__main__':
    params = []
    gen_script = '/wholebrain/u/amancu/Projects/SyConn/scripts/amancu/generate_mergeError_samples.py'
    sets = ['training','test']
    one = ['100','500','5000']
    two = ['1000','2000']

    for x in sets:
        params.append(
            [gen_script,
             dict(set=x, nproc=15, r=one)])
    params.append([gen_script, dict(set='test', nproc=15, r=two)])
    params = list(basics.chunkify_successive(params, 1))
    batchjob_script(params, 'launch_trainer', n_cores=15,
                    additional_flags='--time=7-0 --qos=720h --mem=125000 --gres=gpu:0',
                    disable_batchjob=False,
                    batchjob_folder=f'/wholebrain/scratch/amancu/batchjobs/mergeError/sample_gen/',
                    remove_jobfolder=False, overwrite=True)
