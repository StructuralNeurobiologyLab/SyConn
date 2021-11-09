import os
from syconn.mp.batchjob_utils import batchjob_script

if __name__ == '__main__':
    params = []

    base = os.path.expanduser('~/working_dir/paper/temp/')
    paths = os.listdir(base)

    for path in paths:
        params.append([(base, path)])

    batchjob_script(params, 'launch_syn_eval', n_cores=10,
                    additional_flags='--mem=125000 --gres=gpu:1',
                    disable_batchjob=False, max_iterations=0,
                    batchjob_folder='/wholebrain/u/jklimesch/working_dir/batchjobs/syn_eval/',
                    remove_jobfolder=False, overwrite=True, exclude_nodes=[])
