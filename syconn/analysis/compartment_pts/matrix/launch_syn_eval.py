import os
from syconn.mp.batchjob_utils import batchjob_script

if __name__ == '__main__':
    params = []

    base = os.path.expanduser('/wholebrain/scratch/jklimesch/paper/dnh_matrix_update_cmn_ads/models/')
    paths = os.listdir(base)

    for path in paths:
        params.append([(base, path)])

    batchjob_script(params, 'launch_syn_eval', n_cores=1,
                    additional_flags='',
                    disable_batchjob=False, max_iterations=0,
                    batchjob_folder='/wholebrain/scratch/pschuber/batchjobs/syn_eval/',
                    remove_jobfolder=False, overwrite=True, exclude_nodes=[])
