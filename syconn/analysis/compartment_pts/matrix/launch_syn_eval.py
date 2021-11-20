import os
from syconn.mp.batchjob_utils import batchjob_script

if __name__ == '__main__':
    params = []
    do_pred_key = 'do_cmn_large'
    base = os.path.expanduser('/wholebrain/scratch/pschuber/syconn_v2_paper/'
                              'supplementals/compartment_pts/dnh_matrix_u'
                              'pdate_cmn_ads/models/')
    paths = os.listdir(base)

    for path in paths:
        params.append([(base, path, do_pred_key)])

    batchjob_script(params, 'launch_syn_eval', n_cores=1,
                    additional_flags='',
                    disable_batchjob=True, max_iterations=0,
                    batchjob_folder='/wholebrain/scratch/pschuber/batchjobs/syn_eval/',
                    remove_jobfolder=False, overwrite=True, exclude_nodes=[])
