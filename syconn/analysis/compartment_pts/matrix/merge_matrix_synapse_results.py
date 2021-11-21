import os
import re
import pickle as pkl

base_dir = '/wholebrain/scratch/pschuber/syconn_v2_paper/supplementals/' \
           'compartment_pts/dnh_matrix_update_cmn_ads/'
base = os.path.expanduser(f'{base_dir}/models/')
save_path = os.path.expanduser(base_dir)
target_path = os.path.expanduser(base_dir)
dirs = [d for d in os.listdir(base) if os.path.isdir(base + d)]
pred_key = 'do_cmn_large'

results = {}
for d in dirs:
    files = os.listdir(base + d)
    report_path = ''
    epoch = None
    for file in files:
        if 'syn_e_final' in file:
            report_path = os.path.expanduser(base + d + '/' + file + '/')
            # epoch = int(re.findall(r"_e(\d+).", file)[0])
            epoch = 'final'
            break
    with open(f'{report_path}log/report_{pred_key}.pkl', 'rb') as f:
        report = pkl.load(f)

    errors = os.listdir(report_path + 'examples/')

    # filter real experiment names
    d = d[11:]

    results[d] = dict(report=report, epoch=epoch, errors=errors)

    with open(f'{target_path}matrix_data_e_final_{pred_key}.pkl', 'wb') as f:
        pkl.dump(results, f)




