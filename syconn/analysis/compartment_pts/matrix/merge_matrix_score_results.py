import os
import pickle as pkl

base_dir = '/wholebrain/scratch/pschuber/syconn_v2_paper/supplementals/' \
           'compartment_pts/dnh_matrix_update_cmn_ads/'
base = os.path.expanduser(f'{base_dir}/models/')
save_path = os.path.expanduser(base_dir)
dirs = [d for d in os.listdir(base) if os.path.isdir(base + d)]
pred_key = 'do_cmn_large'

f1_scores = []
names = []
for d in dirs:
    files = os.listdir(base + d)
    report_path = ''
    for file in files:
        if 'syn_e_final' in file:
            report_path = os.path.expanduser(base + d + '/' + file + '/')
            break
    with open(f'{report_path}log/report_{pred_key}.pkl', 'rb') as f:
        report = pkl.load(f)
    f1_scores.append(report['weighted avg']['f1-score'])

    # filter real experiment names
    d = d[11:]
    if d[-2:] in ['_1', '_2', '_3']:
        d = d[:-2]
    names.append(d)

unique = {}
for ix, s in enumerate(names):
    if s in unique:
        unique[s].append(f1_scores[ix])
    else:
        unique[s] = [f1_scores[ix]]

with open(save_path + f'dnh_matrix_syn_e_final_{pred_key}.pkl', 'wb') as f:
    pkl.dump(unique, f)
