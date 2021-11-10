import os
import pickle as pkl

base = os.path.expanduser('/wholebrain/scratch/jklimesch/paper/dnh_matrix_update_cmn_ads/models/')
save_path = os.path.expanduser('/wholebrain/scratch/jklimesch/paper/dnh_matrix_update_cmn_ads/')
dirs = [d for d in os.listdir(base) if os.path.isdir(base + d)]

f1_scores = []
names = []
for d in dirs:
    files = os.listdir(base + d)
    report_path = ''
    for file in files:
        if 'syn_e' in file:
            report_path = os.path.expanduser(base + d + '/' + file + '/')
            break
    with open(report_path + 'log/report.pkl', 'rb') as f:
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

with open(save_path + f'dnh_matrix_syn.pkl', 'wb') as f:
    pkl.dump(unique, f)
