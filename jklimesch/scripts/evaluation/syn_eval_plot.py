import os
import pickle as pkl

base = os.path.expanduser('~/working_dir/paper/dnh_model_comparison/')
dirs = [d for d in os.listdir(base) if os.path.isdir(base + d)]

scores = {}
for item in ['5', '5_border']:
    f1_scores = []
    names = []
    for d in dirs:
        if d == '2020_11_16_8000_8192_cp_cp_q_nn_3' or d == '2020_11_08_2000_2048_cp_cp_q':
            continue
        try:
            with open(base + d + f'/syn_eval_dnh_red' + item + '/report.pkl', 'rb') as f:
                report = pkl.load(f)
            f1_scores.append(report['weighted avg']['f1-score'])
        except:
            f1_scores.append(0)
            continue

        # filter real experiment names
        d = d[11:]
        if d[-2:] == '_2' or d[-2:] == '_3' or d[-2:] == '_4':
            d = d[:-2]
        names.append(d)

    unique = {}
    for ix, s in enumerate(names):
        if s in unique:
            unique[s].append(f1_scores[ix])
        else:
            unique[s] = [f1_scores[ix]]

    scores[item] = unique


with open(base + f'syn_scores_border.pkl', 'wb') as f:
    pkl.dump(scores, f)
