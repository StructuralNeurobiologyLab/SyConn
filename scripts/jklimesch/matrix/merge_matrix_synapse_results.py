import os
import re
import pickle as pkl

base = os.path.expanduser('~/working_dir/paper/dnh_matrix/')
save_path = os.path.expanduser('~/working_dir/paper/')
target_path = os.path.expanduser('~/working_dir/paper/')
dirs = [d for d in os.listdir(base) if os.path.isdir(base + d)]

results = {}
for d in dirs:
    files = os.listdir(base + d)
    report_path = ''
    epoch = None
    for file in files:
        if 'syn_e' in file:
            report_path = os.path.expanduser(base + d + '/' + file + '/')
            epoch = int(re.findall(r"_e(\d+).", file)[0])
            break
    with open(report_path + 'log/report.pkl', 'rb') as f:
        report = pkl.load(f)

    errors = os.listdir(report_path + 'examples/')

    # filter real experiment names
    d = d[11:]

    results[d] = dict(report=report, epoch=epoch, errors=errors)

    with open(target_path + 'matrix_data.pkl', 'wb') as f:
        pkl.dump(results, f)




