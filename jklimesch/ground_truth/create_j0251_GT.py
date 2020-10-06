import os
import pandas
import numpy as np
from syconn.handler.prediction import str2

csv_p = "/wholebrain/songbird/j0251/groundtruth/j0251_celltype_gt_v2.csv"
csv_p = os.path.expanduser(csv_p)
df = pandas.io.parsers.read_csv(csv_p, header=None, names=['ID', 'type']).values
ssv_ids = df[:, 0].astype(np.uint)
if len(np.unique(ssv_ids)) != len(ssv_ids):
    ixs, cnt = np.unique(ssv_ids, return_counts=True)
    raise ValueError(f'Multi-usage of IDs! {ixs[cnt > 1]}')
str_labels = df[:, 1]
ssv_labels = np.array([str2int_converter(el, gt_type='ctgt_j0251_v2') for el in str_labels], dtype=np.uint16)