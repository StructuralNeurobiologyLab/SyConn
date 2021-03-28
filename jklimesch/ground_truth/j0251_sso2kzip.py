import os
import pandas
import numpy as np
from syconn.reps.super_segmentation import SuperSegmentationDataset

csv_p = "/wholebrain/songbird/j0251/groundtruth/celltypes/j0251_celltype_gt_v2.csv"
csv_p = os.path.expanduser(csv_p)
df = pandas.io.parsers.read_csv(csv_p, header=None, names=['ID', 'type']).values
ssv_ids = df[:, 0].astype(np.uint)
if len(np.unique(ssv_ids)) != len(ssv_ids):
    ixs, cnt = np.unique(ssv_ids, return_counts=True)
    raise ValueError(f'Multi-usage of IDs! {ixs[cnt > 1]}')
str_labels = df[:, 1]

ssd = SuperSegmentationDataset(working_dir='/ssdscratch/pschuber/songbird/j0251/rag_flat_Jan2019_v2/')
for ct in np.unique(str_labels):
    ids = ssv_ids[str_labels == ct]
    for i in range(8):
        sso = ssd.get_super_segmentation_object(ids[i])
        sso.load_attr_dict()
        sso.load_skeleton()
        sso.meshes2kzip(f'/wholebrain/u/jklimesch/thesis/gt/j0251/{ct}/{sso.id}.k.zip', synssv_instead_sj=True)
        sso.save_skeleton_to_kzip(f'/wholebrain/u/jklimesch/thesis/gt/j0251/{ct}/{sso.id}.k.zip')
