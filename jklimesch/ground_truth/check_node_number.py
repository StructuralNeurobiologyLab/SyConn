import os
import re
import glob
import numpy as np
import pickle as pkl
from morphx.processing import ensembles

annotations = os.path.expanduser('/wholebrain/scratch/pschuber/cmn_paper/compartment_gt_ncomm_CMN_2019_remapped/axoness_comparison/mapped_pkls_correctsupport/')
ces = os.path.expanduser('~/working_dir/gt/cmn/ads/test/voxeled/')

afiles = glob.glob(annotations + '*.pkl')

nodes = 0
n_dict = {0: 0, 1: 0, 2: 0}
for afile in afiles:
    with open(afile, 'rb') as f:
        data = pkl.load(f)
    sso_id = int(re.findall(r"/(\d+).", afile)[0])
    ce = ensembles.ensemble_from_pkl(ces + 'sso_' + str(sso_id) + '.pkl')
    uniques = np.unique(ce.node_labels, return_counts=True)
    for ix, label in enumerate(uniques[0]):
        n_dict[label] += uniques[1][ix]
    assert len(ce.nodes) == len(data['nodes'])
    nodes += len(data['compartment_label'])

print(n_dict)
print(nodes)