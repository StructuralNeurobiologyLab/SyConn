import sys
from syconn.brainqueries import remap_skeletons
from syconn.processing.learning_rfc import start_multiprocess
import cPickle as pickle
__author__ = 'pschuber'


def multi_helper_remap(para):
    para = {'mapped_skel_paths': [para[0]], 'output_dir': para[1],
            'recalc_prop_only': para[2], 'method': para[3],
            'context_range': para[4]}
    remap_skeletons(**para)

path_storage_file = sys.argv[1]
path_out_file = sys.argv[2]

with open(path_storage_file) as f:
    nml_list = pickle.load(f)
    output_dir = pickle.load(f)
    recalc_prop_only = pickle.load(f)
    method = pickle.load(f)
    dist = pickle.load(f)

params = [[nml_list[i], output_dir, recalc_prop_only, method, dist] for i in
          range(len(nml_list))]

start_multiprocess(multi_helper_remap, params, debug=True, nb_cpus=10)


