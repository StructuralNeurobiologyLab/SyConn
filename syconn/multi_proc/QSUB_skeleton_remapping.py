import sys
__author__ = 'pschuber'
sys.path.append("/home/pschuber/skeleton-analysis/DatasetUtilities/")
sys.path.append("/home/pschuber/skeleton-analysis/Sven/functional/")
sys.path.append("/home/pschuber/skeleton-analysis/ChunkUtilities/")
sys.path.append("/home/pschuber/skeleton-analysis/NewSkeleton/")
sys.path.append("/home/pschuber/skeleton-analysis/EMUtilities/")
sys.path.append("/home/pschuber/skeleton-analysis/GeneralUtilities/")
sys.path.append("/home/pschuber/skeleton-analysis/Sven/CNN_v45/")
sys.path.append("/home/pschuber/skeleton-analysis/")
sys.path.append("/home/pschuber/")
sys.path.append("/home/pschuber/skeleton-analysis/LukasB/")
sys.path.append("/home/pschuber/knossos_python_tools/knossos_utils/")
sys.path.append("/home/pschuber/knossos_python_tools/")
from Philipp.annotator.map_apps import remap_skeletons
from Philipp.predictor.learning_ut import start_multiprocess
import cPickle as pickle


def multi_helper_remap(para):
    para = {'mapped_skel_paths': [para[0]], 'output_dir': para[1],
              'az_kd_set': para[2], 'filter_size': [2786, 1594, para[3]],
              'az_min_votes': para[4], 'recalc_prop_only': para[5], 'method':
            para[6], 'context_range': para[7]}
    remap_skeletons(**para)

path_storage_file = sys.argv[1]
path_out_file = sys.argv[2]

with open(path_storage_file) as f:
    nml_list = pickle.load(f)
    output_dir = pickle.load(f)
    az_kd_set = pickle.load(f)
    az_size_threshold = pickle.load(f)
    az_min_votes = pickle.load(f)
    recalc_prop_only = pickle.load(f)
    method = pickle.load(f)
    dist = pickle.load(f)

params = [[nml_list[i], output_dir, az_kd_set, az_size_threshold,
          az_min_votes, recalc_prop_only, method, dist] for i in range(len(nml_list))]

# remap_skeletons(mapped_skel_paths=nml_list, output_dir=output_dir,
#                 az_kd_set=az_kd_set, filter_size=az_size_threshold,
#                 az_min_votes=az_min_votes)
start_multiprocess(multi_helper_remap, params, debug=True, nb_cpus=10)


