__author__ = 'pschuber'
import sys

sys.path.append("/home/pschuber/skeleton-analysis/DatasetUtilities/")
sys.path.append("/home/pschuber/skeleton-analysis/Sven/functional/")
sys.path.append("/home/pschuber/skeleton-analysis/ChunkUtilities/")
sys.path.append("/home/pschuber/skeleton-analysis/NewSkeleton/")
sys.path.append("/home/pschuber/skeleton-analysis/EMUtilities/")
sys.path.append("/home/pschuber/skeleton-analysis/GeneralUtilities/")
sys.path.append("/home/pschuber/skeleton-analysis/Sven/CNN_v45/")
sys.path.append("/home/pschuber/skeleton-analysis/")
sys.path.append("/home/pschuber/skeleton-analysis/Philipp/annotator/")
sys.path.append("/home/pschuber/")
sys.path.append("/home/pschuber/knossos_python_tools/")

from Philipp.annotator.SynapseMapper import prepare_syns_btw_annos
import cPickle as pickle

path_storage_file = sys.argv[1]
path_out_file = sys.argv[2]


with open(path_storage_file) as f:
    pairwise_paths = pickle.load(f)
    dest_path = pickle.load(f)
    max_hull_dist = pickle.load(f)

prepare_syns_btw_annos(pairwise_paths, dest_path, max_hull_dist=max_hull_dist)