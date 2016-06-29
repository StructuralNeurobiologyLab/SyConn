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
sys.path.append("/home/pschuber/")
sys.path.append("/home/pschuber/skeleton-analysis/LukasB/")
sys.path.append("/home/pschuber/knossos_python_tools/knossos_utils/")
sys.path.append("/home/pschuber/knossos_python_tools/")

from Philipp.annotator.map_apps import annotate_annos
import cPickle as pickle

path_storage_file = sys.argv[1]
path_out_file = sys.argv[2]

with open(path_storage_file) as f:
    nml_list = pickle.load(f)
    output_dir = pickle.load(f)

overwrite=True
annotate_annos(nml_list, overwrite=overwrite, output_dir=output_dir)