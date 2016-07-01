__author__ = 'pschuber'
import sys
from syconn.brainqueries import annotate_annos
import cPickle as pickle


path_storage_file = sys.argv[1]
path_out_file = sys.argv[2]

with open(path_storage_file) as f:
    nml_list = pickle.load(f)
    output_dir = pickle.load(f)

overwrite = True
annotate_annos(nml_list, overwrite=overwrite, output_dir=output_dir)