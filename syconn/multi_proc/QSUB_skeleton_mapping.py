"""Executable file for QSUB job to calculate cell hull and map cell objects to
 tracings

QSUB wrapper for cell object mapping using enrich_tracings from brainqueries.
"""
import sys
from syconn.brainqueries import enrich_tracings
import cPickle as pickle
__author__ = 'pschuber'

if __name__ == '__main__':

    path_storage_file = sys.argv[1]
    path_out_file = sys.argv[2]

    with open(path_storage_file) as f:
        nml_list = pickle.load(f)
        overwrite = pickle.load(f)

    enrich_tracings(nml_list, overwrite=overwrite)