from processing import initialization, objectextraction as oe
from knossos_utils import knossosdataset
from knossos_utils import chunky

import numpy as np
import os
import sys

# home_path = sys.argv[1]
home_path = "/mnt/axon/home/sdorkenw/SyConnDenseCube"

assert os.path.exists(home_path + "/models/")
assert os.path.exists(home_path + "/knossosdatasets/raw/")

# define paths, create folders, initialize datasets
kd_raw = knossosdataset.KnossosDataset()
kd_raw.initialize_from_knossos_path(home_path + "/knossosdatasets/raw/")
cset = initialization.initialize_cset(kd_raw, home_path, [500, 500, 250])

# CNN - Synapses

# CNN - Barrier


kd_bar = knossosdataset.KnossosDataset()
kd_bar.initialize_from_knossos_path(home_path + "/knossosdatasets/rrbarrier/")

# Object Extraction
# TODO: use specific thresholds only for example run
# oe.from_probabilities_to_objects(cset, "ARGUS",
#                                   ["sj", "vc", "mi"],
#                                   thresholds=[100, 100, 100],
#                                   debug=False,
#                                   membrane_kd_path=kd_bar.knossos_path)

oe.from_probabilities_to_objects_parameter_sweeping(cset,
                                                    "ARGUS",
                                                    ["sj", "vc", "mi"],
                                                    20,
                                                    membrane_kd_path=None,
                                                    hdf5_name_membrane=kd_bar.knossos_path,
                                                    use_qsub=False)

