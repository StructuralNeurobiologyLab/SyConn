# SyConn-dev
# Copyright (c) 2016 Philipp J. Schubert
# All rights reserved

import numpy as np
from syconn import global_params
from syconn.reps.super_segmentation import SuperSegmentationDataset
from syconn.handler.prediction import int2str_converter

# set SyConn's working directory to the following path
global_params.wd = '/ssdscratch/pschuber/songbird/j0251/rag_flat_Jan2019_v3/'

# instantiate neuron collection
ssd_neuron = SuperSegmentationDataset()

neuron_example = ssd_neuron.get_super_segmentation_object(163615279)

celltype = neuron_example.celltype()
# use int2str_converter to get actual names of cell types
celltype_name_as_str = int2str_converter(celltype, 'ctgt_j0251_v2')
print(celltype_name_as_str, celltype)

# get total path length of neuron
path_length = neuron_example.total_edge_length()

# get neuron mitos
mitochondria = neuron_example.mis

# get neuron synapse objects
mitochondrien = neuron_example.syn_ssv

# get neuron vesicles
vcs = neuron_example.vcs

# neuron skeleton
neuron_example.load_skeleton()
skel = neuron_example.skeleton
print(skel.keys())

# write neuron mesh to file
fname = f'/wholebrain/scratch/cschick/neuron_{neuron_example.id}.k.zip'
neuron_example.mesh2kzip(fname)
# write skeleton with predicted properties to file
neuron_example.save_skeleton_to_kzip(fname, additional_keys=['axoness_avg10000', 'spiness', 'myelin_avg10000'])


# write neuron mesh and meshes of its cell organelles to file
fname_with_organelles = f'/wholebrain/scratch/cschick/neuron_{neuron_example.id}_with_cellorganelles.k.zip'
neuron_example.meshes2kzip(fname_with_organelles)
neuron_example.save_skeleton_to_kzip(
    fname_with_organelles, additional_keys=['axoness_avg10000', 'spiness', 'myelin_avg10000',
                                            'axoness_avg10000_comp_maj'])

# find any neuron with myelin
for neuron in ssd_neuron.ssvs:
    neuron.load_skeleton()
    myelin_prediction = neuron.skeleton['myelin_avg10000']
    myelin_present = np.any(myelin_prediction)
    celltype = neuron.celltype()
    if myelin_present:
        celltype_name_as_str = int2str_converter(celltype, 'ctgt_j0251_v2')
        print(f'Neuron with ID={neuron.id} and of type={celltype_name_as_str} contains at least one '
              f'skeleton node with myelin.')
