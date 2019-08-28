from syconn import global_params
global_params.wd = '/wholebrain/songbird/j0126/areaxfs_v6/'
import numpy as np
from scipy.spatial import cKDTree

from syconn.reps.super_segmentation import *
from syconn.reps.segmentation import SegmentationDataset

ssd = SuperSegmentationDataset()  # class holding all cell representations

sd_synssv = SegmentationDataset(obj_type='syn_ssv')  # class holding all synapse candidates between cells

for syn_id in sd_synssv.ids[:10]:  # some arbitrary synapse IDs
   syn_obj = sd_synssv.get_segmentation_object(syn_id)
   syn_obj.load_attr_dict()
   c1, c2 = syn_obj.attr_dict['neuron_partners']
   cell_obj1, cell_obj2 = ssd.get_super_segmentation_object([c1, c2])

   # TODO: implement a method (merge_superseg_objects) which merges 2 cell objects into a single
   #  object (2x  SuperSegmentationObject -> 1x SuperSegmentationObject)
   # Important: Merge skeleton AND mesh (for now, there will be more)
   merged_cell = merge_superseg_objects(cell_obj1, cell_obj2)
   merged_cell.load_skeleton()

   cell_nodes = merged_cell.skeleton['nodes'] * merged_cell.scaling  # coordinates of all nodes
   node_labels = np.zero((len(cell_nodes), )) * -1  # this should store 1 for false merger,
   # 0 for true merger (and -1 for ignore, optional!)

   syn_coord = syn_obj.rep_coord * merged_cell.scaling

   # find medium cube around artificial merger and set it to 0
   kdtree = cKDTree(...)  # initialize tree with all cell skeleton nodes
   ixs = kdtree.query_ball_point(..., r=20e3)  # find all skeleton nodes which are close to the
   # synapse
   node_labels[ixs] = 0

   # find small cube around artificial merger and set it to 1
   kdtree = cKDTree(...)  # initialize tree with all cell skeleton nodes
   ixs = kdtree.query_ball_point(..., r=5e3)  # find all skeleton nodes which are close to the
   # synapse
   node_labels[ixs] = 1
   merged_cell.skeleton['merger_gt'] = node_labels

   # write out annotated skeletons (see additional_keys=['merger_gt'])
   # TODO: out in only a single kzip file
   fname = f'syn{syn_obj.id}_cells{cell_obj1.id}_{cell_obj2.id}'
   cell_obj1.save_skeleton_to_kzip(fname, additional_keys=['merger_gt'])
   merged_cell.meshes2kzip(fname)
   



# # code snippet for merging two skeletons:
#    cell_obj1.skeleton['edges'] = np.concatenate([cell_obj1.skeleton['edges'],
#                                                  cell_obj2.skeleton['edges'] +
#                                                  len(cell_obj1.skeleton['node'])])  # additional offset
#    cell_obj1.skeleton['nodes'] = np.concatenate([cell_obj1.skeleton['nodes'],
#                                                  cell_obj2.skeleton['nodes']])

# The following meshes need to merged. For that see the method: merge_meshes in proc/meshes.py
# for example:
# merged_cell = SuperSegmentationObject(..., working_dir='~/tmp/', version='tmp')
# for mesh_type in ['mi', 'sv', 'vc', 'sj', 'syn_ssv']:
#   merged_cell._meshes[mesh_type] = merge_meshes(cell_obj1.load_mesh(mesh_type), \
#   cell_obj2.load_mesh(mesh_type))

