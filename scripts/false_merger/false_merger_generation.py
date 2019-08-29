from syconn import global_params
global_params.wd = '/wholebrain/songbird/j0126/areaxfs_v6/'
import numpy as np
from scipy.spatial import cKDTree

from syconn.reps.super_segmentation import *
from syconn.reps.segmentation import SegmentationDataset
from syconn.proc.meshes import merge_meshes

def merge_superseg_objects(cell_obj1, cell_obj2):

    # TODO: test why working_dir='/tmp/' raise some errors
    # merged_cell = SuperSegmentationObject(ssv_id=-1, working_dir='/tmp/', version='tmp') 
    merged_cell = SuperSegmentationObject(ssv_id=-1, working_dir=None, version='tmp')
    for mesh_type in ['sv', 'sj', 'syn_ssv', 'vc', 'mi']:
        # TEST
        # mesh1 = cell_obj1.load_mesh(mesh_type)
        # mesh2 = cell_obj2.load_mesh(mesh_type)
        # print(">>>>mesh_type: {}".format(mesh_type))
        # if mesh1 is None:
        #     print("mesh1 is None")
        # else:
        #     print("type: {}, len: {}".format( type(mesh1), len(mesh1) ))
        #     for items in zip(mesh1, mesh2):
        #         print("shape1: {}, shape2: {}".format( items[0].shape, items[0].shape ))
        # TEST
        merged_cell._meshes[mesh_type] = merge_meshes(cell_obj1.load_mesh(mesh_type), cell_obj2.load_mesh(mesh_type))
        
    return merged_cell


if __name__=="__main__":

    ssd = SuperSegmentationDataset()  # class holding all cell representations

    sd_synssv = SegmentationDataset(obj_type='syn_ssv')  # class holding all synapse candidates between cells

    for syn_id in sd_synssv.ids[:2]:  # some arbitrary synapse IDs
        syn_obj = sd_synssv.get_segmentation_object(syn_id)
        syn_obj.load_attr_dict()
        c1, c2 = syn_obj.attr_dict['neuron_partners']
        # TEST
        print(">>>>>>>>>>> syn_id: {}".format(syn_id))
        # print("Neuron_parners: {}, {}".format(c1, c2))
        # TEST
        # cell_obj: SuperSegmentationObject
        cell_obj1, cell_obj2 = ssd.get_super_segmentation_object([c1, c2])

        # TODO: implement a method (merge_superseg_objects) which merges 2 cell objects into a single
        #       object (2x  SuperSegmentationObject -> 1x SuperSegmentationObject)
        # Important: Merge skeleton AND mesh (for now, there will be more)
        merged_cell = merge_superseg_objects(cell_obj1, cell_obj2)
        # TEST
        print("---------------------------------------------")
        for key in merged_cell._meshes:
            if key != 'conn':
                print(key)
                for item in merged_cell._meshes[key]:
                    print("shape: {}".format(item.shape))
        print("--------------------------------------------")
        print("")
        # TEST
        # merged_cell.load_skeleton()

        # cell_nodes = merged_cell.skeleton['nodes'] * merged_cell.scaling  # coordinates of all nodes
        # node_labels = np.zero((len(cell_nodes), )) * -1  # this should store 1 for false merger,
        # # 0 for true merger (and -1 for ignore, optional!)

        # syn_coord = syn_obj.rep_coord * merged_cell.scaling

        # # find medium cube around artificial merger and set it to 0
        # kdtree = cKDTree(...)  # initialize tree with all cell skeleton nodes
        # ixs = kdtree.query_ball_point(..., r=20e3) ## 20e3 nanometer  # find all skeleton nodes which are close to the
        # # synapse
        # node_labels[ixs] = 0

        # # find small cube around artificial merger and set it to 1
        # kdtree = cKDTree(...)  # initialize tree with all cell skeleton nodes
        # ixs = kdtree.query_ball_point(..., r=5e3)  # find all skeleton nodes which are close to the
        # # synapse
        # node_labels[ixs] = 1
        # merged_cell.skeleton['merger_gt'] = node_labels

        # # write out annotated skeletons (see additional_keys=['merger_gt'])
        # # TODO: out in only a single kzip file
        # fname = f'syn{syn_obj.id}_cells{cell_obj1.id}_{cell_obj2.id}'
        # cell_obj1.save_skeleton_to_kzip(fname, additional_keys=['merger_gt'])
        # merged_cell.meshes2kzip(fname)




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

