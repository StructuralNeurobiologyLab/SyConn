from syconn import global_params
global_params.wd = '/wholebrain/songbird/j0126/areaxfs_v6/'
import numpy as np
from scipy.spatial import cKDTree

from syconn.reps.super_segmentation import *
from syconn.reps.segmentation import SegmentationDataset
from syconn.proc.meshes import merge_meshes

def merge_superseg_objects(cell_obj1, cell_obj2):

    # TODO: test why working_dir='/tmp/' raise some errors
    # merge meshes
    merged_cell = SuperSegmentationObject(ssv_id=-1, working_dir=None, version='tmp')
    for mesh_type in ['sv', 'sj', 'syn_ssv', 'vc', 'mi']:
        mesh1 = cell_obj1.load_mesh(mesh_type)
        mesh2 = cell_obj2.load_mesh(mesh_type)
        ind_lst = [mesh1[0], mesh2[0]]
        vert_lst = [mesh1[1], mesh2[1]]
        merged_cell._meshes[mesh_type] = merge_meshes(ind_lst, vert_lst)
        merged_cell._meshes[mesh_type] += ([None, None], ) # add normals
    
    # merge skeletons
    merged_cell.skeleton = {}
    cell_obj1.load_skeleton()
    cell_obj2.load_skeleton()
    merged_cell.skeleton['edges'] = np.concatenate([cell_obj1.skeleton['edges'],
                                                    cell_obj2.skeleton['edges'] +
                                                    len(cell_obj1.skeleton['nodes'])]) # additional offset
    merged_cell.skeleton['nodes'] = np.concatenate([cell_obj1.skeleton['nodes'],
                                                    cell_obj2.skeleton['nodes']])
    merged_cell.skeleton['diameters'] = np.concatenate([cell_obj1.skeleton['diameters'],
                                                        cell_obj2.skeleton['diameters']])
    return merged_cell


if __name__=="__main__":

    ssd = SuperSegmentationDataset() # class holding all cell representations

    sd_synssv = SegmentationDataset(obj_type='syn_ssv') # class holding all synapse candidates between cells

    for syn_id in sd_synssv.ids[1:2]:  # some arbitrary synapse IDs
        syn_obj = sd_synssv.get_segmentation_object(syn_id)
        syn_obj.load_attr_dict()
        c1, c2 = syn_obj.attr_dict['neuron_partners']

        cell_obj1, cell_obj2 = ssd.get_super_segmentation_object([c1, c2])
        merged_cell = merge_superseg_objects(cell_obj1, cell_obj2)

        cell_nodes = merged_cell.skeleton['nodes'] * merged_cell.scaling # coordinates of all nodes

        # labels:
        # 1 for false-merger, 0 for true merger, -1 for ignore
        node_labels = np.zeros((len(cell_nodes), )) #* -1

        syn_coord = syn_obj.rep_coord * merged_cell.scaling

        # find medium cube around artificial merger and set it to 0 (true merger)
        kdtree = cKDTree(cell_nodes)
        # find all skeleton nodes which are close to the synapse
        ixs = kdtree.query_ball_point(syn_coord, r=20e3) ## 20e3 nanometer  
        node_labels[ixs] = 1 # correct: 0

        # find small cube around artificial merger and set it to 1 (false merger)
        kdtree = cKDTree(cell_nodes)
        # find all skeleton nodes which are close to the synapse
        ixs = kdtree.query_ball_point(syn_coord, r=5e3) 
        node_labels[ixs] = 2 # correct: 1
        # write out annotated skeletons to ['merger_gt']
        merged_cell.skeleton['merger_gt'] = node_labels

        # write all data to kzip (skeleton and mesh)
        fname = f'test_meta_syn{syn_obj.id}_cells{cell_obj1.id}_{cell_obj2.id}.k.zip'

        target_fnames = []
        tmp_dest_p = []
        tmp_dest_p.append('{}_{}.pkl'.format(fname, 'meta'))
        target_fnames.append('{}.pkl'.format('meta'))
        write_obj2pkl(tmp_dest_p[-1], {'version_dict': merged_cell.version_dict,
                                       'scaling': merged_cell.scaling,
                                       'working_dir': merged_cell.working_dir,
                                       'sso_id': merged_cell.id})

        data2kzip(fname, tmp_dest_p, target_fnames)

        merged_cell.save_skeleton_to_kzip(fname, additional_keys=['merger_gt'])
        merged_cell.meshes2kzip(fname)

