from syconn.reps.super_segmentation import SuperSegmentationDataset
from syconn.reps.segmentation import SegmentationDataset
from syconn.handler.compression import AttributeDict, MeshDict, VoxelDict
from syconn.mp.mp_utils import start_multiprocess_imap, start_multiprocess
from syconn.proc.meshes import triangulation
from syconn.config.global_params import MESH_DOWNSAMPLING, MESH_CLOSING, wd, \
    get_dataset_scaling
import itertools
import numpy as np


def mesh_creator_sso(ssv):
    ssv.enable_locking = False
    ssv.load_attr_dict()
    _ = ssv._load_obj_mesh(obj_type="mi", rewrite=False)
    _ = ssv._load_obj_mesh(obj_type="sj", rewrite=False)
    _ = ssv._load_obj_mesh(obj_type="vc", rewrite=False)
    _ = ssv._load_obj_mesh(obj_type="sv", rewrite=False)
    try:
        ssv.attr_dict["conn"] = ssv.attr_dict["conn_ids"]
        _ = ssv._load_obj_mesh(obj_type="conn", rewrite=False)
    except KeyError:
        print("Loading 'conn' objects failed for SSV %s."
              % ssv.id)
    ssv.clear_cache()


def mesh_chunk(args):
    scaling = get_dataset_scaling()
    attr_dir, obj_type = args
    ad = AttributeDict(attr_dir + "/attr_dict.pkl", disable_locking=True)
    obj_ixs = list(ad.keys())
    if len(obj_ixs) == 0:
        print("EMPTY ATTRIBUTE DICT", attr_dir)
        return
    voxel_dc = VoxelDict(attr_dir + "/voxel.pkl", disable_locking=True)
    md = MeshDict(attr_dir + "/mesh.pkl", disable_locking=True, read_only=False)
    valid_obj_types = ["vc", "sj", "mi", "con"]
    if not obj_type in valid_obj_types:
        raise NotImplementedError("Object type must be one of the following:\n"
                                  "%s" % str(valid_obj_types))
    for ix in obj_ixs:
        # create voxel_list
        bin_arrs, block_offsets = voxel_dc[ix]
        voxel_list = np.array([], dtype=np.int32)
        for i_bin_arr in range(len(bin_arrs)):
            block_voxels = np.array(zip(*np.nonzero(bin_arrs[i_bin_arr])),
                                    dtype=np.int32)
            block_voxels += np.array(block_offsets[i_bin_arr])

            if len(voxel_list) == 0:
                voxel_list = block_voxels
            else:
                voxel_list = np.concatenate([voxel_list, block_voxels])
        # create mesh
        indices, vertices, normals = triangulation(np.array(voxel_list),
                                     downsampling=MESH_DOWNSAMPLING[obj_type],
                                     n_closings=MESH_CLOSING[obj_type])
        vertices *= scaling
        md[ix] = [indices.flatten(), vertices.flatten(), normals.flatten()]
    md.save2pkl()


def mesh_proc_chunked(obj_type, working_dir):
    sd = SegmentationDataset(obj_type, working_dir=working_dir)
    multi_params = sd.so_dir_paths
    print("Processing %d mesh dicts of %s." % (len(multi_params), obj_type))
    start_multiprocess_imap(mesh_chunk, multi_params, nb_cpus=20, debug=False)


if __name__ == "__main__":
    # preprocess meshes of all objects
    mesh_proc_chunked("conn", wd)
    mesh_proc_chunked("sj", wd)
    mesh_proc_chunked("vc", wd)
    mesh_proc_chunked("mi", wd)
    # cache meshes of SSV objects, here for axon ground truth,
    # e.g. change version to "0" for initial run on all SSVs in the segmentation
    ssds = SuperSegmentationDataset(working_dir=wd,)
                                    #version="axgt", ssd_type="ssv")
    start_multiprocess(mesh_creator_sso, list(ssds.ssvs), nb_cpus=20, debug=False)

