from syconn.reps.super_segmentation import SuperSegmentationDataset
from syconn.reps.segmentation import SegmentationDataset
from syconn.handler.compression import AttributeDict, MeshDict, VoxelDict
from syconn.mp.shared_mem import start_multiprocess
from syconn.proc.meshes import triangulation
from syconn.config.global_params import MESH_DOWNSAMPLING, MESH_CLOSING
orig_mesh_dir = "/u/pschuber/areaxfs/sv_0/so_storage/"
import itertools
import numpy as np


def mesh_creator_sso(ssv):
    ssv.load_attr_dict()
    _ = ssv._load_obj_mesh(obj_type="mi", rewrite=False)
    _ = ssv._load_obj_mesh(obj_type="sj", rewrite=False)
    _ = ssv._load_obj_mesh(obj_type="vc", rewrite=False)
    _ = ssv._load_obj_mesh(obj_type="sv", rewrite=False)
    ssv.clear_cache()


def mesh_chunk(args):
    attr_dir, obj_type = args
    ad = AttributeDict(attr_dir + "/attr_dict.pkl", disable_locking=True)
    obj_ixs = ad.keys()
    if len(obj_ixs) == 0:
        print "EMPTY ATTRIBUTE DICT", attr_dir
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
                                     scaling=SCALING, n_closings=MESH_CLOSING[obj_type])
        vertices *= SCALING
        md[ix] = [indices.flatten(), vertices.flatten(), normals.flatten()]
    md.save2pkl()
    print attr_dir


def mesh_proc_chunked(obj_type, working_dir, n_folders_fs=10000):
    sds = SegmentationDataset(obj_type, working_dir=working_dir, n_folders_fs=n_folders_fs)
    fold = sds.so_storage_path
    f1 = np.arange(0, 100)
    f2 = np.arange(0, 100)
    all_poss_attr_dicts = list(itertools.product(f1, f2))
    assert len(all_poss_attr_dicts) == 10000
    print "Processing %d mesh dicts of %s." % (len(all_poss_attr_dicts), obj_type)
    multi_params = [["%s/%02d/%02d/" % (fold, par[0], par[1]), obj_type] for par in all_poss_attr_dicts]
    start_multiprocess(mesh_chunk, multi_params, nb_cpus=20, debug=False)



if __name__ == "__main__":
    wd = "/wholebrain/scratch/areaxfs3/"
    dummy_ssd = SuperSegmentationDataset(working_dir=wd)
    # define SCALING variable for methods above
    global SCALING
    SCALING = dummy_ssd.scaling

    # preprocess meshes of all objects
    # TODO: CHECK IF n_folders_fs MAKES SENSE (has to be read out from config or something @sven)
    mesh_proc_chunked("conn", wd, n_folders_fs=10000)
    mesh_proc_chunked("sj", wd, n_folders_fs=10000)
    mesh_proc_chunked("vc", wd, n_folders_fs=10000)
    mesh_proc_chunked("mi", wd, n_folders_fs=10000)

    # cache meshes of SSV objects, here for axon ground truth,
    # e.g. change version to "0" for initial run on all SSVs in the segmentation
    ssds = SuperSegmentationDataset(working_dir=wd,
                                    version="axgt", ssd_type="ssv")
    start_multiprocess(mesh_creator_sso, list(ssds.ssvs), nb_cpus=20, debug=False)

