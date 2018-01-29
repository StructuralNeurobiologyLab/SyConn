from syconn.reps.super_segmentation import SuperSegmentationDataset, SuperSegmentationObject
from syconn.reps.segmentation import SegmentationDataset, SegmentationObject
from syconn.reps.rep_helper import subfold_from_ix
from syconn.handler.basics import chunkify
from syconn.handler.compression import AttributeDict, MeshDict, VoxelDict
from syconn.proc.meshes import write_mesh2kzip
from syconn.mp.shared_mem import start_multiprocess
from syconn.proc.meshes import triangulation
from syconn.config.global_params import MESH_DOWNSAMPLING
import shutil
import sys
import os
orig_mesh_dir = "/u/pschuber/areaxfs/sv_0/so_storage/"
import itertools
import numpy as np
from shutil import copyfile

# def copy_mesh_for_so(so):
#     so_id = so.id
#     if not os.path.isdir(so.segobj_dir):
#         print so.segobj_dir
#         return
#         os.makedirs(so.segobj_dir)
#     try:
#         shutil.copy(orig_mesh_dir + subfold_from_ix(so_id)+ "mesh.pkl", so.segobj_dir)
#     except:
#         pass
#
# def copy_ssds_meshs(ssd):
#     for sso in ssd.ssvs:
#         # sso.load_attr_dict()
#         svs = list(sso.svs)
#         if len(svs) > 1e4:
#             print "Skipped %d with %d SV's." % (sso.id, len(svs))
#             continue
#         start_multiprocess(copy_mesh_for_so, svs, nb_cpus=20)
#         sys.stdout.write("\r%d" % sso.id)
#         sys.stdout.flush()


def mesh_creator(so, force=True):
    if not force:
        _ = so.mesh
    else:
        indices, vertices = so._mesh_from_scratch()
        try:
            so._save_mesh(indices, vertices)
        except Exception, e:
            print "\n-----------------------\n" \
                  "Mesh of %s %d could not be saved:\n%s\n" \
                  "-------------------------\n" % (so.type, so.id, e)


def mesh_creator_sso(ssv):
    # try:
    ssv.load_attr_dict()
    _ = ssv._load_obj_mesh(obj_type="mi", rewrite=False)
    _ = ssv._load_obj_mesh(obj_type="sj", rewrite=False)
    _ = ssv._load_obj_mesh(obj_type="vc", rewrite=False)
    _ = ssv._load_obj_mesh(obj_type="sv", rewrite=False)
    ssv.calculate_skeleton()
    ssv.clear_cache()
    # except Exception, e:
    #     print "Error occurred:", e, ssv.id


def mesh_creator_mi(ixs):
    for ix in ixs:
        so = SegmentationObject(ix, "mi", working_dir="/wholebrain/scratch/areaxfs3/", n_folders_fs=10000)
        mesh_creator(so)
        del so


def mesh_creator_sj(ixs):
    for ix in ixs:
        so = SegmentationObject(ix, "sj", working_dir="/wholebrain/scratch/areaxfs3/", n_folders_fs=10000)
        mesh_creator(so)
        del so


def mesh_creator_vc(ixs):
    for ix in ixs:
        so = SegmentationObject(ix, "vc", working_dir="/wholebrain/scratch/areaxfs3/", n_folders_fs=10000)
        mesh_creator(so)
        del so


def mesh_chunck(attr_dir):
    ad = AttributeDict(attr_dir + "/attr_dict.pkl", disable_locking=True)
    obj_ixs = ad.keys()
    if len(obj_ixs) == 0:
        print "EMPTY MESH DICT", attr_dir
        return
    voxel_dc = VoxelDict(attr_dir + "/voxel.pkl", disable_locking=True)
    md = MeshDict(attr_dir + "/mesh.pkl", disable_locking=True, read_only=False)
    if "sj" in attr_dir:
        obj_type = "sj"
    elif "vc" in attr_dir:
        obj_type = "vc"
    elif "mi" in attr_dir:
        obj_type = "mi"
    else:
        assert 0
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
                                          scaling=SCALING)
        vertices *= SCALING
        md[ix] = [indices.flatten(), vertices.flatten(), normals.flatten()]
    md.save2pkl()
    print attr_dir


def mesh_proc_chunked(obj_type):
    sds = SegmentationDataset(obj_type, working_dir="/wholebrain/scratch/areaxfs3/",
                              n_folders_fs=10000)
    fold = sds.so_storage_path
    f1 = np.arange(0, 10)
    f2 = np.arange(0, 100)
    all_poss_attr_dicts = list(itertools.product(f1, f2))
    all_poss_attr_dicts += list(itertools.product(f2, f1))
    assert len(all_poss_attr_dicts) == 2000
    print "Processing %d mesh dicts of %s." % (len(all_poss_attr_dicts), obj_type)
    multi_params = ["%s/%02d/%02d/" % (fold, par[0], par[1]) for par in all_poss_attr_dicts]
    start_multiprocess(mesh_chunck, multi_params, nb_cpus=20, debug=False)
#
# def preproc_meshs(ssd):
#     for sso in ssd.ssvs:
#         start_multiprocess(mesh_creator, sso.svs, nb_cpus=20, debug=False)
#         sys.stdout.write("\r%d" % sso.id)
#         sys.stdout.flush()
#
#
# def write_meshs_helper(sso):
#     sso.meshs2kzip()
#     sso.mergelist2kzip()


def copy_sv_skeletons():
    sds = SegmentationDataset("sv", working_dir="/wholebrain/scratch/areaxfs/", version="0", n_folders_fs=100000)
    new_sds = SegmentationDataset("sv", working_dir="/wholebrain/scratch/areaxfs3/", version="0", n_folders_fs=100000)
    fold = sds.so_storage_path
    new_fold = new_sds.so_storage_path
    f1 = np.arange(0, 100)
    f2 = np.arange(0, 100)
    f3 = np.arange(0, 10)
    all_poss_attr_dicts = list(itertools.product(f1, f2, f3))
    assert len(all_poss_attr_dicts) == 100*100*10
    multi_params = [["%s/%d/%d/%d/" % (fold, par[0], par[1], par[2]), "%s/%d/%d/%d/" % (new_fold, par[0], par[1], par[2])] for par in all_poss_attr_dicts]
    start_multiprocess(copy_sv_skeleton, multi_params, nb_cpus=20, debug=False)


def copy_sv_skeleton(dirs):
    if os.path.isfile(dirs[1] + "skeletons.pkl"):
        return
    copyfile(dirs[0] + "skeletons.pkl", dirs[1] + "skeletons.pkl")
    if os.path.isfile(dirs[1] + "skeleton.pkl"):
        os.remove(dirs[1] + "skeleton.pkl")
    print dirs[1]


def copy_axoness():
    sds = SegmentationDataset("sv", working_dir="/wholebrain/scratch/areaxfs/", version="0")
    new_sds = SegmentationDataset("sv", working_dir="/wholebrain/scratch/areaxfs3/", version="0", n_folders_fs=100000)
    fold = sds.so_storage_path
    new_fold = new_sds.so_storage_path
    f1 = np.arange(0, 100)
    f2 = np.arange(0, 100)
    f3 = np.arange(0, 10)
    all_poss_attr_dicts = list(itertools.product(f1, f2, f3))
    assert len(all_poss_attr_dicts) == 100*100*10
    multi_params = [["%s/%d/%d/%d/" % (fold, par[0], par[1], par[2]), "%s/%d/%d/%d/" % (new_fold, par[0], par[1], par[2])] for par in all_poss_attr_dicts][2:]
    start_multiprocess(copy_axoness_single, multi_params, nb_cpus=20, debug=False)


def copy_axoness_single(dirs):
    old_ad = AttributeDict(dirs[0] + "attr_dict.pkl")
    new_ad = AttributeDict(dirs[1] + "attr_dict.pkl", read_only=False)
    for k, v in old_ad.iteritems():
        try:
            sv_attr_dc = new_ad[k]
            if not type(sv_attr_dc) == dict:
                sv_attr_dc = {}
            sv_attr_dc["axoness_proba"] = v["axoness_proba"]
            new_ad[k] = sv_attr_dc
            assert "axoness_proba" in new_ad[k]
        except KeyError:
            if "size" in v and v["size"] > 1e5:
                print "Couldn't find axoness for SV with ID", k, "and size", v["size"]
        assert type(new_ad[k]) is dict and len(new_ad[k]) >= 7
    new_ad.save2pkl()
    print dirs[1]


if __name__ == "__main__":
    # copy_sv_skeletons()
    # copy_axoness()
    ssds = SuperSegmentationDataset(working_dir="/wholebrain/scratch/areaxfs3/",
                                    version="0", ssd_type="ax_gt")
    global SCALING
    SCALING = ssds.scaling
    # mesh_proc_chunked("sj")
    # mesh_proc_chunked("vc")
    # mesh_proc_chunked("mi")
    start_multiprocess(mesh_creator_sso, list(ssds.ssvs), nb_cpus=20, debug=False)
    # start_multiprocess(write_meshs_helper, list(ssds.ssvs), nb_cpus=20)
    # start_multiprocess(write_meshs_helper, list(ssds.ssvs), nb_cpus=20)

    # copy_ssds_meshs(ssds)
    # sds = SegmentationDataset("sj", working_dir="/wholebrain/scratch/areaxfs3/", n_folders_fs=10000)
    # global SCALING
    # SCALING = sds.scaling.astype(np.float32)
    # print SCALING






    # sds = SegmentationDataset("sj", working_dir="/wholebrain/scratch/areaxfs3/", n_folders_fs=10000)
    # usable_ids = sds.ids#[sds.sizes > 498]
    # print "Processing %d sj's." % (len(usable_ids))
    # chs = chunkify(usable_ids, 1000)
    # start_multiprocess(mesh_creator_sj, chs, nb_cpus=20, debug=False)
    #
    #
    # sds = SegmentationDataset("vc", working_dir="/wholebrain/scratch/areaxfs3/", n_folders_fs=10000)
    # usable_ids = sds.ids#[sds.sizes > 1584]
    # print "Processing %d vc's." % (len(usable_ids))
    # chs = chunkify(usable_ids, 1000)
    # start_multiprocess(mesh_creator_vc, chs, nb_cpus=20, debug=False)
    #
    #
    # sds = SegmentationDataset("mi", working_dir="/wholebrain/scratch/areaxfs3/", n_folders_fs=10000)
    # usable_ids = sds.ids#[sds.sizes > 2786]
    # print "Processing %d mi's." % (len(usable_ids))
    # chs = chunkify(usable_ids, 1000)
    # start_multiprocess(mesh_creator_mi, chs, nb_cpus=20, debug=False)
    # copy_ssds_meshs(ssds)
    # preproc_meshs(ssds)
    # ssds = SuperSegmentationDataset(working_dir="/wholebrain/scratch/areaxfs/",
    #                                 version="ctgt")
    # copy_ssds_meshs(ssds)
    # preproc_meshs(ssds)