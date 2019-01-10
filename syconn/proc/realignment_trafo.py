# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Philipp Schubert, Joergen Kornfeld
try:
    import cPickle as pkl
except ImportError:
    import pickle as pkl
import numpy as np
import time
from knossos_utils import KnossosDataset

from ..reps import segmentation, rep_helper as rh


def source_to_target(orig_coord, realign_map):
    """
    This is at 1/16 scale. The two 32-bit float channel values {u, v} at
    source_image_map voxel (x y, z) indicate that pixel location (16x, 16y, z)
     in the source image stack is mapped to (u, v, z) in the realigned volume.
      If {u, v} are NaN values, that source pixel didn't fall within the realigned
       volume (there are probably no cases of this in J0126_13).

    This mapping should be valid regardless of whether the coordinate systems
     locating (16x, 16y, z) and (u, v, z) both have their origins at the top-left
      corner of the top-left pixel, or at the center of the top-left pixel,
      provided they both follow the same convention.
    """
    x_rem, y_rem = orig_coord[:, 1] % 16, orig_coord[:, 0] % 16

    x, y, z = (orig_coord[:, 1]/16, orig_coord[:, 0]/16, orig_coord[:, 2])
    realigned_coord = np.array([np.round(realign_map[0, x, y, z]) + x_rem,
                                np.round(realign_map[1, x, y, z]) + y_rem, z],
                               dtype=np.int).T
    return realigned_coord


def source_to_target_single(orig_coord, realign_map):
    """
    This is at 1/16 scale. The two 32-bit float channel values {u, v} at
    source_image_map voxel (x y, z) indicate that pixel location (16x, 16y, z)
     in the source image stack is mapped to (u, v, z) in the realigned volume.
      If {u, v} are NaN values, that source pixel didn't fall within the realigned
       volume (there are probably no cases of this in J0126_13).

    This mapping should be valid regardless of whether the coordinate systems
     locating (16x, 16y, z) and (u, v, z) both have their origins at the top-left
      corner of the top-left pixel, or at the center of the top-left pixel,
      provided they both follow the same convention.
    """
    x_rem, y_rem = orig_coord[1] % 16, orig_coord[0] % 16

    x, y, z = (orig_coord[1]/16, orig_coord[0]/16, orig_coord[2])
    realigned_coord = (int(round(realign_map[0, x, y, z])) + x_rem,
                       int(round(realign_map[1, x, y, z])) + y_rem, int(z))
    return realigned_coord


def realign_coords(from_coords, realign_map):
    rem = from_coords % np.array([16, 16, 1], dtype=np.float) / 16
    rem = rem[:, ::-1]
    scaled_coords = from_coords / np.array([16, 16, 1], dtype=np.int)

    target_coords = np.array([realign_map[1, scaled_coords[:, 0],
                                          scaled_coords[:, 1],
                                                   scaled_coords[:, 2]],
                              realign_map[0, scaled_coords[:, 0],
                                                   scaled_coords[:, 1],
                                                   scaled_coords[:, 2]]]).T

    step_target_coords = np.array([realign_map[1, scaled_coords[:, 0] + 1,
                                                           scaled_coords[:, 1] + 1,
                                                           scaled_coords[:, 2]],
                                   realign_map[0, scaled_coords[:, 0] + 1,
                                                        scaled_coords[:, 1] + 1,
                                                        scaled_coords[:, 2]]]).T

    target_coords += (step_target_coords - target_coords) * rem[:, 1:]
    # target_coords += np.array([16, 16]) * rem[:, :2]
    target_coords = np.round(target_coords)
    return np.concatenate((target_coords,
                           scaled_coords[:, 2, None]), axis=1).astype(np.int)


def trafo_object(voxels, realign_map):
    # realigned_voxels = source_to_target(voxels, realign_map=realign_map)

    try:
        realigned_voxels = np.array([source_to_target_single(vx, realign_map) for vx in voxels])
    except:
        return None, None

    voxels = voxels[:, [1, 0, 2]]
    mean_shift = np.round(np.median(realigned_voxels - voxels, axis=0)).astype(np.int)

    return voxels + mean_shift, mean_shift


def trafo_objects_to_kd(realign_map, obj_ids=None, label_id=3,
                        kd_path='/wholebrain/scratch/areaxfs3/knossosdatasets/j0126_realigned_v4b_cbs_ext0_fix/',
                        path_folder='/wholebrain/scratch/areaxfs3/j0126_cset_paper/obj_mito_1037_3d_8'):
    with open(path_folder + '/direct_map.pkl', 'rb') as f:
        obj_map = pkl.load(f)

    kd = KnossosDataset()
    kd.initialize_from_knossos_path(kd_path)

    if obj_ids is None:
        obj_ids = list(obj_map.keys())

    time_start = time.time()
    for i_obj_id, obj_id in enumerate(obj_ids):
        voxels = np.array([], dtype=np.int32).reshape(0, 3)

        for voxel_file in obj_map[obj_id]:
            voxels = np.concatenate([voxels, np.load(path_folder + '/voxels/%d_%d.npz' %
                                             (voxel_file[0], voxel_file[1]))["%d" % obj_id]])

        voxels_t, mean_shift = trafo_object(voxels, realign_map)

        bb = np.array([np.min(voxels_t, axis=0), np.max(voxels_t, axis=0)],
                      dtype=np.int)

        if np.product(bb.shape) > 1e10:
            print("Obj %d too large" % obj_id)
            continue

        voxels_t -= bb[0]

        vx_cube = np.zeros(bb[1] - bb[0] + 1, dtype=np.uint64)
        vx_cube[voxels_t[:, 0], voxels_t[:, 1], voxels_t[:, 2]] = label_id

        kd.from_matrix_to_cubes(bb[0], data=vx_cube,
                                as_raw=False, overwrite=False)
        dt = time.time() - time_start
        print("Written obj %d at bb %s - time / object: %.3fs - eta: %.3fh" %
              (obj_id, bb[0], dt / (i_obj_id + 1),
               dt / (i_obj_id + 1) * len(obj_ids) / 3600))

        # if i_obj_id > 2000:
        #     break


def trafo_objects_to_sd(realign_map, sd_obj_type, working_dir,
                        sd_n_folders_fs, sd_version='new', obj_ids=None,
                        path_folder='/wholebrain/scratch/areaxfs3/j0126_cset_paper/obj_mito_1037_3d_8'):
    with open(path_folder + '/direct_map.pkl', 'rb') as f:
        obj_map = pkl.load(f)

    segdataset = segmentation.SegmentationDataset(obj_type=sd_obj_type,
                                                  working_dir=working_dir,
                                                  version=sd_version,
                                                  n_folders_fs=sd_n_folders_fs,
                                                  create=True)

    if obj_ids is None:
        obj_ids = np.array(list(obj_map.keys()))

    sub_fold_ids = np.array([int(rh.subfold_from_ix(obj_id, 10000).replace('/', '')) for obj_id in obj_ids])
    obj_id_order = np.argsort(sub_fold_ids)
    obj_ids = np.array(obj_ids)[obj_id_order]

    oobs = []
    otls = []

    voxel_rel_path = rh.subfold_from_ix(obj_ids[0], sd_n_folders_fs)

    mean_shift_dict = {}
    time_start = time.time()
    for i_obj_id, obj_id in enumerate(obj_ids):
        voxels = np.array([], dtype=np.int32).reshape(0, 3)

        for voxel_file in obj_map[obj_id]:
            voxels = np.concatenate([voxels, np.load(path_folder + '/voxels/%d_%d.npz' %
                                             (voxel_file[0], voxel_file[1]))["%d" % obj_id]])

        voxels_t, mean_shift = trafo_object(voxels, realign_map)

        if voxels_t is None:
            oobs.append(obj_id)
            continue

        mean_shift_dict[obj_id] = mean_shift

        bb = np.array([np.min(voxels_t, axis=0), np.max(voxels_t, axis=0)],
                      dtype=np.int)

        if np.product(bb.shape) > 1e11:
            print("Obj %d too large" % obj_id)
            otls.append(obj_id)
            continue

        voxels_t -= bb[0]

        vx_cube = np.zeros(bb[1] - bb[0] + 1, dtype=np.bool)
        vx_cube[voxels_t[:, 0], voxels_t[:, 1], voxels_t[:, 2]] = True

        obj = segdataset.get_segmentation_object(obj_id, create=True)
        obj.save_voxels(vx_cube, bb[0])

        if i_obj_id % 100 == 0:
            dt = time.time() - time_start
            eta = (dt / (i_obj_id + 1) * len(obj_ids) - dt) / 3600,
            print("Written obj %d - time / object: %.3fs - eta: %.3fh - "
                  "oob %d - otl %d" %
                  (obj_id, dt / (i_obj_id + 1), eta, len(oobs), len(otls)))

        if i_obj_id % 1000 == 0:
            with open(segdataset.path + '/trafo_mean_shift_dict.pkl', 'wb') as f:
                pkl.dump(mean_shift_dict, f)

    with open(segdataset.path + '/trafo_mean_shift_dict.pkl', 'wb') as f:
        pkl.dump(mean_shift_dict, f)

    print(oobs)
    print("%d out of bounds objects" % len(oobs))
    print(otls)
    print("%d objects were too large" % len(otls))


def load_realign_map():
    return np.load('/wholebrain/scratch/jkornfeld/coord_map.npy')

