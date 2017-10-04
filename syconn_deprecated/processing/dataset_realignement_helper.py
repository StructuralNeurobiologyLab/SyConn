from collections import Counter
import itertools
from knossos_utils import knossosdataset
import numpy as np
import scipy.interpolate
import scipy.spatial
import time

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
    return np.concatenate((target_coords, scaled_coords[:, 2, None]), axis=1).astype(np.int)


def realign_from_source_w_kd(source_coords, source_to_target_map, kd):
    target_coords = realign_coords(source_coords, source_to_target_map)

    min_target_coords = np.min(target_coords, axis=0)
    max_target_coords = np.max(target_coords, axis=0)

    raw_values = kd.from_raw_cubes_to_list(source_coords)

    target_coords -= min_target_coords

    target_sh = max_target_coords - min_target_coords + 1
    block = np.zeros(target_sh, dtype=np.int)
    occ = np.zeros(target_sh, dtype=np.int)

    np.add.at(block, [target_coords[:, 0],
                      target_coords[:, 1],
                      target_coords[:, 2]], raw_values)
    np.add.at(occ, [target_coords[:, 0],
                    target_coords[:, 1],
                    target_coords[:, 2]], 1)

    block[occ != 0] /= occ[occ != 0]
    block[occ == 0] = 0

    return block, min_target_coords


def realign_from_target_w_kd(target_coords, target_to_source_map, kd):
    source_coords = realign_coords(target_coords, target_to_source_map)

    min_target_coords = np.min(target_coords, axis=0)
    max_target_coords = np.max(target_coords, axis=0)

    raw_values = kd.from_raw_cubes_to_list(source_coords)

    target_coords -= min_target_coords

    target_sh = max_target_coords - min_target_coords + 1
    block = np.zeros(target_sh, dtype=np.int)

    block[target_coords[:, 0], target_coords[:, 1], target_coords[:, 2]] = raw_values

    return block


def realign_from_target_w_kd_thread(args):
    # time_start = time.time()
    offsets = args[0]
    size = args[1]
    inv_coord_map_path = args[2]
    kd_from_path = args[3]
    kd_to_path = args[4]

    inv_coord_map = np.load(inv_coord_map_path)

    kd_from = knossosdataset.KnossosDataset()
    kd_from.initialize_from_knossos_path(kd_from_path)

    kd_to = knossosdataset.KnossosDataset()
    kd_to.initialize_from_knossos_path(kd_to_path)

    # print "Time prep: %.2fs" % (time.time() - time_start)
    for i_offset in range(len(offsets)):
        # time_start = time.time()
        offset = offsets[i_offset]
        coords = np.array(list(itertools.product(
            *[range(size[i]) for i in range(3)]))) + np.array(offset)
        block = realign_from_target_w_kd(coords, inv_coord_map, kd_from)
        kd_to.from_matrix_to_cubes(offset, data=block, as_raw=True,
                                   datatype=np.uint8)

        # print "Time step %d: %.2fs; %.2f MV/s" % (i_offset, time.time() - time_start, np.product(size) / 1.e6 / (time.time() - time_start))


def invert_coord_map_z(source_to_target_map_z, target_shape):
    cx_map = source_to_target_map_z[0] / 16
    cy_map = source_to_target_map_z[1] / 16
    t_x, t_y = np.indices(target_shape)
    s_x, _ = np.indices(cx_map.shape)
    _, s_y = np.indices(cx_map.shape)

    zx = scipy.interpolate.griddata((cx_map.ravel(), cy_map.ravel()), s_x.ravel(),
                                    (t_x.ravel(), t_y.ravel()), method="cubic")
    zy = scipy.interpolate.griddata((cx_map.ravel(), cy_map.ravel()), s_y.ravel(),
                                    (t_x.ravel(), t_y.ravel()), method="cubic")

    zx = np.nan_to_num(zx).reshape(target_shape) * 16
    zy = np.nan_to_num(zy).reshape(target_shape) * 16

    zx[np.where(zx < 0)] = 0
    zy[np.where(zy < 0)] = 0

    return np.array([zx, zy])


def invert_coord_map(source_to_target_map, n_z=None, target_shape=None):
    if target_shape is not None:
        n_z = target_shape[-1]
    elif n_z is not None:
        target_shape = np.ceil([2, np.max(source_to_target_map[0]),
                                np.max(source_to_target_map[1]),
                                n_z]).astype(np.int)
    else:
        target_shape = np.ceil([2, np.max(source_to_target_map[0]),
                                np.max(source_to_target_map[1]),
                                source_to_target_map.shape[-1]]).astype(np.int)
        n_z = len(source_to_target_map.shape[-1])

    target_to_source_map = np.zeros(target_shape)
    for i_z in range(n_z):
        print i_z, target_shape[1:-1]
        target_to_source_map[..., i_z] = \
            invert_coord_map_z(source_to_target_map[..., i_z].copy(),
                               target_shape[1:-1])

    return target_to_source_map


def invert_coord_map_thread(args):
    zs = args[0]
    path_coord_map = args[1]
    target_shape = args[2]

    target_shape[-1] = len(zs)

    print zs
    coord_map = np.load(path_coord_map)[..., zs]
    inv_coord_map = invert_coord_map(coord_map, len(zs), target_shape)

    return zs, inv_coord_map




