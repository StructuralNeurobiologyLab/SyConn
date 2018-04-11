# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max-Planck-Institute for Medical Research, Heidelberg, Germany
# Authors: Sven Dorkenwald, Philipp Schubert, Joergen Kornfeld
import numpy as np
import os
try:
    import cPickle as pickle
except ImportError:
    import pickle
from knossos_utils import chunky
import re
from scipy import ndimage
from scipy import spatial
import shutil
from multiprocessing import Pool
import glob


try:
    import QSUB_MAIN as qm
    qsub_available = True
except:
    qsub_available = False

from ..utils import basics


def get_rel_path(obj_name, filename, suffix=""):
    """
    Returns path from ChunkDataset foler to UltrastructuralDataset folder

    Parameters
    ----------
    obj_name: str
        ie. hdf5name
    filename: str
        Filename of the prediction in the chunkdataset
    suffix: str
        suffix of name

    Returns
    -------
    rel_path: str
        relative path from ChunkDataset folder to UltrastructuralDataset folder

    """
    if len(suffix) > 0 and not suffix[0] == "_":
        suffix = "_" + suffix
    return "/obj_" + obj_name + "_" + \
           filename + suffix + "/"


def path_in_voxel_folder(obj_id, chunk_nb):
    lower = int(obj_id / 1000) * 1000
    upper = lower + 999
    return str(lower) + "_" + str(upper) + "/" + str(obj_id) + "_" + \
           str(chunk_nb) + ".npz"


def extract_and_save_all_hull_voxels_thread(args):
    path = args[0]
    object_dataset_path = args[1]
    set_cnt = args[2]
    obj_key_list = args[3]

    object_dataset = load_dataset(object_dataset_path)
    set_dict = {}
    for obj_key in obj_key_list:
        obj = object_dataset.object_dict[obj_key]
        vx = obj.voxels
        hull_vx = vx[obj.create_hull_ids()]
        set_dict[str(obj.obj_id)] = hull_vx
    np.savez_compressed(path + "/%d" % set_cnt, **set_dict)


def extract_and_save_all_hull_voxels(object_dataset_path, overwrite=False,
                                     nb_processes=1, use_qsub=False,
                                     queue="full"):
    object_dataset = load_dataset(object_dataset_path)
    path = object_dataset.path + "/hull_voxels/"

    if os.path.exists(path) and overwrite:
        shutil.rmtree(path)
    elif not os.path.exists(path):
        pass
    else:
        raise Exception("Hull '%s' directory already exists." % path)
    os.makedirs(path)

    obj_keys = object_dataset.object_dict.keys()

    objs_p_job = 2000
    while len(obj_keys)/float(objs_p_job) > 1000:
        objs_p_job += 500
        if objs_p_job >= 4000:
            break

    print("%d objects per jobs" % objs_p_job)

    set_cnt = 0
    multi_params = []
    obj_key_list = []
    for ii, obj_key in enumerate(obj_keys):
        object_dataset.object_dict[obj_key]._path_to_hull_voxel = \
            path + "/%d.npz" % set_cnt
        obj_key_list.append(obj_key)

        if (len(obj_key_list) == objs_p_job) or (ii+1 == len(obj_keys)):
            multi_params.append([path + "/%d" % set_cnt, object_dataset_path,
                                 obj_key_list])
            obj_key_list = []
            set_cnt += 1

    if use_qsub and qsub_available:
        path_to_out = qm.QSUB_script(multi_params, "extract_and_save_all_hull_voxels", queue=queue)
    else:
        if nb_processes > 1:
            pool = Pool(nb_processes)
            pool.map(extract_and_save_all_hull_voxels_thread, multi_params)
            pool.close()
            pool.join()
        else:
            map(extract_and_save_all_hull_voxels_thread, multi_params)

    save_dataset(object_dataset)


def extract_and_save_all_hull_voxels_loop(object_dataset_paths, overwrite=False,
                                          nb_processes=1, use_qsub=False, queue="full"):
    for object_dataset_path in object_dataset_paths:
        extract_and_save_all_hull_voxels(object_dataset_path, overwrite=overwrite,
                                         nb_processes=nb_processes, use_qsub=use_qsub, queue=queue)


def check_all_hulls_thread(args):
    object_dataset = load_dataset(args[0])
    step = args[1]

    misses = []
    for key in object_dataset.object_dict.keys()[step*100000: (step+1)*100000]:
        if not isinstance(object_dataset.object_dict[key].hull_voxels, np.ndarray):
            misses.append(key)

    return misses


def check_all_hulls(object_dataset_path, nb_processes, use_qsub=False, queue="full"):
    object_dataset = load_dataset(object_dataset_path)
    multi_params = []
    for step in range(int(np.ceil(len(object_dataset.object_dict)/10000))):
        multi_params.append([object_dataset_path, step])

    if use_qsub and qsub_available:
        path_to_out = qm.QSUB_script(multi_params, "check_all_hulls", queue=queue)

        out_files = glob.glob(path_to_out + "/*")
        misses = []
        for out_file in out_files:
            with open(out_file) as f:
                misses += pickle.load(f)

    else:
        if nb_processes > 1:
            pool = Pool(nb_processes)
            results = pool.map(check_all_hulls_thread, multi_params)
            pool.close()
            pool.join()
        else:
            results = map(check_all_hulls_thread, multi_params)

        misses = []
        for result in results:
            misses += result
    return misses


def check_all_hulls_loop(object_dataset_paths, nb_processes=1, use_qsub=False, queue="full"):
    misses = {}
    for object_dataset_path in object_dataset_paths:
        misses[object_dataset_path] = check_all_hulls(object_dataset_path, nb_processes=nb_processes,
                                                      use_qsub=use_qsub, queue=queue)
        print("%s: %d" % (object_dataset_path, len(misses[object_dataset_path]))+
              misses[object_dataset_path]+
              "\n---------------------------------------------------------\n")
    return misses


def save_dataset(object_dataset, path="auto"):
    if path == "auto":
        path = object_dataset.path

    object_dataset._temp_sizes = []
    object_dataset._temp_rep_coords = []
    object_dataset._temp_ids = []
    object_dataset._temp_rep_coord_kdtree = None

    # with closing(shelve.open(path+"/shelve_object_dictionary", flag="n", protocol=2)) as sdict:
    #     for key in object_dataset.object_dict.keys():
    #         sdict[str(key)] = object_dataset.object_dict[key]

    # with open(path+"object_dictionary.pkl", "w") as file:
    #     pickle.dump(object_dataset.object_dict, file, protocol=2)
    #
    # object_dataset.object_dict = {}

    with open(path+"/object_dataset.pkl", 'wb') \
            as output:
        pickle.dump(object_dataset, output, pickle.HIGHEST_PROTOCOL)


def load_dataset(path, use_shelve=False):
    with open(path+"/object_dataset.pkl", 'rb') as input:
        this_input = pickle.load(input)

    # if use_shelve:
    #     this_input.object_dict = {}
    #     this_input.object_dict = shelve.open(path+"/shelve_object_dictionary", flag='r', protocol=2)

    if os.path.normpath(this_input.path).strip("/") != os.path.normpath(path).strip("/"):
        print("Updating paths...")
        this_input._path_to_chunk_dataset_head =""
        if path[-1] == "/":
            for part in path.split("/")[:-2]:
                this_input._path_to_chunk_dataset_head += "/" + part
        else:
             for part in path.split("/")[:-1]:
                this_input._path_to_chunk_dataset_head += "/" + part

        for key in this_input.object_dict.keys():
            this_input.object_dict[key]._path_dataset = path
            for nb_voxel_path in \
                    range(len(this_input.object_dict[key].path_to_voxel)):
                this_path = this_input.object_dict[key]._path_to_voxel[nb_voxel_path]
                voxel_path = path
                for part in this_path.split("/")[-2:]:
                    voxel_path += "/" + part

            try:
                this_path = this_input.object_dict[key]._path_to_hull_voxel
            except:
                this_path = None
            if this_path is not None:
                hull_voxel_path = path
                for part in this_path.split("/")[-2:]:
                    hull_voxel_path += "/" + part

                this_input.object_dict[key]._path_to_voxel[nb_voxel_path] = voxel_path

        print("... finished.")
    this_input._temp_sizes = []
    this_input._temp_rep_coords = []
    return this_input


def update_dataset(object_dataset, update_objects=False, recalculate_rep_coords=False, overwrite=False):
    if recalculate_rep_coords:
        update_objects = True

    up_object_dataset = UltrastructuralDataset(object_dataset._type,
                                            object_dataset._rel_path_home,
                                            object_dataset._path_to_chunk_dataset_head)
    if update_objects:
        for key in object_dataset.object_dict.keys():
            obj = object_dataset.object_dict[key]
            up_object_dataset.object_dict[key] = SegmentationObject(key,
                                                        obj._path_dataset,
                                                        obj._path_to_voxel)
            if recalculate_rep_coords:
                up_object_dataset.object_dict[key].calculate_rep_coord()
            else:
                up_object_dataset.object_dict[key]._rep_coord = obj.rep_coord
            up_object_dataset.object_dict[key]._size = obj.size
            up_object_dataset.object_dict[key]._obj_id = obj.obj_id
            try:
                up_object_dataset.object_dict[key]._path_to_hull_voxel = \
                    obj._path_to_hull_voxel
            except:
                up_object_dataset.object_dict[key]._path_to_hull_voxel = \
                    None
    else:
        up_object_dataset.object_dict = object_dataset.object_dict

    if overwrite:
        save_dataset(up_object_dataset)
    return up_object_dataset


def updating_segmentationDatasets_thread(args):
    path = args[0]
    update_objects = args[1]
    recalculate_rep_coords = args[2]

    sset = load_dataset(path)
    sset = update_dataset(sset, update_objects=update_objects,
                          recalculate_rep_coords=recalculate_rep_coords, overwrite=True)


def update_multiple_datasets(paths, update_objects=False, recalculate_rep_coords=False, nb_processes=1, use_qsub=True, queue="full"):
    multi_params = []
    for path in paths:
        multi_params.append([path, update_objects, recalculate_rep_coords])

    if use_qsub and qsub_available:
        path_to_out = qm.QSUB_script(multi_params, "updating_segmentationDatasets", queue=queue)
    else:
        if nb_processes > 1:
            pool = Pool(nb_processes)
            pool.map(updating_segmentationDatasets_thread, multi_params)
            pool.close()
            pool.join()
        else:
            map(updating_segmentationDatasets_thread, multi_params)


class UltrastructuralDataset(object):
    def __init__(self, obj_type, rel_path_home, path_to_chunk_dataset_head):
        self._type = obj_type
        self._rel_path_home = rel_path_home
        self._path_to_chunk_dataset_head = path_to_chunk_dataset_head
        self.object_dict = {}
        self._temp_sizes = []
        self._temp_rep_coords = []
        self._temp_ids = []
        self._temp_rep_coord_kdtree = None
        self._node_ids = None

    # enables efficient pickling
    def __getstate__(self):
        paths_to_voxel = []
        paths_to_hull_voxel = []
        sizes = []
        obj_ids = []
        rep_coords = []
        bounding_boxes = []
        most_distant_voxels = []

        for key, obj in self.object_dict.iteritems():
            paths_to_voxel.append(obj._path_to_voxel)
            try:
                paths_to_hull_voxel.append(obj._path_to_hull_voxel)
            except:
                paths_to_hull_voxel.append(None)
            sizes.append(obj.size)
            obj_ids.append(obj.obj_id)
            rep_coords.append(obj.rep_coord)
            bounding_boxes.append(obj._bounding_box)
            try:
                most_distant_voxels.append(obj._most_distant_voxel)
            except:
                most_distant_voxels.append(None)

        try:
            scaling = obj._scaling
        except:
            scaling = np.array([10., 10., 20.])

        return (paths_to_voxel, paths_to_hull_voxel, sizes, obj_ids, rep_coords, bounding_boxes, most_distant_voxels,
                scaling, self._type, self._rel_path_home,
                self._path_to_chunk_dataset_head)

    # enables efficient pickling
    def __setstate__(self, state):
        paths_to_voxel, paths_to_hull_voxel, sizes, obj_ids, rep_coords, bounding_boxes, most_distant_voxels, \
            scaling, self._type, self._rel_path_home, self._path_to_chunk_dataset_head = state

        self.object_dict = {}
        self._temp_sizes = []
        self._temp_rep_coords = []
        self._temp_ids = []
        self._temp_rep_coord_kdtree = None
        self._node_ids = None

        for ii in range(len(paths_to_hull_voxel)):
            obj = SegmentationObject(obj_ids[ii], self._path_to_chunk_dataset_head+self._rel_path_home,
                                     paths_to_voxel[ii])
            obj._path_to_hull_voxel = paths_to_hull_voxel[ii]
            obj._size = sizes[ii]
            obj._rep_coord = rep_coords[ii]
            obj._bounding_box = bounding_boxes[ii]
            obj._most_distant_voxel = most_distant_voxels[ii]
            obj._scaling = scaling
            self.object_dict[obj_ids[ii]] = obj

    @property
    def type(self):
        return self._type

    @property
    def path(self):
        return self._path_to_chunk_dataset_head + self._rel_path_home

    @property
    def chunk_dataset(self):
        return chunky.load_dataset(self._path_to_chunk_dataset_head)

    @property
    def knossos_dataset(self):
        return self.chunk_dataset.dataset

    @property
    def sizes(self, as_list=True):
        if self._temp_sizes == []:
            self.init_properties()
        return self._temp_sizes

    @property
    def rep_coords(self, as_list=True):
        if self._temp_rep_coords == []:
            self.init_properties()
        return self._temp_rep_coords

    @property
    def ids(self):
        if  self._temp_ids == []:
            self.init_properties()
        return self._temp_ids

    def init_properties(self):
        self._temp_ids = []
        self._temp_sizes = []
        self._temp_rep_coords = []
        for key in self.object_dict.keys():
            self._temp_ids.append(key)
            self._temp_rep_coords.append(self.object_dict[key].rep_coord)
            self._temp_sizes.append(self.object_dict[key].size)
        return

    def find_object_with_coordinate(self, coordinate, initial_radius_nm=1000,
                                    hull_search_radius_nm=500,
                                    hull_reduction_by=20,
                                    scaling=None, return_if_contained=True):
        if hull_reduction_by < 1:
            hull_reduction_by = 1
        hull_reduction_by = int(hull_reduction_by)

        if scaling is None:
            scaling =np.array([9., 9., 21.])
        else:
            scaling = np.array(scaling)

        coordinate = np.array(coordinate)
        if self._temp_rep_coord_kdtree is None:
            self._temp_rep_coord_kdtree = spatial.cKDTree(np.array(self.rep_coords)*scaling)
        candidate_indices = \
            self._temp_rep_coord_kdtree.query_ball_point(coordinate*scaling,
                                                         initial_radius_nm)
        objs_ids = []
        obj_dist_list = []
        for index in candidate_indices:
            obj = self.object_dict[self.ids[index]]
            objs_ids.append(obj.obj_id)
            if obj.check_if_coordinate_is_contained(coordinate):
                if return_if_contained:
                    return [obj, 0]
                else:
                    obj_dist_list.append([obj, 0])

        if hull_search_radius_nm > 0 and len(candidate_indices) > 0:
            hull_coordinates = []
            length_list = []
            for index in candidate_indices:
                obj = self.object_dict[self.ids[index]]
                obj_hull_vx = obj.hull_voxels[::hull_reduction_by]
                hull_coordinates += obj_hull_vx.tolist()
                length_list.append(len(obj_hull_vx.tolist()))

            hull_coordinates = np.array(hull_coordinates)
            hull_tree = spatial.cKDTree(hull_coordinates*scaling)
            closest_hull_indices = hull_tree.query_ball_point(coordinate*scaling,
                                                              hull_search_radius_nm)

            distances = np.ones(len(hull_coordinates))*np.inf
            distances[closest_hull_indices] = \
                spatial.distance.cdist([coordinate*scaling],
                                       hull_coordinates[closest_hull_indices]*scaling)

            distance_list = []
            for ii, index in enumerate(candidate_indices):
                cur_pos = int(np.sum(length_list[:(ii)]))
                this_distances = distances[cur_pos: (cur_pos+length_list[ii])]
                closest = np.min(this_distances)
                distance_list.append([self.ids[index], closest])

            distance_list = np.array(distance_list)
            distance_list = distance_list[distance_list[:, 1].argsort()].tolist()
            for entry in distance_list:
                obj_dist_list.append([self.object_dict[entry[0]], entry[1]])
            return obj_dist_list
        else:
            return None


class SegmentationObject(object):
    def __init__(self, obj_id, path_dataset, path_to_voxels):
        self._path_to_voxel = path_to_voxels
        self._path_to_hull_voxel = None
        self._size = None
        self._obj_id = obj_id
        self._rep_coord = None
        self._bounding_box = None
        self._path_dataset = path_dataset
        self._most_distant_voxel = None

    @property
    def path_dataset(self):
        return self._path_dataset

    @property
    def bounding_box(self):
        if self._bounding_box is None:
            min = np.min(self.voxels, axis=0)
            max = np.max(self.voxels, axis=0)
            self._bounding_box = [min, max]
        return self._bounding_box

    @property
    def voxels(self):
        this_voxels = []
        for path in self.path_to_voxel:
            new_voxels = np.load(path)[str(self.obj_id)]
            if len(this_voxels) == 0:
                this_voxels = np.copy(new_voxels)
            else:
                this_voxels = np.concatenate((this_voxels,
                                              new_voxels))
        return this_voxels

    @property
    def hull_voxels(self):
        if self.path_to_hull_voxels is not None:
            try:
                return np.load(self.path_to_hull_voxels)[str(self.obj_id)]
            except:
                # print "Hull Voxels need to be calculated"
                return self.create_hull_voxels()
        else:
            # print "Hull Voxels need to be calculated"
            return self.create_hull_voxels()

    @property
    def size(self):
        if self._size is None:
            self.calculate_size()
        return self._size

    @property
    def obj_id(self):
        return self._obj_id

    @property
    def rep_coord(self):
        return self._rep_coord

    @property
    def path_to_hull_voxels(self):
        return self._path_to_hull_voxel

    @property
    def path_to_voxel(self):
        if isinstance(self._path_to_voxel, basestring):
            self._path_to_voxel = [self._path_to_voxel]
        return self._path_to_voxel

    @property
    def scaling(self):
        raise NotImplementedError()

    @property
    def most_distant_voxel(self):
        if self._most_distant_voxel is None:
            voxel_distances = spatial.distance.cdist([self.rep_coord*self.scaling],
                                                      self.voxels*self.scaling)
            voxel_distances = np.array(voxel_distances)
            self._most_distant_voxel = self.voxels[np.argmax(voxel_distances == np.max(voxel_distances))]
        return self._most_distant_voxel

    def closest_point(self, coordinate, scaling=None):
        if scaling is None:
            scaling = np.array([10., 10., 10.])
        else:
            scaling = np.array(scaling)

        if self.path_to_hull_voxels is None:
            vx = np.array(self.hull_voxels)*scaling
        else:
            vx = np.array(self.voxels)*scaling

        distances = spatial.distance.cdist([np.array(coordinate)*scaling], vx)
        return vx[distances == np.min(distances)][0]

    def check_if_coordinate_is_contained(self, coordinate):
        a = self.voxels
        b = np.array(coordinate)
        return (a == b).all(np.arange(a.ndim - b.ndim, a.ndim)).any()

    def add_path(self, other_obj):
        for path in other_obj._path_to_voxel:
            self._path_to_voxel.append(path)
        self._size = None

    def write_to_overlaycube(self, kd, mags=None, size_filter=0, entry=None):
        self.write_to_cube(kd, mags, size_filter, entry)

    def write_to_cube(self, kd, mags=None, size_filter=0, entry=None,
                      as_raw=False, overwrite=True):
        if mags is None:
            mags = kd.mag
        this_bb = np.copy(self.bounding_box)
        this_bb[1] += 1
        if not overwrite:
            for dim in range(3):
                if this_bb[0][dim] % 128 != 0:
                    this_bb[0][dim] -= this_bb[0][dim] % 128
                if this_bb[1][dim] % 128 != 0:
                    this_bb[1][dim] -= this_bb[0][dim] % 128 + 128
        bb_size = this_bb[1]-this_bb[0]

        if as_raw:
            datatype = np.uint8
        else:
            datatype = np.uint32

        if self.size > size_filter:
            voxels = np.array(np.array(self.voxels) - this_bb[0],
                              dtype=np.uint32)
            if np.ndim(voxels) != 2:
                print("No voxels found in %s. Shape: %s" % (self.type, str(voxels.shape)))
                return
            else:
                if overwrite:
                    matrix = np.zeros((np.max(voxels[:, 0])+1, np.max(voxels[:, 1])+1,
                                       np.max(voxels[:, 2])+1), dtype=datatype)
                else:
                    if as_raw:
                        matrix = kd.from_raw_cubes_to_matrix(bb_size, this_bb[0])
                    else:
                        matrix = kd.from_overlay_cubes_to_matrix(bb_size, this_bb[0])

            if entry is None:
                matrix[voxels[:, 0], voxels[:, 1], voxels[:, 2]] = self.obj_id
            else:
                matrix[voxels[:, 0], voxels[:, 1], voxels[:, 2]] = entry
            kd.from_matrix_to_cubes(this_bb[0], data=[matrix], nb_threads=1, mags=mags, overwrite=not overwrite)

    def create_hull_voxels(self):
        voxels = np.copy(self.voxels)
        if len(voxels.shape) > 1:
            voxels_array = np.array(voxels, dtype=np.int)
            x_min = np.min(voxels_array[:, 0]) - 2
            x_max = np.max(voxels_array[:, 0]) + 2
            y_min = np.min(voxels_array[:, 1]) - 2
            y_max = np.max(voxels_array[:, 1]) + 2
            z_min = np.min(voxels_array[:, 2]) - 2
            z_max = np.max(voxels_array[:, 2]) + 2

            matrix = np.zeros((x_max-x_min, y_max-y_min, z_max-z_min),
                              dtype=np.uint8)

            lower_boarder = np.array([basics.negative_to_zero(x_min),
                                      basics.negative_to_zero(y_min),
                                      basics.negative_to_zero(z_min)],
                                     dtype=np.int)

            voxels = np.array(voxels, dtype=np.int) - lower_boarder
            matrix[voxels[:, 0], voxels[:, 1], voxels[:, 2]] = 1

            k = np.array([[[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                          [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
                          [[0, 0, 0], [0, 1, 0], [0, 0, 0]]])
            coords = np.argwhere((ndimage.convolve(matrix, k, mode="constant", cval=0.) < 7)*matrix == 1) + lower_boarder
        else:
            coords = voxels
        return coords

    def create_hull_ids(self):
        voxels = np.copy(self.voxels)
        if len(voxels.shape) > 1:
            voxels_array = np.array(voxels, dtype=np.int)
            # print voxels_array.shape
            x_min = np.min(voxels_array[:, 0]) - 2
            x_max = np.max(voxels_array[:, 0]) + 2
            y_min = np.min(voxels_array[:, 1]) - 2
            y_max = np.max(voxels_array[:, 1]) + 2
            z_min = np.min(voxels_array[:, 2]) - 2
            z_max = np.max(voxels_array[:, 2]) + 2

            matrix = np.zeros((x_max-x_min, y_max-y_min, z_max-z_min),
                              dtype=np.uint8)

            lower_boarder = np.array([basics.negative_to_zero(x_min),
                                      basics.negative_to_zero(y_min),
                                      basics.negative_to_zero(z_min)],
                                     dtype=np.int)

            voxels = np.array(voxels, dtype=np.int) - lower_boarder
            matrix[voxels[:, 0], voxels[:, 1], voxels[:, 2]] = 1

            hull_voxel_ids = []
            for nb_voxel in range(len(voxels)):
                voxel = np.array(voxels[nb_voxel], dtype=np.int)
                if matrix[voxel[0]-1, voxel[1], voxel[2]] != 1:
                    hull_voxel_ids.append(nb_voxel)
                elif matrix[voxel[0]+1, voxel[1], voxel[2]] != 1:
                    hull_voxel_ids.append(nb_voxel)
                elif matrix[voxel[0], voxel[1]-1, voxel[2]] != 1:
                    hull_voxel_ids.append(nb_voxel)
                elif matrix[voxel[0], voxel[1]+1, voxel[2]] != 1:
                    hull_voxel_ids.append(nb_voxel)
                elif matrix[voxel[0], voxel[1], voxel[2]-1] != 1:
                    hull_voxel_ids.append(nb_voxel)
                elif matrix[voxel[0], voxel[1], voxel[2]+1] != 1:
                    hull_voxel_ids.append(nb_voxel)
        else:
            hull_voxel_ids = [0]

        return hull_voxel_ids

    def add_voxel(self, coordinate):
       raise NotImplementedError

    def remove_voxel_file(self):
        raise NotImplementedError()
        # for path in self.path_to_voxel:
        #     os.remove(path)

    def calculate_rep_coord(self, calculate_size=False, voxels=None, sample_size=200):
        if voxels is None:
            voxels = self.voxels

        if len(voxels) > 1:
            # self._rep_coord = np.array([np.mean(voxels[:, 0]),
            #                             np.mean(voxels[:, 1]),
            #                             np.mean(voxels[:, 2])],
            #                            dtype=np.int)
            np.random.shuffle(voxels)
            dist = spatial.distance.cdist(voxels[:sample_size], voxels[:sample_size])
            sum = np.sum(dist, 1)
            pos = np.where(sum == np.min(sum))[0][0]
            self._rep_coord = np.array(voxels[:sample_size][pos])

            # if not self.check_if_coordinate_is_contained(self._rep_coord):
            #     if fast_correction:
            #         self._rep_coord = np.array(voxels[int(len(voxels)/2)], dtype=np.int)
            #     else:
            #         step = len(voxels)/300
            #         if step == 0:
            #             step = 1
            #         dist = spatial.distance.cdist(voxels[::step], voxels[::step])
            #         sum = np.sum(dist, 1)
            #         pos = np.where(sum == np.min(sum))[0]
            #         self._rep_coord = np.array(voxels[::step][pos])
        else:
            self._rep_coord = voxels[0]

        if calculate_size:
            self.calculate_size(voxels=voxels)

    def calculate_size(self, voxels=None):
        if voxels is None:
            voxels = self.voxels

        if len(voxels.shape) > 1:
            self._size = len(voxels)
        else:
            self._size = 1
