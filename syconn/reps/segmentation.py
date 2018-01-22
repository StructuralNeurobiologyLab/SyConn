# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max-Planck-Institute for Medical Research, Heidelberg, Germany
# Authors: Sven Dorkenwald, Philipp Schubert, Joergen Kornfeld
import errno
import os
import re
import shutil
import networkx as nx

from scipy import ndimage, spatial
from knossos_utils import knossosdataset
script_folder = os.path.abspath(os.path.dirname(__file__) + "/../QSUB_scripts/")
try:
    default_wd_available = True
    from ..config.global_params import wd
except:
    default_wd_available = False
from ..config import parser
from ..config.global_params import MESH_DOWNSAMPLING
from ..handler.compression import LZ4Dict
from ..handler.basics import load_pkl2obj, write_obj2pkl
from .rep_helper import subfold_from_ix, surface_samples, knossos_ml_from_svixs
from ..handler.basics import get_filepaths_from_dir, safe_copy, write_txt2kzip
import warnings
from .segmentation_helper import *
from ..proc import meshs
from skimage.measure import mesh_surface_area


class SegmentationDataset(object):
    def __init__(self, obj_type, version=None, working_dir=None, scaling=None,
                 version_dict=None, create=False, config=None,
                 n_folders_fs=None):
        """ Dataset Initialization

        :param obj_type: str
            type of objects; usually one of: vc, sj, mi, cs, sv
        :param version: str || int
            version of dataset to distinguish it from others of the same type
        :param working_dir: str
            path to working directory
        :param scaling: list || array of three ints
            scaling of the raw data to nanometer
        :param version_dict: dict
            versions of datasets of other types that correspond with this dataset
        :param create: bool
            whether or not to create this dataset on disk
        :param config: str
            content of configuration file
        :param n_folders: int

        """

        self._type = obj_type
        self.object_dict = {}

        self._n_folders_fs = n_folders_fs

        self._sizes = None
        self._ids = None
        self._rep_coords = None
        self._config = config

        if n_folders_fs is not None:
            if not n_folders_fs in [10**i for i in range(6)]:
                raise Exception("n_folders_fs must be in",
                                [10**i for i in range(6)])

        if working_dir is None:
            if default_wd_available:
                self._working_dir = wd
            else:
                raise Exception("No working directory (wd) specified in config")
        else:
            self._working_dir = working_dir

        if not self._working_dir.endswith("/"):
            self._working_dir += "/"

        # self._config = parser.Config(self.working_dir)

        self._scaling = scaling

        if version is None:
            try:
                self._version = self.config.entries["Versions"][self.type]
            except:
                raise Exception("unclear value for version")
        elif version == "new":
            other_datasets = \
                glob.glob(self.working_dir + "/%s_[0-9]" % self.type) + \
                glob.glob(self.working_dir + "/%s_[0-9][0-9]" % self.type) + \
                glob.glob(self.working_dir + "/%s_[0-9][0-9][0-9]" % self.type)

            max_version = -1

            for other_dataset in other_datasets:
                other_version = \
                    int(re.findall("[\d]+",
                                   os.path.basename(other_dataset.strip('/')))[-1])
                if max_version < other_version:
                    max_version = other_version

            self._version = max_version + 1
        else:
            self._version = version

        if version_dict is None:
            try:
                self.version_dict = self.config.entries["Versions"]
            except:
                raise Exception("No version dict specified in config")
        else:
            if isinstance(version_dict, dict):
                self.version_dict = version_dict
            elif isinstance(version_dict, str) and version_dict == "load":
                if self.version_dict_exists:
                    self.load_version_dict()
            else:
                raise Exception("No version dict specified in config")

        if create and not os.path.exists(self.path):
            os.makedirs(self.path)

    @property
    def type(self):
        return self._type

    @property
    def n_folders_fs(self):
        if self._n_folders_fs is None:
            ps = glob.glob("%s/%s*/" % (self.path, self.so_storage_path_base))
            if len(ps) == 0:
                raise Exception("No storage folder found and no number of "
                                "subfolders specified (n_folders_fs))")


            bp = os.path.basename(ps[0].strip('/'))
            for p in ps:
                bp = os.path.basename(p.strip('/'))
                if bp == self.so_storage_path_base:
                    bp = os.path.basename(p.strip('/'))
                    break

            if bp == self.so_storage_path_base:
                self._n_folders_fs = 100000
            else:
                self._n_folders_fs = int(re.findall('[\d]+', bp)[-1])

        return self._n_folders_fs

    @property
    def working_dir(self):
        return self._working_dir

    @property
    def version(self):
        return str(self._version)

    @property
    def path(self):
        return "%s/%s_%s/" % (self._working_dir, self.type, self.version)

    @property
    def path_sizes(self):
        return self.path + "/sizes.npy"

    @property
    def path_rep_coords(self):
        return self.path + "/rep_coords.npy"

    @property
    def path_ids(self):
        return self.path + "/ids.npy"

    @property
    def version_dict_path(self):
        return self.path + "/version_dict.pkl"

    @property
    def version_dict_exists(self):
        return os.path.exists(self.version_dict_path)

    @property
    def so_storage_path_base(self):
        return "so_storage"

    @property
    def so_storage_path(self):
        if self._n_folders_fs is None and os.path.exists("%s/so_storage/" % self.path):
            return "%s/so_storage/" % self.path
        elif self._n_folders_fs == 100000 and os.path.exists("%s/so_storage/" % self.path):
            return "%s/so_storage/" % self.path
        else:
            return "%s/%s_%d/" % (self.path, self.so_storage_path_base,
                                  self.n_folders_fs)

    @property
    def so_dir_paths(self):
        depth = int(np.log10(self.n_folders_fs) // 2 + np.log10(self.n_folders_fs) % 2)
        p = "".join([self.so_storage_path] + ["/*" for _ in range(depth)])

        return glob.glob(p)

    @property
    def config(self):
        if self._config is None:
            self._config = parser.Config(self.working_dir)
        return self._config

    @property
    def sizes(self):
        if self._sizes is None:
            if os.path.exists(self.path_sizes):
                self._sizes = np.load(self.path_sizes)
            else:
                print("sizes were not calculated...")
        return self._sizes

    @property
    def rep_coords(self):
        if self._rep_coords is None:
            if os.path.exists(self.path_rep_coords):
                self._rep_coords = np.load(self.path_rep_coords)
            else:
                print("rep coords were not calculated...")
        return self._rep_coords

    @property
    def ids(self):
        if self._ids is None:
            acquire_obj_ids(self)
        return self._ids

    @property
    def scaling(self):
        if self._scaling is None:
            try:
                self._scaling = \
                    np.array(self.config.entries["Dataset"]["scaling"])
            except:
                self._scaling = np.array([1, 1, 1])

        return self._scaling

    @property
    def sos(self):
        ix = 0
        tot_nb_sos = len(self.ids)
        while ix < tot_nb_sos:
            yield self.get_segmentation_object(self.ids[ix])
            ix += 1

    def load_cached_data(self, name):
        if os.path.exists(self.path + name + "s.npy"):
            return np.load(self.path + name + "s.npy")

    def get_segmentationdataset(self, obj_type):
        assert obj_type in self.version_dict
        return SegmentationDataset(obj_type,
                                   version=self.version_dict[obj_type],
                                   working_dir=self.working_dir)

    def get_segmentation_object(self, obj_id, create=False):
        return SegmentationObject(obj_id=obj_id,
                                  obj_type=self.type,
                                  version=self.version,
                                  working_dir=self.working_dir,
                                  scaling=self.scaling,
                                  create=create,
                                  n_folders_fs=self.n_folders_fs)

    def save_version_dict(self):
        write_obj2pkl(self.version_dict_path, self.version_dict)

    def load_version_dict(self):
        assert self.version_dict_exists
        self.version_dict = load_pkl2obj(self.version_dict_path)


class SegmentationObject(object):
    def __init__(self, obj_id, obj_type="sv", version=None, working_dir=None,
                 rep_coord=None, size=None, scaling=(10, 10, 20), create=False,
                 voxel_caching=True, mesh_caching=False, view_caching=False,
                 config=None, n_folders_fs=None, enable_locking=True,
                 skeleton_caching=True):
        self._id = int(obj_id)
        self._type = obj_type
        self._rep_coord = rep_coord
        self._size = size
        self._n_folders_fs = n_folders_fs

        self.attr_dict = {}
        self._bounding_box = None
        self._paths_to_voxels = None
        self.enable_locking = enable_locking

        self._voxel_caching = voxel_caching
        self._mesh_caching = mesh_caching
        self._mesh_bb = None
        self._view_caching = view_caching
        self._voxels = None
        self._voxel_list = None
        self._mesh = None
        self._config = config
        self._views = None
        self._skeleton = None
        self._skeleton_caching = skeleton_caching

        if working_dir is None:
            if default_wd_available:
                self._working_dir = wd
            else:
                raise Exception("No working directory (wd) specified in config")
        else:
            self._working_dir = working_dir

        self._scaling = scaling

        if version is None:
            try:
                self._version = self.config.entries["Versions"][self.type]
            except:
                raise Exception("unclear value for version")
        else:
            self._version = version

        if create:
            try:
                os.makedirs(self.segobj_dir)
            except OSError as exc:  # Python >2.5
                if exc.errno == errno.EEXIST and os.path.isdir(self.segobj_dir):
                    pass
                else:
                    raise Exception(exc)

    #                                                       IMMEDIATE PARAMETERS
    def __hash__(self):
        return hash((self.id, self.type.__hash__()))

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return self.id == other.id and self.type == other.type

    def __ne__(self, other):
        return not self.__eq__(other)

    @property
    def type(self):
        return self._type

    @property
    def n_folders_fs(self):
        if self._n_folders_fs is None:
            ps = glob.glob(
                "%s/%s*/" % (self.segds_dir, self.so_storage_path_base))
            if len(ps) == 0:
                raise Exception("No storage folder found and no number of "
                                "subfolders specified (n_folders_fs))")

            bp = os.path.basename(ps[0].strip('/'))
            for p in ps:
                bp = os.path.basename(p.strip('/'))
                if bp == self.so_storage_path_base:
                    bp = os.path.basename(p.strip('/'))
                    break

            if bp == self.so_storage_path_base:
                self._n_folders_fs = 100000
            else:
                self._n_folders_fs = int(re.findall('[\d]+', bp)[-1])

        return self._n_folders_fs

    @property
    def id(self):
        return self._id

    @property
    def version(self):
        return str(self._version)

    @property
    def voxel_caching(self):
        return self._voxel_caching

    @property
    def mesh_caching(self):
        return self._mesh_caching

    @property
    def skeleton_caching(self):
        return self._skeleton_caching

    @property
    def view_caching(self):
        return self._view_caching

    @property
    def scaling(self):
        if self._scaling is None:
            try:
                self._scaling = \
                    np.array(self.config.entries["Dataset"]["scaling"])
            except:
                self._scaling = np.array([1, 1, 1])

        return self._scaling

    @property
    def dataset(self):
        return SegmentationDataset(self.type, self.version, self._working_dir)

    @property
    def config(self):
        if self._config is None:
            self._config = parser.Config(self.working_dir)
        return self._config

    #                                                                      PATHS

    @property
    def working_dir(self):
        return self._working_dir

    @property
    def identifier(self):
        return "%s_%s" % (self.type, self.version.lstrip("_"))

    @property
    def segds_dir(self):
        return "%s/%s/" % (self.working_dir, self.identifier)

    @property
    def so_storage_path_base(self):
        return "so_storage"

    @property
    def so_storage_path(self):
        if self._n_folders_fs is None and os.path.exists("%s/%s/" % (self.segds_dir, self.so_storage_path_base)):
            return "%s/%s/" % (self.segds_dir, self.so_storage_path_base)
        elif self._n_folders_fs == 100000 and os.path.exists("%s/%s/" % (self.segds_dir, self.so_storage_path_base)):
            return "%s/%s/" % (self.segds_dir, self.so_storage_path_base)
        else:
            return "%s/%s_%d/" % (self.segds_dir, self.so_storage_path_base,
                                  self.n_folders_fs)

    # @property
    # def segobj_dir(self):
    #     return "%s/so_storage/%s/" % (self.segds_dir,
    #                                   subfold_from_ix(self.id))

    @property
    def segobj_dir(self):
        if os.path.exists("%s/%s/voxel.pkl" % (self.so_storage_path,
                                      subfold_from_ix(self.id, self.n_folders_fs))):
            return "%s/%s/" % (self.so_storage_path,
                               subfold_from_ix(self.id, self.n_folders_fs))
        else:
            return "%s/%s/" % (self.so_storage_path,
                               subfold_from_ix(self.id, self.n_folders_fs, old_version=True))

    @property
    def mesh_path(self):
        return self.segobj_dir + "mesh.pkl"

    @property
    def skeleton_path(self):
        return self.segobj_dir + "skeletons.pkl"

    @property
    def attr_dict_path(self):
        return self.segobj_dir + "attr_dict.pkl"

    def view_path(self, woglia=True):
        if woglia:
            return self.segobj_dir + "views_woglia.pkl"
        return self.segobj_dir + "views.pkl"

    @property
    def locations_path(self):
        return self.segobj_dir + "locations.pkl"

    @property
    def voxel_path(self):
        return self.segobj_dir + "/voxel.pkl"

    #                                                                 PROPERTIES
    @property
    def cs_partner(self):
        if "cs" in self.type:
            partner = [self.id >> 32]
            partner.append(self.id - (partner[0] << 32))
            return partner
        else:
            return None

    @property
    def size(self):
        if self._size is None and self.attr_dict_exists:
            self._size = self.lookup_in_attribute_dict("size")

        if self._size is None:
            self.calculate_size()

        return self._size

    @property
    def shape(self):
        return self.bounding_box[1] - self.bounding_box[0]

    @property
    def bounding_box(self):
        if self._bounding_box is None and self.attr_dict_exists:
            self._bounding_box = self.lookup_in_attribute_dict("bounding_box")

        if self._bounding_box is None:
            self.calculate_bounding_box()

        return self._bounding_box

    @property
    def rep_coord(self):
        if self._rep_coord is None and self.attr_dict_exists:
            self._rep_coord = self.lookup_in_attribute_dict("rep_coord")

        if self._rep_coord is None:
            self.calculate_rep_coord()

        return self._rep_coord

    @property
    def attr_dict_exists(self):
        if not os.path.isfile(self.attr_dict_path):
            return False
        glob_attr_dc = AttributeDict(self.attr_dict_path,
                                     disable_locking=not self.enable_locking)
        return self.id in glob_attr_dc

    @property
    def voxels_exist(self):
        voxel_dc = VoxelDict(self.voxel_path, read_only=True,
                             disable_locking=True)
        return self.id in voxel_dc


    @property
    def voxels(self):
        if self._voxels is None:
            if self.voxel_caching:
                self._voxels = load_voxels(self)
                return self._voxels
            else:
                return load_voxels(self)
        else:
            return self._voxels

    @property
    def voxel_list(self):
        if self._voxel_list is None:
            if self.voxel_caching:
                self._voxel_list = load_voxel_list(self)
                return self._voxel_list
            else:
                return load_voxel_list(self)
        else:
            return self._voxel_list

    @property
    def mesh_exists(self):
        mesh_dc = MeshDict(self.mesh_path,
                           disable_locking=not self.enable_locking)
        return self.id in mesh_dc

    @property
    def skeleton_exists(self):
        skeleton_dc = SkeletonDict(self.skeleton_path,
                           disable_locking=not self.enable_locking)
        return self.id in skeleton_dc

    @property
    def mesh(self):
        if self._mesh is None:
            if self.mesh_caching:
                self._mesh = load_mesh(self)
                return self._mesh
            else:
                return load_mesh(self)
        else:
            return self._mesh

    @property
    def skeleton(self):
        if self._skeleton is None:
            if self.skeleton_caching:
                self._skeleton = load_skeleton(self)
                return self._skeleton
            else:
                return load_skeleton(self)
        else:
            return self._skeleton

    @property
    def mesh_bb(self):
        if self._mesh_bb is None:
            self._mesh_bb = [np.min(self.mesh[1].reshape((-1, 3)), axis=0),
                             np.max(self.mesh[1].reshape((-1, 3)), axis=0)]
        return self._mesh_bb

    @property
    def mesh_size(self):
        return np.linalg.norm(self.mesh_bb[1] - self.mesh_bb[0], ord=2)

    @property
    def mesh_area(self):
        """

        Returns
        -------
        float
            Mesh area in um^2
        """
        return mesh_surface_area(self.mesh[1].reshape(-1, 3),
                                 self.mesh[0].reshape(-1, 3)) / 1e6

    @property
    def sample_locations_exist(self):
        location_dc = LZ4Dict(self.locations_path,
                              disable_locking=not self.enable_locking)
        return self.id in location_dc

    @property
    def views_exist(self, woglia=True):
        view_dc = LZ4Dict(self.view_path(woglia=woglia),
                          disable_locking=not self.enable_locking)
        return self.id in view_dc

    @property
    def views(self):
        assert self.type == "sv"
        if self._views is None:
            if self.views_exist:
                if self.view_caching:
                    self._views = self.load_views()
                    return self._views
                else:
                    return self.load_views()
            else:
                return -1
        else:
            return self._views

    def sample_locations(self, force=False):
        assert self.type == "sv"
        if self.sample_locations_exist and not force:
            return LZ4Dict(self.locations_path,
                           disable_locking=not self.enable_locking)[self.id]
        else:
            coords = surface_samples(self.mesh[1].reshape(-1, 3))
            loc_dc = LZ4Dict(self.locations_path, read_only=False,
                             disable_locking=not self.enable_locking)
            loc_dc[self.id] = coords.astype(np.float32)
            loc_dc.save2pkl()
            return coords.astype(np.float32)

    def save_voxels(self, bin_arr, offset):
        save_voxels(self, bin_arr, offset)

    def load_voxels(self, voxel_dc=None):
        return load_voxels(self, voxel_dc=voxel_dc)

    def load_voxels_downsampled(self, downsampling=(2, 2, 1)):
        return load_voxels_downsampled(self, downsampling=downsampling)

    def load_voxel_list(self):
        return load_voxel_list(self)

    def load_voxel_list_downsampled(self, downsampling=(2, 2, 1)):
        return load_voxel_list_downsampled(self, downsampling=downsampling)

    def load_mesh(self, recompute=False):
        return load_mesh(self, recompute=recompute)

    def load_skeleton(self, recompute=False):
        return load_skeleton(self, recompute=recompute)

    def glia_pred(self, thresh=0.168, pred_key_appendix=""):
        return glia_pred_so(self, thresh, pred_key_appendix)

    def axoness_preds(self, pred_key_appendix=""):
        assert self.type == "sv"
        pred_key = "axoness_proba" + pred_key_appendix
        if not pred_key in self.attr_dict:
            self.load_attr_dict()
        pred = np.argmax(self.attr_dict[pred_key], axis=1)
        return pred

    def axoness_probas(self, pred_key_appendix=""):
        assert self.type == "sv"
        pred_key = "axoness_proba" + pred_key_appendix
        if not pred_key in self.attr_dict:
            self.load_attr_dict()
        return self.attr_dict[pred_key]

    #                                                                  FUNCTIONS

    def extent(self):
        return np.linalg.norm(self.shape * self.scaling)

    def _mesh_from_scratch(self):
        return meshs.get_object_mesh(self, MESH_DOWNSAMPLING[self.type])

    def _save_mesh(self, ind, vert, normals):
        mesh_dc = MeshDict(self.mesh_path, read_only=False,
                           disable_locking=not self.enable_locking)
        mesh_dc[self.id] = [ind, vert, normals]
        mesh_dc.save2pkl()

    def mesh2kzip(self, dest_path, ext_color=None, ply_name=""):
        mesh = self.mesh
        if self.type == "sv":
            color = (130, 130, 130, 160)
        elif self.type == "cs":
            color = (100, 200, 30, 255)
        elif self.type == "sj":
            color = (int(0.849 * 255), int(0.138 * 255), int(0.133 * 255), 255)
        elif self.type == "vc":
            color = (int(0.175 * 255), int(0.585 * 255), int(0.301 * 255), 255)
        elif self.type == "mi":
            color = (0, 153, 255, 255)
        elif ext_color is not None:
            if ext_color == 0:
                color = None
            else:
                color = ext_color
        else:
            raise ("Given object type '%s' does not exist." % self.type,
                   TypeError)

        if ply_name == "":
            ply_name = str(self.id)
        meshs.write_mesh2kzip(dest_path, mesh[0], mesh[1], mesh[2], color,
                              ply_fname=ply_name + ".ply")

    def mergelist2kzip(self, dest_path):
        self.load_attr_dict()
        kml = knossos_ml_from_svixs([self.id], coords=[self.rep_coord])
        write_txt2kzip(dest_path, kml, "mergelist.txt")

    def load_views(self, woglia=True, raw_only=False):
        view_dc = LZ4Dict(self.view_path(woglia=woglia),
                          disable_locking=not self.enable_locking)
        views = view_dc[self.id]
        if raw_only:
            views = views[:, :1]
        return np.array(views, dtype=np.float32)

    def save_views(self, views, woglia=True, cellobjects_only=False):
        view_dc = LZ4Dict(self.view_path(woglia=woglia), read_only=False,
                          disable_locking=not self.enable_locking)
        if cellobjects_only:
            assert self.id in view_dc, "SV must already contain raw views " \
                                       "if adding views for cellobjects only."
            view_dc[self.id] = np.concatenate([view_dc[self.id][:, :1], views],
                                              axis=1)
        else:
            view_dc[self.id] = views
        view_dc.save2pkl()

    def load_attr_dict(self):
        try:
             glob_attr_dc = AttributeDict(self.attr_dict_path,
                                          disable_locking=not self.enable_locking)
             self.attr_dict = glob_attr_dc[self.id]
        except (IOError, EOFError):
            return -1  #

    def save_attr_dict(self):
        glob_attr_dc = AttributeDict(self.attr_dict_path, read_only=False,
                                     disable_locking=not self.enable_locking)
        if self.id in glob_attr_dc:
            orig_dc = glob_attr_dc[self.id]
            orig_dc.update(self.attr_dict)
        else:
            orig_dc = self.attr_dict
        glob_attr_dc[self.id] = orig_dc
        glob_attr_dc.save2pkl()

    def save_attributes(self, attr_keys, attr_values):
        """
        Writes attributes to attribute dict on file system. Does not care about
        self.attr_dict.

        Parameters
        ----------
        attr_keys : tuple of str
        attr_values : tuple of items
        """

        if not hasattr(attr_keys, "__len__"):
            attr_keys = [attr_keys]
        if not hasattr(attr_values, "__len__"):
            attr_values = [attr_values]
        assert len(attr_keys) == len(attr_values), "Key-value lengths did not" \
                                                   " agree while saving attri" \
                                                   "butes of SSO %d." % self.id
        glob_attr_dc = AttributeDict(self.attr_dict_path, read_only=False,
                                     disable_locking=not self.enable_locking)
        for k, v in zip(attr_keys, attr_values):
            glob_attr_dc[self.id][k] = v
        glob_attr_dc.save2pkl()

    def attr_exists(self, attr_key):
        if len(self.attr_dict) == 0:
            self.load_attr_dict()
        try:
            _ = self.attr_dict[attr_key]
        except (KeyError, EOFError):
            return False
        return True

    def lookup_in_attribute_dict(self, attr_key):
        if len(self.attr_dict) == 0:
            self.load_attr_dict()
        if self.attr_exists(attr_key):
            return self.attr_dict[attr_key]
        else:
            return None

    def calculate_rep_coord(self, voxel_dc=None, fast=False):
        if voxel_dc is None:
           voxel_dc = VoxelDict(self.voxel_path, read_only=True,
                                disable_locking=not self.enable_locking)

        if not self.id in voxel_dc:
            self._bounding_box = np.array([[-1, -1, -1], [-1, -1, -1]])
            print("No voxels found in VoxelDict!")
            return

        bin_arrs, block_offsets = voxel_dc[self.id]
        block_offsets = np.array(block_offsets)

        if len(bin_arrs) > 1:
            sizes = []
            for i_bin_arr in range(len(bin_arrs)):
                sizes.append(np.sum(bin_arrs[i_bin_arr]))

            self._size = np.sum(sizes)
            block_offsets = np.array(block_offsets)
            sizes = np.array(sizes)
            center_of_gravity = [np.mean(block_offsets[:, 0] * sizes) / self.size,
                                 np.mean(block_offsets[:, 1] * sizes) / self.size,
                                 np.mean(block_offsets[:, 2] * sizes) / self.size]
            center_of_gravity = np.array(center_of_gravity)

            dists = spatial.distance.cdist(block_offsets,
                                           np.array([center_of_gravity]))

            central_block_id = np.argmin(dists)
        else:
            central_block_id = 0

        vx = bin_arrs[central_block_id].copy()
        central_block_offset = block_offsets[central_block_id]

        vx = ndimage.morphology.distance_transform_edt(
            np.pad(vx, 1, mode="constant", constant_values=0))[1:-1, 1:-1, 1:-1]

        max_locs = np.where(vx == vx.max())

        max_loc_id = int(len(max_locs[0]) / 2)
        max_loc = np.array([max_locs[0][max_loc_id],
                            max_locs[1][max_loc_id],
                            max_locs[2][max_loc_id]])

        if not fast:
            vx = ndimage.gaussian_filter(vx, sigma=[15, 15, 7])
            max_locs = np.where(vx == vx.max())

            max_loc_id = int(len(max_locs[0]) / 2)
            better_loc = np.array([max_locs[0][max_loc_id],
                                   max_locs[1][max_loc_id],
                                   max_locs[2][max_loc_id]])

            if bin_arrs[central_block_id][better_loc[0], better_loc[1], better_loc[2]]:
                max_loc = better_loc

        self._rep_coord = max_loc + central_block_offset

    def calculate_bounding_box(self):
        _ = load_voxels(self)

    def calculate_size(self):
        _ = load_voxels(self)

    def save_kzip(self, path, kd=None, write_id=None):
        if write_id is None:
            write_id = self.id

        if kd is None:
            try:
                kd = knossosdataset.KnossosDataset()
                kd.initialize_from_knossos_path(
                    self.config.entries["Dataset"]["seg_path"])
            except:
                raise("KnossosDataset could not be loaded")

        kd.from_matrix_to_cubes(self.bounding_box[0],
                                data=self.voxels.astype(np.uint64) * write_id,
                                datatype=np.uint64,
                                kzip_path=path,
                                overwrite=False)

    def clear_cache(self):
        self._voxels = None
        self._voxel_list = None
        self._mesh = None
        self._views = None
        self._skeleton = None

    # SKELETON
    @property
    def skeleton_dict_path(self):
        print(self.segobj_dir)
        return self.segobj_dir + "/skeletons.pkl"


    def copy2dir(self, dest_dir, safe=True):
        # get all files in home directory
        fps = get_filepaths_from_dir(self.segobj_dir, ending="")
        fnames = [os.path.split(fname)[1] for fname in fps]
        if not os.path.isdir(dest_dir):
            os.makedirs(dest_dir)
        for i in range(len(fps)):
            src_filename = fps[i]
            dest_filename = dest_dir + "/" + fnames[i]
            try:
                safe_copy(src_filename, dest_filename, safe=safe)
            except Exception, e:
                print(e)
                print("Skipped", fnames[i])
                pass
        # copy attr_dict values
        self.load_attr_dict()
        if os.path.isfile(dest_dir+"/atrr_dict.pkl"):
            dest_attr_dc = load_pkl2obj(dest_dir+"/atrr_dict.pkl")
        else:
             dest_attr_dc = {}
        # overwrite existing keys in the destination attribute dict
        dest_attr_dc.update(self.attr_dict)
        self.attr_dict = dest_attr_dc
        self.save_attr_dict()

    def split_component(self, dist, new_sd, new_id):
        kdtree = spatial.cKDTree(self.voxel_list)

        graph = nx.from_edgelist(kdtree.query_pairs(dist))
        ccs = list(nx.connected_components(graph))

        partner_ids = [self.id - ((self.id >> 32) << 32), self.id >> 32]

        if len(ccs) == 1:
            new_so_obj = new_sd.get_segmentation_object(new_id, create=True)
            new_id += 1

            new_so_obj.attr_dict["paths_to_voxels"] = self.paths_to_voxels
            new_so_obj.attr_dict["%s_partner_ids" % self.type] = partner_ids
            new_so_obj.save_attr_dict()
        else:
            for cc in ccs:
                new_so_obj = new_sd.get_segmentation_object(new_id, create=True)
                new_so_obj.attr_dict["%s_partner_ids" % self.type] = partner_ids
                new_so_obj.save_attr_dict()
                new_id += 1

                voxel_ids = np.array(list(cc), dtype=np.int32)
                this_voxel_list = self.voxel_list[voxel_ids]

                bb = [np.min(this_voxel_list, axis=0),
                      np.max(this_voxel_list, axis=0)]

                this_voxel_list -= bb[0]

                this_voxels = np.zeros(bb[1]-bb[0]+1, dtype=np.bool)
                this_voxels[this_voxel_list[:, 0],
                            this_voxel_list[:, 1],
                            this_voxel_list[:, 2]] = True

                save_voxels(new_so_obj, this_voxels, bb[0], size=len(voxel_ids))