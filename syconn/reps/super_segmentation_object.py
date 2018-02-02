# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max-Planck-Institute for Medical Research, Heidelberg, Germany
# Authors: Sven Dorkenwald, Philipp Schubert, Joergen Kornfeld

import glob
import networkx as nx
import numpy as np
import os
import re
import scipy.spatial
import seaborn as sns
import shutil
import sys
import time
import warnings
from collections import Counter

import syconn.reps.super_segmentation_helper
from knossos_utils import skeleton, knossosdataset

try:
    from knossos_utils.skeleton_utils import load_skeleton, write_skeleton
except:
    pass

from scipy import spatial
import segmentation
import super_segmentation_helper as ssh
from .segmentation import SegmentationObject
from ..proc.sd_proc import predict_sos_views
from .rep_helper import knossos_ml_from_sso, colorcode_vertices, \
    knossos_ml_from_svixs, subfold_from_ix, subfold_from_ix_SSO
from ..config import parser
from ..handler.basics import write_txt2kzip, get_filepaths_from_dir, safe_copy, \
    coordpath2anno, load_pkl2obj, write_obj2pkl, flatten_list, chunkify
from ..handler.compression import AttributeDict, MeshDict, LZ4Dict
from ..proc.image import single_conn_comp_img
from ..proc.graphs import split_glia, split_subcc, create_mst_skeleton
from ..proc.meshes import write_mesh2kzip, merge_someshs, compartmentalize_mesh
from ..proc.rendering import render_sampled_sso, comp_window, \
    multi_render_sampled_svidlist, render_sso_coords
try:
    from knossos_utils import mergelist_tools
except ImportError:
    from knossos_utils import mergelist_tools_fallback as mergelist_tools

from ..mp import qsub_utils as qu
from ..mp import shared_mem as sm
script_folder = os.path.abspath(os.path.dirname(__file__) + "/../QSUB_scripts/")

try:
    default_wd_available = True
    from ..config.global_params import wd
except:
    default_wd_available = False


class SuperSegmentationObject(object):
    def __init__(self, ssv_id, version=None, version_dict=None,
                 working_dir=None, create=True, sv_ids=None, scaling=None,
                 object_caching=True, voxel_caching=True, mesh_caching=True,
                 view_caching=False, config=None, nb_cpus=1,
                 enable_locking=True, ssd_type="ssv"):
        self.nb_cpus = nb_cpus
        self._id = ssv_id
        self.attr_dict = {} # dict(mi=[], sj=[], vc=[], sv=[])
        self.enable_locking = enable_locking

        self._type = ssd_type
        self._rep_coord = None
        self._size = None
        self._bounding_box = None
        self._config = config

        self._object_caching = object_caching
        self._voxel_caching = voxel_caching
        self._mesh_caching = mesh_caching
        self._view_caching = view_caching
        self._objects = {}
        self.skeleton = None
        self._voxels = None
        self._voxels_xy_downsampled = None
        self._voxels_downsampled = None
        self._edge_graph = None

        # init mesh dicts
        self._meshes = {"sv": None, "sj": None,
                        "vc": None, "mi": None}
        self._mesh_compartments = None

        self._views = None
        self._dataset = None
        self._weighted_graph = None
        self._sample_locations = None

        if sv_ids is not None:
            self.attr_dict["sv"] = sv_ids

        try:
            self._scaling = np.array(scaling)
        except:
            print("Currently, scaling has to be set in the config")
            self._scaling = np.array([1, 1, 1])

        if working_dir is None:
            try:
                self._working_dir = wd
            except:
                raise Exception("No working directory (wd) specified in config")
        else:
            self._working_dir = working_dir

        if scaling is None:
            try:
                self._scaling = \
                    np.array(self.config.entries["Dataset"]["scaling"])
            except:
                self._scaling = np.array([1, 1, 1])
        else:
            self._scaling = scaling

        if version is None:
            try:
                self._version = self.config.entries["Versions"][self.type]
            except:
                raise Exception("unclear value for version")
        elif version == "new":
            other_datasets = glob.glob(self.working_dir + "/%s_*" % self.type)
            max_version = -1
            for other_dataset in other_datasets:
                other_version = \
                    int(re.findall("[\d]+",
                                   os.path.basename(other_dataset))[-1])
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
            # TODO @Sven: only valid for SSDS!
            elif isinstance(version_dict, str) and version_dict == "load":
                if self.version_dict_exists:
                    self.load_version_dict()
            else:
                raise Exception("No version dict specified in config")

        if create and not os.path.exists(self.ssv_dir):
            os.makedirs(self.ssv_dir)

    def __hash__(self):
        return hash((self.id, self.type, frozenset(self.sv_ids)))

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return self.id == other.id and self.type == other.type and \
               frozenset(self.sv_ids) == frozenset(other.sv_ids)

    def __ne__(self, other):
        return not self.__eq__(other)

    #                                                       IMMEDIATE PARAMETERS

    @property
    def type(self):
        return self._type

    @property
    def id(self):
        return self._id

    @property
    def version(self):
        return str(self._version)

    @property
    def object_caching(self):
        return self._object_caching

    @property
    def voxel_caching(self):
        return self._voxel_caching

    @property
    def mesh_caching(self):
        return self._mesh_caching

    @property
    def view_caching(self):
        return self._view_caching

    @property
    def scaling(self):
        return self._scaling

    # @property
    # def dataset(self):
    #     if self._dataset is None:
    #         self._dataset = SuperSegmentationDataset(
    #             working_dir=self.working_dir,
    #             version=self.version,
    #             scaling=self.scaling,
    #             version_dict=self.version_dict)
    #     return self._dataset

    #                                                                      PATHS

    @property
    def working_dir(self):
        return self._working_dir

    @property
    def identifier(self):
        return "%s_%s" % (self.type, self.version.lstrip("_"))

    @property
    def ssds_dir(self):
        return "%s/%s/" % (self.working_dir, self.identifier)

    @property
    def ssv_dir(self):
        return "%s/so_storage/%s/" % (self.ssds_dir, subfold_from_ix_SSO(self.id))

    @property
    def attr_dict_path(self):
        return self.ssv_dir + "atrr_dict.pkl"

    @property
    def attr_dict_path_new(self):
        return self.ssv_dir + "attr_dict.pkl"

    @property
    def skeleton_kzip_path(self):
        return self.ssv_dir + "skeleton.k.zip"

    @property
    def skeleton_kzip_path_views(self):
        return self.ssv_dir + "skeleton_views.k.zip"

    @property
    def objects_dense_kzip_path(self):
        return self.ssv_dir + "objects_overlay.k.zip"

    @property
    def skeleton_path(self):
        return self.ssv_dir + "skeleton.pkl"

    @property
    def skeleton_path_views(self):
        return self.ssv_dir + "skeleton_views.pkl"

    @property
    def edgelist_path(self):
        return self.ssv_dir + "edge_list.bz2"

    @property
    def view_path(self):
        return self.ssv_dir + "views.lz4"

    @property
    def mesh_dc_path(self):
        return self.ssv_dir + "mesh_dc.pkl"

    #                                                                        IDS

    @property
    def sv_ids(self):
        return self.lookup_in_attribute_dict("sv")

    @property
    def sj_ids(self):
        return self.lookup_in_attribute_dict("sj")

    @property
    def mi_ids(self):
        return self.lookup_in_attribute_dict("mi")

    @property
    def vc_ids(self):
        return self.lookup_in_attribute_dict("vc")

    @property
    def dense_kzip_ids(self):
        return dict([("mi", 1), ("vc", 2), ("sj", 3)])

    #                                                        SEGMENTATIONOBJECTS

    @property
    def svs(self):
        return self.get_seg_objects("sv")

    @property
    def sjs(self):
        return self.get_seg_objects("sj")

    @property
    def mis(self):
        return self.get_seg_objects("mi")

    @property
    def vcs(self):
        return self.get_seg_objects("vc")

    #                                                                     MESHES

    @property
    def mesh(self):
        return self.load_mesh("sv")

    def load_mesh(self, mesh_type):
        if not mesh_type in self._meshes:
            return None
        if self._meshes[mesh_type] is None:
            if not self.mesh_caching:
                return self._load_obj_mesh(mesh_type)
            self._meshes[mesh_type] = self._load_obj_mesh(mesh_type)
        return self._meshes[mesh_type]

    @property
    def sj_mesh(self):
        return self.load_mesh("sj")

    @property
    def vc_mesh(self):
        return self.load_mesh("vc")

    @property
    def mi_mesh(self):
        return self.load_mesh("mi")
    #                                                                 PROPERTIES

    @property
    def cell_type(self):
        if self.cell_type_ratios is not None:
            return np.argmax(self.cell_type_ratios)
        else:
            return None

    @property
    def cell_type_ratios(self):
        return self.lookup_in_attribute_dict("cell_type_ratios")

    @property
    def weighted_graph(self):
        if self._weighted_graph is None:
            if self.skeleton is None:
                self.load_skeleton()

            node_scaled = self.skeleton["nodes"] * self.scaling
            edge_coords = node_scaled[self.skeleton["edges"]]
            weights = np.linalg.norm(edge_coords[:, 0] - edge_coords[:, 1],
                                     axis=1)

            self._weighted_graph = nx.Graph()
            self._weighted_graph.add_weighted_edges_from(
                np.concatenate((self.skeleton["edges"], weights[:, None]),
                               axis=1))
        return self._weighted_graph

    @property
    def config(self):
        if self._config is None:
            self._config = parser.Config(self.working_dir)
        return self._config

    @property
    def size(self):
        if self._size is None:
            self._size = self.lookup_in_attribute_dict("size")

        if self._size is None:
            self.calculate_size()

        return self._size

    @property
    def bounding_box(self):
        if self._bounding_box is None:
            self._bounding_box = self.lookup_in_attribute_dict("bounding_box")

        if self._bounding_box is None:
            self.calculate_bounding_box()

        return self._bounding_box

    @property
    def shape(self):
        return self.bounding_box[1] - self.bounding_box[0]

    @property
    def rep_coord(self):
        if self._rep_coord is None:
            self._rep_coord = self.lookup_in_attribute_dict("rep_coord")

        if self._rep_coord is None:
            self._rep_coord = self.svs[0].rep_coord

        return self._rep_coord

    @property
    def attr_dict_exists(self):
        return os.path.isfile(self.attr_dict_path)

    def mesh_exists(self, obj_type):
        mesh_dc = MeshDict(self.mesh_dc_path,
                           disable_locking=not self.enable_locking)
        return obj_type in mesh_dc

    @property
    def voxels(self):
        if len(self.sv_ids) == 0:
            return None

        if self._voxels is None:
            voxels = np.zeros(self.bounding_box[1] - self.bounding_box[0],
                              dtype=np.bool)
            for sv in self.svs:
                sv._voxel_caching = False
                if sv.voxels_exist:
                    print(np.sum(sv.voxels), sv.size)
                    box = [sv.bounding_box[0] - self.bounding_box[0],
                           sv.bounding_box[1] - self.bounding_box[0]]

                    voxels[box[0][0]: box[1][0],
                           box[0][1]: box[1][1],
                           box[0][2]: box[1][2]][sv.voxels] = True
                else:
                    print("missing voxels from %d" % sv.id)

            if self.voxel_caching:
                self._voxels = voxels
            else:
                return voxels

        return self._voxels

    @property
    def voxels_xy_downsampled(self):
        if self._voxels_xy_downsampled is None:
            if self.voxel_caching:
                self._voxels_xy_downsampled = \
                    self.load_voxels_downsampled((2, 2, 1))
            else:
                return self.load_voxels_downsampled((2, 2, 1))

        return self._voxels_xy_downsampled

    @property
    def edge_graph(self):
        if self._edge_graph is None:
            self._edge_graph = self.load_graph()
        return self._edge_graph

    def load_voxels_downsampled(self, downsampling=(2, 2, 1), nb_threads=10):
        syconn.reps.super_segmentation_helper.load_voxels_downsampled(self, downsampling=downsampling,
                                                                      nb_threads=nb_threads)

    def get_seg_objects(self, obj_type):
        if obj_type not in self._objects:
            objs = []

            for obj_id in self.lookup_in_attribute_dict(obj_type):
                objs.append(self.get_seg_obj(obj_type, obj_id))

            if self.object_caching:
                self._objects[obj_type] = objs
            else:
                return objs

        return self._objects[obj_type]

    def get_seg_obj(self, obj_type, obj_id):
        return segmentation.SegmentationObject(obj_id=obj_id,
                                               obj_type=obj_type,
                                               version=self.version_dict[
                                                   obj_type],
                                               working_dir=self.working_dir,
                                               create=False,
                                               scaling=self.scaling)

    def get_seg_dataset(self, obj_type):
        return segmentation.SegmentationDataset(obj_type,
                                                version_dict=self.version_dict,
                                                version=self.version_dict[
                                                    obj_type],
                                                scaling=self.scaling,
                                                working_dir=self.working_dir)

    def load_attr_dict(self):
        try:
            self.attr_dict = load_pkl2obj(self.attr_dict_path)
            return 0
        except (IOError, EOFError):
            return -1

    def load_graph(self):
        G = nx.read_edgelist(self.edgelist_path, nodetype=int)
        new_G = nx.Graph()
        for e in G.edges_iter():
            new_G.add_edge(self.get_seg_obj("sv", e[0]),
                           self.get_seg_obj("sv", e[1]))
        return new_G

    def load_edgelist(self):
        g = self.load_graph()
        return g.edges()

    def _load_obj_mesh(self, obj_type="sv", rewrite=False):
        if not rewrite and self.mesh_exists(obj_type) and not \
                        self.version == "tmp":
            mesh_dc = MeshDict(self.mesh_dc_path,
                               disable_locking=not self.enable_locking)
            if len(mesh_dc[obj_type]) == 3:
                ind, vert, normals = mesh_dc[obj_type]
            else:
                ind, vert = mesh_dc[obj_type]
                normals = np.zeros((0, ), dtype=np.float32)
        else:
            ind, vert, normals = merge_someshs(self.get_seg_objects(obj_type),
                                      nb_cpus=self.nb_cpus)
            if not self.version == "tmp":
                mesh_dc = MeshDict(self.mesh_dc_path, read_only=False,
                                   disable_locking=not self.enable_locking)
                mesh_dc[obj_type] = [ind, vert, normals]
                mesh_dc.save2pkl()
        return np.array(ind, dtype=np.int), np.array(vert, dtype=np.int),\
               np.array(normals, dtype=np.float32)

    def _load_obj_mesh_compr(self, obj_type="sv"):
        mesh_dc = MeshDict(self.mesh_dc_path,
                           disable_locking=not self.enable_locking)
        return mesh_dc._dc_intern[obj_type]

    def load_svixs(self):
        if not os.path.isfile(self.edgelist_path):
            warnings.warn("Edge list of SSO %d does not exist. Return empty "
                          "list.", RuntimeWarning)
            return []
        edges = self.load_edgelist()
        return np.unique(np.concatenate([[a.id, b.id] for a, b in edges]))

    def save_attr_dict(self):
        try:
            orig_dc = load_pkl2obj(self.attr_dict_path)
        except IOError:
            orig_dc = {}
        orig_dc.update(self.attr_dict)
        write_obj2pkl(orig_dc, self.attr_dict_path)

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
        try:
            attr_dict = load_pkl2obj(self.attr_dict_path)
        except IOError, e:
            if not "[Errno 13] Permission denied" in str(e):
                pass
            else:
                warnings.warn("Could not load SSO attributes to %s due to "
                              "missing permissions." % self.attr_dict_path,
                              RuntimeWarning)
            attr_dict = {}
        for k, v in zip(attr_keys, attr_values):
            attr_dict[k] = v
        try:
            write_obj2pkl(self.attr_dict_path, attr_dict)
        except IOError, e:
            if not "[Errno 13] Permission denied" in str(e):
                raise (IOError, e)
            else:
                warnings.warn("Could not save SSO attributes to %s due to "
                              "missing permissions." % self.attr_dict_path,
                              RuntimeWarning)

    def attr_exists(self, attr_key):
        return attr_key in self.attr_dict

    def lookup_in_attribute_dict(self, attr_key):
        if attr_key in self.attr_dict:
            return self.attr_dict[attr_key]
        elif len(self.attr_dict) <= 4:
            if self.load_attr_dict() == -1:
                return None
        if attr_key in self.attr_dict:
            return self.attr_dict[attr_key]
        else:
            return None

    def calculate_size(self):
        self._size = 0
        for sv in self.svs:
            self._size += sv.size

    def calculate_bounding_box(self):
        self._bounding_box = np.ones((2, 3), dtype=np.int) * np.inf
        self._bounding_box[1] = 0

        self._size = 0
        real_sv_cnt = 0
        for sv in self.svs:
            if sv.voxels_exist:
                real_sv_cnt += 1
                sv_bb = sv.bounding_box
                sv.clear_cache()
                for dim in range(3):
                    if self._bounding_box[0, dim] > sv_bb[0, dim]:
                        self._bounding_box[0, dim] = sv_bb[0, dim]
                    if self._bounding_box[1, dim] < sv_bb[1, dim]:
                        self._bounding_box[1, dim] = sv_bb[1, dim]

                self._size += sv.size

        if real_sv_cnt > 0:
            self._bounding_box = self._bounding_box.astype(np.int)
        else:
            self._bounding_box = np.zeros((2, 3), dtype=np.int)

    def calculate_skeleton(self, size_threshold=1e20, kd=None,
                           coord_scaling=(8, 8, 4), plain=False, cleanup=True,
                           nb_threads=1):
        if np.product(self.shape) < size_threshold:
            # vx = self.load_voxels_downsampled(coord_scaling)
            # vx = self.voxels[::coord_scaling[0],
            #                  ::coord_scaling[1],
            #                  ::coord_scaling[2]]
            vx = self.load_voxels_downsampled(downsampling=coord_scaling)
            vx = scipy.ndimage.morphology.binary_closing(
                np.pad(vx, 3, mode="constant", constant_values=0), iterations=3)
            vx = vx[3: -3, 3: -3, 3:-3]

            if plain:
                nodes, edges, diameters = \
                    ssh.reskeletonize_plain(vx, coord_scaling=coord_scaling)
                nodes = np.array(nodes, dtype=np.int) + self.bounding_box[0]
            else:
                nodes, edges, diameters = \
                    ssh.reskeletonize_chunked(self.id, self.shape,
                                              self.bounding_box[0],
                                              self.scaling,
                                              voxels=vx,
                                              coord_scaling=coord_scaling,
                                              nb_threads=nb_threads)

        elif kd is not None:
            nodes, edges, diameters = \
                ssh.reskeletonize_chunked(self.id, self.shape,
                                          self.bounding_box[0], self.scaling,
                                          kd=kd, coord_scaling=coord_scaling,
                                          nb_threads=nb_threads)
        else:
            return

        nodes = np.array(nodes, dtype=np.int)
        edges = np.array(edges, dtype=np.int)
        diameters = np.array(diameters, dtype=np.float)

        self.skeleton = {}
        self.skeleton["edges"] = edges
        self.skeleton["nodes"] = nodes
        self.skeleton["diameters"] = diameters

        if cleanup:
            for i in range(2):
                if len(self.skeleton["edges"]) > 2:
                    self.skeleton = ssh.cleanup_skeleton(self.skeleton,
                                                         coord_scaling)

    def save_skeleton_to_kzip(self, dest_path=None):
        try:
            if self.skeleton is None:
                self.load_skeleton()
            a = skeleton.SkeletonAnnotation()
            a.scaling = self.scaling
            a.comment = "skeleton"

            skel_nodes = []
            for i_node in range(len(self.skeleton["nodes"])):
                c = self.skeleton["nodes"][i_node]
                r = self.skeleton["diameters"][i_node] / 2
                skel_nodes.append(skeleton.SkeletonNode().
                                  from_scratch(a, c[0], c[1], c[2], radius=r))
                if "axoness" in self.skeleton:
                    skel_nodes[-1].data["axoness"] = self.skeleton["axoness"][
                        i_node]
                if "cell_type" in self.skeleton:
                    skel_nodes[-1].data["cell_type"] = \
                    self.skeleton["cell_type"][i_node]
                if "meta" in self.skeleton:
                    skel_nodes[-1].data["meta"] = self.skeleton["meta"][i_node]

                a.addNode(skel_nodes[-1])

            for edge in self.skeleton["edges"]:
                a.addEdge(skel_nodes[edge[0]], skel_nodes[edge[1]])

            if dest_path is None:
                dest_path = self.skeleton_kzip_path
            write_skeleton(dest_path, [a])
        except Exception, e:
            print("[SSO: %d] Could not load/save skeleton:\n%s" % (self.id, e))

    def save_objects_to_kzip_sparse(self, obj_types=("sj", "mi", "vc"),
                                    dest_path=None):
        annotations = []
        for obj_type in obj_types:
            assert obj_type in self.attr_dict
            map_ratio_key = "mapping_%s_ratios" % obj_type
            if not map_ratio_key in self.attr_dict.keys():
                print "%s not yet mapped. Object nodes are not written to " \
                      "k.zip." % obj_type
                continue
            overlap_ratios = np.array(self.attr_dict[map_ratio_key])
            overlap_ids = np.array(self.attr_dict["mapping_%s_ids" % obj_type])

            a = skeleton.SkeletonAnnotation()
            a.scaling = self.scaling
            a.comment = obj_type

            so_objs = self.get_seg_objects(obj_type)
            for so_obj in so_objs:
                c = so_obj.rep_coord

                # somewhat approximated from sphere volume:
                r = np.power(so_obj.size / 3., 1 / 3.)
                skel_node = skeleton.SkeletonNode(). \
                    from_scratch(a, c[0], c[1], c[2], radius=r)
                skel_node.data["overlap"] = \
                    overlap_ratios[overlap_ids == so_obj.id][0]
                skel_node.data["size"] = so_obj.size
                skel_node.data["shape"] = so_obj.shape

                a.addNode(skel_node)

            annotations.append(a)

        if dest_path is None:
            dest_path = self.skeleton_kzip_path
        write_skeleton(dest_path, annotations)

    def save_objects_to_kzip_dense(self, obj_types):
        if os.path.exists(self.objects_dense_kzip_path[:-6]):
            shutil.rmtree(self.objects_dense_kzip_path[:-6])
        if os.path.exists(self.objects_dense_kzip_path):
            os.remove(self.objects_dense_kzip_path)

        for obj_type in obj_types:
            so_objs = self.get_seg_objects(obj_type)
            for so_obj in so_objs:
                print so_obj.id
                so_obj.save_kzip(path=self.objects_dense_kzip_path,
                                 write_id=self.dense_kzip_ids[obj_type])

    def save_skeleton(self, to_kzip=False, to_object=True):
        if to_object:
            write_obj2pkl(self.skeleton, self.skeleton_path)

        if to_kzip:
            self.save_skeleton_to_kzip()

    def load_skeleton(self):
        try:
            self.skeleton = load_pkl2obj(self.skeleton_path)
            return True
        except:
            return False

    def aggregate_segmentation_object_mappings(self, obj_types, save=False):
        assert isinstance(obj_types, list)

        mappings = dict((obj_type, Counter()) for obj_type in obj_types)
        for sv in self.svs:
            sv.load_attr_dict()

            for obj_type in obj_types:
                if "mapping_%s_ids" % obj_type in sv.attr_dict:
                    keys = sv.attr_dict["mapping_%s_ids" % obj_type]
                    values = sv.attr_dict["mapping_%s_ratios" % obj_type]
                    mappings[obj_type] += Counter(dict(zip(keys, values)))

        for obj_type in obj_types:
            if obj_type in mappings:
                self.attr_dict["mapping_%s_ids" % obj_type] = \
                    mappings[obj_type].keys()
                self.attr_dict["mapping_%s_ratios" % obj_type] = \
                    mappings[obj_type].values()

        if save:
            self.save_attr_dict()

    def apply_mapping_decision(self, obj_type, correct_for_background=True,
                               lower_ratio=None, upper_ratio=None,
                               sizethreshold=None, save=True):
        assert obj_type in self.version_dict

        self.load_attr_dict()
        if not "mapping_%s_ratios" % obj_type in self.attr_dict:
            print("No mapping ratios found")
            return

        if not "mapping_%s_ids" % obj_type in self.attr_dict:
            print("no mapping ids found")
            return

        if lower_ratio is None:
            try:
                lower_ratio = self.config.entries["LowerMappingRatios"][
                    obj_type]
            except:
                raise ("Lower ratio undefined")

        if upper_ratio is None:
            try:
                upper_ratio = self.config.entries["UpperMappingRatios"][
                    obj_type]
            except:
                print("Upper ratio undefined - 1. assumed")
                upper_ratio = 1.

        if sizethreshold is None:
            try:
                sizethreshold = self.config.entries["Sizethresholds"][obj_type]
            except:
                raise ("Size threshold undefined")

        obj_ratios = np.array(self.attr_dict["mapping_%s_ratios" % obj_type])

        if correct_for_background:
            for i_so_id in range(
                    len(self.attr_dict["mapping_%s_ids" % obj_type])):
                so_id = self.attr_dict["mapping_%s_ids" % obj_type][i_so_id]
                obj_version = self.config.entries["Versions"][obj_type]
                this_so = SegmentationObject(so_id, obj_type,
                                             version=obj_version,
                                             scaling=self.scaling,
                                             working_dir=self.working_dir)
                this_so.load_attr_dict()

                if 0 in this_so.attr_dict["mapping_ids"]:
                    ratio_0 = this_so.attr_dict["mapping_ratios"][
                        this_so.attr_dict["mapping_ids"] == 0][0]

                    obj_ratios[i_so_id] /= (1 - ratio_0)

        id_mask = obj_ratios > lower_ratio
        if upper_ratio < 1.:
            id_mask[obj_ratios > upper_ratio] = False

        candidate_ids = \
            np.array(self.attr_dict["mapping_%s_ids" % obj_type])[id_mask]

        self.attr_dict[obj_type] = []
        for candidate_id in candidate_ids:
            obj = segmentation.SegmentationObject(candidate_id,
                                                  obj_type=obj_type,
                                                  version=self.version_dict[
                                                      obj_type],
                                                  working_dir=self.working_dir,
                                                  config=self.config)
            if obj.size > sizethreshold:
                self.attr_dict[obj_type].append(candidate_id)

        if save:
            self.save_attr_dict()

    def map_cellobjects(self, obj_types=None, save=False):
        if obj_types is None:
            obj_types = ["mi", "sj", "vc"]
        self.aggregate_segmentation_object_mappings(obj_types, save=save)
        for obj_type in obj_types:
            self.apply_mapping_decision(obj_type, save=save,
                                        correct_for_background=obj_type == "sj")

    def clear_cache(self):
        self._objects = {}
        self._voxels = None
        self._voxels_xy_downsampled = None
        self._views = None
        self._sample_locations = None
        self._mesh_compartments = None
        self._meshes = None
        self.skeleton = None

    def copy2dir(self, dest_dir, safe=True):
        raise ("To be tested.", NotImplementedError)
        # get all files in home directory
        fps = get_filepaths_from_dir(self.ssv_dir, ending="")
        fnames = [os.path.split(fname)[1] for fname in fps]
        # Open the file and raise an exception if it exists
        if not os.path.isdir(dest_dir):
            os.makedirs(dest_dir)
        for i in range(len(fps)):
            src_filename = fps[i]
            dest_filename = dest_dir + "/" + fnames[i]
            try:
                safe_copy(src_filename, dest_filename, safe=safe)
            except Exception:
                print("Skipped", fnames[i])
                pass
        self.load_attr_dict()
        if os.path.isfile(dest_dir + "/atrr_dict.pkl"):
            dest_attr_dc = load_pkl2obj(dest_dir + "/atrr_dict.pkl")
        else:
            dest_attr_dc = {}
        dest_attr_dc.update(self.attr_dict)
        self.attr_dict = dest_attr_dc
        self.save_attr_dict()

    def partition_cc(self, max_nb=25):
        """
        Splits connected component into subgraphs.

        Parameters
        ----------
        max_nb : int
            Number of SV per CC

        Returns
        -------
        dict
        """
        init_g = self.edge_graph
        partitions = split_subcc(init_g, max_nb)
        return partitions

    # -------------------------------------------------------------------- VIEWS

    def load_views(self, woglia=True, raw_only=False):
        """

        Parameters
        ----------
        woglia : bool

        Returns
        -------
        list of array
            Views for each SV in self.svs
        """
        params = [[sv, {"woglia": woglia, "raw_only": raw_only}] for sv in self.svs]
        # list of arrays
        views = sm.start_multiprocess_obj("load_views", params,
                                          nb_cpus=self.nb_cpus)
        return views

    def view_existence(self, woglia=True):
        params = [[sv, {"woglia": woglia}] for sv in self.svs]
        so_views_exist = sm.start_multiprocess_obj("views_exist", params,
                                                   nb_cpus=self.nb_cpus)
        return so_views_exist

    def render_views(self, add_cellobjects=False, random_processing=True,
                     qsub_pe=None, overwrite=False, cellobjects_only=False,
                     woglia=True):
        if len(self.sv_ids) > 5e3:
            part = self.partition_cc()
            if 0:#not overwrite: # check existence of glia preds
                views_exist = np.array(self.view_existence(), dtype=np.int)
                print("Rendering huge SSO. %d/%d views left to process." \
                      % (np.sum(~views_exist), len(self.svs)))
                ex_dc = {}
                for ii, k in enumerate(self.svs):
                    ex_dc[k] = views_exist[ii]
                for k in part.keys():
                    if ex_dc[k]: # delete SO's with existing pred
                        del part[k]
                        continue
                del ex_dc
            else:
                print("Rendering huge SSO. %d views left to process." \
                      % len(self.svs))
            for k in part.keys():
                val = part[k]
                part[k] = [so.id for so in val]
            params = part.values()
            if random_processing:
                np.random.seed(int(time.time() * 1e4 % 1e6))
                np.random.shuffle(params)
            if qsub_pe is None:
                sm.start_multiprocess(multi_render_sampled_svidlist, params,
                                      nb_cpus=self.nb_cpus, debug=False)
            elif qu.__QSUB__:
                params = chunkify(params, 700)
                params = [[par, {"overwrite": overwrite,
                                 "render_first_only": True,
                                 "cellobjects_only": cellobjects_only}] for par in params]
                qu.QSUB_script(params, "render_views", pe=qsub_pe, queue=None,
                               script_folder=script_folder, n_max_co_processes=100)
            else:
                raise Exception("QSUB not available")
        else:
            render_sampled_sso(self, add_cellobjects=add_cellobjects,
                               verbose=False, overwrite=overwrite,
                               cellobjects_only=cellobjects_only, woglia=woglia)

    def sample_locations(self, force=False, cache=False, verbose=False):
        """

        Parameters
        ----------
        force : bool
            force resampling of locations
        cache : bool

        Returns
        -------
        list of array
            Sample coordinates for each SV in self.svs.
        """
        if not force and self._sample_locations:
            return self._sample_locations
        if verbose:
            start = time.time()
        if not force:
            if self.attr_exists("sample_locations"):
                return self.attr_dict["sample_locations"]
        params = [[sv, {"force": force}] for sv in self.svs]
        # list of arrays
        locs = sm.start_multiprocess_obj("sample_locations", params,
                                         nb_cpus=self.nb_cpus)
        if cache:
            self.save_attributes(["sample_locations"], [locs])
        if verbose:
            dur = time.time() - start
            print("Sampling locations from %d SVs took %0.2fs. %0.4fs/SV (in" \
                  "cl. read/write)" % (len(self.svs), dur, dur / len(self.svs)))
        return locs

    # ------------------------------------------------------------------ EXPORTS

    def pklskel2kzip(self):
        self.load_skeleton()
        es = self.skeleton["edges"]
        ns = self.skeleton["nodes"]
        a = skeleton.SkeletonAnnotation()
        a.scaling = self.scaling
        a.comment = "skeleton"
        for e in es:
            n0 = skeleton.SkeletonNode().from_scratch(a, ns[e[0]][0],
                                                      ns[e[0]][1], ns[e[0]][2])
            n1 = skeleton.SkeletonNode().from_scratch(a, ns[e[1]][0],
                                                      ns[e[1]][1], ns[e[1]][2])
            a.addNode(n0)
            a.addNode(n1)
            a.addEdge(n0, n1)
        write_skeleton(self.skeleton_kzip_path_views, a)

    def write_locations2kzip(self, dest_path=None):
        if dest_path is None:
            dest_path = self.skeleton_kzip_path_views
        loc = np.concatenate(self.sample_locations())
        new_anno = coordpath2anno(loc, add_edges=False)
        new_anno.setComment("sample_locations")
        write_skeleton(dest_path, [new_anno])

    def mergelist2kzip(self, dest_path=None):
        self.load_attr_dict()
        kml = knossos_ml_from_sso(self)
        if dest_path is None:
            dest_path = self.skeleton_kzip_path_views
        write_txt2kzip(dest_path, kml, "mergelist.txt")

    def mesh2kzip(self, obj_type="sv", dest_path=None, ext_color=None):
        if dest_path is None:
            dest_path = self.skeleton_kzip_path_views
        if obj_type == "sv":
            mesh = self.mesh
            color = (130, 130, 130, 160)
        elif obj_type == "sj":
            mesh = self.sj_mesh
            color = (int(0.849 * 255), int(0.138 * 255), int(0.133 * 255), 255)
        elif obj_type == "vc":
            mesh = self.vc_mesh
            color = (int(0.175 * 255), int(0.585 * 255), int(0.301 * 255), 255)
        elif obj_type == "mi":
            mesh = self.mi_mesh
            color = (0, 153, 255, 255)
        else:
            raise ("Given object type '%s' does not exist." % obj_type,
                   TypeError)
        if ext_color is not None:
            if ext_color == 0:
                color = None
            else:
                color = ext_color
        write_mesh2kzip(dest_path, mesh[0], mesh[1], mesh[2], color,
                        ply_fname=obj_type + ".ply")

    def meshs2kzip(self, dest_path=None, sv_color=None):
        if dest_path is None:
            dest_path = self.skeleton_kzip_path_views
        for ot in ["sj", "vc", "mi",
                   "sv"]:  # determins rendering order in KNOSSOS
            self.mesh2kzip(ot, dest_path=dest_path, ext_color=sv_color if
            ot == "sv" else None)

    def export_kzip(self, dest_path=None, sv_color=None):
        """
        Writes the sso to a KNOSSOS loadable kzip.
        Color is specified as rgba, 0 to 255.
        
        Parameters
        ----------
        dest_path : str
        sv_color : 4-tuple of int

        Returns
        -------

        """

        self.load_attr_dict()
        self.save_skeleton_to_kzip(dest_path=dest_path)
        self.save_objects_to_kzip_sparse(["mi", "sj", "vc"],
                                         dest_path=dest_path)
        self.meshs2kzip(dest_path=dest_path, sv_color=sv_color)
        self.mergelist2kzip(dest_path=dest_path)

    def write_svmeshs2kzip(self, dest_path=None):
        if dest_path is None:
            dest_path = self.skeleton_kzip_path_views
        for ii, sv in enumerate(self.svs):
            mesh = sv.mesh
            write_mesh2kzip(dest_path, mesh[0], mesh[1], None,
                            ply_fname="sv%d.ply" % ii)

    def _svattr2mesh(self, dest_path, attr_key, cmap, normalize_vals=False):
        sv_attrs = np.array([sv.lookup_in_attribute_dict(attr_key)
                             for sv in self.svs])
        if normalize_vals:
            min_val = sv_attrs.min()
            sv_attrs -= min_val
            sv_attrs /= sv_attrs.max()
        ind, vert, col = merge_someshs(self.svs, color_vals=sv_attrs, cmap=cmap)
        write_mesh2kzip(dest_path, ind, vert, col, "%s.ply" % attr_key)

    def svprobas2mergelist(self, key="glia_probas", dest_path=None):
        if dest_path is None:
            dest_path = self.skeleton_kzip_path_views
        coords = np.array([sv.rep_coord for sv in self.svs])
        sv_comments = ["%s; %s" % (str(np.mean(sv.attr_dict[key], axis=0)),
                       str(sv.attr_dict[key]).replace('\n', ''))
                       for sv in self.svs]
        kml = knossos_ml_from_svixs([sv.id for sv in self.svs], coords,
                                    comments=sv_comments)
        write_txt2kzip(dest_path, kml, "mergelist.txt")

    def _pred2mesh(self, pred_coords, preds, ply_fname=None, dest_path=None,
                   colors=None, k=1):
        """
        If dest_path or ply_fname is None then indices, vertices, colors are
        returned. Else Mesh is written to k.zip file as specified.

        Parameters
        ----------
        pred_coords : np.array
            N x 3
        preds : np.array
            N x 1
        ply_fname : str
        dest_path : str
        colors : np.array
            Color for each possible prediction value (range(np.max(preds))
        k : int
            Number of nearest neighbors (average prediction)
        Returns
        -------
        None or [np.array, np.array, np.array]
        """
        mesh = self.mesh
        col = colorcode_vertices(mesh[1].reshape((-1, 3)), pred_coords,
                                 preds, colors=colors, k=k)
        if dest_path is None or ply_fname is None:
            return mesh[0], mesh[1], col
        else:
            write_mesh2kzip(dest_path, mesh[0], mesh[1], col,
                            ply_fname=ply_fname)

    def single_compartment_meshs(self, comp_type):
        """

        Parameters
        ----------
        comp_type : int or str
            0 /  "dendrite"
            1 / "axon"
            2 "soma"
        """
        if not self._mesh_compartments:
            compartmentalize_mesh(self)
        if type(comp_type) == int:
            comp_type = {0: "dendrite", 1: "axon", 2: "soma"}[comp_type]
        return self._mesh_compartments[comp_type]


    # --------------------------------------------------------------------- GLIA
    def gliaprobas2mesh(self, dest_path=None, pred_key_appendix=""):
        if dest_path is None:
            dest_path = self.skeleton_kzip_path_views
        mcmp = sns.diverging_palette(250, 15, s=99, l=60, center="dark",
                                     as_cmap=True)
        self._svattr2mesh(dest_path, "glia_probas" + pred_key_appendix,
                          cmap=mcmp)

    def gliapred2mesh(self, dest_path=None, thresh=0.161489, pred_key_appendix=""):
        self.load_attr_dict()
        for sv in self.svs:
            sv.load_attr_dict()
        glia_svs = [sv for sv in self.svs if
                    sv.glia_pred(thresh, pred_key_appendix) == 1]
        nonglia_svs = [sv for sv in self.svs if
                       sv.glia_pred(thresh, pred_key_appendix) == 0]
        if dest_path is None:
            dest_path = self.skeleton_kzip_path_views
        mesh = merge_someshs(glia_svs)
        write_mesh2kzip(dest_path, mesh[0], mesh[1], None,
                        ply_fname="glia_%0.2f.ply" % thresh)
        mesh = merge_someshs(nonglia_svs)
        write_mesh2kzip(dest_path, mesh[0], mesh[1], None,
                        ply_fname="nonglia_%0.2f.ply" % thresh)

    def gliapred2mergelist(self, dest_path=None, thresh=0.161489,
                           pred_key_appendix=""):
        if dest_path is None:
            dest_path = self.skeleton_kzip_path_views
        params = [[sv, ] for sv in self.svs]
        coords = sm.start_multiprocess_obj("rep_coord", params,
                                           nb_cpus=self.nb_cpus)
        coords = np.array(coords)
        params = [[sv, {"thresh": thresh, "pred_key_appendix":
            pred_key_appendix}] for sv in self.svs]
        glia_preds = sm.start_multiprocess_obj("glia_pred", params,
                                               nb_cpus=self.nb_cpus)
        glia_preds = np.array(glia_preds)
        glia_comments = ["%0.4f" % gp for gp in glia_preds]
        kml = knossos_ml_from_svixs([sv.id for sv in self.svs], coords,
                                    comments=glia_comments)
        write_txt2kzip(dest_path, kml, "mergelist.txt")

    def gliasplit(self, dest_path=None, recompute=False, thresh=0.161489,
                  write_shortest_paths=False, verbose=False,
                  pred_key_appendix=""):
        if recompute or not (
                    self.attr_exists("glia_svs") and self.attr_exists(
                    "nonglia_svs")):

            # # HACK
            dest_dir = "/wholebrain/scratch/pschuber/ssv3_splits_v2/%s" % subfold_from_ix_SSO(
                self.id)
            ad = AttributeDict(dest_dir + "attr_dict.pkl", read_only=True,
                           disable_locking=not self.enable_locking)
            if self.id in ad:
                return
            # # HACK END

            if write_shortest_paths:
                shortest_paths_dir = os.path.split(dest_path)[0]
            else:
                shortest_paths_dir = None
            if verbose:
                print "Splitting glia in SSV %d with %d SV's." % \
                      (self.id, len(self.svs))
                start = time.time()
            nonglia_ccs, glia_ccs = split_glia(self, thresh=thresh,
                        pred_key_appendix=pred_key_appendix,
                        shortest_paths_dest_dir=shortest_paths_dir)

            # from neuropatch.nets.prediction import get_glia_model
            # m = get_glia_model()
            # self.predict_views_gliaSV(m)
            # del m
            # nonglia_ccs, glia_ccs = split_glia(self, thresh=thresh,
            #                                    shortest_paths_dest_dir=shortest_paths_dir)
            if verbose:
                print "Splitting glia in SSV %d with %d SV's finished after " \
                      "%.4gs." % (self.id, len(self.svs), time.time() - start)
            non_glia_ccs_ixs = [[so.id for so in nonglia] for nonglia in
                                nonglia_ccs]
            glia_ccs_ixs = [[so.id for so in glia] for glia in
                            glia_ccs]

            # self.attr_dict["glia_svs"] = glia_ccs_ixs
            # self.attr_dict["nonglia_svs"] = non_glia_ccs_ixs
            # self.save_attributes(["glia_svs", "nonglia_svs"],
            #                      [glia_ccs_ixs, non_glia_ccs_ixs])

            # HACK
            if not os.path.isdir(dest_dir):
                os.makedirs(dest_dir)
            ad = AttributeDict(dest_dir + "attr_dict.pkl", read_only=False,
                           disable_locking=not self.enable_locking)
            ad[self.id]["glia_svs"] = glia_ccs_ixs
            ad[self.id]["nonglia_svs"] = non_glia_ccs_ixs
            ad.save2pkl()
            # HACK END

    def load_gliasplit_ad(self):
        dest_dir = "/wholebrain/scratch/pschuber/ssv3_splits_v2/%s" % subfold_from_ix_SSO(
            self.id)
        ad = AttributeDict(dest_dir + "attr_dict.pkl", read_only=True,
                           disable_locking=not self.enable_locking)
        return ad[self.id]

    def gliasplit2mesh(self, dest_path=None):
        """

        Parameters
        ----------
        dest_path :

        Returns
        -------

        """
        attr_dict = self.load_gliasplit_ad()
        if dest_path is None:
            dest_path = self.skeleton_kzip_path_views
        # write meshes of CC's
        glia_ccs = attr_dict["glia_svs"]
        for kk, glia in enumerate(glia_ccs):
            mesh = merge_someshs([self.get_seg_obj("sv", ix) for ix in
                                  glia])
            write_mesh2kzip(dest_path, mesh[0], mesh[1], None,
                            "glia_cc%d.ply" % kk)
        non_glia_ccs = attr_dict["nonglia_svs"]
        for kk, nonglia in enumerate(non_glia_ccs):
            mesh = merge_someshs([self.get_seg_obj("sv", ix) for ix in
                                  nonglia])
            write_mesh2kzip(dest_path, mesh[0], mesh[1], None,
                            "nonglia_cc%d.ply" % kk)

    def write_gliapred_cnn(self, dest_path=None):
        if dest_path is None:
            dest_path = self.skeleton_kzip_path_views
        skel = load_skeleton(self.skeleton_kzip_path_views)[
            "sample_locations"]
        n_nodes = [n for n in skel.getNodes()]
        pred_coords = [n.getCoordinate() * np.array(self.scaling) for n in
                       n_nodes]
        preds = [int(n.data["glia_pred"]) for n in n_nodes]
        self._pred2mesh(pred_coords, preds, "gliapred.ply",
                        dest_path=dest_path,
                        colors=[[11, 129, 220, 255], [218, 73, 58, 255]])

    def predict_views_gliaSV(self, model, woglia=True, verbose=True,
                             overwrite=False, pred_key_appendix=""):
        # params = self.svs
        # if check_view_existence:
        #     ex_views = self.view_existence()
        #     if not np.all(ex_views):
        #         self.render_views(add_cellobjects=False)
        existing_preds = sm.start_multiprocess(ssh.glia_pred_exists, self.svs,
                                               nb_cpus=self.nb_cpus)
        if overwrite:
            missing_sos = self.svs
        else:
            missing_sos = np.array(self.svs)[~np.array(existing_preds,
                                                       dtype=np.bool)]
        if verbose:
            print "Predicting %d/%d SV's of SSV %d." % (len(missing_sos),
                                                        len(self.svs),
                                                        self.id)
            start = time.time()
        if len(missing_sos) == 0:
            return
        pred_key = "glia_probas"
        if woglia:
            pred_key += "woglia"
        pred_key += pred_key_appendix
        try:
            predict_sos_views(model, missing_sos, pred_key,
                              nb_cpus=self.nb_cpus, verbose=True,
                              woglia=woglia, raw_only=True)
        except KeyError:
            self.render_views(add_cellobjects=False)
            predict_sos_views(model, missing_sos, pred_key,
                              nb_cpus=self.nb_cpus, verbose=True,
                              woglia=woglia, raw_only=True)
        if verbose:
            end = time.time()
            print "Prediction of %d SV's took %0.2fs (incl. read/write). " \
                  "%0.4fs/SV" % (len(missing_sos), end - start,
                                 float(end - start) / len(missing_sos))
            # self.save_attributes(["gliaSV_model"], [model._fname])

    def predict_views_glia(self, model, thresh=0.5, dest_path=None,
                           woglia=False):
        raise (
        NotImplementedError, "Change code to use 'predict_sos_views'.")
        if dest_path is None:
            dest_path = self.skeleton_kzip_path_views
        loc_coords = self.sample_locations()
        views = self.load_views(woglia=woglia)
        assert len(views) == len(loc_coords)
        views = np.concatenate(views)
        loc_coords = np.concatenate(loc_coords)
        # get single connected component in img
        for i in range(len(views)):
            sing_cc = np.concatenate([single_conn_comp_img(views[i, 0, :1]),
                                      single_conn_comp_img(
                                          views[i, 0, 1:])])
            views[i, 0] = sing_cc
        probas = model.predict_proba(views)
        locs = skeleton.SkeletonAnnotation()
        locs.scaling = self.scaling
        locs.comment = "sample_locations"
        for ii, c in enumerate(loc_coords):
            n = skeleton.SkeletonNode().from_scratch(locs,
                                                     c[0] / self.scaling[0],
                                                     c[1] / self.scaling[1],
                                                     c[2] / self.scaling[2])
            n.data["glia_proba"] = probas[ii][1]
            n.data["glia_pred"] = int(probas[ii][1] > thresh)
            locs.addNode(n)
        write_skeleton(dest_path, [locs])
        self.save_attributes(["glia_model"], [model._fname])

    # ------------------------------------------------------------------ AXONESS
    def _load_skelfeatures(self, key):
        if not self.skeleton:
            self.load_skeleton()
        assert self.skeleton is not None, "Skeleton does not exist."
        if key in self.skeleton:
            assert len(self.skeleton["nodes"]) == len(self.skeleton[key]), \
                "Length of skeleton features is not equal to number of nodes."
            return self.skeleton[key]
        else:
            return None

    def _save_skelfeatures(self, key, features, overwrite=False):
        if not self.skeleton:
            self.load_skeleton()
        assert self.skeleton is not None, "Skeleton does not exist."
        if key in self.skeleton and not overwrite:
            raise ValueError("Key %s already exists in skeleton feature dict.")
        self.skeleton[key] = features
        assert len(self.skeleton["nodes"]) == len(self.skeleton[key]), \
            "Length of skeleton features is not equal to number of nodes."
        self.save_skeleton()

    def skel_features(self, feature_context_nm=5000):
        features = self._load_skelfeatures(feature_context_nm)
        if not features:
            features = ssh.extract_skel_features(self, feature_context_nm=
            feature_context_nm)
            self._save_skelfeatures(feature_context_nm, features)
        return features

    def write_axpred_rfc(self, dest_path=None, k=1):
        if dest_path is None:
            dest_path = self.skeleton_kzip_path
        if self.load_skeleton():
            if not "axoness" in self.skeleton:
                return False
            axoness = self.skeleton["axoness"].copy()
            axoness[self.skeleton["axoness"] == 1] = 0
            axoness[self.skeleton["axoness"] == 0] = 1
            print np.unique(axoness, return_counts=True)
            self._pred2mesh(self.skeleton["nodes"] * self.scaling, axoness,
                            k=k, dest_path=dest_path)

    def write_axpred_cnn(self, dest_path=None, k=1, pred_key_appendix=""):
        if dest_path is None:
            dest_path = self.skeleton_kzip_path_views
        preds = np.array(sm.start_multiprocess_obj("axoness_preds",
                [[sv, {"pred_key_appendix": pred_key_appendix}]
                 for sv in self.svs], nb_cpus=self.nb_cpus))
        preds = np.concatenate(preds)
        print "Collected axoness:", Counter(preds).most_common()
        locs = self.sample_locations()
        print "Collected locations."
        pred_coords = np.concatenate(locs)
        assert pred_coords.ndim == 2
        assert pred_coords.shape[1] == 3
        self._pred2mesh(pred_coords, preds, "axoness.ply", dest_path=dest_path,
                        k=k)
        # self.save_objects_to_kzip_sparse(obj_types=obj_types)

    def predict_nodes(self, sc, clf_name="rfc", feature_context_nm=5000,
                      avg_window=10000):
        """
        Predicting class c
        Parameters
        ----------
        sc : SkelClassifier
            Classifier to predict "axoness" or "spiness" for every node on
            self.skeleton["nodes"]. Target type is defined in SkelClassifier

        clf_name : str

        feature_context_nm : int

        avg_window : int
            Defines the maximum path length from a source node for collecting
            neighboring nodes to calculate an average prediction for
            the source node.

        Returns
        -------

        """
        assert sc.target_type in ["axoness", "spiness"]
        clf = sc.load_classifier(clf_name, feature_context_nm)
        probas = clf.predict_proba(self.skel_features(feature_context_nm))
        pred = []
        class_weights = np.array([1, 1, 1])
        for i_node in range(len(self.skeleton["nodes"])):
            paths = nx.single_source_dijkstra_path(self.weighted_graph, i_node,
                                                   avg_window)
            neighs = np.array(paths.keys(), dtype=np.int)
            c = np.argmax(np.sum(probas[neighs], axis=0) * class_weights)
            pred.append(c)

        self.skeleton[sc.target_type] = np.array(pred, dtype=np.int)
        self.save_skeleton(to_object=True, to_kzip=False)

    def axoness_for_coords(self, coords, radius_nm=4000):
        coords = np.array(coords)

        self.load_skeleton()
        kdtree = scipy.spatial.cKDTree(self.skeleton["nodes"] * self.scaling)
        close_node_ids = kdtree.query_ball_point(coords * self.scaling,
                                                 radius_nm)

        axoness_pred = []
        for i_coord in range(len(coords)):
            cls, cnts = np.unique(
                self.skeleton["axoness"][close_node_ids[i_coord]],
                return_counts=True)
            if len(cls) > 0:
                axoness_pred.append(cls[np.argmax(cnts)])
            else:
                axoness_pred.append(-1)

        return np.array(axoness_pred)

    def cnn_axoness_2_skel(self, dest_path=None, pred_key_appendix="", k=10):
        # probas = np.array(sm.start_multiprocess_obj("axoness_probas",
        #                           [[sv, {"pred_key_appendix": pred_key_appendix}] for sv in self.svs], nb_cpus=self.nb_cpus))
        preds = np.array(sm.start_multiprocess_obj("axoness_preds",
                [[sv, {"pred_key_appendix": pred_key_appendix}]
                 for sv in self.svs], nb_cpus=self.nb_cpus))
        preds = np.concatenate(preds)
        loc_coords = np.array(sm.start_multiprocess_obj("sample_locations",
                                  [[sv, ] for sv in self.svs], nb_cpus=self.nb_cpus))
        loc_coords = np.concatenate(loc_coords)
        assert len(loc_coords) == len(preds)
        # find kNN in loc_coords for every skeleton node and use their majority
        # prediction
        node_preds = colorcode_vertices(self.skeleton["nodes"], loc_coords,
                                        preds, colors=[0, 1, 2], k=k)
        self.skeleton["axoness_cnn"] = node_preds
        self.save_skeleton()

    # --------------------------------------------------------------- CELL TYPES
    # def predict_cell_type(self, ssd_version="ctgt", clf_name="rfc",
    #                       feature_context_nm=25000):
    #     # TODO: outsource to sbc module
    #     sc = sbc.SkelClassifier(working_dir=self.working_dir,
    #                             ssd_version=ssd_version,
    #                             create=False)
    #
    #     # if feature_context_nm is None:
    #     #     if np.linalg.norm(self.shape * self.scaling) > 24000:
    #     #         radius = 12000
    #     #     else:
    #     #         radius = nx.diameter(self.weighted_graph) / 2
    #     #
    #     #     if radius > 12000:
    #     #         radius = 12000
    #     #     elif radius < 2000:
    #     #         radius = 2000
    #     #
    #     #     avail_fc = sc.avail_feature_contexts(clf_name)
    #     #     feature_context_nm = avail_fc[np.argmin(np.abs(avail_fc - radius))]
    #
    #     features = ssh.extract_skel_features(feature_context_nm=
    #                                         feature_context_nm,
    #                                         downsample_to=200)
    #     clf = sc.load_classifier(clf_name, feature_context_nm)
    #
    #     probs = clf.predict_proba(features)
    #
    #     ratios = np.sum(probs, axis=0)
    #     ratios /= np.sum(ratios)
    #
    #     self.attr_dict["cell_type_ratios"] = ratios
    #     self.save_attr_dict()

    def gen_skel_from_sample_locs(self, dest_path=None, pred_key_appendix=""):
        try:
            if os.path.isfile(self.skeleton_path_views):
                return
            if dest_path is None:
                dest_path = self.skeleton_kzip_path_views
            locs = np.concatenate(self.sample_locations())
            edge_list = create_mst_skeleton(locs)
            self.skeleton = {}
            self.skeleton["nodes"] = locs / np.array(self.scaling)
            self.skeleton["edges"] = edge_list
            self.skeleton["diameters"] = np.ones(len(locs))
            ax_probas = np.array(sm.start_multiprocess_obj("axoness_probas",
                                 [[sv, {"pred_key_appendix": pred_key_appendix}] for sv in self.svs], nb_cpus=self.nb_cpus))
            ax_probas = np.concatenate(ax_probas)
            # first stage averaging
            curr_ax_preds = np.argmax(ax_probas, axis=1)
            ax_preds = np.zeros((len(locs)), dtype=np.int)
            for i_node in range(len(self.skeleton["nodes"])):
                paths = nx.single_source_dijkstra_path(self.weighted_graph, i_node,
                                                       30000)
                neighs = np.array(paths.keys(), dtype=np.int)
                cnt = Counter(curr_ax_preds[neighs])
                loc_average = np.zeros((3, ))
                for k, v in cnt.items():
                    loc_average[k] = v
                loc_average /= float(len(neighs))
                if (curr_ax_preds[i_node] == 2 and loc_average[2] >= 0.20) or (loc_average[2] >= 0.98):
                    ax_preds[i_node] = 2
                else:
                    ax_preds[i_node] = np.argmax(loc_average[:2])
            # second stage averaging, majority vote on every branch
            curr_ax_preds = np.array(ax_preds, dtype=np.int)
            edge_coords = locs[self.skeleton["edges"]]
            edge_ax = curr_ax_preds[self.skeleton["edges"]]
            edges = []
            for i in range(len(edge_coords)):
                if 2 in edge_ax[i]:
                    continue
                edges.append(self.skeleton["edges"][i])
            edges = np.array(edges)
            g = nx.Graph()
            g.add_edges_from(edges)
            ccs = nx.connected_components(g)
            for cc in ccs:
                curr_ixs = np.array(list(cc), dtype=np.int)
                cnt = Counter(ax_preds[curr_ixs])
                loc_average = np.zeros((3, ))
                for k, v in cnt.items():
                    loc_average[k] = v
                curr_ax_preds[curr_ixs] = np.argmax(loc_average)
            self.skeleton["axoness"] = curr_ax_preds
            # self.save_skeleton_to_kzip(dest_path=dest_path)
            write_obj2pkl(self.skeleton_path_views, self.skeleton)
        except Exception as e:
            if "null graph" in str(e) and len(self.sv_ids) == 2:
                print "Null graph error with 2 nodes, falling back to " \
                      "original classification and one edge."
                locs = np.concatenate(self.sample_locations())
                self.skeleton = {}
                self.skeleton["nodes"] = locs / np.array(self.scaling)
                self.skeleton["edges"] = np.array([[0, 1]])
                self.skeleton["diameters"] = np.ones(len(locs))
                ax_probas = np.array(sm.start_multiprocess_obj("axoness_probas",
                                                               [[sv, {
                                                                   "pred_key_appendix": pred_key_appendix}]
                                                                for sv in
                                                                self.svs],
                                                               nb_cpus=self.nb_cpus))
                ax_probas = np.concatenate(ax_probas)
                # first stage averaging
                curr_ax_preds = np.argmax(ax_probas, axis=1)
                self.skeleton["axoness"] = curr_ax_preds
                write_obj2pkl(self.skeleton_path_views, self.skeleton)
            else:
                print "Error %s occured with SSO %d  (%d SVs)." % (e, self.id,  len(self.sv_ids))

    def predict_celltype_cnn(self, model):
        ssh.predict_sso_celltype(self, model)


# ------------------------------------------------------------------------------
# SO rendering code

def render_sampled_sos_cc(sos, ws=(256, 128), verbose=False, woglia=True,
                          render_first_only=False, add_cellobjects=True,
                          overwrite=False, cellobjects_only=False):
    """
    Renders for each SV views at sampled locations (number is dependent on
    SV mesh size with scaling fact) from combined mesh of all SV.
    ----------------------------------------------------------------------
    USED FOR GLIA DETECTION
    ----------------------------------------------------------------------

    Parameters
    ----------
    sos : list of SegmentationObject
    ws : tuple
    verbose : bool
    woglia : bool
        without glia components
    render_first_only : bool
    add_cellobjects : bool
    overwrite : bool
    """
    # initilaize temporary SSO
    if not overwrite:
        if render_first_only:
            if sos[0].views_exist:
                sys.stdout.write("\r%d" % sos[0].id)
                sys.stdout.flush()
                return
        else:
            if np.all([sv.views_exist for sv in sos]):
                return
    sso = SuperSegmentationObject(np.random.randint(0, sys.maxint),
                                  create=False,
                                  working_dir=sos[0].working_dir,
                                  version="tmp", scaling=(10, 10, 20))
    sso._objects["sv"] = sos
    if render_first_only:
        coords = [sos[0].sample_locations()]
    else:
        coords = sso.sample_locations(cache=False)
    if add_cellobjects:
        sso.map_cellobjects()
    part_views = np.cumsum([0] + [len(c) for c in coords])
    views = render_sso_coords(sso, flatten_list(coords), add_cellobjects=add_cellobjects,
                              ws=ws, verbose=verbose, cellobjects_only=cellobjects_only)
    for i in range(len(coords)):
        v = views[part_views[i]:part_views[i+1]]
        if np.sum(v) == 0 or np.sum(v) == np.prod(v.shape):
            warnings.warn("Empty views detected after rendering.",
                          RuntimeWarning)
        sv_obj = sos[i]
        sv_obj.save_views(views=v, woglia=woglia, cellobjects_only=cellobjects_only)
        print sv_obj.segobj_dir


def render_so(so, ws=(256, 128), add_cellobjects=True, verbose=False):
    """
    Render super voxel views located at given locations. Does not write views
    to so.views_path

    Parameters
    ----------
    so : SegmentationObject
        super voxel ID
    ws : tuple of int
        Rendering windows size
    add_cellobjects : bool
    verbose : bool

    Returns
    -------
    np.array
        views
    """
    # initilaize temporary SSO for cellobject mapping purposes
    sso = SuperSegmentationObject(np.random.randint(0, sys.maxint),
                                  create=False,
                                  working_dir="/wholebrain/scratch/areaxfs/",
                                  version="tmp", scaling=(10, 10, 20))
    sso._objects["sv"] = [so]
    coords = sso.sample_locations(cache=False)[0]
    if add_cellobjects:
        sso.map_cellobjects()
    views = render_sso_coords(sso, coords, ws=ws, add_cellobjects=add_cellobjects,
                              verbose=verbose)
    return views


