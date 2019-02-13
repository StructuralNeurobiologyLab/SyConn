# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Philipp Schubert, Joergen Kornfeld

import glob
import networkx as nx
import numpy as np
import os
import re
import scipy.spatial
import shutil
import time
import tqdm
from collections import Counter
from scipy.misc import imsave
from knossos_utils import skeleton
from knossos_utils.skeleton_utils import load_skeleton as load_skeleton_kzip
from knossos_utils.skeleton_utils import write_skeleton as write_skeleton_kzip

try:
    from knossos_utils import mergelist_tools
except ImportError:
    from knossos_utils import mergelist_tools_fallback as mergelist_tools

from . import super_segmentation_helper as ssh
from .segmentation import SegmentationObject, SegmentationDataset
from ..proc.sd_proc import predict_sos_views
from .rep_helper import knossos_ml_from_sso, colorcode_vertices, \
    knossos_ml_from_svixs, subfold_from_ix_SSO
from ..handler import config
from ..handler.basics import write_txt2kzip, get_filepaths_from_dir, safe_copy, \
    coordpath2anno, load_pkl2obj, write_obj2pkl, flatten_list, chunkify
from ..backend.storage import CompressedStorage, MeshStorage
from ..proc.graphs import split_glia, split_subcc_join, create_graph_from_coords
from ..proc.meshes import write_mesh2kzip, merge_someshes, \
    compartmentalize_mesh, mesh2obj_file, write_meshes2kzip
from ..proc.rendering import render_sampled_sso, multi_view_sso, \
    render_sso_coords, render_sso_coords_index_views
from ..mp import batchjob_utils as qu
from ..mp import mp_utils as sm
from ..reps import log_reps
from .. import global_params


class SuperSegmentationObject(object):
    def __init__(self, ssv_id, version=None, version_dict=None,
                 working_dir=None, create=True, sv_ids=None, scaling=None,
                 object_caching=True, voxel_caching=True, mesh_caching=True,
                 view_caching=False, config=None, nb_cpus=1,
                 enable_locking=True, enable_locking_so=False, ssd_type="ssv"):
        """
        Class to represent an agglomeration of supervoxels (SegmentationObjects).

        Parameters
        ----------
        ssv_id : int
            unique SSV ID
        version : str
            version string identifier. if 'tmp' is used, no data will be saved
            to disk
        version_dict : dict
        working_dir : str
            path to working directory
        create : bool
            whether to create a folder to store cache data
        sv_ids : np.array
        scaling : tuple
        object_caching : bool
        voxel_caching : bool
        mesh_caching : bool
        view_caching : bool
        config : bool
        nb_cpus : int
            Number of cpus for parallel jobs. will only be used in some
            processing steps
        enable_locking : bool
            Locking flag for all SegmentationObjects
            (SV, mitochondria, vesicle clouds, ...)
        ssd_type : str
        """
        if version == 'temp':
            version = 'tmp'
        if version == "tmp":
            self._object_caching = False
            self._voxel_caching = False
            self._mesh_caching = False
            self._view_caching = False
            self.enable_locking = False
            create = False
        else:
            self.enable_locking = enable_locking
            self._object_caching = object_caching
            self._voxel_caching = voxel_caching
            self._mesh_caching = mesh_caching
            self._view_caching = view_caching

        self.enable_locking_so = enable_locking_so
        self.nb_cpus = nb_cpus
        self._id = ssv_id
        self.attr_dict = {}  # dict(mi=[], sj=[], vc=[], sv=[])

        self._type = ssd_type
        self._rep_coord = None
        self._size = None
        self._bounding_box = None
        self._config = config

        self._objects = {}
        self.skeleton = None
        self._voxels = None
        self._voxels_xy_downsampled = None
        self._voxels_downsampled = None
        self._rag = None

        # init mesh dicts
        self._meshes = {"sv": None, "sj": None, "syn_ssv": None,
                        "vc": None, "mi": None, "conn": None}

        self._views = None
        self._dataset = None
        self._weighted_graph = None
        self._sample_locations = None
        self._rot_mat = None
        self._label_dict = {}

        if sv_ids is not None:
            self.attr_dict["sv"] = sv_ids

        if working_dir is None:
            try:
                self._working_dir = global_params.wd
            except:
                raise Exception("No working directory (wd) specified in config")
        else:
            self._working_dir = working_dir

        if scaling is None:
            try:
                self._scaling = \
                    np.array(self.config.entries["Dataset"]["scaling"])
            except KeyError:
                msg = 'Scaling not set and could not be found in config ' \
                      'with entries: {}'.format(self.config.entries)
                log_reps.error(msg)
                raise KeyError(msg)
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
        return "%s/so_storage/%s/" % (self.ssds_dir,
                                      subfold_from_ix_SSO(self.id))

    @property
    def attr_dict_path(self):
        # Kept for backwards compatibility, remove if not needed anymore
        if os.path.isfile(self.ssv_dir + "atrr_dict.pkl"):
            return self.ssv_dir + "atrr_dict.pkl"
        return self.ssv_dir + "attr_dict.pkl"

    @property
    def skeleton_kzip_path(self):
        return self.ssv_dir + "skeleton.k.zip"

    @property
    def skeleton_kzip_path_views(self):
        return self.ssv_dir + "skeleton_views.k.zip"

    @property
    def objects_dense_kzip_path(self):
        """Identifier of cell organell overlays"""
        return self.ssv_dir + "objects_overlay.k.zip"

    @property
    def skeleton_path(self):
        """Identifier of SSV skeleton"""
        return self.ssv_dir + "skeleton.pkl"

    @property
    def edgelist_path(self):
        """Identifier of SSV graph"""
        return self.ssv_dir + "edge_list.bz2"

    @property
    def view_path(self):
        """Identifier of view storage"""
        return self.ssv_dir + "views.pkl"

    @property
    def mesh_dc_path(self):
        """Identifier of mesh storage"""
        return self.ssv_dir + "mesh_dc.pkl"

    @property
    def vlabel_dc_path(self):
        """Identifier of vertex label storage"""
        return self.ssv_dir + "vlabel_dc.pkl"

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
    def load_mesh(self, mesh_type):
        if not mesh_type in self._meshes:
            return None
        if self._meshes[mesh_type] is None:
            if not self.mesh_caching:
                return self._load_obj_mesh(mesh_type)
            self._meshes[mesh_type] = self._load_obj_mesh(mesh_type)
        return self._meshes[mesh_type]

    @property
    def mesh(self):
        return self.load_mesh("sv")

    @property
    def sj_mesh(self):
        return self.load_mesh("sj")

    @property
    def vc_mesh(self):
        return self.load_mesh("vc")

    @property
    def mi_mesh(self):
        return self.load_mesh("mi")

    def label_dict(self, data_type='vertex'):
        if data_type == 'vertex':
            if data_type in self._label_dict:
                pass
            else:
                self._label_dict[data_type] = CompressedStorage(
                    None if self.version == 'tmp' else self.vlabel_dc_path)
            return self._label_dict[data_type]
        else:
            raise ValueError('Label dict for data type "{}" not supported.'
                             ''.format(data_type))

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

    def weighted_graph(self, add_node_attr=()):
        """
        Creates a distance weighted graph representation of the SSV skeleton.

        Parameters
        ----------
        add_node_attr : tuple of keys

        Returns
        -------
        nx.Graph
        """
        if self._weighted_graph is None or np.any([len(nx.get_node_attributes(
                self._weighted_graph, k)) == 0 for k in add_node_attr]):
            if self.skeleton is None:
                self.load_skeleton()

            node_scaled = self.skeleton["nodes"] * self.scaling
            edges = np.array(self.skeleton["edges"], dtype=np.uint)
            edge_coords = node_scaled[edges]
            weights = np.linalg.norm(edge_coords[:, 0] - edge_coords[:, 1],
                                     axis=1)
            self._weighted_graph = nx.Graph()
            self._weighted_graph.add_weighted_edges_from(
                [(edges[ii][0], edges[ii][1], weights[ii]) for
                 ii in range(len(weights))])
            for k in add_node_attr:
                dc = {}
                for n in self._weighted_graph.nodes():
                    dc[n] = self.skeleton[k][n]
                nx.set_node_attributes(self._weighted_graph, dc, k)
        return self._weighted_graph

    @property
    def config(self):
        if self._config is None:
            self._config = global_params.config
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
        mesh_dc = MeshStorage(self.mesh_dc_path,
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
                    log_reps.debug(np.sum(sv.voxels), sv.size)
                    box = [sv.bounding_box[0] - self.bounding_box[0],
                           sv.bounding_box[1] - self.bounding_box[0]]

                    voxels[box[0][0]: box[1][0],
                    box[0][1]: box[1][1],
                    box[0][2]: box[1][2]][sv.voxels] = True
                else:
                    log_reps.warning("missing voxels from %d" % sv.id)

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
    def rag(self):
        if self._rag is None:
            self._rag = self.load_sv_graph()
        return self._rag

    @property
    def compartment_meshes(self):
        if not "axon" in self._meshes:
            self._load_mesh_compartments()
        return {k: self._meshes[k] for k in ["axon", "dendrite", "soma"]}

    def _load_mesh_compartments(self, rewrite=False):
        mesh_dc = MeshStorage(self.mesh_dc_path,
                              disable_locking=not self.enable_locking)
        if not "axon" in mesh_dc or rewrite:
            mesh_compartments = compartmentalize_mesh(self)
            mesh_dc["axon"] = mesh_compartments["axon"]
            mesh_dc["dendrite"] = mesh_compartments["dendrite"]
            mesh_dc["soma"] = mesh_compartments["soma"]
            mesh_dc.push()
        comp_meshes = {k: mesh_dc[k] for k in ["axon", "dendrite", "soma"]}
        self._meshes.update(comp_meshes)

    def load_voxels_downsampled(self, downsampling=(2, 2, 1), nb_threads=10):
        return ssh.load_voxels_downsampled(self, downsampling=downsampling,
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
        return SegmentationObject(obj_id=obj_id, obj_type=obj_type,
                                  version=self.version_dict[obj_type],
                                  working_dir=self.working_dir, create=False,
                                  scaling=self.scaling,
                                  enable_locking=self.enable_locking_so)

    def get_seg_dataset(self, obj_type):
        return SegmentationDataset(obj_type, version_dict=self.version_dict,
                                   version=self.version_dict[obj_type],
                                   scaling=self.scaling,
                                   working_dir=self.working_dir)

    def load_attr_dict(self):
        try:
            self.attr_dict = load_pkl2obj(self.attr_dict_path)
            return 0
        except (IOError, EOFError):
            return -1

    def load_sv_graph(self):
        if os.path.isfile(self.edgelist_path):
            G = nx.read_edgelist(self.edgelist_path, nodetype=np.uint)
        # # Might be useful as soon as global graph path is available
        # else:
        #     if os.path.isfile("{}neuron_rag_{}.bz2".format(
        #             self.working_dir, global_params.rag_suffix)):
        #         G_glob = nx.read_edgelist(self.working_dir + "neuron_rag.bz2",
        #                                   nodetype=np.uint)
        #         G = nx.Graph()
        #         cc = nx.node_connected_component(G_glob, self.sv_ids[0])
        #         assert len(set(cc).difference(set(self.sv_ids))) == 0, \
        #             "SV IDs in graph differ from SSV SVs."
        #         for e in G_glob.edges(cc):
        #             G.add_edge(*e)
        else:
            raise ValueError("Could not find graph data for SSV {}."
                             "".format(self.id))
        if len(set(list(G.nodes())).difference(set(self.sv_ids))) != 0:
            msg = "SV IDs in graph differ from SSV SVs."
            log_reps.error(msg)
            raise ValueError(msg)
        # create graph with SV nodes
        new_G = nx.Graph()
        for e in G.edges():
            new_G.add_edge(self.get_seg_obj("sv", e[0]),
                           self.get_seg_obj("sv", e[1]))
        return new_G

    def load_edgelist(self):
        g = self.load_sv_graph()
        return list(g.edges())

    def _load_obj_mesh(self, obj_type="sv", rewrite=False):
        """
        TODO: Currently does not support color array!

        Parameters
        ----------
        obj_type : str
        rewrite : bool

        Returns
        -------
        np.array, np.array, np.array
            ind, vert, normals
        """
        if not rewrite and self.mesh_exists(obj_type) and not \
                self.version == "tmp":
            mesh_dc = MeshStorage(self.mesh_dc_path,
                                  disable_locking=not self.enable_locking)
            if len(mesh_dc[obj_type]) == 3:
                ind, vert, normals = mesh_dc[obj_type]
            else:
                ind, vert = mesh_dc[obj_type]
                normals = np.zeros((0,), dtype=np.float32)
        else:
            ind, vert, normals = merge_someshes(self.get_seg_objects(obj_type),
                                                nb_cpus=self.nb_cpus)
            if not self.version == "tmp":
                mesh_dc = MeshStorage(self.mesh_dc_path, read_only=False,
                                      disable_locking=not self.enable_locking)
                mesh_dc[obj_type] = [ind, vert, normals]
                mesh_dc.push()
        # Changed vertex dtype to float32, as they actually should. PS, 22Oct2018
        return np.array(ind, dtype=np.int), np.array(vert, dtype=np.float32), \
               np.array(normals, dtype=np.float32)

    def _load_obj_mesh_compr(self, obj_type="sv"):
        mesh_dc = MeshStorage(self.mesh_dc_path,
                              disable_locking=not self.enable_locking)
        return mesh_dc._dc_intern[obj_type]

    def load_svixs(self):
        if not os.path.isfile(self.edgelist_path):
            log_reps.warn("Edge list of SSO {} does not exist. Return empty "
                          "list.".format(self.id))
            return []
        edges = self.load_edgelist()
        return np.unique(np.concatenate([[a.id, b.id] for a, b in edges]))

    def save_attr_dict(self):
        if self.version == 'tmp':
            log_reps.warning('"save_attr_dict" called but this SSV '
                             'has version "tmp", attribute dict will'
                             ' not be saved to disk.')
            return
        try:
            orig_dc = load_pkl2obj(self.attr_dict_path)
        except IOError:
            orig_dc = {}
        orig_dc.update(self.attr_dict)
        write_obj2pkl(self.attr_dict_path + '.tmp', orig_dc)
        shutil.move(self.attr_dict_path + '.tmp', self.attr_dict_path)

    def save_attributes(self, attr_keys, attr_values):
        """
        Writes attributes to attribute dict on file system. Does not care about
        self.attr_dict.

        Parameters
        ----------
        attr_keys : tuple of str
        attr_values : tuple of items
        """
        if self.version == 'tmp':
            log_reps.warning('"save_attributes" called but this SSV '
                             'has version "tmp", attributes will'
                             ' not be saved to disk.')
            return
        if not hasattr(attr_keys, "__len__"):
            attr_keys = [attr_keys]
        if not hasattr(attr_values, "__len__"):
            attr_values = [attr_values]
        try:
            attr_dict = load_pkl2obj(self.attr_dict_path)
        except IOError as e:
            if not "[Errno 13] Permission denied" in str(e):
                pass
            else:
                log_reps.warn("Could not load SSO attributes to %s due to "
                              "missing permissions." % self.attr_dict_path,
                              RuntimeWarning)
            attr_dict = {}
        for k, v in zip(attr_keys, attr_values):
            attr_dict[k] = v
        try:
            write_obj2pkl(self.attr_dict_path, attr_dict)
        except IOError as e:
            if not "[Errno 13] Permission denied" in str(e):
                raise (IOError, e)
            else:
                log_reps.warn("Could not save SSO attributes to %s due to "
                              "missing permissions." % self.attr_dict_path,
                              RuntimeWarning)

    def attr_exists(self, attr_key):
        return attr_key in self.attr_dict

    def lookup_in_attribute_dict(self, attr_key):
        if attr_key in self.attr_dict:
            return self.attr_dict[attr_key]
        # TODO: this is somehow arbitrary
        elif len(self.attr_dict) <= 4:
            if self.load_attr_dict() == -1:
                return None
        if attr_key in self.attr_dict:
            return self.attr_dict[attr_key]
        else:
            return None

    def load_so_attributes(self, obj_type, attr_keys, nb_cpus=None):
        """
        Loads list of attributes from all SOs of certain type.
        Ordering of attributes within each key is the same as self.svs.

        Parameters
        ----------
        obj_type : str
        attr_keys : List[str]
        nb_cpus : int

        Returns
        -------
        list[list]
            list for each key in 'attr_keys'
        """
        if nb_cpus is None:
            nb_cpus = self.nb_cpus
        params = [[obj, dict(attr_keys=attr_keys)]
                  for obj in self.get_seg_objects(obj_type)]
        attr_values = sm.start_multiprocess_obj('load_attributes', params,
                                                nb_cpus=nb_cpus)
        attr_values = [el for sublist in attr_values for el in sublist]
        return [attr_values[ii::len(attr_keys)] for ii in range(len(attr_keys))]

    def calculate_size(self, nb_cpus=None):
        """
        Calculates SSV size.
        Parameters
        ----------
        nb_cpus
        """
        self._size = np.sum(self.load_so_attributes('sv', ['size'],
                                                    nb_cpus=nb_cpus))

    def calculate_bounding_box(self, nb_cpus=None):
        """
        Calculates SSV bounding box (and size).

        Parameters
        ----------
        nb_cpus : int
        """
        if len(self.svs) == 0:
            self._bounding_box = np.zeros((2, 3), dtype=np.int)
            self._size = 0
            return

        self._bounding_box = np.ones((2, 3), dtype=np.int) * np.inf
        self._size = np.inf
        bounding_boxes, sizes = self.load_so_attributes(
            'sv', ['bounding_box', 'size'], nb_cpus=nb_cpus)
        self._size = np.sum(sizes)
        self._bounding_box[0] = np.min(bounding_boxes, axis=0)[0]
        self._bounding_box[1] = np.max(bounding_boxes, axis=0)[1]
        self._bounding_box = self._bounding_box.astype(np.int)

    def calculate_skeleton(self, force=False):
        self.load_skeleton()
        if self.skeleton is not None and len(self.skeleton["nodes"]) != 0 \
                and not force:
            return
        ssh.create_sso_skeleton(self)
        if len(self.skeleton["nodes"]) == 0:
            log_reps.warning("%s has zero nodes." % repr(self))
        self.save_skeleton()

    def save_skeleton_to_kzip(self, dest_path=None, additional_keys=None):
        if type(additional_keys) == str:
            additional_keys = [additional_keys]
        try:
            if self.skeleton is None:
                self.load_skeleton()
            if additional_keys is not None:
                for k in additional_keys:
                    assert k in self.skeleton, "Additional key %s is not " \
                                               "part of SSV %d self.skeleton.\nAvailable keys: %s" % \
                                               (k, self.id, repr(self.skeleton.keys()))
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
                if additional_keys is not None:
                    for k in additional_keys:
                        skel_nodes[-1].data[k] = self.skeleton[k][i_node]

                a.addNode(skel_nodes[-1])

            for edge in self.skeleton["edges"]:
                a.addEdge(skel_nodes[edge[0]], skel_nodes[edge[1]])

            if dest_path is None:
                dest_path = self.skeleton_kzip_path
            write_skeleton_kzip(dest_path, [a])
        except Exception as e:
            log_reps.warning("[SSO: %d] Could not load/save skeleton:\n%s" % (self.id, repr(e)))

    def save_objects_to_kzip_sparse(self, obj_types=("sj", "mi", "vc"),
                                    dest_path=None):
        annotations = []
        for obj_type in obj_types:
            assert obj_type in self.attr_dict
            map_ratio_key = "mapping_%s_ratios" % obj_type
            if not map_ratio_key in self.attr_dict.keys():
                log_reps.warning("%s not yet mapped. Object nodes are not "
                                 "written to k.zip." % obj_type)
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
        write_skeleton_kzip(dest_path, annotations)

    def save_objects_to_kzip_dense(self, obj_types):
        if os.path.exists(self.objects_dense_kzip_path[:-6]):
            shutil.rmtree(self.objects_dense_kzip_path[:-6])
        if os.path.exists(self.objects_dense_kzip_path):
            os.remove(self.objects_dense_kzip_path)

        for obj_type in obj_types:
            so_objs = self.get_seg_objects(obj_type)
            for so_obj in so_objs:
                so_obj.save_kzip(path=self.objects_dense_kzip_path,
                                 write_id=self.dense_kzip_ids[obj_type])

    def total_edge_length(self):
        if self.skeleton is None:
            self.load_skeleton()
        nodes = self.skeleton["nodes"]
        edges = self.skeleton["edges"]
        return np.sum([np.linalg.norm(
            self.scaling * (nodes[e[0]] - nodes[e[1]])) for e in edges])

    def save_skeleton(self, to_kzip=False, to_object=True):
        if self.version == 'tmp':
            log_reps.warning('"save_skeleton" called but this SSV '
                             'has version "tmp", skeleton will'
                             ' not be saved to disk.')
            return
        if to_object:
            write_obj2pkl(self.skeleton_path, self.skeleton)

        if to_kzip:
            self.save_skeleton_to_kzip()

    def load_skeleton(self):
        try:
            self.skeleton = load_pkl2obj(self.skeleton_path)
            # stored as uint32, if used for computations
            # e.g. edge length then it will overflow
            self.skeleton["nodes"] = self.skeleton["nodes"].astype(np.float32)
            return True
        except:
            if global_params.config.allow_skel_gen:
                self.gen_skel_from_sample_locs()
                return True
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
                    list(mappings[obj_type].keys())
                self.attr_dict["mapping_%s_ratios" % obj_type] = \
                    list(mappings[obj_type].values())

        if save:
            self.save_attr_dict()

    def apply_mapping_decision(self, obj_type, correct_for_background=True,
                               lower_ratio=None, upper_ratio=None,
                               sizethreshold=None, save=True):
        assert obj_type in self.version_dict

        self.load_attr_dict()
        if not "mapping_%s_ratios" % obj_type in self.attr_dict:
            log_reps.error("No mapping ratios found")
            return

        if not "mapping_%s_ids" % obj_type in self.attr_dict:
            log_reps.error("no mapping ids found")
            return

        if lower_ratio is None:
            try:
                lower_ratio = self.config.entries["LowerMappingRatios"][
                    obj_type]
            except KeyError:
                msg = "Lower ratio undefined"
                log_reps.error(msg)
                raise ValueError(msg)

        if upper_ratio is None:
            try:
                upper_ratio = self.config.entries["UpperMappingRatios"][
                    obj_type]
            except:
                log_reps.critical("Upper ratio undefined - 1. assumed")
                upper_ratio = 1.

        if sizethreshold is None:
            try:
                sizethreshold = self.config.entries["Sizethresholds"][obj_type]
            except KeyError:
                msg = "Size threshold undefined"
                log_reps.error(msg)
                raise ValueError(msg)

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
            obj = SegmentationObject(candidate_id, obj_type=obj_type,
                                     version=self.version_dict[obj_type],
                                     working_dir=self.working_dir, config=self.config)
            if obj.size > sizethreshold:
                self.attr_dict[obj_type].append(candidate_id)

        if save:
            self.save_attr_dict()

    def _map_cellobjects(self, obj_types=None, save=True):
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
        self._meshes = None
        self.skeleton = None

    def preprocess(self):
        self.load_attr_dict()
        self._map_cellobjects()
        _ = self._load_obj_mesh(obj_type="mi", rewrite=False)
        _ = self._load_obj_mesh(obj_type="sj", rewrite=False)
        _ = self._load_obj_mesh(obj_type="vc", rewrite=False)
        _ = self._load_obj_mesh(obj_type="sv", rewrite=False)
        self.calculate_skeleton()
        self.clear_cache()

    def copy2dir(self, dest_dir, safe=True):
        """
        Usually dest_dir set to the 'ssv_dir' attribute of the target SSV (ssv_target).
        E.g. if one wants to use this SSV (self), lets call it ssv_orig,
        for GT purposes, then on can call ssv_orig.copy2dir(ssv_target.ssv_dir)
         and all data contained in the SSD of ssv_orig will be copied to
         the SSD of ssv_target.
        Parameters
        ----------
        dest_dir : str
            target directory to which data will be copied
        safe : bool
            if True, will not overwrite existing data
        """
        # get all files in home directory
        fps = get_filepaths_from_dir(self.ssv_dir, ending=["pkl", "k.zip"])
        fnames = [os.path.split(fname)[1] for fname in fps]
        # Open the file and raise an exception if it exists
        if not os.path.isdir(dest_dir):
            os.makedirs(dest_dir)
        for i in range(len(fps)):
            src_filename = fps[i]
            dest_filename = dest_dir + "/" + fnames[i]
            try:
                safe_copy(src_filename, dest_filename, safe=safe)
                log_reps.info("Copied %s to %s." % (src_filename, dest_filename))
            except Exception as e:
                log_reps.error("Skipped", fnames[i], str(e))
                pass
        self.load_attr_dict()
        if os.path.isfile(dest_dir + "/attr_dict.pkl"):
            dest_attr_dc = load_pkl2obj(dest_dir + "/attr_dict.pkl")
        else:
            dest_attr_dc = {}
        dest_attr_dc.update(self.attr_dict)
        write_obj2pkl(dest_dir + "/attr_dict.pkl", dest_attr_dc)

    def partition_cc(self, max_nb_sv=None, lo_first_n=None):
        """
        Splits connected component into subgraphs.

        Parameters
        ----------
        max_nb_sv : int
            Number of SV per CC
        lo_first_n : int
            do not use first n traversed nodes for new bfs traversals.

        Returns
        -------
        dict
        """
        if lo_first_n is None:
            lo_first_n = global_params.SUBCC_CHUNK_SIZE_BIG_SSV
        if max_nb_sv is None:
            max_nb_sv = global_params.SUBCC_SIZE_BIG_SSV + 2 * (lo_first_n - 1)
        init_g = self.rag
        partitions = split_subcc_join(init_g, max_nb_sv, lo_first_n=lo_first_n)
        return partitions

    # -------------------------------------------------------------------- VIEWS
    def save_views(self, views, view_key="views"):
        """
        This will only save views on SSV level and not for each individual SV!

        Parameters
        ----------
        views : np.array
        view_key : str

        Returns
        -------

        """
        if self.version == 'tmp':
            log_reps.warning('"save_views" called but this SSV '
                             'has version "tmp", views will'
                             ' not be saved to disk.')
            return
        view_dc = CompressedStorage(self.view_path, read_only=False,
                                    disable_locking=not self.enable_locking)
        view_dc[view_key] = views
        view_dc.push()

    def load_views(self, view_key=None, woglia=True, raw_only=False,
                   force_reload=False, cache_default_views=False, nb_cpus=None,
                   ignore_missing=False, index_views=False):
        """
        Load views which were stored by 'save_views' given the key 'view_key',
        i.e. this operates on SSV level.
        If 'view_key' is not given, then returns the views of the SSV's
        SVs (woglia and raw_only is then used).

        Parameters
        ----------
        woglia : bool
        view_key : str
        raw_only : bool
        force_reload : bool
            if True will force reloading the SV views.
        ignore_missing : bool
            if True, it will not raise KeyError if SV does not exist
        cache_default_views : bool
            Stores views in SSV cache if True.
        nb_cpus : int
        index_views : bool
            load index views

        Returns
        -------
        np.array
            Concatenated views for each SV in self.svs with shape
             [N_LOCS, N_CH, N_VIEWS, X, Y]
        """
        view_dc = CompressedStorage(self.view_path, read_only=True,
                                    disable_locking=not self.enable_locking)
        # Disable view caching on SSV
        # if view_key is None:
        #     if index_views:
        #         view_key = "%d%d%d" % (int(woglia), int(raw_only), int(index_views))
        #     else:  # only kept for backwards compat.
        #         view_key = "%d%d" % (int(woglia), int(raw_only))
        # else:
        #     # check if random sv has view_key
        #     sv_ad = self.svs[0]
        #     random_sv_contains_viewkey = sv_ad.views_exist(woglia=woglia,
        #                                                    view_key=view_key)
        #     if not view_key in view_dc and not random_sv_contains_viewkey:
        #         raise KeyError("Given view key '{}' does not exist in view di"
        #                        "ctionary of SSV {} at {}. Existing keys: {}\n"
        #                        "".format(view_key, self.id, self.view_path,
        #                                  str(view_dc.keys())))
        if view_key in view_dc and not force_reload:
            return view_dc[view_key]
        del view_dc  # delete previous initialized view dictionary
        params = [[sv, {'woglia': woglia, 'raw_only': raw_only, 'index_views':
            index_views, 'ignore_missing': ignore_missing,
                        'view_key': view_key}] for sv in self.svs]
        # load views from underlying SVs
        views = sm.start_multiprocess_obj("load_views", params,
                                          nb_cpus=self.nb_cpus
                                          if nb_cpus is None else nb_cpus)
        views = np.concatenate(views)

        # Disable view caching on SSV

        # view_dc = CompressedStorage(self.view_path, read_only=False,
        #                             disable_locking=not self.enable_locking)
        # if cache_default_views:
        #     log_reps.info("Loaded and cached default views of SSO %d at %s."
        #                   " (raw_only: %d, woglia: %d; #views: %d)" % (
        #         self.id, self.view_path, int(raw_only), int(woglia), len(views)))
        #     view_dc[view_key] = views
        #     view_dc.push()
        return views

    def view_existence(self, woglia=True):
        params = [[sv, {"woglia": woglia}] for sv in self.svs]
        so_views_exist = sm.start_multiprocess_obj("views_exist", params,
                                                   nb_cpus=self.nb_cpus)
        return so_views_exist

    def render_views(self, add_cellobjects=False, verbose=False,
                     qsub_pe=None, overwrite=False, cellobjects_only=False,
                     woglia=True, skip_indexviews=False, qsub_co_jobs=300,
                     resume_job=False):
        """
        Renders views for each SV based on SSV context and stores them
        on SV level. Usually only used once: for initial glia or axoness
        prediction.
        THIS WILL BE SAVED DISTRIBUTED AT EACH SV VIEW DICTIONARY
        IS NOT CACHED IN THE ATTR-DICT OF THE SSV.
        See '_render_rawviews' for storing the views in the SSV folder.

        Parameters
        ----------
        add_cellobjects : bool
        qsub_pe : str
        overwrite : bool
        cellobjects_only : bool
        woglia : bool
        skip_indexviews : bool
            Index views will not be generated, import for e.g. initial SSV
            rendering prior to glia-splitting.
        qsub_co_jobs : int
            Number of parallel jobs if QSUB is used
        resume_job : bool
        Returns
        -------

        """
        if len(self.sv_ids) > global_params.RENDERING_MAX_NB_SV:
            part = self.partition_cc()
            log_reps.info('Partitioned hugh SSV into {} subgraphs with each {}'
                          ' SVs.'.format(len(part), len(part[0])))
            if not overwrite:  # check existence of glia preds
                views_exist = np.array(self.view_existence(), dtype=np.int)
                log_reps.info("Rendering huge SSO. {}/{} views left to process"
                              ".".format(np.sum(views_exist == 0), len(self.svs)))
                ex_dc = {}
                for ii, k in enumerate(self.svs):
                    ex_dc[k] = views_exist[ii]
                for k in part.keys():
                    if ex_dc[k]:  # delete SO's with existing views
                        del part[k]
                        continue
                del ex_dc
            else:
                log_reps.info("Rendering huge SSO. {} SVs left to process"
                              ".".format(len(self.svs)))
            params = [[so.id for so in el] for el in part]
            if qsub_pe is None:
                raise RuntimeError('QSUB has to be enabled when processing '
                                   'huge SSVs.')
            elif qu.batchjob_enabled():
                params = chunkify(params, 2000)
                so_kwargs = {'version': self.svs[0].version,
                             'working_dir': self.working_dir,
                             'obj_type': self.svs[0].type}
                render_kwargs = {"overwrite": overwrite, 'woglia': woglia,
                                 "render_first_only": global_params.SUBCC_CHUNK_SIZE_BIG_SSV,
                                 'add_cellobjects': add_cellobjects,
                                 "cellobjects_only": cellobjects_only,
                                 'skip_indexviews': skip_indexviews}
                params = [[par, so_kwargs, render_kwargs] for par in params]
                qu.QSUB_script(
                    params, "render_views_partial", suffix="_SSV{}".format(self.id),
                    pe=qsub_pe, queue=None, script_folder=None, n_cores=2,
                    n_max_co_processes=qsub_co_jobs, resume_job=resume_job)
            else:
                raise Exception("QSUB not available")
        else:
            # render raw data
            render_sampled_sso(self, add_cellobjects=add_cellobjects,
                               verbose=verbose, overwrite=overwrite,
                               cellobjects_only=cellobjects_only, woglia=woglia)
            if skip_indexviews:
                return
            # render index views
            render_sampled_sso(self, verbose=verbose, overwrite=overwrite,
                               index_views=True)

    def _render_indexviews(self, nb_views=2, save=True, force_recompute=False,
                           verbose=False):
        """
        Render SSV raw views in case non-default number of views is required.
        Will be stored in SSV view dict. Default raw/index/prediction views are
        stored decentralized in corresponding SVs.

        Parameters
        ----------
        nb_views : int
        save : bool
        force_recompute : bool
        verbose : bool

        Returns
        -------
        np.array
        """
        if not force_recompute:
            try:
                views = self.load_views('index{}'.format(nb_views))
                if not save:
                    return views
                else:
                    return
            except KeyError:
                pass
        locs = np.concatenate(self.sample_locations(cache=False))
        start = time.time()
        if self._rot_mat is None:
            index_views, rot_mat = render_sso_coords_index_views(
                self, locs, nb_views=nb_views, verbose=verbose,
                return_rot_matrices=True)
            self._rot_mat = rot_mat
        else:
            index_views = render_sso_coords_index_views(
                self, locs, nb_views=nb_views, verbose=verbose, rot_mat=self._rot_mat)
        end_ix_views = time.time()
        log_reps.debug("Rendering views took {:.2f} s. {:.2f} views/s".format(
            end_ix_views - start, len(index_views) / (end_ix_views - start)))
        log_reps.debug("Mapping rgb values to vertex indices took {:.2f}s.".format(
            time.time() - end_ix_views))
        if not save:
            return index_views
        self.save_views(index_views, "index{}".format(nb_views))

    def _render_rawviews(self, nb_views=2, save=True, force_recompute=False,
                         add_cellobjects=True, verbose=False):
        """
        Render SSV raw views in case non-default number of views is required.
        Will be stored in SSV view dict. Default raw/index/prediction views are
        stored decentralized in corresponding SVs.

        Parameters
        ----------
        nb_views : int
        save : bool
        force_recompute : bool
        add_cellobjects : bool
        verbose : bool

        Returns
        -------
        np.array
        """
        if not force_recompute:
            try:
                views = self.load_views('raw{}'.format(nb_views))
                if not save:
                    return views
                return
            except KeyError:
                pass
        locs = np.concatenate(self.sample_locations(cache=False))
        if self._rot_mat is None:
            views, rot_mat = render_sso_coords(self, locs, verbose=verbose,
                                               add_cellobjects=add_cellobjects,
                                               nb_views=nb_views, return_rot_mat=True)
            self._rot_mat = rot_mat
        else:
            views = render_sso_coords(self, locs, verbose=verbose,
                                      add_cellobjects=add_cellobjects,
                                      nb_views=nb_views, rot_mat=self._rot_mat)
        if save:
            self.save_views(views, "raw{}".format(nb_views))
        else:
            return views

    def predict_semseg(self, m, semseg_key, nb_views=None, verbose=False):
        """
        Generates label views based on input model and stores it under the key
        'semseg_key', either within the SSV's SVs or in an extra view-storage
        according to input parameters:
        Default situation:
            semseg_key = 'spiness', nb_views=None
            This will load the raw views stored at the SSV's SVs.
        Non-default:
            semseg_key = 'spiness4', nb_views=4
            This requires to run 'self._render_rawviews(nb_views=4)'
            This method then has to be called like:
                'self.predict_semseg('spiness4', nb_views=4)'

        Parameters
        ----------
        semseg_key : str
        nb_views : Optional[int]
        dest_path : str
        k : int
        verbose : bool
        """
        # views have shape [N, 4, 2, 128, 256]
        if nb_views is not None and nb_views != global_params.NB_VIEWS:
            # treat as special view rendering
            try:
                views = self.load_views('raw{}'.format(nb_views))
            except KeyError:
                log_reps.warning('Could not find raw-views. Re-rendering now.')
                self._render_rawviews(nb_views)
                views = self.load_views('raw{}'.format(nb_views))
            if len(views) != len(np.concatenate(self.sample_locations(cache=False))):
                raise ValueError("Unequal number of views and redering locations.")
            labeled_views = ssh.predict_views_semseg(views, m, verbose=verbose)
            assert labeled_views.shape[2] == nb_views, \
                "Predictions have wrong shape."
            self.save_views(labeled_views, semseg_key)
        else:
            # treat as default view rendering
            views = self.load_views()
            assert len(views) == len(
                np.concatenate(self.sample_locations(cache=False))), \
                "Unequal number of views and redering locations."
            # re-order number of views according to SV rendering locations
            # TODO: move view reordering to 'pred_svs_semseg', check other usages before!
            locs = self.sample_locations()
            reordered_views = []
            cumsum = np.cumsum([0] + [len(el) for el in locs])
            for ii in range(len(locs)):
                sv_views = views[cumsum[ii]:cumsum[ii + 1]]
                reordered_views.append(sv_views)
            if self.version == 'tmp':
                log_reps.warning('"predict_semseg" called but this SSV '
                                 'has version "tmp", results will'
                                 ' not be saved to disk.')
            ssh.pred_svs_semseg(m, reordered_views, semseg_key, self.svs,
                                nb_cpus=self.nb_cpus, verbose=verbose,
                                return_pred=self.version == 'tmp')  # do not write to disk

    def semseg2mesh(self, semseg_key, dest_path=None, nb_views=None, k=1,
                    force_overwrite=False):
        """
        Generates vertex labels and stores it in the SSV's label storage under
        the key 'semseg_key'.
        Default situation:
            semseg_key = 'spiness', nb_views=None
            This will load the index and label views stored at the SSV's SVs.
        Non-default:
            semseg_key = 'spiness4', nb_views=4
            This requires to run 'self._render_rawviews(nb_views=4)',
            'self._render_indexviews(nb_views=4)' and 'predict_semseg(MODEL,
            'spiness4', nb_views=4)
            This method then has to be called like:
                'self.semseg2mesh('spiness4', nb_views=4)'

        Parameters
        ----------
        semseg_key : str
        nb_views : Optional[int]
        dest_path : str
        k : int
        force_overwrite : bool
        """
        if 'spiness' in semseg_key:
            # colors are only needed if dest_path is given
            cols = np.array([[0.6, 0.6, 0.6, 1], [0.9, 0.2, 0.2, 1],
                             [0.1, 0.1, 0.1, 1], [0.05, 0.6, 0.6, 1],
                             [0.9, 0.9, 0.9, 1], [0.1, 0.1, 0.9, 1]])
            cols = (cols * 255).astype(np.uint8)
            return ssh.semseg2mesh(self, semseg_key, nb_views, dest_path, k,
                                   cols, force_overwrite=force_overwrite)
        else:
            raise ValueError('Sematic segmentation of "" is not supported.'
                             ''.format(semseg_key))

    def semseg_for_coords(self, coords, semseg_key, k=5, ds_vertices=20):
        """
        Dies not need to be axoness, it supports any attribut stored in
        self.skeleton.

        Parameters
        ----------
        coords : np.array
            Voxel coordinates, unscaled! [N, 3]
        semseg_key : str
        k : int
            Number of nearest neighbors (NN) during k-NN classification
        ds_vertices : int
            striding factor for vertices

        Returns
        -------
        np.array
            Same length as coords. For every coordinate in coords returns the
            majority label within radius_nm
        """
        coords = np.array(coords) * self.scaling
        vertex_labels = self.label_dict('vertex')[semseg_key][::ds_vertices]
        vertices = self.mesh[1].reshape((-1, 3))[::ds_vertices]
        if len(vertex_labels) != len(vertices):
            raise ValueError('Size of vertices and their labels does not match!')
        maj_vote = colorcode_vertices(coords, vertices, vertex_labels, k=k,
                                      return_color=False, nb_cpus=self.nb_cpus)
        return maj_vote

    def get_spine_compartments(self, semseg_key='spiness', k=1,
                               min_spine_cc_size=None, dest_folder=None):
        """
        Retrieve connected components of vertex spine predictions

        Parameters
        ----------
        semseg_key : str
        k : int
            number of nearest neighbors for majority label vote (smoothing of
             classification results).
        min_spine_cc_size : int
            Minimum number of vertices to consider a connected component a
             valid object
        dest_folder : str
            Default is None, else provide a path (str) to a folder. The mean
            location and size of the head and neck connected components will be
             stored as numpy array file (npy)

        Returns
        -------
        np.array, np.array, np.array, np.array
            Neck locations, neck sizes, head locations, head sizes. Location
             and size arrays have the same ordering.

        """
        if min_spine_cc_size is None:
            min_spine_cc_size = global_params.min_spine_cc_size
        vertex_labels = self.label_dict('vertex')[semseg_key]
        vertices = self.mesh[1].reshape((-1, 3))
        max_dist = global_params.min_edge_dist_spine_graph
        g = create_graph_from_coords(vertices, force_single_cc=True,
                                     max_dist=max_dist)
        g_orig = g.copy()
        for e in g_orig.edges():
            l0 = vertex_labels[e[0]]
            l1 = vertex_labels[e[1]]
            if l0 != l1:
                g.remove_edge(e[0], e[1])
        log_reps.info("Starting connected components for SSV {}."
                      "".format(self.id))
        all_ccs = list(sorted(nx.connected_components(g), key=len,
                              reverse=True))
        log_reps.info("Finished connected components for SSV {}."
                      "".format(self.id))
        sizes = np.array([len(c) for c in all_ccs])
        thresh_ix = np.argmax(sizes < min_spine_cc_size)
        all_ccs = all_ccs[:thresh_ix]
        sizes = sizes[:thresh_ix]
        cc_labels = []
        cc_coords = []
        for c in all_ccs:
            curr_v_ixs = list(c)
            curr_v_l = vertex_labels[curr_v_ixs]
            curr_v_c = vertices[curr_v_ixs]
            if len(np.unique(curr_v_l)) != 1:
                msg = '"get_spine_compartments": Connected component ' \
                      'contains multiple labels.'
                log_reps.error(msg)
                raise ValueError(msg)
            cc_labels.append(curr_v_l[0])
            cc_coords.append(np.mean(curr_v_c, axis=0))
        cc_labels = np.array(cc_labels)
        cc_coords = np.array(cc_coords)
        np.random.seed(0)
        neck_c = (cc_coords[cc_labels == 0] / self.scaling).astype(np.uint)
        neck_s = sizes[cc_labels == 0]
        head_c = (cc_coords[cc_labels == 1] / self.scaling).astype(np.uint)
        head_s = sizes[cc_labels == 1]
        if dest_folder is not None:
            np.save("{}/neck_coords_ssv{}_k{}_{}_ccsize{}.npy".format(
                dest_folder, self.id, k, semseg_key, min_spine_cc_size), neck_c)
            np.save("{}/head_coords_ssv{}_k{}_{}_ccsize{}.npy".format(
                dest_folder, self.id, k, semseg_key, min_spine_cc_size), head_c)
        return neck_c, neck_s, head_c, head_s

    def sample_locations(self, force=False, cache=True, verbose=False):
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
        if self.version == 'tmp' and cache:
            cache = False
        if not force and self._sample_locations is not None:
            return self._sample_locations
        if not force:
            if self.attr_exists("sample_locations"):
                return self.attr_dict["sample_locations"]
        if verbose:
            start = time.time()
        params = [[sv, {"force": force}] for sv in self.svs]

        # list of arrays
        locs = sm.start_multiprocess_obj("sample_locations", params,
                                         nb_cpus=self.nb_cpus)
        if cache:
            self.save_attributes(["sample_locations"], [locs])
        if verbose:
            dur = time.time() - start
            log_reps.info("Sampling locations from {} SVs took {:.2f}s."
                          " {.4f}s/SV (incl. read/write)".format(
                len(self.svs), dur, dur / len(self.svs)))
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
        write_skeleton_kzip(self.skeleton_kzip_path, a)

    def write_locations2kzip(self, dest_path=None):
        if dest_path is None:
            dest_path = self.skeleton_kzip_path_views
        loc = np.concatenate(self.sample_locations())
        new_anno = coordpath2anno(loc, add_edges=False)
        new_anno.setComment("sample_locations")
        write_skeleton_kzip(dest_path, [new_anno])

    def mergelist2kzip(self, dest_path=None):
        self.load_attr_dict()
        kml = knossos_ml_from_sso(self)
        if dest_path is None:
            dest_path = self.skeleton_kzip_path
        write_txt2kzip(dest_path, kml, "mergelist.txt")

    def mesh2kzip(self, dest_path=None, obj_type="sv", ext_color=None):
        """
        Writes mesh of SSV to kzip as .ply file.
        Parameters
        ----------
        dest_path :
        obj_type : str
            'sv' for cell surface, 'mi': mitochondria, 'vc': vesicle clouds,
            'sj': synaptic junctions
        ext_color : np.array of scalar
            If scalar, it has to be an integer between 0 and 255.
            If array, it has to be of type uint/int and of shape (N, 4) while N
            is the number of vertices of the SSV cell surface mesh:
            N = len(self.mesh[1].reshape((-1, 3)))

        Returns
        -------

        """
        color = None
        if dest_path is None:
            dest_path = self.skeleton_kzip_path
        if obj_type == "sv":
            mesh = self.mesh
        elif obj_type == "sj":
            mesh = self.sj_mesh
            # color = np.array([int(0.849 * 255), int(0.138 * 255),
            #                   int(0.133 * 255), 255])
        elif obj_type == "vc":
            mesh = self.vc_mesh
            # color = np.array([int(0.175 * 255), int(0.585 * 255),
            #                   int(0.301 * 255), 255])
        elif obj_type == "mi":
            mesh = self.mi_mesh
            # color = np.array([0, 153, 255, 255])
        else:
            mesh = self._meshes[obj_type]
        if ext_color is not None:
            if type(ext_color) is list:
                ext_color = np.array(ext_color)
            if np.isscalar(ext_color) and ext_color == 0:
                color = None
            elif np.isscalar(ext_color):
                color = ext_color
            elif type(ext_color) is np.ndarray:
                if ext_color.ndim != 2:
                    msg = "'ext_color' is numpy array of dimension {}." \
                          " Only 2D arrays are allowed.".format(ext_color.ndim)
                    log_reps.error(msg)
                    raise ValueError(msg)
                if ext_color.shape[1] == 3:
                    # add alpha channel
                    alpha_sh = (len(ext_color), 1)
                    alpha_arr = (np.ones(alpha_sh) * 255).astype(ext_color.dtype)
                    ext_color = np.concatenate([ext_color, alpha_arr], axis=1)
                color = ext_color.flatten()
        write_mesh2kzip(dest_path, mesh[0], mesh[1], mesh[2], color,
                        ply_fname=obj_type + ".ply")

    def meshes2kzip(self, dest_path=None, sv_color=None):
        """
        Writes SV, mito, vesicle cloud and synaptic junction meshes to k.zip

        Parameters
        ----------
        dest_path : str
        sv_color : np.array
            array with RGBA values or None to use default values
            (see 'mesh2kzip')

        Returns
        -------

        """
        if dest_path is None:
            dest_path = self.skeleton_kzip_path
        for ot in ["sj", "vc", "mi",
                   "sv"]:  # determins rendering order in KNOSSOS
            self.mesh2kzip(obj_type=ot, dest_path=dest_path, ext_color=sv_color if
            ot == "sv" else None)

    def mesh2file(self, dest_path=None, center=None, color=None):
        """
        Writes mesh to file (e.g. .ply, .stl, .obj) via the 'openmesh' library.
        If possible, writes it as binary.

        Parameters
        ----------
        dest_path : str
        center : np.array
            scaled center coordinates (in nm)
        color: np.array
            Either single color (will be applied to all vertices) or
            per-vertex color array
        """
        mesh2obj_file(dest_path, self.mesh, center=center, color=color)

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
        self.meshes2kzip(dest_path=dest_path, sv_color=sv_color)
        self.mergelist2kzip(dest_path=dest_path)

    def write_svmeshes2kzip(self, dest_path=None):
        if dest_path is None:
            dest_path = self.skeleton_kzip_path
        for ii, sv in enumerate(self.svs):
            mesh = sv.mesh
            write_mesh2kzip(dest_path, mesh[0], mesh[1], mesh[2], None,
                            ply_fname="sv%d.ply" % ii)

    def _svattr2mesh(self, dest_path, attr_key, cmap, normalize_vals=False):
        sv_attrs = np.array([sv.lookup_in_attribute_dict(attr_key).squeeze()
                             for sv in self.svs])
        if normalize_vals:
            min_val = sv_attrs.min()
            sv_attrs -= min_val
            sv_attrs /= sv_attrs.max()
        ind, vert, norm, col = merge_someshes(self.svs, color_vals=sv_attrs,
                                              cmap=cmap)
        write_mesh2kzip(dest_path, ind, vert, norm, col, "%s.ply" % attr_key)

    def svprobas2mergelist(self, key="glia_probas", dest_path=None):
        if dest_path is None:
            dest_path = self.skeleton_kzip_path
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
            N x 3; scaled to nm
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
        if not ply_fname.endswith(".ply"):
            ply_fname += ".ply"
        mesh = self.mesh
        col = colorcode_vertices(mesh[1].reshape((-1, 3)), pred_coords,
                                 preds, colors=colors, k=k)
        if dest_path is None or ply_fname is None:
            if not dest_path is None and ply_fname is None:
                log_reps.warning("Specify 'ply_fanme' in order to save colored"
                                 " mesh to k.zip.")
            return mesh[0], mesh[1], col
        else:
            write_mesh2kzip(dest_path, mesh[0], mesh[1], mesh[2], col,
                            ply_fname=ply_fname)

    # --------------------------------------------------------------------- GLIA
    def gliaprobas2mesh(self, dest_path=None, pred_key_appendix=""):
        if dest_path is None:
            dest_path = self.skeleton_kzip_path_views
        import seaborn as sns
        mcmp = sns.diverging_palette(250, 15, s=99, l=60, center="dark",
                                     as_cmap=True)
        self._svattr2mesh(dest_path, "glia_probas" + pred_key_appendix,
                          cmap=mcmp)

    def gliapred2mesh(self, dest_path=None, thresh=None, pred_key_appendix=""):
        if thresh is None:
            thresh = global_params.glia_thresh
        self.load_attr_dict()
        for sv in self.svs:
            sv.load_attr_dict()
        glia_svs = [sv for sv in self.svs if sv.glia_pred(thresh, pred_key_appendix) == 1]
        nonglia_svs = [sv for sv in self.svs if
                       sv.glia_pred(thresh, pred_key_appendix) == 0]
        if dest_path is None:
            dest_path = self.skeleton_kzip_path_views
        mesh = merge_someshes(glia_svs)
        neuron_mesh = merge_someshes(nonglia_svs)
        write_meshes2kzip(dest_path, [mesh[0], neuron_mesh[0]], [mesh[1], neuron_mesh[1]],
                          [mesh[2], neuron_mesh[2]], [None, None],
                          ["glia_%0.2f.ply" % thresh, "nonglia_%0.2f.ply" % thresh])

    def gliapred2mergelist(self, dest_path=None, thresh=None,
                           pred_key_appendix=""):
        if thresh is None:
            thresh = global_params.glia_thresh
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

    def gliasplit(self, dest_path=None, recompute=False, thresh=None,
                  write_shortest_paths=False, verbose=False,
                  pred_key_appendix=""):
        if thresh is None:
            thresh = global_params.glia_thresh
        if recompute or not (self.attr_exists("glia_svs") and
                             self.attr_exists("nonglia_svs")):
            if write_shortest_paths:
                shortest_paths_dir = os.path.split(dest_path)[0]
            else:
                shortest_paths_dir = None
            if verbose:
                log_reps.debug("Splitting glia in SSV {} with {} SV's.".format(
                    self.id, len(self.svs)))
                start = time.time()
            nonglia_ccs, glia_ccs = split_glia(self, thresh=thresh,
                                               pred_key_appendix=pred_key_appendix, verbose=verbose,
                                               shortest_paths_dest_dir=shortest_paths_dir)
            if verbose:
                log_reps.debug("Splitting glia in SSV %d with %d SV's finished "
                               "after %.4gs." % (self.id, len(self.svs),
                                                 time.time() - start))
            non_glia_ccs_ixs = [[so.id for so in nonglia] for nonglia in
                                nonglia_ccs]
            glia_ccs_ixs = [[so.id for so in glia] for glia in glia_ccs]
            self.attr_dict["glia_svs"] = glia_ccs_ixs
            self.attr_dict["nonglia_svs"] = non_glia_ccs_ixs
            self.save_attributes(["glia_svs", "nonglia_svs"],
                                 [glia_ccs_ixs, non_glia_ccs_ixs])

    def gliasplit2mesh(self, dest_path=None):
        """

        Parameters
        ----------
        dest_path :

        Returns
        -------

        """
        # TODO: adapt writemesh2kzip to work with multiple writes to same file or use write_meshes2kzip here.
        if dest_path is None:
            dest_path = self.skeleton_kzip_path_views
        # write meshes of CC's
        glia_ccs = self.attr_dict["glia_svs"]
        for kk, glia in enumerate(glia_ccs):
            mesh = merge_someshes([self.get_seg_obj("sv", ix) for ix in
                                   glia])
            write_mesh2kzip(dest_path, mesh[0], mesh[1], mesh[2], None,
                            "glia_cc%d.ply" % kk)
        non_glia_ccs = self.attr_dict["nonglia_svs"]
        for kk, nonglia in enumerate(non_glia_ccs):
            mesh = merge_someshes([self.get_seg_obj("sv", ix) for ix in
                                   nonglia])
            write_mesh2kzip(dest_path, mesh[0], mesh[1], mesh[2], None,
                            "nonglia_cc%d.ply" % kk)

    def write_gliapred_cnn(self, dest_path=None):
        if dest_path is None:
            dest_path = self.skeleton_kzip_path_views
        skel = load_skeleton_kzip(self.skeleton_kzip_path_views)[
            "sample_locations"]
        n_nodes = [n for n in skel.getNodes()]
        pred_coords = [n.getCoordinate() * np.array(self.scaling) for n in
                       n_nodes]
        preds = [int(n.data["glia_pred"]) for n in n_nodes]
        self._pred2mesh(pred_coords, preds, "gliapred.ply",
                        dest_path=dest_path,
                        colors=[[11, 129, 220, 255], [218, 73, 58, 255]])

    def predict_views_gliaSV(self, model, verbose=True,
                             pred_key_appendix=""):
        if self.version == 'tmp':
            log_reps.warning('"predict_views_gliaSV" called but this SSV '
                             'has version "tmp", results will'
                             ' not be saved to disk.')
        start = time.time()
        pred_key = "glia_probas"
        pred_key += pred_key_appendix
        # try:
        predict_sos_views(model, self.svs, pred_key,
                          nb_cpus=self.nb_cpus, verbose=verbose,
                          woglia=False, raw_only=True,
                          return_proba=self.version == 'tmp')  # do not write to disk
        # except KeyError:
        #     self.render_views(add_cellobjects=False, woglia=False)
        #     predict_sos_views(model, self.svs, pred_key,
        #                       nb_cpus=self.nb_cpus, verbose=verbose,
        #                       woglia=False, raw_only=True)
        end = time.time()
        log_reps.debug("Prediction of %d SV's took %0.2fs (incl. read/write). "
                       "%0.4fs/SV" % (len(self.svs), end - start,
                                      float(end - start) / len(self.svs)))

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

    def _save_skelfeatures(self, k, features, overwrite=False):
        if not self.skeleton:
            self.load_skeleton()
        assert self.skeleton is not None, "Skeleton does not exist."
        if k in self.skeleton and not overwrite:
            raise ValueError("Key {} already exists in skeleton"
                             " feature dict.".format(k))
        self.skeleton[k] = features
        assert len(self.skeleton["nodes"]) == len(self.skeleton[k]), \
            "Length of skeleton features is not equal to number of nodes."
        self.save_skeleton()

    def skel_features(self, feature_context_nm, overwrite=False):
        features = self._load_skelfeatures(feature_context_nm)
        if features is None or overwrite:
            if not "assoc_sj" in self.skeleton:
                ssh.associate_objs_with_skel_nodes(self)
            features = ssh.extract_skel_features(self, feature_context_nm=
            feature_context_nm)
            self._save_skelfeatures(feature_context_nm, features,
                                    overwrite=True)
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
            self._pred2mesh(self.skeleton["nodes"] * self.scaling, axoness,
                            k=k, dest_path=dest_path)

    def skelproperty2mesh(self, key, dest_path=None, k=1):
        if self.skeleton is None:
            self.load_skeleton()
        if dest_path is None:
            dest_path = self.skeleton_kzip_path
        self._pred2mesh(self.skeleton["nodes"] * self.scaling,
                        self.skeleton[key], k=k, dest_path=dest_path,
                        ply_fname=key + ".ply")

    def predict_nodes(self, sc, clf_name="rfc", feature_context_nm=None,
                      max_dist=0, leave_out_classes=()):
        """
        Predicting class c
        Parameters
        ----------
        sc : SkelClassifier
            Classifier to predict "axoness" or "spiness" for every node on
            self.skeleton["nodes"]. Target type is defined in SkelClassifier
        clf_name : str
        feature_context_nm : int
        max_dist : int
            Defines the maximum path length from a source node for collecting
            neighboring nodes to calculate an average prediction for
            the source node.

        Returns
        -------

        """
        if feature_context_nm is None:
            feature_context_nm = global_params.SKEL_FEATURE_CONTEXT[sc.target_type]
        assert sc.target_type in ["axoness", "spiness"]
        clf = sc.load_classifier(clf_name, feature_context_nm, production=True,
                                 leave_out_classes=leave_out_classes)
        probas = clf.predict_proba(self.skel_features(feature_context_nm))
        pred = []
        if max_dist == 0:
            pred = np.argmax(probas, axis=1)
        else:
            for i_node in range(len(self.skeleton["nodes"])):
                paths = nx.single_source_dijkstra_path(
                    self.weighted_graph(), i_node, max_dist)
                neighs = np.array(list(paths.keys()), dtype=np.int)
                c = np.argmax(np.sum(probas[neighs], axis=0))
                pred.append(c)

        pred_key = "%s_fc%d_avgwind%d" % (sc.target_type, feature_context_nm,
                                          max_dist)
        self.skeleton[pred_key] = np.array(pred, dtype=np.int)
        self.skeleton[pred_key + "_proba"] = np.array(probas, dtype=np.float32)
        self.save_skeleton(to_object=True, to_kzip=False)

    def axoness_for_coords(self, coords, radius_nm=4000, pred_type="axoness"):
        """
        Dies not need to be axoness, it supports any attribut stored in self.skeleton.

        Parameters
        ----------
        coords : np.array
            Voxel coordinates, unscaled! [N, 3]
        radius_nm : float
        pred_type : str

        Returns
        -------
        np.array
            Same length as coords. For every coordinate in coords returns the
            majority label within radius_nm
        """
        coords = np.array(coords)
        self.load_skeleton()
        if self.skeleton is None or len(self.skeleton["nodes"]) == 0:
            log_reps.warn("Skeleton did not exist for SSV {} (size: {}; rep. coord.: "
                          "{}).".format(self.id, self.size, self.rep_coord))
            return [-1]
        if pred_type not in self.skeleton:  # for glia SSV this does not exist.
            return [-1]
        kdtree = scipy.spatial.cKDTree(self.skeleton["nodes"] * self.scaling)
        close_node_ids = kdtree.query_ball_point(coords * self.scaling,
                                                 radius_nm)
        axoness_pred = []
        for i_coord in range(len(coords)):
            curr_close_node_ids = close_node_ids[i_coord]
            if len(curr_close_node_ids) == 0:
                dist, curr_close_node_ids = kdtree.query(coords * self.scaling)
                log_reps.info(
                    "Couldn't find skeleton nodes within {} nm. Using nearest "
                    "one with distance {} nm. SSV ID {}, coordinate at {}."
                    "".format(radius_nm, dist[0], self.id, coords[i_coord]))
            cls, cnts = np.unique(
                np.array(self.skeleton[pred_type])[np.array(curr_close_node_ids)],
                return_counts=True)
            if len(cls) > 0:
                axoness_pred.append(cls[np.argmax(cnts)])
            else:
                log_reps.info("Did not find any skeleton node within {} nm at {}."
                              " SSV {} (size: {}; rep. coord.: {}).".format(
                    radius_nm, i_coord, self.id, self.size, self.rep_coord))
                axoness_pred.append(-1)

        return np.array(axoness_pred)

    def predict_views_axoness(self, model, verbose=False,
                              pred_key_appendix=""):
        start = time.time()
        pred_key = "axoness_probas"
        pred_key += pred_key_appendix
        if self.version == 'tmp':
            log_reps.warning('"predict_views_axoness" called but this SSV '
                             'has version "tmp", results will'
                             ' not be saved to disk.')
        try:
            predict_sos_views(model, self.svs, pred_key,
                              nb_cpus=self.nb_cpus, verbose=verbose,
                              woglia=True, raw_only=False,
                              return_proba=self.version == 'tmp')  # do not write to disk
        except KeyError:
            log_reps.error("Re-rendering SSV %d (%d SVs), because views are missing."
                           % (self.id, len(self.sv_ids)))
            self.render_views(add_cellobjects=True, woglia=True, overwrite=True)
            predict_sos_views(model, self.svs, pred_key,
                              nb_cpus=self.nb_cpus, verbose=verbose,
                              woglia=True, raw_only=False,
                              return_proba=self.version == 'tmp')  # do not write to disk)
        end = time.time()
        log_reps.debug("Prediction of %d SV's took %0.2fs (incl. read/write). "
                       "%0.4fs/SV" % (len(self.svs), end - start,
                                      float(end - start) / len(self.svs)))

    def cnn_axoness2skel(self, **kwargs):
        locking_tmp = self.enable_locking
        self.enable_locking = False  # all SV operations are read-only
        # (enable_locking is inherited by sso.svs);
        # SSV operations not, but SSO file structure is not chunked
        res = ssh._cnn_axoness2skel(self, **kwargs)
        self.enable_locking = locking_tmp
        return res

    def average_node_axoness_views(self, **kwargs):
        locking_tmp = self.enable_locking
        self.enable_locking = False  # all SV operations are read-only
        # (enable_locking is inherited by sso.svs);
        # SSV operations not, but SSO file structure is not chunked
        res = ssh._average_node_axoness_views(self, **kwargs)
        self.enable_locking = locking_tmp
        return res

    def axoness2mesh(self, dest_path, k=1, pred_key_appendix=''):
        ssh.write_axpred_cnn(self, pred_key_appendix=pred_key_appendix, k=k,
                             dest_path=dest_path)

    # --------------------------------------------------------------- CELL TYPES
    def predict_cell_type(self, ssd_version="ctgt", clf_name="rfc",
                          feature_context_nm=25000):
        raise DeprecationWarning('This method is deprecated. Use '
                                 '"predict_nodes" instead!')

    def gen_skel_from_sample_locs(self, dest_path=None):
        """

        Parameters
        ----------
        dest_path : str
            Path to kzip
        """
        if dest_path is None:
            dest_path = self.skeleton_kzip_path_views
        locs = np.concatenate(self.sample_locations())
        g = create_graph_from_coords(locs, mst=True)
        if g.number_of_edges() == 1:
            edge_list = np.array(list(g.edges()))
        else:
            edge_list = np.array(g.edges())
        del g
        assert edge_list.ndim == 2
        self.skeleton = dict()
        self.skeleton["nodes"] = locs / np.array(self.scaling)
        self.skeleton["edges"] = edge_list
        self.skeleton["diameters"] = np.ones(len(locs))
        self.save_skeleton()
        if dest_path is not None:
            self.save_skeleton_to_kzip(dest_path=dest_path)

    def predict_celltype_cnn(self, model, **kwargs):
        ssh.predict_sso_celltype(self, model, **kwargs)

    def render_ortho_views_vis(self, dest_folder=None, colors=None, ws=(2048, 2048),
                               obj_to_render=("sv", )):
        if colors is None:
            colors = {"sv": (0.5, 0.5, 0.5, 0.5), "mi": (0, 0, 1, 1),
                      "vc": (0, 1, 0, 1), "sj": (1, 0, 0, 1)}
        views = multi_view_sso(self, colors, ws=ws, obj_to_render=obj_to_render)
        if dest_folder:
            for ii, v in enumerate(views):
                imsave("%s/SSV_%d_%d.png" % (dest_folder, self.id, ii), v)
        else:
            return views

    def majority_vote(self, prop_key, max_dist):
        """
        Smoothes (average using sliding window of 2 times max_dist and majority
        vote) property prediction in annotation, whereas for axoness somata are
        untouched.

        Parameters
        ----------
        prop_key : str
            which property to average
        max_dist : int
            maximum distance (in nm) for sliding window used in majority voting
        """
        assert prop_key in self.skeleton, "Given key does not exist in self.skeleton"
        prop_array = self.skeleton[prop_key]
        assert prop_array.squeeze().ndim == 1, "Property array has to be 1D."
        maj_votes = np.zeros_like(prop_array)
        for ii in range(len(self.skeleton["nodes"])):
            paths = nx.single_source_dijkstra_path(self.weighted_graph(),
                                                   ii, max_dist)
            neighs = np.array(list(paths.keys()), dtype=np.int)
            labels, cnts = np.unique(prop_array[neighs], return_counts=True)
            maj_label = labels[np.argmax(cnts)]
            maj_votes[ii] = maj_label
        return maj_votes


# ------------------------------------------------------------------------------
# SO rendering code

def render_sampled_sos_cc(sos, ws=(256, 128), verbose=False, woglia=True,
                          render_first_only=0, add_cellobjects=True,
                          overwrite=False, cellobjects_only=False,
                          index_views=False):
    """
    Renders for each SV views at sampled locations (number is dependent on
    SV mesh size with scaling fact) from combined mesh of all SV.

    Parameters
    ----------
    sos : list of SegmentationObject
    ws : tuple
    verbose : bool
    woglia : bool
        without glia components
    render_first_only : int
    add_cellobjects : bool
    overwrite : bool
    """
    # initilaize temporary SSO
    if not overwrite:
        if render_first_only:
            if np.all([sos[ii].views_exist(woglia=woglia) for ii in range(render_first_only)]):
                return
        else:
            if np.all([sv.views_exist(woglia=woglia) for sv in sos]):
                return
    sso = SuperSegmentationObject(sos[0].id,
                                  create=False, enable_locking=False,
                                  working_dir=sos[0].working_dir,
                                  version="tmp", scaling=sos[0].scaling)
    sso._objects["sv"] = sos
    if render_first_only:
        coords = [sos[ii].sample_locations() for ii in range(render_first_only)]
    else:
        coords = sso.sample_locations(cache=False)
    if add_cellobjects:
        sso._map_cellobjects(save=False)
    part_views = np.cumsum([0] + [len(c) for c in coords])
    if index_views:
        views = render_sso_coords_index_views(sso, flatten_list(coords),
                                              ws=ws, verbose=verbose)
    else:
        views = render_sso_coords(sso, flatten_list(coords),
                                  add_cellobjects=add_cellobjects,
                                  ws=ws, verbose=verbose,
                                  cellobjects_only=cellobjects_only)
    for i in range(len(coords)):
        v = views[part_views[i]:part_views[i + 1]]
        if np.sum(v) == 0 or np.sum(v) == np.prod(v.shape):
            log_reps.warn("Empty views detected after rendering.",
                          RuntimeWarning)
        sv_obj = sos[i]
        sv_obj.save_views(views=v, woglia=woglia, index_views=index_views,
                          cellobjects_only=cellobjects_only)


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
    sso = SuperSegmentationObject(so.id,
                                  create=False,
                                  working_dir=so.working_dir,
                                  version="tmp", scaling=so.scaling)
    sso._objects["sv"] = [so]
    coords = sso.sample_locations(cache=False)[0]
    if add_cellobjects:
        sso._map_cellobjects()
    views = render_sso_coords(sso, coords, ws=ws, add_cellobjects=add_cellobjects,
                              verbose=verbose)
    return views


def merge_axis02(arr):
    arr = arr.swapaxes(1, 2)  # swap channel and view axis
    # N, 2, 4, 128, 256
    orig_shape = arr.shape
    # reshape to predict single projections
    arr = arr.reshape([-1] + list(orig_shape[2:]))
    return arr


def celltype_predictor(args):
    """

    Parameters
    ----------
    args :

    Returns
    -------

    """
    from ..handler.prediction import get_celltype_model
    ssv_ids = args
    # randomly initialize gpu
    m = get_celltype_model(init_gpu=0)
    pbar = tqdm.tqdm(total=len(ssv_ids))
    missing_ssvs = []
    for ix in ssv_ids:
        ssv = SuperSegmentationObject(ix, working_dir=global_params.config.working_dir)
        ssv.nb_cpus = 1
        try:
            ssh.predict_sso_celltype(ssv, m, overwrite=True)
        except Exception as e:
            missing_ssvs.append((ssv.id, e))
            log_reps.error(repr(e))
        pbar.update(1)
    pbar.close()
    return missing_ssvs
