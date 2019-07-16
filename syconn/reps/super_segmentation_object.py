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
from collections import Counter, defaultdict
from scipy import spatial
from typing import Optional, Dict, List, Tuple, Any, Union, TYPE_CHECKING, Iterable
if TYPE_CHECKING:
    from torch.nn import Module
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
from ..handler.basics import write_txt2kzip, get_filepaths_from_dir, safe_copy, \
    coordpath2anno, load_pkl2obj, write_obj2pkl, flatten_list, chunkify, data2kzip
from ..backend.storage import CompressedStorage, MeshStorage
from ..proc.graphs import split_glia, split_subcc_join, create_graph_from_coords
from ..proc.meshes import write_mesh2kzip, merge_someshes, \
    compartmentalize_mesh, mesh2obj_file, write_meshes2kzip
from ..proc.rendering import render_sampled_sso, multi_view_sso, \
    render_sso_coords, render_sso_coords_index_views
from ..mp import batchjob_utils as qu
from ..mp import mp_utils as sm
from ..reps import log_reps
from ..handler.config import DynConfig
from .. import global_params
MeshType = Union[Tuple[np.ndarray, np.ndarray, np.ndarray],
                 Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]


class SuperSegmentationObject(object):
    """
    Class instances represent individual neuron reconstructions, defined by a
    list of agglomerated supervoxels (see :class:`~syconn.reps.segmentation.SegmentationObject`).

    Examples:
        This class can be used to create a cell reconstruction object after successful executing
        :func:`syconn.exec.exec_multiview.run_create_neuron_ssd` as follows::

            from syconn import global_params
            # import SuperSegmentationObject and SuperSegmentationDataset
            from syconn.reps.super_segmentation import *
            # set the current working directory SyConn-wide
            global_params.wd = '~/SyConn/example_cube1/'

            ssd = SuperSegmentationDataset()
            cell_reconstr_ids = ssd.ssv_ids
            # call constructor explicitly ...
            cell = SuperSegmentationObject(cell_reconstr_ids[0])
            # ... or via the SuperSegmentationDataset - both contain the same data
            cell = ssd.get_super_segmentation_object(cell_reconstr_ids[0])
            # inspect existing attributes
            cell.load_attr_dict()
            print(cell.attr_dict.keys())
            # get

        Subsequent analysis steps (see the ``SyConn/scripts/example_run/start.py``) augment the
        cell reconstruction with more properties::

            # to iterate over all cell reconstructions use the generator:
            for cell in ssd.ssvs:
                # e.g. collect some analysis results
                cell.load_attr_dict()
                n_synapses = len(cell.syn_ssv)
                celltype = cell.attr_dict["celltype_cnn_e3"]
                ...
                # write out cell mesh
                cell.mesh2kzip('~/cell{}_mesh.k.zip'.format(cell.id))
                # write out cell mesh and meshes of all existing cell organelles
                cell.meshes2kzip('~/cell{}_meshes.k.zip'.format(cell.id))
                # color the cell mesh according to a semantic prediction
                cell.semseg2mesh(semseg_key='spiness', dest_path='~/cell{}_spines.k.zip'.format(cell.id))


        See also `SyConn/docs/api.md` (WIP).

    Attributes:
        attr_dict: Attribute dictionary which serves as a general-purpose container. Accessed via
            the :class:`~syconn.backend.storage.AttributeDict` interface. After successfully
            executing :func:`syconn.exec.exec_multiview.run_create_neuron_ssd`
            and subsequent analysis steps (see the ``SyConn/scripts/example_run/start.py``) the
            following keys are present in :attr:`~attr_dict`:
                * 'id': ID array, identical to :py:attr:`~ssv_ids`. All other properties have the same
                  ordering as this array, i.e. if SSV with ID 1234 has index 42 in the 'id'-array
                  you will find its properties at index 42 in all other cache-arrays.
                * 'bounding_box': Bounding box of every SSV.
                * 'size': Number voxels of each SSV.
                * 'rep_coord': Representative coordinates for each SSV.
                * 'sv': Supervoxel IDs for every SSV.
                * 'sample_locations': Lists of rendering locations for each SSV. Each entry is a
                  list (length corresponds to the number of supervoxels) of coordinate arrays for
                  the corresponding SSV.
                * 'celltype_cnn_e3': Celltype classifications based on the elektronn3 CMN.
                * 'celltype_cnn_e3_probas': Celltype logits for the different types as an array of
                  shape (M, C; M: Number of predicted random multi-view sets, C: Number of
                  classes). In the example run there are currently 9 predicted classes:
                  STN=0, DA=1, MSN=2, LMAN=3, HVC=4, GP=5, FS=6, TAN=7, INT=8.
                * 'syn_ssv': Synapse IDs assigned to each SSV.
                * 'syn_sign_ratio': Area-weighted atio of symmetric synapses, see
                  :func:`~syconn.reps.super_segmentation_object.SuperSegmentationObject.syn_sign_ratio`.
                * 'sj': Synaptic junction object IDs which were mapped to each SSV. These are used
                  for view rendering and also to generate the 'syn_ssv' objects in combination
                  with contact sites (see corresponding section in the documentation).
                * 'mapping_sj_ids': Synaptic junction objects which overlap with the respective
                  SSVs.
                * 'mapping_sj_ratios': Overlap ratio of the synaptic junctions.
                * 'vc': Vesicle clouds mapped to each SSV.
                * 'mapping_vc_ids': Vesicle cloud objects which overlap with the respective SSVs.
                * 'mapping_vc_ratios': Overlap ratio of the vesicle clouds.
                * 'mi': Mitochondria mapped to each SSV.
                * 'mapping_mi_ids': Mitochondria objects which overlap with the respective SSVs.
                * 'mapping_mi_ratios': Overlap ratio of the mitochondria.

        skeleton: The skeleton representation of this super-supervoxel. Keys which are
            currently in use:
                * 'nodes': Array of the node coordinates (in nanometers).
                * 'edges': Edges between nodes.
                * 'diameters': Estimated cell diameter at every node.
                * various node properties, e.g. 'axoness' and 'axoness_avg10000'. Check the
                available keys ``sso.skeleton.keys()`` of an initialized :class:`~SuperSegmentationObject`
                object ``sso`` after loading the skeleton (``sso.load_skeleton()``).
        enable_locking_so: Locking flag for all
            :class:`syconn.reps.segmentation.SegmentationObject` assigned to this
            object (e.g. SV, mitochondria, vesicle clouds, ...)
        nb_cpus: Number of cpus for parallel jobs. will only be used in some
            processing steps.
        view_dict: A dictionary for caching 2D projection views. Those are stored as
            a numpy array of shape (M, N, CH, x, y). M: Length of :py:attr:`~sample_locations` and
            has the same ordering; N: Number of views per location; CH: Number of channels (1 for
            glia prediction containing only the cell shape and 4 for neuron analysis containing
            cell and cell organelle shapes. Stored at :py:attr:`~view_path` and accessed via the
            :class:`~syconn.backend.storage.CompressedStorage` interface.
        version_dict: A dictionary which contains the versions of other dataset types which share
            the same working directory. Defaults to the `Versions` entry in the `config.ini` file.

    Todo:
        * add examples
        * add most important attributes
    """
    def __init__(self, ssv_id: int, version: Optional[str] = None,
                 version_dict: Optional[Dict[str, str]] = None,
                 working_dir: Optional[str] = None, create: bool = True,
                 sv_ids: Optional[np.ndarray] = None,
                 scaling: Optional[np.ndarray] = None,
                 object_caching: bool = True, voxel_caching: bool = True,
                 mesh_caching: bool = True, view_caching: bool = False,
                 config: Optional[DynConfig] = None, nb_cpus: int = 1,
                 enable_locking: bool = True, enable_locking_so: bool = False,
                 ssd_type: str = "ssv"):
        """

        Args:
            ssv_id: unique SSV ID.
            version: Version string identifier. if 'tmp' is used, no data will
                be saved to disk.
            version_dict: Dictionary which contains the versions of other dataset types which share
                the same working directory. Defaults to the `Versions` entry in the `config.ini`file.
            working_dir (): Path to the working directory.
            create: If True, the folder to its storage location :py:attr:`~ssv_dir` will be created.
            sv_ids: List of agglomerated supervoxels which define the neuron reconstruction.
            scaling: Array defining the voxel size in nanometers (XYZ).
            object_caching: :class:`~syconn.reps.segmentation.SegmentationObject` retrieved by
                :func:`~syconn.reps.segmentation.SegmentationObject.get_seg_objects`
                will be cached in a dictionary.
            voxel_caching: Voxel array will be cached at
                :py:attr:`~syconn.reps.segmentation.SegmentationObject._voxels`.
            mesh_caching: Meshes (cell fragments, mitos, vesicles, ..) will be cached at
                :py:attr:`~syconn.reps.segmentation.SegmentationObject._meshes`.
            view_caching: Views can be cached at :py:attr:`~view_dict`.
            config: Retrieved from :py:attr:`~syconn.global_params.config`, otherwise must be
                initialized with a :class:`~syconn.handler.config.DynConfig`
            nb_cpus: Number of cpus for parallel jobs. will only be used in some
                processing steps.
            enable_locking: Enable posix locking for IO operations.
            enable_locking_so: Locking flag for all :class:`syconn.reps.segmen
                tation.SegmentationObject` assigned
            to this object (e.g. SV, mitochondria, vesicle clouds, ...)
            ssd_type: -
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
        self.attr_dict = {}

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
        self._sv_graph = None

        # init mesh dicts
        self._meshes = {"sv": None, "sj": None, "syn_ssv": None,
                        "vc": None, "mi": None, "conn": None}

        self._views = None
        self._dataset = None
        self._weighted_graph = None
        self._sample_locations = None
        self._rot_mat = None
        self._label_dict = {}
        self.view_dict = {}

        if sv_ids is not None:
            self.attr_dict["sv"] = sv_ids

        if working_dir is None:
            if global_params.wd is not None or version == 'tmp':
                self._working_dir = global_params.wd
            else:
                msg = "No working directory (wd) given. It has to be" \
                      " specified either in global_params, via kwarg " \
                      "`working_dir` or `config`."
                log_reps.error(msg)
                raise ValueError(msg)
        else:
            self._working_dir = working_dir
            self._config = DynConfig(working_dir)

        if global_params.wd is None:
            global_params.wd = self._working_dir

        if scaling is None:
            try:
                self._scaling = \
                    np.array(self.config.entries["Dataset"]["scaling"])
            except KeyError:
                msg = 'Scaling not set and could not be found in config ("{}"' \
                      ') with entries: {}'.format(self.config.path_config, self.config.entries)
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

    def __repr__(self):
        return 'SSO object (ID: {}, type: "{}", version: "{}", path: "{}"'.format(
            self.id, self.type, self.version, self.ssv_dir)

    #                                                       IMMEDIATE PARAMETERS
    @property
    def type(self) -> str:
        """
        The type of this super-sueprvoxel. Default: 'ssv'.

        Returns:
            String identifier of the object type.
        """
        return self._type

    @property
    def id(self) -> int:
        """
        Default value is the smalles SV ID which is part of this cell
        reconstruction.

        Returns:
              Globally unique identifier of this object.
          """
        return self._id

    @property
    def version(self) -> str:
        """
        Version of the `~SuperSegmentationDataset` this object
        belongs to. Can be any character or string like '0' or 'axongroundtruth'.

        Returns:
            String identifier of the object's version.
        """
        return str(self._version)

    @property
    def object_caching(self) -> bool:
        """If True, :class:`~syconn.reps.segmentation.SegmentationObject`s which
        are part of this cell reconstruction are cached."""
        return self._object_caching

    @property
    def voxel_caching(self) -> bool:
        """If True, voxel data is cached."""
        return self._voxel_caching

    @property
    def mesh_caching(self) -> bool:
        """If True, mesh data is cached."""
        return self._mesh_caching

    @property
    def view_caching(self) -> bool:
        """If True, view data is cached."""
        return self._view_caching

    @property
    def scaling(self) -> np.ndarray:
        """
        Voxel size in nanometers (XYZ). Default is taken from the `config.ini` file and
        accessible via :py:attr:`~config`.
        """
        return self._scaling

    #                                                                      PATHS
    @property
    def working_dir(self) -> str:
        """
        Working directory.
        """
        return self._working_dir

    @property
    def identifier(self) -> str:
        """
        Identifier used to create the folder name of the
        :class: `~syconn.reps.super_segmentation_dataset.SuperSegmentationDataset`
        this object belongs to.
        """
        return "%s_%s" % (self.type, self.version.lstrip("_"))

    @property
    def ssds_dir(self) -> str:
        """
        Path to the
        :class:`~syconn.reps.super_segmentation_dataset.SuperSegmentationDataset`
        directory this object belongs to.
        """
        return "%s/%s/" % (self.working_dir, self.identifier)

    @property
    def ssv_dir(self) -> str:
        """
        Path to the folder where the data of this super-supervoxel is stored.
        """
        return "%s/so_storage/%s/" % (self.ssds_dir,
                                      subfold_from_ix_SSO(self.id))

    @property
    def attr_dict_path(self) -> str:
        """
         Path to the attribute storage. :py:attr:`~attr_dict` can be loaded from here.
         """
        # Kept for backwards compatibility, remove if not needed anymore
        if os.path.isfile(self.ssv_dir + "atrr_dict.pkl"):
            return self.ssv_dir + "atrr_dict.pkl"
        return self.ssv_dir + "attr_dict.pkl"

    @property
    def skeleton_kzip_path(self) -> str:
        """
        Path to the skeleton storage.
        """
        return self.ssv_dir + "skeleton.k.zip"

    @property
    def skeleton_kzip_path_views(self) -> str:
        """
        Path to the skeleton storage.

        Todo:
            * probably deprecated.
        """
        return self.ssv_dir + "skeleton_views.k.zip"

    @property
    def objects_dense_kzip_path(self) -> str:
        """Identifier of cell organell overlays"""
        return self.ssv_dir + "objects_overlay.k.zip"

    @property
    def skeleton_path(self) -> str:
        """Identifier of SSV skeleton"""
        return self.ssv_dir + "skeleton.pkl"

    @property
    def edgelist_path(self) -> str:
        """Identifier of SSV graph"""
        return self.ssv_dir + "edge_list.bz2"

    @property
    def view_path(self) -> str:
        """Identifier of view storage"""
        return self.ssv_dir + "views.pkl"

    @property
    def mesh_dc_path(self) -> str:
        """Identifier of mesh storage"""
        return self.ssv_dir + "mesh_dc.pkl"

    @property
    def vlabel_dc_path(self) -> str:
        """Identifier of vertex label storage"""
        return self.ssv_dir + "vlabel_dc.pkl"

    #                                                                        IDS
    @property
    def sv_ids(self) -> np.ndarray:
        """
        All cell supervoxel IDs which are assigned to this cell reconstruction.
        """
        return self.lookup_in_attribute_dict("sv")

    @property
    def sj_ids(self) -> np.ndarray:
        """
        All synaptic junction (sj) supervoxel IDs which are assigned to this
        cell reconstruction.
        """
        return self.lookup_in_attribute_dict("sj")

    @property
    def mi_ids(self) -> np.ndarray:
        """
        All mitochondria (mi) supervoxel IDs which are assigned to this
        cell reconstruction.
        """
        return self.lookup_in_attribute_dict("mi")

    @property
    def vc_ids(self) -> np.ndarray:
        """
        All vesicle cloud (vc) supervoxel IDs which are assigned to this
        cell reconstruction.
        """
        return self.lookup_in_attribute_dict("vc")

    @property
    def dense_kzip_ids(self) -> Dict[str, int]:
        """
        ?
        """
        return dict([("mi", 1), ("vc", 2), ("sj", 3)])

    #                                                        SEGMENTATIONOBJECTS
    @property
    def svs(self) -> List[SegmentationObject]:
        """
        All cell :class:`~syconn.reps.segmentation.SegmentationObjects` objects
        which are assigned to this cell reconstruction.
        """
        return self.get_seg_objects("sv")

    @property
    def sjs(self) -> List[SegmentationObject]:
        """
        All synaptic junction (sj) :class:`~syconn.reps.segmentation.SegmentationObjects` objects
        which are assigned to this cell reconstruction. These objects are based on the
        initial synapse predictions and may contain synapse-synapse merger.
        See :py:attr:`~syn_ssv` for merger-free inter-neuron synapses.
        """
        return self.get_seg_objects("sj")

    @property
    def mis(self) -> List[SegmentationObject]:
        """
        All mitochondria (mi) :class:`~syconn.reps.segmentation.SegmentationObjects` objects
        which are assigned to this cell reconstruction.
        """
        return self.get_seg_objects("mi")

    @property
    def vcs(self) -> List[SegmentationObject]:
        """
        All vesicle cloud (vc) :class:`~syconn.reps.segmentation.SegmentationObjects`
        objects
        which are assigned to this cell reconstruction.
        """
        return self.get_seg_objects("vc")

    @property
    def syn_ssv(self) -> List[SegmentationObject]:
        """
        All inter-neuron synapse (syn_ssv) :class:`~syconn.reps.segmentation.SegmentationObject`
        objects which are assigned to this cell reconstruction. These objects are generated
        as a combination of synaptic junction (sj) and contact site (cs) objects.
        """
        return self.get_seg_objects("syn_ssv")

    #                                                                     MESHES
    def load_mesh(self, mesh_type) -> Optional[MeshType]:
        """
        Load mesh of a specific type, e.g. 'mi', 'sv', etc.

        Args:
            mesh_type: Type of :class:`~syconn.reps.segmentation.SegmentationObject` used for
                mesh retrieval.

        Returns:
            Three flat arrays: indices, vertices, normals
        """
        if not mesh_type in self._meshes:
            return None
        if self._meshes[mesh_type] is None:
            if not self.mesh_caching:
                return self._load_obj_mesh(mesh_type)
            self._meshes[mesh_type] = self._load_obj_mesh(mesh_type)
        return self._meshes[mesh_type]

    @property
    def mesh(self) -> Optional[MeshType]:
        """
        Mesh of all cell supervoxels.
        """
        return self.load_mesh("sv")

    @property
    def sj_mesh(self) -> Optional[MeshType]:
        """
        Mesh of all synaptic junction (sj) supervoxels. These objects are based
        on the original synapse prediction and might contain merger.
        """
        return self.load_mesh("sj")

    @property
    def vc_mesh(self) -> Optional[MeshType]:
        """
        Mesh of all vesicle clouds (vc) supervoxels.
        """
        return self.load_mesh("vc")

    @property
    def mi_mesh(self) -> Optional[MeshType]:
        """
        Mesh of all mitochondria (mi) supervoxels.
        """
        return self.load_mesh("mi")

    @property
    def syn_ssv_mesh(self) -> Optional[MeshType]:
        """
        Mesh of all inter-neuron synapses junction (syn_ssv) supervoxels. These
        objects are generated as a combination of contact sites and synaptic
        junctions (sj).
        """
        return self.load_mesh("syn_ssv")

    def label_dict(self, data_type='vertex') -> CompressedStorage:
        """
        Dictionary which stores various predictions. Currently used keys:

            * 'vertex': Labels associated with the mesh vertices. The ordering
              is the same as the vertex order in ``self.mesh[1]``.

        Uses the :class:`~syconn.backend.storage.CompressedStorage` interface.

        Args:
            data_type: Key for the stored labels.

        Returns:
            The stored array.
        """
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
        log_reps.warning('DEPRECATED USE OF `cell_type` attribute in SSV'
                         ' {}'.format(self.id))
        if self.cell_type_ratios is not None:
            return np.argmax(self.cell_type_ratios)
        else:
            return None

    @property
    def cell_type_ratios(self):
        log_reps.warning('DEPRECATED USE OF `cell_type_ratios` attribute in SSV'
                         ' {}'.format(self.id))
        return self.lookup_in_attribute_dict("cell_type_ratios")

    def weighted_graph(self, add_node_attr: Tuple[str] = ()) -> nx.Graph:
        """
        Creates a Euclidean distance weighted graph representation of the
        skeleton of this SSV object.

        Args:
            add_node_attr: To-be-added node attributes. Must exist in
            :py:attr`~skeleton`.

        Returns:
            The skeleton of this SSV object as a graph.
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
    def config(self) -> DynConfig:
        """
        The configuration object which contain all dataset-specific parameters.
        See :class:`~syconn.handler.config.DynConfig`.

        Returns:
            The configuration object.
        """
        if self._config is None:
            self._config = global_params.config
        return self._config

    @property
    def size(self) -> int:
        """
        Returns:
            The number of voxels associated with this SSV object.
        """
        if self._size is None:
            self._size = self.lookup_in_attribute_dict("size")

        if self._size is None:
            self.calculate_size()

        return self._size

    @property
    def bounding_box(self) -> List[np.ndarray]:
        if self._bounding_box is None:
            self._bounding_box = self.lookup_in_attribute_dict("bounding_box")

        if self._bounding_box is None:
            self.calculate_bounding_box()

        return self._bounding_box

    @property
    def shape(self) -> np.ndarray:
        """
        The XYZ extent of this SSV object in voxels.

        Returns:
            The shape/extent of thiss SSV object in voxels (XYZ).
        """
        return self.bounding_box[1] - self.bounding_box[0]

    @property
    def rep_coord(self) -> np.ndarray:
        """
        Representative coordinate of this SSV object. Will be the representative
        coordinate of the first supervoxel in :py:attr:`~svs`.

        Returns:
            1D array of the coordinate (XYZ).
        """
        if self._rep_coord is None:
            self._rep_coord = self.lookup_in_attribute_dict("rep_coord")

        if self._rep_coord is None:
            self._rep_coord = self.svs[0].rep_coord

        return self._rep_coord

    @property
    def attr_dict_exists(self) -> bool:
        """
        Checks if a attribute dictionary file exists at :py:attr:`~attr_dict_path`.

        Returns:
            True if the attribute dictionary file exists.
        """
        return os.path.isfile(self.attr_dict_path)

    def mesh_exists(self, obj_type: str) -> bool:
        """
        Checks if the mesh of :class:`~syconn.reps.segmentation.SegmentationObject`s
        of type `obj_type` exists in the :class:`~syconn.backend.storage.MeshStorage`
        located at :py:attr:`~mesh_dc_path`.

        Args:
            obj_type: Type of requested :class:`~syconn.reps.segmentation.SegmentationObject`s.

        Returns:
            True if the mesh exists.
        """
        mesh_dc = MeshStorage(self.mesh_dc_path,
                              disable_locking=not self.enable_locking)
        return obj_type in mesh_dc

    @property
    def voxels(self) -> Optional[np.ndarray]:
        """
        Voxels associated with this SSV object.

        Returns:
            3D binary array indicating voxel locations.
        """
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
    def voxels_xy_downsampled(self) -> Optional[np.ndarray]:
        if self._voxels_xy_downsampled is None:
            if self.voxel_caching:
                self._voxels_xy_downsampled = \
                    self.load_voxels_downsampled((2, 2, 1))
            else:
                return self.load_voxels_downsampled((2, 2, 1))

        return self._voxels_xy_downsampled

    @property
    def rag(self) -> nx.Graph:
        """
        The region adjacency graph (defining the supervoxel graph) of this SSV
        object.

        Returns:
            Supervoxel graph with nodes of type
            :class:`~syconn.reps.segmentation.SegmentationObject`.
        """
        if self._rag is None:
            self._rag = self.load_sv_graph()
        return self._rag

    @property
    def compartment_meshes(self) -> dict:
        """
        Compartment mesh storage.

        Returns:
            A dictionary which contains the meshes of each compartment.
        """
        if not "axon" in self._meshes:
            self._load_compartment_meshes()
        return {k: self._meshes[k] for k in ["axon", "dendrite", "soma"]}

    def _load_compartment_meshes(self, overwrite: bool = False):
        """
        Loading compartment meshes in-place as 'axon', 'dendrite', 'soma' to
        :py:attr:`~syconn.reps.super_segmentation_object.SuperSegmentationObject._meshes`.

        Args:
            overwrite: Overwrites existing compartment meshes
        """
        mesh_dc = MeshStorage(self.mesh_dc_path,
                              disable_locking=not self.enable_locking)
        if "axon" not in mesh_dc or overwrite:
            mesh_compartments = compartmentalize_mesh(self)
            mesh_dc["axon"] = mesh_compartments["axon"]
            mesh_dc["dendrite"] = mesh_compartments["dendrite"]
            mesh_dc["soma"] = mesh_compartments["soma"]
            mesh_dc.push()
        comp_meshes = {k: mesh_dc[k] for k in ["axon", "dendrite", "soma"]}
        self._meshes.update(comp_meshes)

    def load_voxels_downsampled(self, downsampling: tuple = (2, 2, 1),
                                nb_threads: int = 10) -> np.ndarray:
        """
        Load all voxels of this SSV object.

        Args:
            downsampling: The downsampling of the returned voxels.
            nb_threads: Number of threads.

        Returns:
            List of downsampled voxel coordinates.
        """
        return ssh.load_voxels_downsampled(self, downsampling=downsampling,
                                           nb_threads=nb_threads)

    def get_seg_objects(self, obj_type: str) -> List[SegmentationObject]:
        """
        Factory method for :class:`~syconn.reps.segmentation.SegmentationObject`s of
        type `obj_type`.

        Args:
            obj_type: Type of requested :class:`~syconn.reps.segmentation.SegmentationObject`s.

        Returns:
            The :class:`~syconn.reps.segmentation.SegmentationObject`s of type `obj_type`
            sharing the same working directory as this SSV object.
        """
        if obj_type not in self._objects:
            objs = []

            for obj_id in self.lookup_in_attribute_dict(obj_type):
                objs.append(self.get_seg_obj(obj_type, obj_id))

            if self.object_caching:
                self._objects[obj_type] = objs
            else:
                return objs

        return self._objects[obj_type]

    def get_seg_obj(self, obj_type: str, obj_id: int) -> SegmentationObject:
        """
        Factory method for :class:`~syconn.reps.segmentation.SegmentationObject` of
        type `obj_type`.

        Args:
            obj_type: Type of requested :class:`~syconn.reps.segmentation.SegmentationObject`.
            obj_id: ID of the requested object.

        Returns:
            The :class:`~syconn.reps.segmentation.SegmentationObject` of type `obj_type`
            sharing the same working directory as this SSV object.
        """
        return SegmentationObject(obj_id=obj_id, obj_type=obj_type,
                                  version=self.version_dict[obj_type],
                                  working_dir=self.working_dir, create=False,
                                  scaling=self.scaling,
                                  enable_locking=self.enable_locking_so)

    def get_seg_dataset(self, obj_type: str) -> SegmentationDataset:
        """
        Factory method for :class:`~syconn.reps.segmentation.SegmentationDataset` of
        type `obj_type`.

        Args:
            obj_type: Type of requested :class:`~syconn.reps.segmentation.SegmentationDataset`.

        Returns:
            The :class:`~syconn.reps.segmentation.SegmentationDataset` of type `obj_type`
            sharing the same working directory as this SSV object.
        """
        return SegmentationDataset(obj_type, version_dict=self.version_dict,
                                   version=self.version_dict[obj_type],
                                   scaling=self.scaling,
                                   working_dir=self.working_dir)

    def load_attr_dict(self) -> int:
        """
        Load the attribute dictionary of this SSV object stored at
        :py:attr:`~ssv_dir`.
        """
        try:
            self.attr_dict = load_pkl2obj(self.attr_dict_path)
            return 0
        except (IOError, EOFError):
            return -1

    def load_sv_graph(self) -> nx.Graph:
        """
        Load the supervoxel graph (node objects will be of type
        :class:`~syconn.reps.segmentation.SegmentationObject`) of this SSV object.
         It is generated from the supervoxel ID graph stored in :py:attr:`_sv_graph`
         or the edge list stored at :py:attr:`edgelist_path`.

        Returns:
            The supervoxel graph with :class:`~syconn.reps.segmentation.SegmentationObject`
            nodes.
        """
        if self._sv_graph is not None:
            G = self._sv_graph
        elif os.path.isfile(self.edgelist_path):
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

    def load_edgelist(self) -> List[Tuple[int, int]]:
        """
        Load the edges within the supervoxel graph.

        Returns:
            Edge list representing the supervoxel graph.
        """
        g = self.load_sv_graph()
        return list(g.edges())

    def _load_obj_mesh(self, obj_type: str = "sv",
                       rewrite: bool = False) -> MeshType:
        """
        Load the mesh of a given `obj_type`. If :func:`~mesh_exists` is False,
        loads the meshes from the underlying sueprvoxel objects.
        TODO: Currently does not support color array!
        TODO: add support for sym. asym synapse type

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

    def _load_obj_mesh_compr(self, obj_type: str = "sv") -> MeshType:
        """
        Load meshes of all objects of type `obj_type` assigned to this SSV.

        Args:
            obj_type: Type of requested objects.

        Returns:
            A single mesh of all objects.
        """
        mesh_dc = MeshStorage(self.mesh_dc_path,
                              disable_locking=not self.enable_locking)
        return mesh_dc._dc_intern[obj_type]

    def save_attr_dict(self):
        """
        Save the SSV's attribute dictionary.
        """
        # TODO: use AttributeDict class
        if self.version == 'tmp':
            log_reps.warning('"save_attr_dict" called but this SSV '
                             'has version "tmp", attribute dict will'
                             ' not be saved to disk.')
            return
        try:
            orig_dc = load_pkl2obj(self.attr_dict_path)
        except (IOError, EOFError, FileNotFoundError) as e:
            if not '[Errno 2] No such file or' in str(e):
                log_reps.critical("Could not load SSO attributes from {} due to "
                                  "{}.".format(self.attr_dict_path, e))
            orig_dc = {}
        orig_dc.update(self.attr_dict)
        write_obj2pkl(self.attr_dict_path + '.tmp', orig_dc)
        shutil.move(self.attr_dict_path + '.tmp', self.attr_dict_path)

    def save_attributes(self, attr_keys : List[str], attr_values: List[Any]):
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
        except (IOError, EOFError, FileNotFoundError) as e:
            if not "[Errno 13] Permission denied" in str(e):
                pass
            else:
                log_reps.critical("Could not load SSO attributes at {} due to "
                                  "{}.".format(self.attr_dict_path, e))
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

    def attr_exists(self, attr_key: str) -> bool:
        """
        Checks if an attribute exists for this SSV object.

        Args:
            attr_key: Attribute key.

        Returns:
            True if the key exists in :py:attr:`~attr_dict`.
        """
        return attr_key in self.attr_dict

    def lookup_in_attribute_dict(self, attr_key: str) -> Optional[Any]:
        """
        Returns the value to `attr_key` stored in :py:attr:`~attr_dict` or
        None if the key is not existent.

        Args:
            attr_key: Attribute key.

        Returns:
            Value to the key ``attr_key``.
        """
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

    def load_so_attributes(self, obj_type: str, attr_keys: List[str],
                           nb_cpus: Optional[int] = None):
        """
        Collect attributes from :class:`~syconn.reps.segmentation.SegmentationObject`
        of type `obj_type`.
        The attribute value ordering for each key is the same as :py:attr:`~svs`.

        Args:
            obj_type: Type of :class:`~syconn.reps.segmentation.SegmentationObject`.
            attr_keys: Keys of desired properties. Must exist for the requested
             `obj_type`.
            nb_cpus: Number of CPUs to use for the calculation.

        Returns:
            Attribute values for each key in `attr_keys`.
        """
        if nb_cpus is None:
            nb_cpus = self.nb_cpus
        params = [[obj, dict(attr_keys=attr_keys)]
                  for obj in self.get_seg_objects(obj_type)]
        attr_values = sm.start_multiprocess_obj('load_attributes', params,
                                                nb_cpus=nb_cpus)
        attr_values = [el for sublist in attr_values for el in sublist]
        return [attr_values[ii::len(attr_keys)] for ii in range(len(attr_keys))]

    def calculate_size(self, nb_cpus: Optional[int] = None):
        """
        Calculates :py:attr:`size`.

        Args:
            nb_cpus: Number of CPUs to use for the calculation.
        """
        self._size = np.sum(self.load_so_attributes('sv', ['size'],
                                                    nb_cpus=nb_cpus))

    def calculate_bounding_box(self, nb_cpus: Optional[int] = None):
        """
        Calculates :py:attr:`~bounding_box` (and :py:attr:`size`).

        Args:
            nb_cpus: Number of CPUs to use for the calculation.
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

    def calculate_skeleton(self, force: bool = False):
        """
        Merges existing supervoxel skeletons (``allow_skel_gen=False``) or calculates them
        from scratch using :func:`~syconn.reps.super_segmentation_helper
        .create_sso_skeletons_thread` otherwise (requires ``allow_skel_gen=True``).
        Skeleton will be saved at :py:attr:`~skeleton_path`.

        Args:
            force: Skips :func:`~load_skeleton` if ``force=True``.
        """
        if force:  #
            return ssh.create_sso_skeletons_thread([self])
        self.load_skeleton()
        if self.skeleton is not None and len(self.skeleton["nodes"]) != 0 \
                and not force:
            return
        ssh.create_sso_skeletons_thread([self])

    def save_skeleton_to_kzip(self, dest_path: Optional[str] = None,
                              additional_keys: Optional[List[str]] = None):
        """

        Args:
            dest_path: Destination path for k.zip file.
            additional_keys: Additional skeleton keys which are converted into
            KNOSSOS skeleton node properties. Will always attempt to write out the
            keys 'axoness', 'cell_type' and 'meta'.

        Returns:
            Saves KNOSSOS compatible k.zip file containing the SSV skeleton and
            its node properties.
        """
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
                pred_key_ax = "{}_avg{}".format(global_params.view_properties_semsegax['semseg_key'],
                                                global_params.DIST_AXONESS_AVERAGING)
                if pred_key_ax in self.skeleton:
                    skel_nodes[-1].data[pred_key_ax] = self.skeleton[pred_key_ax][
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

    def save_objects_to_kzip_sparse(self, obj_types: Optional[Iterable[str]] = None,
                                    dest_path: Optional[str] = None):
        """
        Export cellular organelles as coordinates with size, shape and overlap
        properties in a KNOSSOS compatible format.

        Args:
            obj_types: Type identifiers of the supervoxel objects which are exported.
            dest_path: Path to the destination file. If None, results will be
                stored at :py:attr:`~skeleton_kzip_path`:

        """
        if obj_types is None:
            obj_types = global_params.existing_cell_organelles
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

    def save_objects_to_kzip_dense(self, obj_types: List[str],
                                   dest_path: Optional[str] = None):
        """
        Export cellular organelles as coordinates with size, shape and overlap
        properties in a KNOSSOS compatible format.

        Args:
            obj_types: Type identifiers of the supervoxel objects which are
                exported.
            dest_path: Path to the destination file. If None, result will be
                stored at :py:attr:`~objects_dense_kzip_path`:


        """
        if dest_path is None:
            dest_path = self.objects_dense_kzip_path
        if os.path.exists(self.objects_dense_kzip_path[:-6]):
            shutil.rmtree(self.objects_dense_kzip_path[:-6])
        if os.path.exists(self.objects_dense_kzip_path):
            os.remove(self.objects_dense_kzip_path)

        for obj_type in obj_types:
            so_objs = self.get_seg_objects(obj_type)
            for so_obj in so_objs:
                so_obj.save_kzip(path=dest_path,
                                 write_id=self.dense_kzip_ids[obj_type])

    def total_edge_length(self) -> float:
        """
        Total edge length of the super-supervoxel :py:attr:`~skeleton` in nanometers.

        Returns:
            Sum of all edge lengths (L2 norm) in :py:attr:`~skeleton`.
        """
        if self.skeleton is None:
            self.load_skeleton()
        nodes = self.skeleton["nodes"]
        edges = self.skeleton["edges"]
        return np.sum([np.linalg.norm(
            self.scaling * (nodes[e[0]] - nodes[e[1]])) for e in edges])

    def save_skeleton(self, to_kzip=False, to_object=True):
        """
        Saves skeleton to default locations as `.pkl` and optionally as `.k.zip`.

        Args:
            to_kzip: Stores skeleton as a KNOSSOS compatible xml inside a k.zip file.
            to_object: Stores skeleton as a dictionary in a pickle file.
        """
        if self.version == 'tmp':
            log_reps.warning('"save_skeleton" called but this SSV '
                             'has version "tmp", skeleton will'
                             ' not be saved to disk.')
            return
        if to_object:
            write_obj2pkl(self.skeleton_path, self.skeleton)

        if to_kzip:
            self.save_skeleton_to_kzip()

    def load_skeleton(self) -> bool:
        """
        Loads skeleton and will compute it if it does not exist yet (requires
        ``allow_skel_gen=True``).

        Returns:
            True if successfully loaded/generated skeleton, else False.
        """
        try:
            self.skeleton = load_pkl2obj(self.skeleton_path)
            # stored as uint32, if used for computations
            # e.g. edge length then it will overflow
            self.skeleton["nodes"] = self.skeleton["nodes"].astype(np.float32)
            return True
        except:
            if global_params.config.allow_skel_gen:
                self.calculate_skeleton()
                return True
            return False

    def syn_sign_ratio(self, weighted: bool = True,
                       recompute: bool = False) -> float:
        """
        Ratio of symmetric synapses (between 0 and 1; -1 if no synapse objects).

        Args:
            weighted: Compute synapse-area weighted ratio.
            recompute: Ignore existing value.

        Returns:
            Type ratio of all synapses (``syn_ssv``) assigned to this object.
        """
        ratio = self.lookup_in_attribute_dict("syn_sign_ratio")
        if not recompute and ratio is not None:
            return ratio
        syn_signs = []
        syn_sizes = []
        for syn in self.syn_ssv:
            syn.load_attr_dict()
            syn_signs.append(syn.attr_dict["syn_sign"])
            syn_sizes.append(syn.mesh_area / 2)
        if len(syn_signs) == 0 or np.sum(syn_sizes) == 0:
            return -1
        syn_signs = np.array(syn_signs)
        syn_sizes = np.array(syn_sizes)
        if weighted:
            ratio = np.sum(syn_sizes[syn_signs == -1]) / float(np.sum(syn_sizes))
        else:
            ratio = np.sum(syn_signs == -1) / float(len(syn_signs))
        self.attr_dict["syn_sign_ratio"] = ratio
        self.save_attributes(["syn_sign_ratio"], [ratio])
        return ratio

    def aggregate_segmentation_object_mappings(self, obj_types: List[str],
                                               save: bool = False):
        """
        Aggregates mapping information of cellular organelles from the SSV's
        supervoxels. After this step, :func:`~apply_mapping_decision` can be
        called to apply final assignments.

        Examples:
            A mitochondrion can extend over multiple supervoxels, so it will
            overlap with all of them partially. Here, the overlap information
            of all supervoxels assigned to this SSV will be aggregated.
        Args:
            obj_types: Cell organelles types to process.
            save: Save :yp:attr:`~attribute_dict` at the end.
        """
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

    def apply_mapping_decision(self, obj_type: str,
                               correct_for_background: bool = True,
                               lower_ratio: Optional[float] = None,
                               upper_ratio: Optional[float] = None,
                               sizethreshold: Optional[float] = None,
                               save: bool = True):
        """
        Applies mapping decision of cellular organelles to this SSV object. A
        :class:`~syconn.reps.segmentation.SegmentationObject` in question is
        assigned to this :class:`~syconn.reps.super_segmentation_object.SuperSegmentationObject`
        if they share the highest overlap. For more details see ``SyConn/docs/object_mapping.md``.
        Default parameters for the mapping will be taken from the `config.ini` file.

        Args:
            obj_type: Type of :class:`~syconn.reps.segmentation.SegmentationObject`
                which are to be mapped.
            correct_for_background: Ignore background ID during mapping
            lower_ratio: Minimum overlap s.t. objects are mapped.
            upper_ratio: Maximum ratio s.t. objects are mapped.
            sizethreshold: Minimum voxel size of an object, objects below will be
                ignored.
            save: If True, :py:attr:`~attr_dict` will be saved.

        Todo:
            * check what ``correct_for_background`` was for. Any usecase for
            ``correct_for_background=False``?

        Returns:

        """
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

    def _map_cellobjects(self, obj_types: Optional[List[str]] = None,
                         save: bool = True):
        """
        Wrapper function for mapping all existing cell organelles (as defined in
        :py:attr:`~syconn.global_params.existing_cell_organelles`).

        Args:
            obj_types: Type of :class:`~syconn.reps.super_segmentation_object
            .SuperSegmentationObject` which should be mapped.
            save: Saves the attribute dict of this SSV object afterwards.
        """
        if obj_types is None:
            obj_types = global_params.existing_cell_organelles
        self.aggregate_segmentation_object_mappings(obj_types, save=save)
        for obj_type in obj_types:
            self.apply_mapping_decision(obj_type, save=save,
                                        correct_for_background=obj_type == "sj")

    def clear_cache(self):
        """
        Clears the following, cached data:
            * :py:attr:`~voxels`
            * :py:attr:`~voxels_xy_downsampled`
            * :py:attr:`~sample_locations`
            * :py:attr:`~_objects`
            * :py:attr:`~_views`
            * :py:attr:`~skeleton`
            * :py:attr:`~_meshes`
        """
        self._objects = {}
        self._voxels = None
        self._voxels_xy_downsampled = None
        self._views = None
        self._sample_locations = None
        self._meshes = None
        self.skeleton = None

    def preprocess(self):
        """
        Process object mapping (requires the prior assignment of object
        candidates), cache object meshes and calculate the SSV skeleton.

        Todo:
            * Check what the ``clear_cache()`` call was for.
        """
        self.load_attr_dict()
        self._map_cellobjects()
        for sv_type in global_params.existing_cell_organelles + ["sv", "syn_ssv"]:
            _ = self._load_obj_mesh(obj_type=sv_type, rewrite=False)
        self.calculate_skeleton()
        self.clear_cache()

    def copy2dir(self, dest_dir: str, safe: bool = True):
        """
        Copies the content at :py:attr:`~ssv_dir` to another directory.

        Examples:
            To copy the data of this SSV object (``ssv_orig``) to another yet not
            existing SSV (``ssv_target``). call ``ssv_orig.copy2dir(ssv_target.ssv_dir)``.
            All files contained in the directory py:attr:`~ssv_dir` of ``ssv_orig``
            will be copied to ``ssv_target.ssv_dir``.

        Args:
            dest_dir: Destination directory where all files contained in
                py:attr:`~ssv_dir` will be copied to.
            safe: If True, will not overwrite existing data.
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
                log_reps.debug("Copied %s to %s." % (src_filename, dest_filename))
            except Exception as e:
                log_reps.error("Skipped '{}', due to the following error: '{}'"
                               "".format(fnames[i], str(e)))
                pass
        self.load_attr_dict()
        if os.path.isfile(dest_dir + "/attr_dict.pkl"):
            dest_attr_dc = load_pkl2obj(dest_dir + "/attr_dict.pkl")
        else:
            dest_attr_dc = {}
        dest_attr_dc.update(self.attr_dict)
        write_obj2pkl(dest_dir + "/attr_dict.pkl", dest_attr_dc)

    def partition_cc(self, max_nb_sv: Optional[int] = None,
                     lo_first_n: Optional[int] = None) -> List[List[Any]]:
        """
        Splits the supervoxel graph of this SSV into subgraphs. Default values
        are generated from :py:attr:`syconn.global_params`.

        Args:
            max_nb_sv: Number of supervoxels per subgraph
                (defines the subgraph context).
            lo_first_n: Do not use first n traversed nodes for new bfs traversals.
                This allows to partition the original supervoxel graph of size `N`
                into ``N//lo_first_n`` subgraphs.

        Returns:

        """
        if lo_first_n is None:
            lo_first_n = global_params.SUBCC_CHUNK_SIZE_BIG_SSV
        if max_nb_sv is None:
            max_nb_sv = global_params.SUBCC_SIZE_BIG_SSV + 2 * (lo_first_n - 1)
        init_g = self.rag
        partitions = split_subcc_join(init_g, max_nb_sv, lo_first_n=lo_first_n)
        return partitions

    # -------------------------------------------------------------------- VIEWS
    def save_views(self, views: np.ndarray, view_key: str = "views"):
        """
        This will only save views on SSV level and not for each individual SV!

        Args:
            views: The view array.
            view_key: The key used for the look-up.

        Returns:

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

    def load_views(self, view_key: Optional[str] = None, woglia: bool = True,
                   raw_only: bool = False, force_reload: bool = False,
                   nb_cpus: Optional[int] = None, ignore_missing: bool = False,
                   index_views: bool = False) -> np.ndarray:
        """
        Load views which are stored in :py:attr:`~view_dict` or if not present attempts
        to retrieve data from :py:attr:`view_path` given the key `view_key`,
        i.e. this operates on SSV level. If the given key does not exist on
        :class:`~SuperSegmentationObject` level or is None, attempts to load the views from
        the underlying :class:`~syconn.reps.segmentation.SegmentationObject`s.

        Args:
            view_key: The key used for the look-up.
            woglia: If True, will load the views render from the glia-free agglomeration.
            raw_only: If True, will only return the cell shape channel in the views.
            force_reload: If True will force reloading the SV views.
            nb_cpus: Number of CPUs.
            ignore_missing: If True, it will not raise KeyError if SV does not exist.
            index_views: Views which contain the indices of the vertices at the respective pixels.
                Used as look-up to map the predicted semantic labels onto the mesh vertices.

        Returns:
            Concatenated views for each SV in self.svs with shape [N_LOCS, N_CH, N_VIEWS, X, Y].
        """
        if self.view_caching and view_key in self.view_dict:
            # self.view_dict stores list of views with length of sample_locations
            return self.view_dict[view_key]
        view_dc = CompressedStorage(self.view_path, read_only=True,
                                    disable_locking=not self.enable_locking)
        if view_key in view_dc and not force_reload:
            if self.view_caching:
                self.view_dict[view_key] = view_dc[view_key]
                return self.view_dict[view_key]
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
        # stores list of views with length of sample_locations
        if self.view_caching and view_key is not None:
            self.view_dict[view_key] = views
        return views

    def view_existence(self, woglia: bool = True, index_views: bool = False,
                       view_key: Optional[str] = None) -> List[bool]:
        """
        Checks whether a specific set of views exists for this object.

        Args:
            woglia: If True, will load the views render from the glia-free agglomeration.
            index_views: Views which contain the indices of the vertices at the respective pixels.
                Used as look-up to map the predicted semantic labels onto the mesh vertices.
            view_key: The key used for the look-up.

        Returns:
            True if the specified views exist.
        """
        view_paths = set([sv.view_path(woglia=woglia, index_views=index_views,
                                       view_key=view_key) for sv in self.svs])
        cached_ids = []
        for vp in view_paths:
            cached_ids += list(CompressedStorage(vp, disable_locking=True).keys())
        cached_ids = set(cached_ids).intersection(self.sv_ids)
        so_views_exist = [svid in cached_ids for svid in self.sv_ids]
        return so_views_exist

    def render_views(self, add_cellobjects: bool = False, verbose: bool = False,
                     overwrite: bool = True, cellobjects_only: bool =False,
                     woglia: bool = True, skip_indexviews: bool = False,
                     qsub_co_jobs: int = 300, resume_job: bool = False):
        """
        Renders views for each SV based on SSV context and stores them
        on SV level. Usually only used once: for initial glia or axoness
        prediction.
        The results will be saved distributed at each
        class:`~syconn.reps.segmentation.SegmentationObject` of this object.
        It is not cached in :py:attr:`view_dict` nor :py:attr:`view_path`.
        Used during initial glia, compartment and cell type predictions.
        See :func:`~_render_rawviews` for how to store views in the SSV storage, which
        is e.g. used during GT generation.

        Args:
            add_cellobjects: Add cellular organelle channels in the 2D projection views.
            verbose: Log additional information.
            overwrite: Re-render at all rendering locations.
            cellobjects_only: Render only cellular organelle channels. Currently not in use.
            woglia: If True, will load the views render from the glia-free agglomeration.
            skip_indexviews: Index views will not be generated, used for initial SSV
                glia-removal rendering.
            qsub_co_jobs: Number of parallel jobs if batchjob is used.
            resume_job: Batchjob parameter. Resumes a interrupted batchjob.
        """
        # TODO: partial rendering currently does not support index view generation (-> vertex
        #  indices will be different for each partial mesh)
        if len(self.sv_ids) > global_params.RENDERING_MAX_NB_SV and not woglia:
            if not skip_indexviews:
                raise ValueError('Index view rendering is currently not supported with partial '
                                 'cell rendering.')
            part = self.partition_cc()
            log_reps.info('Partitioned huge SSV into {} subgraphs with each {}'
                          ' SVs.'.format(len(part), len(part[0])))
            log_reps.info("Rendering SSO. {} SVs left to process"
                          ".".format(len(self.svs)))
            params = [[so.id for so in el] for el in part]

            params = chunkify(params, global_params.NGPU_TOTAL * 2)
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
                n_cores=global_params.NCORES_PER_NODE // global_params.NGPUS_PER_NODE,
                n_max_co_processes=qsub_co_jobs, remove_jobfolder=True, allow_resubm_all_fail=True,
                resume_job=resume_job, additional_flags="--gres=gpu:1")
        else:
            # render raw data
            rot_mat = render_sampled_sso(
                self, add_cellobjects=add_cellobjects, verbose=verbose, overwrite=overwrite,
                return_rot_mat=True, cellobjects_only=cellobjects_only, woglia=woglia)
            if skip_indexviews:
                return
            # render index views
            render_sampled_sso(self, verbose=verbose, overwrite=overwrite,
                               index_views=True, rot_mat=rot_mat)

    def render_indexviews(self, nb_views=2, save=True, force_recompute=False,
                          verbose=False, view_key=None, ws=None, comp_window=None):
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
        view_key : Optional[str]
            key used for storing view array. Default: 'index{}'.format(nb_views)
        ws : Tuple[int]
            Window size in pixels [y, x]
        comp_window : float
            Physical extent in nm of the view-window along y (see `ws` to infer pixel size)

        Returns
        -------
        np.array
        """
        if view_key is None:
            view_key = 'index{}'.format(nb_views)
        if not force_recompute:
            try:
                views = self.load_views(view_key)
                if not save:
                    return views
                else:
                    return
            except KeyError:
                pass
        locs = np.concatenate(self.sample_locations(cache=False))
        if self._rot_mat is None:
            index_views, rot_mat = render_sso_coords_index_views(
                self, locs, nb_views=nb_views, verbose=verbose,
                return_rot_matrices=True, ws=ws, comp_window=comp_window)
            self._rot_mat = rot_mat
        else:
            index_views = render_sso_coords_index_views(self, locs, nb_views=nb_views,
                                                        verbose=verbose,
                                                        rot_mat=self._rot_mat, ws=ws,
                                                        comp_window=comp_window)
        if self.view_caching:
            self.view_dict[view_key] = index_views
        if not save:
            return index_views
        self.save_views(index_views, view_key)

    def _render_rawviews(self, nb_views=2, save=True, force_recompute=False,
                         add_cellobjects=True, verbose=False, view_key=None,
                         ws=None, comp_window=None):
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
        view_key : Optional[str]
            key used for storing view array. Default: 'raw{}'.format(nb_views)
        ws : Tuple[int]
            Window size in pixels [y, x]
        comp_window : float
            Physical extent in nm of the view-window along y (see `ws` to infer pixel size)

        Returns
        -------
        np.array
        """
        if view_key is None:
            view_key = 'raw{}'.format(nb_views)
        if not force_recompute:
            try:
                views = self.load_views(view_key)
                if not save:
                    return views
                return
            except KeyError:
                pass
        locs = np.concatenate(self.sample_locations(cache=False))
        if self._rot_mat is None:
            views, rot_mat = render_sso_coords(self, locs, verbose=verbose, ws=ws,
                                               add_cellobjects=add_cellobjects, comp_window=comp_window,
                                               nb_views=nb_views, return_rot_mat=True)
            self._rot_mat = rot_mat
        else:
            views = render_sso_coords(self, locs, verbose=verbose, ws=ws,
                                      add_cellobjects=add_cellobjects, comp_window=comp_window,
                                      nb_views=nb_views, rot_mat=self._rot_mat)
        if self.view_caching:
            self.view_dict[view_key] = views
        if save:
            self.save_views(views, view_key)
        else:
            return views

    def predict_semseg(self, m, semseg_key, nb_views=None, verbose=False,
                       raw_view_key=None, save=False, ws=None, comp_window=None):
        """
        Generates label views based on input model and stores it under the key
        'semseg_key', either within the SSV's SVs or in an extra view-storage
        according to input parameters:
        Default situation (nb_views and raw_view_key is None):
            semseg_key = 'spiness', nb_views=None
            This will load the raw views stored at the SSV's SVs.
        Non-default (nb_views or raw_view_key is not None):
            semseg_key = 'spiness4', nb_views=4
            This requires to run 'self._render_rawviews(nb_views=4)'
            This method then has to be called like:
                'self.predict_semseg(m, 'spiness4', nb_views=4)'

        Parameters
        ----------
        semseg_key : str
        nb_views : Optional[int]
        k : int
        verbose : bool
        raw_view_key : str
            key used for storing view array within SSO directory. Default: 'raw{}'.format(nb_views)
            If key does not exist, views will be re-rendered with properties defined
            in global_params.py or as given in the kwargs `ws`, `nb_views` and `comp_window`.
        save : bool
            If True, views will be saved.
        ws : Tuple[int]
            Window size in pixels [y, x]
        comp_window : float
            Physical extent in nm of the view-window along y (see `ws` to infer pixel size)
        """
        if (nb_views is not None) or (raw_view_key is not None):
            # treat as special view rendering
            if nb_views is None:
                nb_views = global_params.NB_VIEWS
            if raw_view_key is None:
                raw_view_key = 'raw{}'.format(nb_views)
            if raw_view_key in self.view_dict:
                views = self.load_views(raw_view_key)
            else:
                # log_reps.warning('Could not find raw-views. Re-rendering now.')
                self._render_rawviews(nb_views, ws=ws, comp_window=comp_window, save=save,
                                      view_key=raw_view_key, verbose=verbose,
                                      force_recompute=True)
                views = self.load_views(raw_view_key)
            if len(views) != len(np.concatenate(self.sample_locations(cache=False))):
                raise ValueError("Unequal number of views and redering locations.")
            labeled_views = ssh.predict_views_semseg(views, m, verbose=verbose)
            assert labeled_views.shape[2] == nb_views, \
                "Predictions have wrong shape."
            if self.view_caching:
                self.view_dict[semseg_key] = labeled_views
            if save:
                self.save_views(labeled_views, semseg_key)
        else:
            # treat as default view rendering
            views = self.load_views()
            locs = self.sample_locations(cache=False)
            assert len(views) == len(np.concatenate(locs)), \
                "Unequal number of views and rendering locations."
            # re-order number of views according to SV rendering locations
            # TODO: move view reordering to 'pred_svs_semseg', check other usages before!
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

    def semseg2mesh(self, semseg_key: str, dest_path: Optional[str] = None,
                    nb_views: Optional[int] = None, k: int = 1,
                    force_recompute: bool = False,
                    index_view_key: Optional[str] = None):
        """
        Generates vertex labels and stores it in the SSV's label storage under
        the key `semseg_key`.

        Examples:
            Default situation:
                ``semseg_key = 'spiness'``, ``nb_views=None``
                This will load the index and label views stored at the SSV's SVs.

            Non-default:
                ``semseg_key = 'spiness4'``, ``nb_views=4``
                This requires to run ``self._render_rawviews(nb_views=4)``,
                ``self.render_indexviews(nb_views=4)`` and ``predict_semseg(MODEL,
                'spiness4', nb_views=4)``.
                This method then has to be called like: ``self.semseg2mesh('spiness4', nb_views=4)``

        Args:
            semseg_key: Key used to retrieve the semantic segmentation results.
            dest_path: Path where the mesh will be stored as .ply in a k.zip.
            nb_views: Number of views used
            k: Number of nearest vertices to average over. If k=0 unpredicted vertices
                will be treated as 'unpredicted' class.
            force_recompute: Force recompute.
            index_view_key: Key usedc to retrieve the index views.

        Returns:

        """
        # colors are only needed if dest_path is given
        # (last two colors correspond to background and undpredicted vertices (k=0))
        cols = None
        if dest_path is not None:
            if 'spiness' in semseg_key:
                cols = np.array([[0.6, 0.6, 0.6, 1], [0.9, 0.2, 0.2, 1],
                                 [0.1, 0.1, 0.1, 1], [0.05, 0.6, 0.6, 1],
                                 [0.9, 0.9, 0.9, 1], [0.1, 0.1, 0.9, 1]])
                cols = (cols * 255).astype(np.uint8)
            elif 'axon' in semseg_key:
                # cols = np.array([[0.6, 0.6, 0.6, 1], [0.9, 0.2, 0.2, 1],
                #                  [0.1, 0.1, 0.1, 1], [0.9, 0.9, 0.9, 1],
                #                  [0.1, 0.1, 0.9, 1]])
                # dendrite, axon, soma, bouton, terminal, background, unpredicted
                cols = np.array([[0.6, 0.6, 0.6, 1], [0.9, 0.2, 0.2, 1],
                                 [0.1, 0.1, 0.1, 1], [0.05, 0.6, 0.6, 1],
                                 [0.8, 0.8, 0.1, 1], [0.9, 0.9, 0.9, 1],
                                 [0.1, 0.1, 0.9, 1]])
                cols = (cols * 255).astype(np.uint8)
            else:
                raise ValueError('Semantic segmentation of "{}" is not supported.'
                                 ''.format(semseg_key))
        return ssh.semseg2mesh(self, semseg_key, nb_views, dest_path, k,
                               cols, force_recompute=force_recompute,
                               index_view_key=index_view_key)

    def semseg_for_coords(self, coords, semseg_key, k=5, ds_vertices=20,
                          ignore_labels=None):
        """
        Get the semantic segmentation with key `semseg_key` from the `k` nearest
        vertices at every coordinate in `coords`.

        Parameters
        ----------
        coords : np.array
            Voxel coordinates, unscaled! [N, 3]
        semseg_key : str
        k : int
            Number of nearest neighbors (NN) during k-NN classification
        ds_vertices : int
            striding factor for vertices
        ignore_labels : List[int]
            Vertices with labels in `ignore_labels` will be ignored during
             majority vote, e.g. used to exclude unpredicted vertices.

        Returns
        -------
        np.array
            Same length as coords. For every coordinate in coords returns the
            majority label within radius_nm
        """
        # TODO: Allow multiple keys as in self.attr_for_coords, e.g. to
        #  include semseg axoness in a single query
        if ignore_labels is None:
            ignore_labels = []
        coords = np.array(coords) * self.scaling
        vertices = self.mesh[1].reshape((-1, 3))
        if len(vertices) < 5e6:
            ds_vertices = max(1, ds_vertices // 10)
        vertex_labels = self.label_dict('vertex')[semseg_key][::ds_vertices]
        vertices = vertices[::ds_vertices]
        for ign_l in ignore_labels:
            vertices = vertices[vertex_labels != ign_l]
            vertex_labels = vertex_labels[vertex_labels != ign_l]
        if len(vertex_labels) != len(vertices):
            raise ValueError('Size of vertices and their labels does not match!')
        maj_vote = colorcode_vertices(coords, vertices, vertex_labels, k=k,
                                      return_color=False, nb_cpus=self.nb_cpus)
        return maj_vote

    def get_spine_compartments(self, semseg_key: str = 'spiness', k: int = 1,
                               min_spine_cc_size: Optional[int] = None,
                               dest_folder: Optional[str] = None) \
            -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Retrieve connected components of vertex spine predictions.

        Args:
            semseg_key: Key of the used semantic segmentation.
            k: Number of nearest neighbors for majority label vote (smoothing of
                classification results).
            min_spine_cc_size: Minimum number of vertices to consider a connected
                component a valid object.
            dest_folder: Default is None, else provide a path (str) to a folder.
                The mean location and size of the head and neck connected
                components will be stored as numpy array file (npy).

        Returns:
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

    def sample_locations(self, force=False, cache=True, verbose=False,
                         ds_factor=None):
        """

        Parameters
        ----------
        force : bool
            force resampling of locations
        cache : bool
            save sample location in SSO attribute dict
        verbose : bool
        ds_factor : float
            Downscaling factor to generate locations

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
        params = [[sv, {"force": force, 'save': cache,
                        'ds_factor': ds_factor}] for sv in self.svs]

        # list of arrays
        # TODO: currently does not support multiprocessing
        locs = sm.start_multiprocess_obj("sample_locations", params,
                                         nb_cpus=1)  #self.nb_cpus)
        if cache:
            self.save_attributes(["sample_locations"], [locs])
        if verbose:
            dur = time.time() - start
            log_reps.debug("Sampling locations from {} SVs took {:.2f}s."
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
        elif obj_type == "syn_ssv":
            mesh = self.syn_ssv_mesh
            # also store it as 'sj' s.t. `init_sso_from_kzip` can use it for rendering.
            # TODO: add option to rendering code which enables rendering of arbitrary cell organelles
            obj_type = 'sj'
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

    def meshes2kzip(self, dest_path=None, sv_color=None, synssv_instead_sj=False):
        """
        Writes SV, mito, vesicle cloud and synaptic junction meshes to k.zip.

        Parameters
        ----------
        dest_path : str
        sv_color : np.array
            array with RGBA values or None to use default values
            (see :func:`~mesh2kzip`).
        synssv_instead_sj : bool

        Returns
        -------

        """
        if dest_path is None:
            dest_path = self.skeleton_kzip_path
        for ot in ["sj", "vc", "mi", "sv"]:  # determins rendering order in KNOSSOS
            if ot == "sj" and synssv_instead_sj:
                ot = 'syn_ssv'
            self.mesh2kzip(obj_type=ot, dest_path=dest_path, ext_color=sv_color if
            ot == "sv" else None)

    def mesh2file(self, dest_path=None, center=None, color=None, scale=None):
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
        scale : float
            Multiplies vertex locations after centering
        """
        mesh2obj_file(dest_path, self.mesh, center=center, color=color,
                      scale=scale)

    def export2kzip(self, dest_path: str, attr_keys: Iterable[str] = (),
                    rag: Optional[nx.Graph] = None,
                    sv_color: Optional[np.ndarray] = None,
                    synssv_instead_sj: bool = False):
        """
        Writes the SSO to a KNOSSOS loadable kzip including the mergelist
        (:func:`~mergelist2kzip`), its meshes (:func:`~meshes2kzip`), data set
        specific information and additional data (`attr_keys`).
        0 to 255. Saved SSO can also be re-loaded as an SSO instance via
        :func:`~syconn.proc.ssd_assembly.init_sso_from_kzip`.

        Notes:
            Will not invoke :func:`~load_attr_dict`.

        Args:
            dest_path: Path to destination kzip file.
            attr_keys: Currently allowed: 'sample_locations', 'skeleton',
                'attr_dict', 'rag'.
            rag: SV graph of SSV with uint nodes.
            sv_color: Cell supervoxel colors. Array with RGBA (0...255) values
                or None to use default values (see :func:`~mesh2kzip`).
            synssv_instead_sj: If True, will use 'syn_ssv' objects instead of 'sj'.

        """
        # # The next two calls are deprecated but might be usefull at some point
        # self.save_skeleton_to_kzip(dest_path=dest_path)
        # self.save_objects_to_kzip_sparse(["mi", "sj", "vc"],
        #                                  dest_path=dest_path)
        tmp_dest_p = []
        target_fnames = []
        attr_keys = list(attr_keys)
        if 'rag' in attr_keys:
            if rag is None and not os.path.isfile(self.edgelist_path):
                log_reps.warn("Could not find SV graph of SSV {}. Please"
                              " pass `sv_graph` as kwarg.".format(self))
            else:
                tmp_dest_p.append('{}_rag.bz2'.format(dest_path))
                target_fnames.append('rag.bz2')
                if rag is None:
                    rag = nx.read_edgelist(self.edgelist_path, nodetype=np.uint)
                nx.write_edgelist(rag, tmp_dest_p[-1])
            attr_keys.remove('rag')

        allowed_attributes = ('sample_locations', 'skeleton', 'attr_dict')
        for attr in attr_keys:
            if attr not in allowed_attributes:
                raise ValueError('Invalid attribute specified. Currently suppor'
                                 'ted attributes for export: {}'.format(allowed_attributes))
            tmp_dest_p.append('{}_{}.pkl'.format(dest_path, attr))
            target_fnames.append('{}.pkl'.format(attr))
            sso_attr = getattr(self, attr)
            if hasattr(sso_attr, '__call__'):
                sso_attr = sso_attr()
            write_obj2pkl(tmp_dest_p[-1], sso_attr)

        # always write meta dict
        tmp_dest_p.append('{}_{}.pkl'.format(dest_path, 'meta'))
        target_fnames.append('{}.pkl'.format('meta'))
        write_obj2pkl(tmp_dest_p[-1], {'version_dict': self.version_dict,
                                       'scaling': self.scaling,
                                       'working_dir': self.working_dir})
        # write all data
        data2kzip(dest_path, tmp_dest_p, target_fnames)
        self.meshes2kzip(dest_path=dest_path, sv_color=sv_color,
                         synssv_instead_sj=synssv_instead_sj)
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
        glia_svs_key = "glia_svs" + pred_key_appendix
        nonglia_svs_key = "nonglia_svs" + pred_key_appendix
        if thresh is None:
            thresh = global_params.glia_thresh
        if recompute or not (self.attr_exists(glia_svs_key) and
                             self.attr_exists(nonglia_svs_key)):
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
            self.attr_dict[glia_svs_key] = glia_ccs_ixs
            self.attr_dict[nonglia_svs_key] = non_glia_ccs_ixs
            self.save_attributes([glia_svs_key, nonglia_svs_key],
                                 [glia_ccs_ixs, non_glia_ccs_ixs])
        else:
            log_reps.critical('Skipping SSO {}, glia splits already exist'
                              '.'.format(self.id))

    def gliasplit2mesh(self, dest_path=None, pred_key_appendix=""):
        """

        Parameters
        ----------
        dest_path :

        Returns
        -------

        """
        # TODO: adapt writemesh2kzip to work with multiple writes
        #  to same file or use write_meshes2kzip here.
        glia_svs_key = "glia_svs" + pred_key_appendix
        nonglia_svs_key = "nonglia_svs" + pred_key_appendix
        if dest_path is None:
            dest_path = self.skeleton_kzip_path_views
        # write meshes of CC's
        glia_ccs = self.attr_dict[glia_svs_key]
        for kk, glia in enumerate(glia_ccs):
            mesh = merge_someshes([self.get_seg_obj("sv", ix) for ix in
                                   glia])
            write_mesh2kzip(dest_path, mesh[0], mesh[1], mesh[2], None,
                            "glia_cc%d.ply" % kk)
        non_glia_ccs = self.attr_dict[nonglia_svs_key]
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
        # 'tmp'-version: do not write to disk
        predict_sos_views(model, self.svs, pred_key,
                          nb_cpus=self.nb_cpus, verbose=verbose,
                          woglia=False, raw_only=True,
                          return_proba=self.version == 'tmp')
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
        return np.array(self.attr_for_coords(coords, [pred_type], radius_nm))

    def attr_for_coords(self, coords, attr_keys, radius_nm=None):
        """
        TODO: move to super_segmentation_helper.py
        Query skeleton node attributes at given coordinates. Supports any
        attribute stored in self.skeleton. If radius_nm is given, will
        assign majority attribute value.

        Parameters
        ----------
        coords : np.array
            Voxel coordinates, unscaled! [N, 3]
        radius_nm : Optional[float]
            If None, will only use attribute of nearest node, otherwise
            majority attribute value is used.
        attr_keys : List[str]
            Attribute identifier
        Returns
        -------
        List
            Same length as coords. For every coordinate in coords returns the
            majority label within radius_nm or [-1] if Key does not exist.
        """
        if type(attr_keys) is str:
            attr_keys = [attr_keys]
        coords = np.array(coords)
        self.load_skeleton()
        if self.skeleton is None or len(self.skeleton["nodes"]) == 0:
            log_reps.warn("Skeleton did not exist for SSV {} (size: {}; rep. coord.: "
                          "{}).".format(self.id, self.size, self.rep_coord))
            return -1 * np.ones((len(coords), len(attr_keys)))

        # get close locations
        kdtree = scipy.spatial.cKDTree(self.skeleton["nodes"] * self.scaling)
        if radius_nm is None:
            _, close_node_ids = kdtree.query(coords * self.scaling, k=1,
                                             n_jobs=self.nb_cpus)
        else:
            close_node_ids = kdtree.query_ball_point(coords * self.scaling,
                                                     radius_nm)
        attr_dc = defaultdict(list)
        for i_coord in range(len(coords)):
            curr_close_node_ids = close_node_ids[i_coord]
            for attr_key in attr_keys:
                if attr_key not in self.skeleton:  # e.g. for glia SSV axoness does not exist.
                    attr_dc[attr_key].append(-1)
                    # # this is commented because there a legitimate cases for missing keys.
                    # # TODO: think of a better warning / error raise
                    # log_reps.warning(
                    #     "KeyError: Could not find key '{}' in skeleton of SSV with ID {}. Setting to -1."
                    #     "".format(attr_key, self.id))
                    continue
                if radius_nm is not None:  # use nodes within radius_nm, there might be multiple node ids
                    if len(curr_close_node_ids) == 0:
                        dist, curr_close_node_ids = kdtree.query(coords * self.scaling)
                        log_reps.info(
                            "Couldn't find skeleton nodes within {} nm. Using nearest "
                            "one with distance {} nm. SSV ID {}, coordinate at {}."
                            "".format(radius_nm, dist[0], self.id, coords[i_coord]))
                    cls, cnts = np.unique(
                        np.array(self.skeleton[attr_key])[np.array(curr_close_node_ids)],
                        return_counts=True)
                    if len(cls) > 0:
                        attr_dc[attr_key].append(cls[np.argmax(cnts)])
                    else:
                        log_reps.info("Did not find any skeleton node within {} nm at {}."
                                      " SSV {} (size: {}; rep. coord.: {}).".format(
                            radius_nm, i_coord, self.id, self.size, self.rep_coord))
                        attr_dc[attr_key].append(-1)
                else:  # only nearest node ID
                    attr_dc[attr_key].append(self.skeleton[attr_key][curr_close_node_ids])
        # safety in case latent morphology was not predicted / needed
        # TODO: refine mechanism for this scenario, i.e. for exporting matrix
        if "latent_morph" in attr_keys:
            latent_morph = attr_dc["latent_morph"]
            for i in range(len(latent_morph)):
                curr_latent = latent_morph[i]
                if np.isscalar(curr_latent) and curr_latent == -1:
                    curr_latent = np.array([np.inf] * global_params.ndim_embedding)
                latent_morph[i] = curr_latent
        return [np.array(attr_dc[k]) for k in attr_keys]

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

    def predict_views_embedding(self, model, pred_key_appendix="",
                                view_key=None):
        """
        This will save a latent vector which captures a local morphology fingerprint for every
        skeleton node location as :py:attr:`~skeleton`['latent_morph'] based on the nearest rendering
        location.

        Parameters
        ----------
        model :
        pred_key_appendix :
        view_key : str
            View identifier, e.g. if views have been pre-rendered and are stored in
            `self.view_dict`
        """
        from ..handler.prediction import naive_view_normalization_new
        pred_key = "latent_morph"
        pred_key += pred_key_appendix
        if self.version == 'tmp':
            log_reps.warning('"predict_views_embedding" called but this SSV '
                             'has version "tmp", results will'
                             ' not be saved to disk.')
        views = self.load_views(view_key=view_key)  # [N, 4, 2, y, x]
        # TODO: add normalization to model - prevent potentially different normalization!
        views = naive_view_normalization_new(views)
        # The inference with TNets can be optimzed, via splititng the views into three equally sized parts.
        inp = (views[:, :, 0], np.zeros_like(views[:, :, 0]), np.zeros_like(views[:, :, 0]))
        # return dist1, dist2, inp1, inp2, inp3 latent
        _, _, latent, _, _ = model.predict_proba(inp)  # only use first view for now

        # map latent vecs at rendering locs to skeleton node locations via nearest neighbor
        self.load_skeleton()
        if not 'view_ixs' in self.skeleton:
            hull_tree = spatial.cKDTree(np.concatenate(self.sample_locations()))
            dists, ixs = hull_tree.query(self.skeleton["nodes"] * self.scaling,
                                         n_jobs=self.nb_cpus, k=1)
            self.skeleton["view_ixs"] = ixs
        self.skeleton[pred_key] = latent[self.skeleton["view_ixs"]]
        self.save_skeleton()

    def cnn_axoness2skel(self, **kwargs):
        locking_tmp = self.enable_locking
        self.enable_locking = False  # all SV operations are read-only
        # (enable_locking is inherited by sso.svs);
        # SSV operations not, but SSO file structure is not chunked
        res = ssh.cnn_axoness2skel(self, **kwargs)
        self.enable_locking = locking_tmp
        return res

    def average_node_axoness_views(self, **kwargs):
        """
        Apply a sliding window averaging along the axon predictions stored at the
        nodes of the :py:attr:`~skeleton`. See
        :func:`~syconn.reps.super_segmentation_helper._average_node_axoness_views`
        for details.

        Args:
            **kwargs: Key word arguments used in
                :func:`~syconn.reps.super_segmentation_helper._average_node_axoness_views`.

        """
        locking_tmp = self.enable_locking
        self.enable_locking = False  # all SV operations are read-only
        # (enable_locking is inherited by sso.svs);
        # SSV operations not, but SSO file structure is not chunked
        res = ssh.average_node_axoness_views(self, **kwargs)
        self.enable_locking = locking_tmp
        return res

    def axoness2mesh(self, dest_path, k=1, pred_key_appendix=''):
        """
        Deprecated. See :func:`~semseg2mesh`. Write the per-location CMN axon
        predictions (img2scalar) to a kzip file.

        Args:
            dest_path: Path to the kzip file.
            k: Number of nearest neighbors used for the majority vote.
            pred_key_appendix: Key to load specific predictions.
        """
        ssh.write_axpred_cnn(self, pred_key_appendix=pred_key_appendix, k=k,
                             dest_path=dest_path)

    # --------------------------------------------------------------- CELL TYPES
    def predict_cell_type(self, ssd_version="ctgt", clf_name="rfc",
                          feature_context_nm=25000):
        raise DeprecationWarning('This method is deprecated. Use '
                                 '"predict_nodes" instead!')

    def predict_celltype_cnn(self, model, pred_key_appendix, model_tnet=None, view_props=None,
                             largeFoV=True):
        """
        Infer celltype classification via `model` (stored as `celltype_cnn_e3` and `celltype_cnn_e3_probas`)
        and an optional cell embedding via `model_tnet` (stored as `latent_morph_ct`).

        Parameters
        ----------
        model : nn.Module
        pred_key_appendix : str
        model_tnet : Optional[nn.Module]
        view_props : Optional[dict]
            Dictionary which contains view properties. If None, default defined in
            `global_params.py` will be used.

        """
        if not largeFoV:
            if view_props is None:
                view_props = {}
            return ssh.predict_sso_celltype(self, model, **view_props)  # OLD
        if view_props is None:
            view_props = global_params.view_properties_large
        ssh.celltype_of_sso_nocache(self, model, pred_key_appendix=pred_key_appendix,
                                    overwrite=False, **view_props)
        if model_tnet is not None:
            ssh.view_embedding_of_sso_nocache(self, model_tnet, pred_key_appendix=pred_key_appendix,
                                              overwrite=True, **view_props)

    def render_ortho_views_vis(self, dest_folder=None, colors=None, ws=(2048, 2048),
                               obj_to_render=("sv", )):
        from scipy.misc import imsave
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
                          index_views=False, enable_locking=True):
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
    enable_locking : bool
        enable system locking when writing views
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
        # TODO: write chunked
        v = views[part_views[i]:part_views[i + 1]]
        if np.sum(v) == 0 or np.sum(v) == np.prod(v.shape):
            log_reps.warn("Empty views detected after rendering.",
                          RuntimeWarning)
        sv_obj = sos[i]
        sv_obj.save_views(views=v, woglia=woglia, index_views=index_views,
                          cellobjects_only=cellobjects_only,
                          enable_locking=True)


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
    # from ..handler.prediction import get_celltype_model
    from ..handler.prediction import get_celltype_model_e3, get_tripletnet_model_e3, \
        get_tripletnet_model_large_e3, get_celltype_model_large_e3
    ssv_ids = args
    # randomly initialize gpu
    # m = get_celltype_model(init_gpu=0)
    if not global_params.config.use_large_fov_views_ct:
        m = get_celltype_model_e3()
        m_tnet = get_tripletnet_model_e3()
    else:
        m = get_celltype_model_large_e3()
        m_tnet = get_tripletnet_model_large_e3()
    pbar = tqdm.tqdm(total=len(ssv_ids))
    missing_ssvs = []
    for ix in ssv_ids:
        ssv = SuperSegmentationObject(ix, working_dir=global_params.config.working_dir)
        ssv.nb_cpus = 1
        ssv._view_caching = True
        try:
            if global_params.config.use_large_fov_views_ct:
                view_props = global_params.view_properties_large
                ssh.celltype_of_sso_nocache(ssv, m, overwrite=True, **view_props)
                ssh.view_embedding_of_sso_nocache(ssv, m_tnet, overwrite=True, **view_props)
            else:
                ssh.predict_sso_celltype(ssv, m, overwrite=True)  # local views
        except Exception as e:
            missing_ssvs.append((ssv.id, str(e)))
            msg = 'ERROR during celltype prediction of SSV {}. {}'.format(ssv.id, repr(e))
            log_reps.error(msg)
        pbar.update(1)
    pbar.close()
    return missing_ssvs


def semsegaxoness_predictor(args):
    """
    Predicts axoness and stores resulting labels at vertex dictionary.

    Parameters
    ----------
    args :

    Returns
    -------

    """
    from ..handler.prediction import get_semseg_axon_model
    ssv_ids, nb_cpus = args
    m = get_semseg_axon_model()
    missing_ssvs = []
    view_props = global_params.view_properties_semsegax
    pbar = tqdm.tqdm(total=len(ssv_ids))
    for ix in ssv_ids:
        ssv = SuperSegmentationObject(ix, working_dir=global_params.config.working_dir)
        ssv.nb_cpus = nb_cpus
        ssv._view_caching = True
        try:
            try:
                ssh.semseg_of_sso_nocache(ssv, m, **view_props)
            except Exception:
                # retry # TODO: facing cuda OOM errors after certain number of iterations
                del m
                del ssv
                ssv = SuperSegmentationObject(ix, working_dir=global_params.config.working_dir)
                ssv.nb_cpus = nb_cpus
                ssv._view_caching = True
                m = get_semseg_axon_model()
                ssh.semseg_of_sso_nocache(ssv, m, **view_props)
        except Exception as e:
            missing_ssvs.append((ssv.id, str(e)))
            msg = 'ERROR during sem. seg. prediction of SSV {}. {}'.format(ssv.id, repr(e))
            log_reps.error(msg)
        pbar.update()
    pbar.close()
    return missing_ssvs


