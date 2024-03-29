# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Philipp Schubert, Joergen Kornfeld
import glob
import os
import re
import shutil
import time
from collections import Counter, defaultdict
from typing import Optional, Dict, List, Tuple, Union, Iterable, Any, TYPE_CHECKING
import pickle as pkl

import networkx as nx
import numpy as np
import scipy.spatial
from scipy import spatial

from . import super_segmentation_helper as ssh
from .rep_helper import knossos_ml_from_sso, colorcode_vertices, knossos_ml_from_svixs, subfold_from_ix_SSO, \
    SegmentationBase
from .segmentation import SegmentationObject, SegmentationDataset
from .segmentation_helper import load_so_attr_bulk
from .. import global_params
from ..backend.storage import CompressedStorage, MeshStorage
from ..handler.basics import write_txt2kzip, get_filepaths_from_dir, safe_copy, coordpath2anno, load_pkl2obj, \
    write_obj2pkl, flatten_list, chunkify, data2kzip
from ..handler.config import DynConfig
from ..handler.prediction import certainty_estimate
from ..mp import batchjob_utils as qu
from ..mp import mp_utils as sm
from ..proc.graphs import split_glia, split_subcc_join, create_graph_from_coords
from ..proc.meshes import write_mesh2kzip, merge_someshes, compartmentalize_mesh, mesh2obj_file, write_meshes2kzip, \
    _calc_pca_components
from ..proc.rendering import render_sampled_sso, load_rendering_func, render_sso_coords, render_sso_coords_index_views
from ..proc.image import normalize_img
from ..proc.sd_proc import predict_sos_views
from ..reps import log_reps

if TYPE_CHECKING:
    from .super_segmentation_dataset import SuperSegmentationDataset
from knossos_utils import skeleton
from knossos_utils.skeleton_utils import load_skeleton as load_skeleton_kzip
from knossos_utils.skeleton_utils import write_skeleton as write_skeleton_kzip

try:
    from knossos_utils import mergelist_tools
except ImportError:
    from knossos_utils import mergelist_tools_fallback as mergelist_tools
from syconn.proc.graphs import stitch_skel_nx

MeshType = Union[Tuple[np.ndarray, np.ndarray, np.ndarray], List[np.ndarray],
                 Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]


class SuperSegmentationObject(SegmentationBase):
    """
    Class instances represent individual neuron reconstructions, defined by a
    list of agglomerated supervoxels (see :class:`~syconn.reps.segmentation.SegmentationObject`).

    Examples:
        This class can be used to create a cell reconstruction object after successful executing
        :func:`~syconn.exec.exec_inference.run_create_neuron_ssd` as follows::

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

        To cache SegmentationObject attributes use the ``cache_properties`` argument during initialization of the
        :class:`~syconn.reps.segmentation.SegmentationDataset` and pass it on to the ``SuperSegmentationDataset``
        instantiation:

            sd_mi = SegmentationDataset(obj_type='mi', cache_properties=['rep_coord'])
            ssd = SuperSegmentationDataset(sd_lookup=dict(mi=sd_mi))
            ssv = ssd.get_super_segmentation_object(ssd.ssv_ids[0])
            # :class:`~syconn.reps.segmentation.SegmentationObject` from ``mis`` don't require loading ``rep_coord``
            # from its storage file.
            for mi in ssv.mis:
                rc = mi.rep_coord  # normally this requires to load the attribute dict storage file.

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


        See also ``SyConn/docs/api.md`` (WIP).

    Attributes:
        attr_dict: Attribute dictionary which serves as a general-purpose container. Accessed via
            the :class:`~syconn.backend.storage.AttributeDict` interface. After successfully
            executing :func:`syconn.exec.exec_init.run_create_neuron_ssd`
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
            the same working directory. Defaults to the `Versions` entry in the `config.yml` file.
    """

    def __init__(self, ssv_id: int, version: Optional[str] = None, version_dict: Optional[Dict[str, str]] = None,
                 working_dir: Optional[str] = None, create: bool = False,
                 sv_ids: Optional[Union[np.ndarray, List[int]]] = None, scaling: Optional[np.ndarray] = None,
                 object_caching: bool = True, voxel_caching: bool = True, mesh_caching: bool = True,
                 view_caching: bool = False, config: Optional[DynConfig] = None, nb_cpus: int = 1,
                 enable_locking: bool = False, enable_locking_so: bool = False, ssd_type: Optional[str] = None,
                 ssd: Optional['SuperSegmentationDataset'] = None, sv_graph: Optional[nx.Graph] = None):
        """

        Args:
            ssv_id: unique SSV ID.
            version: Version string identifier. if 'tmp' is used, no data will
                be saved to disk.
            version_dict: Dictionary which contains the versions of other dataset types which share
                the same working directory. Defaults to the `versions` entry in the
                `config.yml`file.
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
            nb_cpus: Number of cpus for parallel jobs. will only be used in some processing steps.
            enable_locking: Enable posix locking for IO operations.
            enable_locking_so: Locking flag for all :class:`syconn.reps.segmentation.SegmentationObject` assigned.
                to this object (e.g. SV, mitochondria, vesicle clouds, ...)
            ssd_type: Type of cell reconstruction. Default: 'ssv'. If speficied and `ssd` is given, types must match.
            ssd: :py:class:`~syconn.reps.super_segmentation_dataset.SuperSegmentationDataset`; if given it will be used
                to check if property caching can be used in `:py:class:`~syconn.reps.segmentation.SegmentationDataset``.
                Property caching can be used by passing the datasets (attributes for caching have to be specified in
                init. via ``cache_properties``) of interest via the kwarg ``sd_lookup``
                during :py:attr:`~syconn.reps.super_segmentation_dataset.SuperSegmentationDataset` initialization.
            sv_graph: Sueprvoxel graph. Nodes must be uint SV IDs.
        """
        if version == 'temp':
            version = 'tmp'
        self._allow_skeleton_calc = False
        if version == "tmp":
            self.enable_locking = False
            create = False
            self._allow_skeleton_calc = True
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
        self._ssd = ssd
        if self._ssd is not None:
            if ssd_type is not None and self._ssd.type != ssd_type:
                raise TypeError(f'Mis-match between given "ssd_type"={ssd_type} and type of "ssd"={ssd}.')
            else:
                ssd_type = self._ssd.type
        elif ssd_type is None:
            ssd_type = 'ssv'
        self._type = ssd_type
        self._rep_coord = None
        self._size = None
        self._bounding_box = None

        self._objects = {}
        self.skeleton = None
        self._voxels = None
        self._voxels_xy_downsampled = None
        self._voxels_downsampled = None
        self._rag = None
        self._sv_graph_uint = sv_graph

        # init mesh dicts
        self._meshes = {"sv": None, "vc": None, "mi": None, "sj": None, "syn_ssv": None, "syn_ssv_sym": None,
                        "syn_ssv_asym": None, "er": None, "golgi": None}

        self._views = None
        self._weighted_graph = None
        self._sample_locations = None
        self._rot_mat = None
        self._label_dict = {}
        self.view_dict = {}

        if sv_ids is not None:
            self.attr_dict["sv"] = sv_ids

        self._setup_working_dir(working_dir, config, version, scaling)

        if version is None:
            try:
                self._version = self.config["versions"][self.type]
            except KeyError:
                raise Exception(f"Unclear version '{version}' during initialization of {self}.")
        elif version == "new":
            other_datasets = glob.glob(self.working_dir + "/%s_*" % self.type)
            max_version = -1
            for other_dataset in other_datasets:
                other_version = int(re.findall(r"[\d]+", os.path.basename(other_dataset))[-1])
                if max_version < other_version:
                    max_version = other_version

            self._version = max_version + 1
        else:
            self._version = version

        if version_dict is None:
            try:
                self.version_dict = self.config["versions"]
            except KeyError:
                raise ValueError(f"Unclear version '{version}' during initialization of {self}.")
        else:
            if isinstance(version_dict, dict):
                self.version_dict = version_dict
            else:
                raise ValueError("No version dict specified in config.")

        if create:
            os.makedirs(self.ssv_dir, exist_ok=True)

    def __hash__(self) -> int:
        return hash((self.id, self.type, frozenset(self.sv_ids)))

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return self.id == other.id and self.type == other.type and frozenset(self.sv_ids) == frozenset(other.sv_ids)

    def __ne__(self, other: Any) -> bool:
        return not self.__eq__(other)

    def __repr__(self) -> str:
        return (f'{type(self).__name__}(ssv_id={self.id}, ssd_type="{self.type}", '
                f'version="{self.version}", working_dir="{self.working_dir}")')

    def __getitem__(self, item):
        return self.attr_dict[item]

    # IMMEDIATE PARAMETERS
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
        Voxel size in nanometers (XYZ). Default is taken from the `config.yml`
        file and accessible via :py:attr:`~config`.
        """
        return self._scaling

    # PATHS
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
    def ssd_dir(self) -> str:
        """
        Path to the
        :class:`~syconn.reps.super_segmentation_dataset.SuperSegmentationDataset`
        directory this object belongs to.
        """
        return "%s/%s/" % (self.working_dir, self.identifier)

    @property
    def ssd_kwargs(self) -> dict:
        return dict(working_dir=self.working_dir, version=self.version, config=self._config,
                    ssd_type=self.type, sso_locking=self.enable_locking)

    @property
    def ssv_kwargs(self) -> dict:
        kw = dict(ssv_id=self.id, working_dir=self.working_dir, version=self.version, config=self._config,
                  ssd_type=self.type, enable_locking=self.enable_locking)
        if self.version == 'tmp':
            kw.update(sv_ids=self.sv_ids)
        return kw

    @property
    def ssv_dir(self) -> str:
        """
        Path to the folder where the data of this super-supervoxel is stored.
        """
        return "%s/so_storage/%s/" % (self.ssd_dir, subfold_from_ix_SSO(self.id))

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

    # IDS
    @property
    def sv_ids(self) -> np.ndarray:
        """
        All cell supervoxel IDs which are assigned to this cell reconstruction.
        """
        return np.array(self.lookup_in_attribute_dict("sv"), dtype=np.uint64)

    @property
    def sj_ids(self) -> np.ndarray:
        """
        All synaptic junction (sj) supervoxel IDs which are assigned to this
        cell reconstruction.
        """
        return np.array(self.lookup_in_attribute_dict("sj"), dtype=np.uint64)

    @property
    def mi_ids(self) -> np.ndarray:
        """
        All mitochondria (mi) supervoxel IDs which are assigned to this
        cell reconstruction.
        """
        return np.array(self.lookup_in_attribute_dict("mi"), dtype=np.uint64)

    @property
    def vc_ids(self) -> np.ndarray:
        """
        All vesicle cloud (vc) supervoxel IDs which are assigned to this
        cell reconstruction.
        """
        return np.array(self.lookup_in_attribute_dict("vc"), dtype=np.uint64)

    @property
    def dense_kzip_ids(self) -> Dict[str, int]:
        """
        ?
        """
        return dict([("mi", 1), ("vc", 2), ("sj", 3)])

    # SegmentationObjects
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
        All synaptic junctions :class:`~syconn.reps.segmentation.SegmentationObject`
        objects which are between super-supervoxels (syn_ssv) and assigned to this cell reconstruction.
        These objects are generated as an agglomeration of 'syn' objects, which themselves have been generation as a
        combination of synaptic junction (sj) and contact site (cs) objects to remove merges in the sj objects.
        """
        return self.get_seg_objects("syn_ssv")

    # MESHES
    def load_mesh(self, mesh_type) -> Optional[MeshType]:
        """
        Load mesh of a specific type, e.g. 'mi', 'sv' (cell supervoxel), 'sj' (connected
        components of the original synaptic junction predictions), 'syn_ssv' (overlap of
        'sj' with cell contact sites), 'syn_ssv_sym' and 'syn_ssv_asym' (only if syn-type
        predictions are available).

        Args:
            mesh_type: Type of :class:`~syconn.reps.segmentation.SegmentationObject` used for
                mesh retrieval.

        Returns:
            Three flat arrays: indices, vertices, normals

        Raises
            ValueError: If `mesh_type` does not exist in :py:attr:`~_meshes`.

        """
        if mesh_type in ('syn_ssv_sym', 'syn_ssv_asym'):
            self.typedsyns2mesh()
        if mesh_type not in self._meshes:
            raise ValueError(f'Unknown mesh type for objects "{mesh_type}" in {self}."')
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

    # PROPERTIES
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
                              disable_locking=True)
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
        if "axon" not in self._meshes:
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
        Factory method for :class:`~syconn.reps.segmentation.SegmentationObject`s of type `obj_type`.

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
        Factory method for :class:`~syconn.reps.segmentation.SegmentationObject` of type `obj_type`.

        Args:
            obj_type: Type of requested :class:`~syconn.reps.segmentation.SegmentationObject`.
            obj_id: ID of the requested object.

        Returns:
            The :class:`~syconn.reps.segmentation.SegmentationObject` of type `obj_type`
            sharing the same working directory as this SSV object.
        """
        kwargs = dict(enable_locking=self.enable_locking_so, mesh_caching=self.mesh_caching,
                      voxel_caching=self.voxel_caching)
        if self._ssd is not None and obj_type in self._ssd.sd_lookup and self._ssd.sd_lookup[obj_type] is not None:
            sd_obj = self._ssd.sd_lookup[obj_type]
            if str(sd_obj.version) != str(self.version_dict[obj_type]) or sd_obj.working_dir != self.working_dir:
                msg = (f'Inconsistent working directory or version for {obj_type} stored in {self} and look up '
                       f'dataset {sd_obj}.')
                log_reps.error(msg)
                raise ValueError(msg)

            return sd_obj.get_segmentation_object(obj_id, **kwargs)

        return SegmentationObject(obj_id=obj_id, obj_type=obj_type, version=self.version_dict[obj_type],
                                  working_dir=self.working_dir, create=False, scaling=self.scaling, config=self.config)

    def get_seg_dataset(self, obj_type: str) -> SegmentationDataset:
        """
        Factory method for :class:`~syconn.reps.segmentation.SegmentationDataset` of type `obj_type`.

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
        except (IOError, EOFError, pkl.UnpicklingError) as e:
            if '[Errno 2] No such file or' not in str(e):
                log_reps.critical(f"Could not load SSO attributes from {self.attr_dict_path} due to '{e}'.")
            return -1

    @property
    def sv_graph_uint(self) -> nx.Graph:
        if self._sv_graph_uint is None:
            if os.path.isfile(self.edgelist_path):
                self._sv_graph_uint = nx.read_edgelist(self.edgelist_path, nodetype=np.uint64)
            else:
                raise ValueError("Could not find graph data for SSV {}.".format(self.id))
        return self._sv_graph_uint

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
        G = self.sv_graph_uint
        # # Might be useful as soon as global graph path is available
        # else:
        #     if os.path.isfile(global_params.config.neuron_svgraph_path):
        #         G_glob = nx.read_edgelist(global_params.config.neuron_svgraph_path,
        #                                    nodetype=np.uint64)
        #         G = nx.Graph()
        #         cc = nx.node_connected_component(G_glob, self.sv_ids[0])
        #         assert len(set(cc).difference(set(self.sv_ids))) == 0, \
        #             "SV IDs in graph differ from SSV SVs."
        #         for e in G_glob.edges(cc):
        #             G.add_edge(*e)
        if len(set(list(G.nodes())).difference(set(self.sv_ids))) != 0:
            msg = "SV IDs in graph differ from SSV SVs."
            log_reps.error(msg)
            raise ValueError(msg)
        # create graph with SV nodes
        new_G = nx.Graph()
        for e in G.edges():
            new_G.add_edge(self.get_seg_obj("sv", e[0]), self.get_seg_obj("sv", e[1]))
        return new_G

    def load_sv_edgelist(self) -> List[Tuple[int, int]]:
        """
        Load the edges within the supervoxel graph.

        Returns:
            Edge list representing the supervoxel graph.
        """
        g = self.load_sv_graph()
        return list(g.edges())

    def _load_obj_mesh(self, obj_type: str = "sv", rewrite: bool = False) -> MeshType:
        """
        Load the mesh of a given `obj_type`. If :func:`~mesh_exists` is False,
        loads the meshes from the underlying sueprvoxel objects.
        TODO: Currently does not support color array!
        TODO: add support for sym. asym synapse type

        Args:
            obj_type (str):
            rewrite (bool):

        Returns:
            np.array, np.array, np.array:
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
            ind, vert, normals = merge_someshes(self.get_seg_objects(obj_type), nb_cpus=self.nb_cpus,
                                                use_new_subfold=self.config.use_new_subfold)
            if not self.version == "tmp":
                mesh_dc = MeshStorage(self.mesh_dc_path, read_only=False, disable_locking=not self.enable_locking)
                mesh_dc[obj_type] = [ind, vert, normals]
                mesh_dc.push()
        return np.array(ind, dtype=np.int32), np.array(vert, dtype=np.float32), np.array(normals, dtype=np.float32)

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
        if self.version == 'tmp':
            log_reps.warning('"save_attr_dict" called but this SSV has version "tmp", attribute dict will'
                             ' not be saved to disk.')
            return
        try:
            orig_dc = load_pkl2obj(self.attr_dict_path)
        except (IOError, EOFError, FileNotFoundError, pkl.UnpicklingError) as e:
            if '[Errno 2] No such file or' not in str(e):
                log_reps.critical(f"Could not load SSO attributes from {self.attr_dict_path} due to '{e}'. Overwriting")
            orig_dc = {}
        orig_dc.update(self.attr_dict)
        write_obj2pkl(self.attr_dict_path, orig_dc)

    def save_attributes(self, attr_keys: List[str], attr_values: List[Any]):
        """
        Writes attributes to attribute dict on file system. Does not care about
        self.attr_dict.

        Args:
            attr_keys (tuple): tuple of str
            attr_values (tuple): tuple of items
        """
        if self.version == 'tmp':
            log_reps.warning('"save_attributes" called but this SSV has version "tmp", attributes will'
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
                log_reps.critical(f"Could not load SSO attributes at {self.attr_dict_path} due to {e}.")
            attr_dict = {}
        for k, v in zip(attr_keys, attr_values):
            attr_dict[k] = v
        try:
            write_obj2pkl(self.attr_dict_path, attr_dict)
        except IOError as e:
            if not "[Errno 13] Permission denied" in str(e):
                raise (IOError, e)
            else:
                log_reps.warn("Could not save SSO attributes to %s due to missing permissions." % self.attr_dict_path,
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
        Returns the value to `attr_key` stored in :py:attr:`~attr_dict` or None if the key is not existent.

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

    def load_so_attributes(self, obj_type: str, attr_keys: List[str]):
        """
        Collect attributes from :class:`~syconn.reps.segmentation.SegmentationObject`
        of type `obj_type`.
        The attribute value ordering for each key is the same as :py:attr:`~svs`.

        Args:
            obj_type: Type of :class:`~syconn.reps.segmentation.SegmentationObject`.
            attr_keys: Keys of desired properties. Must exist for the requested
             `obj_type`.

        Returns:
            Attribute values for each key in `attr_keys`.
        """
        attr_values = [[] for _ in range(len(attr_keys))]
        for obj in self.get_seg_objects(obj_type):
            for ii, attr_key in enumerate(attr_keys):
                # lookup_in_attribute_dict uses attribute caching of the obj itself or, if enabled,
                # the SegmentationDataset cache in the SSD of this SSO.
                attr = obj.lookup_in_attribute_dict(attr_key)
                attr_values[ii].append(attr)
        return attr_values

    def calculate_size(self):
        """
        Calculates :py:attr:`size`.
        """
        self._size = np.sum(self.load_so_attributes('sv', ['size']))

    def calculate_bounding_box(self):
        """
        Calculates :py:attr:`~bounding_box` (and :py:attr:`size`).
        """
        if len(self.sv_ids) == 0:
            self._bounding_box = np.zeros((2, 3), dtype=np.int32)
            self._size = 0
            return
        self._bounding_box = np.ones((2, 3), dtype=np.int32) * np.inf
        self._size = np.inf
        bounding_boxes, sizes = self.load_so_attributes('sv', ['bounding_box', 'size'])
        self._size = np.sum(sizes)
        self._bounding_box[0] = np.min(bounding_boxes, axis=0)[0]
        self._bounding_box[1] = np.max(bounding_boxes, axis=0)[1]
        self._bounding_box = self._bounding_box.astype(np.int32)

    def calculate_skeleton(self, force: bool = False, **kwargs):
        """
        Merges existing supervoxel skeletons (``allow_ssv_skel_gen=False``) or calculates them
        from scratch using :func:`~syconn.reps.super_segmentation_helper
        .create_sso_skeletons_wrapper` otherwise (requires ``allow_ssv_skel_gen=True``).
        Skeleton will be saved at :py:attr:`~skeleton_path`.

        Args:
            force: Skips :func:`~load_skeleton` if ``force=True``.
        """
        if force or self._allow_skeleton_calc:
            return ssh.create_sso_skeletons_wrapper([self], **kwargs)
        if self.skeleton is not None and len(self.skeleton["nodes"]) != 0 \
                and not force:
            return
        ssh.create_sso_skeletons_wrapper([self], **kwargs)

    def save_skeleton_to_kzip(self, dest_path: Optional[str] = None, name: str = 'skeleton',
                              additional_keys: Optional[List[str]] = None,
                              comments: Optional[Union[np.ndarray, List[str]]] = None):
        """

        Args:
            dest_path: Destination path for k.zip file.
            name: identifier / name of saved skeleton which appears in KNOSSOS
            additional_keys: Additional skeleton keys which are converted into
            KNOSSOS skeleton node properties. Will always attempt to write out the
            keys 'axoness', 'cell_type' and 'meta'.
            comments: np.ndarray of strings or list of strings of length N where N
            equals the number of skeleton nodes. Comments will be converted into
            KNOSSOS skeleton node comments.

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
            a.comment = name

            skel_nodes = []
            for i_node in range(len(self.skeleton["nodes"])):
                c = self.skeleton["nodes"][i_node]
                r = self.skeleton["diameters"][i_node] / 2
                skel_nodes.append(skeleton.SkeletonNode().
                                  from_scratch(a, c[0], c[1], c[2], radius=r))
                pred_key_ax = "{}_avg{}".format(self.config['compartments'][
                                                    'view_properties_semsegax']['semseg_key'],
                                                self.config['compartments'][
                                                    'dist_axoness_averaging'])
                if pred_key_ax in self.skeleton:
                    skel_nodes[-1].data[pred_key_ax] = self.skeleton[pred_key_ax][
                        i_node]
                if "meta" in self.skeleton:
                    skel_nodes[-1].data["meta"] = self.skeleton["meta"][i_node]
                if additional_keys is not None:
                    for k in additional_keys:
                        skel_nodes[-1].data[k] = self.skeleton[k][i_node]
                if comments is not None:
                    skel_nodes[-1].setComment(str(comments[i_node]))

                a.addNode(skel_nodes[-1])

            for edge in self.skeleton["edges"]:
                a.addEdge(skel_nodes[edge[0]], skel_nodes[edge[1]])

            if dest_path is None:
                dest_path = self.skeleton_kzip_path
            elif not dest_path.endswith('.k.zip'):
                dest_path += '.k.zip'
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
            obj_types = self.config['process_cell_organelles']
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
        elif not dest_path.endswith('.k.zip'):
            dest_path += '.k.zip'
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

    def total_edge_length(self, compartments_of_interest: Optional[List[int]] = None,
                          ax_pred_key: str = 'axoness_avg10000') -> Union[np.ndarray, float]:
        """
        Total edge length of the super-supervoxel :py:attr:`~skeleton` in nanometers.

        Args:
            compartments_of_interest: Which compartments to take into account for calculation.
                axon: 1, dendrite: 0, soma: 2.
            ax_pred_key: Key of compartment prediction stored in :attr:`~skeleton`, only used if
                `compartments_of_interest` was set.

        Returns:
            Sum of all edge lengths (L2 norm) in :py:attr:`~skeleton`.
        """
        if self.skeleton is None:
            self.load_skeleton()
        nodes = self.skeleton["nodes"]
        edges = self.skeleton["edges"]
        if compartments_of_interest is None:
            return np.sum([np.linalg.norm(
                self.scaling * (nodes[e[0]] - nodes[e[1]])) for e in edges])
        else:
            node_labels = self.skeleton[ax_pred_key]
            edge_length = 0
            for e in edges:
                if (node_labels[e[0]] in compartments_of_interest) and (node_labels[e[1]] in compartments_of_interest):
                    edge_length += np.linalg.norm(self.scaling * (nodes[e[0]] - nodes[e[1]]))
            return edge_length

    def save_skeleton(self, to_kzip=False, to_object=True):
        """
        Saves skeleton to default locations as `.pkl` and optionally as `.k.zip`.

        Args:
            to_kzip: Stores skeleton as a KNOSSOS compatible xml inside a k.zip file.
            to_object: Stores skeleton as a dictionary in a pickle file.
        """
        if self.version == 'tmp':
            log_reps.debug('"save_skeleton" called but this SSV '
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
        ``allow_ssv_skel_gen=True``).

        Returns:
            True if successfully loaded/generated skeleton, else False.
        """
        if self.skeleton is not None:
            return True
        try:
            self.skeleton = load_pkl2obj(self.skeleton_path)
            self.skeleton["nodes"] = self.skeleton["nodes"].astype(np.float32)
            return True
        except:
            if global_params.config.allow_ssv_skel_gen:
                if global_params.config.use_kimimaro:
                    # add per ssv skeleton generation for kimimaro
                    raise NotImplementedError('Individual cells cannot be processed with kimimaro.')
                else:
                    self.calculate_skeleton()
                return True
            return False

    def celltype(self, key: Optional[str] = None) -> int:
        """
        Returns the cell type classification result. Default: CMN model, if
        `key` is specified returns the corresponding value loaded
        by :func:`~lookup_in_attribute_dict`.
        Args:
            key: Key where classification result is stored.

        Returns:
            Cell type classification.
        """
        if key is None:
            key = 'celltype_cnn_e3'
        return self.lookup_in_attribute_dict(key)

    def weighted_graph(self, add_node_attr: Iterable[str] = ()) -> nx.Graph:
        """
        Creates a Euclidean distance (in nanometers) weighted graph representation of the
        skeleton of this SSV object. The node IDs represent the index in
        the ``'node'`` array part of :py:attr:`~skeleton`. Weights are stored
        as 'weight' in the graph, this allows to use e.g.
        ``nx.single_source_dijkstra_path(..)``.

        Args:
            add_node_attr: To-be-added node attributes. Must exist in
            :py:attr`~skeleton`.

        Returns:
            The skeleton of this SSV object as a networkx graph.
        """
        if self._weighted_graph is None or np.any([len(nx.get_node_attributes(
                self._weighted_graph, k)) == 0 for k in add_node_attr]):
            if self.skeleton is None:
                self.load_skeleton()

            node_scaled = self.skeleton["nodes"] * self.scaling

            edges = np.array(self.skeleton["edges"], dtype=np.int64)
            edge_coords = node_scaled[edges]
            weights = np.linalg.norm(edge_coords[:, 0] - edge_coords[:, 1], axis=1)
            self._weighted_graph = nx.Graph()
            self._weighted_graph.add_nodes_from(
                [(ix, dict(position=coord)) for ix, coord in
                 enumerate(self.skeleton['nodes'])])
            self._weighted_graph.add_weighted_edges_from(
                [(edges[ii][0], edges[ii][1], weights[ii]) for
                 ii in range(len(weights))])

            for k in add_node_attr:
                dc = {}
                for n in self._weighted_graph.nodes():
                    dc[n] = self.skeleton[k][n]
                nx.set_node_attributes(self._weighted_graph, dc, k)
        return self._weighted_graph

    def syn_sign_ratio(self, weighted: bool = True,
                       recompute: bool = True,
                       comp_types: Optional[List[int]] = None,
                       comp_types_partner: Optional[List[int]] = None) -> float:
        """
        Ratio of symmetric synapses (between 0 and 1; -1 if no synapse objects)
        between functional compartments specified via `comp_types` and
        `comp_types_partner`.

        Todo:
            * Check default of synapse type if synapse type predictions are not
              available -> propagate to this method and return -1.

        Notes:
            Bouton predictions are converted into axon label,
            i.e. 3 (en-passant) -> 1 and 4 (terminal) -> 1.

        Args:
            weighted: Compute synapse-area weighted ratio.
            recompute: Ignore existing value.
            comp_types: All synapses that are formed on any of the
                functional compartment types given in `comp_types` are used
                for computing the ratio (0: dendrite,
                1: axon, 2: soma). Default: [1, ].
            comp_types_partner: Compartment type of the partner cell. Default:
                [0, ].

        Returns:
            (Area-weighted) ratio of symmetric synapses or -1 if no synapses.
        """
        if comp_types is None:
            comp_types = [1, ]
        if comp_types_partner is None:
            comp_types_partner = [0, ]
        ratio = self.lookup_in_attribute_dict("syn_sign_ratio")
        if not recompute and ratio is not None:
            return ratio
        syn_signs = []
        syn_sizes = []
        props = load_so_attr_bulk(self.syn_ssv, ('partner_axoness', 'syn_sign', 'mesh_area', 'neuron_partners'),
                                  use_new_subfold=self.config.use_new_subfold)
        for syn in self.syn_ssv:
            ax = np.array(props['partner_axoness'][syn.id])
            # convert boutons to axon class
            ax[ax == 3] = 1
            ax[ax == 4] = 1
            partners = props['neuron_partners'][syn.id]
            this_cell_ix = list(partners).index(self.id)
            other_cell_ix = 1 - this_cell_ix
            if ax[this_cell_ix] not in comp_types:
                continue
            if ax[other_cell_ix] not in comp_types_partner:
                continue
            syn_signs.append(props['syn_sign'][syn.id])
            syn_sizes.append(props['mesh_area'][syn.id] / 2)
        log_reps.debug(f'Used {len(syn_signs)} synapses with a total size of {np.sum(syn_sizes)} um^2 between {comp_types} '
                       f'(this cell) and {comp_types_partner} (other cells).')
        if len(syn_signs) == 0 or np.sum(syn_sizes) == 0:
            return -1
        syn_signs = np.array(syn_signs)
        syn_sizes = np.array(syn_sizes)
        if weighted:
            ratio = np.sum(syn_sizes[syn_signs == -1]) / float(np.sum(syn_sizes))
        else:
            ratio = np.sum(syn_signs == -1) / float(len(syn_signs))
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
                self.attr_dict["mapping_%s_ids" % obj_type] = list(mappings[obj_type].keys())
                self.attr_dict["mapping_%s_ratios" % obj_type] = list(mappings[obj_type].values())

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
        Default parameters for the mapping will be taken from the `config.yml` file.

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
            * duplicate of ssd_proc._apply_mapping_decisions_thread, implement common-use method

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
                lower_ratio = self.config['cell_objects']["lower_mapping_ratios"][
                    obj_type]
            except KeyError:
                msg = "Lower ratio undefined"
                log_reps.error(msg)
                raise ValueError(msg)

        if upper_ratio is None:
            try:
                upper_ratio = self.config['cell_objects']["upper_mapping_ratios"][
                    obj_type]
            except:
                log_reps.critical("Upper ratio undefined - 1. assumed")
                upper_ratio = 1.

        if sizethreshold is None:
            try:
                sizethreshold = self.config['cell_objects']["sizethresholds"][obj_type]
            except KeyError:
                msg = "Size threshold undefined"
                log_reps.error(msg)
                raise ValueError(msg)

        obj_ratios = np.array(self.attr_dict["mapping_%s_ratios" % obj_type])

        if correct_for_background:
            for i_so_id in range(
                    len(self.attr_dict["mapping_%s_ids" % obj_type])):
                so_id = self.attr_dict["mapping_%s_ids" % obj_type][i_so_id]
                obj_version = self.config["versions"][obj_type]
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
        :py:attr:`~config['process_cell_organelles']`).

        Args:
            obj_types: Type of :class:`~syconn.reps.super_segmentation_object
            .SuperSegmentationObject` which should be mapped.
            save: Saves the attribute dict of this SSV object afterwards.
        """
        if obj_types is None:
            obj_types = self.config['process_cell_organelles']
        self.aggregate_segmentation_object_mappings(obj_types, save=save)
        for obj_type in obj_types:
            # TODO: remove handling of sj?
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
        self._meshes = {"sv": None, "sj": None, "syn_ssv": None,
                        "vc": None, "mi": None, "conn": None,
                        "syn_ssv_sym": None, "syn_ssv_asym": None}
        self.skeleton = None

    def preprocess(self):
        """
        Process object mapping (requires the prior assignment of object
        candidates), cache object meshes and calculate the SSV skeleton.
        """
        self.load_attr_dict()
        self._map_cellobjects()
        for sv_type in self.config['process_cell_organelles'] + ["sv", "syn_ssv"]:
            _ = self._load_obj_mesh(obj_type=sv_type, rewrite=False)
        self.calculate_skeleton()

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
        are generated from :py:attr:`~.config`.

        Args:
            max_nb_sv: Number of supervoxels per sub-graph. This defines the sub-graph context.
            lo_first_n: Do not use first n traversed nodes for new bfs traversals.
                This allows to partition the original supervoxel graph of size `N`
                into ``N//lo_first_n`` sub-graphs.

        Returns:

        """
        if lo_first_n is None:
            lo_first_n = self.config['glia']['subcc_chunk_size_big_ssv']
        if max_nb_sv is None:
            max_nb_sv = self.config['glia']['subcc_size_big_ssv'] + 2 * (lo_first_n - 1)
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
                     overwrite: bool = True, cellobjects_only: bool = False,
                     woglia: bool = True, skip_indexviews: bool = False):
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
        """
        # TODO: partial rendering currently does not support index view generation (-> vertex
        #  indices will be different for each partial mesh)
        if len(self.sv_ids) > self.config['glia']['rendering_max_nb_sv'] and not woglia:
            if not skip_indexviews:
                raise ValueError('Index view rendering is currently not supported with partial '
                                 'cell rendering.')
            part = self.partition_cc()
            log_reps.info('Partitioned huge SSV into {} subgraphs with each {}'
                          ' SVs.'.format(len(part), len(part[0])))
            log_reps.info("Rendering SSO. {} SVs left to process"
                          ".".format(len(self.sv_ids)))
            params = [[so.id for so in el] for el in part]

            params = chunkify(params, self.config.ngpu_total * 2)
            so_kwargs = {'version': self.svs[0].version,
                         'working_dir': self.working_dir,
                         'obj_type': self.svs[0].type}
            render_kwargs = {"overwrite": overwrite, 'woglia': woglia,
                             "render_first_only": self.config['glia']['subcc_chunk_size_big_ssv'],
                             'add_cellobjects': add_cellobjects,
                             "cellobjects_only": cellobjects_only,
                             'skip_indexviews': skip_indexviews}
            params = [[par, so_kwargs, render_kwargs] for par in params]
            qu.batchjob_script(
                params, "render_views_partial", suffix="_SSV{}".format(self.id),
                n_cores=self.config['ncores_per_node'] // self.config['ngpus_per_node'],
                remove_jobfolder=True, additional_flags="--gres=gpu:1")
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

        Args:
            nb_views: int
            save: bool
            force_recompute: bool
            verbose: bool
            view_key: Optional[str]
                key used for storing view array. Default: 'index{}'.format(nb_views)
            ws: Tuple[int]
                Window size in pixels [y, x]
            comp_window: float
                Physical extent in nm of the view-window along y (see `ws` to infer pixel size)

        Returns: np.array

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

        Args:
            nb_views: int
            save: bool
            force_recompute: bool
            add_cellobjects: bool
            verbose: bool
            view_key: Optional[str]
                key used for storing view array. Default: 'raw{}'.format(nb_views)
            ws: Tuple[int]
                Window size in pixels [y, x]
            comp_window: float
                Physical extent in nm of the view-window along y (see `ws` to infer pixel size)

        Returns: np.array

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
                       raw_view_key=None, save=False, ws=None, comp_window=None,
                       add_cellobjects: Union[bool, Iterable] = True, bs: int = 10):
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

        Args:
            semseg_key (str):
            nb_views (Optional[int]):
            k (int):
            verbose (bool):
            raw_view_key (str):
                Key used for storing view array within SSO directory. Default: 'raw{}'.format(nb_views)
                If key does not exist, views will be re-rendered with properties defined
                in :py:attr:`~config` or as given in the kwargs `ws`, `nb_views` and `comp_window`.
            save (bool):
                If True, views will be saved.
            ws (tuple[int]):
                Window size in pixels [y, x]
            comp_window (float):
                Physical extent in nm of the view-window along y (see `ws` to infer pixel size)
            add_cellobjects: 
                Add cell objects. Either bool or list of structures used to render. Only
                used when `raw_view_key` or `nb_views` is None - then views are rendered on-the-fly.
            bs: 
                Batch size during inference.
        """
        view_props_default = self.config['views']['view_properties']
        if (nb_views is not None) or (raw_view_key is not None):
            # treat as special view rendering
            if nb_views is None:
                nb_views = view_props_default['nb_views']
            if raw_view_key is None:
                raw_view_key = 'raw{}'.format(nb_views)
            if raw_view_key in self.view_dict:
                views = self.load_views(raw_view_key)
            else:
                self._render_rawviews(nb_views, ws=ws, comp_window=comp_window, save=save,
                                      view_key=raw_view_key, verbose=verbose,
                                      force_recompute=True, add_cellobjects=add_cellobjects)
                views = self.load_views(raw_view_key)
            if len(views) != len(np.concatenate(self.sample_locations(cache=False))):
                raise ValueError("Unequal number of views and redering locations.")
            labeled_views = ssh.predict_views_semseg(views, m, verbose=verbose, batch_size=bs)
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
                                return_pred=self.version == 'tmp', bs=bs)  # do not write to disk

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
            if 'spiness' in semseg_key or 'dnho' in semseg_key or 'do' in semseg_key:
                cols = np.array([[0.6, 0.6, 0.6, 1], [0.9, 0.2, 0.2, 1],
                                 [0.1, 0.1, 0.1, 1], [0.05, 0.6, 0.6, 1],
                                 [0.9, 0.9, 0.9, 1], [0.1, 0.1, 0.9, 1]])
            elif 'axon' in semseg_key:
                # cols = np.array([[0.6, 0.6, 0.6, 1], [0.9, 0.2, 0.2, 1],
                #                  [0.1, 0.1, 0.1, 1], [0.9, 0.9, 0.9, 1],
                #                  [0.1, 0.1, 0.9, 1]])
                # dendrite, axon, soma, bouton, terminal, background, unpredicted
                cols = np.array([[0.6, 0.6, 0.6, 1], [0.9, 0.2, 0.2, 1],
                                 [0.1, 0.1, 0.1, 1], [0.05, 0.6, 0.6, 1],
                                 [0.8, 0.8, 0.1, 1], [0.9, 0.9, 0.9, 1],
                                 [0.1, 0.1, 0.9, 1]])
            elif 'ads' in semseg_key:
                # dendrite, axon, soma, unpredicted
                cols = np.array([[0.6, 0.6, 0.6, 1], [0.9, 0.2, 0.2, 1],
                                 [0.1, 0.1, 0.1, 1], [0.1, 0.1, 0.9, 1]])
            elif 'abt' in semseg_key:
                # axon, bouton, terminal, unpredicted
                cols = np.array([[0.9, 0.2, 0.2, 1], [0.05, 0.6, 0.6, 1],
                                 [0.8, 0.8, 0.1, 1], [0.1, 0.1, 0.9, 1]])
            elif 'dnh' in semseg_key:
                # dendrite, neck, head, unpredicted
                cols = np.array([[0.6, 0.6, 0.6, 1], [0.1, 0.1, 0.1, 1],
                                 [0.9, 0.2, 0.2, 1], [0.1, 0.1, 0.9, 1]])
            elif '3models' in semseg_key or 'dasbt' in semseg_key:
                # dendrite, axon, soma, bouton, terminal, neck, head, unpredicted
                cols = np.array([[0.6, 0.6, 0.6, 1], [0.6, 0.1, 0.1, 1],
                                 [0.1, 0.1, 0.1, 1], [0.05, 0.6, 0.6, 1],
                                 [0.4, 0.4, 0.8, 1], [0.8, 0.8, 0.1, 1],
                                 [0.9, 0.4, 0.4, 1], [0.1, 0.1, 0.9, 1]])
            else:
                raise ValueError('Semantic segmentation of "{}" is not supported.'
                                 ''.format(semseg_key))
            cols = (cols * 255).astype(np.uint8)
        return ssh.semseg2mesh(self, semseg_key, nb_views, dest_path, k,
                               cols, force_recompute=force_recompute,
                               index_view_key=index_view_key)

    def semseg_for_coords(self, coords: np.ndarray, semseg_key: str, k: int = 5,
                          ds_vertices: int = 20,
                          ignore_labels: Optional[Iterable[int]] = None):
        """
        Get the semantic segmentation with key `semseg_key` from the `k` nearest
        vertices at every coordinate in `coords`.

        Args:
            coords: np.array
                Voxel coordinates, unscaled! [N, 3]
            semseg_key: str
            k: int
                Number of nearest neighbors (NN) during k-NN classification
            ds_vertices: int
                striding factor for vertices, uses ``max(1, ds_vertices // 10)`` if
                ``len(vertices) < 5e6``.
            ignore_labels: List[int]
                Vertices with labels in `ignore_labels` will be ignored during
                majority vote, e.g. used to exclude unpredicted vertices.

        Returns: np.array
            Same length as `coords`. For every coordinate in `coords` returns the
            majority label based on its k-nearest neighbors.

        """
        # TODO: Allow multiple keys as in self.attr_for_coords, e.g. to
        #  include semseg axoness in a single query
        if ignore_labels is None:
            ignore_labels = []
        coords = np.array(coords) * self.scaling
        vertices = self.mesh[1].reshape((-1, 3))
        if len(vertices) == 0:
            return np.zeros((0, ), dtype=np.int32)
        if len(vertices) < 5e6:
            ds_vertices = max(1, ds_vertices // 10)
        vertex_labels = self.label_dict('vertex')[semseg_key][::ds_vertices]
        if np.ndim(vertex_labels) == 2:
            vertex_labels = vertex_labels.squeeze(1)
        vertices = vertices[::ds_vertices]
        for ign_l in ignore_labels:
            vertices = vertices[vertex_labels != ign_l]
            vertex_labels = vertex_labels[vertex_labels != ign_l]
        if len(vertex_labels) != len(vertices):
            raise ValueError('Size of vertices and their labels does not match!')
        if len(vertices) < k:
            log_reps.warning(f'Number of vertices ({len(vertices)}) is less than the given '
                             f'value of k ({k}). Setting k to lower value.')
            k = len(vertices)
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
            min_spine_cc_size = self.config['spines']['min_spine_cc_size']
        vertex_labels = self.label_dict('vertex')[semseg_key]
        vertices = self.mesh[1].reshape((-1, 3))
        max_dist = self.config['spines']['min_edge_dist_spine_graph']
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
        neck_c = (cc_coords[cc_labels == 0] / self.scaling).astype(np.uint64)
        neck_s = sizes[cc_labels == 0]
        head_c = (cc_coords[cc_labels == 1] / self.scaling).astype(np.uint64)
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

        Args:
            force: bool
                force resampling of locations
            cache: bool
                save sample location in SSO attribute dict
            verbose: bool
            ds_factor: float
                Downscaling factor to generate locations

        Returns: list of array
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
                                         nb_cpus=1)  # self.nb_cpus)
        if cache:
            self.save_attributes(["sample_locations"], [locs])
        if verbose:
            dur = time.time() - start
            log_reps.debug("Sampling locations from {} SVs took {:.2f}s."
                           " {.4f}s/SV (incl. read/write)".format(
                len(self.sv_ids), dur, dur / len(self.sv_ids)))
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

    def write_locations2kzip(self, dest_path: Optional[str] = None):
        if dest_path is None:
            dest_path = self.skeleton_kzip_path_views
        elif not dest_path.endswith('.k.zip'):
            dest_path += '.k.zip'
        loc = np.concatenate(self.sample_locations())
        new_anno = coordpath2anno(loc, add_edges=False)
        new_anno.setComment("sample_locations")
        write_skeleton_kzip(dest_path, [new_anno])

    def mergelist2kzip(self, dest_path: Optional[str] = None):
        if len(self.attr_dict) == 0:
            self.load_attr_dict()
        kml = knossos_ml_from_sso(self)
        if dest_path is None:
            dest_path = self.skeleton_kzip_path
        write_txt2kzip(dest_path, kml, "mergelist.txt")

    def mesh2kzip(self, dest_path: Optional[str] = None, obj_type: str = "sv",
                  ext_color: Optional[np.ndarray] = None, **kwargs):
        """
        Writes mesh of SSV to kzip as .ply file.

        Args:
            dest_path:
            obj_type: str
                'sv' for cell surface, 'mi': mitochondria, 'vc': vesicle clouds,
                'sj': synaptic junctions
            ext_color: np.array of scalar
                If scalar, it has to be an integer between 0 and 255.
                If array, it has to be of type uint/int and of shape (N, 4) while N
                is the number of vertices of the SSV cell surface mesh:
                N = len(self.mesh[1].reshape((-1, 3)))

        Returns:

        """
        color = None
        if dest_path is None:
            dest_path = self.skeleton_kzip_path
        # TODO: revisit re-definition of `obj_type` to 'sj'.
        if obj_type == "syn_ssv":
            mesh = self.syn_ssv_mesh
            # also store it as 'sj' s.t. `init_sso_from_kzip` can use it for rendering.
            # TODO: add option to rendering code which enables rendering of arbitrary cell organelles
            obj_type = 'sj'
        else:
            mesh = self.load_mesh(obj_type)
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
                        ply_fname=obj_type + ".ply", **kwargs)

    def meshes2kzip(self, dest_path: Optional[str] = None, sv_color: Optional[np.ndarray] = None,
                    synssv_instead_sj: bool = True, object_types: Optional[List[str]] = None, **kwargs):
        """
        Writes SV, mito, vesicle cloud and synaptic junction meshes to k.zip.

        Args:
            dest_path: str
            sv_color: np.array
                array with RGBA values or None to use default values
                (see :func:`~mesh2kzip`).
            synssv_instead_sj: bool
            object_types: List[str]
                Objects to export.

        Returns:

        """
        if dest_path is None:
            dest_path = self.skeleton_kzip_path
        if object_types is None:
            object_types = ["sj", "vc", "mi", "sv"]
        for ot in object_types:  # determines rendering order in KNOSSOS
            if ot == "sj" and synssv_instead_sj:
                ot = 'syn_ssv'
            self.mesh2kzip(obj_type=ot, dest_path=dest_path,
                           ext_color=sv_color if ot == "sv" else None, **kwargs)

    def mesh2file(self, dest_path=None, center=None, color=None, scale=None, obj_type='sv'):
        """
        Writes mesh to file (e.g. .ply, .stl, .obj) via the 'openmesh' library.
        If possible, writes it as binary.

        Args:
            dest_path: str
            center: np.array
                scaled center coordinates (in nm).
            color: np.array
                Either single color (1D; will be applied to all vertices) or
                per-vertex color array (2D).
            scale: float
                Multiplies vertex locations after centering.
            obj_type: str
                Defines the object type which is used for loading the mesh
                via :func:`~load_mesh`.
        """
        mesh2obj_file(dest_path, self.load_mesh(obj_type), center=center, color=color,
                      scale=scale)

    def export2kzip(self, dest_path: str, attr_keys: Iterable[str] = ('skeleton',),
                    rag: Optional[nx.Graph] = None,
                    sv_color: Optional[np.ndarray] = None, individual_sv_meshes: bool = True,
                    object_meshes: Optional[tuple] = None, synssv_instead_sj: bool = True):
        """
        Writes the SSO to a KNOSSOS loadable kzip including the mergelist
        (:func:`~mergelist2kzip`), its meshes (:func:`~meshes2kzip`), data set
        specific information and additional data (`attr_keys`).
        0 to 255. Saved SSO can also be re-loaded as an SSO instance via
        :func:`~syconn.proc.ssd_assembly.init_sso_from_kzip`.

        Todo:
            * Switch to .json format for storing meta information.

        Notes:
            Will not invoke :func:`~load_attr_dict`.

        Args:
            dest_path: Path to destination kzip file.
            attr_keys: Currently allowed: 'sample_locations', 'skeleton',
                'attr_dict', 'rag'.
            rag: SV graph of SSV with uint nodes.
            sv_color: Cell supervoxel colors. Array with RGBA (0...255) values
                or None to use default values (see :func:`~mesh2kzip`).
            individual_sv_meshes: Export meshes of cell supervoxels individually.
            object_meshes: Defaults to subcellular organelles defined in config.yml
                ('process_cell_organelles').
            synssv_instead_sj: If True, will use 'syn_ssv' objects instead of 'sj'.


        """
        # # The next two calls are deprecated but might be useful at some point
        # self.save_skeleton_to_kzip(dest_path=dest_path)
        # self.save_objects_to_kzip_sparse(["mi", "sj", "vc"],
        #                                  dest_path=dest_path)
        if not dest_path.endswith('.k.zip'):
            dest_path += '.k.zip'
        if os.path.isfile(dest_path):
            raise FileExistsError(f'k.zip file already exists at "{dest_path}".')
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
                    rag = self.sv_graph_uint
                nx.write_edgelist(rag, tmp_dest_p[-1])
            attr_keys.remove('rag')

        if object_meshes is None:
            object_meshes = list(self.config['process_cell_organelles']) + ['sv', 'syn_ssv']
        else:
            object_meshes = list(object_meshes)

        allowed_attributes = ('sample_locations', 'skeleton', 'attr_dict')
        for attr in attr_keys:
            if attr not in allowed_attributes:
                raise ValueError('Invalid attribute specified. Currently suppor'
                                 'ted attributes for export: {}'.format(allowed_attributes))
            if attr == 'skeleton' and self.skeleton is None:
                self.load_skeleton()
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
                                       'working_dir': self.working_dir,
                                       'sso_id': self.id})
        # write all data
        data2kzip(dest_path, tmp_dest_p, target_fnames)
        if individual_sv_meshes and 'sv' in object_meshes:
            object_meshes.remove('sv')
            self.write_svmeshes2kzip(dest_path, force_overwrite=False)
        self.meshes2kzip(dest_path=dest_path, sv_color=sv_color, force_overwrite=False,
                         synssv_instead_sj=synssv_instead_sj, object_types=object_meshes)
        self.mergelist2kzip(dest_path=dest_path)
        if 'skeleton' in attr_keys:
            self.save_skeleton_to_kzip(dest_path=dest_path)

    def typedsyns2mesh(self, dest_path: Optional[str] = None, rewrite: bool = False):
        """
        Generates typed meshes of 'syn_ssv' and stores it at :py:attr:`~mesh_dc_path`
        (keys: ``'syn_ssv_sym'`` and ``'syn_ssv_asym'``) and writes it to `dest_path` (if given).
        Accessed with the respective keys via :py:attr:`~load_mesh`.

        Synapse types are looked up in the 'syn_ssv' AttributeDicts and treated as follows:
            * excitatory / asymmetric: 1
            * inhibitory / symmetric: -1

        Args:
            dest_path: Optional output path for the synapse meshes.
            rewrite: Ignore existing meshes in :py:attr:`~_meshes` or at :py:attr:`~mesh_dc_path`.
        """
        if not rewrite and self.mesh_exists('syn_ssv_sym') and self.mesh_exists('syn_ssv_asym') \
                and not self.version == "tmp":
            return
        syn_signs = load_so_attr_bulk(self.syn_ssv, 'syn_sign', use_new_subfold=self.config.use_new_subfold)
        sym_syns = []
        asym_syns = []
        for syn in self.syn_ssv:
            syn_sign = syn_signs[syn.id]
            if syn_sign == -1:
                sym_syns.append(syn)
            elif syn_sign == 1:
                asym_syns.append(syn)
            else:
                raise ValueError(f'Unknown synapse sign {syn_sign}.')
        sym_syn_mesh = list(merge_someshes(sym_syns, use_new_subfold=self.config.use_new_subfold))
        asym_syn_mesh = list(merge_someshes(asym_syns, use_new_subfold=self.config.use_new_subfold))
        if self.version is not "tmp":
            mesh_dc = MeshStorage(self.mesh_dc_path, read_only=False,
                                  disable_locking=not self.enable_locking)
            mesh_dc['syn_ssv_sym'] = sym_syn_mesh
            mesh_dc['syn_ssv_asym'] = asym_syn_mesh
            mesh_dc.push()
        self._meshes['syn_ssv_sym'] = sym_syn_mesh
        self._meshes['syn_ssv_asym'] = asym_syn_mesh
        if dest_path is None:
            return
        # TODO: add appropriate ply fname and/or comment
        write_mesh2kzip(dest_path, asym_syn_mesh[0], asym_syn_mesh[1],
                        asym_syn_mesh[2], color=np.array((240, 50, 50, 255)), ply_fname='10.ply')
        write_mesh2kzip(dest_path, sym_syn_mesh[0], sym_syn_mesh[1],
                        sym_syn_mesh[2], color=np.array((50, 50, 240, 255)), ply_fname='11.ply')

    def write_svmeshes2kzip(self, dest_path: Optional[str] = None, **kwargs):
        """
        Write individual cell supervoxel ('sv') meshes in ply format to kzip file.

        Args:
            dest_path: Target file name.
        """
        if dest_path is None:
            dest_path = self.skeleton_kzip_path
        inds, verts, norms, cols, ply_fnames = [], [], [], [], []
        for sv in self.svs:
            inds.append(sv.mesh[0])
            verts.append(sv.mesh[1])
            norms.append(sv.mesh[2])
            cols.append(None)
            ply_fnames.append(f"sv_{sv.id}.ply")
        write_meshes2kzip(dest_path, inds, verts, norms, cols, ply_fnames=ply_fnames, **kwargs)

    def _svattr2mesh(self, dest_path, attr_key, cmap, normalize_vals=False):
        sv_attrs = np.array([sv.lookup_in_attribute_dict(attr_key).squeeze()
                             for sv in self.svs])
        if normalize_vals:
            min_val = sv_attrs.min()
            sv_attrs -= min_val
            sv_attrs /= sv_attrs.max()
        ind, vert, norm, col = merge_someshes(self.svs, color_vals=sv_attrs, cmap=cmap,
                                              use_new_subfold=self.config.use_new_subfold)
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

    def _pred2mesh(self, pred_coords: np.ndarray, preds: np.ndarray, ply_fname: Optional[str] = None,
                   dest_path: Optional[str] = None, colors: Optional[Union[tuple, np.ndarray, list]] = None,
                   k: int = 1, **kwargs):
        """
        If dest_path or ply_fname is None then indices, vertices, colors are
        returned. Else Mesh is written to k.zip file as specified.

        Args:
            pred_coords: N x 3; scaled to nm
            preds: Label array (N x 1).
            ply_fname: str
            dest_path: str
            colors: Color for each possible prediction value (range(np.max(preds))
            k: Number of nearest neighbors (average prediction)
            **kwargs: Keyword arguments passed to `colorcode_vertices`.

        Returns: None or [np.array, np.array, np.array]

        """
        if ply_fname is not None and not ply_fname.endswith(".ply"):
            ply_fname += ".ply"
        if dest_path is not None and ply_fname is None:
            msg = "Specify 'ply_fanme' in order to save colored " \
                  "mesh to k.zip."
            log_reps.error(msg)
            raise ValueError(msg)
        mesh = self.mesh
        col = colorcode_vertices(mesh[1].reshape((-1, 3)), pred_coords,
                                 preds, colors=colors, k=k, **kwargs)
        if dest_path is None:
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
            thresh = self.config['glia']['glia_thresh']
        astrocyte_svs = [sv for sv in self.svs if sv.glia_pred(thresh, pred_key_appendix) == 1]
        nonastrocyte_svs = [sv for sv in self.svs if sv.glia_pred(thresh, pred_key_appendix) == 0]
        if dest_path is None:
            dest_path = self.skeleton_kzip_path_views
        mesh = merge_someshes(astrocyte_svs, use_new_subfold=self.config.use_new_subfold)
        neuron_mesh = merge_someshes(nonastrocyte_svs, use_new_subfold=self.config.use_new_subfold)
        write_meshes2kzip(dest_path, [mesh[0], neuron_mesh[0]], [mesh[1], neuron_mesh[1]],
                          [mesh[2], neuron_mesh[2]], [None, None],
                          ["glia_%0.2f.ply" % thresh, "nonglia_%0.2f.ply" % thresh])

    def gliapred2mergelist(self, dest_path=None, thresh=None,
                           pred_key_appendix=""):
        if thresh is None:
            thresh = self.config['glia']['glia_thresh']
        if dest_path is None:
            dest_path = self.skeleton_kzip_path_views
        params = [[sv, ] for sv in self.svs]
        coords = sm.start_multiprocess_obj("rep_coord", params, nb_cpus=self.nb_cpus)
        coords = np.array(coords)
        params = [[sv, {"thresh": thresh, "pred_key_appendix": pred_key_appendix}]
                  for sv in self.svs]
        glia_preds = sm.start_multiprocess_obj("glia_pred", params,
                                               nb_cpus=self.nb_cpus)
        glia_preds = np.array(glia_preds)
        glia_comments = ["%0.4f" % gp for gp in glia_preds]
        kml = knossos_ml_from_svixs([sv.id for sv in self.svs], coords,
                                    comments=glia_comments)
        write_txt2kzip(dest_path, kml, "mergelist.txt")

    def gliasplit(self, recompute=False, thresh=None, verbose=False, pred_key_appendix=""):
        astrocyte_svs_key = "astrocyte_svs" + pred_key_appendix
        neuron_svs_key = "neuron_svs" + pred_key_appendix
        if thresh is None:
            thresh = self.config['glia']['glia_thresh']
        if recompute or not (self.attr_exists(astrocyte_svs_key) and
                             self.attr_exists(neuron_svs_key)):
            if verbose:
                log_reps.debug("Splitting glia in SSV {} with {} SV's.".format(
                    self.id, len(self.sv_ids)))
                start = time.time()
            nonglia_ccs, astrocyte_ccs = split_glia(self, thresh=thresh,
                                                    pred_key_appendix=pred_key_appendix)
            if verbose:
                log_reps.debug("Splitting glia in SSV %d with %d SV's finished "
                               "after %.4gs." % (self.id, len(self.sv_ids),
                                                 time.time() - start))
            non_glia_ccs_ixs = [[so.id for so in nonglia] for nonglia in
                                nonglia_ccs]
            astrocyte_ccs_ixs = [[so.id for so in glia] for glia in astrocyte_ccs]
            self.attr_dict[astrocyte_svs_key] = astrocyte_ccs_ixs
            self.attr_dict[neuron_svs_key] = non_glia_ccs_ixs
            self.save_attributes([astrocyte_svs_key, neuron_svs_key],
                                 [astrocyte_ccs_ixs, non_glia_ccs_ixs])
        else:
            log_reps.critical('Skipping SSO {}, glia splits already exist'
                              '.'.format(self.id))

    def gliasplit2mesh(self, dest_path=None, pred_key_appendix=""):
        """

        Args:
            dest_path:
            pred_key_appendix:

        Returns:

        """
        # TODO: adapt writemesh2kzip to work with multiple writes
        #  to same file or use write_meshes2kzip here.
        astrocyte_svs_key = "astrocyte_svs" + pred_key_appendix
        neuron_svs_key = "neuron_svs" + pred_key_appendix
        if dest_path is None:
            dest_path = self.skeleton_kzip_path_views
        # write meshes of CC's
        astrocyte_ccs = self.attr_dict[astrocyte_svs_key]
        for kk, astrocyte in enumerate(astrocyte_ccs):
            mesh = merge_someshes([self.get_seg_obj("sv", ix) for ix in
                                   astrocyte], use_new_subfold=self.config.use_new_subfold)
            write_mesh2kzip(dest_path, mesh[0], mesh[1], mesh[2], None,
                            "astrocyte_cc%d.ply" % kk)
        non_glia_ccs = self.attr_dict[neuron_svs_key]
        for kk, nonglia in enumerate(non_glia_ccs):
            mesh = merge_someshes([self.get_seg_obj("sv", ix) for ix in
                                   nonglia], use_new_subfold=self.config.use_new_subfold)
            write_mesh2kzip(dest_path, mesh[0], mesh[1], mesh[2], None,
                            "nonglia_cc%d.ply" % kk)

    def morphembed2mesh(self, dest_path, pred_key='latent_morph', whiten=True):
        """
        Write morphology embedding as RGB to k.zip file.

        Args:
            dest_path:
            pred_key:
            whiten:

        Returns:

        """
        if self.skeleton is None:
            self.load_skeleton()
        d = np.array(self.skeleton[pred_key])
        if whiten:
            d -= d.mean(axis=0)
        eig = _calc_pca_components(d)
        d_transf = np.dot(d, eig[:, :3])
        d_transf -= d_transf.min(axis=0)
        d_transf /= d_transf.max(axis=0)
        vert_col = colorcode_vertices(self.mesh[1].reshape((-1, 3)), self.skeleton['nodes'] * self.scaling,
                                      np.arange(len(self.skeleton['nodes'])), normalize_img(d_transf))
        self.mesh2kzip(dest_path, ext_color=vert_col)

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
                       "%0.4fs/SV" % (len(self.sv_ids), end - start,
                                      float(end - start) / len(self.sv_ids)))

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

    def axoness_for_coords(self, coords, radius_nm=4000, pred_type="axoness"):
        """
        Dies not need to be axoness, it supports any attribut stored in self.skeleton.

        Args:
            coords: np.array
                Voxel coordinates, unscaled! [N, 3]
            radius_nm: float
            pred_type: str

        Returns: np.array
            Same length as coords. For every coordinate in coords returns the
            majority label within radius_nm

        """
        return np.array(self.attr_for_coords(coords, [pred_type], radius_nm))

    def attr_for_coords(self, coords, attr_keys, radius_nm=None, k=1):
        """
        TODO: move to super_segmentation_helper.py
        Query skeleton node attributes at given coordinates. Supports any
        attribute stored in self.skeleton. If radius_nm is given, will
        assign majority attribute value.

        Args:
            coords (np.array):
                Voxel coordinates, unscaled! [N, 3]
            radius_nm (Optional[float]):
                If None, will only use attribute of nearest node, otherwise
                majority attribute value is used.
            attr_keys (List[str]):
                Attribute identifier
            k (int):
                Number of nearest neighbors, only if `radius_nm` is None.

        Returns:
            list:
                Same length as coords. For every coordinate in coords returns the
                majority label within radius_nm or [-1] if Key does not exist.
        """
        if type(attr_keys) is str:
            attr_keys = [attr_keys]
        coords = np.array(coords)
        if self.skeleton is None:
            self.load_skeleton()
        if self.skeleton is None or len(self.skeleton["nodes"]) == 0:
            log_reps.warn("Skeleton did not exist for SSV {} (size: {}; rep. coord.: "
                          "{}).".format(self.id, self.size, self.rep_coord))
            return -1 * np.ones((len(coords), len(attr_keys)))

        # get close locations
        if k > 1 and len(self.skeleton["nodes"]) < k:
            log_reps.warn(f'Number of skeleton nodes ({len(self.skeleton["nodes"])}) '
                          f'is smaller than k={k} in SSO {self.id}. Lowering k.')
            k = len(self.skeleton["nodes"])
        kdtree = scipy.spatial.cKDTree(self.skeleton["nodes"] * self.scaling)
        if radius_nm is None:
            _, close_node_ids = kdtree.query(coords * self.scaling, k=k, n_jobs=self.nb_cpus)
        else:
            close_node_ids = kdtree.query_ball_point(coords * self.scaling, radius_nm)
        attr_dc = defaultdict(list)
        for i_coord in range(len(coords)):
            curr_close_node_ids = close_node_ids[i_coord]
            for attr_key in attr_keys:
                # e.g. for glia SSV axoness does not exist.
                if attr_key not in self.skeleton:
                    el = -1 if k == 1 else [-1] * k
                    attr_dc[attr_key].append(el)
                    continue
                # use nodes within radius_nm, there might be multiple node ids
                if radius_nm is not None:
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
        # in case latent morphology was not predicted / needed
        if "latent_morph" in attr_keys:
            latent_morph = attr_dc["latent_morph"]
            for i in range(len(latent_morph)):
                curr_latent = latent_morph[i]
                if np.isscalar(curr_latent) and curr_latent == -1:
                    curr_latent = np.array([np.inf] * self.config['tcmn']['ndim_embedding'])
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
                       "%0.4fs/SV" % (len(self.sv_ids), end - start,
                                      float(end - start) / len(self.sv_ids)))

    def predict_views_embedding(self, model, pred_key_appendix="", view_key=None):
        """
        This will save a latent vector which captures a local morphology fingerprint for every
        skeleton node location as :py:attr:`~skeleton`['latent_morph'] based on the nearest rendering
        location.

        Notes:
            * This method requires existing :py:attr:`~views`. For on the fly view rendering use
              :py:func:`~syconn.reps.super_segmentation_helper.view_embedding_of_sso_nocache`

        Todo:
            * Add option for on the fly rendering and call
              :py:func:`~syconn.reps.super_segmentation_helper.view_embedding_of_sso_nocache` in here.

        Args:
            model:
            pred_key_appendix:
            view_key: str
                View identifier, e.g. if views have been pre-rendered and are stored in
                `self.view_dict`

        Returns:

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
        if 'view_ixs' not in self.skeleton:
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
        for details. Will call :func:`~save_skeleton`.

        Args:
            **kwargs: Key word arguments used in
                :func:`~syconn.reps.super_segmentation_helper._average_node_axoness_views`.

        """
        locking_tmp = self.enable_locking
        self.enable_locking = False  # all SV operations are read-only
        # (enable_locking is inherited by sso.svs);
        # SSV operations not, but SSO file structure is not chunked
        res = ssh.average_node_axoness_views(self, **kwargs)
        self.save_skeleton()
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
    def predict_celltype_multiview(self, model, pred_key_appendix, model_tnet=None, view_props=None,
                                   onthefly_views=False, overwrite=True, model_props=None,
                                   verbose: bool = False, save_to_attr_dict: bool = True):
        """
        Infer celltype classification via `model` (stored as ``celltype_cnn_e3`` and
        ``celltype_cnn_e3_probas`` in the :py:attr:`~attr_dict`) and an optional
        cell morphology embedding via `model_tnet` (stored as ``latent_morph_ct``).

        Args:
            model: nn.Module
            pred_key_appendix: str
            model_tnet: Optional[nn.Module]
            view_props: Optional[dict]
                Dictionary which contains view properties. If None, default defined in
                :py:attr:`~config` will be used.
            onthefly_views: bool
            overwrite:
            model_props: Model properties. See config.yml for an example.
            verbose:
            save_to_attr_dict: Save prediction in attr_dict.

        """
        if model_props is None:
            model_props = {}
        view_props_def = self.config['views']['view_properties']
        if view_props is not None:
            view_props_def.update(view_props)
        view_props = view_props_def
        if not onthefly_views:
            ssh.predict_sso_celltype(self, model, pred_key_appendix=pred_key_appendix,
                                     save_to_attr_dict=save_to_attr_dict, overwrite=overwrite, **model_props)
        else:
            ssh.celltype_of_sso_nocache(self, model, pred_key_appendix=pred_key_appendix,
                                        save_to_attr_dict=save_to_attr_dict,
                                        overwrite=overwrite, verbose=verbose, **view_props, **model_props)
        if model_tnet is not None:
            view_props = dict(view_props)  # create copy
            if 'use_syntype' in view_props:
                del view_props['use_syntype']
            ssh.view_embedding_of_sso_nocache(self, model_tnet, pred_key_appendix=pred_key_appendix,
                                              overwrite=True, **view_props)

    def predict_cell_morphology_pts(self, **kwargs):
        """
        Store local cell morphology with key 'latent_morph' (+ `pred_key_appendix`) in the SSV skeleton.

        Args:
            **kwargs:
        """
        from syconn.handler.prediction_pts import infere_cell_morphology_ssd
        ssd_kwargs = dict(working_dir=self.working_dir, config=self.config)
        ssv_params = [dict(ssv_id=self.id, **ssd_kwargs)]
        infere_cell_morphology_ssd(ssv_params, **kwargs)

    def render_ortho_views_vis(self, dest_folder=None, colors=None, ws=(2048, 2048),
                               obj_to_render=("sv",)):
        multi_view_sso = load_rendering_func('multi_view_sso')
        if colors is None:
            colors = {"sv": (0.5, 0.5, 0.5, 0.5), "mi": (0, 0, 1, 1),
                      "vc": (0, 1, 0, 1), "sj": (1, 0, 0, 1)}
        views = multi_view_sso(self, colors, ws=ws, obj_to_render=obj_to_render)
        if dest_folder:
            from scipy.misc import imsave  # TODO: use new imageio package
            for ii, v in enumerate(views):
                imsave("%s/SSV_%d_%d.png" % (dest_folder, self.id, ii), v)
        else:
            return views

    def certainty_celltype(self, pred_key: Optional[str] = None) -> float:
        """_probas
        Certainty estimate of the celltype prediction:
            1. If `is_logit` is True, Generate pseudo-probabilities from the
               input using softmax.
            2. Sum the evidence per class and (re-)normalize.
            3. Compute the entropy, scale it with the maximum entropy (equal
               probabilities) and subtract it from 1.

        Notes:
            See :func:`~syconn.handler.prediction.certainty_estimate`

        Args:
            pred_key: Key of classification results (one C-class probability
                vector for every multi-view sample). ``pred_key + '_probas'`` must exist in
                :py:attr:`~attr_dict`.

        Returns:
            Certainty measure based on the entropy of the cell type logits.
        """
        if pred_key is None:
            pred_key = 'celltype_cnn_e3'
        cert = self.lookup_in_attribute_dict(pred_key + '_certainty')
        if cert is not None:
            return cert

        logits = self.lookup_in_attribute_dict(pred_key + '_probas')
        return certainty_estimate(logits, is_logit=True)

    def majority_vote(self, prop_key: str, max_dist: float) -> np.ndarray:
        """
        Smooths (average using sliding window of 2 times max_dist and majority
        vote) property prediction in annotation.

        Args:
            prop_key: Property to average.
            max_dist: Maximum distance (in nm) for sliding window used in majority voting.

        Returns:

        """
        assert prop_key in self.skeleton, "Given key does not exist in self.skeleton"
        prop_array = self.skeleton[prop_key]
        assert prop_array.squeeze().ndim == 1, "Property array has to be 1D."
        maj_votes = np.zeros_like(prop_array)
        for ii in range(len(self.skeleton["nodes"])):
            paths = nx.single_source_dijkstra_path(self.weighted_graph(),
                                                   ii, max_dist)
            neighs = np.array(list(paths.keys()), dtype=np.int64)
            labels, cnts = np.unique(prop_array[neighs], return_counts=True)
            maj_label = labels[np.argmax(cnts)]
            maj_votes[ii] = maj_label
        return maj_votes

    def shortestpath2soma(self, coordinates: np.ndarray,
                          axoness_key: Optional[str] = None) -> List[float]:
        """
        Computes the shortest path to the soma along :py:attr:`~skeleton`.
        Cell compartment predictions must exist in ``self.skeleton['axoness_avg10000']``,
        see :func:`~syconn.exec.exec_inference.run_semsegaxoness_mapping`.
        Requires a populated :py:attr:`~skeleton`, e.g. via :func:`~load_skeleton`.

        Args:
            coordinates: Starting coordinates in voxel coordinates; shape of (N, 3).
            axoness_key: Key to axon prediction stored in :py:attr:`~skeleton`.

        Raises:
            KeyError: If axon prediction does not exist.

        Examples:
            To get the shortest paths between all synapses and the soma use::

                from syconn.reps.super_segmentation import *
                from syconn import global_params

                global_params.wd = '~/SyConn/example_cube1/'
                ssd = SuperSegmentationDataset()
                # get any cell reconstruction
                ssv = ssd.get_super_segmentation_object(ssd.ssv_ids[0])
                # get synapse coordinates in voxels.
                syns = np.array([syn.rep_coord for syn in ssv.syn_ssv])
                shortest_paths = ssv.shortestpath2soma(syns)

        Returns:
            The shortest path in nanometers for each start coordinate.
        """
        if axoness_key is None:
            axoness_key = 'axoness_avg{}'.format(self.config['compartments'][
                                                     'dist_axoness_averaging'])
        nodes = self.skeleton['nodes']
        soma_ixs = np.nonzero(self.skeleton[axoness_key] == 2)[0]
        if np.sum(soma_ixs) == 0:
            return [np.inf] * len(coordinates)
        graph = self.weighted_graph(add_node_attr=[axoness_key])
        kdt = scipy.spatial.cKDTree(nodes)
        dists, start_ixs = kdt.query(coordinates, n_jobs=self.nb_cpus)
        log_reps.debug(f'Computing shortest paths to soma for {len(start_ixs)} '
                       f'starting nodes.')
        shortest_paths_of_interest = []
        for ix in start_ixs:
            shortest_paths = nx.single_source_dijkstra_path_length(graph, ix)
            # get the shortest path to a soma
            curr_path = np.min([shortest_paths[soma_ix] for soma_ix in soma_ixs])
            shortest_paths_of_interest.append(curr_path)
        return shortest_paths_of_interest

    def path_density_seg_obj(self, obj_type: str, compartments_of_interest: Optional[List[int]] = None,
                             ax_pred_key: str = 'axoness_avg10000') -> float:
        """

        Args:
            obj_type: Key to any available sub-cellular structure.
            compartments_of_interest: Which compartments to take into account for calculation.
                axon: 1, dendrite: 0, soma: 2
            ax_pred_key: Key of compartment prediction stored in :attr:`~skeleton`, only used if
                `compartments_of_interest` was set.

        Returns:
            Average volume per path length (um^3 / um).
        """
        objs = np.array(self.get_seg_objects(obj_type))
        if self.skeleton is None:
            self.load_skeleton()
        skel = self.skeleton
        if compartments_of_interest is not None:
            node_labels = skel[ax_pred_key]
            node_labels[node_labels == 3] = 1
            node_labels[node_labels == 4] = 1
            tree = spatial.cKDTree(skel['nodes'] * self.scaling)
            _, ixs = tree.query(np.array([obj.rep_coord for obj in objs]) * self.scaling, k=1, n_jobs=self.nb_cpus)
            obj_labels = node_labels[ixs]
            mask = np.zeros(len(objs), dtype=np.bool)
            for comp_label in compartments_of_interest:
                mask = mask | (obj_labels == comp_label)
            objs = objs[mask]
        if len(objs) > 0:
            vx_count = np.sum([obj.size for obj in objs])
        else:
            vx_count = 0
        obj_vol = vx_count * np.prod(self.scaling) / 1e9  # in um^3
        path_length = self.total_edge_length(compartments_of_interest) / 1e3  # in um
        
        if path_length == 0:
            return 0.0
        else:
            return obj_vol / path_length


# ------------------------------------------------------------------------------
# SO rendering code
def render_sampled_sos_cc(sos, ws=(256, 128), verbose=False, woglia=True,
                          render_first_only=0, add_cellobjects=True,
                          overwrite=False, cellobjects_only=False,
                          index_views=False, enable_locking=True):
    """
    Renders for each SV views at sampled locations (number is dependent on
    SV mesh size with scaling fact) from combined mesh of all SV.

    Args:
        sos: list of SegmentationObject
        ws: tuple
        verbose: bool
        woglia: bool
            without glia components
        render_first_only: int
        add_cellobjects: bool
        overwrite: bool
        cellobjects_only: bool
        index_views: bool
        enable_locking: bool
            enable system locking when writing views

    Returns:

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

    Args:
        so: SegmentationObject
            super voxel ID
        ws: tuple of int
            Rendering windows size
        add_cellobjects: bool
        verbose: bool

    Returns: np.array
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
        sso._map_cellobjects(save=False)
    views = render_sso_coords(sso, coords, ws=ws, add_cellobjects=add_cellobjects,
                              verbose=verbose)
    return views


def celltype_predictor(args) -> Iterable:
    """

    Args:
        args:

    Returns:

    """
    from ..handler.prediction import get_celltype_model_e3
    ssv_ids, nb_cpus, model_props = args
    use_onthefly_views = global_params.config.use_onthefly_views
    view_props = global_params.config['views']['view_properties']
    m = get_celltype_model_e3()
    missing_ssvs = []
    for ix in ssv_ids:
        ssv = SuperSegmentationObject(ix, working_dir=global_params.config.working_dir)
        ssv.nb_cpus = nb_cpus
        ssv._view_caching = True
        try:
            ssv.predict_celltype_multiview(m, pred_key_appendix="", onthefly_views=use_onthefly_views,
                                           overwrite=True, view_props=view_props, model_props=model_props)
        except RuntimeError as e:
            missing_ssvs.append(ssv.id)
            msg = 'ERROR during celltype prediction of SSV {}. {}'.format(ssv.id, repr(e))
            log_reps.error(msg)
    return missing_ssvs


def semsegaxoness_predictor(args) -> List[int]:
    """
    Predicts axoness and stores resulting labels at vertex dictionary.

    Args:
        args: (ssv_ids, view_props, nb_cpus, map_properties, pred_key, max_dist)

    Returns:
        IDs of missing/failed SSVs.
    """
    from ..handler.prediction import get_semseg_axon_model
    ssv_ids, view_props, nb_cpus, map_properties, pred_key, max_dist, bs = args
    m = get_semseg_axon_model()
    missing_ssvs = []
    for ix in ssv_ids:
        ssv = SuperSegmentationObject(ix, working_dir=global_params.config.working_dir)
        ssv.nb_cpus = nb_cpus
        ssv._view_caching = True
        try:
            ssh.semseg_of_sso_nocache(ssv, m, bs=bs, **view_props)
            semsegaxoness2skel(ssv, map_properties, pred_key, max_dist)
        except RuntimeError as e:
            missing_ssvs.append(ssv.id)
            msg = 'Error during sem. seg. prediction of SSV {}. {}'.format(ssv.id, repr(e))
            log_reps.error(msg)
        del ssv
    return missing_ssvs


def semsegaxoness2skel(sso: SuperSegmentationObject, map_properties: dict,
                       pred_key: str, max_dist: int):
    """
    Populate the following two skeleton keys:
        * "{}_avg{}".format(pred_key, max_dist)
        * "{}_avg{}_comp_maj".format(pred_key, max_dist)
    Args:
        sso: SuperSegmentationObject.
        map_properties: Properties used to map the vertex predictions to the skeleton nodes.
        pred_key: Used for retrieving vertex labels and to store the mapped node labels in the skeleton.
        max_dist: Distance used for majority vote in ``majorityvote_skeleton_property``.

    Notes:
        * Node predictions will be zero if no mesh vertices are available or no nodes exist.

    Returns:

    """
    if sso.skeleton is None:
        sso.load_skeleton()
    if sso.skeleton is None:
        log_reps.warning(f"Skeleton of {sso} hdoes not exist.")
        return
    if len(sso.skeleton["nodes"]) == 0 or len(sso.mesh[1]) == 0:
        log_reps.warning(f"Skeleton of {sso} has zero nodes or no mesh vertices.")
        sso.skeleton["{}_avg{}".format(pred_key, max_dist)] = np.zeros((len(sso.skeleton['nodes']), 1))
        sso.skeleton["{}_avg{}_comp_maj".format(pred_key, max_dist)] = np.zeros((len(sso.skeleton['nodes']), 1))
        sso.save_skeleton()
        return
    # vertex predictions
    node_preds = sso.semseg_for_coords(
        sso.skeleton['nodes'], semseg_key=pred_key,
        **map_properties)

    # perform average only on axon dendrite and soma predictions
    nodes_ax_den_so = np.array(node_preds, dtype=np.int32)
    # set en-passant and terminal boutons to axon class for averaging
    # bouton labels are stored in node_preds
    nodes_ax_den_so[nodes_ax_den_so == 3] = 1
    nodes_ax_den_so[nodes_ax_den_so == 4] = 1
    sso.skeleton[pred_key] = nodes_ax_den_so

    # average along skeleton, stored as: "{}_avg{}".format(pred_key, max_dist)
    ssh.majorityvote_skeleton_property(sso, prop_key=pred_key,
                                       max_dist=max_dist)
    # suffix '_avg{}' is added by `_average_node_axoness_views`
    nodes_ax_den_so = sso.skeleton["{}_avg{}".format(pred_key, max_dist)]
    # recover bouton predictions within axons and store smoothed result
    nodes_ax_den_so[(node_preds == 3) & (nodes_ax_den_so == 1)] = 3
    nodes_ax_den_so[(node_preds == 4) & (nodes_ax_den_so == 1)] = 4
    sso.skeleton["{}_avg{}".format(pred_key, max_dist)] = nodes_ax_den_so

    # will create a compartment majority voting after removing all soma nodes
    # the restul will be written to: ``ax_pred_key + "_comp_maj"``
    ssh.majority_vote_compartments(sso, "{}_avg{}".format(pred_key, max_dist))
    nodes_ax_den_so = sso.skeleton["{}_avg{}_comp_maj".format(pred_key, max_dist)]
    # recover bouton predictions within axons and store majority result
    nodes_ax_den_so[(node_preds == 3) & (nodes_ax_den_so == 1)] = 3
    nodes_ax_den_so[(node_preds == 4) & (nodes_ax_den_so == 1)] = 4
    sso.skeleton["{}_avg{}_comp_maj".format(pred_key, max_dist)] = nodes_ax_den_so
    sso.save_skeleton()


def semsegspiness_predictor(args) -> List[int]:
    """
    Predicts spiness and stores resulting labels at vertex dictionary.

    Args:
        args: (ssv_ids, view_props, nb_cpus, kwargs_semseg2mesh, kwargs_semsegforcoords)

    Returns:

    """
    from ..handler.prediction import get_semseg_spiness_model
    m = get_semseg_spiness_model()
    ssv_ids, view_props, nb_cpus, kwargs_semseg2mesh, kwargs_semsegforcoords = args
    missing_ssvs = []

    for ix in ssv_ids:
        ssv = SuperSegmentationObject(ix, working_dir=global_params.config.working_dir)
        ssv.nb_cpus = nb_cpus
        ssv._view_caching = True
        try:
            ssh.semseg_of_sso_nocache(ssv, m, **view_props, **kwargs_semseg2mesh)
            # map to skeleton
            ssv.load_skeleton()
            if ssv.skeleton is None or len(ssv.skeleton["nodes"]) == 0:
                log_reps.warning(f"Skeleton of SSV {ssv.id} has zero nodes.")
                node_preds = np.zeros((0, ), dtype=np.int32)
            else:
                # vertex predictions
                node_preds = ssv.semseg_for_coords(ssv.skeleton['nodes'],
                                                   kwargs_semseg2mesh['semseg_key'],
                                                   **kwargs_semsegforcoords)
            ssv.skeleton[kwargs_semseg2mesh['semseg_key']] = node_preds
            ssv.save_skeleton()
        except RuntimeError as e:
            missing_ssvs.append(ssv.id)
            msg = 'Error during sem. seg. prediction of SSV {}. {}'.format(ssv.id, repr(e))
            log_reps.error(msg)
    return missing_ssvs
