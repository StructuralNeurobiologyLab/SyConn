# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max-Planck-Institute of Neurobiology, Munich, Germany
# Authors: Philipp Schubert, Joergen Kornfeld
import copy
import re
from typing import Union, Tuple, List, Optional, Dict, Generator, Any, Iterator

import networkx as nx
from knossos_utils import knossosdataset
from scipy import spatial

from .rep_helper import subfold_from_ix, knossos_ml_from_svixs, SegmentationBase, get_unique_subfold_ixs
from .segmentation_helper import *
from ..handler.basics import get_filepaths_from_dir, safe_copy, \
    write_txt2kzip
from ..handler.basics import load_pkl2obj, write_obj2pkl, kd_factory
from ..handler.config import DynConfig
from ..proc import meshes
from ..proc.meshes import mesh_area_calc
from ..backend.storage import VoxelStorageDyn

MeshType = Union[Tuple[np.ndarray, np.ndarray, np.ndarray], List[np.ndarray],
                 Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]


class SegmentationObject(SegmentationBase):
    """
    Represents individual supervoxels. Used for cell shape ('sv'), cell organelles,
    e.g. mitochondria ('mi'), vesicle clouds ('vc') and synaptic junctions ('sj').
    
    Examples:
            Can be used to initialized single :class:`~SegmentationObject` object of
            a specific type, is also returned by :func:`~SegmentationDataset.get_segmentation_object`::
    
                from syconn.reps.segmentation import SegmentationObject, SegmentationDataset
                cell_sv = SegmentationObject(obj_id=.., obj_type='sv', working_dir='..')
                cell_sv.load_attr_dict()  # populates `cell_sv.attr_dict`
    
                cell_sd = SegmentationDataset(obj_type='sv', working_dir='..')
                cell_sv_from_sd = cell_sd.get_segmentation_object(obj_id=cell_sv.id)
                cell_sv_from_sd.load_attr_dict()
    
                keys1 = set(cell_sv.attr_dict.keys())
                keys2 = set(cell_sv_from_sd.attr_dict.keys())
                print(keys1 == keys2)
    
    Attributes:
        attr_dict: Attribute dictionary which serves as a general-purpose container. Accessed via
            the :class:`~syconn.backend.storage.AttributeDict` interface.
        enable_locking: If True, enables file locking.
    """

    def __init__(self, obj_id: int, obj_type: str = "sv",
                 version: Optional[str] = None, working_dir: Optional[str] = None,
                 rep_coord: Optional[np.ndarray] = None, size: Optional[int] = None,
                 scaling: Optional[np.ndarray] = None, create: bool = False,
                 voxel_caching: bool = True, mesh_caching: bool = False,
                 view_caching: bool = False, config: DynConfig = None,
                 n_folders_fs: int = None, enable_locking: bool = True,
                 skeleton_caching: bool = True, mesh: Optional[MeshType] = None):
        """
        Initializes a SegmentationObject with the given parameters. This object represents
        a supervoxel, which can be a cell shape, organelle, vesicle cloud, or synaptic junction.
        If `working_dir` is given and it contains a valid `config.yml` file, other optional
        parameters will be defined by the DynConfig object from the global_params module.
        
        Args:
            obj_id: Unique identifier for the supervoxel.
            obj_type: Type of the supervoxel, which can be 'mi' for Mitochondria, 'vc' for 
                      Vesicle clouds, 'sj' for Synaptic junction, 'syn_ssv' or 'syn' for
                      Synapses, or 'cs' for Contact site.
            version: Optional version identifier for the dataset. If 'tmp' is used, no data
                     will be saved to disk.
            working_dir: Optional path to the working directory containing the SegmentationDataset.
                         If provided, it must contain a valid `config.yml` file.
            rep_coord: Optional representative coordinate for the supervoxel.
            size: Optional number of voxels in the supervoxel.
            scaling: Optional array defining the voxel size in nanometers (XYZ).
            create: If True, creates the storage location for the supervoxel.
            voxel_caching: If True, enables caching for voxel data.
            mesh_caching: If True, enables caching for mesh data.
            view_caching: If True, enables caching for view data.
            skeleton_caching: If True, enables caching for skeleton data.
            config: Optional DynConfig object containing dataset-specific parameters.
            n_folders_fs: Optional number of folders within the SegmentationDataset's folder structure.
            enable_locking: If True, enables file locking to prevent concurrent write access.
            mesh: Optional mesh data provided as flat arrays (indices, vertices, normals).
        """
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
        self._mesh = mesh
        self._config = config
        self._views = None
        self._skeleton = None
        self._skeleton_caching = skeleton_caching
        if version == 'temp':
            version = 'tmp'

        self._setup_working_dir(working_dir, config, version, scaling)

        if version is None:
            try:
                self._version = self.config["versions"][self.type]
            except KeyError:
                raise Exception(f"Unclear version '{version}' during initialization of {self}.")
        else:
            self._version = version

        if create:
            os.makedirs(self.segobj_dir, exist_ok=True)

    #                                                       IMMEDIATE PARAMETERS
    def __hash__(self):
        """
        Generates a hash value for a SegmentationObject.
        
        Returns:
            An integer hash value.
        """
        return hash((self.id, self.type.__hash__()))

    def __eq__(self, other):
        """
        Checks equality between this SegmentationObject and another object.
        
        Args:
            other: The object to compare with.
        
        Returns:
            True if `other` is a SegmentationObject with the same id and type, False otherwise.
        """
        if not isinstance(other, self.__class__):
            return False
        return self.id == other.id and self.type == other.type

    def __ne__(self, other):
        """
        Checks inequality between this SegmentationObject and another object.
        
        Args:
            other: The object to compare with.
        
        Returns:
            True if `other` is not a SegmentationObject or has different id or type, False otherwise.
        """
        return not self.__eq__(other)

    def __repr__(self):
        """
        Returns a string representation of the SegmentationObject.
        
        Returns:
            A string that represents the SegmentationObject.
        """
        return (f'{type(self).__name__}(obj_id={self.id}, obj_type="{self.type}", '
                f'version="{self.version}", working_dir="{self.working_dir}")')

    def __reduce__(self):
        """
        Support pickling of class instances.
        """
        return self.__class__, (self._id, self._type, self._version, self._working_dir,
                                self._rep_coord, self._size, self._scaling, False,
                                self._voxel_caching, self._mesh_caching, self._view_caching,
                                self._config, self._n_folders_fs, self.enable_locking,
                                self._skeleton_caching, self._mesh)

    @property
    def type(self) -> str:
        """
        Retrieves the type of the supervoxel. This can be used to initialize a single `SegmentationObject` of
        a specific type or the corresponding dataset collection handled with the `SegmentationDataset` class.
        
        Keys used include:
        * 'mi': Mitochondria.
        * 'vc': Vesicle clouds.
        * 'sj': Synaptic junction.
        * 'syn_ssv': Synapses between two.
        * 'syn': Synapse fragment between two `SegmentationObject` objects.
        * 'cs': Contact site.
        
        Example usage:
        
        from syconn.reps.segmentation import SegmentationObject, SegmentationDataset
        cell_sv = SegmentationObject(obj_id=.., obj_type='sv', working_dir='..')
        cell_sv.load_attr_dict()  # populates `cell_sv.attr_dict`
        
        cell_sd = SegmentationDataset(obj_type='sv', working_dir='..')
        cell_sv_from_sd = cell_sd.get_segmentation_object(obj_id=cell_sv.id)
        cell_sv_from_sd.load_attr_dict()
        
        keys1 = set(cell_sv.attr_dict.keys())
        keys2 = set(cell_sv_from_sd.attr_dict.keys())
        print(keys1 == keys2)
        
        Returns:
            The type of the supervoxel as a string identifier.
        """
        return self._type

    @property
    def n_folders_fs(self) -> int:
        """
        Retrieves the number of folders used to store the data of SegmentationObjects. This 
        value defines the hierarchy of the folder structure organized by SegmentationDataset.
        
        Returns:
            The number of leaf folders used for storing supervoxel data.
        """
        if self._n_folders_fs is None:
            ps = glob.glob(
                "%s/%s*/" % (self.segds_dir, self.so_storage_path_base))
            if len(ps) == 0:
                raise Exception("No storage folder found at '{}' and no number of "
                                "subfolders specified (n_folders_fs))".format(self.segds_dir))

            bp = os.path.basename(ps[0].strip('/'))
            for p in ps:
                bp = os.path.basename(p.strip('/'))
                if bp == self.so_storage_path_base:
                    bp = os.path.basename(p.strip('/'))
                    break

            if bp == self.so_storage_path_base:
                self._n_folders_fs = 100000
            else:
                self._n_folders_fs = int(re.findall(r'[\d]+', bp)[-1])

        return self._n_folders_fs

    @property
    def id(self) -> int:
        """
        Retrieves the globally unique identifier of the SegmentationObject.
        
        Returns:
            Globally unique identifier of this object.
        """
        return self._id

    @property
    def version(self) -> str:
        """
        Retrieves the version of the SegmentationDataset this object belongs to.
        
        Returns:
            String identifier of the object's version.
        """
        return str(self._version)

    @property
    def voxel_caching(self) -> bool:
        """
        Specifies whether the voxel data should be cached after loading. If True, voxel data is retained in cache post load process.
        
        Returns:
            A boolean indicating whether voxel data is cached (True) or not (False) after loading.
        """
        return self._voxel_caching

    @property
    def mesh_caching(self) -> bool:
        """
        Indicates if mesh data is cached. 
        
        Returns:
            True if mesh data is cached, False otherwise.
        """
        return self._mesh_caching

    @property
    def skeleton_caching(self):
        """
        Indicates if skeleton data is cached.
        
        Returns:
            True if skeleton data is cached, False otherwise.
        """
        return self._skeleton_caching

    @property
    def view_caching(self):
        """
        Indicates if view data is cached. If True, view data is stored for future use. 
        
        Returns:
            True if view data is cached, False otherwise.
        """
        return self._view_caching

    @property
    def scaling(self):
        """
        Retrieves the voxel size in nanometers (XYZ). Default value is taken from the 
        `config.yml` file and accessible via `self.config`.
        
        Returns:
            An array representing the voxel size in nanometers.
        """
        if self._scaling is None:
            try:
                self._scaling = \
                    np.array(self.config['scaling'],
                             dtype=np.float32)
            except:
                self._scaling = np.array([1, 1, 1])

        return self._scaling

    @property
    def dataset(self) -> 'SegmentationDataset':
        """
        Factory method to create a `~syconn.reps.segmentation.SegmentationDataset` 
        instance to which this object belongs. 
        
        Returns: 
            `~syconn.reps.segmentation.SegmentationDataset`: An instance of SegmentationDataset.
        """
        return SegmentationDataset(self.type, self.version, self._working_dir)

    @property
    def config(self) -> DynConfig:
        """
        Retrieves the configuration object containing all dataset-specific parameters.
        
        Returns:
            The configuration object.
        """
        if self._config is None:
            self._config = global_params.config
        return self._config

    #                                                                      PATHS

    @property
    def working_dir(self) -> str:
        """
        Retrieves the working directory of the SegmentationObject.
        
        Returns:
            The working directory path as a string. This working directory 
            corresponds to the SegmentationObject's operating environment.
        """
        return self._working_dir

    @property
    def identifier(self) -> str:
        """
        Retrieves the identifier used to form the folder name of the 
        `~syconn.reps.segmentation.SegmentationDataset`.
        
        Returns:
            The identifier as a string.
        """
        return "%s_%s" % (self.type, self.version.lstrip("_"))

    @property
    def segds_dir(self) -> str:
        """
        Retrieves the path to the `~syconn.reps.segmentation.SegmentationDataset` directory.
        
        Returns:
            The path to the `~syconn.reps.segmentation.SegmentationDataset` directory.
        """
        return "%s/%s/" % (self.working_dir, self.identifier)

    @property
    def so_storage_path_base(self) -> str:
        """
        Retrieves the base folder name for the SegmentationObject storage. 
        
        Todo: 
            * refactor.
        
        Returns: 
            The base folder name as a string.
        """
        return "so_storage"

    @property
    def so_storage_path(self) -> str:
        """
        Retrieves the path to the entry folder of the directory tree containing all supervoxel data 
        of the corresponding `~syconn.reps.segmentation.SegmentationDataset`.
        
        Returns:
            The path to the supervoxel data storage.
        """
        if self._n_folders_fs is None and os.path.exists("%s/%s/" % (
                self.segds_dir, self.so_storage_path_base)):
            return "%s/%s/" % (self.segds_dir, self.so_storage_path_base)
        elif self._n_folders_fs == 100000 and os.path.exists("%s/%s/" % (
                self.segds_dir, self.so_storage_path_base)):
            return "%s/%s/" % (self.segds_dir, self.so_storage_path_base)
        else:
            return "%s/%s_%d/" % (self.segds_dir, self.so_storage_path_base,
                                  self.n_folders_fs)

    @property
    def segobj_dir(self) -> str:
        """
        Retrieves the path to the folder where the data of this supervoxel is stored.
        
        Returns:
            str: The path to the supervoxel data folder.
        """
        base_path = f"{self.so_storage_path}/" \
                    f"{subfold_from_ix(self.id, self.n_folders_fs)}/"
        if os.path.exists(f"{base_path}/voxel.pkl"):
            return base_path
        else:
            # use old folder scheme with leading 0s, e.g. '09'
            return "%s/%s/" % (self.so_storage_path, subfold_from_ix(
                self.id, self.n_folders_fs, old_version=True))

    @property
    def mesh_path(self) -> str:
        """
        Retrieves the path to the mesh storage. 
        
        Returns: 
            The path to where the mesh data is stored.
        """
        return self.segobj_dir + "mesh.pkl"

    @property
    def skeleton_path(self) -> str:
        """
        Retrieves the path to the skeleton storage.
        
        Returns:
            str: The path to the skeleton storage.
        """
        return self.segobj_dir + "skeletons.pkl"

    @property
    def attr_dict_path(self) -> str:
        """
        Retrieves the path to the attribute storage.
        
        Returns:
            str: The path to the attribute storage.
        """
        return self.segobj_dir + "attr_dict.pkl"

    def view_path(self, woglia=True, index_views=False, view_key=None) -> str:
        """
        Retrieves the path to the view storage.
        
        Args:
            woglia: If True, looks for views without glia, i.e., after astrocyte separation.
            index_views: If True, refers to index views.
            view_key: Identifier of the requested views.
        
        Returns:
            Path to the view storage.
        """
        if view_key is not None and not (woglia and not index_views):
            raise ValueError('view_path with custom view key is only allowed for default settings.')
        # TODO: change bool index_views and bool woglia to respective view_key identifier
        if view_key is not None:
            return self.segobj_dir + 'views_{}.pkl'.format(view_key)
        if index_views:
            return self.segobj_dir + "views_index.pkl"
        elif woglia:
            return self.segobj_dir + "views_woglia.pkl"

        return self.segobj_dir + "views.pkl"

    @property
    def locations_path(self) -> str:
        """
        Retrieves and returns the path to the rendering location storage.
        """
        return self.segobj_dir + "locations.pkl"

    @property
    def voxel_path(self) -> str:
        """
        Retrieves the path to the voxel storage. See :class:`~syconn.backend.storage.VoxelStorageDyn` for more details.
        
        Returns:
            The path to the voxel storage.
        """
        # file type is inferred by either VoxelStorageLazyLoading or VoxelStorageDyn
        return self.segobj_dir + "/voxel"

    #                                                                 PROPERTIES
    @property
    def cs_partner(self) -> Optional[List[int]]:
        """
        Retrieves the IDs of the two supervoxels that are part of the contact site 
        specific attribute. 
        
        Returns:
            A list of two IDs if the object type is 'cs', otherwise None.
        """
        # TODO: use `cs_id_to_partner_ids_vec`  (single source of truth)
        if self.type in ['cs', 'syn']:
            partner = [self.id >> 32]
            partner.append(self.id - (partner[0] << 32))
            return partner
        else:
            return None

    @property
    def size(self) -> int:
        """
        Retrieves the number of voxels in the SegmentationObject.
        
        Returns:
            The number of voxels, represented as an integer, in the SegmentationObject.
        """
        if self._size is None and 'size' in self.attr_dict:
            self._size = self.attr_dict['size']
        elif self._size is None and self.attr_dict_exists:
            self._size = self.lookup_in_attribute_dict("size")
        if self._size is None:
            self.calculate_size()

        return self._size

    @property
    def shape(self) -> np.ndarray:
        """
        Retrieves the XYZ extent of this SSV object in voxels.
        
        Returns:
            An array representing the shape/extent of this SSV object in voxels (XYZ).
        """
        return self.bounding_box[1] - self.bounding_box[0]

    @property
    def bounding_box(self) -> np.ndarray:
        """
        Retrieves the bounding box of the SegmentationObject.
        
        Returns:
            An array representing the bounding box (XYZ).
        """
        if self._bounding_box is None and 'bounding_box' in self.attr_dict:
            self._bounding_box = self.attr_dict['bounding_box']
        elif self._bounding_box is None and self.attr_dict_exists:
            self._bounding_box = self.lookup_in_attribute_dict('bounding_box')
        if self._bounding_box is None:
            self.calculate_bounding_box()

        return self._bounding_box

    @property
    def rep_coord(self) -> np.ndarray:
        """
        Retrieves the representative coordinate of this SegmentationObject, 
        which will be the 'rep_coord' of the first supervoxel in 'svs'.
        
        Returns:
            A 1D array representing the representative coordinate (XYZ).
        """
        if self._rep_coord is None and 'rep_coord' in self.attr_dict:
            self._rep_coord = self.attr_dict['rep_coord']
        elif self._rep_coord is None and self.attr_dict_exists:
            self._rep_coord = self.lookup_in_attribute_dict("rep_coord")
        if self._rep_coord is None:
            self.calculate_rep_coord()

        return self._rep_coord

    @property
    def attr_dict_exists(self) -> bool:
        """
        Checks if an attribute dictionary file exists at :py:attr:`~attr_dict_path` for the SegmentationObject.
        
        Returns:
            True if the attribute dictionary file exists, False otherwise.
        """
        if self.version == 'tmp':
            return False
        if not os.path.isfile(self.attr_dict_path):
            return False
        glob_attr_dc = AttributeDict(self.attr_dict_path,
                                     disable_locking=True)  # look-up only, PS 12Dec2018
        return self.id in glob_attr_dc

    @property
    def voxels_exist(self) -> bool:
        """
        Checks if voxel data exists for the SegmentationObject.
        
        Returns:
            True if voxel data exists, False otherwise.
        """
        if self.version == 'tmp':
            return False
        if self.type in ['syn', 'syn_ssv']:
            voxel_dc = VoxelStorageLazyLoading(self.voxel_path)
            exists = self.id in voxel_dc
            voxel_dc.close()
        else:
            voxel_dc = VoxelStorageDyn(self.voxel_path, read_only=True,
                                       disable_locking=True)
            self.id in voxel_dc
        return exists

    @property
    def voxels(self) -> np.ndarray:
        """
        Retrieves the voxels associated with the SSV object.
        
        Returns:
            A 3D binary array indicating voxel locations.
        """
        if self._voxels is None:
            return self.load_voxels()
        else:
            return self._voxels

    @property
    def voxel_list(self) -> np.ndarray:
        """
        Retrieves the voxels associated with the SSV object.
        
        Returns:
            A 2D array with sparse voxel coordinates.
        """
        if self._voxel_list is None:
            voxel_list = load_voxel_list(self)
            if self.voxel_caching:
                self._voxel_list = voxel_list
            return voxel_list
        else:
            return self._voxel_list

    @property
    def mesh_exists(self) -> bool:
        """
        Checks if mesh data exists for the SegmentationObject.
        
        Returns:
            True if mesh data exists, False otherwise.
        """
        if self.version == 'tmp':
            return False
        mesh_dc = MeshStorage(self.mesh_path, disable_locking=True)
        return self.id in mesh_dc

    @property
    def skeleton_exists(self) -> bool:
        """
        Checks if skeleton data exists for the SegmentationObject.
        
        Returns:
            True if skeleton exists, False otherwise.
        """
        if self.version == 'tmp':
            return False
        skeleton_dc = SkeletonStorage(self.skeleton_path, disable_locking=True)
        return self.id in skeleton_dc

    @property
    def mesh(self) -> MeshType:
        """
        Retrieves the mesh data of this object.
        
        Returns:
            A tuple of three flat arrays: indices, vertices, normals.
        """
        if self._mesh is None:
            if self.mesh_caching:
                self._mesh = load_mesh(self)
                return self._mesh
            else:
                return load_mesh(self)
        else:
            return self._mesh

    @property
    def skeleton(self) -> dict:
        """
        Retrieves the skeleton representation of this supervoxel.
        
        Returns:
            A dictionary containing at least three numpy arrays: "nodes", 
            estimated node "diameters", and "edges".
        """
        if self._skeleton is None:
            if self.skeleton_caching:
                self._skeleton = load_skeleton(self)
                return self._skeleton
            else:
                return load_skeleton(self)
        else:
            return self._skeleton

    @property
    def mesh_bb(self) -> np.ndarray:
        """
        Retrieves the bounding box of the object meshes in nanometers. This is approximately 
        the same as the scaled 'bounding_box'. Returns an array representing the bounding box 
        of the meshes.
        """
        if self._mesh_bb is None and 'mesh_bb' in self.attr_dict:
            self._mesh_bb = self.attr_dict['mesh_bb']
        elif self._mesh_bb is None:
            if len(self.mesh[1]) == 0 or len(self.mesh[0]) == 0:
                self._mesh_bb = self.bounding_box * self.scaling
            else:
                verts = self.mesh[1].reshape(-1, 3)
                self._mesh_bb = np.array([np.min(verts, axis=0), np.max(verts, axis=0)], dtype=np.float32)
        return self._mesh_bb

    @property
    def mesh_size(self) -> float:
        """
        Retrieves the length of the bounding box diagonal (BBD) of the object meshes.
        
        Returns:
            The diagonal (BBD) length of the mesh bounding box in nanometers.
        """
        return np.linalg.norm(self.mesh_bb[1] - self.mesh_bb[0], ord=2)

    @property
    def mesh_area(self) -> float:
        """
        Retrieves the mesh surface area of the supervoxel.
        
        Returns:
            float: Mesh surface area in um^2.
        """
        # TODO: decide if caching should be possible
        mesh_area = self.lookup_in_attribute_dict('mesh_area')
        if mesh_area is None:
            mesh_area = mesh_area_calc(self.mesh)
            if np.isnan(mesh_area) or np.isinf(mesh_area):
                raise ValueError('Invalid mesh area.')
        return mesh_area

    @property
    def sample_locations_exist(self) -> bool:
        """
        Checks if rendering locations have been stored.
        
        Returns:
            bool: True if rendering locations have been stored at 
            :py:attr:`~locations_path`.
        """
        if self.version == 'tmp':
            return False
        location_dc = CompressedStorage(self.locations_path,
                                        disable_locking=True)
        return self.id in location_dc

    def views_exist(self, woglia: bool, index_views: bool = False,
                    view_key: Optional[str] = None) -> bool:
        """
        Determines if rendering locations have been stored at the specified view path.
        
        Args:
            woglia: If True, checks for views without glia, i.e. after astrocyte separation.
            index_views: If True, checks for index views.
            view_key: Identifier of the requested views.
        
        Returns:
            bool: True if rendering locations have been stored.
        """
        if self.version == 'tmp':
            return False
        view_dc = CompressedStorage(self.view_path(woglia=woglia, index_views=index_views, view_key=view_key),
                                    disable_locking=True)
        return self.id in view_dc

    def views(self, woglia: bool, index_views: bool = False,
              view_key: Optional[str] = None) -> Union[np.ndarray, int]:
        """
        Retrieves the views of this supervoxel. Valid only for cell fragments, i.e. 
        :py:attr:`~type` must be `sv`.
        
        Args:
            woglia: If True, retrieves views without glia, i.e., after astrocyte separation.
            index_views: If True, retrieves index views.
            view_key: Identifier of the requested views.
        
        Returns:
            Union[np.ndarray, int]: The requested view array or `-1` if it does not exist.
        """
        assert self.type == "sv"
        if self._views is None:
            if self.views_exist(woglia):
                if self.view_caching:
                    self._views = self.load_views(woglia=woglia, index_views=index_views,
                                                  view_key=view_key)
                    return self._views
                else:
                    return self.load_views(woglia=woglia, index_views=index_views,
                                           view_key=view_key)
            else:
                return -1
        else:
            return self._views

    def sample_locations(self, force=False, save=True, ds_factor=None):
        """
        Retrieves the rendering locations of this supervoxel. This is only valid for cell 
        fragments, i.e., `type` must be `sv`.
        
        Args:
            force: If True, overwrites existing data.
            save: If True, saves the result at `locations_path`. Utilizes 
            `CompressedStorage`.
            ds_factor: Down sampling factor used to generate the rendering locations.
        
        Returns:
            np.ndarray: Array of rendering locations (XYZ) with shape (N, 3) in nanometers.
        """
        assert self.type == "sv"
        if self.sample_locations_exist and not force:
            return CompressedStorage(self.locations_path, disable_locking=True)[self.id]
        else:
            verts = self.mesh[1].reshape(-1, 3)
            if len(verts) == 0:  # only return scaled rep. coord as [1, 3] array
                return np.array([self.rep_coord, ], dtype=np.float32) * self.scaling
            if ds_factor is None:
                ds_factor = 2000
            if self.config.use_new_renderings_locs:
                coords = generate_rendering_locs(verts, ds_factor).astype(np.float32)
            else:
                coords = surface_samples(verts, [ds_factor] * 3, r=ds_factor / 2).astype(np.float32)
            if save:
                loc_dc = CompressedStorage(self.locations_path, read_only=False,
                                           disable_locking=not self.enable_locking)
                loc_dc[self.id] = coords.astype(np.float32)
                loc_dc.push()
            return coords.astype(np.float32)

    def load_voxels(self, voxel_dc: Optional[Union[VoxelStorageDyn, VoxelStorage]] = None) -> np.ndarray:
        """
        Loads the voxels associated with this supervoxel.
        
        Args:
            voxel_dc: Pre-loaded dictionary which contains the voxel data of this object.
        
        Returns:
            np.ndarray: 3D array of all voxels which belong to this supervoxel.
        """
        # syn_ssv do not have a segmentation KD; voxels are cached in their VoxelStorage
        if self.type in ['syn', 'syn_ssv']:
            vxs_list = self.voxel_list - self.bounding_box[0]
            voxels = np.zeros(self.bounding_box[1] - self.bounding_box[0] + 1, dtype=np.bool)
            voxels[vxs_list[..., 0], vxs_list[..., 1], vxs_list[..., 2]] = True
        else:
            if voxel_dc is None:
                voxel_dc = VoxelStorageDyn(self.voxel_path, read_only=True, disable_locking=True)
            voxels = voxel_dc.get_voxel_data_cubed(self.id)[0]
        if self.voxel_caching:
            self._voxels = voxels
        return voxels

    def load_voxels_downsampled(self, downsampling=(2, 2, 1)):
        """
        Loads downsampled voxels.
        
        Args:
            downsampling: Tuple specifying the downsampling factor for each axis.
        
        Returns:
            np.ndarray: Downsampled voxel data.
        """
        return load_voxels_downsampled(self, ds=downsampling)

    def load_voxel_list(self):
        """
        Loader method of :py:attr:`~voxel_list`. Loads a sparse, 2-dimensional array 
        of voxel coordinates.
        
        Returns:
            np.ndarray: Sparse, 2-dimensional array of voxel coordinates.
        """
        return load_voxel_list(self)

    def load_voxel_list_downsampled(self, downsampling=(2, 2, 1)):
        """
        Loads a downsampled list of voxel coordinates.
        
        Args:
            downsampling: Tuple specifying the downsampling factor for each axis.
        
        Returns:
            np.ndarray: Downsampled list of voxel coordinates.
        """
        return load_voxel_list_downsampled(self, downsampling=downsampling)

    def load_voxel_list_downsampled_adapt(self, downsampling=(2, 2, 1)):
        """
        Loads a downsampled list of voxel coordinates with adaptive scaling.
        
        Args:
            downsampling: Tuple specifying the downsampling factor for each axis.
        
        Returns:
            np.ndarray: Downsampled list of voxel coordinates with adaptive scaling.
        """
        return load_voxel_list_downsampled_adapt(self, downsampling=downsampling)

    def load_skeleton(self, recompute: bool = False) -> dict:
        """
        Loads the skeleton representation of this supervoxel.
        
        Args:
            recompute: Recompute the skeleton. Currently not implemented.
        
        Returns:
            dict: Dictionary containing flat arrays of indices, vertices, diameters, 
            and attributes.
        """
        return load_skeleton(self, recompute=recompute)

    def save_skeleton(self, overwrite: bool = False):
        """
        Saves the skeleton data of this supervoxel.
        
        Args:
            overwrite: Overwrite existing skeleton entry.
        
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Flat arrays of indices, vertices, and normals.
        """
        return save_skeleton(self, overwrite=overwrite)

    def glia_pred(self, thresh: float, pred_key_appendix: str = "") -> int:
        """
        Predicts if the supervoxel is a glia or neuron. Only valid if :py:attr:`type` is `sv`.
        
        Args:
            thresh: The classification threshold.
            pred_key_appendix: An identifier for specific glia predictions. Only used
                during development.
        
        Returns:
            int: The glia prediction for this supervoxel (0: neuron, 1: glia).
        """
        assert self.type == "sv"
        if self.config.use_point_models:
            return int(self.glia_proba(pred_key_appendix) >= thresh)
        return glia_pred_so(self, thresh, pred_key_appendix)

    def glia_proba(self, pred_key_appendix: str = "") -> float:
        """
        Retrieves the glia probability of this supervoxel. This is only valid if :py:attr:`type` is `sv`.
        
        Args:
            pred_key_appendix: Identifier for specific glia predictions. This is only used
            during development.
        
        Returns:
            float: The glia prediction of this supervoxel (0: neuron, 1: glia).
        """
        assert self.type == "sv"
        return glia_proba_so(self, pred_key_appendix)

    def axoness_preds(self, pred_key_appendix: str = "") -> np.ndarray:
        """
        Predicts the axoness of this supervoxel at every sample location based on `img2scalar` CMN.
        
        Args:
            pred_key_appendix: Identifier for specific axon predictions. Used only
                during development.
        
        Returns:
            np.ndarray: The axon prediction of this supervoxel at each sample location
                (0: dendrite, 1: axon, 2: soma).
        """
        pred = np.argmax(self.axoness_probas(pred_key_appendix), axis=1)
        return pred

    def axoness_probas(self, pred_key_appendix: str = "") -> np.ndarray:
        """
        Retrieves the axon probabilities (0: dendrite, 1: axon, 2: soma) based on `img2scalar` 
        CMN of this supervoxel at every sample location. The probabilities underlie the attribute 
        :py:attr:`axoness_preds` and are only valid if :py:attr:`type` is `sv`.
        
        Args:
            pred_key_appendix: Identifier for specific axon predictions. Only used during 
            development.
        
        Returns:
            np.ndarray: The axon probabilities of this supervoxel at every 
            :py:attr:`~sample_locations`.
        """
        assert self.type == "sv"
        pred_key = "axoness_probas" + pred_key_appendix
        if pred_key not in self.attr_dict:
            self.load_attr_dict()
        if pred_key not in self.attr_dict:
            msg = (f"WARNING: Requested axoness {pred_key} for SV {self.id} is not available. Existing "
                   f"keys: {self.attr_dict.keys()}")
            raise ValueError(msg)
        return self.attr_dict[pred_key]

    #                                                                  FUNCTIONS
    def total_edge_length(self) -> Union[np.ndarray, float]:
        """
        Calculates the total edge length of the supervoxel :py:attr:`~skeleton` in nanometers.
        
        Returns:
            float: Sum of all edge lengths (L2 norm) in :py:attr:`~skeleton`.
        """
        if self.skeleton is None:
            self.load_skeleton()
        nodes = self.skeleton['nodes'].astype(np.float32)
        edges = self.skeleton['edges']
        return np.sum([np.linalg.norm(self.scaling * (nodes[e[0]] - nodes[e[1]])) for e in edges])

    def mesh_from_scratch(self, ds: Optional[Tuple[int, int, int]] = None,
                          **kwargs: dict) -> List[np.ndarray]:
        """
        Calculates the mesh based on :func:`~syconn.proc.meshes.get_object_mesh`.
        
        Args:
            ds: Downsampling of the object's voxel data.
            **kwargs: Additional keyword arguments passed to :func:`~syconn.proc.meshes.triangulation`.
        
        Returns:
            List[np.ndarray]: Mesh data including indices, vertices, and normals.
        """
        _supported_types = ['syn_ssv', 'syn', 'cs_ssv', 'cs']
        if self.type not in _supported_types:
            raise ValueError(f'"mesh_from_scratch" does not support type "{self.type}". Supported types: '
                             f'{_supported_types}')
        if ds is None:
            ds = self.config['meshes']['downsampling'][self.type]
        return meshes.get_object_mesh(self, ds, mesher_kwargs=kwargs)

    def _save_mesh(self, ind: np.ndarray, vert: np.ndarray,
                   normals: np.ndarray):
        """
        Saves the given mesh data at :py:attr:`~mesh_path`. Uses the 
        :class:`~syconn.backend.storage.MeshStorage` interface.
        
        Args:
            ind: Flat array of mesh indices.
            vert: Flat array of mesh vertices.
            normals: Flat array of mesh normals.
        """
        mesh_dc = MeshStorage(self.mesh_path, read_only=False,
                              disable_locking=not self.enable_locking)
        mesh_dc[self.id] = [ind, vert, normals]
        mesh_dc.push()

    def mesh2kzip(self, dest_path: str, ext_color: Optional[Union[
        Tuple[int, int, int, int], List, np.ndarray]] = None,
                  ply_name: str = ""):
        """
        Writes the :py:attr:`~mesh` to a k.zip file.
        
        Args:
            dest_path: Path to the k.zip file which contains the :py:attr:`~mesh`.
            ext_color: Optional RGBA color tuple. If set to 0, no color will be written out, use to adapt color in Knossos.
            ply_name: Name of the ply file within the k.zip archive, must not end with `.ply`.
        """
        mesh = self.mesh
        if self.type == "sv":
            color = (130, 130, 130, 160)
        elif self.type == "cs":
            color = (100, 200, 30, 255)
        elif self.type == "syn":
            color = (150, 50, 200, 255)
        elif self.type == "syn_ssv":
            color = (240, 50, 50, 255)
        elif self.type == "cs_ssv":
            color = (100, 200, 30, 255)
        elif self.type == "sj":
            color = (int(0.849 * 255), int(0.138 * 255), int(0.133 * 255), 255)
        elif self.type == "vc":
            color = (int(0.175 * 255), int(0.585 * 255), int(0.301 * 255), 255)
        elif self.type == "mi":
            color = (0, 153, 255, 255)
        else:
            raise TypeError("Color for bbject type '{}' does not exist."
                            "".format(self.type))
        color = np.array(color, dtype=np.uint8)
        if ext_color is not None:
            if ext_color == 0:
                color = None
            else:
                color = ext_color
        if ply_name == "":
            ply_name = str(self.id)
        meshes.write_mesh2kzip(dest_path, mesh[0], mesh[1], mesh[2], color,
                               ply_fname=ply_name + ".ply")

    def mergelist2kzip(self, dest_path: str):
        """
        Writes the supervoxel agglomeration to a KNOSSOS compatible format.
        
        Args:
            dest_path: Path to the k.zip file.
        """
        self.load_attr_dict()
        kml = knossos_ml_from_svixs([self.id], coords=[self.rep_coord])
        write_txt2kzip(dest_path, kml, "mergelist.txt")

    def load_views(self, woglia: bool = True, raw_only: bool = False,
                   ignore_missing: bool = False, index_views: bool = False,
                   view_key: Optional[str] = None):
        """
        Loads views with specified properties.
        
        Args:
            woglia: If True, looks for views without glia, i.e. after astrocyte separation.
            raw_only: If True, ignores cell organelles projections.
            ignore_missing: If True, will not throw ValueError if views do not exist.
            index_views: If True, refers to index views.
            view_key: Identifier of the requested views.
        
        Returns:
            np.ndarray: Views with the requested properties.
        """
        view_p = self.view_path(woglia=woglia, index_views=index_views,
                                view_key=view_key)
        view_dc = CompressedStorage(view_p, disable_locking=not self.enable_locking)
        try:
            views = view_dc[self.id]
        except KeyError as e:
            if ignore_missing:
                log_reps.warning("Views of SV {} were missing. Skipping.".format(self.id))
                views = np.zeros((0, 4, 2, 128, 256), dtype=np.uint8)
            else:
                raise KeyError(e)
        if raw_only:
            views = views[:, :1]
        return views

    def save_views(self, views: np.ndarray, woglia: bool = True,
                   cellobjects_only: bool = False, index_views: bool = False,
                   view_key: Optional[str] = None,
                   enable_locking: Optional[bool] = None):
        """
        Saves views according to its properties. If a particular `view_key` is provided, it must correspond to a specific type of view, like spine predictions, and all other kwargs should be set to their default values to avoid errors. 
        
        Args:
            views: Array of views to be saved.
            woglia: If True, saves views that do not contain glia, i.e., after astrocyte separation.
            cellobjects_only: If True, only cell organelle views are saved (deprecated).
            index_views: If True, saves index views.
            view_key: Identifier for the specific views to be saved.
            enable_locking: If True, activates file locking during the save operation.
        
        Todo:
            * Remove `cellobjects_only`.
        """
        if not (woglia and not cellobjects_only and not index_views) and view_key is not None:
            raise ValueError('If views are saved to custom key, all other settings have to be defaults!')
        if enable_locking is None:
            enable_locking = self.enable_locking
        view_dc = CompressedStorage(self.view_path(woglia=woglia, index_views=index_views, view_key=view_key),
                                    read_only=False, disable_locking=not enable_locking)
        if cellobjects_only:
            assert self.id in view_dc, "SV must already contain raw views " \
                                       "if adding views for cellobjects only."
            view_dc[self.id] = np.concatenate([view_dc[self.id][:, :1], views],
                                              axis=1)
        else:
            view_dc[self.id] = views
        view_dc.push()

    def load_attr_dict(self) -> int:
        """
        Loads the :py:attr:`~attr_dict`.
        
        Returns:
            int: 0 if successful, -1 if attribute dictionary storage does not exist.
        """
        try:
            glob_attr_dc = AttributeDict(self.attr_dict_path,
                                         disable_locking=True)  # disable locking, PS 07June2019
            self.attr_dict = glob_attr_dc[self.id]
        except (IOError, EOFError) as e:
            log_reps.critical("Could not load SSO attributes at {} due to "
                              "{}.".format(self.attr_dict_path, e))
            return -1

    def save_attr_dict(self):
        """
        Saves the attribute dictionary to the attribute dictionary path. Any existing dictionary will be updated.
        """
        glob_attr_dc = AttributeDict(self.attr_dict_path, read_only=False,
                                     disable_locking=not self.enable_locking)
        if self.id in glob_attr_dc:
            orig_dc = glob_attr_dc[self.id]
            orig_dc.update(self.attr_dict)
        else:
            orig_dc = self.attr_dict
        glob_attr_dc[self.id] = orig_dc
        glob_attr_dc.push()

    def save_attributes(self, attr_keys: List[str], attr_values: List[Any]):
        """
        Writes attributes to attribute storage. Ignores :py:attr:`~attr_dict`.
        Values have to be serializable and will be written via the
        :class:`~syconn.backend.storage.AttributeDict` interface.
        
        Args:
            attr_keys: List of attribute keys which will be written to
                :py:attr:`~attr_dict_path`.
            attr_values: List of attribute values which will be written to
                :py:attr:`~attr_dict_path`.
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
        glob_attr_dc.push()

    def load_attributes(self, attr_keys: List[str]) -> List[Any]:
        """
        Reads attributes from attribute storage, ignoring self.attr_dict and always pulling them from storage. It does not throw KeyError but returns None for missing keys.
        
        Args:
            attr_keys: List of attribute keys to be loaded from attribute storage.
        
        Returns:
            List[Any]: Attribute values corresponding to `attr_keys`, returns None for missing keys.
        """
        glob_attr_dc = AttributeDict(self.attr_dict_path, read_only=True,
                                     disable_locking=not self.enable_locking)
        return [glob_attr_dc[self.id][attr_k] if attr_k in glob_attr_dc[self.id]
                else None for attr_k in attr_keys]

    def attr_exists(self, attr_key: str) -> bool:
        """
        Checks if `attr_key` exists in either :py:attr:`~attr_dict` or at
        :py:attr:`~attr_dict_path`.
        
        Args:
            attr_key: The attribute key to check.
        
        Returns:
            bool: True if the attribute exists, False otherwise.
        """
        if len(self.attr_dict) == 0:
            self.load_attr_dict()
        try:
            _ = self.attr_dict[attr_key]
        except (KeyError, EOFError):
            return False
        return True

    def lookup_in_attribute_dict(self, attr_key: str) -> Any:
        """
        Looks up a value in the attribute dictionary.
        
        Args:
            attr_key: Attribute key to look for.
        
        Returns:
            Any: The value of `attr_key` in :py:attr:`~attr_dict` or None if it does
            not exist. If key does not exist in :py:attr:`~attr_dict`, tries to
            load from :py:attr:`~attr_dict_path`.
        """
        if len(self.attr_dict) == 0:
            self.load_attr_dict()
        if self.attr_exists(attr_key):
            return self.attr_dict[attr_key]
        else:
            return None

    def calculate_rep_coord(self, voxel_dc: Optional[Dict[int, np.ndarray]] = None):
        """
        Calculates or loads the supervoxel representative coordinate.
        
        Args:
            voxel_dc: Pre-loaded dictionary which contains the voxel data of
                this object.
        """
        if voxel_dc is None:
            if self.type in ['syn', 'syn_ssv']:
                voxel_dc = VoxelStorageLazyLoading(self.voxel_path)
            else:
                voxel_dc = VoxelStorageDyn(self.voxel_path, read_only=True,
                                           disable_locking=True)

        if self.id not in voxel_dc:
            self._bounding_box = np.array([[-1, -1, -1], [-1, -1, -1]])
            log_reps.warning("No voxels found in VoxelDict!")
            return

        if isinstance(voxel_dc, VoxelStorageDyn):
            self._rep_coord = voxel_dc.object_repcoord(self.id)
            return
        elif isinstance(voxel_dc, VoxelStorageLazyLoading):
            self._rep_coord = voxel_dc[self.id][len(voxel_dc[self.id]) // 2]  # any rep coord
            return
        else:
            raise ValueError(f'Invalid voxel storage class: {type(voxel_dc)}')

    def calculate_bounding_box(self, voxel_dc: Optional[Dict[int, np.ndarray]] = None):
        """
        Calculates the supervoxel :py:attr:`~bounding_box`.
        
        Args:
            voxel_dc: Pre-loaded dictionary which contains the voxel data of this object.
        """
        if voxel_dc is None:
            if self.type in ['syn', 'syn_ssv']:
                voxel_dc = VoxelStorageLazyLoading(self.voxel_path)
            else:
                voxel_dc = VoxelStorageDyn(self.voxel_path, read_only=True,
                                           disable_locking=True)
        if not isinstance(voxel_dc, VoxelStorageDyn):
            _ = self.load_voxels(voxel_dc=voxel_dc)
        else:
            bbs = voxel_dc.get_boundingdata(self.id)
            bb = np.array([bbs[:, 0].min(axis=0), bbs[:, 1].max(axis=0)])
            self._bounding_box = bb

    def calculate_size(self, voxel_dc: Optional[Union[VoxelStorageDyn, VoxelStorage]] = None):
        """
        Calculates the size of the supervoxel object :py:attr:`~size`.
        
        Args:
            voxel_dc: Pre-loaded dictionary which contains the voxel data of this object.
        """
        if voxel_dc is None:
            if self.type in ['syn', 'syn_ssv']:
                voxel_dc = VoxelStorageLazyLoading(self.voxel_path)
            else:
                voxel_dc = VoxelStorageDyn(self.voxel_path, read_only=True,
                                           disable_locking=True)
        if not isinstance(voxel_dc, VoxelStorageDyn):
            _ = self.load_voxels(voxel_dc=voxel_dc)
        else:
            size = voxel_dc.object_size(self.id)
            self._size = size

    def save_kzip(self, path: str,
                  kd: Optional[knossosdataset.KnossosDataset] = None,
                  write_id: Optional[int] = None):
        """
        Writes the supervoxel segmentation to a k.zip file.
        
        Args:
            path (str): The file path where the k.zip file will be saved.
            kd (Optional[knossosdataset.KnossosDataset], optional): The KnossosDataset
                object. If None, it will be loaded from the configuration. Defaults to None.
            write_id (Optional[int], optional): The supervoxel ID. If None, the ID of the
                current supervoxel object will be used. Defaults to None.
        
        Todo:
            * Broken, segmentation not rendered in Knossos.
        """
        if write_id is None:
            write_id = self.id

        if kd is None:
            try:
                kd = kd_factory(self.config.kd_seg_path)
            except:
                raise ValueError("KnossosDataset could not be loaded")

        kd.save_to_kzip(offset=self.bounding_box[0], data=self.voxels.astype(np.uint64).swapaxes(0, 2) * write_id,
                        kzip_path=path, data_mag=1, mags=[1])

    def clear_cache(self):
        """
        Clears the cached data for the following:
            * voxels
            * voxel lists
            * views
            * skeletons
        Note: It doesn't clear cached data for meshes as indicated in the previous version.
        """
        self._voxels = None
        self._voxel_list = None
        self._mesh = None
        self._views = None
        self._skeleton = None

    # SKELETON
    @property
    def skeleton_dict_path(self) -> str:
        """
        Retrieves the path to the skeleton storage.
        
        Returns:
            str: The file path to the skeleton storage.
        """
        return self.segobj_dir + "/skeletons.pkl"

    def copy2dir(self, dest_dir, safe=True):
        """
        Copies all files from the supervoxel object's directory to a specified destination directory.
        
        Args:
            dest_dir (str): Destination directory where all files contained in
                py:attr:`~segobj_dir` will be copied to.
            safe (bool): If True, existing files at the destination will not be 
                overwritten. Defaults to True.
        
        Examples:
            To copy the content of this supervoxel object (`sv_orig`) to the 
            destination of another (e.g., not yet existing) supervoxel object 
            (`sv_target`), use: `sv_orig.copy2dir(sv_target.segobj_dir)`.
        """
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
            except Exception as e:
                log_reps.warning("{}. Skipped {}.".format(e, fnames[i]))
                pass
        # copy attr_dict values
        self.load_attr_dict()
        if os.path.isfile(dest_dir + "/attr_dict.pkl"):
            dest_attr_dc = load_pkl2obj(dest_dir + "/attr_dict.pkl")
        else:
            dest_attr_dc = {}
        # overwrite existing keys in the destination attribute dict
        dest_attr_dc.update(self.attr_dict)
        self.attr_dict = dest_attr_dc
        self.save_attr_dict()

    def split_component(self, dist, new_sd, new_id):
        """
        Splits the supervoxel into components based on a distance threshold and assigns them new IDs.
        
        Args:
            dist (float): The distance threshold for splitting the supervoxel.
            new_sd ('SegmentationDataset'): The SegmentationDataset to which the new components will belong.
            new_id (int): The starting ID for the new components.
        
        Todo:
            * Refactor to use VoxelStorageDyn.
        """
        raise NotImplementedError('WORK IN PROGRESS')
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

                this_voxels = np.zeros(bb[1] - bb[0] + 1, dtype=np.bool)
                this_voxels[this_voxel_list[:, 0],
                            this_voxel_list[:, 1],
                            this_voxel_list[:, 2]] = True
                save_voxels(new_so_obj, this_voxels, bb[0])


class SegmentationDataset(SegmentationBase):
    """
    Represents a set of supervoxel objects within connectomics data. Each supervoxel corresponds to a distinct anatomical or functional region. This class provides utilities for managing and accessing these objects, their properties, and their relationships.
    
    Attributes:
        _type (str): Defines the type of supervoxel objects in the dataset.
        _n_folders_fs (int): Number of folders in the dataset directory tree.
        _sizes (np.ndarray): Array of all supervoxel sizes in the dataset.
        _ids (np.ndarray): Array of unique identifiers for all supervoxels.
        _rep_coords (np.ndarray): Array of representative coordinates for all supervoxels.
        _config (DynConfig): Configuration object with dataset-specific parameters.
        _soid2ix (dict): Maps supervoxel IDs to their index in the dataset.
        _property_cache (dict): Cache for quick access to supervoxel properties.
        _version (str): Version identifier for the dataset.
        _scaling (np.ndarray): Voxel size in nanometers (XYZ).
        _working_dir (str): Path to the dataset working directory.
    
    Examples:
        To initialize the SegmentationDataset for cell supervoxels:
            sd_cell = SegmentationDataset('sv')
    
        After running dataset analysis, load properties from numpy arrays:
            sd_cell.load_numpy_data('size')
    
    Note: More detailed information about supervoxel properties and additional functionality can be found in the original docstring.
    """

    def __init__(self, obj_type: str, version: Optional[Union[str, int]] = None, working_dir: Optional[str] = None,
                 scaling: Optional[Union[List, Tuple, np.ndarray]] = None,
                 version_dict: Optional[Dict[str, str]] = None, create: bool = False,
                 config: Optional[Union[str, DynConfig]] = None,
                 n_folders_fs: Optional[int] = None, cache_properties: Optional[List[str]] = None):
        """
        Initializes a SegmentationDataset with the specified parameters.
        
        Args:
            obj_type: Type of :class:`~syconn.reps.segmentation.SegmentationObject`, such as 'vc', 'sj', 'mi', 'cs', 'sv'.
            version: Version identifier used to distinguish this dataset from others of the same type.
            working_dir: Path to the working directory where the dataset is located or will be created.
            scaling: Scaling factors used to convert voxel dimensions to nanometers.
            version_dict: Dictionary mapping other dataset types to their versions, sharing the same working directory.
            create: If True, creates this dataset's directory if it does not exist.
            config: Config. object or path to a config. file, see :class:`~syconn.handler.config.DynConfig`. Will
                be copied and then fixed by setting :py:attr:`~syconn.handler.config.DynConfig.fix_config` to True.
            n_folders_fs: Number of folders within the dataset's folder structure for organizing data.
            cache_properties: List of supervoxel properties to cache for faster access when initializing
                :py:class:`~syconn.reps.segmentation.SegmentationObject` via :py:func:`~get_segmentation_object`.
        """

        self._type = obj_type

        self._n_folders_fs = n_folders_fs

        self._sizes = None
        self._ids = None
        self._rep_coords = None
        self._config = config
        self._soid2ix = None
        self._property_cache = dict()
        self._version = None
        if cache_properties is None:
            cache_properties = tuple()

        if n_folders_fs is not None:
            if n_folders_fs not in [10 ** i for i in range(6)]:
                raise Exception("n_folders_fs must be in", [10 ** i for i in range(6)])

        if version == 'temp':
            version = 'tmp'
        self._setup_working_dir(working_dir, config, version, scaling)
        if version is not 'tmp' and self._config is not None:
            self._config = copy.copy(self._config)
            self._config.fix_config = True

        if create and (version is None):
            version = 'new'

        if version is None and create is False:
            try:
                self._version = self.config["versions"][self.type]
            except KeyError:
                raise Exception(f"Unclear version '{version}' during initialization of {self}.")
        elif version == "new":
            other_datasets = \
                glob.glob(self.working_dir + "/%s_[0-9]" % self.type) + \
                glob.glob(self.working_dir + "/%s_[0-9][0-9]" % self.type) + \
                glob.glob(self.working_dir + "/%s_[0-9][0-9][0-9]" % self.type)

            max_version = -1

            for other_dataset in other_datasets:
                other_version = \
                    int(re.findall(r"[\d]+",
                                   os.path.basename(other_dataset.strip('/')))[-1])
                if max_version < other_version:
                    max_version = other_version

            self._version = max_version + 1
        else:
            self._version = version

        if version_dict is None:
            try:
                self.version_dict = self.config["versions"]
            except KeyError:
                raise Exception("No version dict specified in config")
        else:
            if isinstance(version_dict, dict):
                self.version_dict = version_dict
            elif isinstance(version_dict, str) and version_dict == "load":
                if self.version_dict_exists:
                    self.load_version_dict()
            else:
                raise Exception("No version dict specified in config")

        if create:
            os.makedirs(self.path, exist_ok=True)
            os.makedirs(self.so_storage_path, exist_ok=True)

        self.enable_property_cache(cache_properties)

    def __repr__(self):
        return (f'{type(self).__name__}(obj_type="{self.type}", version="{self.version}", '
                f'working_dir="{self.working_dir}")')

    @property
    def type(self) -> str:
        """
        Retrieves the type of :class:`~syconn.reps.segmentation.SegmentationObject`s
        contained in this :class:`~syconn.reps.segmentation.SegmentationDataset`.
        
        Returns:
            String identifier of the object type.
        """
        return self._type

    @property
    def n_folders_fs(self) -> int:
        """
        Retrieves the number of folders in the :class:`~syconn.reps.segmentation.SegmentationDataset` directory tree.
        
        Returns:
            The number of folders in this directory tree as an integer.
        """
        if self._n_folders_fs is None:
            ps = glob.glob("%s/%s*/" % (self.path, self.so_storage_path_base))
            if len(ps) == 0:
                raise Exception("No storage folder found at '{}' and no number of "
                                "subfolders specified (n_folders_fs))".format(self.path))

            bp = os.path.basename(ps[0].strip('/'))
            for p in ps:
                bp = os.path.basename(p.strip('/'))
                if bp == self.so_storage_path_base:
                    bp = os.path.basename(p.strip('/'))
                    break

            if bp == self.so_storage_path_base:
                self._n_folders_fs = 100000
            else:
                self._n_folders_fs = int(re.findall(r'[\d]+', bp)[-1])

        return self._n_folders_fs

    @property
    def working_dir(self) -> str:
        """
        Retrieves the working directory of the SegmentationDataset instance.
        
        Returns:
            The working directory of this :class:`~syconn.reps.segmentation.SegmentationDataset` as a string.
        """
        return self._working_dir

    @property
    def version(self) -> str:
        """
        Retrieves the version identifier of the dataset.
        
        Returns:
            String identifier of the version.
        """
        return str(self._version)

    @property
    def path(self) -> str:
        """
        Retrieves the full path to the :class:`~syconn.reps.segmentation.SegmentationDataset`.
        
        Returns:
            The path to this SegmentationDataset as a string.
        """
        return "%s/%s_%s/" % (self._working_dir, self.type, self.version)

    @property
    def exists(self) -> bool:
        """
        Verifies if the dataset directory, referenced by :py:attr:`~path`, exists.
        
        Returns:
            True if the dataset directory exists, False otherwise.
        """
        return os.path.isdir(self.path)

    @property
    def path_sizes(self) -> str:
        """
        Retrieves the path to the cached array of the object voxel sizes.
        
        Returns:
            The path to the sizes.npy file as a string.
        """
        return self.path + "/sizes.npy"

    @property
    def path_rep_coords(self) -> str:
        """
        Retrieves the path to the cached array of the object representative coordinates.
        
        Returns:
            The path to the rep_coords.npy file as a string.
        """
        return self.path + "/rep_coords.npy"

    @property
    def path_ids(self) -> str:
        """
        Retrieves the path to the cached array of the object IDs.
        
        Returns:
            The path to the ids.npy file as a string.
        """
        return self.path + "/ids.npy"

    @property
    def version_dict_path(self) -> str:
        """
        Retrieves the path to the version dictionary pickle file.
        
        Returns:
            The path to the version_dict.pkl file as a string.
        """
        return self.path + "/version_dict.pkl"

    @property
    def version_dict_exists(self) -> bool:
        """
        Checks whether the version dictionary file exists, referred to by :py:attr:`~version_dict_path`.
        
        Returns:
            True if the version_dict.pkl file exists, False otherwise.
        """
        return os.path.exists(self.version_dict_path)

    @property
    def so_storage_path_base(self) -> str:
        """
        Retrieves the base name ('so_storage') of the root folder for supervoxel storage.
        
        Returns:
            The base name of the root folder as a string.
        """
        return "so_storage"

    @property
    def so_storage_path(self) -> str:
        """
        Provides the path to the root folder for supervoxel storage.
        
        Returns:
            str: The path to the root directory.
        """
        if self._n_folders_fs is None and os.path.exists("%s/so_storage/" % self.path):
            return "%s/so_storage/" % self.path
        elif self._n_folders_fs == 100000 and os.path.exists("%s/so_storage/" % self.path):
            return "%s/so_storage/" % self.path
        else:
            return "%s/%s_%d/" % (self.path, self.so_storage_path_base,
                                  self.n_folders_fs)

    @property
    def so_dir_paths(self) -> List[str]:
        """
        Retrieves a sorted list of paths to all supervoxel object directories in the directory tree from the :py:attr:`~so_storage_path`.
        
        Returns:
            A sorted list of paths to supervoxel object directories.
        """
        depth = int(np.log10(self.n_folders_fs) // 2 + np.log10(self.n_folders_fs) % 2)
        p = "".join([self.so_storage_path] + ["/*" for _ in range(depth)])
        return sorted(glob.glob(p))

    def iter_so_dir_paths(self) -> Iterator[str]:
        """
        Iterates over all possible `SegmentationObject` storage base directories. This iterator may return paths to storages that do not exist if no object fell into its ID bucket. 
        
        Returns:
            An iterator yielding paths to ID storage base folders.
        """
        storage_location_ids = get_unique_subfold_ixs(self.n_folders_fs)
        for ix in storage_location_ids:
            yield self.so_storage_path + subfold_from_ix(ix, self.n_folders_fs)

    @property
    def config(self) -> DynConfig:
        """
        Retrieves the configuration object which contains all dataset-specific parameters. 
        See :class:`~syconn.handler.config.DynConfig`.
        
        Returns:
            The configuration object.
        """
        if self._config is None:
            self._config = global_params.config
        return self._config

    @property
    def sizes(self) -> np.ndarray:
        """
        Retrieves the array of sizes for all supervoxels in the dataset.
        
        Returns:
            A size array of all supervoxel which are part of this dataset.
            The ordering of the returned array will correspond to :py:attr:`~ids`.
        """
        if self._sizes is None:
            if os.path.exists(self.path_sizes):
                self._sizes = np.load(self.path_sizes)
            else:
                msg = "sizes were not calculated... Please run dataset_analysis"
                log_reps.error(msg)
                raise ValueError(msg)
        return self._sizes

    @property
    def rep_coords(self) -> np.ndarray:
        """
        Retrieves the array of representative coordinates for all supervoxels part of
        this dataset. The ordering corresponds to :py:attr:`~ids`.
        
        Returns:
            An array of representative coordinates of all supervoxel.
        """
        if self._rep_coords is None:
            if os.path.exists(self.path_rep_coords):
                self._rep_coords = np.load(self.path_rep_coords)
            else:
                msg = "rep_coords were not calculated... Please run dataset_analysis"
                log_reps.error(msg)
                raise ValueError(msg)
        return self._rep_coords

    @property
    def ids(self) -> np.ndarray:
        """
        Retrieves the array of IDs for all supervoxels that are part of this dataset.
        
        Returns:
            An array of all supervoxel IDs which are part of this dataset.
        """
        if self._ids is None:
            acquire_obj_ids(self)
        return self._ids

    @property
    def scaling(self) -> np.ndarray:
        """
        Retrieves the voxel size in nanometers (XYZ).
        
        Returns:
            Voxel size in nanometers (XYZ).
        """
        if self._scaling is None:
            self._scaling = np.array(self.config['scaling'], dtype=np.float32)
        return self._scaling

    @property
    def sos(self) -> Generator[SegmentationObject, None, None]:
        """
        Generator that yields all SegmentationObject instances associated with this dataset.
        
        Yields:
            :class:`~syconn.reps.segmentation.SegmentationObject`
        """
        ix = 0
        tot_nb_sos = len(self.ids)
        while ix < tot_nb_sos:
            yield self.get_segmentation_object(self.ids[ix])
            ix += 1

    def load_numpy_data(self, prop_name, allow_nonexisting: bool = True) -> np.ndarray:
        """
        Loads a cached array of supervoxel properties. The ordering of the returned 
        array will correspond to :py:attr:`~ids`. 
        
        Todo:
            * Remove 's' appendix in file names.
            * Remove 'celltype' replacement for 'celltype_cnn_e3' as soon as 
              'celltype_cnn_e3' was renamed package-wide
        
        Args:
            prop_name: Identifier of the requested cache array.
            allow_nonexisting: If False, raises an error for missing numpy files.
        
        Returns:
            A numpy array of the requested property `prop_name`.
        """
        if prop_name == 'celltype':
            prop_name = 'celltype_cnn_e3'
        if os.path.exists(self.path + prop_name + "s.npy"):
            return np.load(self.path + prop_name + "s.npy", allow_pickle=True)
        else:
            msg = f'Requested data cache "{prop_name}" did not exist in {self}.'
            if not allow_nonexisting:
                log_reps.error(msg)
                raise FileNotFoundError(msg)
            log_reps.warning(msg)

    def get_segmentationdataset(self, obj_type: str) -> 'SegmentationDataset':
        """
        Factory method to retrieve a SegmentationDataset of a specified supervoxel type.
        
        Args:
            obj_type: Dataset of supervoxels with type `obj_type`.
        
        Returns:
            The requested SegmentationDataset object containing the specified supervoxel type.
        """
        if obj_type not in self.version_dict:
            raise ValueError('Requested object type {} not part of version_dict '
                             '{}.'.format(obj_type, self.version_dict))
        return SegmentationDataset(obj_type, version=self.version_dict[obj_type], working_dir=self.working_dir)

    def get_segmentation_object(self, obj_id: Union[int, List[int]],
                                create: bool = False, **kwargs) -> Union[SegmentationObject, List[SegmentationObject]]:
        """
        Factory method for retrieving :class:`~syconn.reps.segmentation.SegmentationObject` instances
        which are part of this dataset.
        
        Args:
            obj_id: A single ID or a list of IDs for the supervoxel(s) to retrieve.
            create: If True, creates the folder hierarchy for the requested supervoxel(s).
        
        Returns:
            The requested :class:`~syconn.reps.segmentation.SegmentationObject` object or a list of 
            SegmentationObject instances.
        """
        if np.isscalar(obj_id):
            return self._get_segmentation_object(obj_id, create, **kwargs)
        else:
            res = []
            for ix in obj_id:
                obj = self._get_segmentation_object(ix, create, **kwargs)
                res.append(obj)
            return res

    def _get_segmentation_object(self, obj_id: int, create: bool, **kwargs) -> SegmentationObject:
        """
        Initializes a SegmentationObject with the specified ID.
        
        Args:
            obj_id: Object ID.
            create: If True, creates the folder structure for the supervoxel object. Default: False.
        
        Returns:
            Supervoxel object.
        """
        kwargs_def = dict(obj_id=obj_id, obj_type=self.type, version=self.version, working_dir=self.working_dir,
                          scaling=self.scaling, create=create, n_folders_fs=self.n_folders_fs, config=self.config)
        kwargs_def.update(kwargs)

        so = SegmentationObject(**kwargs_def)
        for k, v in self._property_cache.items():
            so.attr_dict[k] = v[self.soid2ix[obj_id]]
        return so

    def save_version_dict(self):
        """
        Saves the version dictionary to a `.pkl` pickle file.
        """
        write_obj2pkl(self.version_dict_path, self.version_dict)

    def load_version_dict(self):
        """
        Loads the version dictionary from a `.pkl` file.
        """
        try:
            self.version_dict = load_pkl2obj(self.version_dict_path)
        except Exception as e:
            raise FileNotFoundError('Version dictionary of SegmentationDataset not found. {}'.format(str(e)))

    @property
    def soid2ix(self):
        """
        Retrieves or creates a mapping from supervoxel IDs to their index in the dataset.
        
        Returns:
            A dictionary mapping supervoxel IDs to indices.
        """
        if self._soid2ix is None:
            self._soid2ix = {k: ix for ix, k in enumerate(self.ids)}
        return self._soid2ix

    def enable_property_cache(self, property_keys: Iterable[str]):
        """
        Adds properties to the cache for faster access.
        
        Args:
            property_keys: Property keys. Numpy cache arrays must exist.
        """
        # look-up for so IDs to index in cache arrays
        property_keys = list(property_keys)  # copy
        for k in self._property_cache:
            if k in property_keys:
                property_keys.remove(k)
        if len(property_keys) == 0:
            return
        # init index array
        _ = self.soid2ix
        self._property_cache.update({k: self.load_numpy_data(k, allow_nonexisting=False) for k in property_keys})

    def get_volume(self, source: str = 'total') -> float:
        """
        Calculates the total volume of the region adjacency graph (RAG).
        
        Args:
            source: Allowed sources: 'total' (all SVs contained in 
            SegmentationDataset('sv')), 'neuron' (use glia-free RAG), 
            'glia' (use glia RAG). Specifies which supervoxels to include 
            in the volume calculation.
        
        Returns:
            The volume of the RAG in cubic millimeters.
        """
        self.enable_property_cache(['size'])
        if source == 'neuron':
            g = nx.read_edgelist(global_params.config.pruned_svgraph_path, nodetype=np.uint64)
            svids = g.nodes()
        elif source == 'glia':
            g = nx.read_edgelist(global_params.config.working_dir + "/glia/astrocyte_svgraph.bz2", nodetype=np.uint64)
            svids = g.nodes()
        elif source == 'total':
            svids = self.ids
        else:
            raise ValueError(f'Unknown source type "{source}".')
        total_size = 0
        for svid in svids:
            total_size += self.get_segmentation_object(svid).size
        total_size_cmm = np.prod(self.scaling) * total_size / 1e18
        return total_size_cmm
