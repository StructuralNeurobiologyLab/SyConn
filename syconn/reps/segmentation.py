# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max-Planck-Institute of Neurobiology, Munich, Germany
# Authors: Philipp Schubert, Joergen Kornfeld
import copy
import re
from typing import Union, Tuple, List, Optional, Dict, Generator, Any

import networkx as nx
from knossos_utils import knossosdataset
from scipy import spatial

from .rep_helper import subfold_from_ix, knossos_ml_from_svixs, SegmentationBase
from .segmentation_helper import *
from ..handler.basics import get_filepaths_from_dir, safe_copy, \
    write_txt2kzip, temp_seed
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
                 skeleton_caching: bool = True):
        """
        If `working_dir` is given and the directory contains a valid `config.yml`file,
        all other optional kwargs will be defined by the :class:`~syconn.handler.config.DynConfig`
        object available in :attr:`~syconn.global_params.config`.

        Args:
            obj_id: Unique supervoxel ID.
            obj_type: Type of the supervoxel, keys used currently are:
                * 'mi': Mitochondria
                * 'vc': Vesicle clouds
                * 'sj': Synaptic junction
                * 'syn_ssv': Synapses between two
                * 'syn': Synapse fragment between two
                  :class:`~syconn.reps.segmentation.SegmentationObject`s.
                  :class:`~syconn.reps.super_segmentation_object.SuperSegmentationObject`s.
                * 'cs': Contact site
            version: Version string identifier. if 'tmp' is used, no data will
                be saved to disk.
            working_dir: Path to folder which contains SegmentationDataset of type 'obj_type'.
            rep_coord: Representative coordinate.
            size: Number of voxels.
            scaling: Array defining the voxel size in nanometers (XYZ).
            create: If True, the folder to its storage location :py:attr:`~segobj_dir` will be
                created.
            voxel_caching: Enables caching for voxel data.
            mesh_caching: Enables caching for mesh data.
            view_caching: Enables caching for view data.
            skeleton_caching: Enables caching for skeleton data.
            config: :class:`~syconn.handler.config.DynConfig` object.
            n_folders_fs: Number of folders within the
                :class:`~syconn.reps.segmentation.SegmentationDataset`'s folder structure.
            enable_locking:  If True, enables file locking.
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
        self._mesh = None
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
        return hash((self.id, self.type.__hash__()))

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return self.id == other.id and self.type == other.type

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
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
                                self._skeleton_caching)

    @property
    def type(self) -> str:
        """
        The `type` of the supervoxel.

        Examples:
            Keys which are currently used:
                * 'mi': Mitochondria.
                * 'vc': Vesicle clouds.
                * 'sj': Synaptic junction.
                * 'syn_ssv': Synapses between two
                * 'syn': Synapse fragment between two :class:`~SegmentationObject` objects.
                * 'cs': Contact site.

            Can be used to initialized single :class:`~SegmentationObject` object of
            a specific type or the corresponding dataset collection handled with the
            :class:`~SegmentationDataset` class::

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
            String identifier.
        """
        return self._type

    @property
    def n_folders_fs(self) -> int:
        """
        Number of folders used to store the data of :class:`~SegmentationObject`s. Defines
        the hierarchy of the folder structure organized by
        :class:`~SegmentationDataset`.

        Returns:
            The number of (leaf-) folders used for storing supervoxel data.
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
        Returns:
            Globally unique identifier of this object.
        """
        return self._id

    @property
    def version(self) -> str:
        """
        Version of the :class:`~SegmentationDataset` this object
        belongs to.

        Returns:
            String identifier of the object's version.
        """
        return str(self._version)

    @property
    def voxel_caching(self) -> bool:
        """If True, voxel data is cached after loading."""
        return self._voxel_caching

    @property
    def mesh_caching(self) -> bool:
        """If True, mesh data is cached."""
        return self._mesh_caching

    @property
    def skeleton_caching(self):
        """If True, skeleton data is cached."""
        return self._skeleton_caching

    @property
    def view_caching(self):
        """If True, view data is cached."""
        return self._view_caching

    @property
    def scaling(self):
        """
        Voxel size in nanometers (XYZ). Default is taken from the `config.yml` file and
        accessible via `self.config`.
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
        Factory method for the `~syconn.reps.segmentation.SegmentationDataset` this object
        belongs to.
        """
        return SegmentationDataset(self.type, self.version, self._working_dir)

    @property
    def config(self) -> DynConfig:
        """
        Config. object which contains all dataset-sepcific parameters.
        """
        if self._config is None:
            self._config = global_params.config
        return self._config

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
        `~syconn.reps.segmentation.SegmentationDataset`.
        """
        return "%s_%s" % (self.type, self.version.lstrip("_"))

    @property
    def segds_dir(self) -> str:
        """
        Path to the `~syconn.reps.segmentation.SegmentationDataset` directory.
        """
        return "%s/%s/" % (self.working_dir, self.identifier)

    @property
    def so_storage_path_base(self) -> str:
        """
        Base folder name.

        Todo:
            * refactor.
        """
        return "so_storage"

    @property
    def so_storage_path(self) -> str:
        """
        Path to entry folder of the directory tree where all supervoxel data of
        the corresponding `~syconn.reps.segmentation.SegmentationDataset` is located.
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
        Path to the folder where the data of this supervoxel is stored.
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
        Path to the mesh storage.
        """
        return self.segobj_dir + "mesh.pkl"

    @property
    def skeleton_path(self) -> str:
        """
        Path to the skeleton storage.
        """
        return self.segobj_dir + "skeletons.pkl"

    @property
    def attr_dict_path(self) -> str:
        """
        Path to the attribute storage.
        """
        return self.segobj_dir + "attr_dict.pkl"

    def view_path(self, woglia=True, index_views=False, view_key=None) -> str:
        """
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
        Path to the rendering location storage.
        """
        return self.segobj_dir + "locations.pkl"

    @property
    def voxel_path(self) -> str:
        """
        Path to the voxel storage. See :class:`~syconn.backend.storage.VoxelStorageDyn`
        for details.
        """
        return self.segobj_dir + "/voxel.pkl"

    #                                                                 PROPERTIES
    @property
    def cs_partner(self) -> Optional[List[int]]:
        """
        Contact site specific attribute.
        Returns:
            None if object is not of type 'cs', else return the IDs to the two
            supervoxels which are part of the contact site.
        """
        if self.type in ['cs', 'syn']:
            partner = [self.id >> 32]
            partner.append(self.id - (partner[0] << 32))
            return partner
        else:
            return None

    @property
    def size(self) -> int:
        """
        Returns:
            Number of voxels.
        """
        if self._size is None and self.attr_dict_exists:
            self._size = self.lookup_in_attribute_dict("size")

        if self._size is None:
            self.calculate_size()

        return self._size

    @property
    def shape(self) -> np.ndarray:
        """
        The XYZ extent of this SSV object in voxels.

        Returns:
            The shape/extent of thiss SSV object in voxels (XYZ).
        """
        return self.bounding_box[1] - self.bounding_box[0]

    @property
    def bounding_box(self) -> np.ndarray:
        if self._bounding_box is None and self.attr_dict_exists:
            self._bounding_box = self.lookup_in_attribute_dict("bounding_box")

        if self._bounding_box is None:
            self.calculate_bounding_box()

        return self._bounding_box

    @property
    def rep_coord(self) -> np.ndarray:
        """
        Representative coordinate of this SSV object. Will be the `rep_coord`
        of the first supervoxel in :py:attr:`~svs`.

        Returns:
            1D array of the coordinate (XYZ).
        """
        if self._rep_coord is None and self.attr_dict_exists:
            self._rep_coord = self.lookup_in_attribute_dict("rep_coord")

        if self._rep_coord is None:
            self.calculate_rep_coord()

        return self._rep_coord

    @property
    def attr_dict_exists(self) -> bool:
        """
        Checks if a attribute dictionary file exists at :py:attr:`~attr_dict_path`.

        Returns:
            True if the attribute dictionary file exists.
        """
        if not os.path.isfile(self.attr_dict_path):
            return False
        glob_attr_dc = AttributeDict(self.attr_dict_path,
                                     disable_locking=True)  # look-up only, PS 12Dec2018
        return self.id in glob_attr_dc

    @property
    def voxels_exist(self) -> bool:
        voxel_dc = VoxelStorage(self.voxel_path, read_only=True,
                                disable_locking=True)  # look-up only, PS 12Dec2018
        return self.id in voxel_dc

    @property
    def voxels(self) -> np.ndarray:
        """
        Voxels associated with this SSV object.

        Returns:
            3D binary array indicating voxel locations.
        """
        if self._voxels is None:
            return self.load_voxels()
        else:
            return self._voxels

    @property
    def voxel_list(self) -> np.ndarray:
        """
        Voxels associated with this SSV object.

        Returns:
            2D array with sparse voxel coordinates.
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
        Returns:
            True if mesh exists.
        """
        mesh_dc = MeshStorage(self.mesh_path, disable_locking=True)
        return self.id in mesh_dc

    @property
    def skeleton_exists(self) -> bool:
        """
        Returns:
            True if skeleton exists.
        """
        skeleton_dc = SkeletonStorage(self.skeleton_path, disable_locking=True)
        return self.id in skeleton_dc

    @property
    def mesh(self) -> MeshType:
        """
        Mesh of this object.
        Returns:
            Three flat arrays: indices, vertices, normals.
        """
        # TODO: use self.load_mesh
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
        The skeleton representation of this supervoxel.

        Returns:
            Dict of at least three numpy arrays: "nodes", estimated node "diameters" and "edges".
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
        Bounding box of the object meshes (in nanometers). Approximately
        the same as scaled 'bounding_box'.
         """
        if self._mesh_bb is None and 'mesh_bb' in self.attr_dict:
            self._mesh_bb = self.attr_dict['mesh_bb']
        elif self._mesh_bb is None:
            if len(self.mesh[1]) == 0 or len(self.mesh[0]) == 0:
                self._mesh_bb = self.bounding_box * self.scaling
            else:
                verts = self.mesh[1].reshape(-1, 3)
                self._mesh_bb = [np.min(verts, axis=0),
                                 np.max(verts, axis=0)]
        return self._mesh_bb

    @property
    def mesh_size(self) -> float:
        """
        Length of bounding box diagonal (BBD).

        Returns:
            Diagonal length of the mesh bounding box in nanometers.
        """
        return np.linalg.norm(self.mesh_bb[1] - self.mesh_bb[0], ord=2)

    @property
    def mesh_area(self) -> float:
        """
        Returns:
            Mesh surface area in um^2
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
        Returns:
            True if rendering locations have been stored at :py:attr:`~locations_path`.
        """
        location_dc = CompressedStorage(self.locations_path,
                                        disable_locking=True)  # look-up only, PS 12Dec2018
        return self.id in location_dc

    def views_exist(self, woglia: bool, index_views: bool = False,
                    view_key: Optional[str] = None) -> bool:
        """
        True if rendering locations have been stored at :func:`~view_path`.

        Args:
            woglia: If True, looks for views without glia, i.e. after glia separation.
            index_views: If True, refers to index views.
            view_key: Identifier of the requested views.
        """
        view_dc = CompressedStorage(self.view_path(woglia=woglia, index_views=index_views, view_key=view_key),
                                    disable_locking=True)  # look-up only, PS 12Dec2018
        return self.id in view_dc

    def views(self, woglia: bool, index_views: bool = False,
              view_key: Optional[str] = None) -> Union[np.ndarray, int]:
        """
        Getter method for the views of this supervoxel. Only valid for cell fragments, i.e.
        :py:attr:`~type` must be `sv`.

        Args:
            woglia: If True, looks for views without glia, i.e. after glia separation.
            index_views: If True, refers to index views.
            view_key: Identifier of the requested views.

        Returns:
            The requested view array or `-1` if it does not exist.
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
        Getter method for the rendering locations of this supervoxel. Only valid for cell
        fragments, i.e. :py:attr:`~type` must be `sv`.

        Args:
            force: Overwrite existing data.
            save: If True, saves the result at :py:attr:`~locations_path`. Uses
            :class:`~syconn.backend.storage.CompressedStorage`.
            ds_factor: Down sampling factor used to generate the rendering locations.

        Returns:
            Array of rendering locations (XYZ) with shape (N, 3) in nanometers!
        """
        assert self.type == "sv"
        if self.sample_locations_exist and not force:
            return CompressedStorage(self.locations_path,
                                     disable_locking=True)[self.id]
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
        Loader method of :py:attr:`~voxels`.

        Args:
            voxel_dc: Pre-loaded dictionary which contains the voxel data of this object.

        Returns:
            3D array of the all voxels which belong to this supervoxel.
        """
        if voxel_dc is None:
            voxel_dc = VoxelStorage(self.voxel_path, read_only=True, disable_locking=True)
        if not isinstance(voxel_dc, VoxelStorageDyn):
            voxels = load_voxels_depr(self, voxel_dc=voxel_dc)
        else:
            voxels = voxel_dc.get_voxel_data_cubed(self.id)[0]
        if self.voxel_caching:
            self._voxels = voxels
        return voxels

    def load_voxels_downsampled(self, downsampling=(2, 2, 1)):
        return load_voxels_downsampled(self, ds=downsampling)

    def load_voxel_list(self):
        """
        Loader method of :py:attr:`~voxel_list`.

        Returns:
            Sparse, 2-dimensional array of voxel coordinates.
        """
        return load_voxel_list(self)

    def load_voxel_list_downsampled(self, downsampling=(2, 2, 1)):
        return load_voxel_list_downsampled(self, downsampling=downsampling)

    def load_voxel_list_downsampled_adapt(self, downsampling=(2, 2, 1)):
        return load_voxel_list_downsampled_adapt(self, downsampling=downsampling)

    def load_skeleton(self, recompute: bool = False) -> dict:
        """
        Loader method of :py:attr:`~skeleton`.

        Args:
            recompute: Recompute the skeleton. Currently not implemented.

        Returns:
            Dict of flat arrays of indices, vertices, diameters and attributes.
        """
        return load_skeleton(self, recompute=recompute)

    def save_skeleton(self, overwrite: bool = False):
        """
        Save method of :py:attr:`~skeleton`.

        Args:
            overwrite: Overwrite existing skeleton entry.

        Returns:
            Flat arrays of indices, vertices, normals.
        """
        return save_skeleton(self, overwrite=overwrite)

    def glia_pred(self, thresh: float, pred_key_appendix: str = "") -> int:
        """
        SV glia prediction (0: neuron, 1: glia). Only valid if :py:attr:`type` is `sv`.

        Args:
            thresh: Classification threshold.
            pred_key_appendix: Identifier for specific glia predictions. Only used
                during development.

        Returns:
            The glia prediction of this supervoxel.
        """
        assert self.type == "sv"
        if self.config.use_point_models:
            return int(self.glia_proba(pred_key_appendix) >= thresh)
        return glia_pred_so(self, thresh, pred_key_appendix)

    def glia_proba(self, pred_key_appendix: str = "") -> float:
        """
        SV glia probability (0: neuron, 1: glia). Only valid if :py:attr:`type` is `sv`.

        Args:
            pred_key_appendix: Identifier for specific glia predictions. Only used
                during development.

        Returns:
            The glia prediction of this supervoxel.
        """
        assert self.type == "sv"
        return glia_proba_so(self, pred_key_appendix)

    def axoness_preds(self, pred_key_appendix: str = "") -> np.ndarray:
        """
        Axon prediction (0: dendrite, 1: axon, 2: soma) based on `img2scalar` CMN.

        Args:
            pred_key_appendix: Identifier for specific axon predictions. Only used
                during development.

        Returns:
            The axon prediction of this supervoxel at every :py:attr:`~sample_locations`.
        """
        pred = np.argmax(self.axoness_probas(pred_key_appendix), axis=1)
        return pred

    def axoness_probas(self, pred_key_appendix: str = "") -> np.ndarray:
        """
        Axon probability (0: dendrite, 1: axon, 2: soma) based on `img2scalar` CMN.
        Probability underlying the attribute :py:attr:`axoness_preds`. Only valid if
        :py:attr:`type` is `sv`.

        Args:
            pred_key_appendix: Identifier for specific axon predictions. Only used during development.

        Returns:
            The axon probabilities of this supervoxel at every :py:attr:`~sample_locations`.
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
        Total edge length of the supervoxel :py:attr:`~skeleton` in nanometers.

        Returns:
            Sum of all edge lengths (L2 norm) in :py:attr:`~skeleton`.
        """
        if self.skeleton is None:
            self.load_skeleton()
        nodes = self.skeleton['nodes'].astype(np.float32)
        edges = self.skeleton['edges']
        return np.sum([np.linalg.norm(self.scaling * (nodes[e[0]] - nodes[e[1]])) for e in edges])

    def mesh_from_scratch(self, ds: Optional[Tuple[int, int, int]] = None,
                          **kwargs: dict) -> List[np.ndarray]:
        """
        Calculate the mesh based on :func:`~syconn.proc.meshes.get_object_mesh`.

        Args:
            ds: Downsampling of the object's voxel data.
            **kwargs: Key word arguments passed to :func:`~syconn.proc.meshes.triangulation`.

        Returns:

        """
        if ds is None:
            ds = self.config['meshes']['downsampling'][self.type]
        return meshes.get_object_mesh(self, ds, mesher_kwargs=kwargs)

    def _save_mesh(self, ind: np.ndarray, vert: np.ndarray,
                   normals: np.ndarray):
        """
        Save given mesh at :py:attr:`~mesh_path`. Uses
        the :class:`~syconn.backend.storage.MeshStorage` interface.

        Args:
            ind: Flat index array.
            vert: Flat vertex array.
            normals: Flat normal array.

        """
        mesh_dc = MeshStorage(self.mesh_path, read_only=False,
                              disable_locking=not self.enable_locking)
        mesh_dc[self.id] = [ind, vert, normals]
        mesh_dc.push()

    def mesh2kzip(self, dest_path: str, ext_color: Optional[Union[
        Tuple[int, int, int, int], List, np.ndarray]] = None,
                  ply_name: str = ""):
        """
        Write :py:attr:`~mesh` to k.zip.

        Args:
            dest_path: Path to the k.zip file which contains the :py:attr:`~mesh`.
            ext_color: If set to 0 no color will be written out. Use to adapt
                color inKnossos.
            ply_name: Name of the ply file in the k.zip, must not
                end with `.ply`.
        """
        mesh = self.mesh
        if self.type == "sv":
            color = (130, 130, 130, 160)
        elif self.type == "cs":
            color = (100, 200, 30, 255)
        elif self.type == "conn":
            color = (150, 50, 200, 255)
        elif self.type == "syn":
            color = (150, 50, 200, 255)
        elif self.type == "syn_ssv":
            color = (240, 50, 50, 255)
        elif self.type == "sj":
            color = (int(0.849 * 255), int(0.138 * 255), int(0.133 * 255), 255)
        elif self.type == "vc":
            color = (int(0.175 * 255), int(0.585 * 255), int(0.301 * 255), 255)
        elif self.type == "mi":
            color = (0, 153, 255, 255)
        else:
            raise TypeError("Given object type '{}' does not exist."
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
            dest_path: Path to k.zip file.
        """
        self.load_attr_dict()
        kml = knossos_ml_from_svixs([self.id], coords=[self.rep_coord])
        write_txt2kzip(dest_path, kml, "mergelist.txt")

    def load_views(self, woglia: bool = True, raw_only: bool = False,
                   ignore_missing: bool = False, index_views: bool = False,
                   view_key: Optional[str] = None):
        """
        Loader method of :py:attr:`~views`.

        Args:
            woglia: If True, looks for views without glia, i.e. after glia separation.
            index_views: If True, refers to index views.
            view_key: Identifier of the requested views.
            raw_only: If True, ignores cell organelles projections.
            ignore_missing: If True, will not throw ValueError if views do not exist.

        Returns:
            Views with requested properties.
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
        Saves views according to its properties. If view_key is given it has
        to be a special type of view, e.g. spine predictions. If in this case
        any other kwarg is not set to default it will raise an error.

        Todo:
            * remove `cellobjects_only`.

        Args:
            woglia: If True, looks for views without glia, i.e. after glia separation.
            index_views: If True, refers to index views.
            view_key: Identifier of the requested views.
            views: View array.
            cellobjects_only: Only render cell organelles (deprecated).
            enable_locking: Enable file locking.
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
        Loader method of :py:attr:`~attr_dict`.

        Returns:
            0 if successful, -1 if attribute dictionary storage does not exist.
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
        Saves :py:attr:`~attr_dict` to attr:`~attr_dict_path`. Already existing
        dictionary will be updated.
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
        Values have be serializable and will be written via the
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
        Reads attributes from attribute storage. It will ignore self.attr_dict and
        will always pull it from the storage. Does not throw KeyError, but returns
        None for missing keys.

        Args:
            attr_keys: List of attribute keys which will be loaded from
            :py:attr:`~attr_dict_path`.

        Returns:
            Attribute values corresponding to `attr_keys`
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
            attr_key: Attribute key to look for.

        Returns:
            True if attribute exists, False otherwise.
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
        Returns

        Args:
            attr_key: Attribute key to look for.

        Returns:
            Value of `attr_key` in :py:attr:`~attr_dict` or None if it does not
            exist. If key does not exist in :py:attr:`~attr_dict`, tries to
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
        Calculate/loads supervoxel representative coordinate.

        Args:
            voxel_dc: Pre-loaded dictionary which contains the voxel data of
                this object.
        """
        if voxel_dc is None:
            voxel_dc = VoxelStorage(self.voxel_path, read_only=True,
                                    disable_locking=True)

        if self.id not in voxel_dc:
            self._bounding_box = np.array([[-1, -1, -1], [-1, -1, -1]])
            log_reps.warning("No voxels found in VoxelDict!")
            return

        if isinstance(voxel_dc, VoxelStorageDyn):
            self._rep_coord = voxel_dc.object_repcoord(self.id)
            return

        bin_arrs, block_offsets = voxel_dc[self.id]
        block_offsets = np.array(block_offsets)

        if len(bin_arrs) > 1:
            sizes = []
            for i_bin_arr in range(len(bin_arrs)):
                sizes.append(np.sum(bin_arrs[i_bin_arr]))

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

        id_locs = np.where(vx == vx.max())
        id_locs = np.array(id_locs)

        # downsampling to ensure fast processing - this is deterministic!
        if len(id_locs[0]) > 1e4:
            with temp_seed(0):
                idx = np.random.randint(0, len(id_locs[0]), int(1e4))
            id_locs = np.array([id_locs[0][idx], id_locs[1][idx], id_locs[2][idx]])

        # calculate COM
        COM = np.mean(id_locs, axis=1)

        # ensure that the point is contained inside of the object, i.e. use closest existing point to COM
        kdtree_array = np.swapaxes(id_locs, 0, 1)
        kdtree = spatial.cKDTree(kdtree_array)
        dd, ii = kdtree.query(COM, k=1)
        found_point = kdtree_array[ii, :]

        self._rep_coord = found_point + central_block_offset

    def calculate_bounding_box(self, voxel_dc: Optional[Dict[int, np.ndarray]] = None):
        """
        Calculate supervoxel :py:attr:`~bounding_box`.

        Args:
            voxel_dc: Pre-loaded dictionary which contains the voxel data of this object.
        """
        if voxel_dc is None:
            voxel_dc = VoxelStorage(self.voxel_path, read_only=True,
                                    disable_locking=True)
        if not isinstance(voxel_dc, VoxelStorageDyn):
            _ = self.load_voxels(voxel_dc=voxel_dc)
        else:
            bbs = voxel_dc.get_boundingdata(self.id)
            bb = np.array([bbs[:, 0].min(axis=0), bbs[:, 1].max(axis=0)])
            self._bounding_box = bb

    def calculate_size(self, voxel_dc: Optional[Union[VoxelStorageDyn, VoxelStorage]] = None):
        """
        Calculate supervoxel object :py:attr:`~size`.

        Args:
            voxel_dc: Pre-loaded dictionary which contains the voxel data of this object.
        """
        if voxel_dc is None:
            voxel_dc = VoxelStorage(self.voxel_path, read_only=True,
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
        Write supervoxel segmentation to k.zip.

        Todo:
            * check usage.

        Args:
            path:
            kd:
            write_id: Supervoxel ID.
        """
        if write_id is None:
            write_id = self.id

        if kd is None:
            try:
                kd = kd_factory(self.config.kd_seg_path)
            except:
                raise ValueError("KnossosDataset could not be loaded")

        kd.from_matrix_to_cubes(self.bounding_box[0],
                                data=self.voxels.astype(np.uint64) * write_id,
                                datatype=np.uint64,
                                kzip_path=path,
                                overwrite=False)

    def clear_cache(self):
        """
        Clears the following, cached data:
            * :py:attr:`~voxels`
            * :py:attr:`~voxel_list`
            * :py:attr:`~views`
            * :py:attr:`~skeleton`
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
        Returns:
            Path to skeleton storage.
        """
        return self.segobj_dir + "/skeletons.pkl"

    def copy2dir(self, dest_dir, safe=True):
        """
        Examples:
            To copy the content of this SV object (``sv_orig``) to the
            destination of another (e.g. yet not existing) SV (``sv_target``),
            call ``sv_orig.copy2dir(sv_target.segobj_dir)``. All files contained
            in the directory py:attr:`~segobj_dir` of ``sv_orig`` will be copied to
            ``sv_target.segobj_dir``.

        Args:
            dest_dir: Destination directory where all files contained in
                py:attr:`~segobj_dir` will be copied to.
            safe: If ``True``, will not overwrite existing data.
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
        Todo:
            * refactor -> VoxelStorageDyn

        Args:
            dist:
            new_sd:
            new_id:

        Returns:

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
    This class represents a set of supervoxels.

    Examples:
        To initialize the :class:`~syconn.reps.segmentation.SegmentationDataset` for
        cell supervoxels you need to call ``sd_cell = SegmentationDataset('sv')``.
        This requires an initialized working directory, for this please refer to
        :class:`~syconn.handler.config.DynConfig` or see::

            $ python SyConn/scripts/example_runs/start.py

        After successfully executing
        :class:`~syconn.exec.exec_init.init_cell_subcell_sds`, *cell* supervoxel properties
        can be loaded from cache via the following keys:
            * 'id': ID array, identical to :py:attr:`~ids`.
            * 'bounding_box': Bounding box of every SV.
            * 'size': Number voxels of each SV.
            * 'rep_coord': Representative coordinates for each SV.
            * 'mesh_area': Surface area as computed from the object mesh triangles.
            * 'mapping_sj_ids': Synaptic junction objects which overlap with the respective SVs.
            * 'mapping_sj_ratios': Overlap ratio of the synaptic junctions.
            * 'mapping_vc_ids': Vesicle cloud objects which overlap with the respective SVs.
            * 'mapping_vc_ratios': Overlap ratio of the vesicle clouds.
            * 'mapping_mi_ids': Mitochondria objects which overlap with the respective SVs.
            * 'mapping_mi_ratios': Overlap ratio of the mitochondria.

        If a glia separation is performed, the following attributes will be cached as well:
            * 'glia_probas': Glia probabilities as array of shape (N, 2; N: Rendering
              locations, 2: 0-index=neuron, 1-index=glia).

        The 'mapping' attributes are only computed for cell supervoxels and not for cellular
        organelles (e.g. 'mi', 'vc', etc.; see
        :py:attr:`~syconn.global_params.config['existing_cell_organelles']`).

        For the :class:`~syconn.reps.segmentation.SegmentationDataset` of type 'syn_ssv'
        (which represent the actual synapses between two cell reconstructions), the following
        properties are cached:
            * 'id': ID array, identical to
              :py:attr:`~ids`.
            * 'bounding_box': Bounding box of every SV.
            * 'size': Number voxels of each SV.
            * 'rep_coord': Representative coordinates of each SV.
            * 'mesh_area': Surface area as computed from the object mesh triangles.
            * 'mesh_bb': Bounding box of the object meshes (in nanometers). Approximately
              the same as scaled 'bounding_box'.
            * 'latent_morph': Latent morphology vector at each rendering location; predicted by
              the tCMN.
            * 'neuron_partners': IDs of the two
              :class:`~syconn.reps.super_segmentation_object.SuperSegmentationObject`
              forming the synapse. The ordering of the subsequent 'partner' attributes is
              identical to 'neuron_partners', e.g. 'neuron_partners'=[3, 49] and
              'partner_celltypes'=[0, 1] means that SSV with ID 3 is an excitatory axon
              targeting the MSN SSV with ID 49.
            * 'partner_celltypes': Celltypes of the two SSVs.
            * 'partner_spiness': Spine predictions (0: neck, 1: head, 2: shaft, 3: other) of the
              two sites.
            * 'partner_axoness': Compartment predictions (0: dendrite, 1: axon, 2: soma,
              3: en-passant bouton, 4: terminal bouton) of the two sites.
            * 'syn_prob': Synapse probability as inferred by the RFC (see corresponding
              section the documentation).
            * 'asym_prop': Mean probability of the 'syn_ssv' object voxels for the asymmetric
              type. See :func:`~syconn.extraction.cs_processing_steps._extract_synapse_type_thread` .
            * 'sym_prop': Mean probability of the 'syn_ssv' object voxels for the symmetric
              type. See :func:`~syconn.extraction.cs_processing_steps._extract_synapse_type_thread` .
            * 'syn_type_sym_ratio': ``sym_prop / float(asym_prop + sym_prop)``.
              See :func:`~syconn.extraction.cs_processing_steps._extract_synapse_type_thread` .
            * 'syn_sign': Synaptic "sign" (-1: symmetric, +1: asymmetric). For threshold see
              :py:attr:`~syconn.global_params.config['cell_objects']['sym_thresh']` .
            * 'cs_ids': Contact site IDs associated with each 'syn_ssv' synapse.
            * 'id_cs_ratio': Overlap ratio between contact site and synaptic junction (sj)
              objects.
    """

    def __init__(self, obj_type: str, version: Optional[Union[str, int]] = None, working_dir: Optional[str] = None,
                 scaling: Optional[Union[List, Tuple, np.ndarray]] = None,
                 version_dict: Optional[Dict[str, str]] = None, create: bool = False,
                 config: Optional[Union[str, DynConfig]] = None,
                 n_folders_fs: Optional[int] = None, cache_properties: Optional[List[str]] = None):
        """
        Args:
            obj_type: Type of :class:`~syconn.reps.segmentation.SegmentationObject`, e.g.: 'vc', 'sj', 'mi', 'cs', 'sv'.
            version: Version of dataset to distinguish it from others of the same type.
            working_dir: Path to the working directory.
            scaling: Scaling of the raw data to nanometer
            version_dict: Dictionary which contains the versions of other dataset types which share
                the same working directory.
            create: Whether or not to create this dataset's directory.
            config: Config. object, see :class:`~syconn.handler.config.DynConfig`. Will be copied and then fixed by
                setting :py:attr:`~syconn.handler.config.DynConfig.fix_config` to True.
            n_folders_fs: Number of folders within the dataset's folder structure.
            cache_properties: Use numpy cache arrays to populate the specified object properties when initializing
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
        The type of :class:`~syconn.reps.segmentation.SegmentationObject`s
        contained in this :class:`~syconn.reps.segmentation.SegmentationDataset`.

        Returns:
            String identifier of the object type.
        """
        return self._type

    @property
    def n_folders_fs(self) -> int:
        """
        Returns:
            The number of folders in this :class:`~syconn.reps.segmentation.SegmentationDataset`
            directory tree.
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
        Returns:
            The working directory of this :class:`~syconn.reps.segmentation.SegmentationDataset`.
        """
        return self._working_dir

    @property
    def version(self) -> str:
        """
        Returns:
            String identifier of the version.
        """
        return str(self._version)

    @property
    def path(self) -> str:
        """
        Returns:
            The path to this :class:`~syconn.reps.segmentation.SegmentationDataset`.
        """
        return "%s/%s_%s/" % (self._working_dir, self.type, self.version)

    @property
    def exists(self) -> bool:
        """
        Checks whether :py:attr:`~path` exists.
        """
        return os.path.isdir(self.path)

    @property
    def path_sizes(self) -> str:
        """
        Path to the cache array of the object voxel sizes.

        Returns:
            Path to the numpy file.
        """
        return self.path + "/sizes.npy"

    @property
    def path_rep_coords(self) -> str:
        """
        Path to the cache array of the object representative coordinates.

        Returns:
            Path to the numpy file.
        """
        return self.path + "/rep_coords.npy"

    @property
    def path_ids(self) -> str:
        """
        Path to the cache array of the object IDs.

        Returns:
            Path to the numpy file.
        """
        return self.path + "/ids.npy"

    @property
    def version_dict_path(self) -> str:
        """
        Path to the version dictionary pickle file.

        Returns:
            Path to the pickle file.
        """
        return self.path + "/version_dict.pkl"

    @property
    def version_dict_exists(self) -> bool:
        """
        Checks whether :py:attr:`~version_dict_path` exists.
        """
        return os.path.exists(self.version_dict_path)

    @property
    def so_storage_path_base(self) -> str:
        """
        Name of the base of the root folder (``'so_storage'``).
        """
        return "so_storage"

    @property
    def so_storage_path(self) -> str:
        """
        Path to the root folder.
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
        Sorted paths to all supervoxel object directories in the directory tree
        :py:attr:`~so_storage_path`.
        """
        depth = int(np.log10(self.n_folders_fs) // 2 + np.log10(self.n_folders_fs) % 2)
        p = "".join([self.so_storage_path] + ["/*" for _ in range(depth)])
        return sorted(glob.glob(p))

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
    def sizes(self) -> np.ndarray:
        """
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
        Returns:
            Representative coordinates of all supervoxel which are part of this dataset.
            The ordering of the returned array will correspond to :py:attr:`~ids`.
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
        Returns:
            All supervoxel IDs which are part of this dataset.
        """
        if self._ids is None:
            acquire_obj_ids(self)
        return self._ids

    @property
    def scaling(self) -> np.ndarray:
        """
        Returns:
            Voxel size in nanometers (XYZ).
        """
        if self._scaling is None:
            self._scaling = np.array(self.config['scaling'], dtype=np.float32)
        return self._scaling

    @property
    def sos(self) -> Generator[SegmentationObject, None, None]:
        """
        Generator for all :class:`~syconn.reps.segmentation.SegmentationObject` objects
        associated with this dataset.

        Yields:
            :class:`~syconn.reps.segmentation.SegmentationObject`
        """
        ix = 0
        tot_nb_sos = len(self.ids)
        while ix < tot_nb_sos:
            yield self.get_segmentation_object(self.ids[ix])
            ix += 1

    def load_cached_data(self, prop_name, allow_nonexisting: bool = True) -> np.ndarray:
        """
        Load cached array. The ordering of the returned array will correspond
        to :py:attr:`~ids`.

        Todo:
            * remove 's' appendix in file names.

        Args:
            prop_name: Identifier of the requested cache array.
            allow_nonexisting: If False, will fail for missing numpy files.

        Returns:
            numpy array of property `prop_name`.
        """
        if os.path.exists(self.path + prop_name + "s.npy"):
            return np.load(self.path + prop_name + "s.npy", allow_pickle=True)
        else:
            if not allow_nonexisting:
                msg = f''
                log_reps.error(msg)
                raise FileNotFoundError(msg)
            log_reps.warning(f'Requested data cache "{prop_name}" did not exist.')

    def get_segmentationdataset(self, obj_type: str) -> 'SegmentationDataset':
        """
        Factory method for :class:`~syconn.reps.segmentation.SegmentationDataset` which are part of this dataset.

        Args:
            obj_type: Dataset of supervoxels with type `obj_type`.

        Returns:
            The requested :class:`~syconn.reps.segmentation.SegmentationDataset` object.
        """
        if obj_type not in self.version_dict:
            raise ValueError('Requested object type {} not part of version_dict '
                             '{}.'.format(obj_type, self.version_dict))
        return SegmentationDataset(obj_type, version=self.version_dict[obj_type], working_dir=self.working_dir)

    def get_segmentation_object(self, obj_id: Union[int, List[int]],
                                create: bool = False, **kwargs) -> Union[SegmentationObject, List[SegmentationObject]]:
        """
        Factory method for :class:`~syconn.reps.segmentation.SegmentationObject` which are
        part of this dataset.

        Args:
            obj_id: Supervoxel ID.
            create: If True, creates the folder hierarchy down to the requested supervoxel.

        Returns:
            The requested :class:`~syconn.reps.segmentation.SegmentationObject` object.
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
        Initialize :py:class:`~SegmentationObject`.

        Args:
            obj_id: Object ID.
            create: Create folder structure. Default: False.

        Returns:
            Supervoxel object.
        """
        kwargs_def = dict(obj_id=obj_id, obj_type=self.type, version=self.version, working_dir=self.working_dir,
                          scaling=self.scaling, create=create, n_folders_fs=self.n_folders_fs, config=self.config)
        kwargs_def.update(kwargs)

        so = SegmentationObject(**kwargs_def)
        for k, v in self._property_cache.items():
            so.attr_dict[k] = v[self._soid2ix[obj_id]]
        return so

    def save_version_dict(self):
        """
        Save the version dictionary to the `.pkl` file.
        """
        write_obj2pkl(self.version_dict_path, self.version_dict)

    def load_version_dict(self):
        """
        Load the version dictionary from the `.pkl` file.
        """
        try:
            self.version_dict = load_pkl2obj(self.version_dict_path)
        except Exception as e:
            raise FileNotFoundError('Version dictionary of SegmentationDataset not found. {}'.format(str(e)))

    def enable_property_cache(self, property_keys: Iterable[str]):
        """
        Add properties to cache.

        Args:
            property_keys: Property keys. Numpy cache arrays must exist.
        """
        # look-up for so IDs to index in cache arrays
        if len(property_keys) == 0:
            return
        if self._soid2ix is None:
            self._soid2ix = {k: ix for ix, k in enumerate(self.ids)}
        self._property_cache.update({k: self.load_cached_data(k, allow_nonexisting=False) for k in property_keys})
