# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Philipp Schubert, Joergen Kornfeld
import copy
import glob
import os
import re
import shutil
from multiprocessing.pool import ThreadPool
from typing import List, Dict, Optional, Union, Tuple, Iterable, Generator

import numpy as np

from . import log_reps
from .rep_helper import SegmentationBase
from .segmentation import SegmentationDataset, SegmentationObject
from .super_segmentation_helper import assemble_from_mergelist
from .super_segmentation_helper import associate_objs_with_skel_nodes
from .super_segmentation_helper import view_embedding_of_sso_nocache
from .super_segmentation_object import SuperSegmentationObject
from .. import global_params
from ..handler.basics import load_pkl2obj, write_obj2pkl, chunkify, kd_factory
from ..handler.config import DynConfig
from ..mp import batchjob_utils as qu
from ..mp import mp_utils as sm

try:
    import cPickle as pkl
except ImportError:
    import pickle as pkl
from knossos_utils import knossosdataset

try:
    from knossos_utils import mergelist_tools
except ImportError:
    from knossos_utils import mergelist_tools_fallback as mergelist_tools


class SuperSegmentationDataset(SegmentationBase):
    """
    This class represents a set of agglomerated supervoxels, which themselves are
    represented by :class:`~syconn.reps.segmentation.SegmentationObject`.

    Examples:
        After successfully executing :py:func:`syconn.exec.exec_init.run_create_neuron_ssd`,
        and subsequent analysis steps (see the ``SyConn/scripts/example_run/start.py``) it is
        possible to load SSV properties via :func:`~load_cached_data` with the following keys
        (the ordering of the arrays corresponds to :py:attr:`~ssv_ids`):
            * 'id': ID array, identical to :py:attr:`~ssv_ids`. All other properties have the same
              ordering as this array, i.e. if SSV with ID 1234 has index 42 in the 'id'-array you
              will find its properties at index 42 in all other cache-arrays.
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

        The following lines initialize the
        :class:`~syconn.reps.super_segmentation_dataset.SuperSegmentationDataset` of the
        example run and explore some of the existing attributes::

            import numpy as np
            from syconn.reps.super_segmentation import *
            ssd = SuperSegmentationDataset(working_dir='~/SyConn/example_cube1/')
            n_synapses = [len(ssv.syn_ssv) for ssv in ssd.ssvs]
            path_length = [ssv.total_edge_length() for ssv in ssd.ssvs]  # in nanometers
            syn_densities = np.array(n_synapses) / np.array(path_length)
            print(np.mean(syn_densities), np.std(syn_densities))

        We can make use of the cached arrays to obtain the total number of synapses per
        cell type as follows::

            celltypes = ssd.load_numpy_data('celltype_cnn_e3')
            n_synapses = np.array([len(el) for el in ssd.load_numpy_data('syn_ssv')])
            n_synapes_per_type = {ct: np.sum(n_synapses[celltypes==ct]) for ct in range(9)}
            print(n_synapes_per_type)

    Attributes:
        sso_caching: WIP, enables caching mechanisms in SuperSegmentationObjects returned via
            `get_super_segmentation_object`
        sso_locking: If True, locking is enabled for SSV files.
    """

    def __init__(self, working_dir: Optional[str] = None, version: Optional[str] = None, ssd_type: str = 'ssv',
                 version_dict: Optional[Dict[str, str]] = None, sv_mapping: Optional[Union[Dict[int, int], str]] = None,
                 scaling: Optional[Union[List, Tuple, np.ndarray]] = None, config: DynConfig = None,
                 sso_caching: bool = False, sso_locking: bool = False, create: bool = False,
                 sd_lookup: Optional[Dict[str, SegmentationDataset]] = None,
                 cache_properties: Optional[List[str]] = None):
        """
        Args:
            working_dir: Path to the working directory.
            version: Indicates the version of the dataset, e.g. '0', 'groundtruth' etc.
            ssd_type: Changes the directory prefix the dataset is stored in. Currently
                there is no real use-case for this.
            version_dict: Dictionary which contains the versions of other dataset types which share
                the same working directory.
            sv_mapping: Dictionary mapping supervoxel IDs (key) to their super-supervoxel ID.
            scaling: Array defining the voxel size in XYZ. Default is taken from the
                `config.yml` file.
            config: Config. object, see :class:`~syconn.handler.config.DynConfig`. Will be copied and then fixed by
                setting :py:attr:`~syconn.handler.config.DynConfig.fix_config` to True.
            sso_caching: WIP, enables caching mechanism in SuperSegmentationObjects returned via
                `get_super_segmentation_object`
            sso_locking: If True, locking is enabled for SSV files.
            sd_lookup: Lookup dict for :py:class:`~syconn.reps.segmentation.SegmentationDataset`, this will enable
                usage of property cache arrays for all attributes which have been specified in `property_cache` during
                init. of `SegmentationDataset` (see :class:`~syconn.reps.segmentation.SegmentationObject`).
            cache_properties: Use numpy cache arrays to populate the specified object properties when initializing
                :py:class:`~syconn.reps.super_segmentation_object.SuperSegmentationObject` via
                :py:func:`~get_super_segmentation_object`.
            create: Create folder.

        """
        self.ssv_dict = {}
        self._mapping_dict = None
        self.sso_caching = sso_caching
        self.sso_locking = sso_locking
        self._mapping_dict_reversed = None

        self._type = ssd_type
        self._id_changer = []
        self._ssv_ids = None
        # cache mechanism
        self._ssoid2ix = None
        self._property_cache = dict()
        if cache_properties is None:
            cache_properties = tuple()

        if version == 'temp':
            version = 'tmp'
        self._setup_working_dir(working_dir, config, version, scaling)
        if version is not 'tmp' and self._config is not None:
            self._config = copy.copy(self._config)
            self._config.fix_config = True

        if version is None:
            try:
                self._version = self.config["versions"][self.type]
            except KeyError:
                raise Exception(f"Unclear version '{version}' during initialization of {self}.")
        elif version == "new":
            other_datasets = glob.glob(self.working_dir + "/%s_*" % self.type)
            max_version = -1
            for other_dataset in other_datasets:
                try:
                    other_version = \
                        int(re.findall(r"[\d]+",
                                       os.path.basename(other_dataset))[-1])
                    if max_version < other_version:
                        max_version = other_version
                # version is not an integer, found version could be e.g. 'tmp'
                except IndexError:
                    pass

            self._version = max_version + 1
        else:
            self._version = version

        # init sd lookup
        if sd_lookup is None:
            sd_lookup = {"sv": None, "vc": None, "mi": None, "sj": None, "syn_ssv": None}
        self.sd_lookup = sd_lookup

        if version_dict is None:
            try:
                self.version_dict = self.config["versions"]
            except KeyError:
                raise ValueError("No version dict specified in config")
        else:
            if isinstance(version_dict, dict):
                self.version_dict = version_dict
            elif isinstance(version_dict, str) and version_dict == "load":
                if self.version_dict_exists:
                    self.load_version_dict()
            else:
                raise ValueError("No version dict specified in config")

        if create:
            os.makedirs(self.path, exist_ok=True)

        if sv_mapping is not None:
            if type(sv_mapping) is dict and 0 in sv_mapping:
                raise ValueError
            self.apply_mergelist(sv_mapping)

        self.enable_property_cache(cache_properties)

    def __repr__(self):
        return (f'{type(self).__name__}(ssd_type="{self.type}", '
                f'version="{self.version}", working_dir="{self.working_dir}")')

    @property
    def type(self) -> str:
        """
        The type of the underlying supervoxel objects. See
        :class:`~syconn.reps.super_segmentation_object.SuperSegmentationObject`.
        """
        return str(self._type)

    @property
    def scaling(self) -> np.ndarray:
        """
        Voxel size in nanometers (XYZ). Default is taken from the `config.yml`
        file and accessible via :py:attr:`~config`.
        """
        return self._scaling

    @property
    def working_dir(self) -> str:
        """
        Working directory.
        """
        return self._working_dir

    @property
    def config(self) -> DynConfig:
        """
        Config. object which contains all dataset-sepcific parameters.
        """
        if self._config is None:
            self._config = global_params.config
        return self._config

    @property
    def path(self) -> str:
        """
        Full path to dataset directory.
        """
        return "%s/%s_%s/" % (self._working_dir, self.type, self.version)

    @property
    def version(self) -> str:
        """
        Indicates the version of the dataset. The version is part of the
        dataset's folder name.
        """
        return str(self._version)

    @property
    def version_dict_path(self) -> str:
        """
        Path to version dictionary file.
        """
        return self.path + "/version_dict.pkl"

    @property
    def mapping_dict_exists(self) -> bool:
        """
        Checks if the mapping dictionary exists (uper-supervoxel ID to sueprvoxel IDs).
        """
        return os.path.exists(self.mapping_dict_path)

    @property
    def mapping_dict_reversed_exists(self) -> bool:
        """
        Checks if the inverse mapping dictionary exists (supervoxel ID to
        super-supervoxel ID).
        """
        return os.path.exists(self.mapping_dict_reversed_path)

    @property
    def mapping_dict_path(self) -> str:
        """
        Path to the mapping dictionary pkl file.
        """
        return self.path + "/mapping_dict.pkl"

    @property
    def mapping_dict_reversed_path(self) -> str:
        """
        Path to the inverse mapping dictionary pkl file.
        """
        return self.path + "/mapping_dict_reversed.pkl"

    @property
    def id_changer_path(self) -> str:
        """
        Path to the ID change array.
        """
        return self.path + "/id_changer.npy"

    @property
    def version_dict_exists(self) -> bool:
        """
        Checks whether the version dictionary exists at :py:attr:`~version_dict_path`.
        """
        return os.path.exists(self.version_dict_path)

    @property
    def id_changer_exists(self) -> bool:
        """
        Checks whether the version dictionary exists at :py:attr:`~id_changer_path`.
        """
        return os.path.exists(self.id_changer_path)

    @property
    def mapping_dict(self) -> Dict[int, np.ndarray]:
        """
        Dictionary which contains the supervoxel IDs for each super-supervoxel.
        """
        if self._mapping_dict is None:
            if self.mapping_dict_exists:
                self.load_mapping_dict()
            else:
                self._mapping_dict = {}
        return self._mapping_dict

    @property
    def mapping_dict_reversed(self) -> Dict[int, int]:
        """
        Dictionary which contains the super-supervoxel ID for every supervoxel.
        """
        if self._mapping_dict_reversed is None:
            if self.mapping_dict_reversed_exists:
                self.load_mapping_dict_reversed()
            else:
                self._mapping_dict_reversed = {}
                self.load_mapping_dict()
                for k, v in self.mapping_dict.items():
                    for ix in v:
                        self._mapping_dict_reversed[ix] = k
                self.save_mapping_dict_reversed()
        return self._mapping_dict_reversed

    @property
    def ssv_ids(self) -> np.ndarray:
        """
        Super-supervoxel IDs which are part of this
        :class:`~syconn.reps.super_segmentation_dataset.SuperSegmentationDataset` object.
        """
        if self._ssv_ids is None:
            # do not change the order of the if statements as it is crucial
            # for the resulting ordering of self.ssv_ids (only ids.npy matches
            # with all the other cached numpy arrays).
            if os.path.exists(self.path + "/ids.npy"):
                self._ssv_ids = np.load(self.path + "/ids.npy")
            elif len(self.mapping_dict) > 0:
                self._ssv_ids = np.array(list(self.mapping_dict.keys()))
            elif self.mapping_dict_exists:
                self.load_mapping_dict()
                self._ssv_ids = np.array(list(self.mapping_dict.keys()))
            else:
                paths = glob.glob(self.path + "/so_storage/*/*/*/")
                self._ssv_ids = np.array([int(os.path.basename(p.strip("/")))
                                          for p in paths], dtype=np.uint)
        return self._ssv_ids

    @property
    def ssvs(self) -> Generator[SuperSegmentationObject, None, None]:
        """
        Generator of :class:`~syconn.reps.super_segmentation_object.SuperSegmentationObject` objects which are part
        of this  :class:`~syconn.reps.super_segmentation_dataset.SuperSegmentationDataset` object.

        Yields:
            :class:`~syconn.reps.super_segmentation_object.SuperSegmentationObject`
        """
        ix = 0
        tot_nb_ssvs = len(self.ssv_ids)
        while ix < tot_nb_ssvs:
            yield self.get_super_segmentation_object(self.ssv_ids[ix])
            ix += 1

    @property
    def sv_ids(self) -> np.ndarray:
        """
        Flat array of supervoxels which are part of all super-supervoxels in this
        :class:`~syconn.reps.super_segmentation_dataset.SuperSegmentationDataset` object.
        """
        self.load_mapping_dict()
        return np.concatenate(list(self.mapping_dict.values()))

    @property
    def id_changer(self) -> List[int]:
        """
        Used to agglomerate of synapse fragments ('syn', supervoxel-level) to whole synapses between cells ('syn_ssv').
        """
        if len(self._id_changer) == 0:
            self.load_id_changer()
        return self._id_changer

    def load_numpy_data(self, prop_name: str, allow_nonexisting: bool = True):
        """
        Todo:
            * remove 's' appendix in file names.

        Args:
            prop_name: Identifier for requested cache array. Ordering of the
                array is the same as :py:attr:`~ssv_ids`.
            allow_nonexisting: If False, will fail for missing numpy files.

        Returns:
            Numpy array of cached property.
        """
        if os.path.exists(self.path + prop_name + "s.npy"):
            return np.load(self.path + prop_name + "s.npy", allow_pickle=True)
        else:
            msg = f'Requested data cache "{prop_name}" did not exist.'
            if not allow_nonexisting:
                log_reps.error(msg)
                raise FileNotFoundError(msg)
            log_reps.warning(msg)

    def sv_id_to_ssv_id(self, sv_id: int) -> int:
        """
        Args:
            sv_id: Supervoxel ID.

        Returns:
            The super-supervoxel ID which `sv_id` is part of.
        """
        return self.id_changer[sv_id]

    def get_segmentationdataset(self, obj_type: str) -> SegmentationDataset:
        assert obj_type in self.version_dict
        return SegmentationDataset(obj_type, version=self.version_dict[obj_type], working_dir=self.working_dir)

    def apply_mergelist(self, sv_mapping: Union[Dict[int, int], str]):
        """
        See :func:`~syconn.reps.super_segmentation_helper.assemble_from_mergelist`.

        Args:
            sv_mapping: Supervoxel agglomeration.

        """
        os.makedirs(self.path, exist_ok=True)
        assemble_from_mergelist(self, sv_mapping)

    def get_super_segmentation_object(self, obj_id: Union[int, Iterable[int]], new_mapping: bool = False,
                                      caching: Optional[bool] = None, create: bool = False) \
            -> Union[SuperSegmentationObject, List[SuperSegmentationObject]]:
        """
        Factory method for
        :class:`~syconn.reps.super_segmentation_object.SuperSegmentationObject`s.
        `bj_id` might be a single ID or list of IDs.

        Args:
            obj_id: ID of the super-supervoxel which should be instantiated. Can also be an
                iterable.
            new_mapping: If True, the returned
                :class:`~syconn.reps.super_segmentation_object.SuperSegmentationObject` object will
                be built on the supervoxel agglomeration stored in
                :py:attr:`~mapping_dict`.
            caching: Enable caching of various attributes.
            create: If True, creates the directory of the super-supervoxel inside the folder
                structure of this dataset.

        Notes:
            * Set the default value of `caching` to False, PS 20Feb2019

        Returns:
            SuperSegmentationObject(s) corresponding to the given `obj_id`
            (int or Iterable[int]).
        """
        kwargs_def = dict(ssd_type=self.type, create=create, scaling=self.scaling, object_caching=caching,
                          voxel_caching=caching, mesh_caching=caching, view_caching=caching, enable_locking_so=False,
                          enable_locking=self.sso_locking, config=self.config)
        if caching is None:
            caching = self.sso_caching
        if np.isscalar(obj_id):
            if new_mapping:
                sso = SuperSegmentationObject(obj_id, self.version, self.version_dict, self.working_dir,
                                              sv_ids=self.mapping_dict[obj_id], **kwargs_def)
            else:
                sso = SuperSegmentationObject(obj_id, self.version, self.version_dict, self.working_dir, **kwargs_def)
            for k, v in self._property_cache.items():
                sso.attr_dict[k] = v[self._ssoid2ix[obj_id]]
            sso._ssd = self
        else:
            sso = []
            for ix in obj_id:
                # call it with scalar input recursively
                sso.append(self.get_super_segmentation_object(ix, create=create, new_mapping=new_mapping,
                                                              caching=caching))
        return sso

    def save_dataset_shallow(self):
        """
        Saves :py:attr:`~version_dict`, :py:attr:`~mapping_dict` and :py:attr:`~id_changer`.
        """
        self.save_version_dict()
        self.save_mapping_dict()
        self.save_mapping_dict_reversed()
        self.save_id_changer()

    def save_dataset_deep(self, extract_only: bool = False, attr_keys: Iterable[str] = (), n_jobs: Optional[int] = None,
                          nb_cpus: Optional[int] = None, use_batchjob=True, new_mapping: bool = True, overwrite=False):
        """
        Saves attributes of all SSVs within the given SSD and computes properties
        like size and representative coordinate. The order of :py:attr:`~ssv_ids`
        may change each run.
        See :func:`~syconn.reps.super_segmentation_dataset.save_dataset_deep`.

        Args:
            extract_only: Only cache attributes `attr_keys` from attribute dict.
                This will add suffix '_sel' to the numpy cache array file names (->
                updates will not apply to the :func:`~load_cached_data` method).
            attr_keys: Attributes to cache, only used if ``extract_only=True``.
            n_jobs: Currently requires any string to enable batch job system,
                will be replaced by a global flag soon.
            nb_cpus: CPUs per worker.
            use_batchjob: Use batchjob processing instead of local multiprocessing.
            new_mapping: Whether to apply new mapping (see :func:`~mapping_dict`).
            overwrite: Remove existing SSD folder, if Fals and a folder already
                exists it raises FileExistsError.

        Returns:

        """
        save_dataset_deep(self, extract_only=extract_only, attr_keys=attr_keys, n_jobs=n_jobs, nb_cpus=nb_cpus,
                          new_mapping=new_mapping, overwrite=overwrite, use_batchjob=use_batchjob)

    def predict_cell_types_skelbased(self, stride: int = 1000,
                                     nb_cpus=1):
        """
        Not used anymore.
        """
        multi_params = []
        for ssv_id_block in [self.ssv_ids[i:i + stride]
                             for i in
                             range(0, len(self.ssv_ids), stride)]:
            multi_params.append([ssv_id_block, self.version, self.version_dict,
                                 self.working_dir])

        if not qu.batchjob_enabled():
            sm.start_multiprocess(predict_cell_type_skelbased_thread,
                                  multi_params, nb_cpus=nb_cpus)

        else:
            qu.batchjob_script(multi_params, "predict_cell_type_skelbased",
                               n_cores=nb_cpus, remove_jobfolder=True)

    def save_version_dict(self):
        """
        Save the version dictionary to a `.pkl` file.
        """
        if len(self.version_dict) > 0:
            write_obj2pkl(self.version_dict_path, self.version_dict)

    def load_version_dict(self):
        """
        Load the version dictionary from the `.pkl` file.
        """
        assert self.version_dict_exists
        self.version_dict = load_pkl2obj(self.version_dict_path)

    def save_mapping_dict(self):
        """
        Save the mapping dictionary to a `.pkl` file.
        """
        if len(self.mapping_dict) > 0:
            write_obj2pkl(self.mapping_dict_path, self.mapping_dict)
        else:
            log_reps.warn(f'No entries in mapping dict of {self}.')

    def save_mapping_dict_reversed(self):
        """
        Save the reversed mapping dictionary to a `.pkl` file.
        """
        if len(self.mapping_dict_reversed) > 0:
            write_obj2pkl(self.mapping_dict_reversed_path,
                          self._mapping_dict_reversed)
        else:
            log_reps.warn(f'No entries in reverse mapping dict of {self}.')

    def load_mapping_dict(self):
        """
        Load the mapping dictionary from the `.pkl` file.
        """
        assert self.mapping_dict_exists
        self._mapping_dict = load_pkl2obj(self.mapping_dict_path)

    def load_mapping_dict_reversed(self):
        """
        Load the reversed mapping dictionary from the `.pkl` file.
        """
        assert self.mapping_dict_reversed_exists
        self._mapping_dict_reversed = load_pkl2obj(self.mapping_dict_reversed_path)

    def save_id_changer(self):
        """
        Save the ID changer as `.npy` file.
        """
        if len(self._id_changer) > 0:
            np.save(self.id_changer_path, self._id_changer)

    def load_id_changer(self):
        """
        Load the ID changer from the `.npy` file.
        """
        assert self.id_changer_exists
        self._id_changer = np.load(self.id_changer_path)

    def enable_property_cache(self, property_keys: List[str]):
        """
        Add properties to cache.

        Args:
            property_keys: Property keys. Numpy cache arrays must exist.
        """
        # look-up for so IDs to index in cache arrays
        if len(property_keys) == 0:
            return
        if self._ssoid2ix is None:
            self._ssoid2ix = {k: ix for ix, k in enumerate(self.ssv_ids)}
        self._property_cache.update({k: self.load_numpy_data(k, allow_nonexisting=False) for k in property_keys})


def save_dataset_deep(ssd: SuperSegmentationDataset, extract_only: bool = False,
                      attr_keys: Iterable = (), n_jobs: Optional[int] = None,
                      nb_cpus: Optional[int] = None, use_batchjob=True,
                      new_mapping: bool = True, overwrite=False):
    """
    Saves attributes of all SSVs within the given SSD and computes properties
    like size and representative coordinate. `ids.npy` order may change after
    repeated runs.

    Todo:
        * extract_only requires refactoring as it stores cache arrays under a
          different filename.
        * allow partial updates of a subset of attributes (e.g. use already
          existing `ids.npy` in case of updating, aka `extract_only=True`).
        * Check consistency of ordering for different runs.

    Args:
        ssd: SuperSegmentationDataset
        extract_only: Only cache attributes (see`attr_keys` from attribute dict.
            This will add a suffix `_sel` to the numpy cache array file names
            (-> updates will not apply to the `load_cached_data` method).
        attr_keys: Attributes to cache, only used if `extract_only=True`
        n_jobs: Currently requires any string to enable batch job system,
            will be replaced by a global flag soon.
        nb_cpus: CPUs per worker.
        use_batchjob: Use batchjob processing instead of local multiprocessing.
        new_mapping: Whether to apply new mapping (see `ssd.mapping_dict`).
        overwrite: Remove existing SSD folder, if Fals and a folder already
            exists it raises FileExistsError.
    """
    if os.path.exists(ssd.path) and len(glob.glob(ssd.path)) > 1:
        if not overwrite:
            msg = f'{ssd} already exists and overwrite is False.'
            log_reps.error(msg)
            raise FileExistsError(msg)
        else:
            shutil.rmtree(ssd.path)

    ssd.save_dataset_shallow()
    if n_jobs is None:
        n_jobs = ssd.config.ncore_total
    multi_params = chunkify(ssd.ssv_ids, n_jobs)
    multi_params = [(ssv_id_block, ssd.version, ssd.version_dict,
                     ssd.working_dir, extract_only, attr_keys,
                     ssd._type, new_mapping) for ssv_id_block in multi_params]

    if not qu.batchjob_enabled() or not use_batchjob:
        results = sm.start_multiprocess(
            _write_super_segmentation_dataset_thread,
            multi_params, nb_cpus=nb_cpus)

    else:
        path_to_out = qu.batchjob_script(multi_params,
                                         "write_super_segmentation_dataset",
                                         n_cores=nb_cpus)

        out_files = glob.glob(path_to_out + "/*")
        results = []
        for out_file in out_files:
            with open(out_file, 'rb') as f:
                results.append(pkl.load(f))
        shutil.rmtree(os.path.abspath(path_to_out + "/../"), ignore_errors=True)
    attr_dict = {}
    for this_attr_dict in results:
        for attribute in this_attr_dict.keys():
            if not attribute in attr_dict:
                attr_dict[attribute] = []

            attr_dict[attribute] += this_attr_dict[attribute]

    if not ssd.mapping_dict_exists or len(ssd.mapping_dict) == 0:
        # initialize mapping dict
        ssd._mapping_dict = dict(zip(attr_dict["id"], attr_dict["sv"]))
        ssd.save_dataset_shallow()

    for attribute in attr_dict.keys():
        if extract_only:
            np.save(ssd.path + "/%ss_sel.npy" % attribute,  # Why '_sel'?
                    attr_dict[attribute])
        else:
            np.save(ssd.path + "/%ss.npy" % attribute, attr_dict[attribute])
    log_reps.info(f'Finished `save_dataset_deep` of {ssd}.')


def _write_super_segmentation_dataset_thread(args):
    ssv_obj_ids = args[0]
    version = args[1]
    version_dict = args[2]
    working_dir = args[3]
    extract_only = args[4]
    attr_keys = args[5]
    ssd_type = args[6]
    new_mapping = args[7]

    ssd = SuperSegmentationDataset(working_dir=working_dir, version=version,
                                   ssd_type=ssd_type, version_dict=version_dict)

    try:
        ssd.load_mapping_dict()
        mapping_dict_avail = True
    except:
        mapping_dict_avail = False

    attr_dict = dict(id=[])

    for ssv_obj_id in ssv_obj_ids:
        ssv_obj = ssd.get_super_segmentation_object(ssv_obj_id,
                                                    new_mapping=new_mapping,
                                                    create=True)
        if ssv_obj.attr_dict_exists:
            ssv_obj.load_attr_dict()

        if not extract_only:

            if len(ssv_obj.attr_dict["sv"]) == 0:
                if mapping_dict_avail:
                    ssv_obj = ssd.get_super_segmentation_object(ssv_obj_id, True)

                    if ssv_obj.attr_dict_exists:
                        ssv_obj.load_attr_dict()
                else:
                    raise Exception("No mapping information found")
        if not extract_only:
            if "rep_coord" not in ssv_obj.attr_dict:
                ssv_obj.attr_dict["rep_coord"] = ssv_obj.rep_coord
            if "bounding_box" not in ssv_obj.attr_dict:
                ssv_obj.attr_dict["bounding_box"] = ssv_obj.bounding_box
            if "size" not in ssv_obj.attr_dict:
                ssv_obj.attr_dict["size"] = ssv_obj.size

        ssv_obj.attr_dict["sv"] = np.array(ssv_obj.attr_dict["sv"],
                                           dtype=np.int)
        if extract_only:
            ignore = False
            for attribute in attr_keys:
                if not attribute in ssv_obj.attr_dict:
                    ignore = True
                    break
            if ignore:
                continue

            attr_dict["id"].append(ssv_obj_id)

            for attribute in attr_keys:
                if attribute not in attr_dict:
                    attr_dict[attribute] = []

                if attribute in ssv_obj.attr_dict:
                    attr_dict[attribute].append(ssv_obj.attr_dict[attribute])
                else:
                    attr_dict[attribute].append(None)
        else:
            attr_dict["id"].append(ssv_obj_id)
            for attribute in ssv_obj.attr_dict.keys():
                if attribute not in attr_dict:
                    attr_dict[attribute] = []

                attr_dict[attribute].append(ssv_obj.attr_dict[attribute])

            ssv_obj.save_attr_dict()
    return attr_dict


def export_to_knossosdataset(ssd, kd, stride=1000, nb_cpus=10):
    multi_params = []
    for ssv_id_block in [ssd.ssv_ids[i:i + stride]
                         for i in range(0, len(ssd.ssv_ids), stride)]:
        multi_params.append([ssv_id_block, ssd.version, ssd.version_dict,
                             ssd.working_dir, kd.knossos_path, nb_cpus])

    if not qu.batchjob_enabled():
        sm.start_multiprocess(_export_ssv_to_knossosdataset_thread,
                              multi_params, nb_cpus=nb_cpus)

    else:
        qu.batchjob_script(
            multi_params, "export_ssv_to_knossosdataset", remove_jobfolder=True)


def _export_ssv_to_knossosdataset_thread(args):
    ssv_obj_ids = args[0]
    version = args[1]
    version_dict = args[2]
    working_dir = args[3]
    kd_path = args[4]
    nb_threads = args[5]

    kd = kd_factory(kd_path)

    ssd = SuperSegmentationDataset(working_dir, version, version_dict)
    ssd.load_mapping_dict()

    for ssv_obj_id in ssv_obj_ids:
        ssv_obj = ssd.get_super_segmentation_object(ssv_obj_id, True)

        offset = ssv_obj.bounding_box[0]
        if not 0 in offset:
            kd.from_matrix_to_cubes(offset,
                                    data=ssv_obj.voxels.astype(np.uint64) *
                                         ssv_obj_id,
                                    overwrite=False,
                                    nb_threads=nb_threads)


def load_voxels_downsampled(sso, downsampling=(2, 2, 1), nb_threads=10):
    def _load_sv_voxels_thread(args):
        sv_id = args[0]
        sv = SegmentationObject(sv_id, obj_type="sv", version=sso.version_dict["sv"],
                                working_dir=sso.working_dir, config=sso.config,
                                voxel_caching=False)
        if sv.voxels_exist:
            box = [np.array(sv.bounding_box[0] - sso.bounding_box[0],
                            dtype=np.int)]

            box[0] /= downsampling
            size = np.array(sv.bounding_box[1] -
                            sv.bounding_box[0], dtype=np.float)
            size = np.ceil(size.astype(np.float) /
                           downsampling).astype(np.int)

            box.append(box[0] + size)

            sv_voxels = sv.voxels
            if not isinstance(sv_voxels, int):
                sv_voxels = sv_voxels[::downsampling[0],
                            ::downsampling[1],
                            ::downsampling[2]]

                voxels[box[0][0]: box[1][0],
                box[0][1]: box[1][1],
                box[0][2]: box[1][2]][sv_voxels] = True

    downsampling = np.array(downsampling, dtype=np.int)

    if len(sso.sv_ids) == 0:
        return None

    voxel_box_size = sso.bounding_box[1] - sso.bounding_box[0]
    voxel_box_size = voxel_box_size.astype(np.float)

    voxel_box_size = np.ceil(voxel_box_size / downsampling).astype(np.int)

    voxels = np.zeros(voxel_box_size, dtype=np.bool)

    multi_params = []
    for sv_id in sso.sv_ids:
        multi_params.append([sv_id])

    if nb_threads > 1:
        pool = ThreadPool(nb_threads)
        pool.map(_load_sv_voxels_thread, multi_params)
        pool.join()
        pool.close()
    else:
        map(_load_sv_voxels_thread, multi_params)

    return voxels


def predict_cell_type_skelbased_thread(args):
    """Skeleton-based celltype prediction"""
    # TODO: check functionality, use 'predict_nodes'!
    ssv_obj_ids = args[0]
    version = args[1]
    version_dict = args[2]
    working_dir = args[3]

    ssd = SuperSegmentationDataset(working_dir, version, version_dict)

    for ssv_id in ssv_obj_ids:
        ssv = ssd.get_super_segmentation_object(ssv_id)

        if not ssv.load_skeleton():
            continue

        ssv.load_attr_dict()
        if "assoc_sj" in ssv.attr_dict:
            ssv.predict_cell_type(feature_context_nm=25000, clf_name="rfc")
        elif len(ssv.skeleton["nodes"]) > 0:
            try:
                associate_objs_with_skel_nodes(ssv, ("sj", "mi", "vc"))
                ssv.predict_cell_type(feature_context_nm=25000, clf_name="rfc")
            except:
                pass


def export_to_knossosdataset_thread(args):
    ssv_obj_ids = args[0]
    version = args[1]
    version_dict = args[2]
    working_dir = args[3]
    kd_path = args[4]
    nb_threads = args[5]

    kd = kd_factory(kd_path)

    ssd = SuperSegmentationDataset(working_dir, version, version_dict)
    ssd.load_mapping_dict()

    for ssv_obj_id in ssv_obj_ids:
        ssv_obj = ssd.get_super_segmentation_object(ssv_obj_id, True)

        offset = ssv_obj.bounding_box[0]
        if not 0 in offset:
            kd.from_matrix_to_cubes(ssv_obj.bounding_box,
                                    data=ssv_obj.voxels.astype(np.uint32) *
                                         ssv_obj_id,
                                    overwrite=False,
                                    nb_threads=nb_threads)


def convert_knossosdataset_thread(args):
    version = args[0]
    version_dict = args[1]
    working_dir = args[2]
    nb_threads = args[3]
    sv_kd_path = args[4]
    ssv_kd_path = args[5]
    offsets = args[6]
    size = args[7]

    sv_kd = kd_factory(sv_kd_path)
    ssv_kd = kd_factory(ssv_kd_path)

    ssd = SuperSegmentationDataset(working_dir, version, version_dict)
    ssd.load_id_changer()

    for offset in offsets:
        block = sv_kd.from_overlaycubes_to_matrix(size, offset,
                                                  datatype=np.uint32,
                                                  nb_threads=nb_threads)

        block = ssd.id_changer[block]

        ssv_kd.from_matrix_to_cubes(offset,
                                    data=block.astype(np.uint32),
                                    datatype=np.uint32,
                                    overwrite=False,
                                    nb_threads=nb_threads)

        raw = sv_kd.from_raw_cubes_to_matrix(size, offset,
                                             nb_threads=nb_threads)

        ssv_kd.from_matrix_to_cubes(offset,
                                    data=raw,
                                    datatype=np.uint8,
                                    as_raw=True,
                                    overwrite=False,
                                    nb_threads=nb_threads)


def write_super_segmentation_dataset_thread(args: Tuple):
    """
    Todo:
        * Check use-cases.
    Args:
        args:

    Returns:

    """
    ssv_obj_ids = args[0]
    version = args[1]
    version_dict = args[2]
    working_dir = args[3]
    extract_only = args[4]
    attr_keys = args[5]

    ssd = SuperSegmentationDataset(working_dir, version, version_dict)

    try:
        ssd.load_mapping_dict()
        mapping_dict_avail = True
    except:
        mapping_dict_avail = False

    attr_dict = dict(id=[])

    for ssv_obj_id in ssv_obj_ids:
        ssv_obj = ssd.get_super_segmentation_object(ssv_obj_id,
                                                    new_mapping=True,
                                                    create=True)

        if ssv_obj.attr_dict_exists:
            ssv_obj.load_attr_dict()

        if not extract_only:

            if len(ssv_obj.attr_dict["sv"]) == 0:
                if mapping_dict_avail:
                    ssv_obj = ssd.get_super_segmentation_object(ssv_obj_id, True)

                    if ssv_obj.attr_dict_exists:
                        ssv_obj.load_attr_dict()
                else:
                    raise Exception("No mapping information found")
        if not extract_only:
            if "rep_coord" not in ssv_obj.attr_dict:
                ssv_obj.attr_dict["rep_coord"] = ssv_obj.rep_coord
            if "bounding_box" not in ssv_obj.attr_dict:
                ssv_obj.attr_dict["bounding_box"] = ssv_obj.bounding_box
            if "size" not in ssv_obj.attr_dict:
                ssv_obj.attr_dict["size"] = ssv_obj.size

        ssv_obj.attr_dict["sv"] = np.array(ssv_obj.attr_dict["sv"],
                                           dtype=np.int)

        if extract_only:
            ignore = False
            for attribute in attr_keys:
                if not attribute in ssv_obj.attr_dict:
                    ignore = True
                    break
            if ignore:
                continue

            attr_dict["id"].append(ssv_obj_id)

            for attribute in attr_keys:
                if attribute not in attr_dict:
                    attr_dict[attribute] = []

                if attribute in ssv_obj.attr_dict:
                    attr_dict[attribute].append(ssv_obj.attr_dict[attribute])
                else:
                    attr_dict[attribute].append(None)
        else:
            attr_dict["id"].append(ssv_obj_id)
            for attribute in ssv_obj.attr_dict.keys():
                if attribute not in attr_dict:
                    attr_dict[attribute] = []

                attr_dict[attribute].append(ssv_obj.attr_dict[attribute])

                ssv_obj.save_attr_dict()

    return attr_dict


def copy_ssvs2new_SSD_simple(ssvs: List[SuperSegmentationObject],
                             new_version: str, target_wd: Optional[str] = None,
                             n_jobs: int = 1, safe: bool = True):
    """
    Creates a new SSD specified with `new_version` and a copy of the given SSVs.
    Usually used for generating distinct GT SSDs. Based on the common
    super-supervoxel dataset (as specified in the `config.yml` file, default:
     ``version=ssv_0``).

    Args:
        ssvs: Source SuperSegmentationObjects taken from default SSD in
            working directory.
        new_version: Version of the new SSV SuperSegmentationDataset where
            SSVs will be copied to.
        target_wd: Path to working directory. If None, the one set in
            :py:attr:`~syconn.gloabal_params` is used.
        n_jobs: Number of jobs used.
        safe: If True, will not overwrite existing data.

    Returns:

    """
    # update existing SSV IDs  # TODO: currently this requires a new mapping dict Unclear what to
    #  do in order to enable updates on existing SSD (e.g. after adding new SSVs)
    # paths = glob.glob(ssd.path + "/so_storage/*/*/*/")
    # ssd._ssv_ids = np.array([int(os.path.basename(p.strip("/")))
    #                           for p in paths], dtype=np.uint)
    if target_wd is None:
        target_wd = global_params.config.working_dir
    scaling = ssvs[0].scaling
    new_ssd = SuperSegmentationDataset(working_dir=target_wd, version=new_version,
                                       scaling=scaling)
    for old_ssv in ssvs:
        new_ssv = SuperSegmentationObject(old_ssv.id, version=new_version,
                                          working_dir=target_wd, sv_ids=old_ssv.sv_ids,
                                          scaling=old_ssv.scaling)
        old_ssv.copy2dir(dest_dir=new_ssv.ssv_dir, safe=safe)
    log_reps.info("Saving dataset deep.")
    new_ssd.save_dataset_deep(new_mapping=False, nb_cpus=n_jobs)


def preproc_sso_skelfeature_thread(args: Tuple):
    """
    Helper function to compute skeleton feature of a cell reconstruction. See
    :func:`~reps.super_segmentation_object.SuperSegmentationObject.skel_features`
    for details.

    Args:
        *args: `ssv_obj_ids`: Cell reconstruction IDs, `args[1:4]` used to
            initialize the :class:`~syconn.reps.super_segmentation_dataset
            .SuperSegmentationDataset`.
    """
    ssv_obj_ids = args[0]
    version = args[1]
    version_dict = args[2]
    working_dir = args[3]

    ssd = SuperSegmentationDataset(working_dir, version, version_dict)

    for ssv_id in ssv_obj_ids:
        ssv = ssd.get_super_segmentation_object(ssv_id)
        ssv.load_skeleton()
        if ssv.skeleton is None or len(ssv.skeleton["nodes"]) == 0:
            log_reps.warning("Skeleton of SSV %d has zero nodes." % ssv_id)
            continue
        for feat_ctx_nm in [500, 1000, 2000, 4000, 8000]:
            try:
                _ = ssv.skel_features(feat_ctx_nm)
            except IndexError as e:
                log_reps.error("Error at SSO %d (context: %d).\n%s" % (
                    ssv.id, feat_ctx_nm, e))


def exctract_ssv_morphology_embedding(args: Union[tuple, list]):
    """
    Helper function to infer local morphology embeddings of a cell
    reconstruction. See :func:`~syconn.reps.super_segmentation_object.SuperSegmentationObject
    .predict_views_embedding` for details.

    Args:
        *args: `ssv_obj_ids`: Cell reconstruction IDs, `args[1:4]` used to
            initialize the :class:`~syconn.reps.super_segmentation_dataset
            .SuperSegmentationDataset`, `pred_key_appendix`: addition to the default
            key for storing the embeddings.
    """
    ssv_obj_ids = args[0]
    nb_cpus = args[1]
    pred_key_appendix = args[2]
    use_onthefly_views = global_params.config.use_onthefly_views
    view_props = global_params.config['views']['view_properties']

    ssd = SuperSegmentationDataset()
    from ..handler.prediction import get_tripletnet_model_e3
    for ssv_id in ssv_obj_ids:
        ssv = ssd.get_super_segmentation_object(ssv_id)
        m = get_tripletnet_model_e3()
        ssv.nb_cpus = nb_cpus
        ssv._view_caching = True
        if use_onthefly_views:
            view_embedding_of_sso_nocache(ssv, m, pred_key_appendix=pred_key_appendix,
                                          overwrite=True, **view_props)
        else:
            ssv.predict_views_embedding(m, pred_key_appendix)
