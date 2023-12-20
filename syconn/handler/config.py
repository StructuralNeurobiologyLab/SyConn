# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Philipp Schubert
import datetime
import glob
import logging
import os
from logging import Logger
from typing import Tuple, Optional, Union, Dict, Any, List

import coloredlogs
import numpy as np
import yaml
from termcolor import colored

from .. import global_params

__all__ = ['DynConfig', 'generate_default_conf', 'initialize_logging']


class Config(object):
    """
    Basic configuration class that handles the creation, comparison, and management of
    configuration files based on YAML format. It checks for the existence of a `config.yml`
    file in the specified working directory and initializes the configuration accordingly.
    If no configuration file is found, the `initialized` attribute is set to False without
    raising an error.
    
    Attributes:
        _config (dict): A dictionary containing the configuration entries.
        _configspec (NoneType): Placeholder for future specifications of the config.
        _working_dir (str): The absolute path to the working directory.
        initialized (bool): Indicates whether the configuration has been
            successfully initialized.
    
    Args:
        working_dir (str): The path to the working directory where the `config.yml`
            file is expected to be located.
    """

    def __init__(self, working_dir):
        """
        Initializes the Config object by setting the working directory and attempting to
        parse the configuration file.
        
        Args:
            working_dir (str): The path to the working directory where the `config.yml`
                file is expected to be located.
        """
        self._config = None
        self._configspec = None
        self._working_dir = working_dir
        self.initialized = False
        if self._working_dir is not None and len(self._working_dir) > 0:
            self._working_dir = os.path.abspath(self._working_dir)
            self.parse_config()

    def __eq__(self, other: 'Config') -> bool:
        """
        Checks equality between two Config objects based on their entries and configuration
        file paths.
        
        Args:
            other (Config): Another Config object to compare with.
        
        Returns:
            bool: True if both Config objects have the same entries and configuration file
                paths, False otherwise.
        """
        return other.entries == self.entries and \
               other.path_config == self.path_config

    @property
    def entries(self) -> dict:
        """
        Retrieves the entries stored in the `config.yml` file.
        
        Raises:
            ValueError: If the Config object is not initialized and entries are not
                available.
        
        Returns:
            All entries as a dictionary containing the configuration entries.
        """
        if not self.initialized:
            raise ValueError('Config object was not initialized. "entries" '
                             'are not available.')
        return self._config

    @property
    def working_dir(self) -> str:
        """
        Retrieves the path to the working directory.
        
        Returns:
            str: The absolute path to the working directory.
        """
        return self._working_dir

    @property
    def path_config(self) -> str:
        """
        Retrieves the path to the configuration file (`config.yml`).
        
        Returns:
            Path to config file (`config.yml`).
        """
        return self._working_dir + "/config.yml"

    @property
    def config_exists(self):
        """
        Checks if the configuration file (`config.yml`) exists.
        
        Returns:
            bool: True if the configuration file exists, False otherwise.
        """
        return os.path.exists(self.path_config)

    @property
    def sections(self) -> List[str]:
        """
        Retrieves the keys to all sections present in the configuration file.
        
        Returns:
            List[str]: A list of keys representing the sections in the configuration file.
        """
        return list(self.entries.keys())

    def parse_config(self):
        """
        Reads and parses the content stored in the configuration file.
        """
        try:
            with open(self.path_config, 'r') as f:
                self._config = yaml.load(f, Loader=yaml.FullLoader)
                
            self.initialized = True
        except FileNotFoundError:
            pass

    def write_config(self, target_dir=None):
        """
        Writes the configuration and its specifications to disk.
        
        Args:
            target_dir (Optional[str]): The directory where the configuration file should
                be written. If None, writes to :py:attr:`~path_config`. Else, writes to
                `target_dir + 'config.yml'`.
        """
        if self._config is None:
            raise ValueError('ConfigObj not yet parsed.')
        if target_dir is None:
            fname_conf = self.path_config
        else:
            fname_conf = target_dir + '/config.yml'
        with open(fname_conf, 'w') as f:
            f.write(yaml.dump(self.entries, default_flow_style=False))

    @staticmethod
    def version():
        """
        Retrieves the current version of SyConn.
        
        Returns:
            str: The current version string of SyConn.
        """
        from syconn import __version__
        return __version__


class DynConfig(Config):
    """
    Enables dynamic and SyConn-wide update of working directory 'wd' and provides an
    interface to all working directory dependent parameters.
    
    Notes:
        * Due to sync. checks it is favorable to not use :func:`~__getitem__`
          inside loops.
    
    Todo:
        * Start to use ``__getitem__`` instead of :py:attr:`~entries`.
        * Adapt all ``global_params.config.`` usages accordingly.
        * Do not replace any property call for now (e.g. `~allow_mesh_gen_cells`)
          because they convey default parameters for old datasets in case they
          are not present in the default ``config.yml``.
    
    Examples:
        To initialize a working directory at the beginning of your script, run::
    
            from syconn import global_params
            global_params.wd = '~/SyConn/example_cube1/'
            cfg = global_params.config  # this is the `DynConfig` object
    """

    def __init__(self, wd: Optional[str] = None, log: Optional[Logger] = None, fix_config: bool = False):
        """
        Initializes the dynamic configuration with an optional working directory, logger, and
        a flag to keep the configuration constant.
        
        Args:
            wd: Optional; The path to the working directory. If not provided, it will
                default to the global parameter.
            log: Optional; The logger to be used for logging information. If not
                provided, a default logger will be used.
            fix_config: Optional; A boolean flag indicating whether the configuration
                should remain constant or be updated dynamically. Defaults to keeping
                the configuration constant if not specified.
        
        Raises:
            ValueError: If `fix_config` is True but no valid working directory is
                provided.
        """
        verbose = False
        if wd is None:
            wd = global_params.wd
            verbose = True if wd is not None else False
        super().__init__(wd)
        self.fix_config = fix_config
        if fix_config and self.working_dir is None:
            raise ValueError('Fixed config must have a valid working directory.')
        self._default_conf = None
        if log is None:
            log = logging.getLogger('syconn')
            coloredlogs.install(level=self['log_level'], logger=log)
            level = logging.getLevelName(self['log_level'])
            log.setLevel(level)

            if not self['disable_file_logging'] and verbose:
                # create file handler
                log_dir = os.path.expanduser('~') + "/SyConn/logs/"

                os.makedirs(log_dir, exist_ok=True)
                fh = logging.FileHandler(log_dir + 'syconn.log')
                fh.setLevel(level)

                # add the handlers to log
                if os.path.isfile(log_dir + 'syconn.log'):
                    os.remove(log_dir + 'syconn.log')
                log.addHandler(fh)
                log.info("Initialized file logging. Log-files are stored at"
                         " {}.".format(log_dir))
        self.log_main = log
        if verbose:
            self.log_main.info("Initialized stdout logging (level: {}). "
                               "Current working directory:"
                               " ".format(self['log_level']) +
                               colored("'{}'".format(self.working_dir), 'red'))
            if self.initialized is False:
                from syconn import handler
                default_conf_p = os.path.dirname(handler.__file__) + 'config.yml'
                self.log_main.warning(f'Initialized working directory without '
                                      f'existing config file at'
                                      f' {self.path_config}. Using default '
                                      f'parameters as defined in {default_conf_p}.')

    def __getitem__(self, item: str) -> Any:
        """
        Retrieves a configuration value for the specified key, falling back to the default
        configuration (from ``config.yml``) if the key is not set in the current configuration.
        
        Args:
            item: The key of the configuration value to retrieve.
        
        Returns:
            The value which corresponds to `item`.
        """
        try:
            return self.entries[item]
        except (KeyError, ValueError, AttributeError):
            return self.default_conf.entries[item]

    def __setitem__(self, key: str, value: Any) -> Any:
        """
        Sets a configuration value for the specified key. If `key` is not present in
        this config, the value will be set in the default `config.yml`.
        
        Args:
            key: The key of the configuration item to set.
            value: The value to set for the configuration item.
        
        Returns:
            The value which corresponds to `key`.
        """
        self.log_main.warning('Modifying DynConfig items via `__setitem__` '
                              'is currently experimental. To change config '
                              'parameters please make changes in the '
                              'corresponding config.yml entries.')
        try:
            self.entries[key] = value
        except (KeyError, ValueError, AttributeError):
            self.default_conf.entries[key] = value

    def _check_actuality(self):
        """
        Checks os.environ and global_params and triggers an update if the therein
        specified WD is not the same as :py:attr:`~working dir`.
        """
        if self.fix_config:
            return
        # first check if working directory was set in environ, else check if it was changed in memory.
        new_wd = None
        if 'syconn_wd' in os.environ and os.environ['syconn_wd'] is not None and len(os.environ['syconn_wd']) > 0 \
                and os.environ['syconn_wd'] != "None":
            if super().working_dir != os.path.abspath(os.environ['syconn_wd']):
                new_wd = os.path.abspath(os.environ['syconn_wd'])
        elif (global_params.wd is not None) and (len(global_params.wd) > 0) and (global_params.wd != "None") and \
                (super().working_dir != os.path.abspath(global_params.wd)):
            new_wd = os.path.abspath(global_params.wd)
        if new_wd is None:
            return
        super().__init__(new_wd)
        self.log_main.info("Initialized stdout logging (level: {}). "
                           "Current working directory:"
                           " ".format(self['log_level']) +
                           colored("'{}'".format(new_wd), 'red'))
        if self.initialized is False:
            from syconn import handler
            default_conf_p = f'{os.path.dirname(handler.__file__)}/config.yml'
            self.log_main.warning(f'Initialized working directory without '
                                  f'existing config file at'
                                  f' {self.path_config}. Using default '
                                  f'parameters as defined in {default_conf_p}.')

    @property
    def default_conf(self) -> Config:
        """
        Loads the default configuration from the `config.yml` file if it has not been loaded
        already.
        
        Returns:
            The default configuration object.
        """
        if self._default_conf is None:
            self._default_conf = Config(os.path.split(os.path.abspath(__file__))[0])
            self._default_conf._working_dir = None
        return self._default_conf

    @property
    def entries(self):
        """
        Retrieves the current configuration entries, checking for updates to the working
        directory before returning.
        
        Returns:
            A dictionary of the current configuration entries.
        """
        self._check_actuality()
        return super().entries

    @property
    def working_dir(self):
        """
        Retrieves the current working directory, checking for updates before returning.
        
        Returns:
            Path to the current working directory.
        """
        self._check_actuality()
        return super().working_dir

    @property
    def kd_seg_path(self) -> str:
        """
        Retrieves the path to the cell supervoxel segmentation `KnossosDataset`.
        
        Returns:
            Path to cell supervoxel segmentation `KnossosDataset`.
        """
        return self.entries['paths']['kd_seg']

    @property
    def kd_sym_path(self) -> str:
        """
        Retrieves the path to the synaptic sym. type probability map stored as a `KnossosDataset`.
        
        Returns:
            Path to synaptic sym. type probability map stored as ``KnossosDataset``.
        """
        return self.entries['paths']['kd_sym']

    @property
    def kd_asym_path(self) -> str:
        """
        Retrieves the path to the synaptic asym. type probability map stored as a
        `KnossosDataset`.
        
        Returns:
            Path to synaptic asym. type probability map stored as `KnossosDataset`.
        """
        return self.entries['paths']['kd_asym']

    @property
    def kd_sj_path(self) -> str:
        """
        Retrieves the path to the synaptic junction probability map or binary predictions stored
        as a `KnossosDataset`.
        
        Returns:
            Path to synaptic junction probability map or binary predictions stored as
            `KnossosDataset`.
        """
        return self.entries['paths']['kd_sj']

    @property
    def kd_vc_path(self) -> str:
        """
        Retrieves the path to the vesicle cloud probability map or binary predictions stored
        as a `KnossosDataset`.
        
        Returns:
            Path to vesicle cloud probability map or binary predictions stored as
            `KnossosDataset`.
        """
        return self.entries['paths']['kd_vc']

    @property
    def kd_mi_path(self) -> str:
        """
        Retrieves the path to the mitochondria probability map or binary predictions stored
        as a `KnossosDataset`.
        
        Returns:
            Path to mitochondria probability map or binary predictions stored as
            `KnossosDataset`.
        """
        return self.entries['paths']['kd_mi']

    @property
    def kd_er_path(self) -> str:
        """
        Retrieves the path to the ER probability map or binary predictions stored as a
        `KnossosDataset`.
        
        Returns:
            Path to ER probability map or binary predictions stored as
            ``KnossosDataset``.
        """
        return self.entries['paths']['kd_er']

    @property
    def kd_golgi_path(self) -> str:
        """
        Retrieves the path to the Golgi probability map or binary predictions stored as a
        `KnossosDataset`.
        
        Returns:
            Path to Golgi probability map or binary predictions stored as
            `KnossosDataset`.
        """
        return self.entries['paths']['kd_golgi']

    @property
    def kd_organelles_paths(self) -> Dict[str, str]:
        """
        Retrieves the paths to `KnossosDataset` of available cellular organelle probability maps.
        
        Returns:
            Dictionary containing the paths to `KnossosDataset` of available cellular
            organelles specified in `global_params.config['process_cell_organelles']`.
        """
        path_dict = {k: self.entries['paths']['kd_{}'.format(k)] for k in
                     self['process_cell_organelles']}
        return path_dict

    @property
    def kd_organelle_seg_paths(self) -> Dict[str, str]:
        """
        Retrieves the paths to `KnossosDataset` of available cellular organelle segmentations.
        
        Returns:
            Dictionary containing the paths to `KnossosDataset` of available
            cellular organelles defined in `global_params.config['process_cell_organelles']`.
        """
        path_dict = {k: "{}/knossosdatasets/{}_seg/".format(
            self.working_dir, k) for k in self['process_cell_organelles']}
        return path_dict

    @property
    def temp_path(self) -> str:
        """
        Retrieves the path to the temporary directory used to store data caches.
        
        Returns:
            Path to temporary directory used to store data caches.
        """
        return "{}/tmp/".format(self.working_dir)

    @property
    def init_svgraph_path(self) -> str:
        """
        Retrieves the path to the initial region adjacency graph (RAG).
        
        Returns:
            Path to initial RAG.
        """
        self._check_actuality()
        p = self.entries['paths']['init_svgraph']
        if p is None or len(p) == 0:
            p = self.working_dir + "/rag.bz2"
        return p

    @property
    def pruned_svgraph_path(self) -> str:
        """
        Retrieves the path to the pruned supervoxel graph after size filtering. It
        excludes cells or cell fragments with a bounding box diagonal smaller than
        the threshold defined by ``global_params.config['min_cc_size_ssv']``.
        
        Returns:
            The path to the pruned supervoxel graph after size filtering.
        """
        self._check_actuality()
        return self.working_dir + '/pruned_svgraph.bz2'

    @property
    def pruned_svagg_list_path(self) -> str:
        """
        Pruned SV lists. All cells or cell fragments with bounding box diagonal
        of less than `global_params.config['min_cc_size_ssv']` are filtered.
        
        Returns:
            Path to pruned agglomeration list (list of SV IDs for every cell) after size filtering.
        """
        self._check_actuality()
        return self.working_dir + '/pruned_svagg_list.txt'

    @property
    def neuron_svgraph_path(self) -> str:
        """
        Retrieves the path to the neuron supervoxel graph.
        
        Returns:
            The path to the neuron supervoxel graph.
        """
        self._check_actuality()
        return "{}/glia/neuron_svgraph.bz2".format(self.working_dir)

    @property
    def neuron_svagg_list_path(self) -> str:
        """
        Retrieves the path to the agglomeration list for neurons.
        
        Returns:
            Path to agglomeration list (list of SV IDs for every cell).
        """
        self._check_actuality()
        return "{}/glia/neuron_svagg_list.txt".format(self.working_dir)

    def astrocyte_svgraph_path(self) -> str:
        """
        Retrieves the path to the astrocyte supervoxel graph.
        
        Returns:
            The path to the astrocyte supervoxel graph.
        """
        self._check_actuality()
        return "{}/glia/astrocyte_svgraph.bz2".format(global_params.config.working_dir)

    @property
    def astrocyte_svagg_list_path(self) -> str:
        """
        Retrieves the path to the agglomeration list for astrocytes.
        
        Returns:
            Path to agglomeration list (list of SV IDs for every cell).
        """
        self._check_actuality()
        return "{}/glia/astrocyte_svagg_list.txt".format(self.working_dir)

    # --------- CLASSIFICATION MODELS
    @property
    def model_dir(self) -> str:
        """
        Retrieves the path to the directory containing the classification models.
        
        Returns:
            Path to model directory.
        """
        return self.working_dir + '/models/'

    @property
    def mpath_tnet(self) -> str:
        """
        Retrieves the path to the tCMN model, an encoder network of local cell morphology trained
        via triplet loss.
        
        Returns:
            Path to tCMN - an encoder network of local cell morphology trained via
            triplet loss.
        """
        return self.model_dir + '/tCMN/model.pts'
        # return self.model_dir + '/tCMN/'

    @property
    def mpath_tnet_pts(self) -> str:
        """
        Retrieves the path to the encoder network of local cell morphology trained via triplet loss
        on point data.
        
        Returns:
            The path to the encoder network model for point data.
        """
        mpath = glob.glob(self.model_dir + '/pts/*tnet*/state_dict.pth')
        if len(mpath) > 1:
            ixs = [int('j0126' in os.path.split(os.path.dirname(m))[1]) for m in mpath]
            if 'j0126' in global_params.config.working_dir and np.sum(ixs) == 1:
                return mpath[ixs.index(1)]
            ixs = [int('j0251' in os.path.split(os.path.dirname(m))[1]) for m in mpath]
            if 'j0251' in global_params.config.working_dir and np.sum(ixs) == 1:
                return mpath[ixs.index(1)]
            # assume its j0126
            if 'j0251' not in global_params.config.working_dir and np.sum(ixs) == 1:
                mpath.pop(ixs.index(1))
        assert len(mpath) == 1
        return mpath[0]

    @property
    def mpath_tnet_pts_wholecell(self) -> str:
        """
        Returns:
            Path to an encoder network of local cell morphology trained via
            triplet loss on point data.
        """
        mpath = glob.glob(self.model_dir + '/pts/whole_cell_embedding/*tnet*/state_dict.pth')
        if len(mpath) > 1:
            ixs = [int('j0126' in os.path.split(os.path.dirname(m))[1]) for m in mpath]
            if 'j0126' in global_params.config.working_dir and np.sum(ixs) == 1:
                return mpath[ixs.index(1)]
            ixs = [int('j0251' in os.path.split(os.path.dirname(m))[1]) for m in mpath]
            if 'j0251' in global_params.config.working_dir and np.sum(ixs) == 1:
                return mpath[ixs.index(1)]
            # assume its j0126
            if 'j0251' not in global_params.config.working_dir and np.sum(ixs) == 1:
                mpath.pop(ixs.index(1))
        assert len(mpath) == 1
        return mpath[0]

    @property
    def mpath_spiness(self) -> str:
        """
        Retrieves the path to the model trained on detecting spine head, neck, dendritic shaft,
        and ``other`` (soma and axon) via 2D projections for semantic segmentation.
        
        Returns:
            The path to the spine detection model.
        """
        return self.model_dir + '/spiness/model.pts'

    @property
    def mpath_axonsem(self) -> str:
        """
        Retrieves the path to the model trained on detecting axon, terminal boutons, en-passant
        boutons, dendrites, and somata via 2D projections.
        
        Returns:
            Path to model trained on detecting axon, terminal boutons and en-passant,
            dendrites and somata via 2D projections.
        """
        return self.model_dir + '/axoness_semseg/model.pts'

    @property
    def mpath_compartment_pts(self) -> str:
        """
        Returns:
            Path to model trained on detecting axon, terminal and en-passant boutons,
            dendritic shaft, spine head and neck, and soma from point data.
        """
        return self.model_dir + '/compartment_pts/'

    @property
    def mpath_celltype_e3(self) -> str:
        """
        Retrieves the path to the model trained on predicting cell types from multi-view sets.
        
        Returns:
            Path to model trained on prediction cell types from multi-view sets.
        """
        return self.model_dir + '/celltype_e3/model.pts'

    @property
    def mpath_celltype_pts(self) -> str:
        """
        Retrieves the path to the model trained on predicting cell types from point data.
        
        Returns:
            Path to model trained on prediction cell types from point data.
        """
        mpath = glob.glob(self.model_dir + '/pts/*celltype*/state_dict.pth')
        if len(mpath) > 1:
            mpath = [m for m in mpath if 'tnet' not in m]
        ixs = [int('j0126' in os.path.split(os.path.dirname(m))[1]) for m in mpath]
        if 'j0126' in global_params.config.working_dir and np.sum(ixs) == 1:
            return mpath[ixs.index(1)]
        ixs = [int('j0251' in os.path.split(os.path.dirname(m))[1]) for m in mpath]
        if 'j0251' in global_params.config.working_dir and np.sum(ixs) == 1:
            return mpath[ixs.index(1)]
        # assume its j0126
        if 'j0251' not in global_params.config.working_dir and np.sum(ixs) == 1:
            mpath.pop(ixs.index(1))
        assert len(mpath) == 1
        return mpath[0]

    @property
    def mpath_glia_e3(self) -> str:
        """
        Retrieves the path to the model trained to classify local 2D projections into glia vs.
        neuron.
        
        Returns:
            Path to model trained to classify local 2D projections into glia vs. neuron
            (img2scalar).
        """
        return self.model_dir + '/glia_e3/'

    @property
    def mpath_glia_pts(self) -> str:
        """
        Retrieves the path to the point-based model trained to classify local 2D projections into
        glia vs. neuron.
        
        Returns:
            Path to point-based model trained to classify local 2D projections into glia
            vs. neuron.
        """
        mpath = glob.glob(self.model_dir + '/pts/*glia*/state_dict.pth')
        assert len(mpath) == 1
        return mpath[0]

    @property
    def mpath_myelin(self) -> str:
        """
        Retrieves the path to the model trained on identifying myelinated cell parts within 3D EM
        raw data.
        
        Returns:
            Path to model trained on identifying myelinated cell parts within 3D EM raw data.
        """
        return self.model_dir + '/myelin/model.pts'

    @property
    def mpath_syntype(self) -> str:
        """
        Returns the path to the model trained on identifying synapse types (symmetric vs.
        asymmetric) within 3D EM raw data.
        
        Returns:
            The path to the synapse type identification model.
        """
        return self.model_dir + '/syntype/model.pts'

    @property
    def mpath_er(self) -> str:
        """
        Retrieves the path to the model trained on identifying cell parts occupied by ER within 3D
        EM raw data.
        
        Returns:
            Path to model trained on identifying cell parts occupied
            by ER within 3D EM raw data.
        """
        return self.model_dir + '/er/model.pts'

    @property
    def mpath_golgi(self) -> str:
        """
        Retrieves the path to the model trained on identifying cell parts occupied by Golgi
        Apparatus within 3D EM raw data.
        
        Returns:
            Path to the Golgi identification model.
        """
        return self.model_dir + '/golgi/model.pts'

    @property
    def mpath_mivcsj(self) -> str:
        """
        Retrieves the path to the model trained on identifying synapse types (symmetric
        vs. asymmetric) within 3D EM raw data.
        
        Returns:
            The path to the synapse type identification model.
        """
        return self.model_dir + '/mivcsj/model.pt'

    @property
    def mpath_syn_rfc(self) -> str:
        """
        Retrieves the path to the Random Forest Classifier model for synapse classification.
        
        Returns:
            The path to the synapse classification RFC model.
        """
        return self.model_dir + '/conn_syn_rfc//rfc'

    @property
    def mpath_syn_rfc_fallback(self) -> str:
        """
        Path to rfc model created with sklearn==0.21.0 for synapse classification.
        
        Returns:
            The path to the fallback synapse classification RFC model.
        """
        return self.model_dir + '/conn_syn_rfc//rfc_fallback'

    @property
    def allow_mesh_gen_cells(self) -> bool:
        """
        If `True`, meshes are not provided for cell supervoxels and will be computed
        from scratch, see :attr:`~syconn.handler.config.DynConf.use_new_meshing`.
        
        Returns:
            A boolean indicating if mesh generation for cell supervoxels is permitted.
        """
        return bool(self['meshes']['allow_mesh_gen_cells'])

    @property
    def allow_ssv_skel_gen(self) -> bool:
        """
        Determines whether skeleton generation for cell supervoxels is allowed,
        or if provided a priori. A naive sampling procedure is currently employed.
        
        Returns:
            A boolean indicating if skeleton generation for cell supervoxels is permitted,
            matching the configuration specified in the config.yml file.
        """
        return bool(self['skeleton']['allow_ssv_skel_gen'])

    @property
    def use_kimimaro(self) -> bool:
        """
        Controls if skeletons should be generated with kimimaro.
        
        Returns:
            The value stored in the config.yml file, indicating if the kimimaro
            algorithm is to be used for skeleton generation.
        """
        return bool(self['skeleton']['use_kimimaro'])

    # New config attributes, enable backwards compat. in case these entries do not exist
    @property
    def syntype_available(self) -> bool:
        """
        Determines whether synaptic types are available as a `KnossosDataset`.
        
        Returns:
            Value stored at the config.yml file indicating if synaptic types are available
            for use in matrix generation.
        """
        return bool(self['syntype_avail'])

    @property
    def use_point_models(self) -> bool:
        """
        Determines whether to use point cloud based models instead of multi-views.
        
        Returns:
            Value stored at the config.yml file indicating if point cloud based
            models should be used.
        """
        return bool(self['use_point_models'])


    @property
    def use_onthefly_views(self) -> bool:
        """
        Determines whether to generate views for cell type prediction on the fly.
        
        Returns:
            A boolean indicating if on-the-fly view generation should be used for cell
            type prediction, as configured in the config.yml file.
        """
        return bool(self['views']['use_onthefly_views'])

    @property
    def use_new_renderings_locs(self) -> bool:
        """
        Determines whether to use new rendering locations for faster computation and closer proximity
        to the neuron surface.
        
        Returns:
            Value stored at the config.yml file.
        """
        return bool(self['views']['use_new_renderings_locs'])

    @property
    def use_new_meshing(self) -> bool:
        """
        Determines whether to use new, dense meshing computed distributed on 3D sub-cubes.
        If ``False``, meshes are computed sparsely, i.e., per object/supervoxel.
        
        Returns:
            A boolean indicating if new meshing should be used, as stored in the config.yml file.
        """
        return bool(self['meshes']['use_new_meshing'])

    @property
    def qsub_work_folder(self) -> str:
        """
        Retrieves the directory where intermediate batchjob results are stored.
        
        Returns:
            Path to the batchjob work directory.
        """
        return f"{self.working_dir}/{self['batch_proc_system']}/"

    @property
    def prior_astrocyte_removal(self) -> bool:
        """
        Determines whether an astrocyte separation procedure should be initiated to create
        a astrocyte-separated RAG (see `glia/neuron_svgraph.bz2` and 
        `glia/astrocyte_svgraph.bz2`).
        
        Returns:
            Value stored in `config.yml`.
        """
        return self.entries['glia']['prior_astrocyte_removal']

    @property
    def use_new_subfold(self) -> bool:
        """
        Determines whether to use a new subfolder hierarchy for storing objects with
        similar IDs.
        
        Returns:
            Value stored in ``config.yml`` indicating if the new subfolder hierarchy
            should be used.
        """
        use_new_subfold = self['paths']['use_new_subfold']
        if use_new_subfold is not None:
            return bool(use_new_subfold)
        else:
            return False

    @property
    def batchjob_script_folder(self) -> str:
        """
        Retrieves the path to the folder containing batchjob scripts.
        
        Returns:
            The path to the batchjob script folder.
        """
        return os.path.abspath(os.path.dirname(os.path.abspath(__file__)) +
                               "/../batchjob_scripts/")

    @property
    def ncore_total(self) -> int:
        """
        Calculates the total number of cores available for processing.
        
        Returns:
            The total number of cores.
        """
        return self['nnodes_total'] * self['ncores_per_node']

    @property
    def ngpu_total(self) -> int:
        """
        Calculates the total number of GPUs available for processing.
        
        Returns:
            The total number of GPUs.
        """
        return self['nnodes_total'] * self['ngpus_per_node']

    @property
    def asym_label(self) -> Optional[int]:
        """
        Retrieves the label used for asymmetric synapses.
        
        Returns:
            The label for asymmetric synapses, if available.
        """
        return self['cell_objects']['asym_label']

    @property
    def sym_label(self) -> Optional[int]:
        """
        Retrieves the label used for symmetric synapses.
        
        Returns:
            The label for symmetric synapses, if available.
        """
        return self['cell_objects']['sym_label']


def generate_default_conf(working_dir: str, scaling: Union[Tuple, np.ndarray],
                          syntype_avail: bool = True,
                          use_new_renderings_locs: bool = True,
                          kd_seg: Optional[str] = None, kd_sym: Optional[str] = None,
                          kd_asym: Optional[str] = None,
                          kd_sj: Optional[str] = None, kd_mi: Optional[str] = None,
                          kd_vc: Optional[str] = None, kd_er: Optional[str] = None,
                          kd_golgi: Optional[str] = None, init_svgraph_path: str = "",
                          prior_astrocyte_removal: bool = True,
                          use_new_meshing: bool = True,
                          allow_mesh_gen_cells: bool = True,
                          use_new_subfold: bool = True, force_overwrite=False,
                          key_value_pairs: Optional[List[tuple]] = None):
    """
    Generates the default SyConn configuration file, including paths to
    KnossosDatasets of e.g. cellular organelle predictions/probability maps and
    the cell supervoxel segmentation, general settings for OpenGL (egl vs osmesa),
    the scheduling system (SLURM vs QSUB vs None), and various parameters for
    processing the data. See `SyConn/scripts/example_run/start.py` for an example.
    `init_svgraph_path` can be set specifically in the config-file which is
    optional. By default, it is set to `init_svgraph_path = working_dir + "rag.bz2"`.
    SyConn then will require an edge list of the supervoxel graph, see also
    `SyConn/scripts/example_run/start.py`.
    Writes the file `config.yml` to `working_dir` after adapting the attributes as
    given by the method input. This file can also only contain the values of
    attributes which should differ from the default config at
    `SyConn/syconn/handlers/config.yml`. SyConn refers to the latter if a parameter
    cannot be found in the config file inside the currently active working
    directory.
    
    The default config content is located at SyConn/syconn/handler/config.yml
    
    Args:
        working_dir: Folder of the working directory.
        scaling: Voxel size in NM.
        syntype_avail: If True, synapse objects will contain additional type
            property (symmetric vs asymmetric).
        use_new_renderings_locs: If True, uses new heuristic for generating
            rendering locations.
        kd_seg: Path to the KnossosDataset which contains the cell segmentation.
        kd_sym: Path to the symmetric type prediction.
        kd_asym: Path to the asymmetric type prediction.
        kd_sj: Path to the synaptic junction predictions.
        kd_mi: Path to the mitochondria predictions.
        kd_vc: Path to the vesicle cloud predictions.
        kd_er: Path to the ER predictions.
        kd_golgi: Path to the Golgi-Apparatus predictions.
        init_svgraph_path: Path to the initial supervoxel graph.
        prior_astrocyte_removal: If True, applies astrocyte separation before
            analyzing cell reconstructions.
        use_new_meshing: If True, uses new meshing procedure based on `zmesh`.
        allow_mesh_gen_cells: If True, meshing of cell supervoxels will be
            permitted.
        use_new_subfold: If True, similar object IDs will be stored in the same
            storage file.
        force_overwrite: Will overwrite existing `config.yml` file.
        key_value_pairs: List of key-value pairs used to modify attributes in
            the config file.
    """
    if kd_seg is None:
        kd_seg = working_dir + 'knossosdatasets/seg/'
    if kd_sym is None:
        kd_sym = working_dir + 'knossosdatasets/sym/'
    if kd_asym is None:
        kd_asym = working_dir + 'knossosdatasets/asym/'
    if kd_sj is None:
        kd_sj = working_dir + 'knossosdatasets/sj/'
    if kd_mi is None:
        kd_mi = working_dir + 'knossosdatasets/mi/'
    if kd_vc is None:
        kd_vc = working_dir + 'knossosdatasets/vc/'
    if kd_er is None:
        kd_er = working_dir + 'knossosdatasets/er/'
    if kd_golgi is None:
        kd_golgi = working_dir + 'knossosdatasets/golgi/'

    default_conf = Config(os.path.split(os.path.abspath(__file__))[0])
    entries = default_conf.entries
    entries['paths']['kd_seg'] = kd_seg
    entries['paths']['kd_sym'] = kd_sym
    entries['paths']['kd_asym'] = kd_asym
    entries['paths']['kd_sj'] = kd_sj
    entries['paths']['kd_vc'] = kd_vc
    entries['paths']['kd_mi'] = kd_mi
    entries['paths']['kd_er'] = kd_er
    entries['paths']['kd_golgi'] = kd_golgi
    entries['paths']['init_svgraph'] = init_svgraph_path
    entries['paths']['use_new_subfold'] = use_new_subfold
    if type(scaling) is np.ndarray:
        scaling = scaling.tolist()
    entries['scaling'] = scaling
    entries['version'] = default_conf.version()
    entries['syntype_avail'] = syntype_avail

    entries['meshes']['allow_mesh_gen_cells'] = allow_mesh_gen_cells
    entries['meshes']['use_new_meshing'] = use_new_meshing

    entries['views']['use_new_renderings_locs'] = use_new_renderings_locs

    entries['glia']['prior_astrocyte_removal'] = prior_astrocyte_removal
    if key_value_pairs is not None:
        _update_key_value_pair_rec(key_value_pairs, entries)
    default_conf._working_dir = working_dir
    if os.path.isfile(default_conf.path_config) and not force_overwrite:
        raise ValueError(f'Overwrite attempt of existing config file at '
                         f'"{default_conf.path_config}".')
    default_conf.write_config(working_dir)


def _update_key_value_pair_rec(key_value_pairs, entries):
    """
    Recursively updates the entries dictionary with the provided key-value pairs.
    
    Args:
        key_value_pairs: A list of tuples containing the key-value pairs to update.
        entries: The dictionary of entries to be updated.
    """
    for k, v in key_value_pairs:
        if k not in entries:
            raise KeyError(f'Key in provided key-value {k}:{v} pair '
                           f'does not exist in default config.')
        if type(v) is dict:
            _update_key_value_pair_rec(list(v.items()), entries[k])
        else:
            entries[k] = v


def initialize_logging(log_name: str, log_dir: Optional[str] = None,
                       overwrite: bool = True):
    """
    Initializes and configures a logger with the given name. If a log directory is
    provided, it will create a file handler and ignore the `disable_file_logging`
    state from the global configuration. For import processing steps, individual
    loggers can be defined (e.g., `proc`, `reps`).
    
    Args:
        log_name: Name of the logger to be initialized.
        log_dir: Specific directory for log files. If provided, a file handler is
                 created regardless of the global configuration.
        overwrite: If True, the existing log file will be overwritten.
    
    Returns:
        A configured logger instance.
    """
    if log_dir is None:
        log_dir = global_params.config['default_log_dir']
    level = global_params.config['log_level']
    logger = logging.getLogger(log_name)
    logger.setLevel(level)
    coloredlogs.install(level=global_params.config['log_level'], logger=logger,
                        reconfigure=False)  # True possibly leads to stderr output
    if not global_params.config['disable_file_logging'] or log_dir is not None:
        # create file handler which logs even debug messages
        if log_dir is None:
            log_dir = os.path.expanduser('~') + "/.SyConn/logs/"
        try:
            os.makedirs(log_dir, exist_ok=True)
        except TypeError:
            if not os.path.isdir(log_dir):
                os.makedirs(log_dir)
        log_fname = log_dir + '/' + log_name + '.log'
        if overwrite and os.path.isfile(log_fname):
            os.remove(log_fname)
        # add the handlers to logger
        fh = logging.FileHandler(log_fname)
        fh.setLevel(level)
        formatter = logging.Formatter(
            '%(asctime)s (%(relative)smin) - %(name)s - %(levelname)s - %(message)s')
        fh.addFilter(TimeFilter())
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger


class TimeFilter(logging.Filter):
    """
    A logging filter that adds a 'relative' attribute to log records, indicating
    the time elapsed since the last log message. This attribute is given in minutes.
    
    This filter is useful for tracking the time between log messages in long-running
    processes. It can help in understanding the flow of events and diagnosing issues
    by showing the duration of operations or time intervals between log entries.
    """

    def filter(self, record):
        """
        Updates the 'relative' attribute of the log record with the time elapsed since
        the last log message.
        
        Args:
            record: The log record to be processed.
        
        Returns:
            True to indicate that the record should be processed, False otherwise.
        """
        try:
            last = self.last
        except AttributeError:
            last = record.relativeCreated

        delta = datetime.datetime.fromtimestamp(record.relativeCreated / 1000.0) - \
                datetime.datetime.fromtimestamp(last / 1000.0)

        record.relative = '{0:.1f}'.format(delta.seconds / 60.)

        self.last = record.relativeCreated
        return True
