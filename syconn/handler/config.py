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
    Basic config object based on yaml. If no ``config.yml`` file exists
    in `working_dir` :py:attr:`~initialized` will be False without raising an
    error.
    """

    def __init__(self, working_dir):
        self._config = None
        self._configspec = None
        self._working_dir = working_dir
        self.initialized = False
        if self._working_dir is not None and len(self._working_dir) > 0:
            self._working_dir = os.path.abspath(self._working_dir)
            self.parse_config()

    def __eq__(self, other: 'Config') -> bool:
        return other.entries == self.entries and \
               other.path_config == self.path_config

    @property
    def entries(self) -> dict:
        """
        Entries stored in the ``config.yml`` file.

        Returns:
            All entries.
        """
        if not self.initialized:
            raise ValueError('Config object was not initialized. "entries" '
                             'are not available.')
        return self._config

    @property
    def working_dir(self) -> str:
        """
        Returns:
            Path to working directory.
        """
        return self._working_dir

    @property
    def path_config(self) -> str:
        """
        Returns:
            Path to config file (``config.yml``).
        """
        return self._working_dir + "/config.yml"

    @property
    def config_exists(self):
        """
        Returns:
            ``True`` if config file exists,
            ``False`` otherwise.
        """
        return os.path.exists(self.path_config)

    @property
    def sections(self) -> List[str]:
        """
        Returns:
            Keys to all sections present in the config file.
        """
        return list(self.entries.keys())

    def parse_config(self):
        """
        Reads the content stored in the config file.
        """
        try:
            self._config = yaml.load(open(self.path_config, 'r'), Loader=yaml.FullLoader)
            self.initialized = True
        except FileNotFoundError:
            pass

    def write_config(self, target_dir=None):
        """
        Write config and configspec to disk.

        Args:
            target_dir: If None, write config to
                :py:attr:`~path_config`. Else,
                writes it to ``target_dir + 'config.yml'``
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
        Args:
            wd: Path to working directory
            log:
            fix_config: Keep config constant.
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
        If `item` is not set in this config, the return value will be taken from
         the default ``config.yml``.

        Args:
            item: Key of the requested value.

        Returns:
            The value which corresponds to `item`.
        """
        try:
            return self.entries[item]
        except (KeyError, ValueError, AttributeError):
            return self.default_conf.entries[item]

    def __setitem__(self, key: str, value: Any) -> Any:
        """
        If `item` is not set in this config, the return value will be taken from
         the default ``config.yml``.

        Args:
            key: Key of the item.
            value: Value of the item.

        Returns:
            The value which corresponds to `item`.
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
        Load default ``config.yml`` if necessary.
        """
        if self._default_conf is None:
            self._default_conf = Config(os.path.split(os.path.abspath(__file__))[0])
            self._default_conf._working_dir = None
        return self._default_conf

    @property
    def entries(self):
        self._check_actuality()
        return super().entries

    @property
    def working_dir(self):
        """
        Returns:
            Path to working directory.
        """
        self._check_actuality()
        return super().working_dir

    @property
    def kd_seg_path(self) -> str:
        """
        Returns:
            Path to cell supervoxel segmentation ``KnossosDataset``.
        """
        return self.entries['paths']['kd_seg']

    @property
    def kd_sym_path(self) -> str:
        """
        Returns:
            Path to synaptic sym. type probability map stored as ``KnossosDataset``.
        """
        return self.entries['paths']['kd_sym']

    @property
    def kd_asym_path(self) -> str:
        """
        Returns:
            Path to synaptic asym. type probability map stored as ``KnossosDataset``.
        """
        return self.entries['paths']['kd_asym']

    @property
    def kd_sj_path(self) -> str:
        """
        Returns:
            Path to synaptic junction probability map or binary predictions stored as
            ``KnossosDataset``.
        """
        return self.entries['paths']['kd_sj']

    @property
    def kd_vc_path(self) -> str:
        """
        Returns:
            Path to vesicle cloud probability map or binary predictions stored as
            ``KnossosDataset``.
        """
        return self.entries['paths']['kd_vc']

    @property
    def kd_mi_path(self) -> str:
        """
        Returns:
            Path to mitochondria probability map or binary predictions stored as
            ``KnossosDataset``.
        """
        return self.entries['paths']['kd_mi']

    @property
    def kd_er_path(self) -> str:
        """
        Returns:
            Path to ER probability map or binary predictions stored as
            ``KnossosDataset``.
        """
        return self.entries['paths']['kd_er']

    @property
    def kd_golgi_path(self) -> str:
        """
        Returns:
            Path to Golgi probability map or binary predictions stored as
            ``KnossosDataset``.
        """
        return self.entries['paths']['kd_golgi']

    @property
    def kd_organells_paths(self) -> Dict[str, str]:
        """
        KDs of subcell. organelle probability maps

        Returns:
            Dictionary containg the paths to ``KnossosDataset`` of available
            cellular containing ``global_params.config['existing_cell_organelles']``.
        """
        path_dict = {k: self.entries['paths']['kd_{}'.format(k)] for k in
                     self['existing_cell_organelles']}
        return path_dict

    @property
    def kd_organelle_seg_paths(self) -> Dict[str, str]:
        """
        KDs of subcell. organelle segmentations.

        Returns:
            Dictionary containing the paths to ``KnossosDataset`` of available
            cellular organelles ``global_params.config['existing_cell_organelles']``.
        """
        path_dict = {k: "{}/knossosdatasets/{}_seg/".format(
            self.working_dir, k) for k in self['existing_cell_organelles']}
        return path_dict

    @property
    def temp_path(self) -> str:
        """

        Returns:
            Path to temporary directory used to store data caches.
        """
        return "{}/tmp/".format(self.working_dir)

    @property
    def init_rag_path(self) -> str:
        """
        Returns:
            Path to initial RAG.
        """
        self._check_actuality()
        p = self.entries['paths']['init_rag']
        if p is None or len(p) == 0:
            p = self.working_dir + "/rag.bz2"
        return p

    @property
    def pruned_rag_path(self) -> str:
        """
        See config parameter
        ``global_params.config['glia']['min_cc_size_ssv']``.

        Returns:
            Path to pruned RAG after size filtering.
        """
        self._check_actuality()
        return self.working_dir + '/pruned_rag.bz2'

    # --------- CLASSIFICATION MODELS
    @property
    def model_dir(self) -> str:
        """
        Returns:
            Path to model directory.
        """
        return self.working_dir + '/models/'

    @property
    def mpath_tnet(self) -> str:
        """
        Returns:
            Path to tCMN - an encoder network of local cell morphology trained via
            triplet loss.
        """
        return self.model_dir + '/tCMN/model.pts'
        # return self.model_dir + '/tCMN/'

    @property
    def mpath_tnet_pts(self) -> str:
        """
        Returns:
            Path to an encoder network of local cell morphology trained via
            triplet loss on point data.
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
        Returns:
            Path to model trained on detecting spine head, neck, dendritic shaft,
            and ``other`` (soma and axon) via 2D projections (-> semantic segmentation).
        """
        return self.model_dir + '/spiness/model.pts'
        # return self.model_dir + '/spiness/'

    @property
    def mpath_axonsem(self) -> str:
        """
        Returns:
            Path to model trained on detecting axon, terminal boutons and en-passant,
            dendrites and somata via 2D projections.
        """
        return self.model_dir + '/axoness_semseg/model.pts'
        # return self.model_dir + '/axoness_semseg/'

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
        Returns:
            Path to model trained on prediction cell types from multi-view sets.
        """
        return self.model_dir + '/celltype_e3/model.pts'

    @property
    def mpath_celltype_pts(self) -> str:
        """
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
        Returns:
            Path to model trained to classify local 2D projections into glia
            vs. neuron (img2scalar).
        """
        return self.model_dir + '/glia_e3/'

    @property
    def mpath_glia_pts(self) -> str:
        """
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
        Returns:
            Path to model trained on identifying myelinated cell parts
            within 3D EM raw data.
        """
        return self.model_dir + '/myelin/model.pts'

    @property
    def mpath_syntype(self) -> str:
        """
        Returns:
            Path to model trained on identifying synapse types (symmetric
            vs. asymmetric) within 3D EM raw data.
        """
        return self.model_dir + '/syntype/model.pts'

    @property
    def mpath_er(self) -> str:
        """
        Returns:
            Path to model trained on identifying cell parts occupied
            by ER within 3D EM raw data.
        """
        return self.model_dir + '/er/model.pts'

    @property
    def mpath_golgi(self) -> str:
        """
        Returns:
            Path to model trained on identifying cell parts occupied
            by Golgi Apparatus within 3D EM raw data.
        """
        return self.model_dir + '/golgi/model.pts'

    @property
    def mpath_syn_rfc(self) -> str:
        return self.model_dir + '/conn_syn_rfc//rfc'

    @property
    def allow_mesh_gen_cells(self) -> bool:
        """
        If ``True``, meshes are not provided for cell supervoxels and will be
        computed from scratch, see :attr:`~syconn.handler.config.DynConf.use_new_meshing`.
        """
        return bool(self['meshes']['allow_mesh_gen_cells'])

    @property
    def allow_ssv_skel_gen(self) -> bool:
        """
        Controls whether cell supervoxel skeletons are provided a priori or
        can be computed from scratch. Currently this is done via a naive sampling
        procedure.

        Returns:
            Value stored at the config.yml file.
        """
        return bool(self['skeleton']['allow_ssv_skel_gen'])

    @property
    def use_kimimaro(self) -> bool:
        """
        Controls if skeletons should be generated with kimimaro
        Returns: value stores in config.yml file

        """
        return bool(self['skeleton']['use_kimimaro'])

    # New config attributes, enable backwards compat. in case these entries do not exist
    @property
    def syntype_available(self) -> bool:
        """
        Synaptic types are available as KnossosDataset. Will be used during the
        matrix generation.

        Returns:
            Value stored at the config.yml file.
        """
        return bool(self['syntype_avail'])

    @property
    def use_point_models(self) -> bool:
        """
        Use point cloud based models instead of multi-views.

        Returns:
            Value stored at the config.yml file.
        """
        return bool(self['use_point_models'])


    @property
    def use_onthefly_views(self) -> bool:
        """
        Generate views for cell type prediction on the fly.

        Returns:
            Value stored at the config.yml file.
        """
        return bool(self['views']['use_onthefly_views'])

    @property
    def use_new_renderings_locs(self) -> bool:
        """
        Use new rendering locations which are faster to computed and are located
        closer to the neuron surface.

        Returns:
            Value stored at the config.yml file.
        """
        return bool(self['views']['use_new_renderings_locs'])

    @property
    def use_new_meshing(self) -> bool:
        """
        Use new, dense meshing (``zmesh``) computed distributed on 3D sub-cubes.
        If ``False`` meshes are computed sparsely, i.e. per object/supervoxel.

        Returns:
            Value stored at the config.yml file.
        """
        return bool(self['meshes']['use_new_meshing'])

    @property
    def qsub_work_folder(self) -> str:
        """
        Directory where intermediate batchjob results are stored.

        Returns:
            Path to directory.
        """
        return f"{self.working_dir}/{self['batch_proc_system']}/"

    @property
    def prior_glia_removal(self) -> bool:
        """
        If ``True`` glia separation procedure will be initiated to create a
        glia-separated RAG (see ``glia/neuron_rag.bz2`` and
        ``glia/glia_rag.bz2``).

        Returns:
            Value stored in ``config.yml``.
        """
        return self.entries['glia']['prior_glia_removal']

    @property
    def use_new_subfold(self) -> bool:
        """
        Use new subfolder hierarchy where objects with similar IDs are stored
        in the same file.

        Returns:
            Value stored in ``config.yml``.
        """
        use_new_subfold = self['paths']['use_new_subfold']
        if use_new_subfold is not None:
            return bool(use_new_subfold)
        else:
            return False

    @property
    def batchjob_script_folder(self) -> str:
        return os.path.abspath(os.path.dirname(os.path.abspath(__file__)) +
                               "/../batchjob_scripts/")

    @property
    def ncore_total(self) -> int:
        return self['nnodes_total'] * self['ncores_per_node']

    @property
    def ngpu_total(self) -> int:
        return self['nnodes_total'] * self['ngpus_per_node']

    @property
    def asym_label(self) -> Optional[int]:
        return self['cell_objects']['asym_label']

    @property
    def sym_label(self) -> Optional[int]:
        return self['cell_objects']['sym_label']


def generate_default_conf(working_dir: str, scaling: Union[Tuple, np.ndarray],
                          syntype_avail: bool = True,
                          use_new_renderings_locs: bool = True,
                          kd_seg: Optional[str] = None, kd_sym: Optional[str] = None,
                          kd_asym: Optional[str] = None,
                          kd_sj: Optional[str] = None, kd_mi: Optional[str] = None,
                          kd_vc: Optional[str] = None, kd_er: Optional[str] = None,
                          kd_golgi: Optional[str] = None, init_rag_p: str = "",
                          prior_glia_removal: bool = True,
                          use_new_meshing: bool = True,
                          allow_mesh_gen_cells: bool = True,
                          use_new_subfold: bool = True, force_overwrite=False,
                          key_value_pairs: Optional[List[tuple]] = None):
    """
    Generates the default SyConn configuration file, including paths to
    ``KnossosDatasets`` of e.g. cellular organelle predictions/prob.
    maps and the cell supervoxel segmentation, general settings for
    OpenGL (egl vs osmesa), the scheduling system (SLURM vs QSUB vs None) and
    various parameters for processing the data. See
    ``SyConn/scripts/example_run/start.py`` for an example.
    ``init_rag`` can be set specifically in the config-file which is optional.
    By default it is set to ``init_rag = working_dir + "rag.bz2"``. SyConn then
    will require an edge list of the supervoxel graph, see also
    ``SyConn/scripts/example_run/start.py``.
    Writes the file ``config.yml`` to `working_dir` after adapting the
    attributes as given by the method input. This file can also only contain
    the values of attributes which should differ from the default config
    at ``SyConn/syconn/handlers/config.yml``. SyConn refers to the latter in
    a parameter cannot be found in the config file inside the currently active
    working directory.

    Examples:
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
        init_rag_p: Path to the initial supervoxel graph.
        prior_glia_removal: If True, applies glia separation before analysing
            cell reconstructions.
        use_new_meshing: If True, uses new meshing procedure based on `zmesh`.
        allow_mesh_gen_cells: If True, meshing of cell supervoxels will be
            permitted.
        use_new_subfold: If True, similar object IDs will be stored in the same
            storage file.
        force_overwrite: Will overwrite existing ``config.yml`` file.
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
    entries['paths']['init_rag'] = init_rag_p
    entries['paths']['use_new_subfold'] = use_new_subfold
    if type(scaling) is np.ndarray:
        scaling = scaling.tolist()
    entries['scaling'] = scaling
    entries['version'] = default_conf.version()
    entries['syntype_avail'] = syntype_avail

    entries['meshes']['allow_mesh_gen_cells'] = allow_mesh_gen_cells
    entries['meshes']['use_new_meshing'] = use_new_meshing

    entries['views']['use_new_renderings_locs'] = use_new_renderings_locs

    entries['glia']['prior_glia_removal'] = prior_glia_removal
    if key_value_pairs is not None:
        _update_key_value_pair_rec(key_value_pairs, entries)
    default_conf._working_dir = working_dir
    if os.path.isfile(default_conf.path_config) and not force_overwrite:
        raise ValueError(f'Overwrite attempt of existing config file at '
                         f'"{default_conf.path_config}".')
    default_conf.write_config(working_dir)


def _update_key_value_pair_rec(key_value_pairs, entries):
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
    Logger for each package module. For import processing steps individual
    logger can be defined (e.g. ``proc``, ``reps``).

    Args:
        log_name: Name of the logger.
        log_dir: Set log_dir specifically. Will then create a filehandler and
            ignore the state of ``global_params.config['disable_file_logging']``
            state.
        overwrite: Overwrite previous log file.

    Returns:
        The logger.
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
    """https://stackoverflow.com/questions/31521859/python-logging-module-time-since-last-log"""

    def filter(self, record):
        try:
            last = self.last
        except AttributeError:
            last = record.relativeCreated

        delta = datetime.datetime.fromtimestamp(record.relativeCreated / 1000.0) - \
                datetime.datetime.fromtimestamp(last / 1000.0)

        record.relative = '{0:.1f}'.format(delta.seconds / 60.)

        self.last = record.relativeCreated
        return True
