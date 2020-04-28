# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Sven Dorkenwald, Philipp Schubert, Joergen Kornfeld
from typing import Tuple, Optional, Union, Dict, Any, List
import yaml
import logging
from logging import Logger
import coloredlogs
import datetime
import numpy as np
from termcolor import colored
import os
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
            self._config = yaml.load(open(self.path_config, 'r'),
                                     Loader=yaml.FullLoader)
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

    def version(self):
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
    def __init__(self, wd: Optional[str] = None, log: Optional[Logger] = None):
        """
        Args:
            wd: Path to working directory
        """
        verbose = False
        if wd is None:
            wd = global_params.wd
            verbose = True if wd is not None else False
        super().__init__(wd)
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
        # first check if working directory was set in environ,
        # else check if it was changed in memory.
        new_wd = None
        if 'syconn_wd' in os.environ and os.environ['syconn_wd'] is not None and \
            len(os.environ['syconn_wd']) > 0 and os.environ['syconn_wd'] != "None":
            if super().working_dir != os.environ['syconn_wd']:
                new_wd = os.environ['syconn_wd']
        elif (global_params.wd is not None) and (len(global_params.wd) > 0) and \
                (global_params.wd != "None") and (super().working_dir != global_params.wd):
            new_wd = global_params.wd
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
        # return "/tmp/{}_syconn/".format(pwd.getpwuid(os.getuid()).pw_name)
        return "{}/tmp/".format(self.working_dir)

    # TODO: Work-in usage of init_rag_path
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
            Path to tCMN - a decoder network of local cell morphology trained via
            triplet loss.
        """
        return self.model_dir + '/tCMN/model.pts'

    @property
    def mpath_spiness(self) -> str:
        """
        Returns:
            Path to model trained on detecting spine head, neck, dendritic shaft,
            and ``other`` (soma and axon) via 2D projections (-> semantic segmentation).
        """
        return self.model_dir + '/spiness/model.pts'

    @property
    def mpath_axonsem(self) -> str:
        """
        Returns:
            Path to model trained on detecting axon, terminal boutons and en-passant,
            dendrites and somata via 2D projections (-> semantic segmentation).
        """
        return self.model_dir + '/axoness_semseg/model.pts'

    @property
    def mpath_celltype_e3(self) -> str:
        """
        Returns:
            Path to model trained on prediction cell types from multi-view sets.
        """
        return self.model_dir + '/celltype_e3/model.pts'

    @property
    def mpath_glia_e3(self) -> str:
        """
        Returns:
            Path to model trained to classify local 2D projections into glia
            vs. neuron (img2scalar).
        """
        return self.model_dir + '/glia_e3/model.pts'

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
    def mpath_syn_rfc(self) -> str:
        return self.model_dir + '/conn_syn_rfc//rfc'

    @property
    def allow_mesh_gen_cells(self) -> bool:
        """
        If ``True``, meshes are not provided for cell supervoxels and will be
        computed from scratch, see :attr:`~syconn.handler.config.DynConf.use_new_meshing`.
        """
        try:
            if self.entries['meshes']['allow_mesh_gen_cells'] is None:
                raise KeyError
            return self.entries['meshes']['allow_mesh_gen_cells']
        except KeyError:
            return False

    @property
    def allow_skel_gen(self) -> bool:
        """
        Controls whether cell supervoxel skeletons are provided a priori or
        can be computed from scratch. Currently this is done via a naive sampling
        procedure.

        Returns:
            Value stored at the config.yml file.
        """
        return self.entries['skeleton']['allow_skel_gen']

    # New config attributes, enable backwards compat. in case these entries do not exist
    @property
    def syntype_available(self) -> bool:
        """
        Synaptic types are available as KnossosDataset. Will be used during the
        matrix generation.

        Returns:
            Value stored at the config.yml file.
        """
        try:
            if self.entries['dataset']['syntype_avail'] is None:
                raise KeyError
            return self.entries['dataset']['syntype_avail']
        except KeyError:
            return True

    @property
    def use_large_fov_views_ct(self) -> bool:
        """
        Use views with large field of view for cell type prediction.

        Returns:
            Value stored at the config.yml file.
        """
        try:
            if self.entries['views']['use_large_fov_views_ct'] is None:
                raise KeyError
            return self.entries['views']['use_large_fov_views_ct']
        except KeyError:
            return False

    @property
    def use_new_renderings_locs(self) -> bool:
        """
        Use new rendering locations which are faster to computed and are located
        closer to the neuron surface.

        Returns:
            Value stored at the config.yml file.
        """
        try:
            if self.entries['views']['use_new_renderings_locs'] is None:
                raise KeyError
            return self.entries['views']['use_new_renderings_locs']
        except KeyError:
            return False

    @property
    def use_new_meshing(self) -> bool:
        """
        Use new, dense meshing (``zmesh``) computed distributed on 3D sub-cubes.
        If ``False`` meshes are computed sparsely, i.e. per object/supervoxel.

        Returns:
            Value stored at the config.yml file.
        """
        try:
            if self.entries['meshes']['use_new_meshing'] is None:
                raise KeyError
            return self.entries['meshes']['use_new_meshing']
        except KeyError:
            return False

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
        try:
            if self['paths']['use_new_subfold'] is not None:
                return self['paths']['use_new_subfold']
            else:
                raise KeyError
        except KeyError:
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
        try:
            return self.entries['cell_objects']['asym_label']
        except KeyError:
            return None

    @property
    def sym_label(self) -> Optional[int]:
        try:
            return self.entries['cell_objects']['sym_label']
        except KeyError:
            return None


def generate_default_conf(working_dir: str, scaling: Union[Tuple, np.ndarray],
                          syntype_avail: bool = True,
                          use_large_fov_views_ct: bool = False,
                          allow_skel_gen: bool = False,
                          use_new_renderings_locs: bool = True,
                          kd_seg: Optional[str] = None, kd_sym: Optional[str] = None,
                          kd_asym: Optional[str] = None,
                          kd_sj: Optional[str] = None,  kd_mi: Optional[str] = None,
                          kd_vc: Optional[str] = None, init_rag_p: str = "",
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
        # General properties of the data set
        scaling: [1, 1, 1]

        # File system, 'FS' is currently the only supported option
        backend: "FS"

        # OpenGL platform: 'egl' (GPU support) or 'osmesa' (CPU rendering)
        pyopengl_platform: 'egl'

        existing_cell_organelles: ['mi', 'sj', 'vc']
        syntype_avail:

        # Compute backend: 'QSUB', 'SLURM', None
        batch_proc_system: 'SLURM'  # If None, fall-back is single node multiprocessing

        # the here defined parameters
        batch_pe: 'default'
        batch_queue: 'all.q'

        mem_per_node: 128000  # in MB
        ncores_per_node: 16
        ngpus_per_node: 1
        nnodes_total: 1

        # --------- LOGGING
        # 'None' disables logging of SyConn modules (e.g. proc, handler, ...) to files.
        # Logs of executed scripts (syconn/scripts) will be stored at the
        # working directory + '/logs/' nonetheless.
        default_log_dir:
        log_level: 10  # INFO: 20, DEBUG: 10
        # file logging for individual modules, and per job. Only use in case of
        # debugging with single core processing. Logs for scripts are located in 'SyConn/scripts/'
        # will be stored at wd + '/logs/'.
        disable_file_logging: True

        # File locking - deprecated.
        disable_locking: False

        # Data paths
        paths:
          kd_seg:
          kd_sym:
          kd_asym:
          kd_sj:
          kd_vc:
          kd_mi:
          init_rag:
          use_new_subfold:

        # (Super-)SegmentationDataset versions
        versions:
          sv: 0
          vc: 0
          sj: 0
          syn: 0
          syn_ssv: 0
          mi: 0
          ssv: 0
          ax_gt: 0
          cs: 0

        # Cell object properties
        cell_objects:
          # threshold applied during object extraction
          min_obj_vx:
            mi: 100
            sj: 100
            vc: 100
            sv: 1  # all cell supervoxels are extracted
            cs: 10  # contact sites tend to be small
            syn: 10  # these are overlayed with contact sites and therefore tend to be small
            syn_ssv: 100 # minimum number of voxel for synapses in SSVs

          lower_mapping_ratios:
            mi: 0.5
            sj: 0.1
            vc: 0.5

          upper_mapping_ratios:
            mi: 1.
            sj: 0.9
            vc: 1.

          # size threshold (in voxels) applied when mapping them to cells
          sizethresholds:
            mi: 2786
            sj: 498
            vc: 1584

          probathresholds:
            mi: 0.428571429
            sj: 0.19047619
            vc: 0.285714286

          # Hook for morphological operations from scipy.ndimage applied during object extraction.
          # e.g. {'sj': ['binary_closing', 'binary_opening'], 'mi': [], 'sv': []}
          extract_morph_op:
            mi: []
            sj: []
            vc: []
            sv: []  # these are the cell supervoxels

          # bounding box criteria for mapping mitochondria objects
          thresh_mi_bbd_mapping: 25000  # bounding box diagonal in NM

          # --------- CONTACT SITE AND SYNAPSE PARAMETERS
          # Synaptic junction bounding box diagonal threshold in nm; objects above will
          # not be used during `syn_gen_via_cset`
          thresh_sj_bbd_syngen: 25000  # bounding box diagonal in NM
          # used for agglomerating 'syn' objects (cell supervoxel-based synapse fragments)
          # into 'syn_ssv'
          cs_gap_nm: 250
          cs_filtersize: [13, 13, 7]
          cs_nclosings: 7
          # Parameters of agglomerated synapses 'syn_ssv'
          # mapping parameters in 'map_objects_to_synssv'; assignment of cellular
          # organelles to syn_ssv
          max_vx_dist_nm: 2000
          max_rep_coord_dist_nm: 4000
          # RFC probability used for classifying whether syn or not
          thresh_synssv_proba: 0.5
          # > sym_thresh will be assigned synaptic sign -1 (inhibitory) and <= will be
          # (1, excitatory)
          sym_thresh: 0.225
          # labels are None by default
          asym_label:
          sym_label:

        meshes:
          allow_mesh_gen_cells:
          use_new_meshing:

          downsampling:
            sv: [4, 4, 2]
            sj: [2, 2, 1]
            vc: [4, 4, 2]
            mi: [8, 8, 4]
            cs: [2, 2, 1]
            syn_ssv: [2, 2, 1]

          closings:
            sv: 0
            s: 0
            vc: 0
            mi: 0
            cs: 0
            syn_ssv: 0

          mesh_min_obj_vx: 100  # adapt to size threshold

          meshing_props:
            normals: False  # True
            simplification_factor: 300
            max_simplification_error: 40  # in nm

        skeleton:
          allow_skel_gen: False
          feature_context_rfc: # in nm
            axoness: 8000
            spiness: 1000

        views:
          use_large_fov_views_ct:
          use_new_renderings_locs:
          nb_views: 2  # used for default view rendering (glia separation, spine detection)

        glia:
          prior_glia_removal: True
          # min. connected component size of glia nodes/SV after thresholding glia proba
          min_cc_size_ssv: 8000  # in nm; L1-norm on vertex bounding box

          # Threshold for glia classification
          glia_thresh: 0.161489
          # number of sv used during local rendering. The total number of SV used are
          # subcc_size_big_ssv + 2*(subcc_chunk_size_big_ssv-1)
          subcc_size_big_ssv: 35
          rendering_max_nb_sv: 5000
          # number of SV for which views are rendered in one pass
          subcc_chunk_size_big_ssv: 9

        # --------- SPINE PARAMETERS
        spines:
          min_spine_cc_size: 10
          min_edge_dist_spine_graph: 110
          gt_path_spineseg: '/wholebrain/scratch/areaxfs3/ssv_spgt/spgt_semseg/'

          # mapping parameters of the semantic segmentation prediction to the cell mesh
          # Note: ``k>0`` means that the predictions are propagated to unpredicted and backround labels
          # via nearest neighbors.
          semseg2mesh_spines:
            semseg_key: "spiness"
            force_recompute: True
            k: 0

          # mapping of vertex labels to skeleton nodes; ignore labels 4 (background)
          # and 5 (unpredicted), use labels of the k-nearest vertices
          semseg2coords_spines:
            k: 50
            ds_vertices: 1
            ignore_labels: [4, 5]


        compartments:
          dist_axoness_averaging: 10000  # also used for myelin averaging
          gt_path_axonseg: '/wholebrain/scratch/areaxfs3/ssv_semsegaxoness/all_bouton_data/'

          # `k=0` will not map predictions to unpredicted vertices -> faster
          # `k` is the parameter used in `semseg2mesh`
          view_properties_semsegax:
            verbose: False
            ws: [1024, 512]
            nb_views: 3
            comp_window: 40960  # in NM
            semseg_key: 'axoness'
            k: 0
          # mapping of vertex labels to skeleton nodes; ignore labels 5 (background)
          # and 6 (unpredicted), use labels of the k-nearest vertices
          map_properties_semsegax:
            k: 50
            ds_vertices: 1
            ignore_labels: [5, 6]


        celltype:
          view_properties_large:
            verbose: False
            ws: [512, 512]
            nb_views_render: 6
            comp_window: 40960
            nb_views_model: 4

        # --------- MORPHOLOGY EMBEDDING
        tcmn:
          ndim_embedding: 10


    Args:
        working_dir: Folder of the working directory.
        scaling: Voxel size in NM.
        syntype_avail: If True, synapse objects will contain additional type
            property (symmetric vs asymmetric).
        use_large_fov_views_ct: If True, uses on-the-fly, large view renderings
            for predicting cell types.
        allow_skel_gen: If True, allow cell skeleton generation from rendering
            locations (inaccurate).
        use_new_renderings_locs: If True, uses new heuristic for generating
            rendering locations.
        kd_seg: Path to the KnossosDataset which contains the cell segmentation.
        kd_sym: Path to the symmetric type prediction.
        kd_asym: Path to the asymmetric type prediction.
        kd_sj: Path to the synaptic junction predictions.
        kd_mi: Path to the mitochondria predictions.
        kd_vc: Path to the vesicle cloud predictions.
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

    default_conf = Config(os.path.split(os.path.abspath(__file__))[0])
    entries = default_conf.entries
    entries['paths']['kd_seg'] = kd_seg
    entries['paths']['kd_sym'] = kd_sym
    entries['paths']['kd_asym'] = kd_asym
    entries['paths']['kd_sj'] = kd_sj
    entries['paths']['kd_vc'] = kd_vc
    entries['paths']['kd_mi'] = kd_mi
    entries['paths']['init_rag'] = init_rag_p
    entries['paths']['use_new_subfold'] = use_new_subfold
    if type(scaling) is np.ndarray:
        scaling = scaling.tolist()
    entries['scaling'] = scaling
    entries['version'] = default_conf.version()
    entries['syntype_avail'] = syntype_avail

    entries['meshes']['allow_mesh_gen_cells'] = allow_mesh_gen_cells
    entries['meshes']['use_new_meshing'] = use_new_meshing

    entries['skeleton']['allow_skel_gen'] = allow_skel_gen

    entries['views']['use_large_fov_views_ct'] = use_large_fov_views_ct
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

        delta = datetime.datetime.fromtimestamp(record.relativeCreated/1000.0) - \
                datetime.datetime.fromtimestamp(last/1000.0)

        record.relative = '{0:.1f}'.format(delta.seconds / 60.)

        self.last = record.relativeCreated
        return True
