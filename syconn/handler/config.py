# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Sven Dorkenwald, Philipp Schubert, Joergen Kornfeld
from configobj import ConfigObj
from typing import Tuple, Optional, Union, Dict, Any, List
import sys
from validate import Validator
import logging
import coloredlogs
import datetime
import pwd
import numpy as np
from termcolor import colored
import os
from .. import global_params

__all__ = ['DynConfig', 'get_default_conf_str', 'initialize_logging']


class Config(object):
    """
    Basic config object base on the package ``configobj``.
    """
    def __init__(self, working_dir, validate=True, verbose=True):
        self._entries = {}
        self._working_dir = working_dir
        self.initialized = False
        self.log_main = get_main_log()
        if self._working_dir is not None and len(self._working_dir) > 0:
            self.parse_config(validate=validate)
            self.initialized = True
            if verbose:
                self.log_main.info("Initialized stdout logging (level: {}). "
                                   "Current working directory:"
                                   " ".format(global_params.log_level) +
                                   colored("'{}'".format(working_dir), 'red'))

    @property
    def entries(self) -> Any:
        """
        Entries stored in the config.ini file.

        Returns:
            All entries.
        """
        if not self.initialized:
            raise ValueError('Config object was not initialized. "entries" '
                             'are not available.')
        return self._entries

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
            Path to config file (``config.ini``).
        """
        return self.working_dir + "/config.ini"

    @property
    def path_configspec(self) -> str:
        """
        Returns:
            Path to specification file for all config. parameters
            (``configspec.ini``).
        """
        return self.working_dir + "/configspec.ini"

    @property
    def is_valid(self) -> bool:
        """
        Valid configuration file.

        Returns:
            ``True`` of any section could be retrieved from the file,
            ``False`` otherwise.
        """
        return len(self.sections) > 0

    @property
    def config_exists(self):
        """
        Returns:
            ``True`` if config file exists,
            ``False`` otherwise.
        """
        return os.path.exists(self.path_config)

    @property
    def configspec_exists(self):
        """
        Returns:
            ``True`` if config. specification file exists,
            ``False`` otherwise.
        """
        return os.path.exists(self.path_configspec)

    @property
    def sections(self) -> List[str]:
        """
        Returns:
            Keys to all sections present in the config file.
        """
        return list(self.entries.keys())

    def parse_config(self, validate: bool = True):
        """
        Reads the content stored in the config file.

        Args:
            validate: Checks if file is a valid config.
        """
        assert self.path_config
        assert self.path_configspec or not validate

        if validate:
            configspec = ConfigObj(self.path_configspec, list_values=False,
                                   _inspec=True)
            config = ConfigObj(self.path_config, configspec=configspec)

            if config.validate(Validator()):
                self._entries = config
            else:
                self.log_main.error('ERROR: Could not parse config at '
                                    '{}.'.format(self.path_config))
        else:
            self._entries = ConfigObj(self.path_config)

    def write_config(self):
        # TODO: implement string conversion
        raise NotImplementedError
        # with open(self.working_dir + 'config.ini', 'w') as f:
            # f.write(config_str)
        # with open(self.working_dir + 'configspec.ini', 'w') as f:
            # f.write(configspec_str)


# TODO: add generic parser method for initial RAG and handle case without glia-splitting, refactor RAG path handling
# TODO:(cover case if glia removal was not performed, change resulting rag paths after glia removal from 'glia' to 'rag'
class DynConfig(Config):
    """
    Enables dynamic and SyConn-wide update of working directory 'wd' and provides an
    interface to all working directory dependent parameters.

    Examples:
        To initialize a working directory at the beginning of your script, run::

            from syconn import global_params
            global_params.wd = '~/SyConn/example_cube1/'
            cfg = global_params.config  # this is the `DynConfig` object

    """
    def __init__(self, wd: Optional[str] = None):
        """
        Args:
            wd: Path to working directory
        """
        if wd is None:
            wd = global_params.wd
            verbose = True
        else:
            verbose = False
        super().__init__(wd, verbose=verbose)

    def _check_actuality(self):
        """
        Checks os.environ and global_params and triggers an update if the therein specified WD is not the same as
         `self.working dir`.
        """
        # first check if working directory was set in environ, else check if it was changed in memory.
        if 'syconn_wd' in os.environ and os.environ['syconn_wd'] is not None and \
            len(os.environ['syconn_wd']) > 0 and os.environ['syconn_wd'] != "None":

            if super().working_dir != os.environ['syconn_wd']:
                super().__init__(os.environ['syconn_wd'])
        elif (global_params.wd is not None) and (len(global_params.wd) > 0) and \
                (global_params.wd != "None") and (super().working_dir != global_params.wd):
            super().__init__(global_params.wd)

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
        return self.entries['Paths']['kd_seg']

    @property
    def kd_sym_path(self) -> str:
        """
        Returns:
            Path to synaptic sym. type probability map stored as ``KnossosDataset``.
        """
        return self.entries['Paths']['kd_sym']

    @property
    def kd_asym_path(self) -> str:
        """
        Returns:
            Path to synaptic asym. type probability map stored as ``KnossosDataset``.
        """
        return self.entries['Paths']['kd_asym']

    @property
    def kd_sj_path(self) -> str:
        """
        Returns:
            Path to synaptic junction probability map or binary predictions stored as
            ``KnossosDataset``.
        """
        return self.entries['Paths']['kd_sj']

    @property
    def kd_vc_path(self) -> str:
        """
        Returns:
            Path to vesicle cloud probability map or binary predictions stored as
            ``KnossosDataset``.
        """
        return self.entries['Paths']['kd_vc']

    @property
    def kd_mi_path(self) -> str:
        """
        Returns:
            Path to mitochondria probability map or binary predictions stored as
            ``KnossosDataset``.
        """
        return self.entries['Paths']['kd_mi']

    @property
    def kd_organells_paths(self) -> Dict[str, str]:
        """
        KDs of subcell. organelle probability maps

        Returns:
            Dictionary containg the paths to ``KnossosDataset`` of available
            cellular containing ``global_params.existing_cell_organelles``.
        """
        path_dict = {k: self.entries['Paths']['kd_{}'.format(k)] for k in
                     global_params.existing_cell_organelles}
        # path_dict = {
        #     'kd_sj': self.kd_sj_path,
        #     'kd_vc': self.kd_vc_path,
        #     'kd_mi': self.kd_mi_path
        # }
        return path_dict

    @property
    def kd_organelle_seg_paths(self)-> Dict[str, str]:
        """
        KDs of subcell. organelle segmentations.

        Returns:
            Dictionary containing the paths to ``KnossosDataset`` of available
            cellular organelles ``global_params.existing_cell_organelles``.
        """
        path_dict = {k: "{}/knossosdatasets/{}_seg/".format(self.working_dir, k) for k in
                     global_params.existing_cell_organelles}
        # path_dict = {
        #     'kd_sj': self.kd_sj_path,
        #     'kd_vc': self.kd_vc_path,
        #     'kd_mi': self.kd_mi_path
        # }
        return path_dict

    @property
    def temp_path(self) -> str:
        """

        Returns:
            Path to temporary directory used to store data caches.
        """
        # return "/tmp/{}_syconn/".format(pwd.getpwuid(os.getuid()).pw_name)
        return "{}/tmp/".format(self.working_dir)

    @property
    # TODO: Not necessarily needed anymore
    def py36path(self) -> str:
        """
        Deprecated.

        Returns:
            Path to python3 interpreter.
        """
        if len(self.entries['Paths']['py36path']) != 0:
            return self.entries['Paths']['py36path']  # python 3.6 path is available
        else:  # python 3.6 path is not set, check current python
            if sys.version_info[0] == 3 and sys.version_info[1] == 6:
                return sys.executable
        raise RuntimeError('Python 3.6 is not available. Please install SyConn'
                           ' within python 3.6 or specify "py36path" in config.ini!')

    # TODO: Work-in usage of init_rag_path
    @property
    def init_rag_path(self) -> str:
        """
        Returns:
            Path to initial RAG.
        """
        self._check_actuality()
        p = self.entries['Paths']['init_rag']
        if len(p) == 0:
            p = self.working_dir + "rag.bz2"
        return p

    @property
    def pruned_rag_path(self) -> str:
        """
        Returns:
            Path to pruned RAG after glia separation.
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
        return self.model_dir + '/tCMN/'

    @property
    def mpath_tnet_large(self) -> str:
        """
        Trained on a large field of view.

        Returns:
            Path to tCMN - a decoder network of cell morphology trained via
            triplet loss.
        """
        return self.model_dir + '/tCMN_large/'

    @property
    def mpath_spiness(self) -> str:
        """
        Returns:
            Path to model trained on detecting spine head, neck, dendritic shaft,
            and ``other`` (soma and axon) via 2D projections (-> semantic segmentation).
        """
        return self.model_dir + '/spiness/'

    @property
    def mpath_axonsem(self) -> str:
        """
        Returns:
            Path to model trained on detecting axon, terminal boutons and en-passant,
            dendrites and somata via 2D projections (-> semantic segmentation).
        """
        return self.model_dir + '/axoness_semseg/'

    @property
    def mpath_celltype(self) -> str:
        """
        Deprecated.
        """
        return self.model_dir + '/celltype/celltype.mdl'

    @property
    def mpath_celltype_e3(self) -> str:
        """
        Returns:
            Path to model trained on prediction cell types from multi-view sets.
        """
        return self.model_dir + '/celltype_e3/'

    @property
    def mpath_celltype_large_e3(self) -> str:
        """
        Trained on a large field of view.

        Returns:
            Path to model trained on prediction cell types from multi-view sets.
        """
        return self.model_dir + '/celltype_large_e3/'

    @property
    def mpath_axoness(self) -> str:
        """
        Deprecated.
        """
        return self.model_dir + '/axoness/axoness.mdl'

    @property
    def mpath_axoness_e3(self) -> str:
        """
        Deprecated.
        """
        return self.model_dir + '/axoness_e3/'

    @property
    def mpath_glia(self) -> str:
        """
        Deprecated.
        """
        return self.model_dir + '/glia/glia.mdl'

    @property
    def mpath_glia_e3(self) -> str:
        """
        Trained on a large field of view.

        Returns:
            Path to model trained on prediction local 2D projections into glia
            vs. neuron (img2scalar).
        """
        return self.model_dir + '/glia_e3/'

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
            return self.entries['Mesh']['allow_mesh_gen_cells']
        except KeyError:
            return False

    @property
    def allow_skel_gen(self) -> bool:
        """
        Controls whether cell supervoxel skeletons are provided a priori or
        can be computed from scratch. Currently this is done via a naive sampling
        procedure.

        Returns:
            Value stored at the config.ini file.
        """
        return self.entries['Skeleton']['allow_skel_gen']

    # New config attributes, enable backwards compat. in case these entries do not exist
    @property
    def syntype_available(self) -> bool:
        """
        Synaptic types are available as KnossosDataset. Will be used during the
        matrix generation.

        Returns:
            Value stored at the config.ini file.
        """
        try:
            return self.entries['Dataset']['syntype_avail']
        except KeyError:
            return True

    @property
    def use_large_fov_views_ct(self) -> bool:
        """
        Use views with large field of view for cell type prediction.

        Returns:
            Value stored at the config.ini file.
        """
        try:
            return self.entries['Views']['use_large_fov_views_ct']
        except KeyError:
            return True

    @property
    def use_new_renderings_locs(self) -> bool:
        """
        Use new rendering locations which are faster to computed and are located
        closer to the neuron surface.

        Returns:
            Value stored at the config.ini file.
        """
        try:
            return self.entries['Views']['use_new_renderings_locs']
        except KeyError:
            return False

    @property
    def use_new_meshing(self) -> bool:
        """
        Use new, dense meshing (``zmesh``) computed distributed on 3D sub-cubes.
        If ``False`` meshes are computed sparsely, i.e. per object/supervoxel.

        Returns:
            Value stored at the config.ini file.
        """
        try:
            return self.entries['Mesh']['use_new_meshing']
        except KeyError:
            return False

    @property
    def qsub_work_folder(self) -> str:
        """
        Directory where intermediate batchjob results are stored.

        Returns:
            Path to directory.
        """
        return "%s/%s/" % (global_params.config.working_dir, # self.temp_path,
                           global_params.BATCH_PROC_SYSTEM)

    @property
    def prior_glia_removal(self) -> bool:
        """
        If ``True`` glia separation procedure will be initiated to create a
        pruned RAG (see :attr:`~syconn/handler.config.DynConfig.pruned_rag_path`).

        Returns:
            Value stored at the config.ini file.
        """
        try:
            return self.entries['Glia']['prior_glia_removal']
        except KeyError:
            return True

    @property
    def use_new_subfold(self) -> bool:
        """
        Use new subfolder hierarchy where objects with similar IDs are stored
        in the same file.

        Returns:
            Value stored at the config.ini file.
        """
        try:
            return self.entries['Paths']['use_new_subfold']
        except KeyError:
            return False


def get_default_conf_str(example_wd: str, scaling: Union[Tuple, np.ndarray],
                         py36path: str = "", syntype_avail: bool = True,
                         use_large_fov_views_ct: bool = False,
                         allow_skel_gen: bool = True,
                         use_new_renderings_locs: bool = False,
                         kd_seg: Optional[str] = None, kd_sym: Optional[str] = None,
                         kd_asym: Optional[str] = None,
                         kd_sj: Optional[str] = None,  kd_mi: Optional[str] = None,
                         kd_vc: Optional[str] = None, init_rag_p: str = "",
                         prior_glia_removal: bool = False,
                         use_new_meshing: bool = False,
                         allow_mesh_gen_cells: bool = True,
                         use_new_subfold: bool = True) -> Tuple[str, str]:
    """
    Default SyConn config and variable type specifications. Paths to ``KnossosDatasets``
    containing various predictions, prob. maps and segmentations have to be given depending on
    what specifically is going to be processed. See ``SyConn/scripts/example_run/start.py``
    for an example.
    ``init_rag`` can be set specifically in the config-file which is optional.
    By default it is set to ``init_rag = working_dir + "rag.bz2"``, which then
    requires manual generation of the file, see ``SyConn/scripts/example_run/start.py``.
    The parameter ``py36path`` is currently not in use.

    Examples:
        Example content of the `config.ini` file::

            [Versions]
                sv = 0
                vc = 0
                sj = 0
                syn = 0
                syn_ssv = 0
                mi = 0
                ssv = 0
                ax_gt = 0
                cs = 0

            [Paths]
                kd_seg = ~/SyConn/example_cube1/knossosdatasets/seg/
                kd_sym = ~/SyConn/example_cube1/knossosdatasets/sym/
                kd_asym = ~/SyConn/example_cube1/knossosdatasets/asym/
                kd_sj = ~/SyConn/example_cube1/knossosdatasets/sj/
                kd_vc = ~/SyConn/example_cube1/knossosdatasets/vc/
                kd_mi = ~/SyConn/example_cube1/knossosdatasets/mi/
                init_rag =
                py36path =
                use_new_subfold = True

            [Dataset]
                scaling = 10, 10, 20
                syntype_avail = True

            [LowerMappingRatios]
                mi = 0.5
                sj = 0.1
                vc = 0.5

            [UpperMappingRatios]
                mi = 1.
                sj = 0.9
                vc = 1.

            [Sizethresholds]
                mi = 2786
                sj = 498
                vc = 1584

            [Probathresholds]
                mi = 0.428571429
                sj = 0.19047619
                vc = 0.285714286

            [Mesh]
                allow_mesh_gen_cells = True
                use_new_meshing = True

            [Skeleton]
                allow_skel_gen = True

            [Views]
                use_large_fov_views_ct = False
                use_new_renderings_locs = True

            [Glia]
                prior_glia_removal = True

    Args:
        example_wd:
        scaling:
        py36path:
        syntype_avail:
        use_large_fov_views_ct:
        allow_skel_gen:
        use_new_renderings_locs:
        kd_seg:
        kd_sym:
        kd_asym:
        kd_sj:
        kd_mi:
        kd_vc:
        init_rag_p:
        prior_glia_removal:
        use_new_meshing:
        allow_mesh_gen_cells:
        use_new_subfold:

    Returns:
        Content of config.ini and configspec.ini
    """
    if kd_seg is None:
        kd_seg = example_wd + 'knossosdatasets/seg/'
    if kd_sym is None:
        kd_sym = example_wd + 'knossosdatasets/sym/'
    if kd_asym is None:
        kd_asym = example_wd + 'knossosdatasets/asym/'
    if kd_sj is None:
        kd_sj = example_wd + 'knossosdatasets/sj/'
    if kd_mi is None:
        kd_mi = example_wd + 'knossosdatasets/mi/'
    if kd_vc is None:
        kd_vc = example_wd + 'knossosdatasets/vc/'
    config_str = """[Versions]
sv = 0
vc = 0
sj = 0
syn = 0
syn_ssv = 0
mi = 0
ssv = 0
ax_gt = 0
cs = 0

[Paths]
kd_seg = {}
kd_sym = {}
kd_asym = {}
kd_sj = {}
kd_vc = {}
kd_mi = {}
init_rag = {}
py36path = {}
use_new_subfold = {}

[Dataset]
scaling = {}, {}, {}
syntype_avail = {}

[LowerMappingRatios]
mi = 0.5
sj = 0.1
vc = 0.5

[UpperMappingRatios]
mi = 1.
sj = 0.9
vc = 1.

[Sizethresholds]
mi = 2786
sj = 498
vc = 1584

[Probathresholds]
mi = 0.428571429
sj = 0.19047619
vc = 0.285714286

[Mesh]
allow_mesh_gen_cells = {}
use_new_meshing = {}

[Skeleton]
allow_skel_gen = {}

[Views]
use_large_fov_views_ct = {}
use_new_renderings_locs = {}

[Glia]
prior_glia_removal = {}
    """.format(kd_seg, kd_sym, kd_asym, kd_sj, kd_vc, kd_mi, init_rag_p,
               py36path, use_new_subfold, scaling[0], scaling[1], scaling[2],
               str(syntype_avail), str(allow_mesh_gen_cells), str(use_new_meshing),
               str(allow_skel_gen), str(use_large_fov_views_ct),
               str(use_new_renderings_locs), str(prior_glia_removal))

    configspec_str = """
[Versions]
__many__ = string

[Paths]
__many__ = string
use_new_subfold = boolean

[Dataset]
scaling = float_list(min=3, max=3)
syntype_avail = boolean

[LowerMappingRatios]
__many__ = float

[UpperMappingRatios]
__many__ = float

[Sizethresholds]
__many__ = integer

[Probathresholds]
__many__ = float

[Mesh]
allow_mesh_gen_cells = boolean
use_new_meshing = boolean

[Skeleton]
allow_skel_gen = boolean

[Views]
use_large_fov_views_ct = boolean
use_new_renderings_locs = boolean

[Glia]
prior_glia_removal = boolean
"""
    return config_str, configspec_str


def get_main_log():
    """
    Initialize main log.

    Returns:
        Main log.

    """
    logger = logging.getLogger('syconn')
    coloredlogs.install(level=global_params.log_level, logger=logger)
    level = logging.getLevelName(global_params.log_level)
    logger.setLevel(level)

    if not global_params.DISABLE_FILE_LOGGING:
        # create file handler which logs even debug messages
        log_dir = os.path.expanduser('~') + "/SyConn/logs/"

        os.makedirs(log_dir, exist_ok=True)
        fh = logging.FileHandler(log_dir + 'syconn.log')
        fh.setLevel(level)

        # add the handlers to logger
        if os.path.isfile(log_dir + 'syconn.log'):
            os.remove(log_dir + 'syconn.log')
        logger.addHandler(fh)
        logger.info("Initialized file logging. Log-files are stored at"
                    " {}.".format(log_dir))
    return logger


def initialize_logging(log_name: str, log_dir: Optional[str] = None,
                       overwrite: bool = True):
    """
    Logger for each package module. For import processing steps individual
    logger can be defined (e.g. ``proc``, ``reps``).

    Args:
        log_name: Name of the logger.
        log_dir: Set log_dir specifically. Will then create a filehandler and
            ignore the state of global_params.DISABLE_FILE_LOGGING state.
        overwrite: Overwrite previous log file.


    Returns:

    """
    if log_dir is None:
        log_dir = global_params.default_log_dir
    level = global_params.log_level
    logger = logging.getLogger(log_name)
    logger.setLevel(level)
    coloredlogs.install(level=global_params.log_level, logger=logger,
                        reconfigure=False)  # True possibly leads to stderr output
    if not global_params.DISABLE_FILE_LOGGING or log_dir is not None:
        # create file handler which logs even debug messages
        if log_dir is None:
            log_dir = os.path.expanduser('~') + "/.SyConn/logs/"
        try:
            os.makedirs(log_dir, exist_ok=True)
        except TypeError:
            if not os.path.isdir(log_dir):
                os.makedirs(log_dir)
        if overwrite and os.path.isfile(log_dir + log_name + '.log'):
            os.remove(log_dir + log_name + '.log')
        # add the handlers to logger
        fh = logging.FileHandler(log_dir + log_name + ".log")
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

        delta = datetime.datetime.fromtimestamp(record.relativeCreated/1000.0) - datetime.datetime.fromtimestamp(last/1000.0)

        record.relative = '{0:.1f}'.format(delta.seconds / 60.)

        self.last = record.relativeCreated
        return True
