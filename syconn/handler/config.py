# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Sven Dorkenwald, Philipp Schubert, Joergen Kornfeld
from configobj import ConfigObj
import sys
from validate import Validator
import logging
import coloredlogs
from termcolor import colored
import os
from .. import global_params

__all__ = ['DynConfig', 'get_default_conf_str', 'initialize_logging']


class Config(object):
    def __init__(self, working_dir, validate=True):
        self._entries = {}
        self._working_dir = working_dir
        self.initialized = False
        self.log_main = get_main_log()
        if self._working_dir is not None and len(self._working_dir) > 0:
            self.parse_config(validate=validate)
            self.log_main.info("Initialized stdout logging (level: {}). "
                               "Current working directory:"
                               " ".format(global_params.log_level) +
                               colored("'{}'".format(working_dir), 'red'))
            self.log_main.info(""
                               "".format())
            self.initialized = True

    @property
    def entries(self):
        if not self.initialized:
            raise ValueError('Config object was not initialized. "entries" '
                             'are not available.')
        return self._entries

    @property
    def working_dir(self):
        return self._working_dir

    @property
    def path_config(self):
        return self.working_dir + "/config.ini"

    @property
    def path_configspec(self):
        return self.working_dir + "/configspec.ini"

    @property
    def is_valid(self):
        return len(self.sections) > 0

    @property
    def config_exists(self):
        return os.path.exists(self.path_config)

    @property
    def configspec_exists(self):
        return os.path.exists(self.path_configspec)

    @property
    def sections(self):
        return list(self.entries.keys())

    def parse_config(self, validate=True):
        assert self.path_config
        assert self.path_configspec or not validate

        if validate:
            configspec = ConfigObj(self.path_configspec, list_values=False,
                                   _inspec=True)
            config = ConfigObj(self.path_config, configspec=configspec)

            if config.validate(Validator()):
                self._entries = config
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
    Enables dynamic and SyConn-wide update of working directory 'wd'.
    """
    def __init__(self):
        super().__init__(global_params.wd)

    def _check_actuality(self):
        """
        Checks os.environ and global_params and triggers an update if the therein specified WD is not the same as
         `self.working dir`.
        """
        # first check if working directory was set in environ, else check if it was changed in memory.
        if 'syconn_wd' in os.environ:
            if super().working_dir != os.environ['syconn_wd']:
                super().__init__(os.environ['syconn_wd'])
        elif super().working_dir != global_params.wd:
            super().__init__(global_params.wd)

    @property
    def entries(self):
        self._check_actuality()
        return super().entries

    @property
    def working_dir(self):
        self._check_actuality()
        return super().working_dir

    @property
    def kd_seg_path(self):
        return self.entries['Paths']['kd_seg']

    @property
    def kd_sym_path(self):
        return self.entries['Paths']['kd_sym']

    @property
    def kd_asym_path(self):
        return self.entries['Paths']['kd_asym']

    @property
    def kd_sj_path(self):
        return self.entries['Paths']['kd_sj']

    @property
    def kd_vc_path(self):
        return self.entries['Paths']['kd_vc']

    @property
    def kd_mi_path(self):
        return self.entries['Paths']['kd_mi']

    @property
    # TODO: Not necessarily needed anymore
    def py36path(self):
        if len(self.entries['Paths']['py36path']) != 0:
            return self.entries['Paths']['py36path']  # python 3.6 path is available
        else:  # python 3.6 path is not set, check current python
            if sys.version_info[0] == 3 and sys.version_info[1] == 6:
                return sys.executable
        raise RuntimeError('Python 3.6 is not available. Please install SyConn within python 3.6 or specify '
                           '"py36path" in config.ini!')

    # TODO: Work-in usage of init_rag_path
    @property
    def init_rag_path(self):
        """
        # currently a mergelist/RAG of the following form is expected:
        # ID, ID
        #    .
        #    .
        # ID, ID

        Returns
        -------
        str
        """
        # self._check_actuality()
        return self.entries['Paths']['init_rag']

    # --------- CLASSIFICATION MODELS
    @property
    def model_dir(self):
        return self.working_dir + '/models/'

    @property
    def mpath_tnet(self):
        return self.model_dir + '/tCMN/'

    @property
    def mpath_tnet_large(self):  # large FoV
        return self.model_dir + '/tCMN_large/'

    @property
    def mpath_spiness(self):
        return self.model_dir + '/spiness/'

    @property
    def mpath_axonsem(self):
        """
        Semantic segmentation moder cellular compartments
        """
        return self.model_dir + '/axon_semseg/'

    @property
    def mpath_celltype(self):
        return self.model_dir + '/celltype/celltype.mdl'

    @property
    def mpath_celltype_e3(self):
        return self.model_dir + '/celltype_e3/'

    @property
    def mpath_celltype_large_e3(self):  # large FoV
        return self.model_dir + '/celltype_large_e3/'

    @property
    def mpath_axoness(self):
        return self.model_dir + '/axoness/axoness.mdl'

    @property
    def mpath_axoness_e3(self):
        return self.model_dir + '/axoness_e3/'

    @property
    def mpath_glia(self):
        return self.model_dir + '/glia/glia.mdl'

    @property
    def mpath_glia_e3(self):
        return self.model_dir + '/glia_e3/'

    @property
    def mpath_syn_rfc(self):
        return self.model_dir + '/conn_syn_rfc//rfc'

    @property
    def allow_mesh_gen_cells(self):
        return self.entries['Mesh']['allow_mesh_gen_cells']

    @property
    def allow_skel_gen(self):
        return self.entries['Skeleton']['allow_skel_gen']

    # New config attributes, enable backwards compat. in case these entries do not exist
    @property
    def syntype_available(self):
        try:
            return self.entries['Dataset']['syntype_avail']
        except KeyError:
            return True

    @property
    def use_large_fov_views_ct(self):
        try:
            return self.entries['Views']['use_large_fov_views_ct']
        except KeyError:
            return True

    @property
    def use_new_renderings_locs(self):
        try:
            return self.entries['Views']['use_new_renderings_locs']
        except KeyError:
            return False

    @property
    def qsub_work_folder(self):
        return "%s/%s/" % (global_params.config.working_dir,
                           global_params.BATCH_PROC_SYSTEM)


def get_default_conf_str(example_wd, scaling, py36path="", syntype_avail=True,
                         use_large_fov_views_ct=True, use_new_renderings_locs=True):
    """
    Default SyConn config and type specification, placed in the working directory.

    Returns
    -------
    str, str
        config.ini and configspec.ini contents
    """
    config_str = """[Versions]
sv = 0
vc = 0
sj = 0
syn = 0
syn_ssv = 0
mi = 0
ssv = 0
cs_agg = 0
ax_gt = 0

[Paths]
kd_seg = {}
kd_sym = {}
kd_asym = {}
kd_sj = {}
kd_vc = {}
kd_mi = {}
init_rag = {}
py36path = {}

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
allow_mesh_gen_cells = True

[Skeleton]
allow_skel_gen = True

[Views]
use_large_fov_views_ct = {}
use_new_renderings_locs = {}
    """.format(example_wd + 'knossosdatasets/seg/',
               example_wd + 'knossosdatasets/sym/',
               example_wd + 'knossosdatasets/asym/',
               example_wd + 'knossosdatasets/sj/',
               example_wd + 'knossosdatasets/vc/',
               example_wd + 'knossosdatasets/mi/', '',
               py36path, scaling[0], scaling[1], scaling[2],
               str(syntype_avail), str(use_large_fov_views_ct),
               str(use_new_renderings_locs))

    configspec_str = """
[Versions]
__many__ = string

[Paths]
__many__ = string

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

[Skeleton]
allow_skel_gen = boolean

[Views]
use_large_fov_views_ct = boolean
use_new_renderings_locs = boolean
"""
    return config_str, configspec_str


def get_main_log():
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


def initialize_logging(log_name, log_dir=None, overwrite=True):
    """
    Logger for each package module. For import processing steps individual
    logger can be defined (e.g. multiviews, skeleton)
    Parameters
    ----------
    log_name : str
        Name of logger
    log_dir : str
        Set log_dir specifically. Will then create a filehandler and ignore the
         state of global_params.DISABLE_FILE_LOGGING state.
    overwrite : bool
        Previous log file will be overwritten

    Returns
    -------

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
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger
