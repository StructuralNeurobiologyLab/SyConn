# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Sven Dorkenwald, Philipp Schubert, Joergen Kornfeld

from configobj import ConfigObj
import os
from validate import Validator


class Config(object):
    def __init__(self, working_dir, validate=True):
        self._entries = {}
        self._working_dir = working_dir
        self.parse_config(validate=validate)

    @property
    def entries(self):
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


#
def get_default_conf_str(example_wd, py36path=""):
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
    """.format(example_wd + 'knossosdatasets/seg/', example_wd + 'knossosdatasets/sym/',
               example_wd + 'knossosdatasets/asym/', example_wd + 'knossosdatasets/sj/',
               example_wd + 'knossosdatasets/vc/', example_wd + 'knossosdatasets/mi/', '', py36path)

    configspec_str = """
[Versions]
__many__ = string

[Paths]
__many__ = string

[Dataset]
scaling = float_list(min=3, max=3)

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
"""
    return config_str, configspec_str
