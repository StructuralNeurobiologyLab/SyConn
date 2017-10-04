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
        self.entries = {}
        self._working_dir = working_dir
        self.parse_config(validate=validate)

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
        return self.entries.keys()

    def parse_config(self, validate=True):
        assert self.path_config
        assert self.path_configspec or not validate

        if validate:
            configspec = ConfigObj(self.path_configspec, list_values=False,
                                   _inspec=True)
            config = ConfigObj(self.path_config, configspec=configspec)

            if config.validate(Validator()):
                self.entries = config
        else:
            self.entries = ConfigObj(self.path_config)
