# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max-Planck-Institute for Medical Research, Heidelberg, Germany
# Authors: Sven Dorkenwald, Philipp Schubert, Jörgen Kornfeld

from .mp_utils import *
from ..handler.logger import initialize_logging
log_mp = initialize_logging('mp')
log_mp.warn("'shared_mem' module is outdated. Please use 'mp_utils'.")