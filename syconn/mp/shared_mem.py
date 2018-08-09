# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max-Planck-Institute of Neurobiology, Munich, Germany
# Authors: Philipp Schubert, Sven Dorkenwald, Jörgen Kornfeld

from ..mp import log_mp
log_mp.warn("'shared_mem' module is outdated. Please use 'mp_utils'.")
from .mp_utils import *

