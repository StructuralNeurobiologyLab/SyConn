# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Philipp Schubert, Sven Dorkenwald, Joergen Kornfeld

from ..config.global_params import backend
from .base import LZ4DictBase, BTBase
from ..handler.logger import initialize_logging
# init backend
if backend == 'FS':
    StorageClass = LZ4DictBase
elif backend == 'BT':
    StorageClass = BTBase
# init log
log_backend = initialize_logging('backend')

__all__ = ['StorageClass', 'log_backend']

