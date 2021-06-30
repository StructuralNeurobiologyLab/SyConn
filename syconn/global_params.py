# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Philipp Schubert, Joergen Kornfeld

import warnings
from .handler.config import DynConfig

# TODO: update knossos_utils usages
# ignore knossos_utils warnings
warnings.filterwarnings("ignore", message=".*You are using implicit channel selection.*")
warnings.filterwarnings("ignore", message=".*You are initializing a KnossosDataset from a path.*")
warnings.filterwarnings("ignore", message=".*dataset.value has been deprecated.*")
# ignore "OpenGL/images.py:142: DeprecationWarning: tostring() is deprecated. Use tobytes() instead"
warnings.filterwarnings('ignore', message=".*tostring() is deprecated.*")
wd = "/ssdscratch/songbird/j0251/rag_flat_Jan2019_v3/"
config = DynConfig()
