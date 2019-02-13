# -*- coding: utf-8 -*-
# Max-Planck-Institute of Neurobiology, Munich, Germany
# Authors: Philipp Schubert, Sven Dorkenwald, Jörgen Kornfeld
# SyConn - Synaptic connectivity inference toolkit
# SyConn Gate is a thin flask server that allows clients
# over a RESTful HTTP API to interact with a SyConn dataset

from ..handler.logger import log_main
log_gate = log_main

__all__ = ['log_gate']