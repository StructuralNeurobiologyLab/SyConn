# -*- coding: utf-8 -*-
# Max-Planck-Institute of Neurobiology, Munich, Germany
# Authors: Philipp Schubert, Sven Dorkenwald, Jörgen Kornfeld
# SyConn - Synaptic connectivity inference toolkit
# SyConn Gate is a thin flask server that allows clients
# over a RESTful HTTP API to interact with a SyConn dataset

from ..handler.logger import initialize_logging
log_gate = initialize_logging('gate')

__all__ = ['log_gate']