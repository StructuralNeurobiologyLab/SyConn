# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Philipp Schubert, Joergen Kornfeld

from syconn.analysis import reconnect_proofreading as repro

if __name__ == "__main__":
    repro.update_RAG_with_reconnects()