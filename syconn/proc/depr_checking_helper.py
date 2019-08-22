# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Philipp Schubert, Joergen Kornfeld
import glob


def find_missing_overlaycubes_thread(args):
    paths = args[0]

    m_paths = []
    for path in paths:
        if len(glob.glob(path + "/*seg.sz.zip")) == 0:
            m_paths.append(path)
    return m_paths
