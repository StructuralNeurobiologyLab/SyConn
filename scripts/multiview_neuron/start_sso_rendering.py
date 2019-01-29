# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max-Planck-Institute of Neurobiology, Munich, Germany
# Authors: Philipp Schubert, Joergen Kornfeld

from syconn.exec import exec_multiview

# Using OSMesa:
# ~5h for 'small' SSV (<5000)
# ~24h for one huge SSV (14k SV, most likely including huge SV)
if __name__ == "__main__":
    exec_multiview.run_neuron_rendering()
