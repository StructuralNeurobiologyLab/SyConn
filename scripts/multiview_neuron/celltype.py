# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Philipp Schubert, Sven Dorkenwald, Joergen Kornfeld

from syconn.exec import exec_multiview


# ~2.5h with 24 gpus
if __name__ == "__main__":
    exec_multiview.run_celltype_prediction()
