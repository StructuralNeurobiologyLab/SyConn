# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Philipp Schubert, Joergen Kornfeld
from syconn.proc.sd_proc import mesh_proc_chunked
from syconn.proc.ssd_proc import mesh_proc_ssv
from syconn.config.global_params import wd

if __name__ == "__main__":
    # preprocess meshes of all objects
    mesh_proc_chunked(wd, "conn")
    mesh_proc_chunked(wd, "sj")
    mesh_proc_chunked(wd, "vc")
    mesh_proc_chunked(wd, "mi")
