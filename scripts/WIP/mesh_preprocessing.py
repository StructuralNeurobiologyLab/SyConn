# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Philipp Schubert, Joergen Kornfeld
from syconn.proc.sd_proc import mesh_proc_chunked
from syconn import global_params

if __name__ == "__main__":
    # preprocess meshes of all objects
    mesh_proc_chunked(global_params.config.working_dir, "conn")
    mesh_proc_chunked(global_params.config.working_dir, "sj")
    mesh_proc_chunked(global_params.config.working_dir, "vc")
    mesh_proc_chunked(global_params.config.working_dir, "mi")
