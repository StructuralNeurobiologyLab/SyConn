# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Philipp Schubert, Sven Dorkenwald, Joergen Kornfeld
from syconn.proc.skel_based_classifier import SkelClassifier
from syconn.config.global_params import wd
from syconn.handler.basics import load_pkl2obj

if __name__ == "__main__":
    # example usage for
    sbc = SkelClassifier("axoness", working_dir=wd)
    ft_context = [1000, 2000, 4000, 8000, 12000]
    sbc.generate_data(feature_contexts_nm=ft_context, nb_cpus=20)
    sbc.classifier_production(ft_context, nb_cpus=20)