# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Philipp Schubert, Sven Dorkenwald, Joergen Kornfeld
from syconn.proc.skel_based_classifier import SkelClassifier
from syconn.config.global_params import wd, NCORES_PER_NODE


if __name__ == "__main__":
    # # run skeleton feature extraction # Not needed anymore, will be kept in
    # case skeleton features should remain a feature of SyConn
    sbc = SkelClassifier("axoness", working_dir=wd)
    ft_context = [1000, 2000, 4000, 8000, 12000]
    sbc.generate_data(feature_contexts_nm=ft_context, nb_cpus=NCORES_PER_NODE)
    sbc.classifier_production(ft_context, nb_cpus=NCORES_PER_NODE)
