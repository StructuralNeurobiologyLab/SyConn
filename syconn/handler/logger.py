# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max-Planck-Institute of Neurobiology, Munich, Germany
# Authors: Philipp Schubert, Joergen Kornfeld
import logging
from ..config.global_params import wd
import os

def get_main_log():
    logger = logging.getLogger('syconn')

    logger.setLevel(logging.INFO)

    # create file handler which logs even debug messages
    log_dir = wd + "/logging/"
    os.makedirs(log_dir, exist_ok=True)
    fh = logging.FileHandler(log_dir + 'syconn')
    fh.setLevel(logging.INFO)

    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s %(message)s')

    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    # add the handlers to logger
    logger.addHandler(ch)
    logger.addHandler(fh)

    return logger


def initialize_logging(log_name):
    """

    Parameters
    ----------
    log_name : str
        Name of logger

    Returns
    -------

    """
    predefined_lognames = ['mp', 'gate', 'proc', 'ui']
    if log_name not in predefined_lognames:
        main_log = get_main_log()
        main_log.warn("Please use logger names as specified"
                      " here: {}".format(predefined_lognames))
    logger = logging.getLogger(log_name)

    logger.setLevel(logging.INFO)

    # create file handler which logs even debug messages
    log_dir = wd + "/logging/"
    os.makedirs(log_dir, exist_ok=True)
    fh = logging.FileHandler(log_dir + log_name)
    fh.setLevel(logging.INFO)

    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s %(message)s')

    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    # add the handlers to logger
    logger.addHandler(ch)
    logger.addHandler(fh)

    return logger
