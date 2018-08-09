# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max-Planck-Institute of Neurobiology, Munich, Germany
# Authors: Philipp Schubert, Joergen Kornfeld
import logging
from ..config.global_params import wd
import os


def get_main_log(working_dir=None):
    if working_dir is None:
        working_dir = wd
    logger = logging.getLogger('syconn')

    logger.setLevel(logging.INFO)

    # create file handler which logs even debug messages
    log_dir = working_dir + "/logging/"
    # Python2 compat.
    try:
        os.makedirs(log_dir, exist_ok=True)
    except TypeError:
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)
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


def initialize_logging(log_name, working_dir=None):
    """

    Parameters
    ----------
    log_name : str
        Name of logger

    Returns
    -------

    """
    if working_dir is None:
        working_dir = wd
    predefined_lognames = ['mp', 'gate', 'proc', 'ui', 'skeleton', 'multiview',
                           'object_extraction']
    if log_name not in predefined_lognames:
        main_log = get_main_log(working_dir=working_dir)
        main_log.warn("Please use logger names as specified"
                      " here: {}".format(predefined_lognames))
    logger = logging.getLogger(log_name)

    logger.setLevel(logging.INFO)

    # create file handler which logs even debug messages
    log_dir = working_dir + "/logging/"
    # Python2 compat.
    try:
        os.makedirs(log_dir, exist_ok=True)
    except TypeError:
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)
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
