# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max-Planck-Institute of Neurobiology, Munich, Germany
# Authors: Philipp Schubert, Joergen Kornfeld
import logging
import coloredlogs
from ..config.global_params import wd
import os


def get_main_log():
    logger = logging.getLogger('syconn')
    coloredlogs.install(level='DEBUG', logger=logger)
    logger.setLevel(logging.INFO)
    # create file handler which logs even debug messages
    log_dir = os.path.expanduser('~') + "/SyConn/logs/"
    os.makedirs(log_dir, exist_ok=True)
    fh = logging.FileHandler(log_dir + 'syconn.log')
    fh.setLevel(logging.INFO)

    # add the handlers to logger
    if os.path.isfile(log_dir + 'syconn.log'):
        os.remove(log_dir + 'syconn.log')
    logger.addHandler(fh)
    logger.info("Initialized logger '{}'. Log-files are stored at"
                " {}.".format('syconn', log_dir + 'syconn.log'))
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
    predefined_lognames = ['mp', 'gate', 'proc', 'ui', 'skeleton', 'multiview']
    if log_name not in predefined_lognames:
        main_log = get_main_log()
        main_log.warn("Please use logger names as specified"
                      " here: {}".format(predefined_lognames))
    logger = logging.getLogger(log_name)
    coloredlogs.install(level='DEBUG', logger=logger)
    logger.setLevel(logging.INFO)
    # create file handler which logs even debug messages
    log_dir = os.path.expanduser('~') + "/SyConn/logs/"
    os.makedirs(log_dir, exist_ok=True)
    fh = logging.FileHandler(log_dir + log_name + ".log")
    fh.setLevel(logging.INFO)

    # add the handlers to logger
    if os.path.isfile(log_dir + log_name + '.log'):
        os.remove(log_dir + log_name + '.log')
    logger.addHandler(fh)
    logger.info("Initialized logger '{}'. Log-files are stored at"
                " {}.".format(log_name, log_dir + log_name + ".log"))
    return logger
