# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max-Planck-Institute of Neurobiology, Munich, Germany
# Authors: Philipp Schubert, Joergen Kornfeld
import logging
import coloredlogs
import os
__all__ = ['initialize_logging']


def get_main_log():
    logger = logging.getLogger('syconn')
    coloredlogs.install(level='DEBUG', logger=logger)
    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    log_dir = os.path.expanduser('~') + "/SyConn/logs/"
    # Python2 compat.
    try:
        os.makedirs(log_dir, exist_ok=True)
    except TypeError:
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)
    fh = logging.FileHandler(log_dir + 'syconn.log')
    fh.setLevel(logging.DEBUG)

    # add the handlers to logger
    if os.path.isfile(log_dir + 'syconn.log'):
        os.remove(log_dir + 'syconn.log')
    logger.addHandler(fh)
    logger.info("Initialized logging. Log-files are stored at"
                " {}.".format(log_dir))
    return logger


def initialize_logging(log_name):
    """
    Logger for each package module. For import processing steps individual
    logger can be defined (e.g. multiviews, skeleton)
    Parameters
    ----------
    log_name : str
        Name of logger

    Returns
    -------

    """
    predefined_lognames = ['mp', 'gate', 'proc', 'ui', 'skeleton', 'multiview',
                           'handler', 'cnn', 'extraction', 'reps',
                           'object_extraction', 'backend']
    if log_name not in predefined_lognames:
        log_main.warning("Please use logger names as specified"
                         " here: {}".format(predefined_lognames))
    logger = logging.getLogger(log_name)
    coloredlogs.install(level='DEBUG', logger=logger)
    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    log_dir = os.path.expanduser('~') + "/SyConn/logs/"
    # Python2 compat.
    try:
        os.makedirs(log_dir, exist_ok=True)
    except TypeError:
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)
    fh = logging.FileHandler(log_dir + log_name + ".log")
    fh.setLevel(logging.DEBUG)

    # add the handlers to logger
    if os.path.isfile(log_dir + log_name + '.log'):
        os.remove(log_dir + log_name + '.log')
    logger.addHandler(fh)
    return logger


# init main logger
log_main = get_main_log()



