# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max-Planck-Institute of Neurobiology, Munich, Germany
# Authors: Philipp Schubert, Joergen Kornfeld
import logging
import coloredlogs
import os

from ..config import global_params

__all__ = ['initialize_logging']


def get_main_log():
    logger = logging.getLogger('syconn')
    coloredlogs.install(level='DEBUG', logger=logger)
    logger.setLevel(logging.DEBUG)

    if not global_params.DISABLE_FILE_LOGGING:
        # create file handler which logs even debug messages
        log_dir = os.path.expanduser('~') + "/SyConn/logs/"

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
        logger.info("Initialized file logging. Log-files are stored at"
                    " {}.".format(log_dir))
    logger.info("Initialized stdout logging.")
    return logger


def initialize_logging(log_name, log_dir=None):
    """
    Logger for each package module. For import processing steps individual
    logger can be defined (e.g. multiviews, skeleton)
    Parameters
    ----------
    log_name : str
        Name of logger
    log_dir : str
        Set log_dir specifically. Will then create a filehandler and ignore the
         state of global_params.DISABLE_FILE_LOGGING state.

    Returns
    -------

    """
    predefined_lognames = ['mp', 'gate', 'proc', 'ui', 'skeleton', 'multiview',
                           'handler', 'cnn', 'extraction', 'reps',
                           'object_extraction', 'backend']
    # not necessary...
    # if log_name not in predefined_lognames:
    #     log_main.warning("Please use logger names as specified"
    #                      " here: {}".format(predefined_lognames))
    logger = logging.getLogger(log_name)
    coloredlogs.install(level='DEBUG', logger=logger)
    logger.setLevel(logging.DEBUG)

    if not global_params.DISABLE_FILE_LOGGING or log_dir is not None:
        # create file handler which logs even debug messages
        if log_dir is None:
            log_dir = os.path.expanduser('~') + "/SyConn/logs/"
        try:
            os.makedirs(log_dir, exist_ok=True)
        except TypeError:
            if not os.path.isdir(log_dir):
                os.makedirs(log_dir)
        if os.path.isfile(log_dir + log_name + '.log'):
            os.remove(log_dir + log_name + '.log')
        # add the handlers to logger
        fh = logging.FileHandler(log_dir + log_name + ".log")
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger


# init main logger
log_main = get_main_log()



